"""Run TVM's paged KV-cache kernels one at a time on a ROCm GPU and check them against NumPy.

``rocm_find_crash.py`` showed a GPU fault inside ``attention_with_fused_qkv`` at the launch of
``tir_kv_cache_transpose_append``. An asynchronous fault can surface one launch late, so this
builds the kernels involved from TVM's own definitions for the local GPU and runs each on its
own, syncing after every launch, with the shapes Qwen2.5-0.5B uses:

  * ``tir_kv_cache_transpose_append`` - writes new K/V into the cache pages
  * ``batch_prefill_ragged_kv``       - causal prefill attention over the new tokens

Each test prints PASS/FAIL with the max error, or the error raised by the launch.

Usage (PowerShell, venv active):

    $env:AMD_SERIALIZE_KERNEL = "3"
    $env:TVM_ROCM_LOG_LAUNCHES = "1"
    python ci\\task\\rocm_kv_kernel_test.py
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
import tvm
from tvm.relax.frontend.nn.llm.kv_cache import (
    _attention_prefill_ragged,
    _kv_cache_transpose_append,
)
from tvm.s_tir import dlight as dl

# Qwen2.5-0.5B-Instruct
NUM_QO_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = "float16"
PAGE_SIZE = 16
NTOKEN = 20  # the prompt length in the failing run


def build(tir_func, target):
    mod = tvm.IRModule({"main": tir_func})
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    return tvm.tirx.build(mod["main"], target=target).main


def run_case(name, fn):
    print(f"--- {name}", flush=True)
    try:
        ok, detail = fn()
    except Exception:
        traceback.print_exc()
        print(f"FAIL {name}: raised", flush=True)
        return False
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    return ok


def test_transpose_append(dev, target, num_pages):
    f = build(_kv_cache_transpose_append(NUM_KV_HEADS, HEAD_DIM, DTYPE, PAGE_SIZE), target)
    rng = np.random.default_rng(0)
    k = rng.standard_normal((NTOKEN, NUM_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    v = rng.standard_normal((NTOKEN, NUM_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    pos = np.arange(NTOKEN, dtype="int32")  # first sequence: page 0, slots 0..19
    pages_shape = (num_pages, 2, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
    pages = tvm.runtime.tensor(np.zeros(pages_shape, DTYPE), dev)
    f(
        pages,
        tvm.runtime.tensor(k, dev),
        tvm.runtime.tensor(v, dev),
        tvm.runtime.tensor(pos, dev),
    )
    dev.sync()
    got = pages.numpy()
    expected = np.zeros(pages_shape, DTYPE)
    for t in range(NTOKEN):
        expected[t // PAGE_SIZE, 0, :, t % PAGE_SIZE, :] = k[t]
        expected[t // PAGE_SIZE, 1, :, t % PAGE_SIZE, :] = v[t]
    err = float(np.max(np.abs(got.astype("float32") - expected.astype("float32"))))
    return err == 0.0, f"max abs error {err}"


def test_prefill_ragged(dev, target):
    f = build(
        _attention_prefill_ragged(
            NUM_KV_HEADS, NUM_QO_HEADS, HEAD_DIM, HEAD_DIM, DTYPE, {}, target
        ),
        target,
    )
    rng = np.random.default_rng(1)
    q = rng.standard_normal((NTOKEN, NUM_QO_HEADS, HEAD_DIM)).astype(DTYPE)
    k = rng.standard_normal((NTOKEN, NUM_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    v = rng.standard_normal((NTOKEN, NUM_KV_HEADS, HEAD_DIM)).astype(DTYPE)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    out = tvm.runtime.empty((NTOKEN, NUM_QO_HEADS, HEAD_DIM), DTYPE, dev)
    lse = tvm.runtime.empty((NTOKEN, NUM_QO_HEADS), "float32", dev)
    indptr = np.array([0, NTOKEN], dtype="int32")
    f(
        tvm.runtime.tensor(q, dev),
        tvm.runtime.tensor(indptr, dev),
        tvm.runtime.tensor(k, dev),
        tvm.runtime.tensor(v, dev),
        tvm.runtime.tensor(indptr, dev),
        tvm.runtime.tensor(np.arange(NTOKEN, dtype="int32"), dev),  # q_rope_position
        tvm.runtime.tensor(np.zeros(1, dtype="int32"), dev),  # k_rope_pos_offset
        out,
        lse,
        1,  # causal
        0,  # rotary_mode: RoPE already applied by the rope kernel
        1.0,  # rope_scale
        1e6,  # rope_theta
        sm_scale,
    )
    dev.sync()

    group = NUM_QO_HEADS // NUM_KV_HEADS
    qf, kf, vf = (x.astype("float32") for x in (q, k, v))
    ref = np.zeros((NTOKEN, NUM_QO_HEADS, HEAD_DIM), "float32")
    ref_lse = np.zeros((NTOKEN, NUM_QO_HEADS), "float32")
    for h in range(NUM_QO_HEADS):
        kh = h // group
        s = (qf[:, h, :] @ kf[:, kh, :].T) * sm_scale
        s = np.where(np.tril(np.ones((NTOKEN, NTOKEN), bool)), s, -np.inf)
        m = s.max(axis=1, keepdims=True)
        p = np.exp(s - m)
        ref[:, h, :] = (p / p.sum(axis=1, keepdims=True)) @ vf[:, kh, :]
        ref_lse[:, h] = (m[:, 0] + np.log(p.sum(axis=1))) / math.log(2)
    err = float(np.max(np.abs(out.numpy().astype("float32") - ref)))
    lse_err = float(np.max(np.abs(lse.numpy() - ref_lse)))
    return err < 2e-2 and lse_err < 2e-2, f"max abs error {err:.2e}, lse error {lse_err:.2e}"


def main() -> int:
    dev = tvm.rocm(0)
    if not dev.exist:
        print("No ROCm device")
        return 2
    target = tvm.target.Target.from_device(dev)
    print(f"device: {dev.device_name}, target: {target}", flush=True)
    results = [
        run_case("transpose_append (8 pages)", lambda: test_transpose_append(dev, target, 8)),
        run_case(
            "transpose_append (2049 pages, like the model's cache)",
            lambda: test_transpose_append(dev, target, 2049),
        ),
        run_case("batch_prefill_ragged_kv", lambda: test_prefill_ragged(dev, target)),
    ]
    print("ALL PASS" if all(results) else "SOME FAILED", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

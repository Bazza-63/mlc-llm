"""Dump the generated code of TVM's paged KV-cache kernels for the local ROCm GPU.

This compiles the kernels exactly as ``rocm_kv_kernel_test.py`` does but never launches
them, so it cannot crash the display driver. For each kernel it writes, under
``rocm_kernel_dump/<kernel>/``:

  * ``scheduled.py``       - the TIR after dlight scheduling (TVMScript)
  * ``device_lowered.py``  - the device TIR right before LLVM codegen
  * ``kernel.ll``          - the LLVM IR handed to the AMDGPU backend
  * ``kernel.s``           - the generated ISA with the kernel descriptor (``.amdhsa_*``)

and prints the resource usage the backend reports (VGPRs, SGPRs, LDS, scratch, spills).

Usage (PowerShell, venv active):

    python ci\\task\\rocm_dump_kernel.py            # both kernels
    python ci\\task\\rocm_dump_kernel.py --only batch_prefill_ragged_kv --out dump
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tvm
from tvm.relax.frontend.nn.llm.kv_cache import (
    _attention_prefill_ragged,
    _kv_cache_transpose_append,
)
from tvm.s_tir import dlight as dl
from tvm.tirx.build import codegen_build, split_host_device_mods

# Qwen2.5-0.5B-Instruct
NUM_QO_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = "float16"
PAGE_SIZE = 16

# Lines of the ISA dump worth showing on the console.
INTERESTING = re.compile(
    r"^\s*(\.amdhsa_(group_segment_fixed_size|private_segment_fixed_size|next_free_[vs]gpr"
    r"|accum_offset|wavefront_size32|uses_dynamic_stack)|; (NumVgprs|NumSgprs|ScratchSize"
    r"|Occupancy|LDSByteSize|codeLenInByte|NumAgprs|TotalNumVgprs|SGPRBlocks|VGPRBlocks"
    r"|NumVGPRsForWavesPerEU|NumSGPRsForWavesPerEU|Spill|spill))"
)


def kernels(target):
    return {
        "tir_kv_cache_transpose_append": lambda: _kv_cache_transpose_append(
            NUM_KV_HEADS, HEAD_DIM, DTYPE, PAGE_SIZE
        ),
        "batch_prefill_ragged_kv": lambda: _attention_prefill_ragged(
            NUM_KV_HEADS, NUM_QO_HEADS, HEAD_DIM, HEAD_DIM, DTYPE, {}, target
        ),
    }


def dump(name, make_func, target, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mod = tvm.IRModule({"main": make_func()})
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    (out / "scheduled.py").write_text(mod.script(), encoding="utf-8")

    # Same steps as tvm.tirx.build, kept apart so the device TIR can be saved.
    target_host = tvm.target.Target("llvm")
    mod = tvm.tirx.transform.BindTarget(target.with_host(target_host))(mod)
    pipeline, _, finalize_device_passes = tvm.tirx.get_default_tir_pipeline(target, mod)
    mod = pipeline(mod)
    _, device_mods = split_host_device_mods(mod)
    (dev_target, dev_mod), *rest = device_mods.items()
    assert not rest, f"expected one device module, got {len(device_mods)}"
    dev_mod = finalize_device_passes()(dev_mod)
    (out / "device_lowered.py").write_text(dev_mod.script(), encoding="utf-8")

    rt_mod = codegen_build(dev_mod, dev_target)
    (out / "kernel.ll").write_text(rt_mod.inspect_source("hip"), encoding="utf-8")
    asm = rt_mod.inspect_source("asm")
    (out / "kernel.s").write_text(asm, encoding="utf-8")

    print(f"=== {name} -> {out}", flush=True)
    for line in asm.splitlines():
        if INTERESTING.match(line):
            print("   ", line.strip(), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="rocm_kernel_dump", help="output directory")
    parser.add_argument("--only", action="append", help="dump only this kernel (repeatable)")
    parser.add_argument(
        "--target",
        default=None,
        help='target string, e.g. "rocm -mcpu=gfx1101 -thread_warp_size=32" '
        "(default: from the local device)",
    )
    args = parser.parse_args()

    if args.target:
        target = tvm.target.Target(args.target)
    else:
        dev = tvm.rocm(0)
        if not dev.exist:
            print("No ROCm device; pass --target")
            return 2
        target = tvm.target.Target.from_device(dev)
    print(f"target: {target}", flush=True)

    out = Path(args.out)
    for name, make_func in kernels(target).items():
        if args.only and name not in args.only:
            continue
        dump(name, make_func, target, out / name)
    print(f"done; zip and share {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

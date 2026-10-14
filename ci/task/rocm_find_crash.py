"""Find which GPU function crashes the driver when running a compiled MLC model.

On Windows a GPU kernel that hangs or faults usually does not raise an error: the display
driver resets (TDR) and the process dies with no message. This script runs the model one
Relax VM call at a time (through ``mlc_llm.testing.debug_chat``) and, for every call:

  1. writes ``BEGIN <n> <name> <argument shapes>`` to the log file and flushes it to disk,
  2. runs the call,
  3. waits for the GPU to finish (device sync) and writes ``END <n>``.

After a crash, the last ``BEGIN`` line without a matching ``END`` names the culprit. Each
line is fsync'ed, so it survives the process being killed.

Usage (PowerShell, from the repo root, with the venv active):

    $env:AMD_SERIALIZE_KERNEL = "3"      # optional: HIP waits for each kernel as well
    python ci\\task\\rocm_find_crash.py `
        --model .\\models\\Qwen2.5-0.5B-Instruct-q4f16_1-MLC `
        --model-lib "$env:LOCALAPPDATA\\mlc_llm\\model_lib\\<hash>.dll"
    Get-Content rocm_crash.log -Tail 8

The model library is the DLL ``mlc_llm chat`` printed ("Using compiled model lib: ...").
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


class CrashLog:
    """Line log that is flushed and fsync'ed after every write."""

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "w", encoding="utf-8")

    def write(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.file.write(f"{stamp} {line}\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        print(line, flush=True)


class SyncInstrument:
    """Relax VM instrument: log before each call, sync the GPU after it."""

    def __init__(self, log: CrashLog, device):
        self.log = log
        self.device = device
        self.counter = 0
        self.phase = "init"

    def reset(self, debug_out: Path) -> None:
        # Called by DebugChat between prefill and each decode step.
        self.phase = Path(debug_out).name
        self.log.write(f"PHASE {self.phase}")

    def __call__(self, func, name, before_run, ret_val, *args):
        import tvm

        if before_run:
            shapes = []
            for arg in args:
                if isinstance(arg, tvm.runtime.Tensor):
                    shapes.append(f"{arg.dtype}{list(arg.shape)}")
            self.log.write(f"BEGIN {self.counter} [{self.phase}] {name} {' '.join(shapes)}")
            return
        self.device.sync()
        self.log.write(f"END   {self.counter}")
        self.counter += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="MLC model directory")
    parser.add_argument("--model-lib", required=True, help="Compiled model library (.dll/.so)")
    parser.add_argument("--device", default="rocm:0")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--generate-len", type=int, default=3)
    parser.add_argument("--log", default="rocm_crash.log")
    args = parser.parse_args()

    log = CrashLog(Path(args.log))
    log.write(f"model={args.model} lib={args.model_lib} device={args.device}")
    log.write(f"AMD_SERIALIZE_KERNEL={os.environ.get('AMD_SERIALIZE_KERNEL', '')}")

    from mlc_llm.support.auto_device import detect_device
    from mlc_llm.testing.debug_chat import DebugChat

    device = detect_device(args.device)
    log.write(f"device: {device} name={device.device_name} warp={device.warp_size}")
    instrument = SyncInstrument(log, device)

    log.write("PHASE load library and parameters")
    with tempfile.TemporaryDirectory() as tmp:
        chat = DebugChat(
            model=args.model,
            model_lib=args.model_lib,
            debug_dir=Path(tmp),
            device=args.device,
            debug_instrument=instrument,
        )
        device.sync()
        log.write("PHASE parameters loaded")
        instrument.phase = "prefill"
        chat.generate(args.prompt, args.generate_len)
        device.sync()
    log.write("DONE: no crash")
    return 0


if __name__ == "__main__":
    sys.exit(main())

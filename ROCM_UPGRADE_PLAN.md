# Plan: upgrade ROCm support to ROCm 10.1 (Linux and Windows)

Status: proposal. MLC-LLM currently documents ROCm 6.1/6.2 only. ROCm 10.1 ships on
Linux and Windows, including as a tarball / TheRock build that can live anywhere on disk.

Most of the work is in the TVM submodule (`3rdparty/tvm`, upstream `mlc-ai/relax`); it
must land there first, then MLC-LLM bumps the submodule.

Items marked **(verify)** depend on ROCm 10.1 details not yet confirmed against AMD's
release notes.

## Reference install (Windows, TheRock build)

AMD's system-wide environment:

```bat
HIP_PATH            = C:\TheRock\build
HIP_DEVICE_LIB_PATH = C:\TheRock\build\lib\llvm\amdgcn\bitcode
HIP_PLATFORM        = amd
LLVM_PATH           = C:\TheRock\build\lib\llvm
PATH               += C:\TheRock\build\bin;C:\TheRock\build\lib\llvm\bin
```

AMD's instructions may later clear these with `setx <VAR> "" /M`, leaving only the `PATH`
entries. Discovery must therefore treat an empty variable as unset and fall back to
`PATH`.

Note there is **no `ROCM_PATH`** and nothing under `/opt/rocm`. Layout relative to the
root: `include/`, `lib/`, `bin/` (DLLs), `lib/llvm/` (LLVM, `ld.lld`, device bitcode).

### Verified on Windows (RX 7800 XT, gfx1101)

Results from `rocm_windows_check.ps1` on the reference install:

- Toolchain: `hipconfig --version` = 7.16 (HIP API numbering stays 7.x under ROCm
  10.x, so every "ROCm 7" API change applies). AMD clang 24 (LLVM 24) with
  `llvm-config.exe`, so TVM can be built against ROCm's own LLVM.
- **TVM's device pipeline works with this toolchain.** Compiling OpenCL with device libs
  from `HIP_DEVICE_LIB_PATH` and running `ld.lld --no-undefined -shared` produced a
  `.hsaco`. What's broken is TVM's discovery and glue code, not the toolchain itself.
- The runtime loads `amdhip64_7.dll` from **`C:\Windows\System32` (the Adrenalin driver's
  copy, HIP 7.12)**, not TheRock's `bin` (7.16). Windows searches System32 before `PATH`.
  TVM would therefore compile with 7.16 headers but run on the driver's 7.12 runtime.
  **Requirement: the driver's `System32\amdhip64_*.dll` must never be loaded.** Load the
  ROCm install's copy by full path (see Phase 3, "Windows DLL loading").
- hipBLAS, hipBLASLt and rocBLAS DLLs are present. RCCL and `hsa-runtime64` are absent.
- Device bitcode: 61 ISA files (including 1101, 1201 and generic targets) and ABI 400/500/600.
  `oclc_wavefrontsize64_on`, `oclc_unsafe_math_*` and `oclc_finite_only_*` **still exist**.
  `oclc_daz_opt_on/off` and `oclc_correctly_rounded_sqrt_on/off` **are gone**, so TVM's
  `callback_rocm_bitcode_path()` raises `could not find bitcode oclc_daz_opt_on`.
- **Full device path verified end to end:** `amdclang++` built a HIP program without any
  Visual Studio environment loaded. The kernel ran on the GPU and matched CPU results, and
  `hipModuleLoad` loaded the `ld.lld`-linked `.hsaco` (TVM's runtime path). `hipcc` failed
  in the same shell because it could not find the C++ standard headers. **Use
  `amdclang`/`amdclang++` on Windows, not `hipcc`.**
- The device reports `warpSize=32` (RDNA3), confirming that TVM's default
  `thread_warp_size=64` is wrong for this GPU.
- The driver's `amdhip64_7.dll` is file 10.0.3683.0 (HIP 7.12); TheRock's is 10.0.3684.0
  (HIP 7.16). Even though the builds are adjacent, only the ROCm install's copy may be used.

Corrections to the analysis below:
- `FindROCM.cmake` does find `lib/amdhip64.lib` when given the root
  (`-DUSE_ROCM=C:/TheRock/build`). It fails only because it ignores `HIP_PATH`.
- `ROCM.cmake` already links `hsa-runtime64` only if found. The blocker is the
  `#include <hsa/hsa.h>` / `hsa_init()` in `rocm_device_api.cc`.
- `TVM_LLVM_VERSION` guards in `codegen_amdgpu.cc` only go up to 21. Building against
  LLVM 24 will likely need new guards (Phase 0).

### Why the current code fails on this install

| Location | Current behaviour | Result |
|---|---|---|
| `3rdparty/tvm/cmake/utils/FindROCM.cmake` | Looks at `USE_ROCM`, `$ROCM_PATH`, `/opt/rocm` (not `HIP_PATH`) | ROCm not found unless `-DUSE_ROCM=<root>` is passed |
| `3rdparty/tvm/python/tvm/support/rocm.py` `find_rocm_path()` | `ROCM_PATH` → `which hipcc` (subprocess) → `/opt/rocm` | `RuntimeError` |
| `rocm.py` `callback_rocm_bitcode_path()` | `<root>/amdgcn/bitcode`, else `/opt/rocm/lib`; requires `oclc_daz_opt_*` and `oclc_correctly_rounded_sqrt_*` | Bitcode not found (real path: `<root>/lib/llvm/amdgcn/bitcode`), and four required files no longer exist |
| `rocm.py` `find_lld()` + `tvm/support/utils.py` `which()` | `which()` never appends `.exe` | `ld.lld.exe` not found despite being on `PATH` |
| `rocm.py` `get_rocm_arch()` | Runs `<root>/bin/rocminfo` | Not available on Windows; silently falls back to `gfx900` |
| `rocm_device_api.cc` | Includes `hsa/hsa.h` and calls `hsa_init` (`ROCM.cmake` already treats the library as optional) | No HSA runtime on Windows, so the link fails |

Check a Windows install with `ci/task/rocm_windows_check.ps1`.

## Phase 0: toolchain baseline

1. TVM links ROCm's device-library bitcode into its own LLVM module
   (`src/backend/rocm/codegen/llvm/codegen_amdgpu.cc` `BuildAMDGPU`). LLVM cannot read
   bitcode written by a newer LLVM, so **TVM's LLVM must be at least as new as ROCm's**.
   Build TVM with `USE_LLVM=<LLVM_PATH>/bin/llvm-config` (Linux: `$ROCM_PATH/lib/llvm`).
   **(verify: TheRock ships `llvm-config` plus LLVM headers and static libraries;
   otherwise build a matching LLVM.)**
2. Re-check every `TVM_LLVM_VERSION` guard in `codegen_amdgpu.cc` against that LLVM.
3. Bring up `USE_ROCM`, `USE_HIPBLAS`, `USE_THRUST` (and `USE_RCCL` on Linux) builds on both OSes, then work
   through the phases below.

## Phase 1: TVM build system

- **`cmake/utils/FindROCM.cmake`**
  - Root search order: `USE_ROCM=<path>` → `$ENV{ROCM_PATH}` → `$ENV{HIP_PATH}` (empty
    values count as unset) → root derived from `PATH` (`find_program(hipcc|amdclang|ld.lld)`
    in `<root>/bin` or `<root>/lib/llvm/bin`) → `/opt/rocm` (non-Windows only) →
    `FATAL_ERROR`.
  - Search `lib`, `lib64`, `bin`; on Windows find `amdhip64.lib` (import lib) and `.dll`.
  - Drop the pre-3.7 `hip_hcc` fallback. Report the ROCm version.
  - Preferably use `find_package(hip CONFIG)` / `hipblas` / `hipblaslt` / `rocthrust`
    with the root on `CMAKE_PREFIX_PATH`.
- **`cmake/modules/ROCM.cmake`**
  - Make `hsa-runtime64` Linux-only.
  - Remove the obsolete `__HIP_PLATFORM_HCC__` define; keep `__HIP_PLATFORM_AMD__`.
  - rocThrust currently requires `CXX=hipcc` (fatal otherwise). Use CMake's `HIP` language
    for `thrust.cu` with `CMAKE_HIP_COMPILER=amdclang++`. On Windows `amdclang++` works
    where `hipcc` fails (verified).
- **`cmake/utils/FindRCCL.cmake`, `CMakeLists.txt` (RCCL block)**: hint
  `<root>/include/rccl` and `<root>/lib`. RCCL is Linux-only, so skip it on Windows.

## Phase 2: TVM runtime (`src/backend/rocm/runtime/`)

- `rocm_device_api.cc`: replace the `hsa_init`/`hsa_shut_down` existence check with
  `hipGetDeviceCount`, which removes the HSA dependency.
- `rocm_device_api.cc`: `hipHostMalloc` has been deprecated since ROCm 6; switch to
  `hipExtHostAlloc`/`hipHostAlloc` **(verify removal status in 10.x)**.
- Compile against 10.1 headers and fix any other removed `hip*` API.

## Phase 3: TVM codegen, target and Python helpers

- `src/backend/rocm/codegen/target_kind.cc`
  - Triple `amdgcn-amd-amdhsa-hcc` → `amdgcn-amd-amdhsa`. It is enforced by
    `CheckOrSetAttr`, so accept both for compatibility.
  - Default `mcpu=gfx900` is likely unsupported now **(verify)**; error out instead of
    guessing.
  - `thread_warp_size` defaults to 64; RDNA (gfx10/11/12) is wave32. Derive it from `mcpu`
    or `hipDeviceAttributeWarpSize`.
  - Delete the `version < 305` / `-code-object-v3` branch (pre-ROCm 4).
- `python/tvm/support/rocm.py`
  - `find_rocm_path()`: same order as CMake (`ROCM_PATH` → `HIP_PATH` → root derived from
    `PATH` via `shutil.which` → `/opt/rocm` on Linux), skipping empty values (today
    `"ROCM_PATH" in os.environ` accepts `""`). No `which` subprocess. Optional fallback for pip/TheRock wheels:
    `rocm-sdk path --root` **(verify)**.
  - `callback_rocm_bitcode_path()`: `HIP_DEVICE_LIB_PATH` (if non-empty) → `<root>/lib/llvm/amdgcn/bitcode`
    → `<root>/amdgcn/bitcode`. Fix the stale list:
    - `irif`, `oclc_daz_opt_on/off` and `oclc_correctly_rounded_sqrt_on/off` no longer
      exist (verified on Windows). Drop them or make them optional.
    - Choose `oclc_isa_version_<gfx>` from the target `mcpu` instead of the fixed
      803/900/906/1030 list (gfx1101 is present on disk but never linked today).
    - `oclc_wavefrontsize64_on` still exists. Link it only for wave64 targets.
    - Match `oclc_abi_version_*` (400/500/600 available) to the code-object version LLVM emits.

    This needs `mcpu` passed into the callback (`codegen_amdgpu.cc` `BuildAMDGPU`).
  - `find_lld()`: `LLVM_PATH/bin` (if non-empty) → `<root>/lib/llvm/bin` → `PATH` via `shutil.which`.
    Drop the hardcoded `/opt/rocm/llvm/bin`.
  - `get_rocm_arch()`: use the runtime's `gcnArchName` (`tvm.rocm(0)`), with `rocminfo`
    only as a Linux fallback.
  - `have_matrixcore()`: the `major >= 8` heuristic is wrong for gfx9/11/12.
- `python/tvm/support/utils.py` `which()`: use `shutil.which` (handles `PATHEXT`).
- **Windows DLL loading: never use the driver's `System32\amdhip64_*.dll`.** `PATH` does
  not help, because the standard search order checks System32 before `PATH`, and
  `os.add_dll_directory` does not outrank it either. Once a module named `amdhip64_7.dll`
  is in the process, every later by-name import (`tvm_runtime_rocm`, hipBLAS, rocBLAS,
  hipBLASLt) binds to it. So:
  - C++ (covers every host, including the MLC-LLM CLI): link `tvm_runtime_rocm` with
    `/DELAYLOAD:amdhip64_*.dll` (+ `delayimp.lib`). Install a `__pfnDliNotifyHook2`
    `dliNotePreLoadLibrary` hook that calls
    `LoadLibraryExW(<root>\bin\amdhip64_*.dll, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`, with `<root>` resolved as in `FindROCM`
    (`ROCM_PATH` → `HIP_PATH`). Fail with a clear error, never fall back silently, if the
    resolved module is in System32.
  - Python: in `tvm/libinfo` / `tvm_ffi` library loading, preload the same DLL with
    `ctypes.WinDLL(<root>\bin\amdhip64_*.dll, winmode=0x1100)` before loading
    `tvm_runtime_rocm` or `tvm_runtime_extra` (hipBLAS).
  - The DLL name (`amdhip64_7.dll`) comes from the import library (`amdhip64.lib`). Detect
    it at configure time instead of hardcoding it.
  - Check: `ci/task/rocm_windows_check.ps1` steps 5 and 7 verify which copy is loaded,
    including during HIP's pre-`main` fat-binary registration.
  - **Verified on the reference machine (VS 2022 Build Tools 14.44):** an `amdclang++`
    program linked with `-Wl,/DELAYLOAD:amdhip64_7.dll -ldelayimp` and the hook above used
    `C:\TheRock\build\bin\amdhip64_7.dll` (HIP 7.16.26385), never System32. It ran the
    kernel correctly and loaded the `ld.lld`-linked `.hsaco`. Loading by full path from
    Python/PowerShell also made later by-name loads bind to the ROCm copy. Windows-only
    code in `.hip`/`.cu` sources must be guarded with
    `#if defined(_WIN32) && !defined(__HIP_DEVICE_COMPILE__)`, since `_WIN32` is also defined
    in the device pass.

## Phase 4: TVM hipBLAS / hipBLASLt

- `src/runtime/extra/contrib/hipblas/hipblas_utils.h` (`GetHipBlasDataType`) and
  `hipblas.cc` (`hipblasGemmEx`, `hipblasGemmStridedBatchedEx`) use the legacy
  `hipblasDatatype_t`/`HIPBLAS_R_*` API, which was removed in hipBLAS 3 (ROCm 7). Port to
  `hipDataType` (`HIP_R_*`) plus `hipblasComputeType_t`.
- Re-check the hipBLASLt descriptor and epilogue enums against the 10.1 headers.
- Optional: FP8 offload for gfx942/gfx950/gfx12 in `python/tvm/relax/backend/rocm/hipblas.py`.

## Phase 5: MLC-LLM

- `python/mlc_llm/support/auto_target.py`: stop appending `rocblas` and `miopen` (unused).
  Add `thrust`/`hipblas` only if the runtime was built with them, e.g.
  `tvm.get_global_func("tvm.contrib.thrust.sort", True)`.
- `python/mlc_llm/compiler_pass/blas_dispatch.py`: regression-test against Phase 4.
- `cpp/serve/engine.cc`: multi-GPU uses RCCL; on Windows give a clear error or force a
  single GPU.
- Windows model libraries must be `.dll`: check the export path and
  `tests/python/integration/test_model_compile.py` (the `rocm: "so"` suffix).
- `tests/python/integration/test_model_compile.py`: add `gfx942` and `gfx1201` next to
  `gfx1100`.
- `cmake/gen_cmake_config.py`: when ROCm is enabled, also ask about hipBLAS, Thrust and
  RCCL.
- The `kDLROCM` paths in `cpp/serve/*`, `interface/compiler_flags.py` and
  `quantization/utils.py` are version-independent; regression-test only.

## Phase 6: CI and packaging

- `ci/task/build_lib.sh`: `USE_RCCL /opt/rocm/rccl/` is the wrong path (use `ON` or the
  root). Update the `auditwheel` excludes: 10.1 removes ROCm SMI from the standard build,
  so drop `librocm_smi64`; check HIP library names **(verify)**.
- `ci/task/test_model_compile.sh`: replace `mlc-ai-nightly-rocm57`.
- Windows: add a ROCm job to `.github/workflows/windows-build.yaml` /
  `ci/task/build_win.bat`. Unpack the tarball and set the variables listed above.
- Publish wheels as `*-rocm101`.

## Phase 7: docs

- `docs/install/gpu.rst`: replace "only supports ROCm 6.1/6.2". Document tarball/pip
  installs on Linux and Windows and the required environment variables.
- `docs/install/mlc_llm.rst`, `docs/install/tvm.rst`: ROCm 10.1 wheel tabs; replace
  "LLVM>=17/18" with "build TVM against ROCm's own LLVM (`LLVM_PATH`)".
- `README.md`: list ROCm under the AMD GPU column for Windows too.

## Order and validation

1. Phases 1 and 2, then Phase 3 (unblocks compiling on the reference install).
2. Phase 4.
3. Upstream TVM changes; bump `3rdparty/tvm`.
4. Phase 5, then Phases 6–7.

Validate with a small model (e.g. Qwen3-0.6B `q4f16_1`) compiled and chatted on one RDNA
(wave32) and one CDNA (wave64) GPU, on both Linux and Windows.

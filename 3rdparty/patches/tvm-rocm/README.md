# TVM patch: ROCm 10.1 (HIP 7.x) on Linux and Windows

`0001-ROCm-Support-ROCm-10.1-HIP-7.x-on-Linux-and-Windows.patch` holds Phases 1–4 of
`ROCM_UPGRADE_PLAN.md` for the TVM submodule (`3rdparty/tvm`, upstream `mlc-ai/relax`),
based on commit `201fa00`. It is kept here until it lands in a fork or upstream. After that,
the submodule pointer is bumped and this directory is deleted.

## Apply

```sh
cd 3rdparty/tvm
git checkout 201fa00
git am ../patches/tvm-rocm/0001-*.patch
```

## Verified so far

- Compiles against the real `therock-10.1` HIP 7.16, hipBLAS and hipBLASLt headers (host
  compiler, as TVM builds them): `rocm_device_api.cc`, `rocm_module.cc`, `hipblas*.cc`,
  `target_kind.cc`, and `codegen_amdgpu.cc` with LLVM 18. Unpatched `hipblas.cc` does
  not compile against these headers (`hipblasDatatype_t` was removed).
- `rocm_dll_loader_win.cc` compiles for Windows with MinGW headers. MSVC's `delayimp.h`
  declares the hook `const`; the same pattern was verified with MSVC 14.44 by
  `ci/task/rocm_windows_check.ps1` step 7.
- `FindROCM.cmake` works with `HIP_PATH` only, empty `ROCM_PATH`, `ROCM_PATH` over
  `HIP_PATH`, root derived from `hipconfig` or `amdclang` on `PATH`, and explicit `USE_ROCM`.
  A full TVM configure with `USE_ROCM=ON USE_HIPBLAS=ON` succeeds.
- `test_rocm_support.py` (12 tests) covers `tvm.support.rocm` discovery and bitcode
  selection against the TheRock/ROCm 10.1 layout and older layouts, with `tvm` stubbed out:

  ```sh
  ROCM_PY=3rdparty/tvm/python/tvm/support/rocm.py python -m pytest 3rdparty/patches/tvm-rocm/test_rocm_support.py
  ```

## Not verified yet (needs the Windows/ROCm machine)

- A full TVM build on Windows with TheRock's LLVM 24 (`USE_LLVM=%LLVM_PATH%\bin\llvm-config.exe`).
  TVM's `TVM_LLVM_VERSION` guards only go up to LLVM 21.
- The delay-load hook inside `tvm_runtime_rocm.dll` / `tvm_runtime_extra.dll`.
- Compiling and running a model end to end.

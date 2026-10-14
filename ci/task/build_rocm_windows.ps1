<#
.SYNOPSIS
  Builds TVM (compiler + ROCm runtime) and MLC-LLM against a ROCm install on Windows.

.DESCRIPTION
  Stages (run in order; -Stage picks one or a range start):
    check  Prerequisites: Python, Rust (cargo), Visual Studio C++ tools, ROCm root, LLVM.
    venv   Python virtual environment (-Venv) with the build tools.
    ffi    apache-tvm-ffi from 3rdparty/tvm/3rdparty/tvm-ffi (same commit TVM is built with).
    tvm    apache-tvm from 3rdparty/tvm with USE_ROCM, USE_HIPBLAS and ROCm's own LLVM.
    mlc    mlc-llm from this checkout with USE_ROCM / USE_HIPBLAS (config.cmake).
    test   Smoke test: ROCm device, target, which amdhip64 DLL is loaded, and a TVM kernel
           compiled for the GPU and run on it.

  Builds use Ninja with one compile job per logical CPU (-Jobs to override, e.g. if memory
  runs out: each LLVM-heavy compile can take 1-2 GB).

  Everything is logged to -Log (default build_rocm_windows.log in the repo root). The
  script stops at the first failing stage; rerun with -Stage <stage> to resume there.

.EXAMPLE
  cd C:\Users\Geeza\code\mlc-llm
  powershell -ExecutionPolicy Bypass -File .\ci\task\build_rocm_windows.ps1
  powershell -ExecutionPolicy Bypass -File .\ci\task\build_rocm_windows.ps1 -Stage tvm
#>
param(
  [ValidateSet('check', 'venv', 'ffi', 'tvm', 'mlc', 'test')]
  [string]$Stage = 'check',
  [switch]$Only,
  [string]$RocmRoot = '',
  [string]$LlvmConfig = '',
  [string]$Venv = '',
  [string]$Python = 'python',
  [int]$Jobs = 0,
  [string]$Log = ''
)

$ErrorActionPreference = 'Continue'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Venv) { $Venv = Join-Path $Repo '.venv-rocm' }
if (-not $Log) { $Log = Join-Path $Repo 'build_rocm_windows.log' }
$Stages = @('check', 'venv', 'ffi', 'tvm', 'mlc', 'test')

function Say([string]$m, [string]$color = 'Gray') {
  Write-Host $m -ForegroundColor $color
  Add-Content -LiteralPath $Log -Value $m
}
function Die([string]$m) {
  Say "[FAIL] $m" 'Red'
  Say "Log: $Log" 'Red'
  exit 1
}
function Run([string]$what, [string]$exe, [string[]]$argList) {
  Say ">> $what" 'Cyan'
  Say "   $exe $($argList -join ' ')"
  & $exe @argList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append | Out-Host
  if ($LASTEXITCODE -ne 0) { Die "$what failed (exit $LASTEXITCODE)" }
}
function ToCMake([string]$p) { return $p.Replace('\', '/') }

# CMake cannot switch generator inside an existing build directory ("generator mismatch"),
# so clear a build directory configured by an earlier Visual Studio / NMake run.
function Reset-NonNinjaBuild([string]$buildDir) {
  $cache = Join-Path $buildDir 'CMakeCache.txt'
  if (-not (Test-Path -LiteralPath $cache)) { return }
  $gen = (Select-String -LiteralPath $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$' |
    Select-Object -First 1).Matches.Groups[1].Value
  if ($gen -ne 'Ninja') {
    Say "Removing $buildDir (configured with '$gen'; switching to Ninja)" 'Yellow'
    Remove-Item -LiteralPath $buildDir -Recurse -Force
  }
}

function Import-VsEnv {
  if ($env:INCLUDE -and $env:VCToolsInstallDir) { return 'already active' }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vs) { return $null }
  $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path -LiteralPath $vcvars)) { return $null }
  cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
  }
  if ($env:INCLUDE) { return $vcvars } else { return $null }
}

Set-Content -LiteralPath $Log -Value "build_rocm_windows.ps1 $(Get-Date -Format s)  stage=$Stage only=$Only"
$start = $Stages.IndexOf($Stage)
$todo = if ($Only) { @($Stage) } else { $Stages[$start..($Stages.Count - 1)] }

# ---------------------------------------------------------------------------
# Always: resolve ROCm, LLVM and the MSVC environment (later stages need them).
if (-not $RocmRoot) {
  foreach ($v in @('ROCM_PATH', 'HIP_PATH')) {
    $val = [Environment]::GetEnvironmentVariable($v)
    if (-not $RocmRoot -and $val -and (Test-Path -LiteralPath $val)) { $RocmRoot = $val }
  }
}
if (-not $RocmRoot) { Die 'ROCm root not found: set ROCM_PATH or HIP_PATH, or pass -RocmRoot' }
$RocmRoot = (Resolve-Path -LiteralPath $RocmRoot).Path.TrimEnd('\')
if (-not $LlvmConfig) {
  $llvmDir = if ($env:LLVM_PATH) { $env:LLVM_PATH } else { Join-Path $RocmRoot 'lib\llvm' }
  $LlvmConfig = Join-Path $llvmDir 'bin\llvm-config.exe'
}
$VenvPy = Join-Path $Venv 'Scripts\python.exe'

$vsEnv = Import-VsEnv
Say "Repo:        $Repo"
Say "ROCm root:   $RocmRoot"
Say "llvm-config: $LlvmConfig"
Say "Venv:        $Venv"
Say "MSVC env:    $(if ($vsEnv) { $vsEnv } else { '<not found>' })"

# ---------------------------------------------------------------------------
if ($todo -contains 'check') {
  Say "`n== check ==" 'Yellow'
  if (-not $vsEnv) { Die 'Visual Studio 2022 / Build Tools with "Desktop development with C++" not found' }
  $py = Get-Command $Python -ErrorAction SilentlyContinue
  if (-not $py) { Die "Python not found ($Python). Install Python 3.10-3.13 or pass -Python" }
  $pyver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
  Say "Python $pyver at $($py.Source)"
  if ([version]$pyver -lt [version]'3.10') { Die 'Python 3.10 or newer is required' }
  if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Die 'cargo (Rust) not found; tokenizers-cpp needs it. Install from https://rustup.rs'
  }
  Say "cargo: $((Get-Command cargo).Source)"
  foreach ($f in @('lib\amdhip64.lib', 'include\hip\hip_runtime_api.h', 'lib\hipblas.lib')) {
    if (-not (Test-Path -LiteralPath (Join-Path $RocmRoot $f))) { Die "Missing $RocmRoot\$f" }
  }
  if (-not (Test-Path -LiteralPath $LlvmConfig)) { Die "llvm-config not found at $LlvmConfig (pass -LlvmConfig)" }
  $llvmVer = & $LlvmConfig --version
  Say "LLVM $llvmVer (TVM's LLVM guards go up to 21; newer versions may need source fixes)"
  if (-not (Test-Path -LiteralPath (Join-Path $Repo '3rdparty\tvm\cmake\modules\ROCM.cmake'))) {
    Die '3rdparty\tvm is empty: initialise submodules first (ci\task\init_submodules.ps1)'
  }
  $tvmHead = & git -C (Join-Path $Repo '3rdparty\tvm') log --oneline -1 2>$null
  Say "3rdparty/tvm: $tvmHead"
  Say '[PASS] prerequisites' 'Green'
}

# ---------------------------------------------------------------------------
if ($todo -contains 'venv') {
  Say "`n== venv ==" 'Yellow'
  if (-not (Test-Path -LiteralPath $VenvPy)) { Run 'create venv' $Python @('-m', 'venv', $Venv) }
  Run 'build tools' $VenvPy @('-m', 'pip', 'install', '-U', 'pip', 'scikit-build-core>=0.11',
    'setuptools-scm>=8', 'cython>=3.2.8', 'cmake>=3.24', 'ninja', 'numpy', 'ml_dtypes',
    'typing_extensions')
}

if (-not (Test-Path -LiteralPath $VenvPy)) { Die "venv missing at ${Venv}: run with -Stage venv" }
# Put the venv's cmake/ninja first on PATH for the builds.
$env:PATH = (Join-Path $Venv 'Scripts') + ';' + $env:PATH

# Parallel builds: Ninja runs one compile per job. The Visual Studio generator (MSBuild) and
# NMake, which CMake otherwise picks here, compile mostly one file at a time.
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
if (Get-Command ninja -ErrorAction SilentlyContinue) {
  $env:CMAKE_GENERATOR = 'Ninja'
} else {
  Say '[WARN] ninja not found; builds may not use all cores (rerun with -Stage venv)' 'Yellow'
}
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$Jobs"
Say "Build:       generator=$($env:CMAKE_GENERATOR) jobs=$Jobs"

# ---------------------------------------------------------------------------
if ($todo -contains 'ffi') {
  Say "`n== ffi ==" 'Yellow'
  Reset-NonNinjaBuild (Join-Path $Repo '3rdparty\tvm\3rdparty\tvm-ffi\build')
  Run 'apache-tvm-ffi' $VenvPy @('-m', 'pip', 'install', '-v', '--no-build-isolation',
    (Join-Path $Repo '3rdparty\tvm\3rdparty\tvm-ffi'))
}

# ---------------------------------------------------------------------------
if ($todo -contains 'tvm') {
  Say "`n== tvm ==" 'Yellow'
  $root = ToCMake $RocmRoot
  Reset-NonNinjaBuild (Join-Path $Repo '3rdparty\tvm\build')
  # USE_Z3 needs mlc-z3-static, which may not have a Windows wheel.
  Run 'apache-tvm (compiler + ROCm runtime)' $VenvPy @('-m', 'pip', 'install', '-v',
    '--no-build-isolation', '--no-deps',
    "--config-settings=cmake.define.USE_ROCM=$root",
    '--config-settings=cmake.define.USE_HIPBLAS=ON',
    "--config-settings=cmake.define.USE_LLVM=$(ToCMake $LlvmConfig)",
    '--config-settings=cmake.define.USE_Z3=OFF',
    '--config-settings=cmake.define.HIDE_PRIVATE_SYMBOLS=ON',
    '--config-settings=build.verbose=true',
    (Join-Path $Repo '3rdparty\tvm'))
  # --no-deps above also skips TVM's prebuilt LLVM JIT runtime (apache-tvm-ffi-orcjit), which
  # running host-side code needs (e.g. calling a compiled module directly). Install the
  # version TVM pins; it has a win_amd64 wheel.
  $pin = Select-String -LiteralPath (Join-Path $Repo '3rdparty\tvm\pyproject.toml') `
    -Pattern '"(apache-tvm-ffi-orcjit[^"]*)"' | Select-Object -First 1
  $orcjit = if ($pin) { $pin.Matches[0].Groups[1].Value } else { 'apache-tvm-ffi-orcjit' }
  Run $orcjit $VenvPy @('-m', 'pip', 'install', '--no-deps', $orcjit)
}

# ---------------------------------------------------------------------------
if ($todo -contains 'mlc') {
  Say "`n== mlc ==" 'Yellow'
  $cfg = Join-Path $Repo 'config.cmake'
  if (Test-Path -LiteralPath $cfg) { Copy-Item -LiteralPath $cfg "$cfg.bak" -Force; Say "Backed up existing config.cmake to config.cmake.bak" }
  Set-Content -LiteralPath $cfg -Encoding ASCII -Value @"
set(TVM_SOURCE_DIR 3rdparty/tvm)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_ROCM $(ToCMake $RocmRoot))
set(USE_HIPBLAS ON)
set(USE_CUDA OFF)
set(USE_VULKAN OFF)
set(USE_METAL OFF)
set(USE_OPENCL OFF)
"@
  # Conda's GNU link.exe can shadow the MSVC linker and break the Rust build.
  $badLink = Get-Command link.exe -All -ErrorAction SilentlyContinue | Where-Object { $_.Source -match 'usr\\bin' }
  if ($badLink) { Say "[WARN] non-MSVC link.exe on PATH: $($badLink.Source -join ', ')" 'Yellow' }
  Reset-NonNinjaBuild (Join-Path $Repo 'build')
  Run 'mlc-llm' $VenvPy @('-m', 'pip', 'install', '-v', $Repo)
}

# ---------------------------------------------------------------------------
if ($todo -contains 'test') {
  Say "`n== test ==" 'Yellow'
  $test = Join-Path $env:TEMP 'mlc_rocm_smoke.py'
  Set-Content -LiteralPath $test -Encoding ASCII -Value @'
import ctypes, os, sys
import numpy as np
import tvm
from tvm.script import tirx as T

print("tvm", tvm.__file__)
dev = tvm.rocm(0)
print("rocm(0).exist:", dev.exist)
if not dev.exist:
    sys.exit("FAIL: TVM does not see a ROCm device")
print("device:", dev.device_name, "warp:", dev.warp_size)

k32 = ctypes.WinDLL("kernel32")
k32.GetModuleHandleW.restype = ctypes.c_void_p
k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
loaded = []
for name in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
    h = k32.GetModuleHandleW(name)
    if h:
        buf = ctypes.create_unicode_buffer(1024)
        k32.GetModuleFileNameW(h, buf, 1024)
        loaded.append(buf.value)
print("HIP runtime loaded from:", loaded)
if any("system32" in p.lower() for p in loaded):
    sys.exit("FAIL: the display driver's System32 HIP runtime is loaded")

target = tvm.target.Target("rocm")
print("target:", target)

n = 1 << 16
num_blocks = n // 256

# Exercises device-library math (ocml exp/sqrt), so the bitcode selection and ld.lld link
# are tested, not just codegen.
@T.prim_func
def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
    for bx in T.thread_binding(num_blocks, thread="blockIdx.x"):
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            B[bx * 256 + tx] = T.exp(A[bx * 256 + tx]) + T.sqrt(A[bx * 256 + tx])

lib = tvm.compile(kernel, target=target)
a_np = (np.arange(n, dtype="float32") % 1000) * 0.001
a = tvm.runtime.tensor(a_np, dev)
b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
lib(a, b)
ref = np.exp(a_np) + np.sqrt(a_np)
err = float(np.max(np.abs(b.numpy() - ref)))
print("max abs error:", err)
if err > 1e-3:
    sys.exit("FAIL: wrong results")

import mlc_llm
print("mlc_llm", mlc_llm.__file__)
print("PASS")
'@
  Run 'smoke test' $VenvPy @($test)
}

Say "`nDone: $($todo -join ', ')" 'Green'
Say "Log: $Log"

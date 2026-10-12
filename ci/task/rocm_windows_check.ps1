<#
.SYNOPSIS
  Checks that a system-wide ROCm (TheRock) install on Windows is configured and functional.

.DESCRIPTION
  1. Environment variables (Machine / User / current process)
  2. PATH (ROCm entries present, setx truncation, duplicates, missing dirs)
  3. Install layout (DLLs, headers, LLVM, device bitcode)
  4. Tools on PATH (amdclang, amdclang++, hipcc, hipconfig, clang, ld.lld, llvm-config) and shadowing
  5. HIP runtime DLL load + device enumeration (no compiler needed)
  6. TVM-style device code path: OpenCL -> amdgcn object (+device libs) -> ld.lld -> .hsaco
  7. HIP program: build with amdclang++ (fallback hipcc), run a kernel (expf/sqrtf) on the GPU, verify results,
     and load the .hsaco from step 6 with hipModuleLoad (what TVM does at runtime)
  8. What TVM/MLC-LLM's current code would find (expected failures are reported as INFO)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\rocm_windows_check.ps1
  powershell -ExecutionPolicy Bypass -File .\rocm_windows_check.ps1 -Arch gfx1100 -KeepTemp

  Run from a fresh terminal. If the HIP build cannot find C++ headers, run from
  "x64 Native Tools Command Prompt for VS" (needs MSVC + Windows SDK).
#>
param(
  [string]$Arch = "",
  [switch]$KeepTemp
)

$ErrorActionPreference = 'Continue'
$script:Fails = 0
$script:Warns = 0

function Pass($m)    { Write-Host "[PASS] $m" -ForegroundColor Green }
function Warn($m)    { Write-Host "[WARN] $m" -ForegroundColor Yellow; $script:Warns++ }
function Fail($m)    { Write-Host "[FAIL] $m" -ForegroundColor Red; $script:Fails++ }
function Info($m)    { Write-Host "[INFO] $m" }
function Section($m) { Write-Host ""; Write-Host "== $m ==" -ForegroundColor Cyan }

function Norm([string]$p) {
  if ([string]::IsNullOrWhiteSpace($p)) { return "" }
  return $p.Trim().Trim('"').TrimEnd('\').ToLowerInvariant()
}

function Run-Native([string]$exe, [string[]]$argList) {
  try {
    $out = & $exe @argList 2>&1 | ForEach-Object { "$_" }
    return [pscustomobject]@{ Code = $LASTEXITCODE; Out = ($out -join "`n") }
  } catch {
    return [pscustomobject]@{ Code = -1; Out = "could not start ${exe}: $($_.Exception.Message)" }
  }
}

function Import-VsEnv {
  if ($env:INCLUDE -and $env:VCToolsInstallDir) { return "already active" }
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

function Find-RocmTool([string]$name) {
  foreach ($d in @($script:llvmBin, $script:bin)) {
    if ($d) {
      $f = Join-Path $d "$name.exe"
      if (Test-Path -LiteralPath $f) { return $f }
    }
  }
  $c = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.Source }
  return $null
}

function Try-LoadLibrary([string]$name) {
  try { $h = [RocmProbe]::LoadLibraryW($name) } catch { return [IntPtr]::Zero }
  if ($null -eq $h) { return [IntPtr]::Zero }
  return $h
}

if (-not [Environment]::Is64BitProcess) {
  Fail "This is a 32-bit PowerShell. Use 64-bit PowerShell; ROCm DLLs are 64-bit."
}

# ---------------------------------------------------------------------------
Section "1. Environment variables"
$vars = @('HIP_PATH', 'ROCM_PATH', 'HIP_DEVICE_LIB_PATH', 'LLVM_PATH', 'HIP_PLATFORM')
$envv = @{}
foreach ($n in $vars) {
  $m = [Environment]::GetEnvironmentVariable($n, 'Machine')
  $u = [Environment]::GetEnvironmentVariable($n, 'User')
  $p = [Environment]::GetEnvironmentVariable($n, 'Process')
  $envv[$n] = $p
  $shown = if ([string]::IsNullOrEmpty($p)) { "<unset/empty>" } else { $p }
  Info ("{0,-20} process={1}" -f $n, $shown)
  if ((Norm $m) -ne (Norm $p) -and [string]::IsNullOrEmpty($u)) {
    Warn "$n differs between Machine ('$m') and this process ('$p'). Open a new terminal."
  }
  if ($n -ne 'HIP_PLATFORM' -and -not [string]::IsNullOrEmpty($p)) {
    if (Test-Path -LiteralPath $p -PathType Container) { Pass "$n points to an existing directory" }
    else { Fail "$n is set but '$p' does not exist" }
  }
}
if (-not [string]::IsNullOrEmpty($envv['HIP_PLATFORM']) -and $envv['HIP_PLATFORM'] -ne 'amd') {
  Fail "HIP_PLATFORM='$($envv['HIP_PLATFORM'])' (expected 'amd' or unset)"
}

# Determine ROCm root: HIP_PATH -> ROCM_PATH -> directory on PATH holding amdhip64*.dll
$root = $null
foreach ($n in @('HIP_PATH', 'ROCM_PATH')) {
  $v = $envv[$n]
  if (-not $root -and -not [string]::IsNullOrEmpty($v) -and (Test-Path -LiteralPath $v)) {
    $root = $v.TrimEnd('\'); Info "ROCm root from ${n}: $root"
  }
}
$procPathEntries = ($env:Path -split ';') | Where-Object { $_ -ne '' }
if (-not $root) {
  foreach ($e in $procPathEntries) {
    $e2 = $e.Trim('"')
    if ((Test-Path -LiteralPath $e2 -PathType Container) -and (Get-ChildItem -LiteralPath $e2 -Filter 'amdhip64*.dll' -ErrorAction SilentlyContinue)) {
      $root = (Split-Path -Parent $e2.TrimEnd('\')); Info "ROCm root derived from PATH (amdhip64*.dll in $e2): $root"; break
    }
  }
}
if (-not $root) {
  Fail "Cannot determine the ROCm root (no HIP_PATH/ROCM_PATH and no amdhip64*.dll on PATH). Stopping."
  exit 1
}
$bin     = Join-Path $root 'bin'
$llvmDir = if (-not [string]::IsNullOrEmpty($envv['LLVM_PATH'])) { $envv['LLVM_PATH'].TrimEnd('\') } else { Join-Path $root 'lib\llvm' }
$llvmBin = Join-Path $llvmDir 'bin'
$bcDir   = if (-not [string]::IsNullOrEmpty($envv['HIP_DEVICE_LIB_PATH'])) { $envv['HIP_DEVICE_LIB_PATH'].TrimEnd('\') } else { Join-Path $llvmDir 'amdgcn\bitcode' }

# ---------------------------------------------------------------------------
Section "2. PATH"
$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
Info "Machine PATH length: $($machinePath.Length) chars; process PATH length: $($env:Path.Length)"
if ($machinePath.Length -eq 1024 -or $machinePath.Length -eq 1023) {
  Fail "Machine PATH is exactly ~1024 chars: likely truncated by setx. Check the last entries."
}
$normProc = $procPathEntries | ForEach-Object { Norm $_ }
foreach ($need in @($bin, $llvmBin)) {
  if ($normProc -contains (Norm $need)) { Pass "PATH contains $need" } else { Fail "PATH is missing $need" }
}
$machineEntries = ($machinePath -split ';') | Where-Object { $_ -ne '' }
$dups = $machineEntries | ForEach-Object { Norm $_ } | Group-Object | Where-Object { $_.Count -gt 1 }
if ($dups) { Warn "Machine PATH has $(@($dups).Count) duplicated entries (typical of setx PATH ""%PATH%;..."")"; $dups | ForEach-Object { Info "  dup: $($_.Name)" } }
else { Pass "No duplicate Machine PATH entries" }
$missing = $machineEntries | Where-Object { -not (Test-Path -LiteralPath ($_.Trim('"')) -ErrorAction SilentlyContinue) -and ($_ -notmatch '%') }
if ($missing) { Warn "Machine PATH entries that do not exist (a cut-off last entry means truncation):"; $missing | ForEach-Object { Info "  missing: $_" } }

# ---------------------------------------------------------------------------
Section "3. Install layout under $root"
$hipDlls = @(Get-ChildItem -LiteralPath $bin -Filter 'amdhip64*.dll' -ErrorAction SilentlyContinue)
if ($hipDlls.Count -gt 0) { Pass "HIP runtime DLL: $($hipDlls.Name -join ', ')" } else { Fail "No amdhip64*.dll in $bin" }
$checks = @(
  @{ P = 'include\hip\hip_runtime.h'; Req = $true;  Why = 'HIP headers' },
  @{ P = 'lib\amdhip64.lib';          Req = $true;  Why = 'HIP import library (needed to build TVM)' },
  @{ P = 'lib\cmake\hip\hip-config.cmake'; Req = $false; Why = 'CMake package config' }
)
foreach ($c in $checks) {
  $f = Join-Path $root $c.P
  if (Test-Path -LiteralPath $f) { Pass "$($c.P) ($($c.Why))" }
  elseif ($c.Req) { Fail "Missing $($c.P) ($($c.Why))" } else { Warn "Missing $($c.P) ($($c.Why))" }
}
foreach ($t in @('clang.exe', 'ld.lld.exe', 'llvm-config.exe')) {
  $f = Join-Path $llvmBin $t
  if (Test-Path -LiteralPath $f) { Pass "$t in $llvmBin" }
  elseif ($t -eq 'llvm-config.exe') { Warn "No llvm-config.exe: TVM cannot be built against ROCm's own LLVM (needs a matching LLVM build)" }
  else { Fail "Missing $f" }
}
foreach ($lib in @('hipblas', 'hipblaslt', 'rocblas', 'rccl', 'hsa-runtime64')) {
  $hit = @(Get-ChildItem -LiteralPath $bin -Filter "*$lib*.dll" -ErrorAction SilentlyContinue)
  if ($hit.Count) { Info "$lib DLL present: $($hit.Name -join ', ')" } else { Info "$lib DLL not present" }
}
if (Test-Path -LiteralPath $bcDir -PathType Container) {
  Pass "Device bitcode dir: $bcDir"
  foreach ($b in @('ocml.bc', 'ockl.bc')) {
    if (Test-Path -LiteralPath (Join-Path $bcDir $b)) { Pass $b } else { Fail "Missing $b in $bcDir" }
  }
  $isa = @(Get-ChildItem -LiteralPath $bcDir -Filter 'oclc_isa_version_*.bc' | ForEach-Object { $_.BaseName -replace 'oclc_isa_version_', '' })
  Info "ISA bitcode files: $($isa.Count) ($($isa -join ' '))"
  $abi = @(Get-ChildItem -LiteralPath $bcDir -Filter 'oclc_abi_version_*.bc' | ForEach-Object { $_.BaseName })
  Info "ABI bitcode files: $($abi -join ' ')"
} else {
  Fail "Device bitcode dir not found: $bcDir"
}

# ---------------------------------------------------------------------------
Section "4. Tools on PATH"
foreach ($t in @('amdclang', 'amdclang++', 'hipcc', 'hipconfig', 'hipInfo', 'clang', 'ld.lld', 'llvm-config', 'amdgpu-arch', 'offload-arch')) {
  $cmd = Get-Command $t -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $cmd) { Info "$t not on PATH"; continue }
  $src = $cmd.Source
  if ((Norm $src).StartsWith((Norm $root))) { Pass "$t -> $src" }
  else { Warn "$t resolves OUTSIDE ROCm: $src (another toolchain earlier on PATH shadows ROCm's)" }
}
$hipconfig = Get-Command hipconfig -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($hipconfig) {
  $r = Run-Native $hipconfig.Source @('--version'); Info "hipconfig --version: $($r.Out.Trim())"
  $r = Run-Native $hipconfig.Source @('--platform'); Info "hipconfig --platform: $($r.Out.Trim())"
}
$clangExe = Find-RocmTool 'amdclang'
if (-not $clangExe) { $clangExe = Join-Path $llvmBin 'clang.exe'; Warn "amdclang not found; using $clangExe" }
$clangxxExe = Find-RocmTool 'amdclang++'
if (-not $clangxxExe) { Warn "amdclang++ not found" }
foreach ($c in @($clangExe, $clangxxExe)) {
  if ($c -and (Test-Path -LiteralPath $c)) { $r = Run-Native $c @('--version'); Info "$(Split-Path -Leaf $c): $(($r.Out -split "`n")[0])" }
}

# ---------------------------------------------------------------------------
Section "5. HIP runtime DLL load (no compiler)"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class RocmProbe {
  [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
  public static extern IntPtr LoadLibraryW(string name);
  [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true)]
  public static extern IntPtr GetProcAddress(IntPtr h, string name);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  public static extern uint GetModuleFileNameW(IntPtr h, StringBuilder sb, uint size);
  [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate int IntOutFn(out int v);
  [UnmanagedFunctionPointer(CallingConvention.Cdecl)] public delegate int NameFn(byte[] buf, int len, int dev);
  public static int CallIntOut(IntPtr h, string fn, out int v) {
    v = 0;
    IntPtr p = GetProcAddress(h, fn);
    if (p == IntPtr.Zero) return -1000;
    IntOutFn d = (IntOutFn)Marshal.GetDelegateForFunctionPointer(p, typeof(IntOutFn));
    return d(out v);
  }
  public static string DeviceName(IntPtr h, int dev) {
    IntPtr p = GetProcAddress(h, "hipDeviceGetName");
    if (p == IntPtr.Zero) return "<hipDeviceGetName missing>";
    NameFn d = (NameFn)Marshal.GetDelegateForFunctionPointer(p, typeof(NameFn));
    byte[] buf = new byte[256];
    int rc = d(buf, buf.Length, dev);
    if (rc != 0) return "<error " + rc + ">";
    int n = Array.IndexOf(buf, (byte)0); if (n < 0) n = buf.Length;
    return Encoding.ASCII.GetString(buf, 0, n);
  }
  public static string ModulePath(IntPtr h) {
    StringBuilder sb = new StringBuilder(1024);
    GetModuleFileNameW(h, sb, 1024);
    return sb.ToString();
  }
}
"@
$deviceCount = 0
if ($hipDlls.Count -gt 0) {
  $dllName = $hipDlls[0].Name
  $h = Try-LoadLibrary $dllName   # bare name: resolved through the DLL search path / PATH
  if ($h -eq [IntPtr]::Zero) {
    $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Fail "LoadLibrary('$dllName') by name failed (Win32 error $err): not resolvable via PATH"
    $h = Try-LoadLibrary (Join-Path $bin $dllName)
    if ($h -eq [IntPtr]::Zero) { Fail "LoadLibrary by full path also failed (Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))" }
  } else {
    $loaded = [RocmProbe]::ModulePath($h)
    Pass "Loaded $dllName by name from $loaded"
    $rockDll = Join-Path $bin $dllName
    if ((Norm $loaded) -ne (Norm $rockDll)) {
      $vLoaded = (Get-Item -LiteralPath $loaded).VersionInfo.FileVersion
      $vRock = (Get-Item -LiteralPath $rockDll).VersionInfo.FileVersion
      Warn "Windows loaded the driver's copy, not ROCm's (System32 is searched before PATH)"
      Info "  loaded : $loaded  (file version $vLoaded)"
      Info "  ROCm   : $rockDll  (file version $vRock)"
    }
  }
  if ($h -ne [IntPtr]::Zero) {
    $v = 0
    if ([RocmProbe]::CallIntOut($h, 'hipRuntimeGetVersion', [ref]$v) -eq 0) {
      Info ("HIP runtime version: {0}.{1}.{2} (raw {3})" -f [math]::Floor($v / 10000000), [math]::Floor(($v % 10000000) / 100000), ($v % 100000), $v)
    }
    if ([RocmProbe]::CallIntOut($h, 'hipDriverGetVersion', [ref]$v) -eq 0) { Info "HIP driver version (raw): $v" }
    $rc = [RocmProbe]::CallIntOut($h, 'hipGetDeviceCount', [ref]$deviceCount)
    if ($rc -eq 0 -and $deviceCount -gt 0) {
      Pass "hipGetDeviceCount = $deviceCount"
      for ($d = 0; $d -lt $deviceCount; $d++) { Info "  device ${d}: $([RocmProbe]::DeviceName($h, $d))" }
    } elseif ($rc -eq 100 -or $deviceCount -eq 0) {
      Fail "HIP sees no GPU (hipGetDeviceCount rc=$rc). Check the Adrenalin driver and that the GPU is supported."
    } else {
      Fail "hipGetDeviceCount failed with HIP error $rc"
    }
  }
}

# ---------------------------------------------------------------------------
Section "Target architecture"
if (-not $Arch) {
  foreach ($t in @('amdgpu-arch', 'offload-arch')) {
    $exe = Join-Path $llvmBin "$t.exe"
    if (-not $Arch -and (Test-Path -LiteralPath $exe)) {
      $r = Run-Native $exe @()
      $m = [regex]::Match($r.Out, 'gfx[0-9a-f]+')
      if ($r.Code -eq 0 -and $m.Success) { $Arch = $m.Value; Info "Detected with ${t}: $Arch" }
    }
  }
}
if (-not $Arch) {
  $hi = Get-Command hipInfo -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($hi) {
    $r = Run-Native $hi.Source @()
    $m = [regex]::Match($r.Out, 'gcnArchName:\s*(gfx[0-9a-f]+)')
    if ($m.Success) { $Arch = $m.Groups[1].Value; Info "Detected with hipInfo: $Arch" }
  }
}
if ($Arch) { $Arch = ($Arch -split ':')[0]; Pass "Using arch $Arch" }
else { Warn "Could not detect GPU arch; pass -Arch gfxXXXX. Steps 6-7 will be skipped." }

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("rocm_check_" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $tmp | Out-Null
$hsaco = $null

# ---------------------------------------------------------------------------
Section "6. TVM-style compile + link (.cl -> amdgcn object -> ld.lld -> .hsaco)"
$lldExe = Join-Path $llvmBin 'ld.lld.exe'
if ($Arch -and (Test-Path -LiteralPath $clangExe) -and (Test-Path -LiteralPath $lldExe)) {
  $cl = Join-Path $tmp 'k.cl'
  Set-Content -LiteralPath $cl -Encoding ASCII -Value @"
__kernel void vexp_cl(__global float* y, __global const float* x, int n) {
  int i = get_global_id(0);
  if (i < n) y[i] = exp(x[i]);
}
"@
  $obj = Join-Path $tmp 'k.o'
  $r = Run-Native $clangExe @('-x', 'cl', '-cl-std=CL2.0', '-Xclang', '-finclude-default-header',
    '--target=amdgcn-amd-amdhsa', "-mcpu=$Arch", "--rocm-device-lib-path=$bcDir", '-O3', '-c', $cl, '-o', $obj)
  if ($r.Code -eq 0) {
    Pass "$(Split-Path -Leaf $clangExe) compiled kernel for $Arch with device libs from $bcDir"
    $out = Join-Path $tmp 'k.hsaco'
    $r = Run-Native $lldExe @('--no-undefined', '-shared', $obj, '-o', $out)
    if ($r.Code -eq 0) { Pass "ld.lld --no-undefined -shared produced $out"; $hsaco = $out }
    else { Fail "ld.lld link failed:`n$($r.Out)" }
  } else {
    Fail "$(Split-Path -Leaf $clangExe) device compile failed:`n$($r.Out)"
  }
} else {
  Info "Skipped (needs arch, clang.exe and ld.lld.exe)"
}

# ---------------------------------------------------------------------------
Section "7. HIP program: build, run kernel on GPU, load .hsaco"
$hipcc = Find-RocmTool 'hipcc'
if ($Arch -and ($clangxxExe -or $hipcc) -and $deviceCount -gt 0) {
  $src = Join-Path $tmp 'check.hip'
  Set-Content -LiteralPath $src -Encoding ASCII -Value @'
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::printf("HIPERR %s: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void vexp(float* y, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = expf(x[i]) + sqrtf(x[i]);
}

int main(int argc, char** argv) {
  int count = 0;
  CHECK(hipGetDeviceCount(&count));
  for (int d = 0; d < count; ++d) {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, d));
    std::printf("DEVICE %d: %s arch=%s warpSize=%d CUs=%d mem=%.1fGiB\n", d, p.name,
                p.gcnArchName, p.warpSize, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
  }
  const int n = 1 << 20;
  std::vector<float> hx(n), hy(n);
  for (int i = 0; i < n; ++i) hx[i] = (i % 1000) * 0.001f;
  float *dx = nullptr, *dy = nullptr;
  CHECK(hipMalloc(&dx, n * sizeof(float)));
  CHECK(hipMalloc(&dy, n * sizeof(float)));
  CHECK(hipMemcpy(dx, hx.data(), n * sizeof(float), hipMemcpyHostToDevice));
  vexp<<<(n + 255) / 256, 256>>>(dy, dx, n);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hy.data(), dy, n * sizeof(float), hipMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    float ref = std::exp(hx[i]) + std::sqrt(hx[i]);
    if (std::fabs(hy[i] - ref) > 1e-4f * (1.0f + std::fabs(ref))) ++bad;
  }
  std::printf("KERNEL_MISMATCHES=%d\n", bad);
  if (argc > 1) {
    hipModule_t m;
    hipFunction_t f;
    CHECK(hipModuleLoad(&m, argv[1]));
    CHECK(hipModuleGetFunction(&f, m, "vexp_cl"));
    std::printf("HSACO_LOAD=OK\n");
    CHECK(hipModuleUnload(m));
  }
  CHECK(hipFree(dx));
  CHECK(hipFree(dy));
  return bad ? 3 : 0;
}
'@
  $vsEnv = Import-VsEnv
  if ($vsEnv) { Info "MSVC environment: $vsEnv" }
  else { Warn "No MSVC environment found (Visual Studio 2022 / Build Tools with 'Desktop development with C++'). On Windows, amdclang++ and hipcc both use the MSVC C++ headers and CRT." }
  $exe = Join-Path $tmp 'check.exe'
  $libDir = Join-Path $root 'lib'
  $attempts = @()
  if ($clangxxExe) {
    $attempts += ,@{ Name = 'amdclang++'; Exe = $clangxxExe; Args = @('-x', 'hip', "--offload-arch=$Arch", "--rocm-path=$root",
      "--rocm-device-lib-path=$bcDir", '-O2', $src, '--hip-link', '-fuse-ld=lld', "-L$libDir", '-lamdhip64', '-o', $exe) }
  }
  if ($hipcc) {
    $attempts += ,@{ Name = 'hipcc'; Exe = $hipcc; Args = @("--offload-arch=$Arch", '-O2', $src, '-o', $exe) }
  }
  $built = $null
  foreach ($a in $attempts) {
    Remove-Item -LiteralPath $exe -ErrorAction SilentlyContinue
    $r = Run-Native $a.Exe $a.Args
    if ($r.Code -eq 0 -and (Test-Path -LiteralPath $exe)) { Pass "$($a.Name) built a HIP program for $Arch"; $built = $a.Name; break }
    Warn "$($a.Name) build failed:`n$($r.Out)"
    if ($r.Out -match 'file not found|Could not find standard C\+\+ header') {
      $v = Run-Native $a.Exe (@('-v', '-fsyntax-only') + ($a.Args | Where-Object { $_ -ne '-o' -and $_ -ne $exe -and $_ -ne '--hip-link' }))
      $inc = $false
      $lines = foreach ($l in ($v.Out -split "`n")) {
        if ($l -match 'search starts here') { $inc = $true; continue }
        if ($l -match 'End of search list') { $inc = $false; continue }
        if ($inc) { $l.Trim() }
      }
      Info "$($a.Name) C++ include search path (no MSVC 'VC\Tools\MSVC\...\include' entry means MSVC headers are not visible):"
      $lines | Select-Object -Unique | ForEach-Object { Info "  $_" }
    }
  }
  if (-not $built) { Fail "HIP program could not be built with amdclang++ or hipcc (see warnings above)" }
  if ($built) {
    $runArgs = @(); if ($hsaco) { $runArgs = @($hsaco) }
    $r = Run-Native $exe $runArgs
    $r.Out -split "`n" | Where-Object { $_ -match '^DEVICE' } | ForEach-Object { Info $_ }
    if ($r.Out -match 'KERNEL_MISMATCHES=0') { Pass "Kernel ran on GPU; expf/sqrtf results match CPU" }
    elseif ($r.Out -match 'KERNEL_MISMATCHES=(\d+)') { Fail "Kernel ran but $($Matches[1]) results are wrong" }
    else { Fail "Program failed (exit $($r.Code)):`n$($r.Out)" }
    if ($hsaco) {
      if ($r.Out -match 'HSACO_LOAD=OK') { Pass "hipModuleLoad loaded the ld.lld-linked .hsaco (TVM runtime path works)" }
      else { Fail "hipModuleLoad of the .hsaco failed:`n$(($r.Out -split "`n" | Where-Object { $_ -match 'HIPERR' }) -join "`n")" }
    }
  }
} else {
  Info "Skipped (needs arch, amdclang++ or hipcc, and a visible GPU)"
}

# ---------------------------------------------------------------------------
Section "8. What TVM/MLC-LLM's current ROCm code would find (expected gaps, INFO only)"
$rp = [Environment]::GetEnvironmentVariable('ROCM_PATH', 'Process')
if ([string]::IsNullOrEmpty($rp)) { Info "find_rocm_path(): ROCM_PATH unset -> falls back to 'which hipcc' (no 'which' on Windows) -> /opt/rocm -> RuntimeError" }
else { Info "find_rocm_path(): would use ROCM_PATH=$rp" }
$tvmBc = Join-Path $root 'amdgcn\bitcode'
if (Test-Path -LiteralPath $tvmBc) { Info "TVM bitcode dir <root>\amdgcn\bitcode exists" }
else { Info "TVM looks for <root>\amdgcn\bitcode: NOT found (actual: $bcDir)" }
if (Test-Path -LiteralPath $bcDir) {
  $tvmRequired = @('oclc_daz_opt_on', 'ocml', 'oclc_correctly_rounded_sqrt_off', 'oclc_correctly_rounded_sqrt_on',
    'oclc_daz_opt_off', 'oclc_finite_only_off', 'oclc_finite_only_on', 'oclc_unsafe_math_off',
    'oclc_unsafe_math_on', 'oclc_wavefrontsize64_on', 'oclc_abi_version_500')
  $gone = $tvmRequired | Where-Object { -not (Test-Path -LiteralPath (Join-Path $bcDir "$_.bc")) }
  if ($gone) { Info "Bitcode files TVM requires but this install lacks: $($gone -join ', ')" }
  else { Info "All bitcode files TVM requires are present" }
  if ($Arch) {
    $isaFile = 'oclc_isa_version_' + ($Arch -replace '^gfx', '') + '.bc'
    $listed = @('803', '900', '906', '1030') -contains ($Arch -replace '^gfx', '')
    if (-not $listed) { Info "  -> TVM would not link oclc_isa_version for $Arch (its list is fixed to 803/900/906/1030)" }
    Info "ISA file for ${Arch}: $(if (Test-Path -LiteralPath (Join-Path $bcDir $isaFile)) {'present'} else {'missing'}); in TVM's fixed list: $listed"
  }
}
Info "TVM's which('ld.lld') does not append .exe, so it will not find ld.lld.exe even though it is on PATH."
Info "hsa-runtime64 (linked by TVM's ROCM.cmake): $(if (Get-ChildItem -LiteralPath $bin -Filter '*hsa-runtime64*' -ErrorAction SilentlyContinue) {'present'} else {'not present'})"

# ---------------------------------------------------------------------------
if (-not $KeepTemp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
else { Info "Temp files kept in $tmp" }

Section "Summary"
if ($script:Fails -eq 0) {
  Write-Host "ROCm environment is functional ($($script:Warns) warning(s))." -ForegroundColor Green
  exit 0
} else {
  Write-Host "$($script:Fails) failure(s), $($script:Warns) warning(s)." -ForegroundColor Red
  exit 1
}

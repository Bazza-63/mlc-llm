<#
.SYNOPSIS
  Initialises all git submodules recursively without `git submodule`.

.DESCRIPTION
  On some Windows machines `git submodule update` fails because Git for Windows runs it
  through its bundled sh, and sh cannot start child processes ("dofork: child ... died
  unexpectedly, exit code 0xC0000142"). This script uses only git's built-in commands
  (config, ls-tree, init, fetch, checkout), so no sh is needed.

  For every submodule in .gitmodules it reads the commit pinned by the parent repo, fetches
  just that commit (shallow by default) and checks it out, then recurses into it. It also
  records the submodule in the parent's .git/config so normal git commands treat it as
  initialised. Submodules that are already at the pinned commit are left alone.

.EXAMPLE
  cd C:\Users\Geeza\code\mlc-llm
  powershell -ExecutionPolicy Bypass -File .\ci\task\init_submodules.ps1
  powershell -ExecutionPolicy Bypass -File .\ci\task\init_submodules.ps1 -Full   # full history
#>
param(
  [string]$Root = (Get-Location).Path,
  [switch]$Full
)

$ErrorActionPreference = 'Continue'
$script:Failed = @()

function Git-Out([string]$repo, [string[]]$argList) {
  $out = & git -C $repo @argList 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  return $out
}

function Git-Run([string]$repo, [string[]]$argList) {
  & git -C $repo @argList
  return ($LASTEXITCODE -eq 0)
}

function Resolve-SubmoduleUrl([string]$repo, [string]$url) {
  if ($url -notmatch '^\.\.?/') { return $url }
  # Relative URL: resolve against the parent's origin.
  $base = (Git-Out $repo @('remote', 'get-url', 'origin'))
  if (-not $base) { throw "Relative submodule URL '$url' but $repo has no origin remote" }
  $base = ($base | Select-Object -First 1).Trim().TrimEnd('/')
  foreach ($part in ($url -split '/')) {
    if ($part -eq '..') { $base = $base.Substring(0, $base.LastIndexOf('/')) }
    elseif ($part -ne '.' -and $part -ne '') { $base = "$base/$part" }
  }
  return $base
}

function Init-Submodules([string]$repo, [string]$label) {
  if (-not (Test-Path -LiteralPath (Join-Path $repo '.gitmodules'))) { return }
  $paths = Git-Out $repo @('config', '-f', '.gitmodules', '--get-regexp', '^submodule\..*\.path$')
  foreach ($line in @($paths)) {
    if (-not $line) { continue }
    $key, $path = $line -split ' ', 2
    $name = $key.Substring(10, $key.Length - 15)   # strip "submodule." and ".path"
    $display = if ($label) { "$label/$path" } else { $path }

    $url = Git-Out $repo @('config', '-f', '.gitmodules', '--get', "submodule.$name.url")
    if (-not $url) { Write-Warning "${display}: no URL in .gitmodules, skipped"; continue }
    $url = Resolve-SubmoduleUrl $repo ($url | Select-Object -First 1).Trim()

    $tree = Git-Out $repo @('ls-tree', 'HEAD', '--', $path)
    if (-not $tree -or ($tree -split '\s+')[0] -ne '160000') {
      Write-Warning "${display}: not a submodule in the parent's HEAD, skipped"
      continue
    }
    $sha = ($tree -split '\s+')[2]
    $dest = Join-Path $repo $path

    $head = $null
    if (Test-Path -LiteralPath (Join-Path $dest '.git')) { $head = Git-Out $dest @('rev-parse', 'HEAD') }
    if ($head -eq $sha) {
      Write-Host "[ok]    $display @ $($sha.Substring(0, 7))"
    } else {
      if (-not (Test-Path -LiteralPath (Join-Path $dest '.git'))) {
        if ((Test-Path -LiteralPath $dest) -and (Get-ChildItem -LiteralPath $dest -Force | Select-Object -First 1)) {
          Write-Warning "${display}: directory is not empty and not a git repo, skipped"
          $script:Failed += $display
          continue
        }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        [void](Git-Run $dest @('init', '-q'))
        [void](Git-Run $dest @('remote', 'add', 'origin', $url))
      }
      Write-Host "[fetch] $display @ $($sha.Substring(0, 7)) from $url"
      $fetchArgs = @('fetch', '-q', 'origin', $sha)
      if (-not $Full) { $fetchArgs = @('fetch', '-q', '--depth', '1', 'origin', $sha) }
      $ok = Git-Run $dest $fetchArgs
      if (-not $ok) {
        # Some servers refuse fetching a commit by SHA; fall back to fetching everything.
        Write-Host "        fetch by commit refused, fetching all branches and tags"
        $ok = Git-Run $dest @('fetch', '-q', '--tags', 'origin', '+refs/heads/*:refs/remotes/origin/*')
      }
      if ($ok) { $ok = Git-Run $dest @('-c', 'advice.detachedHead=false', 'checkout', '-q', '--detach', $sha) }
      if (-not $ok) {
        Write-Warning "${display}: could not check out $sha"
        $script:Failed += $display
        continue
      }
    }

    # Record it in the parent so `git status` / `git submodule status` see it as initialised.
    [void](Git-Run $repo @('config', "submodule.$name.url", $url))
    [void](Git-Run $repo @('config', "submodule.$name.active", 'true'))

    Init-Submodules $dest $display
  }
}

$Root = (Resolve-Path -LiteralPath $Root).Path
if (-not (Git-Out $Root @('rev-parse', '--show-toplevel'))) { throw "$Root is not a git repository" }
Init-Submodules $Root ''

Write-Host ''
if ($script:Failed.Count -gt 0) {
  Write-Host "Failed: $($script:Failed -join ', ')" -ForegroundColor Red
  exit 1
}
Write-Host 'All submodules are at their pinned commits.' -ForegroundColor Green

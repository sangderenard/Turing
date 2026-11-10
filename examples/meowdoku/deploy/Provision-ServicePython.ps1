<#
.SYNOPSIS  Provision the PRIVATE service interpreter under <Root>/python from an already-downloaded python-build-standalone archive.
Verifies the archive sha256 (refuses on mismatch), extracts with tar into <Root>, installs requirements-service.txt (including the exact Torch build) with that interpreter's
own pip, writes <Root>/python/VERSION.txt. No elevation. Touches nothing outside <Root>: no PATH, no registry, no system-wide Python.
Download is NOT done here. Asset: cpython-3.11.18+20261026-x86_64-pc-windows-msvc-install_only.tar.gz (release 20261026 of astral-sh/python-build-standalone).
-WhatIf prints every command and runs none (the hash is still checked). See LAUNCH_2026-11-10.md "Private service interpreter".
#>
param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [string]$Root = 'C:/dev/Powershell/srv',
    [string]$Requirements = '',   # default: requirements-service.txt beside this script (Torch build); pass requirements-service-notorch.txt for the Torch-free service
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'
$AssetName    = 'cpython-3.11.18+20261026-x86_64-pc-windows-msvc-install_only.tar.gz'
$ExpectedSha  = '75cb501dc11c7b55c6b447c58e4910bce5b1f609f1017328b4f3a858dfaf5428'
if (-not $Requirements) { $Requirements = Join-Path $PSScriptRoot 'requirements-service.txt' }
$PyDir = "$Root/python"
$Py    = "$PyDir/python.exe"

if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "REFUSED: archive not found: $Archive" }
if (-not (Test-Path -LiteralPath $Requirements)) { throw "REFUSED: missing $Requirements" }
$actual = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "archive  : $Archive"
Write-Host "sha256   : $actual"
if ($actual -ne $ExpectedSha) { Write-Host "REFUSED: sha256 mismatch (expected $ExpectedSha)."; exit 1 }
Write-Host 'sha256 matches.'
if (Test-Path -LiteralPath $PyDir) { throw "REFUSED: $PyDir already exists; remove it by hand to re-provision." }

function Cmd($t) { Write-Host "cmd: $t" }
Cmd "New-Item -ItemType Directory -Force $Root"
# Windows' own bsdtar by absolute path: a Git-for-Windows tar earlier on PATH reads "C:" as a remote host.
$Tar = Join-Path $env:SystemRoot 'System32\tar.exe'
Cmd "$Tar -xzf $Archive -C $Root     (creates $PyDir)"
if (-not $WhatIf) {
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    & $Tar -xzf $Archive -C $Root
    if ($LASTEXITCODE -ne 0) { throw "tar failed with exit $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $Py)) { throw "extraction did not produce $Py" }
}
Write-Host 'NOTE: the install downloads about 2.5 GB (the torch 2.5.1+cu124 wheel dominates); --no-cache-dir keeps it from being cached.'
Cmd "$Py -m pip install --no-cache-dir -r $Requirements"
if (-not $WhatIf) {
    & $Py -m pip install --no-cache-dir -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit $LASTEXITCODE" }
}
Cmd "write $PyDir/VERSION.txt (asset name + sha256)"
if (-not $WhatIf) {
    Set-Content -LiteralPath "$PyDir/VERSION.txt" -Value @($AssetName, "sha256 $ExpectedSha") -Encoding ascii
}
Write-Host ''
Write-Host 'Verify with:'
Write-Host "$Py -c `"import sys,numpy,sympy,llvmlite,torch; print(sys.version); print(torch.__version__, torch.version.cuda)`""
Write-Host "NODUS_TORCH_ROOT for the Nodus service build: $PyDir/Lib/site-packages/torch"
if ($WhatIf) { Write-Host '[WhatIf] nothing run.' }

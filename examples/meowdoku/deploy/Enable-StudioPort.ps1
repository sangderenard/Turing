<#
.SYNOPSIS  One stop window that gives the live canvas its Studio contract port and brings the stack back up.
A canvas host serves only the contract ports its canvas file declares, and the file can only be extended while the
host that owns its ports is stopped. This script: refuses if a photo exposure is running; stops the stack through
Stop-Apache.ps1 (which carries the operator gate); waits for httpd to exit; backs up the canvas; adds the port with
nodus tools/add_contract_port.py; replaces the canvas only if that succeeded; starts the stack through
Start-Apache.ps1. Running it again after success changes nothing but restarts the stack.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Enable-StudioPort.ps1
#>
param([string]$Prefix = 'studio.product', [int]$Port = 47613, [int]$Capacity = 4, [int]$Loan = 7200,
      [string]$Keep = 'gui.proof:4:54000', [switch]$Force)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$workspace = Split-Path $repo -Parent
$canvas = Join-Path $repo 'build\meowdoku\canvas\messenger.canvas'
$python = Join-Path $workspace 'srv\python\python.exe'
$library = Join-Path $workspace 'nodus\build-service\Release'
if (-not $env:MEOWDOKU_STATE_DIR) { throw 'Set MEOWDOKU_STATE_DIR to the live state root first.' }

# A running exposure dies with the stack. Its worker is a child of the stable server.
$photos = Join-Path $env:MEOWDOKU_STATE_DIR 'content\marble_photos'
$recent = Get-ChildItem $photos -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName 'status.json' } |
    Where-Object { (Test-Path $_) -and ((Get-Item $_).LastWriteTime -gt (Get-Date).AddMinutes(-10)) } |
    Where-Object { (Get-Content $_ -Raw | ConvertFrom-Json).state -notin @('complete', 'error', 'canceled', 'failed') }
if ($recent -and -not $Force) { throw "A photo job changed state in the last ten minutes and is not finished: $($recent -join ', '). Wait, or pass -Force." }

& (Join-Path $PSScriptRoot 'Stop-Apache.ps1')
$end = (Get-Date).AddSeconds(90)
while ((Get-Process httpd -ErrorAction SilentlyContinue) -and (Get-Date) -lt $end) { Start-Sleep -Milliseconds 500 }
if (Get-Process httpd -ErrorAction SilentlyContinue) { throw 'httpd is still running after 90 s; the stack was not started again.' }

if (Select-String -Path $canvas -Pattern ("^CONTRACT_PORT " + [regex]::Escape($Prefix) + " ") -Quiet) {
    "The canvas already declares $Prefix; nothing to add."
} else {
    Copy-Item $canvas "$canvas.before-$Prefix" -Force
    Remove-Item "$canvas.new" -ErrorAction SilentlyContinue
    $saved = $env:PATH
    $env:NODUS_HEADLESS_TEST_DLL = (Join-Path $library 'nodus_headless.dll').Replace('\', '/')
    $env:PYTHONUTF8 = '1'
    $env:PATH = (Join-Path $workspace 'srv\python\Lib\site-packages\torch\lib') + ';' + $library + ';' + $env:PATH
    $ErrorActionPreference = 'Continue'
    & $python (Join-Path $workspace 'nodus\tools\add_contract_port.py') --canvas $canvas --out "$canvas.new" --prefix $Prefix `
        --capacity $Capacity --port $Port --loan $Loan --keep $Keep 2>&1 | Select-String 'ports|rror|refused'
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $env:PATH = $saved
    if ($code -eq 0 -and (Test-Path "$canvas.new")) { Move-Item "$canvas.new" $canvas -Force; "Canvas now declares $Prefix on port $Port." }
    else { "The port was NOT added (exit $code); the old canvas is unchanged and pictures stay off." }
}
& (Join-Path $PSScriptRoot 'Start-Apache.ps1')

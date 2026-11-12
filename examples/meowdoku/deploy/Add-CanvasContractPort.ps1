<#
.SYNOPSIS  One stop window that gives the live canvas one more contract port and brings the stack back up.
A canvas host serves only the contract ports its canvas file declares, and the file can only be extended while the
host that owns its ports is stopped. This script: refuses if a photo exposure is running; stops the stack through
Stop-Apache.ps1 (which carries the operator gate); waits for httpd to exit; optionally rebuilds the live Nodus
library (-BuildLibrary, for a port whose tool is newer than the library); backs up the canvas; adds the port with
nodus tools/add_contract_port.py, keeping every port the canvas already declares; replaces the canvas only if that
succeeded; starts the stack through Start-Apache.ps1. Running it again after success only restarts the stack.
Whatever fails in the middle, the stack is started again with what it had.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Add-CanvasContractPort.ps1 -Prefix studio.product -Port 47613 -Loan 7200
#>
param([Parameter(Mandatory)][string]$Prefix, [Parameter(Mandatory)][int]$Port, [int]$Capacity = 4, [int]$Loan = 60,
      [switch]$BuildLibrary, [switch]$Force)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$workspace = Split-Path $repo -Parent
$canvas = Join-Path $repo 'build\meowdoku\canvas\messenger.canvas'
$python = Join-Path $workspace 'srv\python\python.exe'
$buildTree = Join-Path $workspace 'nodus\build-service'
$library = Join-Path $buildTree 'Release'
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

try {
    if ($BuildLibrary) {
        $ErrorActionPreference = 'Continue'
        $started = Get-Date
        & cmake --build $buildTree --config Release --target nodus_headless 2>&1 | Select-String 'error|nodus_headless.vcxproj ->'
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        "Library build exit $code in $([int]((Get-Date) - $started).TotalSeconds) s."
        if ($code -ne 0) { 'The library was NOT rebuilt; the old one stays in use and a tool that needs the new one stays off.' }
    }
    if (Select-String -Path $canvas -Pattern ("^CONTRACT_PORT " + [regex]::Escape($Prefix) + " ") -Quiet) {
        "The canvas already declares $Prefix; nothing to add."
    } else {
        # Keep every port the canvas declares: CONTRACT_PORT <prefix> <capacity> <a> <ai> <b> <bi> <port> [loan]
        $keep = @()
        foreach ($line in (Select-String -Path $canvas -Pattern '^CONTRACT_PORT ').Line) {
            $f = $line.Trim() -split '\s+'
            $keep += '--keep'
            $keep += if ($f.Count -ge 9) { "$($f[1]):$($f[2]):$($f[8])" } else { "$($f[1]):$($f[2])" }
        }
        Copy-Item $canvas "$canvas.before-$Prefix" -Force
        Remove-Item "$canvas.new" -ErrorAction SilentlyContinue
        $saved = $env:PATH
        $env:NODUS_HEADLESS_TEST_DLL = (Join-Path $library 'nodus_headless.dll').Replace('\', '/')
        $env:PYTHONUTF8 = '1'
        $env:PATH = (Join-Path $workspace 'srv\python\Lib\site-packages\torch\lib') + ';' + $library + ';' + $env:PATH
        $ErrorActionPreference = 'Continue'
        & $python (Join-Path $workspace 'nodus\tools\add_contract_port.py') --canvas $canvas --out "$canvas.new" --prefix $Prefix `
            --capacity $Capacity --port $Port --loan $Loan @keep 2>&1 | Select-String "'ports'|rror|refused"
        $code = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        $env:PATH = $saved
        if ($code -eq 0 -and (Test-Path "$canvas.new")) { Move-Item "$canvas.new" $canvas -Force; "Canvas now declares $Prefix on port $Port." }
        else { "The port was NOT added (exit $code); the old canvas is unchanged." }
    }
} finally {
    $ErrorActionPreference = 'Stop'
    & (Join-Path $PSScriptRoot 'Start-Apache.ps1')
}

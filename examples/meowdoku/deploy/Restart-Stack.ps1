<#
.SYNOPSIS  Restart the whole stack safely: refuse while a photo exposure runs, stop, wait for httpd to exit, start.
Stop-Apache.ps1 returns before the old httpd has fully gone, and a Start-Apache.ps1 run too soon fails
("slotmem_create failed", no 443 listener). This waits in between. Both scripts carry the operator gate.
A restart runs the working tree as it is and does not re-export the site's files (web_export.py is a separate step).
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Restart-Stack.ps1
#>
param([switch]$Force)

$ErrorActionPreference = 'Stop'
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
& (Join-Path $PSScriptRoot 'Start-Apache.ps1')

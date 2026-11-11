<#
.SYNOPSIS  Turn on the Nodus cron tool for the live service: install the schedule, then one stop window.
The schedule (deploy\cron.table) is installed as <state>\jobs\cron\cron.table only when that file does not exist;
an installed table is never overwritten. The stop window (Add-CanvasContractPort.ps1) rebuilds the live Nodus
library, because the cron tool is newer than it, and gives the canvas the `cron` contract port. The instance
configuration already names the prefix, the tool's directory and this server as the performer of its tasks.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Enable-CronTool.ps1
#>
param([int]$Port = 47615, [switch]$Force)

$ErrorActionPreference = 'Stop'
if (-not $env:MEOWDOKU_STATE_DIR) { throw 'Set MEOWDOKU_STATE_DIR to the live state root first.' }
$jobs = Join-Path $env:MEOWDOKU_STATE_DIR 'jobs\cron'
New-Item -ItemType Directory -Force $jobs | Out-Null
$table = Join-Path $jobs 'cron.table'
if (Test-Path $table) { "The schedule is already installed: $table" }
else { Copy-Item (Join-Path $PSScriptRoot 'cron.table') $table; "Installed the schedule: $table" }
# A loan of 36000 market turns is ten minutes at the canvas host's 60 turns a second.
& (Join-Path $PSScriptRoot 'Add-CanvasContractPort.ps1') -Prefix cron -Port $Port -Loan 36000 -BuildLibrary -Force:$Force

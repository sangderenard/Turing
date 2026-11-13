<#
.SYNOPSIS  Turn on the Nodus email manager table for the live service: install its schedule, then one stop window.
The table's schedule (deploy\mail.cron.table) is installed as <state>\jobs\mail\cron.table only when that file does not
exist; an installed table is never overwritten. The stop window (Add-CanvasContractPort.ps1) gives the canvas the
`mail` contract port and brings the stack back up. The instance configuration (instances.json) already names the
prefix, the table's cron tool and directory, the `mail` instance as the table's host and meowdoku-a as an engine.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Enable-MailTable.ps1
#>
param([int]$Port = 47619, [switch]$Force)

$ErrorActionPreference = 'Stop'
if (-not $env:MEOWDOKU_STATE_DIR) { throw 'Set MEOWDOKU_STATE_DIR to the live state root first.' }
$jobs = Join-Path $env:MEOWDOKU_STATE_DIR 'jobs\mail'
New-Item -ItemType Directory -Force $jobs | Out-Null
$table = Join-Path $jobs 'cron.table'
if (Test-Path $table) { "The schedule is already installed: $table" }
else { Copy-Item (Join-Path $PSScriptRoot 'mail.cron.table') $table; "Installed the schedule: $table" }
# A loan of 36000 market turns is ten minutes at the canvas host's 60 turns a second: a sweep may wait on the provider.
& (Join-Path $PSScriptRoot 'Add-CanvasContractPort.ps1') -Prefix mail -Port $Port -Loan 36000 -Force:$Force

<#
.SYNOPSIS  The single entry: httpd -t, start Apache (not as a service), wait for :443, then wait for the supervisor to report the instances.
Apache starts deploy\apache_supervisor.py as the 443 vhost's piped CustomLog program; the supervisor leases the lowering binaries,
launches the Meowdoku and Nodus instances in instances.json and supervises them until Apache closes its stdin (Stop-Apache.ps1).
Do not also run Start-Meowdoku.ps1 for an instance listed in instances.json: a port that already has a listener is left alone (state 'external').
#>
param([switch]$WhatIf, [int]$InstanceTimeout = 300)
$ErrorActionPreference = 'Stop'
$Httpd = 'C:\Apache24\bin\httpd.exe'
$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$Repo = 'C:\dev\Powershell\turing'
$Sup = Join-Path $Repo 'examples\meowdoku\deploy\apache_supervisor.py'
Write-Host "cmd: $Httpd -t"
Write-Host "cmd: Start-Process $Httpd -WindowStyle Hidden -WorkingDirectory C:\Apache24\bin"
Write-Host "then: poll '$Python $Sup --status' until every instance is running/external/skipped/failed (timeout $InstanceTimeout s)"
Write-Host "cmd: curl.exe -k https://127.0.0.1/api/whoami"
if ($WhatIf) { Write-Host '[WhatIf] nothing run.'; return }
& $Httpd -t
if ($LASTEXITCODE -ne 0) { throw 'httpd -t failed' }
if (Get-NetTCPConnection -LocalPort 443 -State Listen -ErrorAction SilentlyContinue) { throw 'Something already listens on 443.' }
$p = Start-Process -FilePath $Httpd -WorkingDirectory 'C:\Apache24\bin' -WindowStyle Hidden -PassThru
Write-Host "PID $($p.Id)"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 443 -State Listen -ErrorAction SilentlyContinue) { break }
}
if (-not (Get-NetTCPConnection -LocalPort 443 -State Listen -ErrorAction SilentlyContinue)) { throw 'No listener on 443 after 30 s; see C:\Apache24\logs\error.log' }
Write-Host '443: listening'

$env:PYTHONUTF8 = '1'
$settled = @('running', 'external', 'skipped', 'failed')
$rows = @()
for ($i = 0; $i -lt $InstanceTimeout; $i += 2) {
    Start-Sleep -Seconds 2
    $json = & $Python $Sup --status 2>$null
    if ($LASTEXITCODE -eq 0 -and $json) { $rows = @($json | ConvertFrom-Json) }
    if ($rows.Count -gt 0 -and -not ($rows | Where-Object { $settled -notcontains $_.state })) { break }
}
if ($rows.Count -eq 0) { Write-Warning "Supervisor reported no instances after $InstanceTimeout s; see build\meowdoku\logs\apache-supervisor.log" }
else { $rows | Format-Table name, kind, port, state, pid, restarts, supervisor_pid -AutoSize }
curl.exe -s -k https://127.0.0.1/api/whoami
Write-Host ''

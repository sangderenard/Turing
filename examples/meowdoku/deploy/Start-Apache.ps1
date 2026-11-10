<#
.SYNOPSIS  httpd -t, then start Apache (not as a service) in the background, wait for :443, curl /api/whoami with the CA.
WARNING: do not run until the server's forwarded-origin change (proxy_trust) is live; see LAUNCH_2026-11-10.md.
#>
param([switch]$WhatIf)
$ErrorActionPreference = 'Stop'
$Httpd = 'C:\Apache24\bin\httpd.exe'
$Ca = 'C:\dev\Powershell\turing\build\meowdoku\tls\ca.crt'
Write-Host "cmd: $Httpd -t"
Write-Host "cmd: Start-Process $Httpd -WindowStyle Hidden -WorkingDirectory C:\Apache24\bin"
Write-Host "cmd: curl.exe --cacert $Ca https://127.0.0.1/api/whoami"
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
curl.exe -s --cacert $Ca https://127.0.0.1/api/whoami
Write-Host ''

<#
.SYNOPSIS  Stop the background Apache: httpd.exe -k stop, falling back to Stop-Process on the httpd listener of 443.
#>
param([switch]$WhatIf)
$ErrorActionPreference = 'Stop'
$Httpd = 'C:\Apache24\bin\httpd.exe'
Write-Host "cmd: $Httpd -k stop   (fallback: Stop-Process on the 443 listener, httpd only)"
if ($WhatIf) { Write-Host '[WhatIf] nothing run.'; return }
$conn = Get-NetTCPConnection -LocalPort 443 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { Write-Host 'Nothing listening on 443.'; return }
$proc = Get-Process -Id $conn.OwningProcess
if ($proc.ProcessName -ne 'httpd') { throw "443 listener is $($proc.ProcessName), not httpd; refusing." }
& $Httpd -k stop 2>&1 | Out-Host
Start-Sleep -Seconds 3
if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) { Write-Warning "httpd -k stop did not end PID $($proc.Id); Stop-Process"; Stop-Process -Id $proc.Id -Force }
Write-Host 'Apache stopped.'

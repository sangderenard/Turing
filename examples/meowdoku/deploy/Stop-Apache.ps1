<#
.SYNOPSIS  Stop Apache; closing httpd's piped CustomLog pipe makes apache_supervisor.py stop its instances gracefully and check its leases in.
httpd.exe -k stop first, falling back to Stop-Process on the httpd listener of 443 (and its httpd parent). Then waits up to 40 s for the
supervisor table to show every instance stopped, falling back to Stop-Process on leftover instance pids from the table.
#>
param([switch]$WhatIf, [int]$InstanceTimeout = 40)
$ErrorActionPreference = 'Stop'
$Httpd = 'C:\Apache24\bin\httpd.exe'
$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$Sup = 'C:\dev\Powershell\turing\examples\meowdoku\deploy\apache_supervisor.py'
Write-Host "cmd: $Httpd -k stop   (fallback: Stop-Process on the 443 httpd listener and its httpd parent only)"
Write-Host "then: poll '$Python $Sup --status' up to $InstanceTimeout s for all instances stopped; fallback: Stop-Process by pid from the table"
if ($WhatIf) { Write-Host '[WhatIf] nothing run.'; return }
$conn = Get-NetTCPConnection -LocalPort 443 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { Write-Host 'Nothing listening on 443.' }
else {
    $proc = Get-Process -Id $conn.OwningProcess
    if ($proc.ProcessName -ne 'httpd') { throw "443 listener is $($proc.ProcessName), not httpd; refusing." }
    $parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)").ParentProcessId
    & $Httpd -k stop 2>&1 | Out-Host
    Start-Sleep -Seconds 3
    $pp = Get-Process -Id $parent -ErrorAction SilentlyContinue
    if ($pp -and $pp.ProcessName -eq 'httpd') { Write-Warning "Stop-Process httpd parent $parent"; Stop-Process -Id $parent -Force }
    if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) { Write-Warning "httpd -k stop did not end PID $($proc.Id); Stop-Process"; Stop-Process -Id $proc.Id -Force }
    Write-Host 'Apache stopped.'
}
$env:PYTHONUTF8 = '1'
$rows = @()
for ($i = 0; $i -lt $InstanceTimeout; $i += 2) {
    $json = & $Python $Sup --status 2>$null
    if ($LASTEXITCODE -eq 0 -and $json) { $rows = @($json | ConvertFrom-Json) }
    $live = @($rows | Where-Object { $_.pid -and (Get-Process -Id $_.pid -ErrorAction SilentlyContinue) })
    if ($live.Count -eq 0 -and -not ($rows | Where-Object { @('running', 'starting', 'backoff', 'stopping') -contains $_.state })) { break }
    Start-Sleep -Seconds 2
}
foreach ($r in $rows) {
    if ($r.pid -and (Get-Process -Id $r.pid -ErrorAction SilentlyContinue)) {
        Write-Warning "instance $($r.name) pid $($r.pid) still running after $InstanceTimeout s; Stop-Process"
        Stop-Process -Id $r.pid -Force
    }
}
if ($rows.Count -gt 0) { $rows | Format-Table name, kind, port, state, pid, restarts -AutoSize }
Write-Host 'Instances stopped.'

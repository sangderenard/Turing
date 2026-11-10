<#  Listeners on 80/443/8766/8767/8768 with owning PID and process name. Read-only. #>
param([switch]$WhatIf)
Write-Host 'cmd: Get-NetTCPConnection -State Listen -LocalPort 80,443,8766,8767,8768'
if ($WhatIf) { Write-Host '[WhatIf] read-only; running the above anyway.' }
$rows = Get-NetTCPConnection -State Listen -LocalPort 80,443,8766,8767,8768 -ErrorAction SilentlyContinue |
    Sort-Object LocalPort, LocalAddress | ForEach-Object {
        $n = (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        [pscustomobject]@{ Port = $_.LocalPort; Address = $_.LocalAddress; PID = $_.OwningProcess; Process = $n }
    }
if ($rows) { $rows | Format-Table -AutoSize } else { Write-Host 'No listeners on those ports.' }

# Supervisor table (apache_supervisor.py --status): instances Apache owns, with pids and states.
$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$Sup = Join-Path $PSScriptRoot 'apache_supervisor.py'
Write-Host "cmd: $Python $Sup --status"
$env:PYTHONUTF8 = '1'
$json = & $Python $Sup --status 2>$null
if ($LASTEXITCODE -eq 0 -and $json) {
    $inst = @($json | ConvertFrom-Json)
    if ($inst.Count) { $inst | Format-Table name, kind, port, route, state, pid, restarts, supervisor_pid -AutoSize } else { Write-Host 'Supervisor table is empty.' }
} else { Write-Host 'Supervisor table unavailable.' }

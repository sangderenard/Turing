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

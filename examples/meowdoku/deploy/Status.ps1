<#  Listeners on 80/443/8766/8767/8768 with owning PID and process name. Read-only. #>
param([switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb)

# ---- Operator gate (authz.py script-auth): only accounts in script_access with deploy_scripts may run this script. Not skippable; -WhatIf is gated too.
$__repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $User) { throw 'REFUSED no operator: pass -User <name> (or set MEOWDOKU_OPERATOR) and -PasswordFile <path>.' }
if (-not $AccountsDb) { $AccountsDb = Join-Path $__repo 'build\meowdoku\accounts.db' }
if (-not $PasswordFile) {
    $__default = Join-Path $__repo "build\meowdoku\operator_credentials\$User.secret"
    if (Test-Path -LiteralPath $__default) { $PasswordFile = $__default }
}
$__args = @((Join-Path $__repo 'examples\meowdoku\authz.py'), 'script-auth', $MyInvocation.MyCommand.Name, '--user', $User, '--db', $AccountsDb)
if ($PasswordFile) { $__args += @('--password-file', $PasswordFile) }
$__eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$__gate = & 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' @__args 2>&1
$__code = $LASTEXITCODE
$ErrorActionPreference = $__eap
if ($__code -ne 0) { throw (($__gate | ForEach-Object { "$_" }) -join ' ') }
Write-Host ($__gate -join ' ')
# ---- end gate
Write-Host 'cmd: Get-NetTCPConnection -State Listen -LocalPort 80,443,8766,8767,8768'
if ($WhatIf) { Write-Host '[WhatIf] read-only; running the above anyway.' }
$rows = Get-NetTCPConnection -State Listen -LocalPort 80,443,8766,8767,8768 -ErrorAction SilentlyContinue |
    Sort-Object LocalPort, LocalAddress | ForEach-Object {
        $n = (Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue).ProcessName
        [pscustomobject]@{ Port = $_.LocalPort; Address = $_.LocalAddress; PID = $_.OwningProcess; Process = $n }
    }
if ($rows) { $rows | Format-Table -AutoSize } else { Write-Host 'No listeners on those ports.' }

# Supervisor table (apache_supervisor.py --status): instances Apache owns, with pids and states.
$ServicePython = 'C:\dev\Powershell\srv\python\python.exe'
if (-not (Test-Path -LiteralPath $ServicePython)) { $ServicePython = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' }   # until provisioned
$Python = $ServicePython
$Sup = Join-Path $PSScriptRoot 'apache_supervisor.py'
Write-Host "cmd: $Python $Sup --status"
$env:PYTHONUTF8 = '1'
$json = & $Python $Sup --status 2>$null
if ($LASTEXITCODE -eq 0 -and $json) {
    $inst = @($json | ConvertFrom-Json)
    if ($inst.Count) { $inst | Format-Table name, kind, port, route, state, pid, restarts, supervisor_pid -AutoSize } else { Write-Host 'Supervisor table is empty.' }
} else { Write-Host 'Supervisor table unavailable.' }

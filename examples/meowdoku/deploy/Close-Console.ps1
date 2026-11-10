<#
.SYNOPSIS  Close the read-only console viewer opened by Open-Console.ps1. Stops only that frontend_shell.exe; never touches the host.
#>
param([Parameter(Mandatory)][string]$Instance, [switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb)

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

$PidFile = Join-Path $__repo "build\meowdoku\console-$Instance.pid"
if (-not (Test-Path -LiteralPath $PidFile)) { Write-Host "No console recorded for $Instance."; return }
$id = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
$proc = Get-Process -Id $id -ErrorAction SilentlyContinue
if ($proc -and $proc.ProcessName -ne 'frontend_shell') { throw "REFUSED pid $id is $($proc.ProcessName), not frontend_shell; leaving it alone (pid file kept)." }
Write-Host "cmd: Stop-Process -Id $id (frontend_shell viewer for $Instance)"
if ($WhatIf) { Write-Host '[WhatIf] not stopping.'; return }
if ($proc) { Stop-Process -Id $id }
Remove-Item -LiteralPath $PidFile -Force
Write-Host "console viewer closed ($Instance)."

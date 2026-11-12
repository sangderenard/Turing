<#
.SYNOPSIS  Local snapshot backup of the Meowdoku state root (state_backup.py snapshot). Safe while the servers are live.
Secrets are copied only with -IncludeSecrets. -WhatIf prints the command only (the gate still runs).
#>
param([string]$StateDir = $env:MEOWDOKU_STATE_DIR, [string]$Out = 'C:\dev\Powershell\srv\backups', [int]$Keep = 14, [switch]$IncludeSecrets, [switch]$IncludeLogs, [switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb)

# ---- Operator gate (authz.py script-auth): only accounts in script_access with deploy_scripts may run this script. Not skippable; -WhatIf is gated too.
$__repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $StateDir) { $StateDir = $env:MEOWDOKU_STATE_DIR }   # empty = the old build\meowdoku layout
$SupArgs = if ($StateDir) { @('--state-dir', $StateDir) } else { @() }
if (-not $User) { throw 'REFUSED no operator: pass -User <name> (or set MEOWDOKU_OPERATOR) and -PasswordFile <path>.' }
if (-not $AccountsDb) { $AccountsDb = if ($StateDir) { Join-Path $StateDir 'sql\accounts.db' } else { Join-Path $__repo 'build\meowdoku\accounts.db' } }
if (-not $PasswordFile) {
    $__default = if ($StateDir) { Join-Path $StateDir "secrets\operator_credentials\$User.secret" } else { Join-Path $__repo "build\meowdoku\operator_credentials\$User.secret" }
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

if (-not $StateDir) { throw 'REFUSED no state dir: pass -StateDir or set MEOWDOKU_STATE_DIR.' }
$Python = if (Test-Path -LiteralPath 'C:\dev\Powershell\srv\python\python.exe') { 'C:\dev\Powershell\srv\python\python.exe' } else { 'python' }
$Tool = Join-Path $__repo 'examples\meowdoku\state_backup.py'
$ToolArgs = @($Tool, 'snapshot', '--state-dir', $StateDir, '--out', $Out, '--keep', "$Keep")
if ($IncludeSecrets) { $ToolArgs += '--include-secrets' }
if ($IncludeLogs) { $ToolArgs += '--include-logs' }
Write-Host "cmd: $Python $($ToolArgs -join ' ')"
if ($WhatIf) { return }
$env:PYTHONUTF8 = '1'
& $Python @ToolArgs
if ($LASTEXITCODE -ne 0) { throw "state_backup.py exited $LASTEXITCODE" }

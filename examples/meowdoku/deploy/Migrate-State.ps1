<#
.SYNOPSIS  Stop-window state migration: copy (never move) the legacy build\meowdoku state into the state root, then verify it.
Refuses while 8766 / 47611 / 47612 is listening. Runs examples\meowdoku\state_migrate.py (plan, copy via the sqlite backup API for the databases, verify by
sha256 / row counts), prints the receipt path, then prints the follow-up steps. The legacy tree is never modified. -DryRun prints the plan only.
The gate authenticates against the LEGACY accounts.db (the new root has none yet) unless -AccountsDb / -PasswordFile are given.
#>
param([Parameter(Mandatory)][string]$LegacyBase, [Parameter(Mandatory)][string]$StateDir, [switch]$DryRun, [switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb)

if (-not $AccountsDb) { $AccountsDb = Join-Path $LegacyBase 'accounts.db' }
if (-not $PasswordFile -and $User) { $__legacySecret = Join-Path $LegacyBase "operator_credentials\$User.secret"; if (Test-Path -LiteralPath $__legacySecret) { $PasswordFile = $__legacySecret } }

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

$LegacyBase = (Resolve-Path -LiteralPath $LegacyBase).Path
$busy = @(Get-NetTCPConnection -State Listen -LocalPort 8766,47611,47612 -ErrorAction SilentlyContinue)
if ($busy.Count) { throw ('REFUSED stop window is not open: listening on ' + (($busy | ForEach-Object { $_.LocalPort } | Sort-Object -Unique) -join ',') + '. Run Stop-Apache.ps1 first.') }

$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$Tool = Join-Path $__repo 'examples\meowdoku\state_migrate.py'
$ToolArgs = @($Tool, '--legacy-base', $LegacyBase, '--state-dir', $StateDir)
if ($DryRun -or $WhatIf) { $ToolArgs += '--dry-run' } else { $ToolArgs += @('--mode', 'copy-verify') }
Write-Host "cmd: $Python $($ToolArgs -join ' ')"
$env:PYTHONUTF8 = '1'
& $Python @ToolArgs
if ($LASTEXITCODE -ne 0) { throw "state_migrate.py exited $LASTEXITCODE (nothing in $LegacyBase was changed)." }
if ($DryRun -or $WhatIf) { return }

Write-Host ''
Write-Host 'Follow-up (nothing below has been done):'
Write-Host "  1. instances.json (object form): set top-level  ""state_dir"": ""$($StateDir -replace '\','/')"""
Write-Host "  2. set the environment for the scripts:  `$env:MEOWDOKU_STATE_DIR = '$StateDir'  (and persist it for the service account)"
Write-Host '  3. Start-Apache.ps1'
Write-Host "  Rollback: Rollback-State.ps1 (prints it); $LegacyBase is untouched."

<#
.SYNOPSIS  Make the dev stack's accounts database: a redacted export of the live one (redactor.py).
Ticket 1ffa389bd91b4288aecda3b8c168d2cf; Development case 7ac83c7b293a40969d94a455e15ffa01.

The dev server, the branch servers and every developer's branch code read ONLY this export, never the live accounts
database. First every table is audited for a declared rule: a missing rule stops here, and nothing is exported. Then:
- the live database is snapshotted read-only;
- log values are aliased with the dev state's own key (<DevState>\secrets\private_hash.key, made when missing);
- bearer secrets are dropped and content private on live is withheld;
- the copy is VACUUMed and integrity-checked, then renamed into place whole at <DevState>\sql\accounts.db.

The live state is never written. If a dev-side server has the export open, the replace fails and the old export stays;
stop that server first (instance control), then run this again.

Runs as the operator (it reads the live database). It is also in the remote script control's list
(admin_script_control.SCRIPTS), so an admin can run it from /operator-control.html.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='admin'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Export-DevDatabase.ps1 -WhatIf
.EXAMPLE   .\Export-DevDatabase.ps1
#>
param([switch]$WhatIf, [string]$DevState = 'C:\dev\Powershell\srv\state-dev', [string]$User = $env:MEOWDOKU_OPERATOR,
      [string]$PasswordFile, [string]$AccountsDb, [string]$StateDir)

# ---- Operator gate (authz.py script-auth): only accounts in script_access with deploy_scripts may run this script. Not skippable; -WhatIf is gated too.
$__repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $StateDir) { $StateDir = $env:MEOWDOKU_STATE_DIR }   # empty = the old build\meowdoku layout
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

$ErrorActionPreference = 'Stop'
$Python = 'C:\dev\Powershell\srv\python\python.exe'      # the private service interpreter (Provision-ServicePython.ps1)
$Redactor = Join-Path $__repo 'examples\meowdoku\redactor.py'
if (-not $StateDir) { throw 'Set MEOWDOKU_STATE_DIR (or -StateDir) to the live state root: the export reads its accounts database.' }
$Source = Join-Path $StateDir 'sql\accounts.db'
$Target = Join-Path $DevState 'sql\accounts.db'
$Key = Join-Path $DevState 'secrets\private_hash.key'
if ((Resolve-Path -LiteralPath $StateDir).Path -eq (Resolve-Path -LiteralPath $DevState -ErrorAction SilentlyContinue).Path) {
    throw 'REFUSED: -DevState is the live state root; the export goes to the dev state only.'
}

Write-Host "== audit: every table of $Source has a redaction rule"
Write-Host "cmd: $Python $Redactor audit --from $Source"
& $Python $Redactor audit --from $Source
if ($LASTEXITCODE -ne 0) { throw 'REFUSED: a table has no redaction rule (redactor.RULES); nothing was exported.' }

Write-Host "== export: $Source (read only) -> $Target, aliased with $Key"
Write-Host "cmd: $Python $Redactor export --from $Source --to $Target --key $Key"
if ($WhatIf) { Write-Host '[WhatIf] not exported.'; exit 0 }
& $Python $Redactor export --from $Source --to $Target --key $Key
if ($LASTEXITCODE -ne 0) {
    throw "the export failed (exit $LASTEXITCODE). The dev database is unchanged. If it is in use by a dev-side server, stop that server first (instance control), then run this again."
}
$item = Get-Item -LiteralPath $Target
Write-Host ("exported: {0} ({1:N0} bytes, {2})" -f $Target, $item.Length, $item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
Write-Host 'The dev-side servers read it at their next start (their --accounts-db names this file).'

<#
.SYNOPSIS  Open the read-only administrative console viewer (frontend_shell --attach) on a running Nodus instance.
The viewer only reads: opening or closing it never touches the running graph. Close with Close-Console.ps1 -Instance <name>.
#>
param([Parameter(Mandatory)][string]$Instance, [switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb, [string]$StateDir)

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

$State = if ($StateDir) { Join-Path $StateDir 'run' } else { Join-Path $__repo 'build\meowdoku' }
$PidFile = Join-Path $State "console-$Instance.pid"
$ServicePython = 'C:\dev\Powershell\srv\python\python.exe'
if (-not (Test-Path -LiteralPath $ServicePython)) { $ServicePython = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' }
$Shell = 'C:/dev/Powershell/nodus/build-bridging/Release/frontend_shell.exe'
$json = & $ServicePython (Join-Path $PSScriptRoot 'apache_supervisor.py') --status @SupArgs 2>$null
if ($LASTEXITCODE -ne 0 -or -not $json) { throw 'REFUSED supervisor table unavailable.' }
$row = @($json | ConvertFrom-Json) | Where-Object { $_.name -eq $Instance } | Select-Object -First 1
if (-not $row) { throw "REFUSED no such instance: $Instance" }
if ($row.state -ne 'running') { throw "REFUSED instance $Instance is not running (state $($row.state))." }
if (-not $row.console_port) { throw "REFUSED instance $Instance has no console port." }
$Token = if ($StateDir) { Join-Path $StateDir "secrets\service_tokens\$Instance.console.secret" }   # state_paths.py service_tokens_dir else { Join-Path $State "operator_credentials\$Instance.console.secret" }
if (-not (Test-Path -LiteralPath $Token)) { throw "REFUSED console token file missing: $Token" }
if (Test-Path -LiteralPath $PidFile) {
    $old = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -eq 'frontend_shell' })) { throw "REFUSED console already open for $Instance (pid $old); use Close-Console.ps1." }
}
$argv = @("--attach=127.0.0.1:$($row.console_port)", "--console-token-file=$Token")
Write-Host "cmd: $Shell $($argv -join ' ')"
if ($WhatIf) { Write-Host '[WhatIf] not launching.'; return }
# The viewer is the Torch build: its DLLs resolve from the build directory and the operator's Torch lib.
$env:PATH = 'C:\dev\Powershell\nodus\build-bridging\Release;C:\Users\alber\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib;' + $env:PATH
$p = Start-Process -FilePath $Shell -ArgumentList $argv -PassThru
Set-Content -LiteralPath $PidFile -Value $p.Id -Encoding ascii
Write-Host "console viewer pid $($p.Id) (recorded in $PidFile)"

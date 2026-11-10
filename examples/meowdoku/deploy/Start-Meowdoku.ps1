<#
.SYNOPSIS  Start a Meowdoku instance with the recorded launch settings (HOW_TO_START.md section 3).
.EXAMPLE   .\Start-Meowdoku.ps1 -Port 8766 -Route a -Bind 127.0.0.1
.EXAMPLE   .\Start-Meowdoku.ps1 -Port 8767 -Dev -WhatIf
No secrets are held here. -WhatIf prints the commands and changes nothing.
#>
param(
    [int]$Port = 8766,
    [string]$Route = 'a',
    [string]$Bind = '0.0.0.0',
    [switch]$Dev,
    [switch]$WhatIf,
    [string]$User = $env:MEOWDOKU_OPERATOR,
    [string]$PasswordFile,
    [string]$AccountsDb
)

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
$ErrorActionPreference = 'Stop'
Write-Warning "Apache owns the live instances (deploy\apache_supervisor.py, instances.json): use Start-Apache.ps1 / Stop-Apache.ps1. This script is for manual dev use; a port the supervisor manages will be left alone or fought over."
$Repo   = 'C:\dev\Powershell\turing'
$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$Tls    = Join-Path $Repo 'build\meowdoku\tls'
$LogDir = Join-Path $Repo 'build\meowdoku\logs'
$Stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'

if ($Dev) {
    if ($Port -eq 8766) { $Port = 8767 }
    if (-not $PSBoundParameters.ContainsKey('Bind')) { $Bind = '127.0.0.1' }
    $argv = @('examples\meowdoku\server.py', '--port', "$Port", '--bind', $Bind,
        '--web-dir', 'C:\dev\Powershell\turing-claude-integration-20261107\build\meowdoku\web_toy_dev',
        '--profiles', 'build/meowdoku/dev_profiles_8767.json',
        '--tls-cert', "$Tls\cert.pem", '--tls-key', "$Tls\key.pem",
        '--no-wasm', '--no-idle-train', '--route', $Route)
} else {
    $argv = @('examples\meowdoku\server.py', '--port', "$Port", '--bind', $Bind,
        '--tls-cert', 'build\meowdoku\tls\cert.pem', '--tls-key', 'build\meowdoku\tls\key.pem',
        '--retain-fountain-products', 'C:\dev\Powershell\turing\build\dot-retained-fountain-20261109\fountain-receipt.json',
        '--route', $Route)
}
$out = Join-Path $LogDir "$Port-$Stamp.out.log"
$err = Join-Path $LogDir "$Port-$Stamp.err.log"

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) { throw "Port $Port already has a listener (PID $($existing[0].OwningProcess)). Use the session-preserving restart (Stop-Meowdoku.ps1); never launch over it." }

Write-Host "cwd:  $Repo"
Write-Host "env:  PYTHONUTF8=1"
Write-Host "cmd:  $Python $($argv -join ' ')"
Write-Host "logs: $out / $err"
if ($WhatIf) {
    Write-Host "[WhatIf] then: netstat -ano | find ':$Port' ; curl.exe -k https://127.0.0.1:$Port/api/ready"
    return
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONUTF8 = '1'
$p = Start-Process -FilePath $Python -ArgumentList $argv -WorkingDirectory $Repo -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
Write-Host "PID $($p.Id)"

# Verify listener, then HTTPS via the local CA (HOW_TO_START.md section 3: a PID alone is not proof).
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { throw "Process exited with code $($p.ExitCode); see $err" }
    if (netstat -ano | Select-String -Pattern "[:.]$Port\s+\S+\s+LISTENING") { $ok = $true; break }
}
if (-not $ok) { throw "No listener on $Port after 180 s; see $err" }
Write-Host "listener on ${Port}: up"
$code = curl.exe -s -o NUL -w '%{http_code}' -k "https://127.0.0.1:$Port/api/ready"
Write-Host "GET https://127.0.0.1:$Port/api/ready -> HTTP $code"
if ($code -ne '200') { Write-Warning "/api/ready did not return 200 yet (may still be booting); re-check with Status.ps1 and the logs." }

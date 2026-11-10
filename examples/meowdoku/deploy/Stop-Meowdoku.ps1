<#
.SYNOPSIS  Session-preserving stop of the Meowdoku server listening on -Port (HOW_TO_START.md section 4).
Posts {op:restart, in:N} to the authenticated lifecycle endpoint (saves sessions, exits 75), then waits for the
process to exit. It does NOT relaunch; run Start-Meowdoku.ps1 afterwards. Refuses unless the listener is
python.exe running server.py. Admin token: build\meowdoku\admin_token_<port>.txt, else admin_token.txt (never printed).
-Force falls back to Stop-Process (loses unsaved session state) only if the lifecycle call fails.
#>
param([Parameter(Mandatory)][int]$Port, [int]$In = 10, [switch]$Force, [switch]$WhatIf, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile, [string]$AccountsDb)

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
$Repo = 'C:\dev\Powershell\turing'
$State = Join-Path $Repo 'build\meowdoku'
$Python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { Write-Host "Nothing listening on $Port."; return }
$procId = $conn.OwningProcess
$cim = Get-CimInstance Win32_Process -Filter "ProcessId=$procId"
if ($cim.Name -ne 'python.exe' -or $cim.CommandLine -notmatch 'server\.py') {
    throw "PID $procId is '$($cim.Name)' and is not running server.py; refusing."
}
$tok = Join-Path $State "admin_token_$Port.txt"
if (-not (Test-Path $tok)) { $tok = Join-Path $State 'admin_token.txt' }
Write-Host "target PID $procId  ($($cim.CommandLine))"
Write-Host "cmd: POST https://127.0.0.1:$Port/api/admin/lifecycle {op:restart,in:$In}  header X-Meowdoku-Admin from $tok  CA $State\tls\ca.crt"
Write-Host "then: wait for PID $procId to exit; fallback only with -Force: Stop-Process -Id $procId"
if ($WhatIf) { Write-Host '[WhatIf] nothing sent.'; return }

$py = @'
import json, ssl, sys, urllib.request
port, tok, ca, secs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ctx = ssl.create_default_context(cafile=ca)
req = urllib.request.Request(f'https://127.0.0.1:{port}/api/admin/lifecycle',
    data=json.dumps({'op': 'restart', 'in': secs}).encode(),
    headers={'Content-Type': 'application/json', 'X-Meowdoku-Admin': open(tok).read().strip()})
with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
    print(r.read().decode())
'@
$env:PYTHONUTF8 = '1'
& $Python -c $py $Port $tok "$State\tls\ca.crt" $In
if ($LASTEXITCODE -ne 0) {
    if (-not $Force) { throw "Lifecycle stop failed (exit $LASTEXITCODE). Re-run with -Force only if losing unsaved sessions is acceptable." }
    Write-Warning "Lifecycle failed; -Force: Stop-Process $procId"
    Stop-Process -Id $procId
}
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { Write-Host "PID $procId exited; port $Port closed."; return }
}
throw "PID $procId still running after 120 s."

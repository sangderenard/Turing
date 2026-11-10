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
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'
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
    Write-Host "[WhatIf] then: netstat -ano | find ':$Port' ; curl.exe --cacert $Tls\ca.crt https://127.0.0.1:$Port/api/ready"
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
$code = curl.exe -s -o NUL -w '%{http_code}' --cacert "$Tls\ca.crt" "https://127.0.0.1:$Port/api/ready"
Write-Host "GET https://127.0.0.1:$Port/api/ready -> HTTP $code"
if ($code -ne '200') { Write-Warning "/api/ready did not return 200 yet (may still be booting); re-check with Status.ps1 and the logs." }

#Requires -Version 7
# Runs inside the isolated agents-gemma Windows account. It owns only its profile state.
# Starts the local Gemma 4 model (llama-server), prepares the Hermes home for the SpeakToMe account gemma, then runs the
# sleep cycle (hermes_agent.py cycle) in the foreground. When the cycle ends or the task is stopped, the llama-server
# this script started is stopped with its whole process tree.
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$workspace = 'C:\dev\Powershell'
$repo = Join-Path $workspace 'turing'
$python = Join-Path $workspace 'srv\python\python.exe'
$agent = Join-Path $repo 'examples\meowdoku\deploy\hermes_agent.py'
$toolRoot = Join-Path $workspace 'srv\tools\llama-cpp\b12416'
$server = Join-Path $toolRoot 'llama-server.exe'
$modelRoot = Join-Path $workspace 'srv\state\models\gemma-4-26b-a4b'
$modelFile = 'google_gemma-4-26B-A4B-it-Q4_K_M.gguf'
$alias = 'gemma-4-26B-A4B-it-Q4_K_M'
$port = 11436
$credentials = Join-Path $env:USERPROFILE 'speaktome-credentials'   # Set-AgentBox -AgentLogins gemma copies the sign-in here
$secret = Join-Path $credentials 'gemma.secret'
$hermesRoot = Join-Path $env:USERPROFILE 'hermes'                    # the Hermes home is $hermesRoot\gemma
$state = Join-Path $env:USERPROFILE 'GemmaAgent'
$speaktomeHermes = Join-Path $workspace 'speaktome\speaktome_hermes'
$hermesExe = Join-Path $workspace 'srv\tools\hermes-agent\0.19.0\venv\Scripts\hermes.exe'
$needed = @($secret, $python, $agent, $server, (Join-Path $modelRoot $modelFile), $hermesExe, $speaktomeHermes)
foreach ($file in $needed) { if (-not (Test-Path -LiteralPath $file)) { throw "Missing gemma prerequisite: $file" } }
if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) { throw "Port $port is already listening; not starting a second model server." }
New-Item -ItemType Directory -Force -Path $state, $hermesRoot | Out-Null
$log = Join-Path $state 'runner.log'
$serverProcess = $null
try {
    # mmap stays on (the default): never --load-mode none, it filled the page file on 2026-11-20.
    $serverProcess = Start-Process -FilePath $server -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $state 'llama-server.out.log') -RedirectStandardError (Join-Path $state 'llama-server.err.log') `
        -ArgumentList @('--model', (Join-Path $modelRoot $modelFile), '--alias', $alias, '--host', '127.0.0.1', '--port', "$port",
                        '--device', 'CUDA0', '--fit', 'on', '-c', '65536', '-np', '1', '--jinja')
    $deadline = (Get-Date).AddMinutes(10)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($serverProcess.HasExited) { throw "llama-server exited early with code $($serverProcess.ExitCode); see $state" }
        try {
            if ((Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) { throw "llama-server did not answer /health within 10 minutes; see $state" }
    & $python $agent prepare --account gemma --provider custom --model $alias --base-url "http://127.0.0.1:$port/v1" `
        --speaktome-hermes $speaktomeHermes --hermes-root $hermesRoot --secrets-dir $credentials `
        --mcp-python $python *>> $log
    if ($LASTEXITCODE -ne 0) { throw "hermes_agent.py prepare failed with $LASTEXITCODE; see $log" }
    & $python $agent cycle --account gemma --hermes-root $hermesRoot *>> $log
    exit $LASTEXITCODE
} finally {
    if ($serverProcess -and -not $serverProcess.HasExited) {
        & taskkill.exe /T /F /PID $serverProcess.Id | Out-Null
    }
}

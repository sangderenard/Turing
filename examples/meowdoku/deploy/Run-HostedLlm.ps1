#Requires -Version 7
# Runs inside the isolated agents-llama Windows account. It owns only its profile state.
$ErrorActionPreference = 'Stop'
$env:SPEAKTOME_USER = 'llama'
$env:SPEAKTOME_BASE = 'https://stichka.duckdns.org:8766'
$env:SPEAKTOME_SECRET_FILE = Join-Path $env:USERPROFILE 'speaktome-credentials\llama.secret'
$env:PYTHONPATH = 'C:\dev\Powershell\speaktome;C:\dev\Powershell\turing\examples'
$env:PYTHONDONTWRITEBYTECODE = '1'
$repo = 'C:\dev\Powershell\turing'
$state = Join-Path $env:USERPROFILE 'LlamaIntern'
$modelRoot = 'C:\dev\Powershell\srv\state\models\llama3.2-1b'
$toolRoot = 'C:\dev\Powershell\srv\tools\llama-cpp\b12416'
# The base model only (owner, 2026-11-17): no adapter, no trainer. Training returns through Llama's own training page.
$needed = @($env:SPEAKTOME_SECRET_FILE, (Join-Path $modelRoot 'Llama-3.2-1B-Instruct-Q8_0.gguf'), (Join-Path $toolRoot 'llama-server.exe'))
foreach ($file in $needed) { if (-not (Test-Path -LiteralPath $file)) { throw "Missing hosted LLM prerequisite: $file" } }
New-Item -ItemType Directory -Force -Path $state | Out-Null
$log = Join-Path $state 'runner.log'
& 'C:\dev\Powershell\srv\python\python.exe' (Join-Path $repo 'examples\meowdoku\deploy\hosted_llm_runner.py') `
    --llama-server (Join-Path $toolRoot 'llama-server.exe') `
    --model (Join-Path $modelRoot 'Llama-3.2-1B-Instruct-Q8_0.gguf') `
    --state-dir $state *>> $log
exit $LASTEXITCODE

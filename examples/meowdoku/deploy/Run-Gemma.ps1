#Requires -Version 7
# Runs inside the isolated agents-gemma Windows account. It owns only its profile state.
# hermes_agent.py serve ties its children to a kill-on-close job, starts llama-server (local Gemma 4), waits for /health,
# runs prepare for the SpeakToMe account gemma, then runs the sleep cycle in-process (as hosted_llm_runner.py does for Llama).
$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$workspace = 'C:\dev\Powershell'
$python = Join-Path $workspace 'srv\python\python.exe'
$agent = Join-Path $workspace 'turing\examples\meowdoku\deploy\hermes_agent.py'
$toolRoot = Join-Path $workspace 'srv\tools\llama-cpp\b12416'
$modelFile = Join-Path $workspace 'srv\state\models\gemma-4-26b-a4b\google_gemma-4-26B-A4B-it-Q4_K_M.gguf'
$credentials = Join-Path $env:USERPROFILE 'speaktome-credentials'   # Set-AgentBox -AgentLogins gemma copies the sign-in here
$secret = Join-Path $credentials 'gemma.secret'
$hermesRoot = Join-Path $env:USERPROFILE 'hermes'                    # the Hermes home is $hermesRoot\gemma
$state = Join-Path $env:USERPROFILE 'GemmaAgent'
$speaktomeHermes = Join-Path $workspace 'speaktome\speaktome_hermes'
$needed = @($secret, $python, $agent, (Join-Path $toolRoot 'llama-server.exe'), $modelFile, $speaktomeHermes,
            (Join-Path $workspace 'srv\tools\hermes-agent\0.19.0\venv\Scripts\hermes.exe'))
foreach ($file in $needed) { if (-not (Test-Path -LiteralPath $file)) { throw "Missing gemma prerequisite: $file" } }
New-Item -ItemType Directory -Force -Path $state, $hermesRoot | Out-Null
$log = Join-Path $state 'runner.log'
& $python $agent serve --account gemma --hermes-root $hermesRoot `
    --llama-server (Join-Path $toolRoot 'llama-server.exe') --model-file $modelFile `
    --alias 'gemma-4-26B-A4B-it-Q4_K_M' --port 11436 --device CUDA0 --context 65536 `
    --state-dir $state --speaktome-hermes $speaktomeHermes --secrets-dir $credentials `
    --mcp-python $python *>> $log
exit $LASTEXITCODE

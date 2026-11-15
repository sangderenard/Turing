#Requires -Version 7
# Runs inside the isolated agents-llama Windows account. It owns only its profile state.
$ErrorActionPreference = 'Stop'
$env:SPEAKTOME_USER = 'llama'
$env:SPEAKTOME_BASE = 'https://stichka.duckdns.org:8766'
$env:SPEAKTOME_SECRET_FILE = Join-Path $env:USERPROFILE 'speaktome-credentials\llama.secret'
$env:PYTHONPATH = 'C:\dev\Powershell\speaktome'
$env:PYTHONDONTWRITEBYTECODE = '1'
$repo = 'C:\dev\Powershell\turing'
$state = Join-Path $env:USERPROFILE 'LlamaIntern'
$modelRoot = 'C:\dev\Powershell\srv\state\models\llama3.2-1b'
$toolRoot = 'C:\dev\Powershell\srv\tools\llama-cpp\b12416'
$trainingPython = 'E:\SpeakToMe\hosted-llm-train\Scripts\python.exe'
$source = Get-ChildItem -LiteralPath (Join-Path $toolRoot 'source') -Directory -Filter 'llama.cpp-*' | Select-Object -First 1 -ExpandProperty FullName
if (-not $source) { throw 'The pinned llama.cpp source converter is missing' }
$needed = @($env:SPEAKTOME_SECRET_FILE, $trainingPython, (Join-Path $modelRoot 'hf\model.safetensors'),
            (Join-Path $modelRoot 'Llama-3.2-1B-Instruct-Q8_0.gguf'), (Join-Path $toolRoot 'llama-server.exe'))
foreach ($file in $needed) { if (-not (Test-Path -LiteralPath $file)) { throw "Missing hosted LLM prerequisite: $file" } }
New-Item -ItemType Directory -Force -Path $state | Out-Null
$log = Join-Path $state 'runner.log'
& 'C:\dev\Powershell\srv\python\python.exe' (Join-Path $repo 'examples\meowdoku\deploy\hosted_llm_runner.py') `
    --llama-server (Join-Path $toolRoot 'llama-server.exe') `
    --model (Join-Path $modelRoot 'Llama-3.2-1B-Instruct-Q8_0.gguf') `
    --adapter (Join-Path $state 'adapter.gguf') `
    --state-dir $state --trainer-python $trainingPython `
    --trainer (Join-Path $repo 'examples\meowdoku\deploy\hosted_llm_train.py') `
    --base (Join-Path $modelRoot 'hf') `
    --converter (Join-Path $source 'convert_lora_to_gguf.py') `
    --adapter-root (Join-Path $state 'adapters') *>> $log
exit $LASTEXITCODE

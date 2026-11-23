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
# gemma's tools (owner, 2026-11-20: it moves things along in the repo, comments, coordinates, and calls an operator in
# the agents room when it has nothing new to explore): the 12 walking/messaging verbs, the rooms, comment threads, the
# reviews waiting for attention and one diff, and experience reports. Bulk listings (reviews_list, reviews_threads,
# research_cases) are left out: one answer is larger than a 64K context.
# reports_issues_list (one page of 50 ticket titles, about 15K characters) and reports_issue_get are offered so the
# ticket triage gemma tries is a call it can make (2026-11-23: without them it posted "calling reports_issues_list"
# thousands of times in the agents room; Hermes offers only the tools in this list).
$tools = 'whoami,where,look,enter,go,leave,examine,man,say,pending,acknowledge,contacts,' +
         'chat_rooms,chat_room,chat_post,comments_thread,comments_post,reviews_attention,reviews_diff,make,' +
         'reports_issues_list,reports_issue_get'
& $python $agent serve --account gemma --hermes-root $hermesRoot `
    --llama-server (Join-Path $toolRoot 'llama-server.exe') --model-file $modelFile `
    --alias 'gemma-4-26B-A4B-it-Q4_K_M' --port 11436 --device CUDA0 --context 65536 `
    --state-dir $state --speaktome-hermes $speaktomeHermes --secrets-dir $credentials `
    --mcp-python $python --tools-include $tools *>> $log
exit $LASTEXITCODE

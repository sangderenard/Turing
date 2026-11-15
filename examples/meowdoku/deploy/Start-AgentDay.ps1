<#
.SYNOPSIS  One task, every agent: each agent's prompt for today, run once (one-shot) as its own Windows account, hidden.
Runs as the operator and needs no elevation.

Prompts: <AgentDays>\<agent>\<yyyy-MM-dd>.md, else <AgentDays>\<agent>\default.md. An agent with neither is skipped.
AgentDays is %USERPROFILE%\AgentDays by default. A Gemini prompt may begin with "room: <room>" (and "tail: <n>") lines:
that room's latest messages go with it as context.

The agents, declared below: claude as agents (claude -p), codex as agents-codex (codex exec), gemini as agents-gemini
(gemini_agent.py). Each runs as its own account with its saved password (Set-AgentBox.ps1 -ResetPassword saves it, for
the operator only), its own environment and its own SpeakToMe sign-in; one after another, each stopped after
-TimeoutMinutes. What each did, the agents report in SpeakToMe; this script prints and appends one JSON line per run to
<AgentDays>\runs.jsonl (agent, start, seconds, exit code, the end of its output).

-Register [-At 09:00,15:00] registers the Windows task "SpeakToMe agent day" for the operator: it runs this script at
those times, only while the operator is signed in, so no password is stored and nothing is elevated. -Unregister removes it.
.EXAMPLE   .\Start-AgentDay.ps1 -List                     # what would run today, nothing started
.EXAMPLE   .\Start-AgentDay.ps1 -Agent claude             # run one now
.EXAMPLE   .\Start-AgentDay.ps1 -Register -At 09:00,15:00
#>
#Requires -Version 7
param(
    [string[]]$Agent = @(),
    [string]$AgentDays = (Join-Path $env:USERPROFILE 'AgentDays'),
    [int]$TimeoutMinutes = 20,
    [string]$GeminiModel = 'gemini-3.5-flash-lite',
    [switch]$List,
    [switch]$Register,
    [string[]]$At = @('09:00'),
    [switch]$Unregister
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$Workspace = 'C:\dev\Powershell'
$TaskName = 'SpeakToMe agent day'
$Base = 'https://stichka.duckdns.org:8766'

# Each agent's one-shot command, run in its own account's PowerShell; $prompt holds the day's prompt.
$Agents = [ordered]@{
    claude = @{ account = 'agents'; run = '& claude -p $prompt --allowedTools mcp__speaktome 2>&1' }
    codex  = @{ account = 'agents-codex'; run = '& codex exec --skip-git-repo-check $prompt 2>&1' }
    gemini = @{ account = 'agents-gemini'; run = @"
`$env:SPEAKTOME_USER = 'gemini'; `$env:SPEAKTOME_BASE = '$Base'
`$env:SPEAKTOME_SECRET_FILE = Join-Path `$HOME 'speaktome-credentials\gemini.secret'
`$env:PYTHONPATH = '$Workspace\speaktome'; `$env:PYTHONDONTWRITEBYTECODE = '1'
`$extra = @()
while (`$prompt -match '^(room|tail):\s*(\S+)\s*\r?\n') { `$extra += "--`$(`$Matches[1])", `$Matches[2]; `$prompt = `$prompt.Substring(`$Matches[0].Length) }
& '$Workspace\srv\python\python.exe' '$Workspace\turing\examples\meowdoku\deploy\gemini_agent.py' --model '$GeminiModel' --prompt `$prompt @extra 2>&1
"@ }
}

if ($Register -or $Unregister) {
    if ($Unregister) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; Write-Host "removed the task '$TaskName'"; exit 0 }
    $action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'pwsh.exe') -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$PSCommandPath`""
    $triggers = @($At | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ })
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 3)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "registered '$TaskName': $PSCommandPath daily at $($At -join ', '), only while $env:USERNAME is signed in (no password stored, not elevated)"
    exit 0
}

$today = Get-Date -Format 'yyyy-MM-dd'
$chosen = if ($Agent.Count) { $Agent } else { @($Agents.Keys) }
foreach ($name in $chosen) {
    if (-not $Agents.Contains($name)) { Write-Host "${name}: not a declared agent (declared: $($Agents.Keys -join ', '))"; continue }
    $declared = $Agents[$name]
    $promptFile = @((Join-Path $AgentDays "$name\$today.md"), (Join-Path $AgentDays "$name\default.md")) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $promptFile) { Write-Host "${name}: skipped (no $name\$today.md or $name\default.md in $AgentDays)"; continue }
    $cred = Read-AgentCredential $declared.account
    if (-not $cred) { Write-Host "${name}: skipped (no saved password for $($declared.account); Set-AgentBox.ps1 -Apply -ResetPassword saves one)"; continue }
    if ($List) { Write-Host "${name}: would run as $($declared.account) with $promptFile"; continue }

    $prompt = Get-Content -Raw -LiteralPath $promptFile -Encoding utf8
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($prompt))
    $script = "`$prompt = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$encoded'))`nSet-Location '$Workspace'`n$($declared.run)`nexit `$LASTEXITCODE"
    $started = Get-Date
    Write-Host "${name}: running as $($declared.account) with $promptFile"
    $result = Invoke-AsAgent $cred $script $Workspace ($TimeoutMinutes * 60)
    $seconds = [int]((Get-Date) - $started).TotalSeconds
    $outcome = if ($result.TimedOut) { "stopped after $TimeoutMinutes min" } else { "exit $($result.ExitCode)" }
    Write-Host "${name}: $outcome in $seconds s"
    if ($result.Output) { Write-Host ($result.Output -replace '(?m)^', '  ') }
    $tail = if ($result.Output.Length -gt 4000) { $result.Output.Substring($result.Output.Length - 4000) } else { $result.Output }
    New-Item -ItemType Directory -Force -Path $AgentDays | Out-Null
    [ordered]@{ agent = $name; account = $declared.account; prompt = $promptFile; started = $started.ToString('o'); seconds = $seconds
                exit = $result.ExitCode; timed_out = $result.TimedOut; output_tail = $tail } |
        ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $AgentDays 'runs.jsonl') -Encoding utf8
}

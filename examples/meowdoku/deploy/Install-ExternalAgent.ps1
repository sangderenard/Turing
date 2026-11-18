<#
.SYNOPSIS  Install an external API agent that an admin declared on /operator-control.html, the way Gemini was installed
by hand on 2026-11-14, in one run from an elevated shell (an SSH session as the owner is one): .\Install-ExternalAgent.ps1 -Name <name>.

What it does, in order, printing each step and never a secret:
  1. reads the declaration (external_agents.py show): provider, base URL, model, tools, minute, brief
  2. the SpeakToMe login <name>: a random secret written only to <state>\secrets\operator_credentials\<name>.secret,
     role member, no script access (as gemini, codex and claude have)
  3. the Windows account agents-<name>: Set-AgentBox.ps1 -Apply -CreateAgentAccount (a generated password, never shown,
     saved with DPAPI for the owner) and the sign-in copied into its profile; an existing account gets -CopyAgentCredentials
  4. asks for the API key (typed, not shown) and writes it AS agents-<name> to its own %USERPROFILE%\<name>.key with
     ConvertFrom-SecureString: DPAPI for that account, so only it can open the key, exactly like ~\gemini.key. The key
     goes to that process on standard input, never on a command line, in a file or a log, and is not kept here.
  5. proves the key as agents-<name> (api_agent.py --check: the provider lists its models)
  6. the character profile Start-AgentDay resolves (Gemini's agent-day documents, then the declared brief)
  7. marks it installed and re-registers the agent ticks (Start-AgentDay.ps1 -Register)
A failed step marks the declaration failed with the step's name; running again resumes (every step keeps what exists).
-RotateKey on an installed agent does steps 4 and 5 only.

This script is NOT in the admin-control action list: it asks for the key interactively and must run in the owner's shell.
.EXAMPLE   .\Install-ExternalAgent.ps1 -Name mistral
.EXAMPLE   .\Install-ExternalAgent.ps1 -Name mistral -RotateKey
#>
#Requires -Version 7
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-z][a-z0-9-]{1,12}$')][string]$Name,
    [switch]$RotateKey,
    [string]$StateDir = $(if ($env:MEOWDOKU_STATE_DIR) { $env:MEOWDOKU_STATE_DIR } else { 'C:\dev\Powershell\srv\state' })
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$Workspace = 'C:\dev\Powershell'
$Python = "$Workspace\srv\python\python.exe"
$Registry = Join-Path $PSScriptRoot '..\external_agents.py'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch { }

$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run elevated (an SSH session as the owner is).' }

function Step([string]$text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Registry([string[]]$arguments) {
    $out = & $Python $Registry --state-dir $StateDir @arguments
    if ($LASTEXITCODE -ne 0) { throw "external_agents.py $($arguments[0]) refused (exit $LASTEXITCODE)" }
    $out | ConvertFrom-Json
}
function Fail([string]$step, [string]$why) {
    & $Python $Registry --state-dir $StateDir mark $Name failed --note "$step`: $why" | Out-Null
    Write-Host "FAILED at $step`: $why" -ForegroundColor Red
    exit 1
}

Step "1. the declaration of $Name"
$agent = Registry @('show', $Name)
Write-Host ("{0}: {1} {2} at {3}, minute :{4:00}, {5} tool(s), status {6}" -f $agent.name, $agent.provider, $agent.model,
    $agent.base_url, $agent.minute, $agent.tools.Count, $agent.status)
if ($agent.status -eq 'withdrawn') { throw "$Name is withdrawn; declare it again on the operator page first" }
if ($RotateKey -and $agent.status -ne 'installed') { throw "-RotateKey is for an installed agent; $Name is $($agent.status)" }
if (-not $RotateKey -and $agent.status -eq 'installed') { throw "$Name is installed already; use -RotateKey for a new key" }
$account = $agent.account

if (-not $RotateKey) {
    Step "2. the SpeakToMe login $Name"
    try { $login = Registry @('ensure-login', $Name) } catch { Fail 'login' $_.Exception.Message }
    Write-Host "login $($login.login_was), role $($login.role), secret in $($login.secret_file) (not shown)"

    Step "3. the Windows account $account"
    $exists = [bool](Get-LocalUser -Name $account -ErrorAction SilentlyContinue)
    $box = @('-NoProfile', '-NonInteractive', '-File', (Join-Path $PSScriptRoot 'Set-AgentBox.ps1'), '-Apply',
             $(if ($exists) { '-CopyAgentCredentials' } else { '-CreateAgentAccount' }), '-AgentAccount', $account, '-AgentLogins', $Name)
    & (Join-Path $PSHOME 'pwsh.exe') @box | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail 'windows account' "Set-AgentBox exited $LASTEXITCODE" }
}
$credential = Read-AgentCredential $account
if (-not $credential) { Fail 'windows account' "no saved password for $account (Set-AgentBox.ps1 -Apply -ResetPassword -AgentAccount $account)" }

Step "4. the $($agent.provider) API key, written as $account"
# The destination is named again where the key is typed: an admin chose the base URL, and the key goes there.
$destination = ([Uri]$agent.base_url).Host
$secure = Read-Host -AsSecureString "API key for $Name ($($agent.provider) at $destination); typing is not shown"
$plain = [Net.NetworkCredential]::new('', $secure).Password
if (-not $plain) { Fail 'key' 'no key was typed' }
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($plain))
$plain = $null; $secure.Dispose()
# The catch prints a fixed line: an error message must never carry the script's text, which holds the key.
$write = @"
`$ErrorActionPreference = 'Stop'
try {
    `$k = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$encoded'))
    ConvertTo-SecureString -String `$k -AsPlainText -Force | ConvertFrom-SecureString | Set-Content -LiteralPath (Join-Path `$HOME '$Name.key') -Encoding ascii
    `$k = `$null
    'KEY WRITTEN'
} catch { 'KEY NOT WRITTEN'; exit 1 }
"@
$encoded = $null
$result = Invoke-AsAgent $credential $write $Workspace 120
$write = $null
if ($result.ExitCode -ne 0 -or $result.Output -notmatch 'KEY WRITTEN') { Fail 'key' "the key file was not written as $account" }
Write-Host "written to $account's own profile as $Name.key (DPAPI for $account; not shown)"

Step "5. the key, proved as $account"
$q = { param($v) "'" + ($v -replace "'", "''") + "'" }
$prove = "& $(& $q $Python) $(& $q (Join-Path $PSScriptRoot 'api_agent.py')) --name $(& $q $Name) --provider $(& $q $agent.provider) " +
         "--base-url $(& $q $agent.base_url) --check 2>&1`nexit `$LASTEXITCODE"
$check = Invoke-AsAgent $credential $prove $Workspace 120
Write-Host $check.Output
if ($check.ExitCode -ne 0) { Fail 'key check' (($check.Output -split "`n" | Select-Object -Last 1) -replace '\s+', ' ') }
if ($RotateKey) { Write-Host "`nNew key in place for $Name."; exit 0 }

Step "6. the character profile"
try { $made = Registry @('ensure-profile', $Name) } catch { Fail 'profile' $_.Exception.Message }
Write-Host "profile $($made.profile_id) $($made.profile_was) (edit it in Agent workspace)"

Step "7. installed; the agent ticks"
Registry @('mark', $Name, 'installed', '--note', "installed by $env:USERNAME") | Out-Null
& (Join-Path $PSHOME 'pwsh.exe') -NoProfile -NonInteractive -File (Join-Path $PSScriptRoot 'Start-AgentDay.ps1') -Register | Out-Host
# Copyable hints: the script path is a quoted PowerShell literal, so a checkout under a folder with spaces stays one
# argument (review 003eb9cb, comment 198).
$day = & $q (Join-Path $PSScriptRoot 'Start-AgentDay.ps1')
if ($LASTEXITCODE -ne 0) { Write-Host "Start-AgentDay -Register exited $LASTEXITCODE; run: pwsh -File $day -Register" -ForegroundColor Yellow }
Write-Host "`n$Name is installed: work run at :$('{0:00}' -f $agent.minute) each hour, and its bell. To start one now: pwsh -File $day -RunOne $Name" -ForegroundColor Green

<#
.SYNOPSIS  SpeakToMe operator shortcuts. Each one calls the repository's deploy scripts as they are; nothing here
replaces them. From your home folder, in cmd (the SSH default) or PowerShell:  .\stm <command>

  status    read-only: Apache, the listeners, what would move, the agents, Llama, the condition card
  backup    snapshot the state root (Backup-State.ps1; safe while live)
  merge     move each live checkout to what landed (Update-LiveCheckouts.ps1; pages republish if turing moved)
  publish   merge, and republish the pages even if nothing moved (Update-LiveCheckouts.ps1 -ExportWeb); then the same
            on the Oracle host (speaktome.dev) over SSH: its checkouts move to what landed, its pages re-export, the
            landed oracle-front.conf is installed if it changed, and its Apache reloads, which restarts ORACLE's
            instances (never this machine's). -NoOracle skips it; an unreachable Oracle warns, home's publish stands.
  restart   stop Llama, restart the stack, then start Llama and the OpenAI MCP tunnel (if it is not running);
            refused while a photo exposure runs unless -Force
  stop      shut it all down: the OpenAI MCP tunnel, Llama (task and model listener), then Apache and the stack it
            supervises; refused while a photo exposure runs unless -Force
  start     bring it all back after stop: Apache and its stack, Llama, then the tunnel
  full      backup, then publish (Oracle included), then shows the table and asks before restarting this machine
  llama     restart the hosted Llama runner (its scheduled task)
  agents    re-register the agent ticks (Start-AgentDay.ps1 -Register)
  agent     external API agents declared on /operator-control.html (Install-ExternalAgent.ps1):
              agent list               what is declared, installed or withdrawn
              agent install <name>     make its login and Windows account, ask for its API key, schedule it
              agent key <name>         a new API key for an installed agent
              agent run <name>         one work run now (Start-AgentDay.ps1 -RunOne)
              agent withdraw <name>    stop scheduling it (its accounts and key stay)

restart, stop, start, full, llama and agent install/key need an elevated shell (an SSH session as alber already is).
merge, publish and full take -MoveIdentical: an untracked file in a live tree that is identical to what landed (an
agent wrote it there, then committed it through review) is moved to <state>\backups\untracked-<repo>-<time>\ and the
fast-forward goes on (Update-LiveCheckouts.ps1 -MoveIdenticalUntracked). A file that differs still blocks.
#>
param([Parameter(Position = 0)][ValidateSet('status', 'backup', 'merge', 'publish', 'restart', 'stop', 'start', 'full', 'llama', 'agents', 'agent', 'help')]
      [string]$Command = 'help',
      [Parameter(Position = 1)][ValidateSet('list', 'install', 'key', 'run', 'withdraw')][string]$Action = 'list',
      [Parameter(Position = 2)][string]$Name = '',
      [switch]$Force, [switch]$MoveIdentical, [switch]$NoOracle)
$Move = @(if ($MoveIdentical) { 'MoveIdenticalUntracked' })
$ErrorActionPreference = 'Stop'
$Deploy = 'C:\dev\Powershell\turing\examples\meowdoku\deploy'
$env:MEOWDOKU_OPERATOR = 'admin'
$env:MEOWDOKU_STATE_DIR = 'C:\dev\Powershell\srv\state'

# Switches go by name (a hashtable): an array would hand '-Status' to the script as a plain word.
function Step([string]$script, [string[]]$switches = @()) {
    Write-Host "`n>> $script $(($switches | ForEach-Object { "-$_" }) -join ' ')" -ForegroundColor Cyan
    $named = @{}; foreach ($name in $switches) { $named[$name] = $true }
    & (Join-Path $Deploy $script) @named | Out-Host
}
function Need-Elevation {
    $me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw "'$Command' needs an elevated shell (Run as administrator, or SSH)." }
}
# The same rule as Restart-Stack.ps1: a photo job that changed state in the last ten minutes and is not finished
# dies with the stack. Restart-Stack.ps1 itself is the console-mode restart; Apache runs as the Apache2.4 service here.
function Assert-NoPhoto {
    $photos = Join-Path $env:MEOWDOKU_STATE_DIR 'content\marble_photos'
    $recent = Get-ChildItem $photos -Directory -ErrorAction SilentlyContinue | ForEach-Object { Join-Path $_.FullName 'status.json' } |
        Where-Object { (Test-Path $_) -and ((Get-Item $_).LastWriteTime -gt (Get-Date).AddMinutes(-10)) } |
        Where-Object { (Get-Content $_ -Raw | ConvertFrom-Json).state -notin @('complete', 'error', 'canceled', 'failed') }
    if ($recent -and -not $Force) { throw "A photo job is running ($($recent -join ', ')). Wait, or add -Force." }
}
function Invoke-Restart {
    Need-Elevation; Assert-NoPhoto
    $taskName = 'SpeakToMe hosted llama'
    # Register the on-demand task and its Apache service permission before touching the live stack.
    # This also migrates an older task that used to start independently at Windows boot.
    Write-Host "`n>> Start-HostedLlm.ps1 -Register" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & (Join-Path $Deploy 'Start-HostedLlm.ps1') -Register | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Could not register $taskName; stack restart canceled." }

    Stop-Llama $taskName

    Write-Host "`n>> Apache-Service.ps1 -Restart" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & (Join-Path $Deploy 'Apache-Service.ps1') -Restart | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Apache restart failed; Llama remains stopped.' }
    if ((Get-Service -Name 'Apache2.4').Status -ne 'Running') { throw 'Apache did not restart; Llama remains stopped.' }
    Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Write-Host "Started $taskName with the stack."
    Start-Tunnel
    Step 'Status.ps1'
}
# Stop the hosted Llama task and any model process it left listening (the pre-migration runner could leave its
# llama-server child behind when the task stops). Throws, with the stack untouched, when it cannot.
function Stop-Llama([string]$taskName) {
    Write-Host "`n>> Stop-ScheduledTask $taskName" -ForegroundColor Cyan
    if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction Stop
    }
    $deadline = (Get-Date).AddSeconds(15)
    do {
        if ((Get-ScheduledTask -TaskName $taskName).State -ne 'Running') { break }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    if ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') { throw "$taskName did not stop; stack restart canceled." }

    # The pre-migration runner could leave its llama-server child behind when the task stops.
    $modelExe = 'C:\dev\Powershell\srv\tools\llama-cpp\b12416\llama-server.exe'
    $listeners = @(Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
        if (-not $process -or $process.ExecutablePath -ne $modelExe) {
            throw "Port 11435 is still in use by a process other than the hosted Llama model; stack restart canceled."
        }
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
    }
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) {
        throw 'Hosted Llama model is still listening; stack restart canceled.'
    }
}
# Shut it all down: the tunnel first (nothing new comes in through it), then Llama, then Apache, which takes down the
# stack it supervises. Nothing is restarted; `stm start` (or `stm restart`) brings it back.
function Invoke-Stop {
    Need-Elevation; Assert-NoPhoto
    $taskName = 'SpeakToMe hosted llama'
    Stop-Tunnel
    Stop-Llama $taskName
    Write-Host "`n>> Apache-Service.ps1 -Stop" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & (Join-Path $Deploy 'Apache-Service.ps1') -Stop | Out-Host
    if ($LASTEXITCODE -ne 0 -or (Get-Service -Name 'Apache2.4').Status -ne 'Stopped') { throw 'Apache did not stop.' }
    Write-Host 'Stopped: the tunnel, Llama, Apache and its stack. Bring it back with: .\stm start' -ForegroundColor Yellow
}
function Invoke-Start {
    Need-Elevation
    $taskName = 'SpeakToMe hosted llama'
    Write-Host "`n>> Apache-Service.ps1 -Start" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & (Join-Path $Deploy 'Apache-Service.ps1') -Start | Out-Host
    if ($LASTEXITCODE -ne 0 -or (Get-Service -Name 'Apache2.4').Status -ne 'Running') { throw 'Apache did not start; Llama and the tunnel were left stopped.' }
    Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Write-Host "Started $taskName with the stack."
    Start-Tunnel
    Step 'Status.ps1'
}
# Stop the tunnel this script starts (tunnel-client run with the speaktome-guest config); nothing when it is not running.
function Stop-Tunnel {
    $config = Join-Path $env:APPDATA 'tunnel-client\speaktome-guest.yaml'
    Write-Host "`n>> stop tunnel-client ($config)" -ForegroundColor Cyan
    $running = @(Get-CimInstance Win32_Process -Filter "Name='tunnel-client.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$config*" })
    if (-not $running) { Write-Host 'The tunnel is not running.'; return }
    foreach ($p in $running) { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop }
    Write-Host "Stopped the tunnel (pid $($running.ProcessId -join ', '))."
}
# The OpenAI MCP tunnel (ChatGPT's way in to /mcp/guest): its runtime 'speaktome-guest' runs with its saved config as
# it is (key, tunnel id, log and health files). Started with the stack when it is not already running.
function Start-Tunnel {
    $tunnel = 'C:\dev\Powershell\srv\tools\openai-tunnel-client\v0.0.21\tunnel-client.exe'
    $config = Join-Path $env:APPDATA 'tunnel-client\speaktome-guest.yaml'
    Write-Host "`n>> tunnel-client run --config $config" -ForegroundColor Cyan
    $running = Get-CimInstance Win32_Process -Filter "Name='tunnel-client.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like "*$config*" }
    if ($running) { Write-Host "The tunnel is already running (pid $($running.ProcessId -join ', '))."; return }
    if (-not (Test-Path -LiteralPath $config)) { throw "No tunnel config at $config; the tunnel was not started." }
    $process = Start-Process -FilePath $tunnel -ArgumentList @('run', '--config', "`"$config`"") -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 5
    if ($process.HasExited) { throw "The tunnel exited at once (code $($process.ExitCode)); see its log." }
    Write-Host "Started the tunnel (pid $($process.Id))."
    & $tunnel runtimes status speaktome-guest | Out-Host
}
# The Oracle A1 host (project ARM cloud-ready): Apache there fronts speaktome.dev and its subdomains
# (deploy/apache/oracle-front.conf) and owns its own stack (deploy/apache/oracle-stack.conf). It is reached over SSH
# as ubuntu with this machine's key; `oracle.speaktome.dev` is its DNS name (Cloudflare A record, DNS only).
# What a publish has to do there, in order:
#   1. as nodus (owner of /srv/powershell and its checkouts): deploy/linux/stm.sh publish --export-web
#      -- each live checkout fast-forwards to what landed (publish.sh; a shallow clone is deepened first), and the
#      pages re-export into turing/build/meowdoku/web, which Oracle's instance serves.
#   2. as root: deploy/linux/stm.sh front -- the landed oracle-front.conf replaces the served one if it differs (the
#      old one is kept as .previous and put back if configtest fails), then a graceful reload. The reload restarts the
#      piped supervisor, so Oracle's instances restart on the moved code.
# Both run the operator gate on Oracle's state root: the one its installed oracle-stack.conf names
# (SPEAKTOME_STACK_STATE), read on the host so this script never guesses it, with that state's operator account.
$OracleHost = 'ubuntu@oracle.speaktome.dev'
$OracleKey = Join-Path $HOME '.ssh\oracle_nodus'
$OracleOperator = 'claude'   # the operator account whose credentials Oracle's state root holds
function Invoke-OraclePublish {
    if ($NoOracle) { Write-Host "`nOracle skipped (-NoOracle)." -ForegroundColor Yellow; return }
    Write-Host "`n>> Oracle ($OracleHost): stm.sh publish --export-web, then stm.sh front" -ForegroundColor Cyan
    $script = @"
set -euo pipefail
stm=/srv/powershell/turing/examples/meowdoku/deploy/linux/stm.sh
[ -f "`$stm" ] || { echo "REFUSED no `$stm: Oracle's turing checkout predates the Linux deploy scripts"; exit 2; }
state=`$(sed -n 's/^Define SPEAKTOME_STACK_STATE *"\(.*\)"/\1/p' /etc/apache2/conf-available/oracle-stack.conf)
[ -n "`$state" ] || { echo 'REFUSED oracle-stack.conf names no SPEAKTOME_STACK_STATE'; exit 2; }
op=$OracleOperator
sudo -u nodus env MEOWDOKU_OPERATOR="`$op" MEOWDOKU_STATE_DIR="`$state" bash "`$stm" publish --export-web
sudo env MEOWDOKU_OPERATOR="`$op" MEOWDOKU_STATE_DIR="`$state" bash "`$stm" front
"@ -replace "`r", ''
    $global:LASTEXITCODE = 0
    $script | & ssh -i $OracleKey -o ConnectTimeout=20 -o BatchMode=yes $OracleHost 'bash -s' | Out-Host
    if ($LASTEXITCODE -ne 0) { Write-Warning "Oracle publish did not finish (exit $LASTEXITCODE); home's publish stands." }
}

switch ($Command) {
    'help'    { $t = Get-Content -Raw $PSCommandPath; Write-Host $t.Substring(2, $t.IndexOf('#>') - 2).Trim() }
    'status'  {
        Step 'Apache-Service.ps1' @('Status')
        Step 'Status.ps1'
        Step 'Update-LiveCheckouts.ps1' @('WhatIf')
        Step 'Start-AgentDay.ps1' @('Status')
        Step 'Start-HostedLlm.ps1' @('Status')
        Step 'Set-SiteCondition.ps1' @('Status')
    }
    'backup'  { Step 'Backup-State.ps1' }
    'merge'   { Step 'Update-LiveCheckouts.ps1' $Move }
    'publish' { Step 'Update-LiveCheckouts.ps1' (@('ExportWeb') + $Move); Invoke-OraclePublish }
    'restart' { Invoke-Restart }
    'stop'    { Invoke-Stop }
    'start'   { Invoke-Start }
    'full'    {
        Need-Elevation; Assert-NoPhoto
        Step 'Backup-State.ps1'
        Step 'Update-LiveCheckouts.ps1' (@('ExportWeb') + $Move)
        Invoke-OraclePublish
        Write-Host "`nRead the table above. A 'skipped', 'ahead', 'diverged', 'blocked' or 'failed' repository did not move." -ForegroundColor Yellow
        if ((Read-Host 'Restart the stack now? (y/n)') -eq 'y') { Invoke-Restart } else { Write-Host 'Not restarted; the moved code loads at the next restart.' }
    }
    'llama'   {
        Need-Elevation
        Stop-ScheduledTask -TaskName 'SpeakToMe hosted llama' -ErrorAction SilentlyContinue
        Step 'Start-HostedLlm.ps1' @('Run')
        Step 'Start-HostedLlm.ps1' @('Status')
    }
    'agents'  { Step 'Start-AgentDay.ps1' @('Register') }
    'agent'   {
        $registry = Join-Path $Deploy '..\external_agents.py'
        if (-not (Test-Path -LiteralPath (Join-Path $Deploy 'Install-ExternalAgent.ps1'))) { throw "Install-ExternalAgent.ps1 is not in the live checkout yet: land case 989e507e and 'stm merge' first." }
        $python = 'C:\dev\Powershell\srv\python\python.exe'
        if ($Action -ne 'list' -and $Name -notmatch '^[a-z][a-z0-9-]{1,12}$') { throw "name the agent: stm agent $Action <name>" }
        switch ($Action) {
            'list'     { (& $python $registry --state-dir $env:MEOWDOKU_STATE_DIR list | ConvertFrom-Json) |
                           Format-Table name, provider, model, @{ n = 'minute'; e = { ':{0:00}' -f $_.minute } }, account, status, note -AutoSize | Out-Host }
            'install'  { Need-Elevation; & (Join-Path $Deploy 'Install-ExternalAgent.ps1') -Name $Name }
            'key'      { Need-Elevation; & (Join-Path $Deploy 'Install-ExternalAgent.ps1') -Name $Name -RotateKey }
            'run'      { & (Join-Path $Deploy 'Start-AgentDay.ps1') -RunOne $Name | Out-Host }
            'withdraw' { & $python $registry --state-dir $env:MEOWDOKU_STATE_DIR mark $Name withdrawn --note "withdrawn by stm" | Out-Null
                         Step 'Start-AgentDay.ps1' @('Register') }
        }
    }
}

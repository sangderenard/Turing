<#
.SYNOPSIS  The round table's runner. A room's round table runs only while it is switched on in the room (the room page,
chat_rooms.py): the room shows that it is on and whose turn it is, so a table is never mistaken for the usual chat.

Runs as the operator and needs no elevation. -Watch keeps a lookout (roundtable_seat.py --agent none, as -LookoutAgent's
account) reading the room's switch. When it is switched on, the runner starts a seat for each agent among the seats, as
that agent's own Windows account (its saved password, Set-AgentBox.ps1 -ResetPassword; AgentAccounts.psm1 Start-AsAgent
-Redirect), hidden, with standard input and output kept open, and passes the turn down them one JSON line each way. On an
agent's turn its seat gives it the room's last n-1 messages (n: the seats) and posts its answer as the agent's own
account. On a person's turn a seat waits for that person to write in the room. Each seat reports whose turn it is; the
runner never signs in and never sees a message.

The table stops at the next turn after it is switched off. When it ends by itself (a person has not written for
-PersonWaitMinutes, or no agent seat is left) a seat switches it off and says why, and the room shows that. Then the
lookout watches again. While a seat is open, Start-AgentDay.ps1 does not start that agent's hourly or bell runs (each seat
is recorded in <AgentDays>\running). A turn that runs past -TurnMinutes is skipped; an agent answers "(pass)" to say
nothing in a round. What the runner does is written to <AgentDays>\roundtable.log for agents, or
<AgentDays>\roundtable-<room>.log for another room. Each room has its own task; the agents task keeps its existing name.
.EXAMPLE   .\Start-RoundTable.ps1 -Register        # the agents room's task, while you are signed in
.EXAMPLE   .\Start-RoundTable.ps1 -Register -Room research  # a separate task for Research
.EXAMPLE   .\Start-RoundTable.ps1 -Watch -Room research  # run Research in this window (Ctrl+C stops it)
.EXAMPLE   .\Start-RoundTable.ps1 -Status -Room research # the task, last lines, and open seats
.EXAMPLE   .\Start-RoundTable.ps1 -Unregister -Room research
#>
#Requires -Version 7
param(
    [switch]$Watch,
    [switch]$Status,
    [switch]$Register,
    [switch]$Unregister,
    [ValidatePattern('^[a-z0-9][a-z0-9-]{2,39}$')][string]$Room = 'agents',
    [string]$LookoutAgent = 'gemini',
    [int]$TurnMinutes = 10,
    [int]$PersonWaitMinutes = 60,
    [string]$AgentDays = (Join-Path $env:USERPROFILE 'AgentDays'),
    [string]$AccountsDb = $(if ($env:MEOWDOKU_STATE_DIR) { Join-Path $env:MEOWDOKU_STATE_DIR 'sql\accounts.db' } else { 'C:\dev\Powershell\srv\state\sql\accounts.db' }),
    [string]$GeminiModel = 'gemini-3.5-flash-lite'
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$Workspace = 'C:\dev\Powershell'
$Python = "$Workspace\srv\python\python.exe"
$TaskName = if ($Room -eq 'agents') { 'SpeakToMe round table' } else { "SpeakToMe round table $Room" }
$LogName = if ($Room -eq 'agents') { 'roundtable.log' } else { "roundtable-$Room.log" }
# The agents' Windows accounts and SpeakToMe logins: the same as Start-AgentDay.ps1's $Agents.
$Known = @{ claude = @{ account = 'agents'; login = 'claude' }; codex = @{ account = 'agents-codex'; login = 'codex' }
            gemini = @{ account = 'agents-gemini'; login = 'gemini' } }

function Say([string]$line) {
    $text = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $line"
    Write-Host $text
    New-Item -ItemType Directory -Force -Path $AgentDays | Out-Null
    $file = Join-Path $AgentDays $LogName
    $kept = @(if (Test-Path -LiteralPath $file) { Get-Content -LiteralPath $file -Tail 999 }) + $text
    Set-Content -LiteralPath $file -Value $kept -Encoding utf8
}

# ---------------------------------------------------------------- the task, and status
if ($Register -or $Unregister) {
    if ($Unregister) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue; Write-Host "removed '$TaskName'"; exit 0 }
    $action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'pwsh.exe') `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$PSCommandPath`" -Watch -Room $Room"
    # Every 5 minutes, and a task never starts a second copy of itself: a runner that ended (or failed) is started again.
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
    Write-Host "registered '$TaskName': the runner for the room $Room, started again within 5 minutes if it ends, only while $env:USERNAME is signed in"
    exit 0
}
if ($Status -or -not $Watch) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Write-Host "task '$TaskName': $(if ($task) { $task.State } else { "not registered (-Register -Room $Room)" })"
    if ($task) { Write-Host "watch action: $($task.Actions[0].Arguments)" }
    $seated = @(Get-AgentRecords $AgentDays | Where-Object { $_.kind -eq 'seat' -and $_.room -eq $Room })
    Write-Host "open seats: $(if ($seated) { ($seated | ForEach-Object { "$($_.agent) (pid $($_.pid), room $($_.room))" }) -join '; ' } else { 'none' })"
    $file = Join-Path $AgentDays $LogName
    if (Test-Path -LiteralPath $file) { Write-Host 'the runner, last lines:'; Get-Content -LiteralPath $file -Tail 12 | ForEach-Object { Write-Host "  $_" } }
    exit 0
}

# ---------------------------------------------------------------- seats
$seats = [ordered]@{}

function Show-SeatErrors {
    foreach ($name in @($seats.Keys)) {
        $seat = $seats[$name]
        while ($seat.err -and $seat.err.IsCompleted) {
            $line = $seat.err.Result
            if ($null -eq $line) { $seat.err = $null; break }
            if ($line.Trim()) { Say "  [$name] $line" }
            $seat.err = $seat.process.StandardError.ReadLineAsync()
        }
    }
}

function Open-Seat([string]$key, [string]$agent, [string]$kind, [string[]]$order, [string]$owner) {
    $declared = $Known[$agent]
    $cred = Read-AgentCredential $declared.account
    if (-not $cred) { Say "${agent}: no saved password for $($declared.account) (Set-AgentBox.ps1 -Apply -ResetPassword)"; return $false }
    $arguments = @("$PSScriptRoot\roundtable_seat.py", '--agent', $(if ($kind -eq 'lookout') { 'none' } else { $agent }),
                   '--login', $declared.login, '--room', $Room, '--order', ($order -join ','), '--owner', "$owner",
                   '--turn-minutes', "$TurnMinutes", '--model', $GeminiModel)
    $profile = ''
    $memoryId = ''
    if ($kind -eq 'seat') {
        $profile = (& $Python (Join-Path $PSScriptRoot 'profile_prompt.py') --accounts-db $AccountsDb --individual $declared.login --job roundtable) -join "`n"
        if ($LASTEXITCODE -notin @(0, 2)) { throw "Roundtable profile resolution failed for $agent (exit $LASTEXITCODE)" }
        $memoryPath = Join-Path $AgentDays "$agent\persistence.id"
        if (Test-Path -LiteralPath $memoryPath) { $memoryId = (Get-Content -Raw -LiteralPath $memoryPath).Trim() }
        if ($memoryId -notmatch '^[a-f0-9]{32}$') {
            $memoryId = [Guid]::NewGuid().ToString('N')
            New-Item -ItemType Directory -Force -Path (Split-Path $memoryPath) | Out-Null
            Set-Content -LiteralPath $memoryPath -Value $memoryId -Encoding ascii
        }
        & $Python (Join-Path $PSScriptRoot 'profile_prompt.py') --accounts-db $AccountsDb --individual $declared.login --ensure-memory $memoryId
        if ($LASTEXITCODE -ne 0) { throw "Could not prepare persistence $memoryId for $agent" }
    }
    $process = Start-AsAgent $cred $Python $arguments $Workspace -Hidden -Redirect
    # A seat is recorded, so the agent's day runs wait for it; the lookout runs no agent and is not.
    $record = if ($kind -eq 'seat') { Add-AgentRecord $AgentDays 'seat' $declared.account $process.Id @{ agent = $agent; room = $Room } }
    $seats[$key] = @{ process = $process; record = $record; out = $process.StandardOutput.ReadLineAsync()
                      err = $process.StandardError.ReadLineAsync(); profile = $profile; memory_id = $memoryId }
    $ready = Read-Seat $key $null 240
    if ($ready.op -eq 'ready') { return $true }
    Say "${agent}: the $kind did not start ($($ready.op)$(if ($ready.error) { ": $($ready.error)" }))"
    Close-Seat $key
    $false
}

# The seat's next answer for this turn (an older turn's late answer is dropped), or a timeout.
function Read-Seat([string]$key, $turn, [int]$seconds) {
    $seat = $seats[$key]
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        Show-SeatErrors
        if ($seat.out.IsCompleted) {
            $line = $seat.out.Result
            if ($null -eq $line) { return @{ op = 'gone' } }
            $seat.out = $seat.process.StandardOutput.ReadLineAsync()
            try { $answer = $line | ConvertFrom-Json -AsHashtable } catch { Say "  [$key] $line"; continue }
            if ($null -ne $turn -and $answer.ContainsKey('turn') -and $answer.turn -ne $turn) { continue }
            return $answer
        }
        Start-Sleep -Milliseconds 200
    }
    @{ op = 'timeout' }
}

function Send-Seat([string]$key, [hashtable]$command) {
    $seat = $seats[$key]
    try { $seat.process.StandardInput.WriteLine(($command | ConvertTo-Json -Compress)); $seat.process.StandardInput.Flush(); $true }
    catch { $false }
}

function Close-Seat([string]$key) {
    $seat = $seats[$key]
    [void](Send-Seat $key @{ op = 'stop' })
    try { $seat.process.StandardInput.Close() } catch { }
    if (-not $seat.process.WaitForExit(20000)) { try { $seat.process.Kill($true) } catch { } }
    if ($seat.record) { Remove-Item -LiteralPath $seat.record -Force -ErrorAction SilentlyContinue }
    $seats.Remove($key)
}

# Switch the table off, saying why, through a seat that sits at it (or the lookout, when its account does).
function End-Table([string]$note, [string[]]$order) {
    $by = @($seats.Keys | Where-Object { $_ -ne 'lookout' }) + @(if ($order -contains $Known[$LookoutAgent].login) { 'lookout' })
    foreach ($key in $by) {
        if ((Send-Seat $key @{ op = 'end'; note = $note }) -and (Read-Seat $key $null 60).op -eq 'ended') { Say "the table is switched off: $note"; return }
    }
    Say "the table ended ($note) but no seat could switch it off; switch it off on the room page"
}

# ---------------------------------------------------------------- one table, while it is on
function Invoke-Table([string[]]$order, [string]$owner) {
    $agents = @($order | Where-Object { $Known.ContainsKey($_) })
    foreach ($agent in $agents) {
        $busy = @(Get-AgentRecords $AgentDays | Where-Object { $_.account -eq $Known[$agent].account })
        if ($busy) { Say "${agent}: a $($busy[0].kind) is already open as $($Known[$agent].account) (pid $($busy[0].pid)); no seat this table"; continue }
        if (Open-Seat $agent $agent 'seat' $order $owner) { Say "${agent}: seated as $($Known[$agent].account)" }
    }
    if (-not @($seats.Keys | Where-Object { $_ -ne 'lookout' }).Count) { End-Table 'no agent seat could start' $order; return }
    $tail = [Math]::Max(1, $order.Count - 1)
    Say "the table runs in $Room, in this order: $($order -join ', '); each agent is given the last $tail messages"
    $turn = 0; $round = 1; $lastId = $null
    while ($true) {
        foreach ($name in $order) {
            $turn++
            $agentSeats = @($seats.Keys | Where-Object { $_ -ne 'lookout' })
            if (-not $agentSeats.Count) { End-Table 'no agent seat is left' $order; return }
            if (-not $Known.ContainsKey($name)) {                        # a person's turn: a seat waits for them to write
                $waiter = $agentSeats[0]
                $command = @{ op = 'wait'; turn = $turn; round = $round; for = $name; seconds = $PersonWaitMinutes * 60 }
                if ($null -ne $lastId) { $command.after = $lastId }
                if (-not (Send-Seat $waiter $command)) { Say "${waiter}: the seat is gone"; Close-Seat $waiter; continue }
                $answer = Read-Seat $waiter $turn ($PersonWaitMinutes * 60 + 120)
                switch ($answer.op) {
                    'posted' { $lastId = $answer.id; Say "round ${round}: $name wrote (message $($answer.id))" }
                    'off' { Say 'the table was switched off'; return }
                    'timeout' { End-Table "$name did not write for $PersonWaitMinutes minutes" $order; return }
                    'gone' { Say "${waiter}: the seat ended"; Close-Seat $waiter }
                    default { End-Table "the wait for $name failed: $($answer.error)" $order; return }
                }
                continue
            }
            if (-not $seats.Contains($name)) { continue }
            if (-not (Send-Seat $name @{ op = 'turn'; turn = $turn; round = $round; tail = $tail;
                                       profile = $seats[$name].profile; memory_id = $seats[$name].memory_id })) { $answer = @{ op = 'gone' } }
            else { $answer = Read-Seat $name $turn ($TurnMinutes * 60 + 60) }
            switch ($answer.op) {
                'posted' { $lastId = $answer.id; Say "round ${round}: $name posted message $($answer.id)" }
                'passed' { Say "round ${round}: $name passed" }
                'off' { Say 'the table was switched off'; return }
                'timeout' { Say "round ${round}: $name gave no answer within $TurnMinutes minutes; skipped" }
                'gone' { Say "${name}: the seat ended"; Close-Seat $name }
                default { Say "round ${round}: ${name}: $($answer.error)" }
            }
        }
        $round++
    }
}

# ---------------------------------------------------------------- the runner
try {
    if (-not $Known.ContainsKey($LookoutAgent)) { Say "the lookout must be an agent: $($Known.Keys -join ', ')"; exit 2 }
    Say "the runner starts for the room $Room"
    $ran = $null                       # when the last table this runner ran was switched on
    while ($true) {
        if (-not $seats.Contains('lookout') -and -not (Open-Seat 'lookout' $LookoutAgent 'lookout' @() '')) {
            Say 'the lookout could not start; the runner ends (the task starts it again)'; exit 1
        }
        if (-not (Send-Seat 'lookout' @{ op = 'watch'; seconds = 3600 })) { Close-Seat 'lookout'; continue }
        $answer = Read-Seat 'lookout' $null 3700
        if ($answer.op -eq 'gone') { Say 'the lookout ended; starting it again'; Close-Seat 'lookout'; continue }
        if ($answer.op -ne 'on') { continue }                            # a watch that ran its hour: watch again
        if ($null -ne $ran -and $answer.switched_at -eq $ran) {           # the table just run, still on: never run it twice
            Start-Sleep -Seconds 30; continue
        }
        $ran = $answer.switched_at
        $order = @($answer.seats); $owner = [string]$answer.owner
        Say "the round table was switched on by $owner, seats: $($order -join ', ')"
        try { Invoke-Table $order $owner }
        finally { foreach ($key in @($seats.Keys | Where-Object { $_ -ne 'lookout' })) { Close-Seat $key; Say "${key}: seat closed" } }
    }
}
finally {
    foreach ($key in @($seats.Keys)) { Close-Seat $key }
    Say 'the runner stopped'
}

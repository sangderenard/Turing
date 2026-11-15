<#
.SYNOPSIS  A round table in a SpeakToMe room: the owner and the agents speak in turn, round after round. Each agent is
one process kept running as its own Windows account for the whole table, and on its turn it is given the room's last
n-1 messages (n: the seats, the owner included), which is everything said since its last turn.

Runs as the operator and needs no elevation. For each agent in -Order it starts roundtable_seat.py as that agent's
account (its saved password, Set-AgentBox.ps1 -ResetPassword; AgentAccounts.psm1 Start-AsAgent -Redirect), hidden,
with its standard input and output kept open: this script passes the turn down them, one JSON line each way. The seat
reads the room, gives the agent the messages, and posts the agent's answer to the room as the agent's own SpeakToMe
account; this script never signs in and never sees a message. On the owner's turn a seat waits for the owner to write
in the room (the room page: /rooms.html?room=<room>).

While a seat is open, Start-AgentDay.ps1 does not start that agent's hourly or bell runs: each seat is recorded in
<AgentDays>\running, as a console is. The table ends after -Rounds rounds, when the owner has not written for
-OwnerWaitMinutes, when no seat is left, or on Ctrl+C; the seats are then stopped.
An agent answers "(pass)" to say nothing in a round. A turn that runs past -TurnMinutes is skipped.
.EXAMPLE   .\Start-RoundTable.ps1                                   # admin, claude, codex, gemini in the agents room
.EXAMPLE   .\Start-RoundTable.ps1 -Order admin,gemini -Rounds 2     # a short trial with one agent
.EXAMPLE   .\Start-RoundTable.ps1 -AgentsFirst                      # the agents answer what is already in the room first
#>
#Requires -Version 7
param(
    [string]$Room = 'agents',
    [string[]]$Order = @('admin', 'claude', 'codex', 'gemini'),
    [string]$Owner = 'admin',
    [int]$Tail = 0,
    [int]$Rounds = 0,
    [int]$TurnMinutes = 10,
    [int]$OwnerWaitMinutes = 60,
    [switch]$AgentsFirst,
    [string]$AgentDays = (Join-Path $env:USERPROFILE 'AgentDays'),
    [string]$GeminiModel = 'gemini-3.5-flash-lite'
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$Workspace = 'C:\dev\Powershell'
$Python = "$Workspace\srv\python\python.exe"
# The agents' Windows accounts and SpeakToMe logins: the same as Start-AgentDay.ps1's $Agents.
$Known = @{ claude = @{ account = 'agents'; login = 'claude' }; codex = @{ account = 'agents-codex'; login = 'codex' }
            gemini = @{ account = 'agents-gemini'; login = 'gemini' } }

function Say([string]$line) { Write-Host "$(Get-Date -Format 'HH:mm:ss') $line" }

$Order = @($Order | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Order -notcontains $Owner) { Say "the order must include the owner $Owner"; exit 2 }
$unknown = @($Order | Where-Object { $_ -ne $Owner -and -not $Known.ContainsKey($_) })
if ($unknown) { Say "not an agent: $($unknown -join ', ') (agents: $($Known.Keys -join ', '))"; exit 2 }
$tailCount = if ($Tail -gt 0) { $Tail } else { [Math]::Max(1, $Order.Count - 1) }
if ($AgentsFirst) {                                   # start just after the owner, so the owner speaks last in each round
    $at = [Array]::IndexOf($Order, $Owner)
    $Order = @($Order | Select-Object -Skip ($at + 1)) + @($Order | Select-Object -First ($at + 1))
}

$seats = [ordered]@{}

function Show-SeatErrors {
    foreach ($name in @($seats.Keys)) {
        $seat = $seats[$name]
        while ($seat.err -and $seat.err.IsCompleted) {
            $line = $seat.err.Result
            if ($null -eq $line) { $seat.err = $null; break }
            if ($line.Trim()) { Write-Host "  [$name] $line" }
            $seat.err = $seat.process.StandardError.ReadLineAsync()
        }
    }
}

# The seat's next answer for this turn (an older turn's late answer is dropped), or a timeout.
function Read-Seat([string]$name, $turn, [int]$seconds) {
    $seat = $seats[$name]
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        Show-SeatErrors
        if ($seat.out.IsCompleted) {
            $line = $seat.out.Result
            if ($null -eq $line) { return @{ op = 'gone' } }
            $seat.out = $seat.process.StandardOutput.ReadLineAsync()
            try { $answer = $line | ConvertFrom-Json -AsHashtable } catch { Write-Host "  [$name] $line"; continue }
            if ($null -ne $turn -and $answer.ContainsKey('turn') -and $answer.turn -ne $turn) { continue }
            return $answer
        }
        Start-Sleep -Milliseconds 200
    }
    @{ op = 'timeout' }
}

function Send-Seat([string]$name, [hashtable]$command) {
    $seat = $seats[$name]
    try { $seat.process.StandardInput.WriteLine(($command | ConvertTo-Json -Compress)); $seat.process.StandardInput.Flush(); $true }
    catch { $false }
}

function Close-Seat([string]$name) {
    $seat = $seats[$name]
    [void](Send-Seat $name @{ op = 'stop' })
    try { $seat.process.StandardInput.Close() } catch { }
    if (-not $seat.process.WaitForExit(20000)) { try { $seat.process.Kill($true) } catch { } }
    Remove-Item -LiteralPath $seat.record -Force -ErrorAction SilentlyContinue
    $seats.Remove($name)
}

try {
    # ---------------------------------------------------------------- the seats
    $records = @(Get-AgentRecords $AgentDays)
    foreach ($name in $Order | Where-Object { $_ -ne $Owner }) {
        $declared = $Known[$name]
        $busy = @($records | Where-Object { $_.account -eq $declared.account })
        if ($busy) { Say "${name}: $($busy[0].kind) already open as $($declared.account) (pid $($busy[0].pid)); left out"; continue }
        $cred = Read-AgentCredential $declared.account
        if (-not $cred) { Say "${name}: no saved password for $($declared.account) (Set-AgentBox.ps1 -Apply -ResetPassword); left out"; continue }
        $arguments = @("$PSScriptRoot\roundtable_seat.py", '--agent', $name, '--login', $declared.login, '--room', $Room,
                       '--order', ($Order -join ','), '--owner', $Owner, '--turn-minutes', "$TurnMinutes", '--model', $GeminiModel)
        $process = Start-AsAgent $cred $Python $arguments $Workspace -Hidden -Redirect
        $record = Add-AgentRecord $AgentDays 'seat' $declared.account $process.Id @{ agent = $name; room = $Room }
        $seats[$name] = @{ process = $process; record = $record; out = $process.StandardOutput.ReadLineAsync()
                           err = $process.StandardError.ReadLineAsync() }
        Say "${name}: seat starting as $($declared.account) (pid $($process.Id))"
    }
    foreach ($name in @($seats.Keys)) {
        $ready = Read-Seat $name $null 240
        if ($ready.op -eq 'ready') { Say "${name}: ready" }
        else { Say "${name}: did not start ($($ready.op)$(if ($ready.error) { ": $($ready.error)" })); left out"; Close-Seat $name }
    }
    if (-not $seats.Count) { Say 'no seat started; the table does not open'; exit 1 }
    $Order = @($Order | Where-Object { $_ -eq $Owner -or $seats.Contains($_) })
    Say "the table is open in the room $Room, in this order: $($Order -join ', '); each agent is given the last $tailCount messages"

    # ---------------------------------------------------------------- the rounds
    $turn = 0
    $round = 1
    $lastId = $null
    :table while ($Rounds -le 0 -or $round -le $Rounds) {
        foreach ($name in $Order) {
            $turn++
            if ($name -eq $Owner) {
                $waiter = @($seats.Keys)[0]
                $command = @{ op = 'wait'; turn = $turn; for = $Owner; seconds = $OwnerWaitMinutes * 60 }
                if ($null -ne $lastId) { $command.after = $lastId }
                Say "round ${round}: ${Owner}'s turn, write in the room $Room"
                if (-not (Send-Seat $waiter $command)) { Say "${waiter}: the seat is gone"; Close-Seat $waiter; break table }
                $answer = Read-Seat $waiter $turn ($OwnerWaitMinutes * 60 + 60)
                if ($answer.op -eq 'posted') { $lastId = $answer.id; Say "${Owner} wrote (message $($answer.id))"; continue }
                Say "$(if ($answer.op -eq 'timeout') { "$Owner did not write within $OwnerWaitMinutes minutes" } else { "the wait failed: $($answer.error)" }); the table ends"
                break table
            }
            if (-not $seats.Contains($name)) { continue }
            Say "round ${round}: ${name}'s turn"
            if (-not (Send-Seat $name @{ op = 'turn'; turn = $turn; round = $round; tail = $tailCount })) { $answer = @{ op = 'gone' } }
            else { $answer = Read-Seat $name $turn ($TurnMinutes * 60 + 60) }
            switch ($answer.op) {
                'posted' { $lastId = $answer.id; Say "${name} posted message $($answer.id)" }
                'passed' { Say "${name} passed" }
                'timeout' { Say "${name}: no answer within $TurnMinutes minutes; skipped" }
                'gone' { Say "${name}: the seat ended; left out"; Close-Seat $name }
                default { Say "${name}: $($answer.error)" }
            }
            if (-not $seats.Count) { Say 'no seat is left; the table ends'; break table }
        }
        $round++
    }
}
finally {
    foreach ($name in @($seats.Keys)) { Close-Seat $name; Say "${name}: seat closed" }
}

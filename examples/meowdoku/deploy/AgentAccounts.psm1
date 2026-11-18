# The agent accounts' saved passwords, how a program is started as one of them, and the records of what is running as
# them. Shared by Set-AgentBox.ps1 (which makes the accounts and saves the passwords) and Start-AgentDay.ps1 (which runs
# the agents).

# Each password is saved by Set-AgentBox with Export-Clixml: DPAPI, the operator's key, so it opens only for the operator.
function Get-AgentCredentialFile([string]$Name) {
    Join-Path (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'SpeakToMe\agent-accounts') "$Name.xml"
}

function Read-AgentCredential([string]$Name) {
    $file = Get-AgentCredentialFile $Name
    if (Test-Path -LiteralPath $file) { Import-Clixml -LiteralPath $file } else { $null }
}

# Start a program as an agent account with that account's own environment. Start-Process -Credential copies the caller's
# environment (HOME, TEMP, APPDATA and PATH pointed at the operator's profile: the first -OpenConsole on 2026-11-14 tried
# to write its history and $HOME\gemini.key into C:\Users\alber and was refused), and -UseNewEnvironment gives only the
# machine scope (USERNAME reads SYSTEM). A ProcessStartInfo whose Environment is never touched passes no environment
# block, so CreateProcessWithLogonW builds one from the account's own profile; it also opens a new console by default.
# Its command line is limited to 1024 characters, so longer input goes in on standard input (-Redirect).
function Start-AsAgent([pscredential]$Credential, [string]$File, [string[]]$Arguments, [string]$Directory, [switch]$Hidden, [switch]$Redirect) {
    $info = [System.Diagnostics.ProcessStartInfo]::new($File)
    foreach ($a in $Arguments) { $info.ArgumentList.Add($a) }
    $info.UserName = $Credential.UserName
    $info.Domain = $env:COMPUTERNAME
    $info.Password = $Credential.Password
    $info.LoadUserProfile = $true
    $info.UseShellExecute = $false
    $info.WorkingDirectory = $Directory
    $info.CreateNoWindow = [bool]$Hidden
    if ($Redirect) { $info.RedirectStandardInput = $info.RedirectStandardOutput = $info.RedirectStandardError = $true }
    [System.Diagnostics.Process]::Start($info)
}

# Run a PowerShell script as an agent account, hidden, the script given on standard input. Returns its exit code, its
# output (stdout then stderr) and whether it ran past the timeout (then it is stopped).
function Invoke-AsAgent([pscredential]$Credential, [string]$Script, [string]$Directory, [int]$TimeoutSeconds = 1200,
                        [switch]$GrantDesktopAccess) {
    $lease = $null
    if ($GrantDesktopAccess) {
        if (-not ('AgentDesktopAccess' -as [type])) { Add-Type -Path (Join-Path $PSScriptRoot 'AgentDesktopAccess.cs') }
        $lease = [AgentDesktopAccess]::Grant($Credential.UserName)
    }
    try {
        $process = Start-AsAgent $Credential (Join-Path $PSHOME 'pwsh.exe') @('-NoProfile', '-NonInteractive', '-Command', '-') $Directory -Hidden -Redirect
        $out = $process.StandardOutput.ReadToEndAsync()
        $err = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.Write(($Script -replace "`r", '') + "`n")
        $process.StandardInput.Close()
        $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
        if ($timedOut) { try { $process.Kill($true) } catch { } ; $process.WaitForExit() }
        $text = ($out.Result.TrimEnd(), $err.Result.TrimEnd() | Where-Object { $_ }) -join "`n"
        [pscustomobject]@{ ExitCode = $(if ($timedOut) { $null } else { $process.ExitCode }); Output = $text; TimedOut = $timedOut }
    } finally {
        if ($lease) { $lease.Dispose() }
    }
}

# ---------------------------------------------------------------- what is running as the agents
# One small file per thing running as an agent account: a scheduled run or a console. A record whose process is gone is
# stale and is removed when read. The records live in <AgentDays>\running.
function Get-AgentDaysDir { Join-Path $env:USERPROFILE 'AgentDays' }

function Add-AgentRecord([string]$AgentDays, [string]$Kind, [string]$Account, [int]$ProcessId, [hashtable]$More = @{}) {
    $dir = Join-Path $AgentDays 'running'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $file = Join-Path $dir "$Kind-$Account-$ProcessId.json"
    ([ordered]@{ kind = $Kind; account = $Account; pid = $ProcessId; started = (Get-Date).ToString('o') } + $More) |
        ConvertTo-Json -Compress | Set-Content -LiteralPath $file -Encoding utf8
    $file
}

# A record is alive only while the process that wrote it lives (case fd54c5e9404140238c7d5f1f56d73feb). Windows reuses
# process ids: a record left by a run that a reboot killed named a svchost after the boot, and Codex started no run from
# 20:10 to 21:12 on 2026-11-17. So: the record is from this boot, its pid exists, and that process did not start after
# the record was written. A process whose start time this account may not read keeps its record.
function Test-AgentRecordAlive($Record, [datetime]$Boot) {
    $written = [datetime]$Record.started
    if ($written -lt $Boot) { return $false }
    $process = Get-Process -Id $Record.pid -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    $began = $null
    try { $began = $process.StartTime } catch { }
    if ($began -and $began -gt $written.AddSeconds(1)) { return $false }      # a later process was given this pid
    $true
}

function Get-AgentRecords([string]$AgentDays) {
    $dir = Join-Path $AgentDays 'running'
    if (-not (Test-Path -LiteralPath $dir)) { return @() }
    $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    foreach ($file in Get-ChildItem -LiteralPath $dir -Filter '*.json') {
        try { $record = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json } catch { continue }
        if (Test-AgentRecordAlive $record $boot) { $record }
        else { Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue }
    }
}

# A console as an agent account, recorded so a scheduled run of that account waits while it is open.
function Open-AgentConsole([string]$Account, [string]$Directory, [string]$AgentDays = (Get-AgentDaysDir)) {
    $credential = Read-AgentCredential $Account
    if (-not $credential) { return $null }
    $process = Start-AsAgent $credential (Join-Path $PSHOME 'pwsh.exe') @() $Directory
    Add-AgentRecord $AgentDays 'console' $Account $process.Id | Out-Null
    $process
}

Export-ModuleMember -Function Get-AgentCredentialFile, Read-AgentCredential, Start-AsAgent, Invoke-AsAgent, Get-AgentDaysDir,
    Add-AgentRecord, Get-AgentRecords, Open-AgentConsole

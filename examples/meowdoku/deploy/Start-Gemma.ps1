#Requires -Version 7
# Register the isolated gemma task under its own Windows account; Apache's supervisor owns when it runs.
param([switch]$Register, [switch]$Status, [switch]$GrantRead, [switch]$Run)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$account = 'agents-gemma'
$task = 'SpeakToMe gemma'
$workspace = 'C:\dev\Powershell'
if ($GrantRead -or $Register) {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'GrantRead and Register require an elevated PowerShell after Set-AgentBox created agents-gemma'
    }
}
if ($GrantRead) {
    foreach ($path in @("$workspace\srv\state\models\gemma-4-26b-a4b",
                        "$workspace\srv\tools\llama-cpp\b12416",
                        "$workspace\srv\tools\hermes-agent\0.19.0")) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Missing path: $path" }
        & icacls.exe $path /grant:r "$($account):(OI)(CI)RX" /T
        if ($LASTEXITCODE -ne 0) { throw "Could not grant model read access: $path" }
    }
    Write-Host "$account may read the exact gemma model, llama.cpp and Hermes directories."
    exit 0
}
if ($Register) {
    $credential = Read-AgentCredential $account
    if (-not $credential) { throw "No saved credential for $account; run Set-AgentBox elevated" }
    # Task Scheduler uses a batch logon for password-backed tasks. Preserve the
    # current holders when adding this account, as Setup-ServiceAccount does for
    # its service logon right.
    $sid = (Get-LocalUser -Name $account).SID.Value
    $prefix = Join-Path ([IO.Path]::GetTempPath()) "gemma-secedit-$PID"
    $export = "$prefix.export.inf"; $apply = "$prefix.apply.inf"; $database = "$prefix.sdb"
    try {
        & secedit.exe /export /cfg $export /areas USER_RIGHTS /quiet
        if ($LASTEXITCODE -ne 0) { throw 'Could not read Windows logon rights' }
        $batch = ''
        foreach ($line in Get-Content -LiteralPath $export) {
            if ($line -match '^SeBatchLogonRight\s*=\s*(.*)$') { $batch = $Matches[1].Trim(); break }
        }
        $holders = @($batch -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        # secedit /export names built-in groups by SID (*S-1-5-32-...) but ordinary accounts by NAME, so resolve each holder to
        # its SID before asking whether this account already holds the right (2026-11-17: the name went unrecognised and
        # every -Register tried, and failed, to grant the right again).
        $heldSids = @($holders | ForEach-Object {
            $holder = $_
            if ($holder.StartsWith('*')) { $holder.Substring(1) }
            else { try { ([Security.Principal.NTAccount]$holder).Translate([Security.Principal.SecurityIdentifier]).Value } catch { $holder } }
        })
        if ($heldSids -notcontains $sid) {
            $holders += "*$sid"
            $policy = "[Unicode]`r`nUnicode=yes`r`n[Version]`r`nsignature=`"`$CHICAGO`$`"`r`nRevision=1`r`n[Privilege Rights]`r`nSeBatchLogonRight = $($holders -join ',')`r`n"
            Set-Content -LiteralPath $apply -Value $policy -Encoding Unicode
            & secedit.exe /configure /db $database /cfg $apply /areas USER_RIGHTS /quiet
            if ($LASTEXITCODE -ne 0) { throw 'Could not grant Log on as a batch job to agents-gemma' }
            Write-Host "Granted Log on as a batch job to $account."
        }
    } finally {
        Remove-Item -LiteralPath $export, $apply, $database -Force -ErrorAction SilentlyContinue
    }
    $action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'pwsh.exe') `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$(Join-Path $PSScriptRoot 'Run-Gemma.ps1')`""
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
    $password = $credential.GetNetworkCredential().Password
    try {
        Register-ScheduledTask -TaskName $task -Action $action -Settings $settings `
            -User "$env:COMPUTERNAME\$account" -Password $password -Force | Out-Null
    } finally { $password = $null }
    # Apache's piped-log supervisor owns stack startup/shutdown. Give only its Windows service account
    # read+execute on this task, so it can Run/Stop the isolated agent without its password.
    $apacheSid = (Get-LocalUser -Name 'svc-meowdoku').SID
    $schedule = New-Object -ComObject 'Schedule.Service'
    $schedule.Connect()
    $registered = $schedule.GetFolder('\').GetTask($task)
    $descriptor = [System.Security.AccessControl.RawSecurityDescriptor]::new($registered.GetSecurityDescriptor(4))
    $mask = 0x1200a9
    $allowed = @($descriptor.DiscretionaryAcl | Where-Object {
        $_.SecurityIdentifier -eq $apacheSid -and $_.AceQualifier -eq [System.Security.AccessControl.AceQualifier]::AccessAllowed -and
        ($_.AccessMask -band $mask) -eq $mask
    })
    if (-not $allowed) {
        $ace = [System.Security.AccessControl.CommonAce]::new(
            [System.Security.AccessControl.AceFlags]::None,
            [System.Security.AccessControl.AceQualifier]::AccessAllowed, $mask, $apacheSid, $false, $null)
        $descriptor.DiscretionaryAcl.InsertAce(0, $ace)
        $registered.SetSecurityDescriptor($descriptor.GetSddlForm([System.Security.AccessControl.AccessControlSections]::Access), 0)
    }
    Write-Host "Registered $task for $account and allowed svc-meowdoku to start and stop it with the stack."
    exit 0
}
if ($Run) {
    Start-ScheduledTask -TaskName $task
    Write-Host "Started $task."
    exit 0
}
$registered = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
Write-Host "task: $(if ($registered) { $registered.State } else { 'none' })"

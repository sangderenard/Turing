#Requires -Version 7
# Register the isolated, continuously running Llama intern under its own Windows account.
param([switch]$Register, [switch]$Status, [switch]$GrantRead, [switch]$Run)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AgentAccounts.psm1') -Force
$account = 'agents-llama'
$task = 'SpeakToMe hosted llama'
$workspace = 'C:\dev\Powershell'
$train = 'E:\SpeakToMe\hosted-llm-train'
if ($GrantRead -or $Register) {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'GrantRead and Register require an elevated PowerShell after Set-AgentBox created agents-llama'
    }
}
if ($GrantRead) {
    foreach ($path in @("$workspace\srv\state\models\llama3.2-1b",
                        "$workspace\srv\tools\llama-cpp\b12416", $train)) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) { throw "Missing path: $path" }
        & icacls.exe $path /grant:r "$($account):(OI)(CI)RX" /T
        if ($LASTEXITCODE -ne 0) { throw "Could not grant model read access: $path" }
    }
    Write-Host "$account may read the exact model, llama.cpp and training environment directories."
    exit 0
}
if ($Register) {
    $credential = Read-AgentCredential $account
    if (-not $credential) { throw "No saved credential for $account; run Set-AgentBox elevated" }
    # Task Scheduler uses a batch logon for password-backed tasks. Preserve the
    # current holders when adding this account, as Setup-ServiceAccount does for
    # its service logon right.
    $sid = (Get-LocalUser -Name $account).SID.Value
    $prefix = Join-Path ([IO.Path]::GetTempPath()) "hosted-llm-secedit-$PID"
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
            if ($LASTEXITCODE -ne 0) { throw 'Could not grant Log on as a batch job to agents-llama' }
            Write-Host "Granted Log on as a batch job to $account."
        }
    } finally {
        Remove-Item -LiteralPath $export, $apply, $database -Force -ErrorAction SilentlyContinue
    }
    $action = New-ScheduledTaskAction -Execute (Join-Path $PSHOME 'pwsh.exe') `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -File `"$(Join-Path $PSScriptRoot 'Run-HostedLlm.ps1')`""
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
    $password = $credential.GetNetworkCredential().Password
    try {
        Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Settings $settings `
            -User "$env:COMPUTERNAME\$account" -Password $password -Force | Out-Null
    } finally { $password = $null }
    Write-Host "Registered $task to run as $account at boot and restart on failure."
    exit 0
}
if ($Run) {
    Start-ScheduledTask -TaskName $task
    Write-Host "Started $task."
    exit 0
}
$registered = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
Write-Host "task: $(if ($registered) { $registered.State } else { 'none' })"

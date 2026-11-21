#Requires -Version 7
<#
.SYNOPSIS
Register the owner-run admin control worker to start at boot.
Run -Register once in an elevated PowerShell as the Windows account that owns deployment.
The credential prompt is for that same account; no password is stored in this script.
Use -Status to inspect the task or -Run to start it after registration.
#>
param(
    [switch]$Register,
    [switch]$Run,
    [switch]$Status,
    [switch]$Unregister,
    [string]$StateDir = 'C:\dev\Powershell\srv\state'
)
$ErrorActionPreference = 'Stop'
$taskName = 'SpeakToMe admin control'
$workerPath = Join-Path $PSScriptRoot '..\admin_script_control.py'
$pythonPath = 'C:\dev\Powershell\srv\python\python.exe'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$modeCount = @(@($Register, $Run, $Status, $Unregister) | Where-Object { $_ }).Count
if ($modeCount -ne 1) { throw 'Choose exactly one of -Register, -Run, -Status or -Unregister.' }
if ($Register -or $Unregister) {
    if (-not $elevated) { throw 'Registration requires an elevated PowerShell.' }
}
if ($Register) {
    if (-not (Test-Path -LiteralPath $workerPath -PathType Leaf)) { throw "Worker script missing: $workerPath" }
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Service Python missing: $pythonPath" }
    if (-not (Test-Path -LiteralPath (Join-Path $StateDir 'sql\accounts.db') -PathType Leaf)) { throw "Accounts database missing in $StateDir" }
    $credential = Get-Credential -UserName $identity.Name -Message "Windows password for $($identity.Name), to run the worker at boot"
    if ($credential.UserName -ne $identity.Name) { throw 'The task must run as the current Windows account.' }
    # Password-backed tasks need the batch logon right. Preserve existing holders.
    $sid = $identity.User.Value
    $prefix = Join-Path ([IO.Path]::GetTempPath()) "admin-control-secedit-$PID"
    $export = "$prefix.export.inf"; $apply = "$prefix.apply.inf"; $database = "$prefix.sdb"
    try {
        & secedit.exe /export /cfg $export /areas USER_RIGHTS /quiet
        if ($LASTEXITCODE -ne 0) { throw 'Could not read Windows logon rights.' }
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
            if ($LASTEXITCODE -ne 0) { throw 'Could not grant batch logon right.' }
        }
    } finally {
        Remove-Item -LiteralPath $export, $apply, $database -Force -ErrorAction SilentlyContinue
    }
    $arguments = "--worker --state-dir `"$StateDir`""
    $action = New-ScheduledTaskAction -Execute $pythonPath -Argument "`"$workerPath`" $arguments" -WorkingDirectory $PSScriptRoot
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
    $password = $credential.GetNetworkCredential().Password
    try {
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User $identity.Name -Password $password -RunLevel Highest -Force | Out-Null
    } finally { $password = $null; $credential = $null }
    Write-Host "Registered $taskName as $($identity.Name) at boot."
    Write-Host "Run .\Admin-Control-Service.ps1 -Run to start it now."
    exit 0
}
if ($Run) {
    Start-ScheduledTask -TaskName $taskName
    Write-Host "Started $taskName."
    exit 0
}
if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed $taskName."
    exit 0
}
$registered = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $registered) { Write-Host "task: none"; exit 0 }
$info = Get-ScheduledTaskInfo -TaskName $taskName
[pscustomobject]@{ Task=$taskName; State=$registered.State; User=$registered.Principal.UserId; LastRun=$info.LastRunTime; LastResult=$info.LastTaskResult; NextRun=$info.NextRunTime } | Format-List

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

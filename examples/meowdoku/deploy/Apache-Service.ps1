<#
.SYNOPSIS  Control the Apache2.4 Windows service (installed by Setup-ServiceAccount.ps1). Start/Stop/Restart need an elevated shell.
When the service does not exist the console-mode scripts (Start-Apache.ps1 / Stop-Apache.ps1) apply.
#>
param([switch]$Start, [switch]$Stop, [switch]$Restart, [switch]$Status)
$ErrorActionPreference = 'Stop'
$Svc = 'Apache2.4'
if ((@($Start, $Stop, $Restart, $Status) | Where-Object { $_ }).Count -ne 1) { Write-Host 'usage: Apache-Service.ps1 -Start | -Stop | -Restart | -Status'; exit 1 }
$s = Get-Service -Name $Svc -ErrorAction SilentlyContinue
if (-not $s) {
    Write-Host "Service $Svc is not installed: the console-mode scripts (Start-Apache.ps1 / Stop-Apache.ps1) apply."
    exit 0
}
if ($Start -or $Restart) {         # the declared inbound ports' firewall rules come up with the stack (never fatal)
    try { & (Join-Path $PSScriptRoot 'Set-ServiceFirewall.ps1') } catch { Write-Warning "firewall: $($_.Exception.Message)" }
}
if ($Start)   { Start-Service -Name $Svc }
if ($Stop)    { Stop-Service -Name $Svc }
if ($Restart) { Restart-Service -Name $Svc }
Get-Service -Name $Svc | Format-Table Status, Name, StartType -AutoSize

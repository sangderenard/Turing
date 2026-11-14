<#
.SYNOPSIS  Control the Apache2.4 Windows service (installed by Setup-ServiceAccount.ps1). Start/Stop/Restart need an elevated shell.
When the service does not exist the console-mode scripts (Start-Apache.ps1 / Stop-Apache.ps1) apply.
#>
param([switch]$Start, [switch]$Stop, [switch]$Restart, [switch]$Status)
$ErrorActionPreference = 'Stop'
$Svc = 'Apache2.4'
# httpd -E target that Setup-ServiceAccount.ps1 puts in the service's ConfigArgs: the errors httpd hits before it opens logs/error.log.
$StartupLog = 'C:/Apache24/logs/startup-errors.log'
if ((@($Start, $Stop, $Restart, $Status) | Where-Object { $_ }).Count -ne 1) { Write-Host 'usage: Apache-Service.ps1 -Start | -Stop | -Restart | -Status'; exit 1 }
$s = Get-Service -Name $Svc -ErrorAction SilentlyContinue
if (-not $s) {
    Write-Host "Service $Svc is not installed: the console-mode scripts (Start-Apache.ps1 / Stop-Apache.ps1) apply."
    exit 0
}

# Start or restart; on failure print what httpd wrote, so a refused start says why instead of only "Failed to start service".
function Invoke-Start([scriptblock]$Action) {
    Remove-Item -LiteralPath $StartupLog -ErrorAction SilentlyContinue   # what is printed below belongs to this attempt
    try { & $Action }
    catch {
        Write-Host $_.Exception.Message
        if (Test-Path -LiteralPath $StartupLog) {
            Write-Host "httpd startup errors ($StartupLog):"
            Get-Content -LiteralPath $StartupLog
        } else {
            Write-Host "No $StartupLog. Either httpd failed after opening its log (see C:/Apache24/logs/error.log), or the service's"
            Write-Host 'ConfigArgs lack -E (re-run Setup-ServiceAccount.ps1).'
        }
        exit 1
    }
}

if ($Start -or $Restart) {         # the declared inbound ports' firewall rules come up with the stack (never fatal)
    try { & (Join-Path $PSScriptRoot 'Set-ServiceFirewall.ps1') } catch { Write-Warning "firewall: $($_.Exception.Message)" }
}
if ($Start)   { Invoke-Start { Start-Service -Name $Svc } }
if ($Stop)    { Stop-Service -Name $Svc }
if ($Restart) { Invoke-Start { Restart-Service -Name $Svc } }
Get-Service -Name $Svc | Format-Table Status, Name, StartType -AutoSize

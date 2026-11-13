<#
.SYNOPSIS  Bring the Windows Firewall in line with the inbound ports instances.json declares. Run by the scripts that bring
the stack up (Start-Apache.ps1, Apache-Service.ps1 -Start/-Restart), so a port and its rule arrive together.
An instance declares what it listens on from outside:
    "firewall": [{"protocol": "TCP", "port": 47625, "program": "C:/dev/Powershell/srv/python/python.exe"}]
Each declaration of an enabled instance becomes one inbound Allow rule, named "SpeakToMe <instance> <protocol> <port>",
in the rule group "SpeakToMe", limited to that program and that port. A rule of that group that no enabled instance
declares any more is removed. Rules made by hand (8766, 443, the old python.exe prompts) are never touched.
Changing rules needs an elevated shell. Without one this only reports what is missing and goes on: it never stops a
bring-up. -WhatIf reports only.
#>
param([string]$Instances = (Join-Path $PSScriptRoot 'instances.json'), [switch]$WhatIf)

$Group = 'SpeakToMe'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
try {
    $spec = Get-Content -LiteralPath $Instances -Raw | ConvertFrom-Json
} catch {
    Write-Warning "firewall: $Instances is unreadable; no rule was checked"
    return
}
$list = if ($spec.PSObject.Properties.Name -contains 'instances') { $spec.instances } else { $spec }

$wanted = @{}
foreach ($instance in $list) {
    if ($instance.enabled -eq $false -or -not $instance.firewall) { continue }
    foreach ($entry in $instance.firewall) {
        $protocol = ([string]$entry.protocol).ToUpperInvariant()
        if ($protocol -notin @('TCP', 'UDP') -or -not ([int]$entry.port -gt 0) -or -not $entry.program) {
            Write-Warning "firewall: $($instance.name) declares an unusable entry (protocol, port and program are required)"
            continue
        }
        $name = "$Group $($instance.name) $protocol $([int]$entry.port)"
        $wanted[$name] = @{ Protocol = $protocol; Port = [string][int]$entry.port; Program = [IO.Path]::GetFullPath([string]$entry.program) }
    }
}

$change = $admin -and -not $WhatIf
$held = @(Get-NetFirewallRule -Group $Group -ErrorAction SilentlyContinue)
foreach ($rule in $held) {
    if ($wanted.ContainsKey($rule.Name)) { continue }
    if ($change) { Remove-NetFirewallRule -Name $rule.Name; Write-Host "firewall: removed $($rule.Name) (no enabled instance declares it)" }
    else { Write-Host "firewall: $($rule.Name) is no longer declared; an elevated run removes it" }
}
$missing = 0
foreach ($name in $wanted.Keys) {
    $want = $wanted[$name]
    $rule = $held | Where-Object { $_.Name -eq $name }
    if ($rule) {
        $port = $rule | Get-NetFirewallPortFilter
        $app = $rule | Get-NetFirewallApplicationFilter
        $same = $port.Protocol -eq $want.Protocol -and [string]$port.LocalPort -eq $want.Port -and $app.Program -eq $want.Program -and
                $rule.Enabled -eq 'True' -and $rule.Action -eq 'Allow'
        if ($same) { Write-Host "firewall: $name is in place"; continue }
        if ($change) { Remove-NetFirewallRule -Name $name } else { $missing++; Write-Host "firewall: $name differs from its declaration"; continue }
    } elseif (-not $change) {
        $missing++
        Write-Host "firewall: $name is missing (inbound $($want.Protocol) $($want.Port) to $($want.Program))"
        continue
    }
    New-NetFirewallRule -Name $name -DisplayName $name -Group $Group -Direction Inbound -Action Allow -Profile Any `
        -Protocol $want.Protocol -LocalPort $want.Port -Program $want.Program `
        -Description 'Declared in turing examples/meowdoku/deploy/instances.json; kept by Set-ServiceFirewall.ps1.' | Out-Null
    Write-Host "firewall: added $name"
}
if ($missing -and -not $admin) {
    Write-Warning "firewall: $missing declared rule(s) are not in place and this shell is not elevated. Run once, elevated: $PSCommandPath"
}

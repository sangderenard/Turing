<#
.SYNOPSIS  Bring the Windows Firewall in line with the inbound ports instances.json declares. Run by the scripts that bring
the stack up (Start-Apache.ps1, Apache-Service.ps1 -Start/-Restart), so a port and its rule arrive together.
An instance declares what it listens on from outside:
    "firewall": [{"protocol": "TCP", "port": 8766, "program": "C:/dev/Powershell/srv/python/python.exe"}]
A program that is not an instance (Apache) is declared at the top of instances.json, with a name:
    "firewall": [{"name": "apache", "protocol": "TCP", "port": 443, "program": "C:/Apache24/bin/httpd.exe"}]
A port is a number or a range "first-last". Each declaration (of an enabled instance, or at the top) becomes one inbound
Allow rule, named "SpeakToMe <name> <protocol> <port>", in the rule group "SpeakToMe", limited to that program and that
port, on every network profile: the rule does not depend on whether Windows calls the network Public or Private. A rule of that group that no enabled instance
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

$owners = @()
if ($spec.PSObject.Properties.Name -contains 'firewall') {
    foreach ($entry in $spec.firewall) { $owners += [pscustomobject]@{ name = [string]$entry.name; entries = @($entry) } }
}
foreach ($instance in $list) {
    if ($instance.enabled -eq $false -or -not $instance.firewall) { continue }
    $owners += [pscustomobject]@{ name = [string]$instance.name; entries = @($instance.firewall) }
}
$wanted = @{}
foreach ($owner in $owners) {
    foreach ($entry in $owner.entries) {
        $protocol = ([string]$entry.protocol).ToUpperInvariant()
        $port = ([string]$entry.port).Trim()
        $ports = @($port -split '-' | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ })
        $valid = $ports.Count -eq ($port -split '-').Count -and $ports.Count -in @(1, 2) -and ($ports | Where-Object { $_ -lt 1 -or $_ -gt 65535 }).Count -eq 0 -and
                 ($ports.Count -eq 1 -or $ports[0] -le $ports[1])
        if (-not $owner.name -or $protocol -notin @('TCP', 'UDP') -or -not $valid -or -not $entry.program) {
            Write-Warning "firewall: '$($owner.name)' declares an unusable entry (name, protocol, port or range, and program are required)"
            continue
        }
        $port = ($ports -join '-')
        $name = "$Group $($owner.name) $protocol $port"
        $wanted[$name] = @{ Protocol = $protocol; Port = $port; Program = [IO.Path]::GetFullPath([string]$entry.program) }
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
        $portFilter = $rule | Get-NetFirewallPortFilter
        $app = $rule | Get-NetFirewallApplicationFilter
        $same = $portFilter.Protocol -eq $want.Protocol -and (@($portFilter.LocalPort) -join ',') -eq $want.Port -and $app.Program -eq $want.Program -and
                $rule.Enabled -eq 'True' -and $rule.Action -eq 'Allow' -and $rule.Profile -eq 'Any'
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

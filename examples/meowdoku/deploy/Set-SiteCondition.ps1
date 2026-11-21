<#
.SYNOPSIS  Post or take down the site's condition card at the Apache front (apache/meowdoku-proxy.conf, "site condition").
.DESCRIPTION
  -Yellow  "We are under mild attack": every new request through Apache gets the yellow card (503, Retry-After 300).
  -Red     "Under genuine threat, or down": the red card instead.
  -Well    Take the card down: requests reach the site again.
  -Status  (the default) Say which card is posted, since when and by whom.
  One card is posted at a time; posting one takes the other down. Apache tests the flag file on every request, so this
  needs no restart and no elevation. Open WebSocket tunnels are not cut. The cards are copied from apache/condition on
  every run, which also keeps the red card ready for a backend Apache cannot reach.
.EXAMPLE  .\Set-SiteCondition.ps1 -Yellow -Note "unexpected MCP traffic"
#>
param([switch]$Yellow, [switch]$Red, [switch]$Well, [switch]$Status, [string]$Note = '',
      [string]$Condition = 'C:\dev\Powershell\srv\state\condition')
$ErrorActionPreference = 'Stop'
if ((@($Yellow, $Red, $Well, $Status) | Where-Object { $_ }).Count -gt 1) {
    Write-Host 'usage: Set-SiteCondition.ps1 -Yellow | -Red | -Well | -Status  [-Note text]'; exit 1
}
New-Item -ItemType Directory -Force $Condition | Out-Null
foreach ($card in 'yellow', 'red') {
    Copy-Item (Join-Path $PSScriptRoot "apache\condition\$card.html") (Join-Path $Condition "$card.html") -Force
}
$flags = @{ yellow = Join-Path $Condition 'yellow.on'; red = Join-Path $Condition 'red.on' }
$posted = if ($Yellow) { 'yellow' } elseif ($Red) { 'red' } else { $null }
if ($posted -or $Well) {
    foreach ($name in $flags.Keys) { if ($name -ne $posted) { Remove-Item $flags[$name] -ErrorAction SilentlyContinue } }
}
if ($posted) {
    $line = '{0} by {1}{2}' -f (Get-Date).ToString('s'), $env:USERNAME, $(if ($Note) { ": $Note" } else { '' })
    # Written to a temporary name and moved, so Apache never sees a half-written flag.
    $temp = "$($flags[$posted]).tmp"
    [IO.File]::WriteAllText($temp, $line + "`n")
    Move-Item $temp $flags[$posted] -Force
}
$found = 'red', 'yellow' | Where-Object { Test-Path $flags[$_] } | Select-Object -First 1
if ($found) {
    Write-Host ("{0} card posted, {1}" -f $found, (Get-Content $flags[$found] -Raw).Trim())
} else {
    Write-Host 'No card posted: the site is open.'
}

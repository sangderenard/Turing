<#
.SYNOPSIS  Copy the public (Let's Encrypt) certificate of stichka.duckdns.org from its Posh-ACME order into the service's
secrets, where the mail receiver reads it (deploy/mail.json tls_cert / tls_key). Run by Start-Apache.ps1, so a renewed
certificate reaches the receiver at the next bring-up. Copies only when the order's files are newer than the copy;
never prints their contents. The service config never points into the owner's profile: this copy is the boundary.
The target folder inherits the secrets folder's access (the operator only).
#>
param([string]$Source = 'C:\Users\alber\AppData\Local\Posh-ACME\LE_PROD\3981975032\stichka.duckdns.org',
      [string]$Target = 'C:\dev\Powershell\srv\state\secrets\tls-public',
      [switch]$WhatIf)

$files = @('fullchain.cer', 'cert.key')
if (-not (Test-Path -LiteralPath $Source)) {
    Write-Host "certificate: $Source is not readable from this account; the copy in $Target stays as it is"
    return
}
if (-not $WhatIf) { New-Item -ItemType Directory -Force -Path $Target | Out-Null }
foreach ($name in $files) {
    $from, $to = (Join-Path $Source $name), (Join-Path $Target $name)
    if (-not (Test-Path -LiteralPath $from)) { Write-Warning "certificate: $from is missing"; continue }
    if ((Test-Path -LiteralPath $to) -and (Get-Item -LiteralPath $to).LastWriteTimeUtc -ge (Get-Item -LiteralPath $from).LastWriteTimeUtc) {
        Write-Host "certificate: $name is current"
        continue
    }
    if ($WhatIf) { Write-Host "certificate: would copy $name"; continue }
    Copy-Item -LiteralPath $from -Destination $to -Force
    (Get-Item -LiteralPath $to).LastWriteTimeUtc = (Get-Item -LiteralPath $from).LastWriteTimeUtc
    Write-Host "certificate: copied $name"
}

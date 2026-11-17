#Requires -Version 7
<#
.SYNOPSIS  Create the dev stack's own low-permission Windows account (Stage 4 of Development case 8bbe0de76497481fbb0b86cc107b13f8).

The account (default svc-meowdoku-dev) is for the dev server, the branch servers and the dev Apache (dev-proxy.conf). It
lets a developer or an operator restart them without elevation and without touching stable.

Unlike Setup-ServiceAccount.ps1, this installs NO Windows service. It never touches Apache2.4, httpd.conf, stable's
account or its state.

Run ELEVATED (exit 2 otherwise; -WhatIf may run unelevated). Every step is idempotent and printed before it runs. The
password is generated here and written ONLY to <StateDir>\secrets\operator_credentials\<Name>.windows.secret (a new file,
never overwritten, never echoed).

What the account gets:
  (a) a local user, not in any group but Users, its password never expiring;
  (b) the right to log on as a batch job (the dev supervisor runs as a scheduled task) and NO interactive logon;
  (c) read and execute on the Apache program files, the private interpreter, the dev trees and the central checkouts;
      Modify on the dev state root and the loans state root (created when missing);
  (d) with -IssueTls, its own loopback TLS pair in <DevState>\tls, issued by the stack's loopback CA for 127.0.0.1 and
      localhost only. Stable's key is never given to it.

What it never gets: anything under <StateDir> (the stable state: secrets, private records, the encrypted logs) or the
live checkouts. The last step lists every entry the account holds on those places, and fails when it finds one.

NOT DONE HERE (a decision for the owner, recorded on the case): the branch servers share stable's accounts database
(srv\state\sql\accounts.db) as the one identity authority. Giving this account access to that file is not done by this
script, because that folder also holds private records.

.EXAMPLE   .\Setup-DevAccount.ps1 -WhatIf
.EXAMPLE   .\Setup-DevAccount.ps1 -IssueTls
#>
param(
    [string]$Name = 'svc-meowdoku-dev',
    [string]$StateDir = 'C:/dev/Powershell/srv/state',
    [string]$DevState = 'C:/dev/Powershell/srv/state-dev',
    [string]$LoansState = 'C:/dev/Powershell/srv/state-loans',
    [string]$DevTrees = 'C:/dev/Powershell/srv/deploy/dev',
    [string]$Apache = 'C:/Apache24',
    [string]$Python = 'C:/dev/Powershell/srv/python',
    [string]$OpenSsl = 'C:/Program Files/Git/usr/bin/openssl.exe',
    # The names the dev pair answers to. Loopback is the default, never the only choice: to simulate a remote compute partner,
    # name the network address the dev stack serves on (for example 'IP:192.168.1.232','DNS:dev.example') and turn local
    # access off in dev-proxy.conf (MEOWDOKU_DEV_NO_LOCAL).
    [string[]]$SubjectAltNames = @('IP:127.0.0.1', 'DNS:localhost'),
    [switch]$IssueTls,
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'
$SecretsDir = Join-Path $StateDir 'secrets/operator_credentials'
$SecretFile = Join-Path $SecretsDir "$Name.windows.secret"
$Checkouts = Join-Path $StateDir 'content/checkouts'
$Live = @($StateDir, 'C:/dev/Powershell/turing', 'C:/dev/Powershell/speaktome')
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    if ($WhatIf) { Write-Host '[WhatIf] not elevated: showing commands only (a real run exits 2 here).' }
    else { Write-Host 'REFUSED: run from an elevated PowerShell.'; exit 2 }
}
function Step($t) { Write-Host ''; Write-Host "== $t" }
function Cmd($display) { Write-Host "cmd: $display" }
function Run([string]$exe, [string[]]$arguments) {
    Cmd "$exe $($arguments -join ' ')"
    if ($WhatIf) { return }
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit $LASTEXITCODE" }
}

# ---------------------------------------------------------------- (a) user
Step "(a) local user $Name"
if (Get-LocalUser -Name $Name -ErrorAction SilentlyContinue) {
    Write-Host "user $Name already exists: creation skipped (password left alone)."
} else {
    if (Test-Path -LiteralPath $SecretFile) { throw "REFUSED: $SecretFile exists but user $Name does not. Resolve by hand; it is never overwritten." }
    Cmd "generate 32 random bytes -> url-safe base64 password (not shown); write it ONLY to $SecretFile (a new file)"
    Cmd "New-LocalUser -Name $Name -Password <REDACTED> -PasswordNeverExpires -UserMayNotChangePassword -Description 'Meowdoku dev stack account'"
    if (-not $WhatIf) {
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $plain = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        New-Item -ItemType Directory -Force -Path $SecretsDir | Out-Null
        $fs = [IO.File]::Open($SecretFile, [IO.FileMode]::CreateNew)
        $b = (New-Object Text.UTF8Encoding($false)).GetBytes($plain); $fs.Write($b, 0, $b.Length); $fs.Close()
        New-LocalUser -Name $Name -Password (ConvertTo-SecureString $plain -AsPlainText -Force) -PasswordNeverExpires `
            -UserMayNotChangePassword -Description 'Meowdoku dev stack account' | Out-Null
        $plain = $null
    }
    Run 'icacls.exe' @($SecretFile, '/inheritance:r', '/grant:r', '*S-1-5-32-544:(F)', '*S-1-5-18:(F)', "$env:USERNAME`:(F)")
}

# ---------------------------------------------------------------- (b) logon rights
Step "(b) SeBatchLogonRight + SeDenyInteractiveLogonRight for $Name (existing holders kept)"
$sid = if (-not $WhatIf) { (Get-LocalUser -Name $Name).SID.Value } else { "<SID-of-$Name>" }
$tmp = Join-Path ([IO.Path]::GetTempPath()) "meowdoku-dev-secedit-$PID"
$exportInf = "$tmp.export.inf"; $applyInf = "$tmp.apply.inf"; $sdb = "$tmp.sdb"
Run 'secedit.exe' @('/export', '/cfg', $exportInf, '/areas', 'USER_RIGHTS', '/quiet')
$cur = @{ SeBatchLogonRight = ''; SeDenyInteractiveLogonRight = '' }
if (-not $WhatIf) {
    foreach ($line in Get-Content -LiteralPath $exportInf) {
        foreach ($k in @($cur.Keys)) { if ($line -match "^$k\s*=\s*(.*)$") { $cur[$k] = $Matches[1].Trim() } }
    }
}
function Merge($list, $sid) {
    $items = @($list -split ',' | Where-Object { $_ })
    if ($items -notcontains "*$sid") { $items += "*$sid" }
    return ($items -join ',')
}
$infText = @"
[Unicode]
Unicode=yes
[Version]
signature="`$CHICAGO`$"
Revision=1
[Privilege Rights]
SeBatchLogonRight = $(Merge $cur.SeBatchLogonRight $sid)
SeDenyInteractiveLogonRight = $(Merge $cur.SeDenyInteractiveLogonRight $sid)
"@
Write-Host $infText
if (-not $WhatIf) { Set-Content -LiteralPath $applyInf -Value $infText -Encoding Unicode }
Run 'secedit.exe' @('/configure', '/db', $sdb, '/cfg', $applyInf, '/areas', 'USER_RIGHTS', '/quiet')
if (-not $WhatIf) { Remove-Item -LiteralPath $exportInf, $applyInf, $sdb -Force -ErrorAction SilentlyContinue }

# ---------------------------------------------------------------- (c) ACLs
Step "(c) NTFS ACLs for $Name"
foreach ($d in @($DevState, $LoansState, $DevTrees)) {
    if (-not (Test-Path -LiteralPath $d)) { Cmd "New-Item -ItemType Directory $d"; if (-not $WhatIf) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }
}
Write-Host 'Read+execute with inheritance:'
foreach ($d in @("$Apache/bin", "$Apache/modules", $Python, $DevTrees, $Checkouts)) {
    if (Test-Path -LiteralPath $d) { Run 'icacls.exe' @($d, '/grant', "${Name}:(OI)(CI)RX") }
    else { Cmd "icacls $d /grant ${Name}:(OI)(CI)RX   (missing now: run this script again once it exists)" }
}
# The checkouts live under the stable state root; to reach them the account lists only the folders on the way, never their files.
foreach ($d in @($StateDir, (Join-Path $StateDir 'content'))) { Run 'icacls.exe' @($d, '/grant', "${Name}:(RX)") }
Write-Host 'Modify with inheritance (its own state):'
foreach ($d in @($DevState, $LoansState)) { Run 'icacls.exe' @($d, '/grant', "${Name}:(OI)(CI)M") }

# ---------------------------------------------------------------- (d) its own loopback TLS pair
Step "(d) the dev stack's own TLS pair"
$Tls = Join-Path $DevState 'tls'
$Ca = Join-Path $StateDir 'secrets/tls'
if (-not $IssueTls) {
    Write-Host "skipped (pass -IssueTls). Until then the dev stack has no TLS key of its own; it is never given stable's."
} elseif ((Test-Path -LiteralPath (Join-Path $Tls 'key.pem')) -and (Test-Path -LiteralPath (Join-Path $Tls 'cert.pem'))) {
    Write-Host "$Tls already holds a pair: left alone."
} else {
    if (-not (Test-Path -LiteralPath $OpenSsl)) { throw "openssl not found at $OpenSsl (pass -OpenSsl)" }
    Cmd "New-Item -ItemType Directory $Tls"
    if (-not $WhatIf) { New-Item -ItemType Directory -Force -Path $Tls | Out-Null }
    $ext = Join-Path $Tls 'ext.cnf'
    # pwsh -File hands a list over as one string: accept commas and quotes either way.
    $SubjectAltNames = @($SubjectAltNames | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().Trim("'").Trim('"') } | Where-Object { $_ })
    if (-not $SubjectAltNames -or ($SubjectAltNames | Where-Object { $_ -notmatch '^(IP|DNS):[A-Za-z0-9.:\-]+$' })) {
        throw "SubjectAltNames are entries like IP:192.168.1.232 or DNS:dev.example"
    }
    $extText = "subjectAltName=$($SubjectAltNames -join ',')`nbasicConstraints=CA:FALSE`nkeyUsage=critical,digitalSignature,keyEncipherment`nextendedKeyUsage=serverAuth`n"
    Cmd "write $ext (subjectAltName=$($SubjectAltNames -join ','))"
    if (-not $WhatIf) { Set-Content -LiteralPath $ext -Value $extText -Encoding ascii -NoNewline }
    Run $OpenSsl @('req', '-new', '-newkey', 'rsa:2048', '-nodes', '-keyout', (Join-Path $Tls 'key.pem'), '-subj', '/CN=meowdoku-dev',
                   '-out', (Join-Path $Tls 'dev.csr'))
    Run $OpenSsl @('x509', '-req', '-in', (Join-Path $Tls 'dev.csr'), '-CA', (Join-Path $Ca 'ca.crt'), '-CAkey', (Join-Path $Ca 'ca.key'),
                   '-CAcreateserial', '-days', '825', '-sha256', '-extfile', $ext, '-out', (Join-Path $Tls 'cert.pem'))
    Cmd "copy $Ca/ca.crt -> $Tls/ca.crt (the CA certificate only; its key stays where it is)"
    if (-not $WhatIf) { Copy-Item -LiteralPath (Join-Path $Ca 'ca.crt') -Destination (Join-Path $Tls 'ca.crt') -Force }
    # Only the dev account and the operator may read the key.
    Run 'icacls.exe' @($Tls, '/inheritance:r', '/grant:r', '*S-1-5-32-544:(OI)(CI)(F)', '*S-1-5-18:(OI)(CI)(F)', "$env:USERNAME`:(OI)(CI)(F)", "${Name}:(OI)(CI)R")
}

# ---------------------------------------------------------------- (e) proof: nothing of stable's
Step "(e) $Name holds nothing below the stable state or in the live checkouts"
if ($WhatIf) {
    Write-Host '[WhatIf] the check runs after a real run.'
} else {
    $found = @()
    $listOnly = @((Resolve-Path $StateDir).Path, (Resolve-Path (Join-Path $StateDir 'content')).Path)   # granted (RX) above, folder only
    foreach ($root in $Live) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $places = @(Get-Item -LiteralPath $root) + @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue)
        foreach ($place in $places) {
            $full = $place.FullName
            if ($full -in $listOnly) { continue }
            $acl = Get-Acl -LiteralPath $full
            foreach ($rule in $acl.Access) { if ($rule.IdentityReference.Value -match [regex]::Escape($Name)) { $found += "$full : $($rule.FileSystemRights)" } }
        }
    }
    if ($found) { $found | ForEach-Object { Write-Host "UNEXPECTED: $_" }; throw "$Name holds entries on stable's state or the live checkouts" }
    Write-Host "OK: $Name holds no entry on $($Live -join ', ') or their top-level folders (beyond listing the state root and content)."
}

Write-Host ''
Write-Host 'Next (Stage 4, the owner):'
Write-Host '  - decide how the dev stack reaches the accounts database (the case records the question);'
Write-Host '  - then the dev supervisor is registered as a scheduled task running as this account.'

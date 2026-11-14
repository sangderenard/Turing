<#
Create the dedicated Windows account and install MeowdokuLiaison. Run elevated.
This script neither migrates nor removes any existing Suno credential. The
web instances continue using the interim vault until the liaison is verified.
#>
param([switch]$WhatIf)
$ErrorActionPreference = 'Stop'
$Name = 'svc-liaison'
$Service = 'MeowdokuLiaison'
$State = 'C:/dev/Powershell/srv/state'
$Secrets = Join-Path $State 'secrets'
$Python = 'C:/dev/Powershell/srv/python/python.exe'
$Entry = Join-Path $PSScriptRoot 'LiaisonService.py'
$CanonicalDeploy = 'C:/dev/Powershell/turing/examples/meowdoku/deploy'
$PasswordFile = Join-Path $Secrets 'operator_credentials/svc-liaison.windows.secret'
$KeyFile = Join-Path $Secrets 'liaison-ipc.key'
$Vault = Join-Path $Secrets 'liaison_vault'
$Config = Join-Path $State 'run/mcp_connections.json'

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated -and -not $WhatIf) { throw 'Run in an elevated PowerShell.' }
if (-not $WhatIf -and (Resolve-Path $PSScriptRoot).Path -ne (Resolve-Path $CanonicalDeploy).Path) {
    throw 'Install only from the canonical deployed Turing checkout.'
}
function Run([string]$Exe, [string[]]$Arguments) {
    Write-Host "$Exe $($Arguments -join ' ')"
    if ($WhatIf) { return }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit $LASTEXITCODE" }
}

if (-not (Test-Path -LiteralPath $Entry)) { throw "Liaison service source is missing: $Entry" }
if (-not (Test-Path -LiteralPath $Config)) { throw "Operator MCP config is missing: $Config" }

$account = Get-LocalUser -Name $Name -ErrorAction SilentlyContinue
if (-not $account) {
    if (Test-Path -LiteralPath $PasswordFile) { throw 'Refusing existing password file for a missing account.' }
    Write-Host "Create local account $Name with an unprinted random password"
    if (-not $WhatIf) {
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $plain = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        New-Item -ItemType Directory -Force -Path (Split-Path $PasswordFile) | Out-Null
        $file = [IO.File]::Open($PasswordFile, [IO.FileMode]::CreateNew)
        try {
            $encoded = (New-Object Text.UTF8Encoding($false)).GetBytes($plain)
            $file.Write($encoded, 0, $encoded.Length)
        } finally { $file.Close() }
        $secure = ConvertTo-SecureString $plain -AsPlainText -Force
        New-LocalUser -Name $Name -Password $secure -PasswordNeverExpires -UserMayNotChangePassword -Description 'Meowdoku outgoing MCP liaison' | Out-Null
        $plain = $null
    }
}
if (-not $WhatIf -and -not (Test-Path -LiteralPath $PasswordFile)) { throw 'The service account password file is missing.' }
if (Test-Path -LiteralPath $PasswordFile) {
    Run 'icacls.exe' @($PasswordFile, '/inheritance:r', '/grant:r', 'alber:(F)')
}

# The service runs as its own Windows user. Give it only code/dependency reads,
# traversal of state/secrets, the operator config, and Modify on its own vault.
Run 'icacls.exe' @('C:/dev/Powershell/turing', '/grant', "${Name}:(RX)")
Run 'icacls.exe' @('C:/dev/Powershell/turing/examples', '/grant', "${Name}:(RX)")
Run 'icacls.exe' @('C:/dev/Powershell/turing/examples/meowdoku', '/grant', "${Name}:(OI)(CI)(RX)")
Run 'icacls.exe' @('C:/dev/Powershell/speaktome', '/grant', "${Name}:(RX)")
Run 'icacls.exe' @('C:/dev/Powershell/speaktome/speaktome_api', '/grant', "${Name}:(OI)(CI)(RX)")
Run 'icacls.exe' @('C:/dev/Powershell/srv/python', '/grant', "${Name}:(OI)(CI)(RX)")
Run 'icacls.exe' @($State, '/grant', "${Name}:(X)")
Run 'icacls.exe' @($Secrets, '/grant', "${Name}:(X)")
Run 'icacls.exe' @((Join-Path $State 'run'), '/grant', "${Name}:(X)")
Run 'icacls.exe' @($Config, '/grant', "${Name}:(R)")

if (-not (Test-Path -LiteralPath $Vault)) {
    Write-Host "Create isolated vault $Vault"
    if (-not $WhatIf) { New-Item -ItemType Directory -Path $Vault | Out-Null }
}
if (Test-Path -LiteralPath $Vault) {
    Run 'icacls.exe' @($Vault, '/inheritance:r', '/grant:r', 'alber:(OI)(CI)(F)', "${Name}:(OI)(CI)(M)")
}
if (-not (Test-Path -LiteralPath $KeyFile)) {
    Write-Host 'Create 32-byte liaison IPC key (contents not shown)'
    if (-not $WhatIf) {
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $file = [IO.File]::Open($KeyFile, [IO.FileMode]::CreateNew)
        try { $file.Write($bytes, 0, $bytes.Length) } finally { $file.Close() }
    }
}
if (Test-Path -LiteralPath $KeyFile) {
    if ((Get-Item -LiteralPath $KeyFile).Length -ne 32) { throw 'Liaison IPC key is not 32 bytes.' }
    Run 'icacls.exe' @($KeyFile, '/inheritance:r', '/grant:r', 'alber:(F)', "${Name}:(R)", 'svc-meowdoku:(R)')
}

# Preserve existing holders; allow service logon and deny interactive logon.
$sid = if ($WhatIf) { '<svc-liaison-SID>' } else { (Get-LocalUser -Name $Name).SID.Value }
$tmp = Join-Path ([IO.Path]::GetTempPath()) "liaison-secedit-$PID"
$export = "$tmp.export.inf"; $apply = "$tmp.apply.inf"; $database = "$tmp.sdb"
Run 'secedit.exe' @('/export', '/cfg', $export, '/areas', 'USER_RIGHTS', '/quiet')
$rights = @{ SeServiceLogonRight = ''; SeDenyInteractiveLogonRight = '' }
if (-not $WhatIf) {
    foreach ($line in Get-Content -LiteralPath $export) {
        foreach ($right in @($rights.Keys)) {
            if ($line -match "^$right\s*=\s*(.*)$") { $rights[$right] = $Matches[1].Trim() }
        }
    }
}
foreach ($right in @($rights.Keys)) {
    $holders = @($rights[$right] -split ',' | Where-Object { $_ })
    if ($holders -notcontains "*$sid") { $holders += "*$sid" }
    $rights[$right] = $holders -join ','
}
$inf = "[Unicode]`r`nUnicode=yes`r`n[Version]`r`nsignature=`"`$CHICAGO`"`r`nRevision=1`r`n[Privilege Rights]`r`nSeServiceLogonRight = $($rights.SeServiceLogonRight)`r`nSeDenyInteractiveLogonRight = $($rights.SeDenyInteractiveLogonRight)`r`n"
Write-Host 'Apply service and denied-interactive logon rights while retaining existing holders'
if (-not $WhatIf) {
    try {
        Set-Content -LiteralPath $apply -Value $inf -Encoding Unicode
        Run 'secedit.exe' @('/configure', '/db', $database, '/cfg', $apply, '/areas', 'USER_RIGHTS', '/quiet')
    } finally { Remove-Item -LiteralPath $export, $apply, $database -Force -ErrorAction SilentlyContinue }
}

if (-not (Get-Service -Name $Service -ErrorAction SilentlyContinue)) {
    Write-Host "Install $Service under $Name (password read inside the installer, never passed as an argument)"
    Run $Python @($Entry, 'install-from-file', $PasswordFile)
}
Write-Host 'Installed. Start the service only after checking the isolated vault ACL, then migrate the existing Suno credential.'

<#
Copy existing Suno OAuth records from the working interim vault to the isolated
liaison vault. Never prints or removes credentials. Run after Setup-LiaisonService
has created the vault and before switching the web route to the liaison.
#>
param([switch]$WhatIf)
$ErrorActionPreference = 'Stop'
$Secrets = 'C:/dev/Powershell/srv/state/secrets'
$SourceRoot = Join-Path $Secrets 'mcp_credentials'
$Vault = Join-Path $Secrets 'liaison_vault'
if (-not (Test-Path -LiteralPath $SourceRoot)) { throw 'Interim Suno vault is missing.' }
if (-not (Test-Path -LiteralPath $Vault)) { throw 'Liaison vault is missing. Run Setup-LiaisonService first.' }

$acl = Get-Acl -LiteralPath $Vault
if (-not $acl.AreAccessRulesProtected) { throw 'Liaison vault still inherits permissions.' }
$unsafe = @($acl.Access | Where-Object { $_.IdentityReference.Value -match 'svc-meowdoku|Authenticated Users|BUILTIN\\Users' })
if ($unsafe.Count) { throw 'Liaison vault admits the web account or general users.' }

$copied = 0
foreach ($account in Get-ChildItem -LiteralPath $SourceRoot -Directory -Force) {
    if ($account.Name -notmatch '^[1-9][0-9]*$') { continue }
    $source = Join-Path $account.FullName 'suno.json'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $targetDir = Join-Path $Vault $account.Name
    $target = Join-Path $targetDir 'suno.json'
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
            throw "Existing liaison record differs for account $($account.Name); refusing overwrite."
        }
        continue
    }
    Write-Host "Copy Suno credential for account $($account.Name) (contents not shown)"
    if ($WhatIf) { continue }
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    $temp = Join-Path $targetDir ('.' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::Copy($source, $temp, $false)
        & icacls.exe $temp /inheritance:r /grant:r 'alber:(F)' 'svc-liaison:(M)' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Could not restrict liaison record for account $($account.Name)." }
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash) {
            throw "Suno credential changed during copy for account $($account.Name); retry after activity stops."
        }
        [IO.File]::Move($temp, $target)
        $copied++
    } finally { Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue }
}
Write-Host "Copied $copied Suno credential record(s). Original records were left in place for rollback."

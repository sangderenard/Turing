<#
Enable the liaison only after installing it and copying the interim vault.
Disable removes only the marker, so a managed web restart restores the working
interim path. Neither mode restarts services or removes credentials.
#>
param([switch]$Enable, [switch]$Disable)
$ErrorActionPreference = 'Stop'
if ($Enable -eq $Disable) { throw 'Choose exactly one of -Enable or -Disable.' }
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elevated) { throw 'Run in an elevated PowerShell.' }

$State = 'C:/dev/Powershell/srv/state'
$Secrets = Join-Path $State 'secrets'
$Marker = Join-Path $State 'run/liaison.enabled'
if ($Disable) {
    Remove-Item -LiteralPath $Marker -Force -ErrorAction SilentlyContinue
    Write-Host 'Liaison marker removed. Use the managed web lifecycle to load the interim path.'
    return
}

$service = Get-Service -Name 'MeowdokuLiaison' -ErrorAction SilentlyContinue
if (-not $service -or $service.Status -ne 'Running') { throw 'MeowdokuLiaison must be running.' }
$key = Join-Path $Secrets 'liaison-ipc.key'
if (-not (Test-Path -LiteralPath $key) -or (Get-Item -LiteralPath $key).Length -ne 32) {
    throw 'Liaison IPC key is missing or invalid.'
}
$vault = Join-Path $Secrets 'liaison_vault'
$acl = Get-Acl -LiteralPath $vault
if (-not $acl.AreAccessRulesProtected -or
    @($acl.Access | Where-Object { $_.IdentityReference.Value -match 'svc-meowdoku|Authenticated Users|BUILTIN\\Users' }).Count) {
    throw 'Liaison vault ACL is not isolated from the web account.'
}

$owners = @()
$sourceRoot = Join-Path $Secrets 'mcp_credentials'
foreach ($account in Get-ChildItem -LiteralPath $sourceRoot -Directory -Force) {
    if ($account.Name -notmatch '^[1-9][0-9]*$') { continue }
    $source = Join-Path $account.FullName 'suno.json'
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $target = Join-Path (Join-Path $vault $account.Name) 'suno.json'
    if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
        (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
        throw "Liaison credential copy is missing or stale for account $($account.Name)."
    }
    $targetAcl = Get-Acl -LiteralPath $target
    if (-not $targetAcl.AreAccessRulesProtected -or
        @($targetAcl.Access | Where-Object { $_.IdentityReference.Value -match 'svc-meowdoku|Authenticated Users|BUILTIN\\Users' }).Count) {
        throw "Liaison credential ACL is not isolated for account $($account.Name)."
    }
    $owners += $account.Name
}
if (-not $owners.Count) { throw 'No interim Suno credential was found to migrate.' }

# Authenticate to the local service as the operator without printing tokens,
# tool results, provider URLs, or account credentials.
$check = @'
import sys
from pathlib import Path
sys.path.insert(0, 'C:/dev/Powershell/turing/examples')
from meowdoku.liaison import LiaisonClient
client = LiaisonClient(Path('C:/dev/Powershell/srv/state/secrets/liaison-ipc.key'))
for owner in sys.argv[1:]:
    result = client.status(int(owner), 'suno')
    if not result.get('configured'):
        raise SystemExit(2)
'@
& 'C:/dev/Powershell/srv/python/python.exe' -c $check @owners
if ($LASTEXITCODE -ne 0) { throw 'Liaison did not confirm the migrated credential state.' }

if (Test-Path -LiteralPath $Marker) { throw 'Liaison marker already exists; inspect it before restarting.' }
$file = [IO.File]::Open($Marker, [IO.FileMode]::CreateNew)
try {
    $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes("v1`n")
    $file.Write($bytes, 0, $bytes.Length)
} finally { $file.Close() }
& icacls.exe $Marker /inheritance:r /grant:r 'alber:(F)' 'svc-meowdoku:(R)' | Out-Null
if ($LASTEXITCODE -ne 0) { Remove-Item -LiteralPath $Marker -Force; throw 'Could not restrict the liaison marker.' }
Write-Host 'Liaison marker enabled. Use the managed web lifecycle to load it; no service was restarted.'

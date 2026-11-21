<#
.SYNOPSIS  Undo Set-AgentBox.ps1's folder permissions from the save it made. Run ELEVATED (restoring needs the restore
privilege). Report only unless -Apply. The account it may have made is left: Remove-LocalUser agents removes it.
.EXAMPLE   .\Restore-AgentBoxAcls.ps1 -Manifest C:\dev\acl-backups\agent-box-20261114-130000.json -Apply
#>
param([Parameter(Mandatory)] [string]$Manifest, [switch]$Apply)
$ErrorActionPreference = 'Stop'
$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($Apply -and -not $elevated) { Write-Host 'REFUSED: -Apply needs an elevated PowerShell.'; exit 2 }
if (-not $Apply) { Write-Host '[report only] nothing is changed; run again with -Apply.' }
$saves = @(Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json)
# A folder's own entry was saved by its name, so it is restored from its parent; what is below it was saved relative to
# the folder, so it is restored from the folder. The inner trees go back before the outer folders.
foreach ($s in ($saves | Sort-Object { $_.root.Length } -Descending)) {
    if ($s.tree_file) {
        Write-Host "cmd: icacls `"$($s.root)`" /restore `"$($s.tree_file)`" /C /Q"
        if ($Apply) { & icacls.exe $s.root /restore $s.tree_file /C /Q; if ($LASTEXITCODE -ne 0) { Write-Warning "icacls exited $LASTEXITCODE for $($s.root)" } }
    }
    Write-Host "cmd: icacls `"$($s.parent)`" /restore `"$($s.root_file)`" /C /Q"
    if ($Apply) { & icacls.exe $s.parent /restore $s.root_file /C /Q; if ($LASTEXITCODE -ne 0) { Write-Warning "icacls exited $LASTEXITCODE for $($s.root)" } }
}
Write-Host 'Done. Restart the stack (Apache-Service.ps1 -Restart) and check the site.'

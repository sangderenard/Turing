<#
.SYNOPSIS  Prepare the agents' box: no account but the operator, the service and Windows itself may change the workspace,
and a standard (non-admin) Windows account "agents" that can read the code and nothing else. Run ELEVATED.

REPORT ONLY unless -Apply: every step is printed before it runs, and without -Apply nothing is changed.

What -Apply does, in this order:
  (a) reports who besides named accounts may touch C:\dev, C:\dev\Powershell and C:\Apache24
  (b) SAVES every folder permission of those trees to -BackupDir, with a manifest; Restore-AgentBoxAcls.ps1 puts them back
  (c) grants the service account Modify, explicitly, on the folders the stack writes that today it can only write through
      the broad "Authenticated Users: Modify" (every state_dir in instances.json, srv\backups, turing\build)
  (d) locks the trees: C:\dev and C:\Apache24 lose "Authenticated Users: Modify" (they keep "Users: Read", which the
      service needs to list the parents of every path it opens); C:\dev\Powershell loses both "Authenticated Users" and
      "Users". The operator keeps Full control; every explicit grant (the service's, codex's sandbox group) is kept.
  (e) with -CreateAgentAccount: the standard account (its password is asked for here and stored nowhere), member of Users
      only, with read on the code repositories of branch_policy.json and nothing on srv\state, the secrets or Apache
  (f) with -CopyAgentCredentials: copies the SpeakToMe sign-in files of -AgentLogins into that account's own profile,
      readable by it alone (it must have signed in once, so the profile exists). Their contents are never printed.
  (g) reports every listening port with its process and account, so each loopback service can be checked for its own
      authentication (any account on this computer reaches a loopback port)

Not here yet: the core service (MeowdokuCore) that will run git and the workspace commands for the agents. Until it is
installed, an agent signed in as "agents" can read the code but not change it, so keep the agents on the operator's
account until then. Undo: Restore-AgentBoxAcls.ps1 -Manifest <the manifest this run printed>.
.EXAMPLE   .\Set-AgentBox.ps1                                  # report only
.EXAMPLE   .\Set-AgentBox.ps1 -Apply -CreateAgentAccount        # lock, make the account
#>
param(
    [switch]$Apply,
    [switch]$CreateAgentAccount,
    [switch]$CopyAgentCredentials,
    [string]$AgentAccount = 'agents',
    [string[]]$AgentLogins = @('claude', 'codex', 'guest_agent'),
    [string]$ServiceAccount = 'svc-meowdoku',
    [string]$Operator = $env:USERNAME,
    [string]$BackupDir = 'C:\dev\acl-backups'
)
$ErrorActionPreference = 'Stop'
$Workspace = 'C:\dev\Powershell'
$Dev = 'C:\dev'
$Apache = 'C:\Apache24'
$Deploy = $PSScriptRoot
$Secrets = Join-Path $Workspace 'srv\state\secrets'
$AUTH_USERS, $USERS, $ADMINS, $SYSTEM = '*S-1-5-11', '*S-1-5-32-545', '*S-1-5-32-544', '*S-1-5-18'

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($Apply -and -not $elevated) { Write-Host 'REFUSED: -Apply needs an elevated PowerShell.'; exit 2 }
if (-not $Apply) { Write-Host '[report only] nothing is changed; run again with -Apply to perform the steps below.' }
function Step($t) { Write-Host ''; Write-Host "== $t" }
function Cmd($display) { Write-Host "cmd: $display" }
function Run([string]$exe, [string[]]$arguments) {
    Cmd "$exe $($arguments -join ' ')"
    if (-not $Apply) { return }
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe exited $LASTEXITCODE" }
}
function Broad([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return '(missing)' }
    $names = @{ 'S-1-5-11' = 'Authenticated Users'; 'S-1-5-32-545' = 'Users'; 'S-1-1-0' = 'Everyone' }
    $rows = (Get-Acl -LiteralPath $path).Access | ForEach-Object {
        try { $sid = $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value } catch { $sid = '' }
        $who = if ($names.ContainsKey($sid)) { $names[$sid] } elseif ($_.IdentityReference.Value -match 'Sandbox') { $_.IdentityReference.Value } else { $null }
        if ($who) { "$who=$($_.FileSystemRights -replace ', Synchronize','')$(if ($_.IsInherited) { ' (inherited)' })" }
    }
    if ($rows) { ($rows | Select-Object -Unique) -join '; ' } else { '(none)' }
}

# ---------------------------------------------------------------- (a)
Step '(a) who besides named accounts may touch the trees now'
foreach ($p in @($Dev, $Workspace, $Apache, (Join-Path $Workspace 'srv\python'), (Join-Path $Workspace 'srv\state'), (Join-Path $Workspace 'srv\state\sql'))) {
    Write-Host ("{0,-40} {1}" -f $p, (Broad $p))
}

# ---------------------------------------------------------------- (b)
Step "(b) save every folder permission to $BackupDir"
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$manifest = Join-Path $BackupDir "agent-box-$stamp.json"
$saves = @()
foreach ($root in @($Dev, $Workspace, $Apache)) {
    $leaf = Split-Path $root -Leaf
    $parent = Split-Path $root -Parent
    $rootFile = Join-Path $BackupDir "$leaf-root-$stamp.acl"
    $treeFile = Join-Path $BackupDir "$leaf-tree-$stamp.acl"
    $saves += [ordered]@{ root = $root; parent = $parent; root_file = $rootFile; tree_file = $(if ($root -eq $Dev) { $null } else { $treeFile }) }
    Cmd "icacls `"$root`" /save `"$rootFile`" /C /Q"
    if ($root -ne $Dev) { Cmd "icacls `"$root\*`" /save `"$treeFile`" /T /C /Q   (every file and folder below; this can take minutes)" }
}
if ($Apply) {
    New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
    foreach ($s in $saves) {
        & icacls.exe $s.root /save $s.root_file /C /Q | Out-Null
        if ($s.tree_file) { & icacls.exe "$($s.root)\*" /save $s.tree_file /T /C /Q | Out-Null }
    }
    $saves | ConvertTo-Json | Set-Content -LiteralPath $manifest -Encoding utf8
    Write-Host "saved; the manifest to restore from: $manifest"
}

# ---------------------------------------------------------------- (c)
Step "(c) explicit Modify for $ServiceAccount where the stack writes"
$instances = Get-Content -Raw -LiteralPath (Join-Path $Deploy 'instances.json') | ConvertFrom-Json
$writes = [System.Collections.Generic.List[string]]::new()
if ($instances.state_dir) { $writes.Add([string]$instances.state_dir) }
foreach ($i in $instances.instances) { if ($i.state_dir) { $writes.Add([string]$i.state_dir) } }
foreach ($extra in @('srv\backups', 'turing\build')) { $writes.Add((Join-Path $Workspace $extra)) }
foreach ($w in ($writes | ForEach-Object { $_.Replace('/', '\').TrimEnd('\') } | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $w) { Run 'icacls.exe' @($w, '/grant', "${ServiceAccount}:(OI)(CI)M") }
    else { Write-Host "skip (missing): $w" }
}

# ---------------------------------------------------------------- (d)
Step '(d) lock the trees (inherited entries are copied first, so every explicit grant stays)'
Run 'icacls.exe' @($Dev, '/inheritance:d')
Run 'icacls.exe' @($Dev, '/remove:g', $AUTH_USERS)
Run 'icacls.exe' @($Dev, '/grant', "${Operator}:(OI)(CI)F")
Run 'icacls.exe' @($Workspace, '/inheritance:d')
Run 'icacls.exe' @($Workspace, '/remove:g', $AUTH_USERS, $USERS)
Run 'icacls.exe' @($Workspace, '/grant', "${Operator}:(OI)(CI)F", "${ADMINS}:(OI)(CI)F", "${SYSTEM}:(OI)(CI)F")
Run 'icacls.exe' @($Apache, '/inheritance:d')
Run 'icacls.exe' @($Apache, '/remove:g', $AUTH_USERS)
Run 'icacls.exe' @($Apache, '/grant', "${Operator}:(OI)(CI)F")

# ---------------------------------------------------------------- (e)
Step "(e) the standard account $AgentAccount"
$policy = Get-Content -Raw -LiteralPath (Join-Path $Deploy 'branch_policy.json') | ConvertFrom-Json
$codeRepos = @($policy.repositories.PSObject.Properties | ForEach-Object { ([string]$_.Value.path).Replace('/', '\').TrimEnd('\') } |
    Where-Object { $_ -ne $Workspace })
if ($CreateAgentAccount) {
    $exists = Get-LocalUser -Name $AgentAccount -ErrorAction SilentlyContinue
    if ($exists) { Write-Host "account $AgentAccount exists: not made again" }
    else {
        Cmd "New-LocalUser $AgentAccount (password asked for now, stored nowhere); Add-LocalGroupMember Users"
        if ($Apply) {
            $password = Read-Host -AsSecureString "A password for $AgentAccount (you sign in with it; keep it in your password manager)"
            New-LocalUser -Name $AgentAccount -Password $password -PasswordNeverExpires -AccountNeverExpires `
                -Description 'The agents box: reads the code; writes only through the system' | Out-Null
            $password = $null
            Add-LocalGroupMember -Group (Get-LocalGroup -SID 'S-1-5-32-545') -Member $AgentAccount
        }
    }
    if ((Get-LocalGroupMember -Group (Get-LocalGroup -SID 'S-1-5-32-544') -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*\$AgentAccount" })) {
        throw "$AgentAccount is an administrator: the box needs a standard account. Remove it from Administrators first."
    }
    Run 'icacls.exe' @($Workspace, '/grant', "${AgentAccount}:RX")         # the folder itself, to reach the repositories
    foreach ($repo in $codeRepos) { if (Test-Path -LiteralPath $repo) { Run 'icacls.exe' @($repo, '/grant', "${AgentAccount}:(OI)(CI)RX") } }
} else { Write-Host 'skipped (no -CreateAgentAccount)' }

# ---------------------------------------------------------------- (f)
Step "(f) the agents' SpeakToMe sign-in files, in $AgentAccount's own profile"
if ($CopyAgentCredentials) {
    $profileDir = (Get-CimInstance Win32_UserProfile | Where-Object { $_.LocalPath -like "*\$AgentAccount" } | Select-Object -First 1).LocalPath
    if (-not $profileDir) { Write-Host "no profile for $AgentAccount yet: sign in as it once, then run this step again" }
    else {
        $target = Join-Path $profileDir 'speaktome-credentials'
        Cmd "New-Item $target; icacls $target /inheritance:r /grant ${AgentAccount}:(OI)(CI)R Administrators:F SYSTEM:F"
        if ($Apply) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            & icacls.exe $target /inheritance:r /grant "${AgentAccount}:(OI)(CI)R" "${ADMINS}:(OI)(CI)F" "${SYSTEM}:(OI)(CI)F" | Out-Null
        }
        foreach ($login in $AgentLogins) {
            $source = Join-Path $Secrets "operator_credentials\$login.secret"
            if (-not (Test-Path -LiteralPath $source)) { Write-Host "no sign-in file for $login"; continue }
            Cmd "copy $login.secret -> $target  (contents not shown)"
            if ($Apply) { Copy-Item -LiteralPath $source -Destination (Join-Path $target "$login.secret") -Force }
        }
    }
} else { Write-Host 'skipped (no -CopyAgentCredentials)' }

# ---------------------------------------------------------------- (g)
Step '(g) listening ports: every account here reaches the loopback ones, so each must check who is asking'
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Sort-Object LocalPort -Unique | ForEach-Object {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($_.OwningProcess)" -ErrorAction SilentlyContinue
    $owner = if ($proc) { (Invoke-CimMethod -InputObject $proc -MethodName GetOwner -ErrorAction SilentlyContinue).User } else { '' }
    "{0,-10} {1,-6} {2,-22} {3}" -f $_.LocalAddress, $_.LocalPort, $(if ($proc) { $proc.Name } else { '?' }), $owner
} | Write-Host

Step 'after (d), check the stack'
Write-Host "Restart it (Apache-Service.ps1 -Restart) and open the site. If anything refuses, undo with:"
Write-Host "  .\Restore-AgentBoxAcls.ps1 -Manifest $manifest"

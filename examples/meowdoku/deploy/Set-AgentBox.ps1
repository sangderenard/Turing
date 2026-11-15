<#
.SYNOPSIS  Prepare the agents' box: no account but the operator, the service and Windows itself may change the workspace,
and standard (non-admin) Windows accounts, one per agent, that can read the code and nothing else. Run ELEVATED.

REPORT ONLY unless -Apply: every step is printed before it runs, and without -Apply nothing is changed.

What -Apply does, in this order:
  (a) reports who besides named accounts may touch C:\dev, C:\dev\Powershell and C:\Apache24
  (b) SAVES every folder permission of those trees to -BackupDir, with a manifest; Restore-AgentBoxAcls.ps1 puts them back
  (c) grants the service account Modify, explicitly, on the folders the stack writes that today it can only write through
      the broad "Authenticated Users: Modify" (every state_dir in instances.json, srv\backups, turing\build), the .git
      folder of every repository in branch_policy.json (the lander fetches, pushes and updates refs there) and the
      removal notices file the cleaner commits
  (d) locks the trees: C:\dev and C:\Apache24 lose "Authenticated Users: Modify" (they keep "Users: Read", which the
      service needs to list the parents of every path it opens); C:\dev\Powershell loses both "Authenticated Users" and
      "Users". The operator keeps Full control; every explicit grant (the service's, codex's sandbox group) is kept.
  (e) with -CreateAgentAccount: one standard account -AgentAccount, member of Users only. Its password is generated here,
      never shown, and saved encrypted (Windows DPAPI) for the operator alone, so nobody types or pastes it and no agent
      holds it. It gets read on the code repositories of branch_policy.json, the service's Python (instances.json
      supervisor_python) and the workspace's tracked top-level files, and nothing on srv\state, the secrets or Apache.
      Its profile is made by starting one hidden process as it. It is refused while any folder on the PATH can be
      written by every account (an agent could plant a program that the operator or the service would then run).
      -ResetPassword gives an existing account a generated, saved password the same way.
  (f) with -AgentLogins <one login> (and -CreateAgentAccount or -CopyAgentCredentials): copies that one SpeakToMe sign-in
      file into the account's own profile, readable by it alone, and removes any other agent's sign-in file found
      there. One Windows account holds one agent's sign-in, so no agent can read another's. Contents are never printed.
  (g) reports every listening port with its process and account, so each loopback service can be checked for its own
      authentication (any account on this computer reaches a loopback port)

Once the trees are locked (C:\dev\Powershell has no "Authenticated Users" or "Users" entry), the account steps skip
(b) to (d): they only add, so no new backup is needed, and each prints its own undo.

-OpenConsole -AgentAccount <name> opens a PowerShell console as that account, in the workspace folder, with its saved
password (never shown, never on the clipboard) and the account's own environment ($HOME, TEMP, APPDATA and PATH are its
own, not the operator's). It changes nothing and needs no elevation.

An agent signed in as one of these accounts changes the code through the central checkout (checkout_* tools; the
service makes the branch and commits) and cannot run commands in it yet. Undo the lock: Restore-AgentBoxAcls.ps1
-Manifest <the manifest this run printed>.
.EXAMPLE   .\Set-AgentBox.ps1                                                                    # report only
.EXAMPLE   .\Set-AgentBox.ps1 -Apply -CreateAgentAccount -AgentAccount agents-codex -AgentLogins codex
.EXAMPLE   .\Set-AgentBox.ps1 -Apply -ResetPassword -AgentAccount agents                         # manage an existing one
.EXAMPLE   .\Set-AgentBox.ps1 -OpenConsole -AgentAccount agents-codex
#>
#Requires -Version 7
param(
    [switch]$Apply,
    [switch]$CreateAgentAccount,
    [switch]$CopyAgentCredentials,
    [switch]$ResetPassword,
    [switch]$OpenConsole,
    [string]$AgentAccount = 'agents',
    [string[]]$AgentLogins = @(),
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
$StoreDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'SpeakToMe\agent-accounts'
$AUTH_USERS, $USERS, $ADMINS, $SYSTEM = '*S-1-5-11', '*S-1-5-32-545', '*S-1-5-32-544', '*S-1-5-18'

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
# Folders on the machine's or the operator's PATH that every account may write: a program planted there by an agent
# would run as whoever next calls that command, the operator or the service included.
function Open-PathFolders {
    $dirs = @([Environment]::GetEnvironmentVariable('Path', 'Machine') -split ';') + @([Environment]::GetEnvironmentVariable('Path', 'User') -split ';')
    foreach ($d in ($dirs | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique)) {
        $w = (Get-Acl -LiteralPath $d).Access | Where-Object {
            $_.AccessControlType -eq 'Allow' -and ($_.FileSystemRights -match 'Modify|FullControl|Write|CreateFiles') -and
            $(try { $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -in @('S-1-5-11', 'S-1-5-32-545', 'S-1-1-0') } catch { $false }) }
        if ($w) { $d }
    }
}
# A generated password: 64 characters with every class Windows' complexity rule asks for. It exists only as a SecureString.
function New-AgentPassword {
    $sets = @('ABCDEFGHJKLMNPQRSTUVWXYZ', 'abcdefghijkmnopqrstuvwxyz', '23456789', '!#%+-.:=?@^_~')
    $all = -join $sets
    $rng = [System.Security.Cryptography.RandomNumberGenerator]
    $chars = [System.Collections.Generic.List[char]]::new()
    foreach ($s in $sets) { $chars.Add($s[$rng::GetInt32($s.Length)]) }
    while ($chars.Count -lt 64) { $chars.Add($all[$rng::GetInt32($all.Length)]) }
    for ($i = $chars.Count - 1; $i -gt 0; $i--) { $j = $rng::GetInt32($i + 1); $t = $chars[$i]; $chars[$i] = $chars[$j]; $chars[$j] = $t }
    $secure = [securestring]::new()
    foreach ($c in $chars) { $secure.AppendChar($c) }
    $chars.Clear()
    $secure.MakeReadOnly()
    $secure
}
function Credential-File([string]$name) { Join-Path $StoreDir "$name.xml" }
# Export-Clixml encrypts the password with the current user's DPAPI key: only the operator, signed in as themself, can read it.
function Save-AgentCredential([pscredential]$cred) {
    if (-not (Test-Path -LiteralPath $StoreDir)) {
        New-Item -ItemType Directory -Force -Path $StoreDir | Out-Null
        & icacls.exe $StoreDir /inheritance:r /grant "${Operator}:(OI)(CI)F" "${SYSTEM}:(OI)(CI)F" | Out-Null
    }
    $cred | Export-Clixml -LiteralPath (Credential-File $cred.UserName)
    Write-Host "  password saved encrypted for $Operator only: $(Credential-File $cred.UserName) (not shown)"
}
function Read-AgentCredential([string]$name) {
    $file = Credential-File $name
    if (Test-Path -LiteralPath $file) { Import-Clixml -LiteralPath $file } else { $null }
}
# Start a program as an agent account with that account's own environment. Start-Process -Credential copies the caller's
# environment (HOME, TEMP, APPDATA and PATH pointed at the operator's profile: the first -OpenConsole on 2026-11-14 tried
# to write its history and $HOME\gemini.key into C:\Users\alber and was refused), and -UseNewEnvironment gives only the
# machine scope (USERNAME reads SYSTEM). A ProcessStartInfo whose Environment is never touched passes no environment
# block, so CreateProcessWithLogonW builds one from the account's own profile; it also opens a new console by default.
function Start-AsAgent([pscredential]$cred, [string]$file, [string[]]$arguments, [string]$directory, [switch]$Hidden) {
    $info = [System.Diagnostics.ProcessStartInfo]::new($file)
    foreach ($a in $arguments) { $info.ArgumentList.Add($a) }
    $info.UserName = $cred.UserName
    $info.Domain = $env:COMPUTERNAME
    $info.Password = $cred.Password
    $info.LoadUserProfile = $true
    $info.UseShellExecute = $false
    $info.WorkingDirectory = $directory
    $info.CreateNoWindow = [bool]$Hidden
    [System.Diagnostics.Process]::Start($info)
}

# ---------------------------------------------------------------- console
if ($OpenConsole) {
    $cred = Read-AgentCredential $AgentAccount
    if (-not $cred) { Write-Host "REFUSED: no saved password for $AgentAccount; run -Apply -ResetPassword -AgentAccount $AgentAccount first."; exit 2 }
    Start-AsAgent $cred (Join-Path $PSHOME 'pwsh.exe') @() $Workspace | Out-Null
    Write-Host "opened a console as $AgentAccount (its password was not shown)"
    exit 0
}

$elevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if ($Apply -and -not $elevated) { Write-Host 'REFUSED: -Apply needs an elevated PowerShell.'; exit 2 }
if (-not $Apply) { Write-Host '[report only] nothing is changed; run again with -Apply to perform the steps below.' }

# ---------------------------------------------------------------- (a)
Step '(a) who besides named accounts may touch the trees now'
foreach ($p in @($Dev, $Workspace, $Apache, (Join-Path $Workspace 'srv\python'), (Join-Path $Workspace 'srv\state'), (Join-Path $Workspace 'srv\state\sql'))) {
    Write-Host ("{0,-40} {1}" -f $p, (Broad $p))
}

$policy = Get-Content -Raw -LiteralPath (Join-Path $Deploy 'branch_policy.json') | ConvertFrom-Json
$instances = Get-Content -Raw -LiteralPath (Join-Path $Deploy 'instances.json') | ConvertFrom-Json
$broadNow = @((Get-Acl -LiteralPath $Workspace).Access | Where-Object {
    try { $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -in @('S-1-5-11', 'S-1-5-32-545') } catch { $false } })
$skipLock = ($broadNow.Count -eq 0) -and ($CreateAgentAccount -or $CopyAgentCredentials -or $ResetPassword)
$manifest = $null
if ($skipLock) {
    Write-Host ''
    Write-Host "== (b) to (d) skipped: $Workspace is already locked, and the account steps only add (no new backup needed)"
} else {

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
$writes = [System.Collections.Generic.List[string]]::new()
if ($instances.state_dir) { $writes.Add([string]$instances.state_dir) }
foreach ($i in $instances.instances) { if ($i.state_dir) { $writes.Add([string]$i.state_dir) } }
foreach ($extra in @('srv\backups', 'turing\build')) { $writes.Add((Join-Path $Workspace $extra)) }
foreach ($w in ($writes | ForEach-Object { $_.Replace('/', '\').TrimEnd('\') } | Select-Object -Unique)) {
    if (Test-Path -LiteralPath $w) { Run 'icacls.exe' @($w, '/grant', "${ServiceAccount}:(OI)(CI)M") }
    else { Write-Host "skip (missing): $w" }
}
# The lander and the cleaner run inside the service's task worker: fetch, push, update-ref and branch in every repository of
# branch_policy.json (the live checkouts' .git), and the removal notices file is rewritten and committed in the workspace.
foreach ($r in $policy.repositories.PSObject.Properties) {
    $gitDir = Join-Path ([string]$r.Value.path).Replace('/', '\') '.git'
    if (Test-Path -LiteralPath $gitDir -PathType Container) { Run 'icacls.exe' @($gitDir, '/grant', "${ServiceAccount}:(OI)(CI)M") }
    else { Write-Host "skip (no .git folder): $gitDir" }
}
$notices = if ($policy.notices_file) { ([string]$policy.notices_file).Replace('/', '\') } else { Join-Path $Workspace 'REMOVAL_NOTICES.csv' }   # branch_gate.py's default
if (Test-Path -LiteralPath $notices -PathType Leaf) { Run 'icacls.exe' @($notices, '/grant', "${ServiceAccount}:M") }
else { Write-Host "skip (missing): $notices" }

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
}   # end of (b) to (d)

# ---------------------------------------------------------------- (e)
Step "(e) the standard account $AgentAccount"
$codeRepos = @($policy.repositories.PSObject.Properties | ForEach-Object { ([string]$_.Value.path).Replace('/', '\').TrimEnd('\') } |
    Where-Object { $_ -ne $Workspace })
$pythonDir = (Split-Path ([string]$instances.supervisor_python).Replace('/', '\') -Parent)
$cred = $null
if ($CreateAgentAccount -or $ResetPassword) {
    $open = @(Open-PathFolders)
    if ($open.Count) {
        Write-Host 'these folders on the PATH can be written by every account; an agent could plant a program there that you or the service would run:'
        $open | ForEach-Object { Write-Host "  $_" }
        Write-Host "close each the way six were closed on 2026-11-14: icacls <folder> /save <file> /C /Q; icacls <folder> /inheritance:d;"
        Write-Host "  icacls <folder> /grant ${Operator}:(OI)(CI)F; then remove the broad group's write entry (icacls <folder> /remove:g *S-1-5-11)"
        if ($Apply) { Write-Host "REFUSED: no agent account while the PATH is open."; exit 2 }
    }
    $exists = Get-LocalUser -Name $AgentAccount -ErrorAction SilentlyContinue
    if (-not $exists -and -not $CreateAgentAccount) { Write-Host "REFUSED: $AgentAccount does not exist; -CreateAgentAccount makes it."; exit 2 }
    if (-not $exists) {
        Cmd "New-LocalUser $AgentAccount with a generated password; Add-LocalGroupMember Users"
        if ($Apply) {
            $password = New-AgentPassword
            New-LocalUser -Name $AgentAccount -Password $password -PasswordNeverExpires -AccountNeverExpires `
                -Description 'Agents box: reads code; writes via the system' | Out-Null
            Add-LocalGroupMember -Group (Get-LocalGroup -SID 'S-1-5-32-545') -Member $AgentAccount
            $cred = [pscredential]::new($AgentAccount, $password)
            Save-AgentCredential $cred
        }
    } elseif ($ResetPassword) {
        Cmd "Set-LocalUser $AgentAccount with a new generated password (a scheduled task registered with the old one must be registered again)"
        if ($Apply) {
            $password = New-AgentPassword
            Set-LocalUser -Name $AgentAccount -Password $password
            $cred = [pscredential]::new($AgentAccount, $password)
            Save-AgentCredential $cred
        }
    } else {
        Write-Host "account $AgentAccount exists: not made again"
        if (-not (Test-Path -LiteralPath (Credential-File $AgentAccount))) { Write-Host "  no saved password for it: add -ResetPassword to give it a generated one, so -OpenConsole works" }
    }
    $password = $null
    if ((Get-LocalGroupMember -Group (Get-LocalGroup -SID 'S-1-5-32-544') -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*\$AgentAccount" })) {
        throw "$AgentAccount is an administrator: the box needs a standard account. Remove it from Administrators first."
    }
    Run 'icacls.exe' @($Workspace, '/grant', "${AgentAccount}:RX")         # the folder itself, to reach the repositories
    foreach ($repo in $codeRepos) { if (Test-Path -LiteralPath $repo) { Run 'icacls.exe' @($repo, '/grant', "${AgentAccount}:(OI)(CI)RX") } }
    if (Test-Path -LiteralPath $pythonDir) { Run 'icacls.exe' @($pythonDir, '/grant', "${AgentAccount}:(OI)(CI)RX") }   # runs the SpeakToMe MCP
    else { Write-Host "skip (missing): the service's Python $pythonDir" }
    # The workspace's tracked top-level files (AGENTS.md, README.md, the SPEAKTOME_* history ...) are published with the
    # repository; everything else at the top level stays closed.
    $topFiles = @(& git -C $Workspace ls-files 2>$null | Where-Object { $_ -and $_ -notmatch '/' } | ForEach-Object { Join-Path $Workspace $_ } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
    Cmd "icacls <each of the $($topFiles.Count) tracked top-level files of $Workspace> /grant ${AgentAccount}:R"
    if ($Apply) { foreach ($f in $topFiles) { & icacls.exe $f /grant "${AgentAccount}:R" | Out-Null; if ($LASTEXITCODE -ne 0) { throw "icacls $f exited $LASTEXITCODE" } } }
    $profileDir = (Get-CimInstance Win32_UserProfile | Where-Object { $_.LocalPath -like "*\$AgentAccount" } | Select-Object -First 1).LocalPath
    if (-not $profileDir) {
        Cmd "make $AgentAccount's profile: start one hidden process as it, with its profile loaded"
        if ($Apply) {
            if (-not $cred) { $cred = Read-AgentCredential $AgentAccount }
            if ($cred) {
                $made = Start-AsAgent $cred (Join-Path $env:SystemRoot 'System32\cmd.exe') @('/c', 'exit') (Join-Path $env:SystemRoot 'System32') -Hidden
                $made.WaitForExit()
            } else { Write-Host "  no saved password: add -ResetPassword, or sign in as $AgentAccount once" }
        }
    }
    Write-Host "undo (e): Remove-LocalUser $AgentAccount; Remove-Item $(Credential-File $AgentAccount); icacls <path> /remove:g $AgentAccount on"
    Write-Host "  $Workspace, $pythonDir, each of $($codeRepos -join ', ') and each tracked top-level file; its profile folder stays"
    Write-Host "  until removed in System Properties > User Profiles"
} else { Write-Host 'skipped (no -CreateAgentAccount or -ResetPassword)' }

# ---------------------------------------------------------------- (f)
Step "(f) the agent's SpeakToMe sign-in file, in $AgentAccount's own profile"
if ($CopyAgentCredentials -or ($CreateAgentAccount -and $AgentLogins.Count)) {
    if ($AgentLogins.Count -ne 1) { Write-Host "REFUSED: name exactly one SpeakToMe login with -AgentLogins; one Windows account holds one agent's sign-in, so no agent can read another's."; exit 2 }
    $login = $AgentLogins[0]
    $profileDir = (Get-CimInstance Win32_UserProfile | Where-Object { $_.LocalPath -like "*\$AgentAccount" } | Select-Object -First 1).LocalPath
    if (-not $profileDir) { Write-Host "no profile for $AgentAccount yet: run -CreateAgentAccount (or -ResetPassword) with -Apply first" }
    else {
        $target = Join-Path $profileDir 'speaktome-credentials'
        $source = Join-Path $Secrets "operator_credentials\$login.secret"
        Cmd "New-Item $target; icacls $target /inheritance:r /grant ${AgentAccount}:(OI)(CI)R Administrators:F SYSTEM:F"
        if ($Apply) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            & icacls.exe $target /inheritance:r /grant "${AgentAccount}:(OI)(CI)R" "${ADMINS}:(OI)(CI)F" "${SYSTEM}:(OI)(CI)F" | Out-Null
        }
        if (-not (Test-Path -LiteralPath $source)) { Write-Host "no sign-in file for $login" }
        else {
            Cmd "copy $login.secret -> $target  (contents not shown)"
            if ($Apply) { Copy-Item -LiteralPath $source -Destination (Join-Path $target "$login.secret") -Force }
        }
        foreach ($other in @(Get-ChildItem -LiteralPath $target -Filter '*.secret' -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne "$login.secret" })) {
            Cmd "remove $($other.Name) from $target (another agent's sign-in; the original stays in $Secrets)"
            if ($Apply) { Remove-Item -LiteralPath $other.FullName -Force }
        }
    }
} else { Write-Host 'skipped (no -AgentLogins with -CreateAgentAccount or -CopyAgentCredentials)' }

# ---------------------------------------------------------------- (g)
Step '(g) listening ports: every account here reaches the loopback ones, so each must check who is asking'
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Sort-Object LocalPort -Unique | ForEach-Object {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($_.OwningProcess)" -ErrorAction SilentlyContinue
    $owner = if ($proc) { (Invoke-CimMethod -InputObject $proc -MethodName GetOwner -ErrorAction SilentlyContinue).User } else { '' }
    "{0,-10} {1,-6} {2,-22} {3}" -f $_.LocalAddress, $_.LocalPort, $(if ($proc) { $proc.Name } else { '?' }), $owner
} | Write-Host

if (-not $skipLock) {
    Step 'after (d), check the stack'
    Write-Host "Restart it (Apache-Service.ps1 -Restart) and open the site. If anything refuses, undo with:"
    Write-Host "  .\Restore-AgentBoxAcls.ps1 -Manifest $manifest"
}

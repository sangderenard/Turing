<#
.SYNOPSIS  Create the local service account for Apache + Meowdoku + Nodus and install Apache as the Windows service Apache2.4 under it.
Run ELEVATED (exit 2 otherwise; -WhatIf may run unelevated). Every step is idempotent and printed before it runs. The password is
generated here, written ONLY to <SecretsDir>\<Name>.windows.secret (never overwritten, never echoed) and shown as <REDACTED> everywhere.
The service never reads the operator's profile (C:/Users/alber): it runs from the private interpreter C:/dev/Powershell/srv/python
(Provision-ServicePython.ps1) and uses no machine-wide resource. See SERVICE_ACCOUNT_2026-11-10.md.
#>
param(
    [string]$Name = 'svc-meowdoku',
    [string]$SecretsDir = 'C:/dev/Powershell/turing/build/meowdoku/operator_credentials',
    [string]$StateDir,   # the state root (examples/meowdoku/state_paths.py); empty (or $env:MEOWDOKU_STATE_DIR) = the old build/meowdoku layout
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'
$Repo      = 'C:/dev/Powershell/turing'
$Nodus     = 'C:/dev/Powershell/nodus'
$Apache    = 'C:/Apache24'
$Srv       = 'C:/dev/Powershell/srv'   # private interpreter tree (srv/python); no grant touches C:/Users/alber
if (-not $StateDir) { $StateDir = $env:MEOWDOKU_STATE_DIR }
if ($StateDir) {
    if (-not $PSBoundParameters.ContainsKey('SecretsDir')) { $SecretsDir = Join-Path $StateDir 'secrets/operator_credentials' }
    $StateBase = $StateDir; $SecretsBase = Join-Path $StateDir 'secrets'; $ModifyDirs = @($StateDir)   # Modify on the root; secrets/ is cut to the operator by operators.py / StatePaths.ensure()
} else {
    $StateBase = "$Repo/build/meowdoku"; $SecretsBase = $StateBase; $ModifyDirs = @("$Repo/build")
}
$Svc       = 'Apache2.4'
$SecretFile = Join-Path $SecretsDir "$Name.windows.secret"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    if ($WhatIf) { Write-Host '[WhatIf] not elevated: showing commands only (a real run exits 2 here).' }
    else { Write-Host 'REFUSED: run from an elevated PowerShell.'; exit 2 }
}

function Step($t) { Write-Host ''; Write-Host "== $t" }
function Cmd($display) { Write-Host "cmd: $display" }
# Run a native command: print it (password redacted), execute unless -WhatIf, fail on non-zero exit.
function Run([string]$exe, [string[]]$arguments, [string]$redact) {
    $shown = ($arguments | ForEach-Object { if ($redact -and $_ -eq $redact) { '<REDACTED>' } else { $_ } }) -join ' '
    Cmd "$exe $shown"
    if ($WhatIf) { return }
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed with exit $LASTEXITCODE" }
}

# ---------------------------------------------------------------- (a) user
Step "(a) local user $Name"
$existing = Get-LocalUser -Name $Name -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "user $Name already exists: skipping creation (password left alone)."
} else {
    if (Test-Path -LiteralPath $SecretFile) { throw "REFUSED: $SecretFile already exists but user $Name does not. Resolve by hand; never overwritten." }
    Cmd "generate 32 random bytes -> url-safe base64 password (not shown)"
    Cmd "write password ONLY to $SecretFile (new file)"
    Cmd "New-LocalUser -Name $Name -Password <REDACTED> -PasswordNeverExpires -UserMayNotChangePassword -Description 'Meowdoku/Nodus/Apache service account'"
    if (-not $WhatIf) {
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $plain = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        New-Item -ItemType Directory -Force -Path $SecretsDir | Out-Null
        $fs = [IO.File]::Open($SecretFile, [IO.FileMode]::CreateNew)   # CreateNew: refuses to overwrite
        $b = (New-Object Text.UTF8Encoding($false)).GetBytes($plain); $fs.Write($b, 0, $b.Length); $fs.Close()
        $sec = ConvertTo-SecureString $plain -AsPlainText -Force
        New-LocalUser -Name $Name -Password $sec -PasswordNeverExpires -UserMayNotChangePassword -Description 'Meowdoku/Nodus/Apache service account' | Out-Null
        $plain = $null
    }
    # the secret file sits under build/meowdoku, where the service gets Modify: cut inheritance so the service cannot read its own password file.
    Run 'icacls.exe' @($SecretFile, '/inheritance:r', '/grant:r', '*S-1-5-32-544:(F)', '*S-1-5-18:(F)', 'alber:(F)')
}

# ---------------------------------------------------------------- (b) logon rights
Step "(b) SeServiceLogonRight + SeDenyInteractiveLogonRight for $Name"
$sid = if (-not $WhatIf) { (Get-LocalUser -Name $Name).SID.Value } else { "<SID-of-$Name>" }
$tmp = Join-Path ([IO.Path]::GetTempPath()) "meowdoku-secedit-$PID"
$exportInf = "$tmp.export.inf"; $applyInf = "$tmp.apply.inf"; $sdb = "$tmp.sdb"
Run 'secedit.exe' @('/export', '/cfg', $exportInf, '/areas', 'USER_RIGHTS', '/quiet')
$cur = @{ SeServiceLogonRight = ''; SeDenyInteractiveLogonRight = '' }
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
SeServiceLogonRight = $(Merge $cur.SeServiceLogonRight $sid)
SeDenyInteractiveLogonRight = $(Merge $cur.SeDenyInteractiveLogonRight $sid)
"@
Write-Host "temporary inf ($applyInf); existing holders of both rights are preserved (WhatIf shows only the new SID):"
Write-Host $infText
if (-not $WhatIf) { Set-Content -LiteralPath $applyInf -Value $infText -Encoding Unicode }
Run 'secedit.exe' @('/configure', '/db', $sdb, '/cfg', $applyInf, '/areas', 'USER_RIGHTS', '/quiet')
if (-not $WhatIf) { Remove-Item -LiteralPath $exportInf, $applyInf, $sdb -Force -ErrorAction SilentlyContinue }

# ---------------------------------------------------------------- (c) ACLs
Step "(c) NTFS ACLs for $Name"
Write-Host 'Read+execute with inheritance (the whole tree must be loadable):'
foreach ($d in @($Apache, $Repo, $Nodus, $Srv)) { Run 'icacls.exe' @($d, '/grant', "${Name}:(OI)(CI)RX") }
# Git reads every repository the branch policy names (the reviews page's commits and diffs, core Git, the task
# worker's landing). A repository inside another one's tree is covered by that grant. The workspace root holds every
# other project in its working tree, so there the service reads only the folder itself and its .git, never the tree.
# git's safe.directory for these paths reaches each Git-running instance through its own environment
# (instances.json "env", GIT_CONFIG_* command-scope config), not through any machine-wide or profile config.
$Policy = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'branch_policy.json') | ConvertFrom-Json
$Workspace = (Resolve-Path (Join-Path $Repo '..')).Path.Replace('\', '/')
foreach ($p in $Policy.repositories.PSObject.Properties) {
    $path = ([string]$p.Value.path).Replace('\', '/').TrimEnd('/')
    if ($path -eq $Workspace) {
        Run 'icacls.exe' @($path, '/grant', "${Name}:RX")
        Run 'icacls.exe' @("$path/.git", '/grant', "${Name}:(OI)(CI)RX")
    } elseif (@($Apache, $Repo, $Nodus, $Srv) -notcontains $path) {
        Run 'icacls.exe' @($path, '/grant', "${Name}:(OI)(CI)RX")
    }
}
Write-Host 'Modify (logs, state, build output):'
foreach ($d in (@("$Apache/logs") + $ModifyDirs + @("$Nodus/out"))) { Run 'icacls.exe' @($d, '/grant', "${Name}:(OI)(CI)M") }
Write-Host 'Secrets the service must read (explicit file-level grants; build/ already gives Modify by inheritance):'
$tokens = @(Get-ChildItem -Path $SecretsBase -Filter 'admin_token*.txt' -ErrorAction SilentlyContinue)
if ($tokens.Count -eq 0) { Cmd "icacls $SecretsBase/admin_token.txt /grant ${Name}:(R)   (no admin_token*.txt present now)" }
foreach ($f in $tokens) { Run 'icacls.exe' @($f.FullName, '/grant', "${Name}:(R)") }
Run 'icacls.exe' @("$SecretsBase/tls/key.pem", '/grant', "${Name}:(R)")
# [MAIL] the mail receiver's public certificate and key (Copy-PublicCertificate.ps1 copies them in): read only, granted on
# the folder and inherited by its files, so the copy a renewal writes is readable too.
# [LANDING] the landing task's GitHub token (instances.json task-worker "env_files" GH_TOKEN): read only, that one file.
$landingToken = "$SecretsBase/github/landing.token"
if (Test-Path -LiteralPath $landingToken) { Run 'icacls.exe' @($landingToken, '/grant', "${Name}:(R)") }
else { Cmd "icacls $landingToken /grant ${Name}:(R)   (no token yet: write it there, then run this script again)" }
$tlsPublic = "$SecretsBase/tls-public"
if (Test-Path -LiteralPath $tlsPublic) { Run 'icacls.exe' @($tlsPublic, '/grant', "${Name}:(OI)R") }
else { Cmd "icacls $tlsPublic /grant ${Name}:(OI)R   (no tls-public folder yet: run Copy-PublicCertificate.ps1, then this script again)" }
# The secrets folder itself (state layout only; the old build/meowdoku layout inherits Modify). Apache resolves every folder of a
# configured path by name (FindFirstFile on the parent folder), so without the right to list secrets/ the service is refused
# tls-public/fullchain.cer although the file grants it read: httpd then exits at startup with "SSLCertificateFile: Invalid file
# path" (2026-11-13). RX on this folder only, no (OI)(CI): the service sees the names, and opens nothing below that is not granted
# above. Not (RD): without Synchronize Windows still refuses the listing. operators.py's /grant:r replaces only the operator's
# entry, so a later restriction of secrets/ keeps this one.
if ($StateDir) { Run 'icacls.exe' @($SecretsBase, '/grant:r', "${Name}:(RX)") }
Write-Host 'NOTE: key.pem keeps its existing entries (alber etc.); the service is only ADDED. Tighten by hand if desired.'
Write-Host "NOTE: other operators' *.secret files under operator_credentials inherit Modify from build/; consider cutting inheritance there."

# ---------------------------------------------------------------- (d) service
Step "(d) Apache service $Svc"
$svcExists = Get-Service -Name $Svc -ErrorAction SilentlyContinue
if ($svcExists) { Write-Host "service $Svc already exists: install skipped, configuration re-applied." }
else {
    # httpd -k install creates the service, then runs a config test. That test exits non-zero while the console-mode Apache still
    # holds :80/:443, although the service was created (2026-11-13). Judge by the service, not by the exit code.
    Cmd "$Apache/bin/httpd.exe -k install -n $Svc"
    if (-not $WhatIf) {
        & "$Apache/bin/httpd.exe" -k install -n $Svc
        $installExit = $LASTEXITCODE
        if (-not (Get-Service -Name $Svc -ErrorAction SilentlyContinue)) { throw "httpd -k install did not create $Svc (exit $installExit)" }
        if ($installExit -ne 0) { Write-Host "NOTE: httpd -k install exited $installExit, but $Svc exists (its config test fails while the console Apache holds the ports); continuing." }
    }
}
$pwArg = '<from-secret-file>'
if (-not $WhatIf) { $pwArg = (Get-Content -LiteralPath $SecretFile -Raw).Trim() }
Run 'sc.exe' @('config', $Svc, 'obj=', ".\$Name", 'password=', $pwArg) $pwArg
Run 'sc.exe' @('config', $Svc, 'start=', 'auto')
Run 'sc.exe' @('failure', $Svc, 'reset=', '86400', 'actions=', 'restart/5000')
$pwArg = $null
# Startup errors. Until httpd opens logs/error.log it can only write to stderr, and a service has none: a config error then shows
# only as "service-specific error: Incorrect function" (2026-11-13). httpd -E <file> sends those errors to a file; the service
# passes ConfigArgs to httpd -k runservice. Apache-Service.ps1 prints the file when a start fails.
$StartupLog = "$Apache/logs/startup-errors.log"
$Params = "HKLM:\SYSTEM\CurrentControlSet\Services\$Svc\Parameters"
Cmd "Set-ItemProperty $Params ConfigArgs = -E $StartupLog  (REG_MULTI_SZ)"
if (-not $WhatIf) {
    if (-not (Test-Path -LiteralPath $Params)) { New-Item -Path $Params -Force | Out-Null }
    Set-ItemProperty -LiteralPath $Params -Name ConfigArgs -Value ([string[]]@('-E', $StartupLog)) -Type MultiString
}
# The front's configuration is the repository's file itself (APACHE_FRONT_2026-11-10.md step 2), never a copy under conf/: a copy
# drifts (2026-11-14: conf/meowdoku-proxy.conf lacked the site-nav filter and held a hand edit the repository had since taken).
# Any other Include of a meowdoku-proxy.conf is replaced by this one; httpd.conf is backed up first and checked after (httpd -t).
$ProxyConf = "$Repo/examples/meowdoku/deploy/apache/meowdoku-proxy.conf"
$HttpdConf = "$Apache/conf/httpd.conf"
$wantInclude = "Include `"$ProxyConf`""
Cmd "httpd.conf: $wantInclude  (replacing any other Include of meowdoku-proxy.conf)"
if (-not $WhatIf) {
    $lines = Get-Content -LiteralPath $HttpdConf
    $found = @($lines | Where-Object { $_ -match '^\s*Include\s+"?[^"]*meowdoku-proxy\.conf"?\s*$' })
    if ($found.Count -eq 1 -and $found[0].Trim() -eq $wantInclude) { Write-Host 'httpd.conf already includes the repository conf.' }
    else {
        $backup = "$HttpdConf.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
        Copy-Item -LiteralPath $HttpdConf -Destination $backup
        $kept = @($lines | Where-Object { $_ -notmatch '^\s*Include\s+"?[^"]*meowdoku-proxy\.conf"?\s*$' }) + $wantInclude
        Set-Content -LiteralPath $HttpdConf -Value $kept -Encoding ascii
        Write-Host "httpd.conf now includes the repository conf (was: $($found -join ' | ')); backup $backup"
    }
    & "$Apache/bin/httpd.exe" -t
    if ($LASTEXITCODE -ne 0) { throw "httpd -t refused the configuration (exit $LASTEXITCODE); httpd.conf backup is beside it" }
}

# ---------------------------------------------------------------- (d2) pre-push hooks
# The gate's pre-push hook runs the interpreter that installed it (branch_gate.py hook_text: sys.executable). Installed from the
# operator's own Python it points into C:/Users/alber, which the service cannot run, and every push the landing task makes fails
# ("cannot execute: Is a directory", 2026-11-14). Installed from the private interpreter, the operator and the service both run it.
Step '(d2) pre-push hooks, installed from the private interpreter'
$Gate = "$Repo/examples/meowdoku/branch_gate.py"
$Policy = "$Repo/examples/meowdoku/deploy/branch_policy.json"
$AccountsDb = if ($StateDir) { Join-Path $StateDir 'sql/accounts.db' } else { "$StateBase/accounts.db" }
Run "$Srv/python/python.exe" @($Gate, '--policy', $Policy, '--db', $AccountsDb, 'install-hook')

# ---------------------------------------------------------------- (e) verify
Step '(e) verification'
Cmd "sc.exe qc $Svc"
if (-not $WhatIf) { sc.exe qc $Svc }
Cmd "groups of ${Name}: Get-LocalGroup | where member"
if (-not $WhatIf) { Get-LocalGroup | Where-Object { Get-LocalGroupMember -Group $_ -Member $Name -ErrorAction SilentlyContinue } | Select-Object -ExpandProperty Name }
foreach ($d in @($Apache, "$Apache/logs", $Repo, "$Repo/build", $Nodus, "$Nodus/out", $Srv, $SecretsBase, "$SecretsBase/tls/key.pem", "$SecretsBase/tls-public", "$SecretsBase/tls-public/cert.key", $SecretFile)) {
    Cmd "icacls $d  (lines matching $Name)"
    if (-not $WhatIf -and (Test-Path -LiteralPath $d)) { icacls.exe $d | Select-String -SimpleMatch $Name }
}
Cmd "ConfigArgs of $Svc (expect -E $StartupLog)"
if (-not $WhatIf) { (Get-ItemProperty -LiteralPath $Params -Name ConfigArgs -ErrorAction SilentlyContinue).ConfigArgs -join ' ' }
Write-Host ''
if ($WhatIf) { Write-Host '[WhatIf] nothing run.' }
else { Write-Host 'Done. Next: stop the console-mode Apache (Stop-Apache.ps1), then Apache-Service.ps1 -Start.' }

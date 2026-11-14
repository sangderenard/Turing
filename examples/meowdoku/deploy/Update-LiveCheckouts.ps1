<#
.SYNOPSIS  After the landing task lands reviewed changes, move each live checkout to what landed: a fast-forward only.
The repositories are branch_policy.json's (the ones the gate lands into). A checkout moves only when it is on its default branch,
has no tracked changes, and its commit is part of the remote's default branch. Anything else is reported and left as it is:
  ahead     the checkout has commits the remote lacks (made in the live tree, never reviewed): put them up for review first
  diverged  both have commits of their own: review the live tree merged with the remote, then run this again
  blocked   untracked files there would be overwritten. With -MoveIdenticalUntracked, files whose content equals what landed
            (line endings aside) are moved to <state>/backups/untracked-<repo>-<time>/ and the fast-forward is tried again.
Nothing is pushed, reset, merged or deleted. A restart (Apache-Service.ps1 -Restart) is what loads the moved code.
The site's pages are a published copy (web_export.py), not the source tree: a restart does not refresh them. When turing or
spectral-analyzer moved, or with -ExportWeb, the pages are exported again from the live turing checkout (web_export.py
--skip-boards, so the board database is kept) into each web folder the servers serve: build\meowdoku\web and every --web-dir
in instances.json. -ExportWeb republishes what the live checkout holds now, even what is not on the remote.
.EXAMPLE   $env:MEOWDOKU_OPERATOR='<operator>'; $env:MEOWDOKU_STATE_DIR='C:\dev\Powershell\srv\state'; .\Update-LiveCheckouts.ps1 -WhatIf
.EXAMPLE   .\Update-LiveCheckouts.ps1 -ExportWeb      # move what landed, and publish the pages even if nothing moved
#>
param([switch]$WhatIf, [switch]$MoveIdenticalUntracked, [switch]$ExportWeb, [string]$User = $env:MEOWDOKU_OPERATOR, [string]$PasswordFile,
      [string]$AccountsDb, [string]$StateDir)

# ---- Operator gate (authz.py script-auth): only accounts in script_access with deploy_scripts may run this script. Not skippable; -WhatIf is gated too.
$__repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $StateDir) { $StateDir = $env:MEOWDOKU_STATE_DIR }   # empty = the old build\meowdoku layout
if (-not $User) { throw 'REFUSED no operator: pass -User <name> (or set MEOWDOKU_OPERATOR) and -PasswordFile <path>.' }
if (-not $AccountsDb) { $AccountsDb = if ($StateDir) { Join-Path $StateDir 'sql\accounts.db' } else { Join-Path $__repo 'build\meowdoku\accounts.db' } }
if (-not $PasswordFile) {
    $__default = if ($StateDir) { Join-Path $StateDir "secrets\operator_credentials\$User.secret" } else { Join-Path $__repo "build\meowdoku\operator_credentials\$User.secret" }
    if (Test-Path -LiteralPath $__default) { $PasswordFile = $__default }
}
$__args = @((Join-Path $__repo 'examples\meowdoku\authz.py'), 'script-auth', $MyInvocation.MyCommand.Name, '--user', $User, '--db', $AccountsDb)
if ($PasswordFile) { $__args += @('--password-file', $PasswordFile) }
$__eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$__gate = & 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe' @__args 2>&1
$__code = $LASTEXITCODE
$ErrorActionPreference = $__eap
if ($__code -ne 0) { throw (($__gate | ForEach-Object { "$_" }) -join ' ') }
Write-Host ($__gate -join ' ')
# ---- end gate

$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'      # a sign-in prompt nobody sees must fail, not wait (the landing task hung on one, 2026-11-14)
$env:GCM_INTERACTIVE = 'never'
$policy = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'branch_policy.json') -Raw | ConvertFrom-Json
$remote = [string]$policy.remote
$backupRoot = if ($StateDir) { Join-Path $StateDir 'backups' } else { Join-Path $__repo 'build\meowdoku\backups' }

function G([string]$path, [string[]]$arguments) {       # git in one checkout; output lines, exit code in $script:gitExit
    $out = & git -C $path @arguments 2>&1
    $script:gitExit = $LASTEXITCODE
    return , [string[]]@($out | ForEach-Object { "$_" })      # always an array: one line must not become a string ([0] = a char)
}
function Short([string]$id) { if ($id.Length -ge 8) { $id.Substring(0, 8) } else { $id } }

$rows = @()
foreach ($property in $policy.repositories.PSObject.Properties) {
    $name = $property.Name; $path = [string]$property.Value.path; $base = [string]$property.Value.default
    $row = [ordered]@{ Repository = $name; State = ''; Detail = '' }
    $rows += $row
    if (-not (Test-Path -LiteralPath (Join-Path $path '.git'))) { $row.State = 'skipped'; $row.Detail = 'no checkout'; continue }
    $branch = (G $path @('rev-parse', '--abbrev-ref', 'HEAD'))[0]
    if ($branch -ne $base) { $row.State = 'skipped'; $row.Detail = "on $branch, not $base"; continue }
    $tracked = G $path @('status', '--porcelain', '--untracked-files=no')
    if ($tracked.Count) { $row.State = 'skipped'; $row.Detail = "$($tracked.Count) tracked change(s)"; continue }
    Write-Host "cmd: git -C $path fetch --quiet $remote $base"
    $null = G $path @('fetch', '--quiet', $remote, $base)
    if ($gitExit -ne 0) { $row.State = 'failed'; $row.Detail = "fetch failed"; continue }
    $here = (G $path @('rev-parse', 'HEAD'))[0]; $there = (G $path @('rev-parse', "$remote/$base"))[0]
    if ($here -eq $there) { $row.State = 'current'; $row.Detail = "at $(Short $here)"; continue }
    $null = G $path @('merge-base', '--is-ancestor', $here, $there); $behind = $gitExit -eq 0
    $null = G $path @('merge-base', '--is-ancestor', $there, $here); $ahead = $gitExit -eq 0
    if ($ahead) {
        $own = G $path @('log', '--format=%h %s', "$remote/$base..HEAD")
        $row.State = 'ahead'; $row.Detail = "$($own.Count) commit(s) not on ${remote}: $($own -join '; ')"; continue
    }
    if (-not $behind) {
        $mine = (G $path @('rev-list', '--count', "$remote/$base..HEAD"))[0]; $theirs = (G $path @('rev-list', '--count', "HEAD..$remote/$base"))[0]
        $row.State = 'diverged'; $row.Detail = "$mine local, $theirs on $remote"; continue
    }
    if ($WhatIf) { $row.State = 'would move'; $row.Detail = "$(Short $here) -> $(Short $there)"; continue }
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        Write-Host "cmd: git -C $path merge --ff-only $remote/$base"
        $said = G $path @('merge', '--ff-only', '--quiet', "$remote/$base")
        if ($gitExit -eq 0) { $row.State = 'moved'; $row.Detail = "$(Short $here) -> $(Short $there)"; break }
        $blocking = @($said | Where-Object { $_ -match '^\s+\S' -and $_ -notmatch '^\s*(Please|Aborting|hint:)' } | ForEach-Object { $_.Trim() })
        $row.State = 'blocked'; $row.Detail = if ($blocking.Count) { "untracked files in the way: $($blocking -join ', ')" } else { ($said -join ' ') }
        if (-not $blocking.Count -or -not $MoveIdenticalUntracked -or $attempt -eq 2) { break }
        $different = @($blocking | Where-Object {
            $file = Join-Path $path $_
            $landed = (G $path @('rev-parse', "${remote}/${base}:$_"))[0]
            -not (Test-Path -LiteralPath $file -PathType Leaf) -or (G $path @('hash-object', '--', $_))[0] -ne $landed })
        if ($different.Count) { $row.Detail += "; not moved, these differ from what landed: $($different -join ', ')"; break }
        $to = Join-Path $backupRoot "untracked-$name-$(Get-Date -Format yyyyMMdd-HHmmss)"
        foreach ($relative in $blocking) {
            $target = Join-Path $to $relative
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
            Move-Item -LiteralPath (Join-Path $path $relative) -Destination $target
        }
        Write-Host "moved $($blocking.Count) untracked file(s) equal to what landed to $to"
    }
}
$rows | ForEach-Object { [pscustomobject]$_ } | Format-Table -AutoSize -Wrap
if ($rows | Where-Object { $_.State -eq 'moved' }) { Write-Host 'Moved checkouts take effect at the next restart (Apache-Service.ps1 -Restart).' }

# The published pages. web_export.py copies the pages from turing and the shaders and wave arena from spectral-analyzer, so a
# move of either makes the served copy stale until it is exported again.
$pagesMoved = @($rows | Where-Object { $_.State -in @('moved', 'would move') -and $_.Repository -in @('turing', 'spectral-analyzer') })
if ($ExportWeb -or $pagesMoved.Count) {
    # Pages are published only from reviewed code: both source checkouts must be on their default branch.
    $offBase = @(foreach ($source in @('turing', 'spectral-analyzer')) {
        $entry = $policy.repositories.$source
        $on = (G ([string]$entry.path) @('rev-parse', '--abbrev-ref', 'HEAD'))[0]
        if ($on -ne [string]$entry.default) { "$source is on $on, not $($entry.default)" }
    })
    if ($offBase.Count) { throw "REFUSED to publish the pages: $($offBase -join '; '). Put the checkout back on its default branch first." }
    $turing = [string]$policy.repositories.turing.path
    $exporter = Join-Path $turing 'examples\meowdoku\web_export.py'
    $instances = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'instances.json') -Raw | ConvertFrom-Json
    $webDirs = @((Join-Path $turing 'build\meowdoku\web'))      # web_export.py's default, which a server without --web-dir serves
    foreach ($instance in $instances.instances) {
        $arguments = @($instance.args)
        $at = [array]::IndexOf($arguments, '--web-dir')
        if ($at -ge 0 -and $at + 1 -lt $arguments.Count) { $webDirs += [string]$arguments[$at + 1] }
    }
    $webDirs = @($webDirs | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') } | Sort-Object -Unique)
    $why = if ($pagesMoved.Count) { "$(($pagesMoved | ForEach-Object { $_.Repository }) -join ' and ') moved" } else { '-ExportWeb' }
    Write-Host ''
    Write-Host "Publishing the pages ($why) into $($webDirs.Count) web folder(s)."
    $python = 'C:\Users\alber\AppData\Local\Programs\Python\Python311\python.exe'   # the compiler's Python (web_export compiles a shader)
    $failed = @()
    foreach ($dir in $webDirs) {
        if (-not (Test-Path -LiteralPath (Join-Path $dir 'boards.json'))) {
            Write-Host "skipped $dir : no boards.json there (not a published web folder; export it once by hand)"; continue
        }
        Write-Host "cmd: $python $exporter --skip-boards --out $dir"
        if ($WhatIf) { continue }
        $env:PYTHONUTF8 = '1'
        & $python $exporter --skip-boards --out $dir
        if ($LASTEXITCODE -ne 0) { $failed += $dir; Write-Warning "web_export.py exited $LASTEXITCODE for $dir" }
    }
    if ($failed.Count) { throw "The pages were not published into: $($failed -join ', ')" }
    if (-not $WhatIf) { Write-Host 'Pages published; a page reload shows them (no restart needed for pages).' }
}

<#
.SYNOPSIS
  Make "review-gate" a required status check of a repository's default branch on GitHub, or remove that rule.

.DESCRIPTION
  A GitHub account setting, changed only by the owner's decision: run it yourself, or tell an agent to in so many
  words. It uses the GitHub CLI's own sign-in (gh auth status); no token is read, printed or stored here.

  With the rule on, GitHub admits a commit as the tip of the branch only when the review-gate check has passed on
  that commit (deploy/review-gate.yml asks the service's SQL). Administrators are included, force pushes and
  deletion of the branch are refused. Put review-gate.yml on the default branch BEFORE turning the rule on, or
  nothing can land.

.EXAMPLE
  ./Protect-DefaultBranch.ps1 -Repository sangderenard/Turing -Branch main
  ./Protect-DefaultBranch.ps1 -Repository sangderenard/Turing -Branch main -Remove
#>
param(
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$')][string]$Repository,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9._/-]+$')][string]$Branch,
    [switch]$Remove
)
$ErrorActionPreference = 'Stop'
$target = "repos/$Repository/branches/$Branch/protection"
if ($Remove) {
    gh api -X DELETE $target
    if ($LASTEXITCODE -ne 0) { throw "GitHub refused to remove the rule on $Repository $Branch" }
    Write-Host "$Repository ${Branch}: the rule is removed"
    return
}
$rule = @{
    required_status_checks        = @{ strict = $false; contexts = @('review-gate') }
    enforce_admins                = $true
    required_pull_request_reviews = $null
    restrictions                  = $null
    allow_force_pushes            = $false
    allow_deletions               = $false
} | ConvertTo-Json -Depth 5
$rule | gh api -X PUT $target --input - | Out-Null
if ($LASTEXITCODE -ne 0) { throw "GitHub refused the rule on $Repository $Branch" }
Write-Host "$Repository ${Branch}: review-gate is required, administrators included"

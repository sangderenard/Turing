# Service account: what changed on 2026-11-12

The owner decided (2026-11-12) to run Apache as the Windows service `Apache2.4` under `svc-meowdoku`, managed from the
machine: whoever has the machine may stop or start the stack (`Apache-Service.ps1`), because a web service should not
hide where it runs. The setup is still the owner's elevated run of `Setup-ServiceAccount.ps1 -StateDir C:/dev/Powershell/srv/state`
(dry run: add `-WhatIf`).

Git under the service account, prepared on our side:

- **Read access.** `Setup-ServiceAccount.ps1` now grants read to every repository `branch_policy.json` names. For the
  workspace root (`C:/dev/Powershell`), whose working tree holds every other project, it grants only the folder itself and
  its `.git`.
- **safe.directory.** Git refuses repositories owned by another user. The Git-running instances (`meowdoku-a`,
  `meowdoku-dev`, `task-worker`) carry `GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_n` / `GIT_CONFIG_VALUE_n` in their
  `instances.json` "env": command-scope config, which Git honours for `safe.directory`, with no machine-wide or profile config.

Still the owner's:

- **A GitHub credential for the service.** Landing pushes to GitHub and now opens a pull request per landed review
  (`branch_gate.record_pull_request`). Today `git` and `gh` use the owner's own login. Under `svc-meowdoku` they need the
  service's own narrow credential (a deploy key or a fine-grained token limited to these repositories). Until then a
  landing still succeeds locally only if the push is possible; a pull request that GitHub refuses is recorded as
  "no pull request (…)" and never holds the landing.

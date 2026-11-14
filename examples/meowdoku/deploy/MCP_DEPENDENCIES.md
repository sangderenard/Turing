# Additive outgoing MCP dependency installation

Use the existing private interpreter provisioner, not a replacement service runtime:

```powershell
./Provision-ServicePython.ps1 -Root C:/dev/Powershell/srv -Requirements ./requirements-mcp-runtime.txt -InstallRequirementsOnly -WhatIf
```

Remove `-WhatIf` only in a coordinated maintenance window with no concurrent
package installations. This is a deployment operation, not an ordinary UI login.
The helper checks the existing dependency inventory, constrains every installed
version, resolves only wheels from PyPI, refuses replacement plans, installs only
resolved missing distributions with dependency processing disabled, and checks
both the old inventory and dependency consistency afterward. The interpreter's
hash must remain unchanged. On failure it uninstalls newly added distributions;
a rollback inventory mismatch requires operator review. This is not an atomic
filesystem transaction; do not run it concurrently with service jobs or installers.

`requirements-mcp-runtime.txt` is an SDK dependency set, not the complete service.
It was provisioned in an isolated Windows Python3.11.18 candidate using the
existing archive-verified provisioner; pip check and nine outgoing MCP fixture
tests passed. Additive installer tests cover refusal of replacement/direct artifact
plans, strict requirement syntax, preservation, and partial-install rollback.
The already-satisfied real candidate path also passed. No Suno authorization or
production-service health claim follows from these tests.

## Mandatory maintenance prerequisites

Before a live additive install, stop every consumer of this interpreter through
the coordinated service lifecycle and ensure no other package installer is active.
Take a complete recoverable copy of the existing private interpreter directory,
including Lib/site-packages and Scripts, outside the repository. Verify the backup
is readable and retain it until service health is confirmed. The helper's rollback
is a best-effort dependency rollback, NOT a substitute for that filesystem backup:
a package wheel can share files with an existing distribution even when their
names differ. If installation or rollback fails, keep consumers stopped and restore
the verified complete backup before resuming. Do not overlay a backup onto a
partially changed runtime and assume removed files are gone.

The installer disables inherited pip configuration (PIP_CONFIG_FILE points at the
platform null device), strips PIP_/PYTHON* environment variables and runs Python
isolated. Resolution uses PyPI only. Every planned wheel must have an HTTPS
files.pythonhosted.org URL and SHA256 digest. Installation uses those exact URLs
and required hashes with no index/dependency resolution; it does not resolve a
second potentially different package set.

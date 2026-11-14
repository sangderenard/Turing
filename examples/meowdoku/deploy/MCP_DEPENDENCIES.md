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

# 2026-11-13 AUTHORED PAGES INTO LIVE-SOURCE BRANCH

Isolated integration branch based on `codex/wave-arena-studio-menu` b841373b
and Turing `origin/main` 53f6dd3c. The single merge conflict in
`examples/meowdoku/web_export.py` retained both the existing Wave Arena copy
and the new direct-game export. This is a source merge, not a service receipt.

With `PYTHONPATH=examples` and `SPEAKTOME_REPO=C:\dev\Powershell\speaktome`,
the selected authored-page, route-switch, public-login and access suites passed:
57 tests in 101.01 seconds. Python compilation, Node syntax and staged diff
whitespace checks passed. The initial collection without `SPEAKTOME_REPO`
stopped because this isolated worktree has no sibling SpeakToMe checkout.

Local listener inspection found no process on stable port 8766. Port 8767 is
an older toy development process. The SpeakToMe messaging connector timed out,
so operator coordination and live exposure checks could not be completed.
No managed restart, web export, browser verification, or public-route check was
performed. The configured full web exporter also requires the Spectral sibling
checkout, absent from this isolated worktree.

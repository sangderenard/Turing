# 2026-11-14 Suno connection host boundary

Base aa5e3c62. Seven focused unittest tests pass for the real route dispatcher,
using injected service responses (not live provider claims). py_compile server
and remote_connections pass; git diff --check passes. No SDK installed into live
service, no live config/restart, no external network requests. Dependency/SDK
integration is separately verified by SpeakToMe remote_mcp tests.

Deployment must apply query-free Apache logging before OAuth callbacks; source
configuration changed only. Original combined logs include secrets in queries.
Existing default balancer stickiness must retain the authorization worker.
Full repository suite not run (documented nonterminating suite).

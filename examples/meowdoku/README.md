# Meowdoku

This folder contains the Meowdoku server, games, Marble Studio, and related
rendering tools. The live web directory is `build/meowdoku/web`, relative to the
Turing repository root; editing source does not by itself publish those assets.

## Start and restart

Follow [HOW_TO_START.md](HOW_TO_START.md) for the Windows startup command,
required inbound TCP 8766 firewall rule, session-preserving restart, and checks.
The firewall rule requires Administrator PowerShell to create. Starting Python
alone does not establish public access through Windows and the routers.

[AGENTS.md](AGENTS.md) records the operational constraints for agents.
[STUDIO_HIDDEN_START_HANDOFF_2026-11-09.md](STUDIO_HIDDEN_START_HANDOFF_2026-11-09.md)
is about staged native renderer work, not the HTTP server startup procedure.

# Meowdoku agent instructions

Read the repository AGENTS.md and AGENT_OBSTINANCE.md first.

## Starting or restarting the live server

Read [HOW_TO_START.md](HOW_TO_START.md) before touching the running server.
A listening Python process is not sufficient proof of public availability.
The Windows inbound TCP 8766 allow rule must be in place. Creating it requires
an Administrator PowerShell and authorization for the firewall change; do not
silently change firewall, router, TLS, or network settings.

Use the existing authenticated lifecycle save/exit path, check active exposures
before restarting, and preserve the recorded argv, cwd, environment, session
files, TLS configuration, and retained fountain receipt. Never kill a server
just to free its port or start a duplicate. Coordinate with other active workers.
Keep the process host alive and verify the replacement survives the launching
command. A returned PID alone is not proof of a running service.

Verify local HTTPS and the actual public route separately. Distinguish TCP
connectivity, TLS identity/trust, HTTP health, and restored sessions. Do not
bypass certificate verification to claim success. TcpClient.Connect and
Get-NetTCPConnection are checks, not commands that enable forwarding.

Preserve shared edits and distinguish source changes from published assets in
build/meowdoku/web. A restart loads current backend source, including other
workers' edits; coordinate that scope before restarting.

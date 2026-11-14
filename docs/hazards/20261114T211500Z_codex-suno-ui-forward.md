# 2026-11-14 CONNECTED SERVICES REAL SUNO HANDOFF UI

Base aa5e3c62. Connected services now uses session-owned backend connection
routes for Suno status, connect, disconnect/cancel, tool discovery, and explicit
tool calls. Provider authorization is an HTTPS link from backend discovery,
not a fabricated client setup. Pending status polls; callback returns here.
Existing Alpaca Keys controls remain unchanged.

Measured: 2 focused unittest checks pass; Node syntax and behavior tests pass.
Covered guest/account reset, repeated clicks, pending authorization, nested
remote schemas retained, explicit JSON call arguments, remote tool errors,
invalid authorization URL rejection, logout/stale response isolation and offline.
Provider descriptions/results are textContent, never executable HTML.
No browser tokens or persisted authorization URLs. No automatic tool call.

Limits: no Suno live connection, browser visual acceptance, site deployment or
restart performed. Requires sibling backend routes and configured outbound MCP
adapter. These are UI contract tests, not an end-to-end provider auth receipt.

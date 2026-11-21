# Guest MCP session quota — 2026-11-16 07:13 UTC

Branch `codex/guest-mcp-session-quota-20261116` starts from Turing main `227666c3`.

ChatGPT created and connected the SpeakToMe guest plugin through an outbound OpenAI Secure MCP Tunnel. Its first chat read reached the server, but MCP initialize received HTTP 429 `guest session limit reached`. Plugin discovery and connection had already used the prior five-session hourly per-IP allowance. This branch raises the existing per-IP hourly cap to thirty while retaining the preview identity, permissions, and publication review flow.

Focused test: `SPEAKTOME_REPO=C:/dev/Powershell/speaktome python -m pytest -q tests/test_meowdoku_oauth_mcp.py` passed (1 test, 9.91 seconds) in the server checkout. It checks that the thirtieth guest session succeeds and the thirty-first returns 429. An initial collection without SPEAKTOME_REPO failed because the sibling SpeakToMe REST dependency was missing; setting it resolved that failure. Full compiler baseline was not rerun. `python tools/hazards_log.py` currently rejects an existing malformed historical hazard filename, so the newest receipt was read directly before this test.

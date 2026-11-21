# MCP session capacity integration — 2026-11-16 20:35 UTC

Base: remote Turing main `ee7c73f465b51f2de022332b0d9642f0f9570c04`. The live `codex/fix-mcp-internal-https` branch has two commits beyond its old base. Its `oauth_mcp.py` session capacity change is retained here; its older `server.py` would remove the later site-condition controls, so that file is deliberately not merged.

MCP sessions now expire after ten idle minutes; the host caps all live sessions at 128 and guest sessions at 96. A full host answers 503 with Retry-After instead of clearing unrelated sessions. The old per-IP hourly creation budget is removed so several connectors behind one address can initialize. Existing guest preview identity and permissions remain the same.

Focused test: `SPEAKTOME_REPO=C:/dev/Powershell/speaktome python -m pytest -q tests/test_meowdoku_oauth_mcp.py` passed (1 test, 5.30 seconds) in the review checkout on Windows. It exercised multiple guest initializations through the TLS listener and the existing guest API. The full compiler baseline was not rerun. The latest hazards receipt before this entry was `20261116T071305Z_codex-guest-mcp-session-quota-20261116.md`.

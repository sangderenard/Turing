2026-11-10 SQL-LEASED LOCALHOST STUDIO/GAME DEV INSTANCES (Windows, shared dirty checkout)

Extended the existing tools/serve_studio_game_layer.py read-only overlay server.
Atomic SQLite identity/name/port lease plus an exclusively bound loopback socket;
heartbeat, generation-fenced release, collision refusal and expired-name recovery.
Default state is ~/.turing-dev/sql/instances.sqlite, outside repo/live state.
No production DB, Apache/Nodus configuration, firewall or live assets changed.
Backend APIs are deliberately unavailable; this is browser/assets QA, not the
full Meowdoku backend. Run command and limitations: docs/studio_dev_instances.md.

python -m pytest -q tests/test_studio_dev_server.py: 6 passed in 1.60s.
Covers simultaneous same-name claim, foreign-port collision without eviction,
expired lease with live listener refusal, dead-listener reclaim, identity-fenced
release, read-only status, real HTTP overlay/fallback and traversal/API/write denial.
python -m py_compile tools/serve_studio_game_layer.py passed.

Started managed session 98499, PID 51316, studio-game on 127.0.0.1:8900 against
build/dot-studio-hud-20261110. Independent HTTP request returned 200, SQL heartbeat
advanced after launcher returned. Windows Get-NetTCPConnection CIM access was
unavailable; no OS listener-table assertion is claimed. Bound address is fixed
in source and asserted by socket tests. Full repository suite not run.

# Localhost Studio/game development instances

Use the existing read-only overlay server independently of Apache and Nodus:

    python -u tools/serve_studio_game_layer.py --name studio-game --stage build/dot-studio-hud-20261110

The first stdout line gives the actual URL, PID and SQL identity. The default
range is 8900..8999; `--port 8910` requests one exact port. There is deliberately
no bind-address option: the listener is always 127.0.0.1. No firewall, public
route, service account, Apache config, or production database is changed.

The default database is `~/.turing-dev/sql/instances.sqlite`, outside all repo
worktrees. This user-home path is intentional: Windows tool sandboxes can
redirect LOCALAPPDATA to a per-session directory. All workers must use the same
default (or the same explicit `--db`) to coordinate. `--status` reads the lease
rows without starting a server or creating a missing database.

`--stage` overrides files in `--assets` (default `build/meowdoku/web`). Both
roots are read-only. This serves the browser Studio/game and compiled assets;
backend APIs, uploads, jobs, login and persistence are NOT available. It is not
a replacement for the full Meowdoku backend or its state-preserving lifecycle.
No rebuild or live asset publication occurs here. Existing HTTP QA imports
using Handler, STAGE and LIVE remain supported.

The lease transaction holds the actual exclusive bound socket before committing
its unique name/port row. Instance identity is a fresh generation UUID, not a
PID inferred from some other service. Heartbeats renew a 15-second lease every
5 seconds. Normal shutdown closes the socket and releases only that identity.
If the process is forcibly stopped, rerun after expiry to reclaim its name.
An expired name whose old port still listens is refused, never killed or
replaced. Port collisions are skipped, never evicted. SQL failure or ownership
loss stops this server rather than silently continuing without a lease.

Keep the launching managed execution session alive. The server is independent
of Apache's process tree but cannot survive termination of its own process or
host. It is respawnable with the same command; no auto-restart service or
scheduled task is installed. Stop only your own managed execution session.

Verification: `python -m pytest -q tests/test_studio_dev_server.py`.

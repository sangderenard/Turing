# 2026-11-15 REVIEW LANDING TIMING

The review policy's default silence window is 30 minutes per submitted commit. The deployed cron template scans every five minutes. A scheduled or requested applying run reserves its start in `branch_land_runs` under an immediate SQLite transaction and defers another applying run for five minutes. One run still handles all eligible reviews and fetches each repository once. Explicit CLI runs remain available to the operator.

Measured on `codex/review-landing-timing-20261115` from `ce5f6c42`: `python -m pytest tests/test_meowdoku_branch_gate.py -q` passed 46 tests in 146.93 seconds. The added check covers separate expiration times and a requested run arriving one second after a scheduled run. The installed service cron table and running task worker were not changed by this review branch; the live schedule still needs installation after review.

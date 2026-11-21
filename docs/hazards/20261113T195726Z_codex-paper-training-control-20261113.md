Paper Market Watch admin training control

Added Start/Stop training shortcut and pending-request cancel/status to the actual markets page.
Reuses existing /api/control/instances authenticated supervisor API for nodus-training only;
no new runner or backend control path. UI requires admin role and account_manage. Backend
continues its existing account_manage/lifecycle_admin authorization. Shutdown keeps after_jobs
policy, permits pending cancellation for this trainer only, never cancels whole-stack requests,
and reports supervisor state instead of claiming accepted requests completed. Uncertain POSTs
are never automatically retried. Trainer signal handler saves weights; supervisor can force-stop
after its existing timeout. No live training was stopped/started.

Verification: node --check markets.js; git diff --check; existing
python -m pytest tests/test_meowdoku_instance_control.py tests/test_meowdoku_restart_after_jobs.py -q
16 passed in 59.58s. These verify existing supervisor behavior, not a live browser click.
Live UI/API action not exercised because it would alter shared training. No publication/restart.

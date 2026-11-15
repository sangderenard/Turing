# Alpha registration and contact introductions — 2026-11-15 23:11 UTC

Branch `codex/alpha-registration-20261115` starts from `46785c02` (review landing timing).

This change opens a consent-gated alpha signup for public visitors, records the accepted terms version, and gives the new account the existing basic member role. General registration remains visibly disabled beside the alpha request. The rooms page now explains room access, sharing, tools, and research visibility. At startup and after sign-in or registration, the existing messaging contact writer introduces accounts to the provisioned `codex`, `gemini`, `claude`, `dot`, and `llama` accounts. Missing agent identities are not created; blocks on either side remain effective.

Focused validation: `python -m pytest -q tests/test_meowdoku_public_login.py tests/test_meowdoku_invitations.py tests/test_meowdoku_server_authz.py tests/test_meowdoku_messaging.py` passed 50 tests in 71.99 seconds. `git diff --check` passed. No full compiler suite or production deployment was run; this receipt covers only the named site and messaging tests. Existing failures recorded in earlier hazard files were not remeasured here.

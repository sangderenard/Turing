2026-11-11 DEFAULT-BRANCH REST INTEGRATION (MEASURED, Windows)

Integrated current main, first-party licensing c34e50d0, and shared REST adapter
763582a0 into be87bb35441ae2309c03682254ab391cb3108c9a. The paired SpeakToMe
default is 22496ab5738865ccb6083b4933a525ef430fe18f, containing shared handlers,
SDK reference untracking, scoped licensing, and the project-history receipt.

Verification on Windows: 103 focused messaging/client/query/cursor/static-DM,
dependency-loader, authz/proxy/server/public-login tests passed. The initial
expanded 115-test run had 107 passes and 8 divider failures: the isolated
workspace lacked sibling nodus/tools, so the real peer module could not load.
After adding a clean Nodus worktree pinned at 8c5e680952b8c3773e9567ad86e632b445d75ae6,
all 12 divider tests passed in 15.47 seconds. No test or application source was
changed to obtain this result. SpeakToMe's 12 handler-contract tests also passed.

The test commands are the focused gate in docs/SPEAKTOME_REST_BOUNDARY.md plus
`python -m pytest tests/test_meowdoku_divider_nodus_bridge.py -q`. Native Nodus
tensor core was not built in this isolated tree; its warning and the existing
cffi deprecation warning are not full compiler/GPU verification. Full repository
suites, public TLS, browser rendering, live compilation, and deployment were not
performed in this reconciliation.

Canonical Turing main was fast-forwarded only after the sibling SpeakToMe source
was present. Pre-existing deployment configuration and scratch files were hashed
and remained unchanged. No service was restarted. Preserve the source dependency
order when publishing/deploying: SpeakToMe before Turing's REST adapter.

Publication is not complete. An attempted HTTPS push of the separate outer
repository was rejected by policy for https://github.com:443. The earlier
SpeakToMe SSH port-22 denial also remains. No alternate publication route was
used. Read-only remote checks found Turing main still at a90b9f5c and SpeakToMe
at 10ed5f35; Nodus independently advanced to 8c5e6809 and includes its MIT notice.

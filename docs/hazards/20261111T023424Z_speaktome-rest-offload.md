2026-11-11 SHARED REST HANDLERS EXTRACTED TO SPEAKTOME (MEASURED)

Turing baseline fd9a9903; SpeakToMe baseline 7362174. Four existing HTTP
handler bodies move to speaktome_api/routes.py. Turing supplies its original
request handler and backend instances, retaining all account/schema/world and
compiler state. Dependency selected by SPEAKTOME_REPO or sibling speaktome.
Missing package now intentionally refuses server startup. Deploy SpeakToMe first
and keep both revision receipts. This is not a second HTTP service.

Matching SpeakToMe local commit: 91add50378fae73baeefe4505a57007fc9a9b0a3.

Before extraction: 45 tests in messaging, Living Data query/cursors/static DM
passed in 63.79s. After: 103 passed in 69.43s across those suites, messaging
clients, dependency loader, authz, proxy trust, server authz and public login.
One existing cffi imp deprecation warning. SpeakToMe's 12 new handler-contract
unit tests passed. Independent AST review confirms behavior-equivalent moved
bodies after dependency substitutions. No compiler source changed.

Full suite, public TLS, live rollout and browser rendering not run. Canonical
source, concurrent work, services, live DBs and security configuration untouched.
Work remains in isolated paired branches. GitHub SSH inspection was blocked by
Permission denied on port 22; no remote publication is asserted and no alternate
transport was attempted after denial. See docs/SPEAKTOME_REST_BOUNDARY.md.

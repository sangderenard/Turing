# Messenger REST relationship handles

Reused the existing Meowdoku accounts, contacts, messaging storage and HTTP
handler. Added HTTPS-only, sender-bound 60-second send handles obtained from an
authenticated conversation picker, digest-only server storage, and transactional
idempotency. No recipient name is required in the new send payload. Legacy
name-based sends remain compatible. Existing session tokens can also be passed
as Bearer only on messaging routes over secure transport; this does not issue
new account credentials or grant capabilities.

CLI and widget default to handles. Failed-response retry keeps its operation
key. Browser Back/Close/reopen shares in-flight state and cannot silently replace
an uncertain operation with changed content. Both clients refuse redirects;
CLI refuses HTTP before login. Server-readable HTTPS messaging is explicitly
not end-to-end encrypted or claimed post-quantum secure.

Security review corrections: sample the expiry clock after the SQLite write
lock; revoke handles transactionally when contact relationships change rather
than relying only on wall-clock stamps; retain a retry key after an uncertain
send even if the next attempt definitively fails; validate response receipts. Final client review also added HTTPException/IncompleteRead normalization, malformed contact-picker response checks, and recovery from definite rejections without losing earlier uncertain-send keys.
Deterministic tests cover lock-delayed expiry and same-timestamp relationship
recreation, plus concurrent idempotent sends, revocation, suspension, auth,
private token storage, and client retry/reopen cases.

Validation: 55 passed in messaging, authz, proxy trust, server authz and public
login suites (34.12 seconds, one existing cffi imp deprecation warning). 20
client regression tests passed (0.32 seconds), including offline Node tests.
Python compilation, Node syntax, git diff --check passed. Full repository suite
not run. HTTP fixtures simulate the configured trusted TLS-termination path;
no real TLS handshake or public route is claimed by this run. No live database,
server process, firewall, trust setting, credential issuance or deployment
was changed. Additive messaging_rest schema v1 is created when updated source
opens a database; coordinate rollout of all processes sharing accounts.db.

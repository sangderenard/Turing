2026-11-10 WORKSPACE ACCOUNTING / LOCAL REVIEW FOUNDATION (MEASURED, offline Windows).

Added isolated workspace_accounting.py and tests/test_workspace_accounting.py.
15 focused unittest cases passed in 1.620s. No compiler imports, large builds,
server restart, endpoints, credentials, live database access, GitHub interaction,
commit or push. Source remains uncommitted alongside other workers' changes.

Atomic SQL reservations use existing users/account_roles identity, owner-scoped
access, live admin rechecks, volume/user budgets and measured-free headroom.
Immutable review receipts do not apply merges. Releasing remains charged until
a trusted removal receipt; independent review caught non-idempotent release
retries and the fix is covered. Worker-only measurements and removal receipts
must not be exposed to owners through a generic HTTP dispatcher.

Initial test fixture cleanup failed on Windows because sqlite3 connection
contexts do not close handles; test fixtures now explicitly close and rerun
passed. No production Store leak was found. The repository hazards log helper
refused the existing malformed 20261109T_material-library-paused-checkpoint.md;
valid newest receipts and baseline were read directly without changing it.

See docs/WORKSPACE_ACCOUNTING_REVIEW_FOUNDATION_2026-11-10.md for integration
limits: state-root/schema_version registration, authenticated routes, real
snapshot measurements, retention/export/apply and OS execution isolation remain
unimplemented. This is not an end-to-end user-to-user safety or deployment pass.

Continuation: added Accounts-session request-facing operations and StatePaths
binding without touching shared auth/server/state source. 22 focused tests
passed in 1.968s against disposable databases, including actual Accounts logout,
expiry and account/database identity. Public session surface cannot invoke
measurement, snapshot evidence or cleanup confirmation. No HTTP routes are
installed. Production schema/version and sandbox gates still apply.

Bounded payload continuation: workspace_review_snapshots stores explicit input
file bytes atomically with ticket and logical encoded-byte accounting. Input cap
1 MiB / 128 files; portable path validation; owner/workspace/base-bound digest;
owner/admin reads verify integrity, with admin revocation rechecked. No host
path reads, mutable-tree walk, archive extraction or HTTP activation.
Independent review caught DB/root filesystem mismatch bypassing correct-volume
headroom. Store now refuses it; regression covered. 30 focused tests pass in
1.768s. SQLite physical page/index/WAL overhead remains outside logical bundle
accounting and is an explicit production admission gate, not claimed solved.

Operator-guide review repairs: shared schema_version.declare integration now
runs before workspace DDL/business writes, reacquiring the transaction after an
initial declaration commit. Tests refuse newer schemas before writes and inject
failure to prove payload/ticket/usage rollback. Explicit closed-review snapshot
retirement preserves ticket audit metadata and enables final lease release.
External cleanup uses CAS plus absolute post-removal measured usage, not a delta;
regression closes independent review's double-subtract race. Owner-scoped size
profiles preserve unknown build peak NULL distinct from zero.
37 workspace tests passed (3.163s) and existing 3 schema-version tests passed
(0.212s) using the private service interpreter. No live DB/stack, shared auth,
state-root or federation edits, ACLs, credentials, build, commit or push.

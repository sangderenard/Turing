# Workspace accounting and local review foundation

> Current integration status, 2026-11-10: the shared schema_version.declare gate and actual StatePaths/Accounts binding are now implemented. Closed-review retention has explicit audited retirement, so leases are no longer permanently blocked by historical ticket rows. Earlier statements below about those missing features describe prior checkpoints, not current code. Nothing is deployed or pointed at live state.

Offline-only addition, 2026-11-10. Module: examples/meowdoku/workspace_accounting.py.

## Authority and scope

Use the existing Accounts database and users.id / account_roles authority. The module is not imported by server.py and does not touch live state unless explicitly initialized by a future integration. It does not create users, roles, credentials, folders, worktrees or snapshots; execute code; apply merges; change ACLs; or contact GitHub. Diagnostic tickets.py is unchanged.

Three announced tables: workspace_volumes, workspace_leases, workspace_review_tickets. The state-root lane owns schema_version: integrate this component with that lane's registration contract before production initialization. Policy/root binding is immutable for now; no implicit policy migration.

## Accounting

A reservation is a total lifetime byte commitment, not additional bytes beyond measured usage. Charge max(reserved_bytes, measured_bytes), and subtract only max(reserved_bytes - measured_bytes, 0) from current filesystem free space when considering future growth. Apply per-user quota, per-volume capacity and a free-space floor atomically with BEGIN IMMEDIATE. Disk sampling cannot fence unrelated processes: operating-system enforcement/admission coordination is still needed before execution is exposed.

One managed root per filesystem per Accounts database prevents accidentally splitting a volume's allowance. Other Accounts databases can still have independent ledgers: production must designate one host-volume accounting authority rather than initialize several.

Trusted reconciliation reports actual overruns rather than clipping them. User-content, private dependencies/build/runtime, retained submission snapshots and logs must all be included. Shared baselines and shared caches need separate once-only accounting before end-to-end capacity claims. No physical allocation scanner or snapshot materializer is supplied by this slice.

## User-to-user boundaries

Every owner operation requires an existing account and filters by owner_id and volume_id. Another user's object and an absent object yield the same NotFound error. Admin-only review_queue/decide recheck the SQL admin role for every transaction, including after revocation; LAN origin and operator/service role do not bypass it.

actor_id comes from server-authenticated Accounts state, never a request parameter. record_usage, submit_review snapshot fields, and confirm_removed are trusted worker inputs, not user-supplied evidence. Do NOT expose a generic method dispatcher. The future owner submit endpoint must ask the worker to seal/measure the owner's workspace and supply that receipt itself. No worker method should be reachable just because a caller owns the workspace.

A user-controlled Python process must not receive this database, server credentials, unrestricted host paths or this Store object. A function-level owner check is not OS/process isolation. Execution/export/HTTP integration remains blocked until its sandbox and negative tests establish that separation.

## Review and release

Tickets retain immutable base_revision, snapshot_digest, snapshot_bytes and owner. Submission must match a trusted sealed workspace revision and generation. Identical retry returns the same ticket; changed size is refused. Later working revisions leave prior tickets unchanged. Every retained snapshot remains charged and prevents workspace release until a future explicit retention/disposal workflow.

An administrator decides approved/rejected for an exact snapshot. This does not mean applied. Future application must recheck target/base revision, validate a scoped content manifest and commit an idempotent application receipt. File traversal, symlink/hardlink escapes, archive filtering, privilege-separated untrusted validation and export without host secrets are intentionally not implemented by this SQL slice.

Release is releasing until a trusted cleanup worker confirms absence/no writers with the current generation. Repeated begin_release is idempotent and returns that generation. No automatic expiry frees unverified bytes.

## Verification

python tests/test_workspace_accounting.py -v: 15 tests passed on Windows, 1.620 seconds. Includes owner/ticket isolation, unknown actors, missing/revoked admin roles, invalid sizes/digests, user/volume quota, physical headroom without double charging, truthful overrun, concurrent admission, stale revisions, immutable/idempotent tickets and retained bytes, idempotent release, and policy/root mismatch refusal.

Initial test run found Windows fixture cleanup failures because sqlite3's connection context manager commits but does not close. Test-only connections now close explicitly; rerun passed. Production Store connections already close in finally.

Independent static review found one release retry issue; fixed and regression-tested. No compiler/build, HTTP, production database migration, full repository suite, snapshot export, actual merge or sandbox test was run. Existing hazards_log.py refuses an unrelated malformed hazard filename; newest valid receipts and baseline were read directly, without renaming foreign files.

## Accounts and StatePaths integration continuation

Added SessionAccess, which resolves each supplied session token through the existing Accounts.user_for_token instead of accepting actor IDs or roles. Every method reauthenticates; logout/expiry are observed on the next call. It exposes owner reservation/read/release-request and ticket read/admin decision only, not record_usage, confirm_removed, or submit_review. Those remain trusted worker calls. This is a Python service boundary, not protection against arbitrary code running in the server interpreter.

for_state binds the same Accounts database to StatePaths.content_dir/workspaces and refuses mismatched databases, absent roots and redirected roots. It never invokes StatePaths.ensure, changes ACLs or creates a directory. The existing state-root owner must provision the root and register schema_version. No server route was added while server.py/authz.py/federation_peer.py contain the other lane's edits.

Updated gate: 22 focused tests passed in 1.968s, including actual Accounts and StatePaths (disposable fixtures with synthetic session rows), logged-out/expired/unknown sessions, database mismatch, cross-user ownership, admin revocation, refusal of caller actor_id injection and no worker-receipt dispatcher. No passwords or production credentials were created. Current inspected source HEAD c60704b7; unrelated changes remain untouched.

Production remains gated: HTTP method/CSRF/body limits/rate limits; schema compatibility; snapshot materialization/retention/application; hard isolation for user execution; shared baseline/cache and SQL/WAL overhead accounting. The ledger's byte reservations are not an OS disk quota. No GitHub, deployment, live request or real database operation was performed.

Final independent review found no blocker in the session adapter. Hardened its database binding further: PRAGMA database_list on the existing Accounts connection now supplies the database filename, so changing the object's path label cannot disguise another open DB. Mismatched-connection regression included; all 22 tests pass again. Database files must remain stable for object lifetimes; state migration must rebuild Accounts and Store together. An operation authenticated immediately before a concurrent logout may finish; subsequent operations reauthenticate and refuse.

## Bounded explicit-byte review bundles

The foundation now implements seal_and_submit: up to 128 explicit files and 1 MiB total input content are canonicalized into an owner/workspace/base-bound review payload. Names are relative portable ASCII paths; traversal, absolute/drive/ADS names, separators/backslashes, .git metadata, Windows reserved names, case collisions and file/directory collisions are refused. No filesystem path is opened and no archive is extracted. This is a bounded submitted bundle, not a complete independently runnable system snapshot.

The workspace_review_snapshots table retains encoded payload bytes. In one SQL transaction, admission checks budgets/headroom, writes the immutable payload and review ticket, and increases measured usage by the encoded payload length. Exact retry is idempotent only if the exact bytes are retained. The owner's read_snapshot and current-admin reviewer_snapshot verify the stored digest before returning bytes; other users cannot read. No snapshot methods were added to SessionAccess or HTTP.

Independent review found that a database on a different filesystem could evade the root's headroom check. Store now refuses database/root device mismatch. A focused regression simulates the alternate filesystem without any cross-drive writes. Logical bundle-byte charging excludes SQLite pages, indexes and journal/WAL overhead; the free-space floor is not a hard OS quota or a proof of peak physical allocation. Shared database overhead and host-volume admission remain deployment gates.

Final focused test run: 30 tests passed in 1.768s. New tests cover immutable payload bytes, idempotence, ownership, administrator revocation, base/owner/workspace identity, rejected paths/collisions, input bounds, corrupted stored data, stale generation, disk pressure, quota rollback and database-volume mismatch. No live database, user filesystem tree, server, network or security configuration was touched.

## Operator-guide review repairs

Read deploy/OPERATOR_GUIDE_2026-11-10.md at current main and Speaktome review 1794338100. No live instance/database access. Our for_state adapter already consumed actual StatePaths; it remains unchanged and covered by integration tests. Shared state_paths.py, schema_version.py, authz.py, server.py and federation files are untouched.

Every connection now calls the shared schema_version.declare for workspace_accounting v1 before DDL or business writes. Because initial declaration may commit, it reacquires BEGIN IMMEDIATE and rechecks the version before proceeding. A newer schema refuses before any workspace DDL or row writes. All subsequent operation writes remain atomic; failure-injection tests prove no retained blob/usage residue if ticket or retirement audit insertion fails. Six owned tables are now announced: workspace_volumes, workspace_leases, workspace_review_tickets, workspace_review_snapshots, workspace_snapshot_retirements, workspace_size_profiles.

Closed reviewed snapshots may be explicitly retired by an administrator. SQL payload retirement atomically removes payload bytes, writes an immutable retirement receipt, updates logical usage and invalidates stale generation. Ticket/decision/digest/size metadata remain for audit. No open review can retire. All unretired reviews still block release; once each is retired, begin_release and trusted confirm_removed complete normally. Payload retirement is not merge application and does not mean database pages physically shrink. Real disposal requires explicit retention authorization; no such production action occurred.

Metadata-only external snapshots need a distinct trusted cleanup receipt. The worker supplies an absolute measured total AFTER verified removal, plus the generation sampled before that measurement. It never subtracts the payload size from a possibly already-reconciled total. A stale measurement refuses; resample and retry. The remaining retained-snapshot floor still applies. Independent review found the earlier double-subtract race; the regression now covers reconciliation occurring between deletion and receipt.

workspace_size_profiles preserves build_peak_bytes=NULL for unknown, separately from measured zero. Profiles are owner scoped and do not create leases or admit builds. Explicit reservations still require a known positive budget; no unknown is coerced to zero.

Final private service interpreter gates: 37 workspace tests passed in 3.163 seconds, and the existing 3 shared schema-version tests passed in 0.212 seconds. No native/compiler builds, HTTP, live DB, ACL, credentials, commit, push, or deployment. Future production work still needs physical database/WAL overhead budgeting, stable state-root lifecycle, authenticated HTTP controls and process/filesystem isolation for user execution.

# State root, leased worktrees, federated Nodus (2026-11-10)

Design only. Read-only analysis; nothing was changed, started or probed. Paths are relative to `examples/meowdoku/` unless prefixed `build/` (= `C:/dev/Powershell/turing/build/meowdoku`) or `nodus/`. The SQL/blob inventory and the cloud options (Litestream, libSQL, Postgres, bucket assets) are in `CLOUD_HOSTING_2026-11-10.md`; this document does not repeat them, it decides where the files live and how instances share them.

Request (operator): a place for user data outside any repo tracking that can later be detached from this server, with logs and the SQL; so worktrees of the project source can be leased out, versions can evolve and coordinate on different machines, over a network connection that lets Nodus spawn FIFO ports across the internet between instances that promise the same API.

## 1. Today's state map

One fact explains most of it: **state is not configured, it is derived from `web_dir.parent`**. `--web-dir` defaults to `ROOT/build/meowdoku/web` (server.py:84, :1913) and every store is `web_dir.parent / <name>`. `lifecycle.STATE_DIR` is `ROOT/build/meowdoku` (lifecycle.py:39) and the supervisor's `STATE` is the same (apache_supervisor.py:46). `ROOT` is `HERE.parents[1]`, i.e. the worktree. So state follows the checkout.

| Group | Path (under `build/meowdoku/`) | Writer / reader | Source |
|---|---|---|---|
| user content | `marble_objects/<digest>.json` | marble_objects.Store | server.py:353 |
| user content | `marble_photos/<digest>/{config,status,images,hdr,orbit.avi}` | marble_photo.PhotoJobs | server.py:354 |
| user content | `obj_assets/` | obj_assets.Store (needs accounts) | server.py:367 |
| user content | `shares/`, `tickets/` | ShareStore, tickets.Store | server.py:345, :357 |
| user content | `profiles.json` (+ `profiles.dev.json`) | profiles.Profiles | server.py:86, :1914, :1959 |
| user content | `loading_anim.json`, `.mla`, `tipjar.json` | anim_mod.Store, tipjar | server.py:352, :174 |
| SQL | `accounts.db` (+`-wal`, `-shm`): users, sessions, owned_assets (accounts.py:74-93); account_roles, script_access, script_audit (authz.py:97-113); lowering_cores/lowerings/lowering_blobs/lowering_leases (lowering_library.py:34-45); service_instances, instance_leases, account_instances (apache_supervisor.py:55-60; server.py:576) | accounts, authz, lowering_library, supervisor, server | server.py:362-363 |
| SQL | `marble_photos/job-library.sqlite` (`job_owners`) | job_library.py:11-14 | |
| policy | `access.json`, `authz_policy.json` (beside accounts.db) | access.py, authz.Policy.beside | server.py:360, :364 |
| logs | `log_ring.jsonl` (size-capped ring), `logs/apache-access.log`, `logs/apache-supervisor.log`, `logs/<instance>-<stamp>.log`, ad-hoc `live-*.std*.log`, `dev_*_std*.log` | logring, supervisor | server.py:1958; apache_supervisor.py:254-258, :420, :443 |
| secrets | `admin_token[_<port>].txt` | lifecycle (never printed) | lifecycle.py:65 |
| secrets | `operator_credentials/<user>.secret`, `<instance>.console.secret` (per run), `svc-meowdoku.windows.secret` | operators.py, supervisor, Setup-ServiceAccount.ps1 | apache_supervisor.py:119-121; Setup-ServiceAccount.ps1:11, :52 |
| secrets | `tls/{cert.pem,key.pem,ca.crt}`, `lichess_token.txt` | server.py (cert flags, bridge CA) | server.py:1424; instances.json args |
| lifecycle/per-process | `sessions[_port]/<sha256(cookie)>.json`, `launch[_port].json`, `last_exit.json`; `console-<instance>.pid`; `service_instances.pid/port/state` | lifecycle, Open-Console.ps1, supervisor | lifecycle.py:62-66, :451; LAUNCH_2026-11-10.md |
| canvases | `canvas/messenger.canvas` (CANVAS V4 body + EDGE_FIFO/PORT/CONTRACT_PORT trailer) | run_canvas.py | instances.json:27 |
| products | `web/**`: `*.wasm`, `*.abi.json`, WGSL, `boards.json`, fountain/policy products (317 MB) | prepare_* lanes, web_export.py, retained_products | server.py:1826-1849, :1937; CLOUD_HOSTING section 1 |
| products (cache, scratch) | `graph_wasm/`, `laws/`, `native_*`, `autopilot/`, `gallery/`, `dictionaries/`, `chess_*.npz`, `*.glsl`, probe dirs | build lanes | directory listing |
| outside | `C:/Apache24/conf/meowdoku-proxy.conf`, `C:/Apache24/logs` | Apache | LAUNCH_2026-11-10.md |

### Which live inside the turing repo tree, and why that blocks leasing

All of the above are under `turing/build/`, which is git-ignored (`.gitignore:10 /build/*`) but physically inside the worktree. Consequences:

1. **A new worktree is an empty `build/`.** `accounts.db`, photos and objects do not exist there; the instance would silently start a second, empty user database (Accounts() creates it, accounts.py:71-74).
2. **Copying the worktree copies or loses secrets.** `tls/key.pem`, `admin_token*.txt`, `operator_credentials/*` sit beside generated products; a worktree archive or `git clean -fdx` removes or leaks them.
3. **ACLs are repo-wide.** `Setup-ServiceAccount.ps1:94-97` gives the service Modify on `turing/build` and file-level read on tokens and `key.pem`; a leased worktree under another owner would need the same grant on a tree that also holds `operator_credentials`.
4. **Path coupling.** `instances.json` args are relative (`build/meowdoku/tls/...`, `build/meowdoku/canvas/messenger.canvas`) and the retained-products receipt is an absolute worktree path (instances.json:15-16). Two worktrees cannot share one instance definition.
5. **Per-port suffixes are the only collision rule** (`lifecycle.state_paths`, lifecycle.py:62-66): two servers on one checkout are kept apart by naming, not by a directory.
6. Logs are mixed with products and are never rotated except `log_ring.jsonl`.

## 2. The state root

One root outside every repo, beside the private interpreter:

```
C:/dev/Powershell/srv/state/
  sql/        accounts.db (+wal/shm)  job-library.sqlite  authz_policy.json  access.json
  content/    marble_objects/  marble_photos/  obj_assets/  shares/  tickets/  profiles.json  tipjar.json  loading_anim.*
  products/   retained compiled products (web/ export per source digest, receipts); lowering blobs if moved out of SQL
  canvas/     *.canvas   (Nodus canvases, incl. trailer records)
  logs/       log_ring.jsonl  apache-*.log  <instance>/<stamp>.log
  secrets/    tls/  admin_token*.txt  operator_credentials/  lichess_token.txt  (ACL: operators + service read only)
  run/        sessions[_port]/  launch*.json  last_exit.json  *.pid  console-*.pid
```
`srv/` is not under any repo (the parent repo ignores every child directory, `.gitignore` line 6 `/*/`, and `srv` is not whitelisted). `service_instances`/`account_instances` stay in `sql/accounts.db`; their pid/port/state columns are host-local and must not be replicated (CLOUD_HOSTING section 6).

Policy files (`access.json`, `authz_policy.json`) go in `sql/` because `authz.Policy.beside(accounts_db)` (server.py:364) already couples one to the db.

### What must learn `--state-dir`

Default = `C:/dev/Powershell/srv/state` (env `MEOWDOKU_STATE_DIR` for the scripts); the old `build/meowdoku` accepted only when passed explicitly (migration, and tests).

| Module | Change |
|---|---|
| server.py | New `--state-dir`; `Sessions.__init__` (server.py:340) takes `state_dir` instead of deriving from `web_dir.parent` for lines 345-367; `logring` path (:1958); `Lifecycle(..., state_dir)` (:1963); `bridge_ca` (:1424); `PROFILES` default (:86) and `--access` default. `--web-dir` stays per worktree: it is a product tree tied to a source revision. |
| lifecycle.py | `STATE_DIR` (:39) and `--state-dir` (:583) default to the root; `state_paths` already takes `state_dir`, with `run/` and `secrets/` subdirs. |
| apache_supervisor.py | `STATE` (:46), `--db` (:511), `--state-dir` (:513) defaults; `log_dir` (:409) -> `<state>/logs`; `console_secret_path` (:121) -> `<state>/secrets/operator_credentials`; instance log to `logs/<instance>/`. |
| operators.py, Setup-ServiceAccount.ps1 | `--db`/`--secrets-dir` and `$SecretsDir` (:12), ACL list (:94-97), token/key grants. |
| deploy/*.ps1 (Start/Stop/Status/Open/Close-Console) | pass the root; pid file location. |
| instances.json | args become `--state-dir`-relative: `tls` and canvas are named by root-relative paths resolved by the supervisor. |

Detachability: the root is the unit to replicate. `sql/` via Litestream per file; `content/`, `canvas/`, `products/` via rsync (content addressed, append-mostly); `secrets/` copied by hand, never replicated automatically; `logs/` and `run/` are not replicated. Nothing inside refers to a worktree path except products built from a specific source revision, and those already carry `source_digest` / `git_head` (lowering_library.py:35, :134; retained_products.py:108). The retained-products receipt path in instances.json becomes root-relative (`products/...`).

ACLs after the move (replaces Setup-ServiceAccount.ps1:94-97): service Modify on `srv/state` except `secrets/` (inheritance cut; read-only on `tls/`, tokens and the instance's own console secret; operator secrets remain operator-only). Worktrees get Read+Execute only, no Modify, so a leased tree cannot write into `build/` of another; its own `build/` stays scratch.

## 3. Leasing worktrees

The supervisor already models an instance as a row in `instances.json` (name, kind, port, route, bind, account, args, leases, python, nodus block) mapped to `service_instances` and `account_instances` (apache_supervisor.py:55-60) and to a service account (ensure_service_user, :307). A leased worktree is one more instance:

| Per instance (never shared) | Shared (state root) |
|---|---|
| name, service account, route, port, bind | `sql/accounts.db` (users, roles, sessions, lowering tables) |
| `python` + `--web-dir` of ITS worktree | `content/` (ownership via `owned_assets` / `job_owners`) |
| `run/` files: `launch_<port>.json`, `sessions_<port>`, pid (lifecycle.py:62-66) | `secrets/tls`, operator credentials |
| `logs/<instance>/` | `products/` (digest-addressed, read-mostly) |
| Nodus: `console_port`, contract-port prefix, canvas file | `canvas/` only if the canvas is meant to be shared; default one canvas per instance |

Two choices per lease, both already expressible: **same root** (a tenant sharing accounts, e.g. a dev build of the live game) or **own root** (`--state-dir srv/state-<lease>`, fully detachable; accounts are separate). Own root is the safe default for another party; it needs a seeded copy of `sql/` (or an empty one with its own `admin`).

### Versions: schema compatibility

Two worktrees at different revisions against one `accounts.db` is currently unprotected: every module does `CREATE TABLE IF NOT EXISTS` and a few `ALTER`/`table_info` adds (the supervisor adds `console_port` conditionally, apache_supervisor.py:75-76). A newer revision that adds a column breaks an older one silently. Proposal: a `schema_version(module TEXT PRIMARY KEY, version INTEGER, written_by TEXT, at REAL)` table, written by each module's connect and compared on open:

- If `version(module)` in the file is greater than what the code knows: refuse to open (read-only mode only if the module declares one). If lower: migrate forward, then write.
- Modules that create tables today and must register: `accounts.Accounts.__init__` (accounts.py:71, tables users/sessions/owned_assets), `authz._connect` (authz.py:197; account_roles, script_access, script_audit), `lowering_library.connect` (:72; lowering_*), `job_library.connect` (job_library.py:11; its own file, same table), `apache_supervisor.connect` (:65; service_instances, instance_leases, account_instances) and `server.py:576` (`account_instances`, duplicate definition that should defer to the supervisor module).
- Content addressed products need nothing: digest names already make versions coexist. `lowering_cores.core_id` (`logical:<logical_abi_digest>`) means two revisions with the same logical API share a core and differ only by `lowering_blobs` rows.

### Accounts and roles (existing vocabulary, authz.py:71-90)

| Actor | Role | Capabilities |
|---|---|---|
| `admin` | admin | all incl. `account_manage`, `deploy_scripts`; one human; bootstrap by operators.py |
| `codex`, `claude`, `guest_agent` | operator (in `script_access`) | studio_write, photo_submit, share_live, `deploy_scripts`, `lifecycle_admin`; no `account_manage`. One account per agent so `script_audit` names the actor. |
| `svc-meowdoku-a`, `svc-nodus-messenger`, one per leased instance | service | studio_write, photo_submit, share_live, library_manage; no deploy or admin. Created by the supervisor (ensure_service_user), password never given to a human. Per-lease accounts allow revoking one lease. |
| players | member | studio_write, photo_submit, share_live |
| anonymous | guest (no account) | tier caps only |

A federating remote instance authenticates as `service` (section 4). A new capability `federate` is cleaner than overloading `share_live` (which also gates photo/client-ray sharing): grant it to `service` and `operator`, not `member`.

## 4. Federating Nodus instances

Facts today: a contract port is a canvas-owned 127.0.0.1 listener (`include/contract_listener.h`, comment: "NO authentication", authorizes by loopback bind and the (role,id) accepted in `on_hello`). One 72-byte Hello (kind 5; work.id = role 1 host / 2 engine; work.engine = participant id; work.revision = 2) is consumed, then 72-byte frames of kinds 1-9 (NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md). The console port is a second loopback listener with its own token (`nodus/tools/run_canvas.py:203-207`, read-only). `nodus.graph-definition.v1` has no transport (`nodus_graph/model.py:364`, TURING_NODUS_GRAPH_DEFINITION.md). Keep all of that: do not widen the native bind.

Transport that fits "Apache should drive":

- Public URL `wss://<host>/ws/contract/<prefix>`, terminated by Apache (mod_proxy_wstunnel is enabled, LAUNCH_2026-11-10.md; `/ws` already returned 101 through Apache). Apache proxies to the Python server (`server.py` already serves `/ws` and `/ws/spectate`, :674, :1062), not to the native port.
- The Python server bridges each WebSocket to the loopback contract port of the instance that owns `<prefix>` (looked up in `service_instances`/the canvas CONTRACT_PORT trailer): binary WebSocket messages carry whole 72-byte frames, so framing is the existing frame. The native side still sees a loopback client; nothing in contract_listener changes.
- Authentication: the account session cookie or a bearer taken from the remote's service-account login, validated by `authz`. Authorization: capability `federate` (new) checked before the bridge opens. `Hello.id` is forced to the remote's service account id (users.id) by the bridge, so a peer cannot claim another id; the Hello role is allowed only as `engine` (2) unless the account is the owning instance's service.
- "Promise the same API", checked before any Hello: the first WebSocket text message is a JSON offer `{graph_definition_digest, cores:[{core_id, logical_abi_digest}], api_modules:[{module, entry}]}`. `graph_definition_digest` = sha256 of the canonical `GraphDefinition.to_mapping()` JSON (schema `nodus.graph-definition.v1`); `logical_abi_digest` comes from the lowering library's `lowering_cores` (lowering_library.py:35) and the module+entry list from the `.api.yaml` that produced them. The bridge compares with its own instance's values and replies `{ok}` or `{refused, reason, mismatched:[...]}`; only on `ok` does it relay the Hello. Mismatch closes the socket. The offer is per connection, not cached.
- FIFO ports across the internet: each side still owns its edges. A federated FIFO is a contract port pair: the remote side's engine (role 2) submits through its own local port, which is bridged to the owner's. No new native feature is needed to try it.

Not covered: NAT traversal (the owner must be reachable at Apache's 443, or the instance dials out), QoS and backpressure beyond the existing bounded queues (a slow WAN peer hits the "capacity of unadmitted Submits" protocol violation and is closed), replay and durability (negotiator state is process-owned; save/load re-declares the port, not contracts), the GUI hosting path, encryption of content beyond TLS, rate limits, and the certificate problem (the current cert has no public SAN; use a DNS name first). This is design only; nothing here exists or was tested.

## 5. Migration plan (each step reversible; one at a time; no builds)

1. **Create the root and ACLs.** `srv/state` with the subdirs; set ACLs; copy nothing. Verify: `icacls` shows service Modify only outside `secrets/`; `git status` in the parent repo shows no change. Undo: delete the empty tree.
2. **Logs.** Add `--state-dir` to supervisor and `log_ring.jsonl`; point `logs/` at the root. Verify: restart via Apache (existing session-preserving stop); new `apache-access.log` and `<instance>/` logs appear in the root; Status.ps1 unchanged. Undo: omit the flag.
3. **Secrets and run/.** Copy (not move) `tls/`, `admin_token*.txt`, `operator_credentials/` to `secrets/`, point lifecycle and operators at it; re-run `script-auth` as claude. Verify: admin token read works, a console token is written under `secrets/`, a script gate still passes. Keep the old copy until step 8.
4. **SQL.** Stop the stack, copy `accounts.db` with the SQLite backup API (not a file copy while WAL is live), also `job-library.sqlite`; start with `--db` / `--state-dir`. Add `schema_version` registration in this step. Verify: login of admin and a member, `service_instances` rows, lease checkout, same row counts and `PRAGMA integrity_check` on both. Undo: restart with the old path.
5. **Content.** rsync `marble_objects`, `marble_photos`, `obj_assets`, `shares`, `tickets`, `profiles.json` into `content/` (source kept). Verify: file counts and digests match; a known saved object and photo load for their owner and 404 for another account; profile write goes to the new file.
6. **Canvas and products.** Move `canvas/` and the retained-product receipts to the root; edit instances.json to root-relative paths. Verify: Nodus host listens on its contract port, an engine Hello completes, whoami reports the instance.
7. **Cut over defaults.** Make the root the default for server.py, lifecycle and scripts; old `build/meowdoku` only by explicit flag. Verify: a server started with no flags writes only under the root (watch `build/meowdoku` mtimes for a day).
8. **Lease the first worktree.** `git worktree add` at a short path, new instance with its own service account, route and port, same root; run with `schema_version` enforcement; then try an own-root lease. Verify: two revisions serve the same accounts, an incompatible schema is refused, then retire the old `build/meowdoku` state copy after one Litestream restore drill (CLOUD_HOSTING section 6 step 5).

Federation (section 4) comes after step 8 and needs its own review before any code.

## Implemented (code only), 2026-11-10

Migration steps 1-3 are code; the live cutover has NOT run (no state moved or copied, no process started or stopped, `build/meowdoku` untouched, no root created).

- `examples/meowdoku/state_paths.py`: `StatePaths(root)` (new layout), `StatePaths.legacy(repo_root)` / `legacy_at(base)` (today's `build/meowdoku` names), `ensure()` (dirs plus the operators.py icacls step on `secrets/`).
- `server.py --state-dir PATH`: absent means the old layout beside `--web-dir`. `Sessions(web_dir, profiles, state)` holds it; accounts.db, access.json, authz policy, profiles, content dirs, log ring, bridge CA and the lifecycle token/session/launch files come from it. `--web-dir` unchanged.
- `lifecycle.py --state-dir`: `state_paths()` takes a StatePaths (sessions/launch in `run/`, token in `secrets/`) or a plain dir (old meaning). `STATE_DIR` stays as the default.
- `deploy/apache_supervisor.py --state-dir` and top-level `"state_dir"` in the object form of instances.json (flag wins): db defaults to `<root>/sql/accounts.db`, logs to `<root>/logs`, console tokens to `<root>/secrets/operator_credentials`, lock to `<root>/run`; meowdoku children get `--state-dir`. Without either, nothing changes. Note: the supervisor's pre-existing `--state-dir` now names the root (new layout), not the old base.
- `deploy/operators.py --state-dir` replaces `--db`/`--secrets-dir`.
- `deploy/*.ps1`: `-StateDir` (or `$env:MEOWDOKU_STATE_DIR`), empty means legacy; threads the gate's `-AccountsDb` default, operator secret, tls, logs, pid files and `apache_supervisor.py --status`; Setup-ServiceAccount.ps1 derives `$SecretsDir` and the token/key grants.
- Not done: `job_library` still resolves beside the photo dir (`job_library_db` is not wired), `tipjar.json` and `lichess_token.txt` still read `build/meowdoku`, instances.json args (tls, canvas) stay literal, the schema_version table, and the default flip (step 7).

## Implemented (phase 1, loopback-tested), 2026-11-10

Code: `authz.py` (capability `federate` for admin/operator/service, never a tier cap; table `peer_grants`; `grant_peer`/`revoke_peer`/`verify_peer`/`list_peers`; CLI `peer-grant|peer-revoke|peer-list`, the secret printed once on a `SECRET` line), `server.py` (`/ws/contract/<prefix>`, `Sessions.federation_identity()`, flags `--federation-port` and `--federation-canvas`; blocks marked `[FEDERATE]`), `federation_peer.py` (stdlib WebSocket client, `identity_digests`, a socket-shaped `PeerSocket` for `ServicePeer(..., announce=False)` / `EnginePeer`), `tests/test_meowdoku_federation.py`.

Decisions: the grant is the credential (service to service, no cookie); the route stays behind the access tier (shader tier refused by the existing gate, `view_site` required, viewer links refused). A grant names peer, prefix, direction (host=Hello role 1, engine=2), allowed frame kinds (Hello is never grantable), expiry, revocation, and a local service account whose id is forced into Hello.id. Only a sha256 of the secret is stored. Refusals before the grant check are unauthenticated and give no detail; a failed grant answers one reason ("grant refused") for wrong secret, revoked, expired, wrong prefix or wrong direction. Only after the grant verifies are digests compared (`graph_definition_digest` = sha256 of the canvas file bytes; `api_digests` = sorted `module:entry` of every `*.api.yaml` under web_dir) and mismatches named. The local contract port comes from `--federation-port` (the supervisor's service_instances does not carry it yet); no flag means every promise is refused with a reason.

Covered (stub contract port, ephemeral ports, temp state): Hello role/id as granted, peer Result reaches the port, Receipt returns, wrong secret / revoked / expired / wrong prefix / digest mismatch / no instance refused, a frame kind outside allowed_ops or a peer-sent Hello closes 1008 and never reaches the port, closing either side closes the other, `EnginePeer` runs over `PeerSocket`.

Not covered: any real Nodus host or cross-machine run; the Apache wstunnel path (untested live, no TLS); NAT traversal, QoS and backpressure beyond the bounded WS/TCP buffers, durability or replay; the graph-definition digest is of the canvas file, not the canonical `GraphDefinition.to_mapping()` JSON, and the api digest omits logical_abi_digest per core (both are section 4's fuller design); no rate limiting of failed grants; a federated peer's own Hello in the Nodus console/ownership model is unexamined.

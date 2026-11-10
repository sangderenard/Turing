# Operator guide for the second agent (2026-11-10)

Audience: the other agent (storage / leases / merge tickets / studio work). Terse by intent. Every claim names its source file (`deploy/X` or `examples/meowdoku/X`; `Mk` below = `examples/meowdoku`). Read-only vs state-changing is marked. Nothing here authorizes touching the live instances.

Standing rules (LAUNCH, lane-split note, meowdoku AGENTS.md): Apache owns the live instances. Never launch over 8766 or 8767. You run an INDEPENDENT dev server: own port, own `--web-dir`, own state root. One window at a time on the operator's desktop, with a stated reason. Do not commit foreign uncommitted edits.

```powershell
$Py     = 'C:\dev\Powershell\srv\python\python.exe'     # private interpreter; use it for the CLIs below
$Mk     = 'C:\dev\Powershell\turing\examples\meowdoku'
$Live   = 'C:\dev\Powershell\srv\state'                  # LIVE root: read-only for you
$LiveDb = "$Live\sql\accounts.db"
```

## 1. Layout

- Live state root `C:/dev/Powershell/srv/state` (STATE_AND_FEDERATION "The state root"; `state_paths.py`): `sql/` (accounts.db, job-library.sqlite, authz_policy.json, access.json), `content/`, `products/`, `canvas/`, `logs/`, `secrets/` (tls, admin tokens, operator_credentials), `run/` (sessions, launch json, pids). Cutover measured 14:04; `instances.json` carries `"state_dir"`; `MEOWDOKU_STATE_DIR` is set at user scope (LAUNCH "State root cutover").
- Still legacy by code: `tipjar.json`, `lichess_token.txt` (read from `build/meowdoku`), products / `web_dir` (retained fountain receipt path), the 8767 dev server (Nov 7 code). `build/meowdoku` is untouched and is the rollback; `deploy/Rollback-State.ps1` only prints steps.
- Private interpreter `C:/dev/Powershell/srv/python` (CPython 3.11.18, Torch-free; `deploy/requirements-service-notorch.txt`; the Torch variant `deploy/requirements-service.txt` is unused). Provisioned by `deploy/Provision-ServicePython.ps1` (needs a user-approved archive; do not run).
- Nodus builds (SERVICE_ACCOUNT, LAUNCH): `nodus/build-bridging` = Torch GUI/viewer (`frontend_shell.exe`, operator account); `nodus/build-service` = Torch-free headless (`nodus_headless.dll`), used by the service. Do not rebuild either unless the operator says so.
- `deploy/Setup-ServiceAccount.ps1` has NOT been run elevated; the stack still runs as the operator account (LAUNCH "Private interpreter live"). Do not run it.

## 2. Accounts and roles

Who exists: operators `admin` (role admin), `codex`, `claude`, `guest_agent` (role operator, in `script_access`); per-instance service accounts `svc-meowdoku-a`, `svc-nodus-messenger` (role service, created by the supervisor); players are members; anonymous is guest (LAUNCH "Who may run the scripts"; STATE_AND_FEDERATION section 3).

Capabilities by role (`authz.py` DEFAULT_POLICY). Tiers add `view_site/play/dev_ui`; tier `edit` grants most others but never `lifecycle_admin`, `account_manage`, `deploy_scripts`, `federate`.

| role | caps |
|---|---|
| guest | none beyond tier |
| member | studio_write, photo_submit, share_live |
| service | member + library_manage, federate |
| operator | member + deploy_scripts, lifecycle_admin, federate |
| admin | all (incl. account_manage) |

`lifecycle_admin` additionally needs origin `lan`; `account_manage` needs role admin (`authz.py` docstring).

Secrets: `C:\dev\Powershell\srv\state\secrets\operator_credentials\<user>.secret`. Never print, copy, log or commit them. Console tokens `<instance>.console.secret` are per-run, same dir.

Authenticating to the scripts: `-User <name>` (default `$env:MEOWDOKU_OPERATOR`), `-PasswordFile` (defaults to that user's `.secret`), `-StateDir`, `-AccountsDb` (tests). Every `deploy\*.ps1` runs `authz.py script-auth` first, even with `-WhatIf`: the account must verify, be in `script_access`, and hold `deploy_scripts`; each attempt writes a `script_audit` row; refusal prints `REFUSED <reason>`, exit 3.

Read-only inspection of the live db:

```powershell
& $Py "$Mk\authz.py" script-list --db $LiveDb
& $Py "$Mk\authz.py" peer-list   --db $LiveDb
& $Py "$Mk\authz.py" schema      --db $LiveDb
& $Py "$Mk\authz.py" show claude --db $LiveDb
& $Py "$Mk\authz.py" caps --role operator --tier edit
# audit trail: script_audit(ts, username, script, allowed, host), newest 20
$uri = 'file:' + ($LiveDb -replace '\\','/') + '?mode=ro'
& $Py -c "import sqlite3,sys;c=sqlite3.connect(sys.argv[1],uri=True);[print(r) for r in c.execute('select ts,username,script,allowed,host from script_audit order by ts desc limit 20')]" $uri
```

STATE-CHANGING (an admin's call; run against YOUR dev db, not the live one, unless told):

```powershell
& $Py "$Mk\deploy\operators.py" bootstrap --state-dir <root>      # creates missing accounts, never overwrites a .secret; prints names and paths only
& $Py "$Mk\deploy\operators.py" rotate <user> --state-dir <root>  # writes a new secret file
& $Py "$Mk\authz.py" grant-admin <user> --db <db> --granted-by <existing-admin-or-operator:prefix>
& $Py "$Mk\authz.py" revoke-admin <user> --db <db> --granted-by <admin>
& $Py "$Mk\authz.py" script-grant <user> --db <db> --granted-by <admin>
& $Py "$Mk\authz.py" script-revoke <user> --db <db>
```

`operators.py` docstring: `admin` gets role admin, every other name role operator, all added to `script_access`. While `account_roles` is empty, `grant-admin` accepts only `operator:<admin_token_sha256_prefix>` (`authz.py` BOOTSTRAP). To give someone operator role, add them to `script_access` and make the account an operator through `operators.py bootstrap --users`.

## 3. Your own development instance (never touches live)

Pick a port outside 80/443/8766/8767/8768 (the ports Status.ps1 watches) and outside 47611/47612 (Nodus host), e.g. 8790, route `dev`. Confirm it is free (read-only):

```powershell
Get-NetTCPConnection -LocalPort 8790 -State Listen -ErrorAction SilentlyContinue
```

`server.py` flags: `--port --bind --web-dir --state-dir --profiles --route --instance --tls-cert --tls-key --no-wasm --no-idle-train`. `--web-dir` defaults to the LIVE `build/meowdoku/web`; with no `--state-dir` state lives beside `--web-dir`, i.e. the live legacy tree. So BOTH flags are mandatory for you.

Fresh dev root (creates `sql/ content/ products/ canvas/ logs/ secrets/ run/` and restricts `secrets/` with icacls; `state_paths.py` `ensure`), your own web dir, your own operator accounts:

```powershell
$Dev    = 'C:\dev\Powershell\srv\dev-<yourname>'       # not under srv\state, not under a repo worktree
$DevWeb = 'C:\dev\Powershell\<your-worktree>\build\meowdoku\web_<yourname>'   # a product tree you own
& $Py -c "import sys;sys.path[:0]=[r'C:\dev\Powershell\turing',r'C:\dev\Powershell\turing\examples'];from meowdoku.state_paths import StatePaths;print(StatePaths(sys.argv[1]).ensure())" $Dev
& $Py "$Mk\deploy\operators.py" bootstrap --state-dir $Dev
```

Launch (foreground, in its own terminal; the live argv of `deploy/Start-Meowdoku.ps1` minus the live-only flags):

```powershell
Set-Location C:\dev\Powershell\turing; $env:PYTHONUTF8 = '1'
& $Py examples\meowdoku\server.py --port 8790 --bind 127.0.0.1 --route dev --web-dir $DevWeb --state-dir $Dev --profiles "$Dev\content\profiles.json" --no-wasm --no-idle-train
```

Gated script form: `.\Start-Meowdoku.ps1 -Port 8790 -Route dev -Bind 127.0.0.1 -StateDir $Dev -User <u> -WhatIf`. Caveats read from the script: it has NO `-WebDir` (non-`-Dev` omits `--web-dir`, so it would serve the live web dir); it passes the live retained-fountain receipt; it expects TLS under `<StateDir>\secrets\tls`; it uses the operator's system Python; its `-Dev` form hardcodes port 8767 and another worktree's web dir. Drop `-WhatIf` only if the printed argv is acceptable; otherwise use the direct command. It refuses when the port has a listener.

Stop: `.\Stop-Meowdoku.ps1 -Port 8790 -StateDir $Dev -User <u> -WhatIf` (session-preserving; reads `$Dev\secrets\admin_token_8790.txt`), or close your own foreground terminal.

Alternative, supervised like live: copy `deploy\instances.json` to your own file, keep one meowdoku entry (your port, route, account, `--web-dir`) and a top-level `"state_dir"` of your root; drop the nodus entry unless needed.

```powershell
& $Py "$Mk\deploy\apache_supervisor.py" --instances C:\path\to\my-instances.json --state-dir $Dev --once --dry-run   # prints the plan, changes nothing
& $Py "$Mk\deploy\apache_supervisor.py" --instances C:\path\to\my-instances.json --state-dir $Dev                   # real run; stdin EOF = graceful stop
```

The supervisor holds an OS lock on `<state>/supervisor.lock`, so a separate root never contends with live (`apache_supervisor.py` docstring).

NEVER: launch or restart anything on 8766 or 8767; use `$Live` as `--state-dir` for experiments; run `tools/publish_*` against `build/meowdoku/web` or live 8766 casually; bind `0.0.0.0` on a dev instance; start, stop or restart Apache for development; kill a server to free a port (meowdoku `AGENTS.md`).

## 4. The live stack (operator-gated)

- `.\Status.ps1 -User <u>`: listeners on 80/443/8766-8768 with PID and process, plus the supervisor table. Read-only.
- `.\Start-Apache.ps1`, `.\Stop-Apache.ps1`: the only sanctioned live start/stop. Not for development. Stop closes httpd's pipe; the supervisor stops the instances (Meowdoku lifecycle restart request, Nodus CTRL_BREAK), checks leases in, marks them `stopped`. `-WhatIf` prints the commands.
- Supervisor (`deploy/apache_supervisor.py`): copies the access log to `<state>/logs/apache-access.log`, ensures service accounts, checks out declared leases, launches and supervises children (crash backoff, max 5 restarts per 10 min, then `failed`), renews leases. A port that already has a listener is left alone (state `external`). Not gated: it runs under Apache's identity.
- Instances table (read-only): `& $Py "$Mk\deploy\apache_supervisor.py" --status --state-dir $Live` prints `service_instances` as JSON (name, kind, port, route, pid, state, restarts, console_port, contract prefix/port).
- Console (read-only viewer): `.\Open-Console.ps1 -Instance nodus-messenger -User <u>` runs `frontend_shell.exe --attach=127.0.0.1:<console_port> --console-token-file=<token>`, refuses if the instance is not running, records `run\console-<instance>.pid`. `.\Close-Console.ps1 -Instance nodus-messenger -User <u>` stops only the viewer. The host accepts no commands (`REFUSED read-only`). It opens a desktop window: one at a time, tell the operator why.
- Logs: `$Live\logs\` (`apache-access.log`, `apache-supervisor.log`, `<instance>-<stamp>.log`, `log_ring.jsonl`); Apache's own in `C:\Apache24\logs`.

## 5. Adding tables or state

Follow `schema_version.py`: in your module's `connect()`, run your DDL, then `schema_version.declare(conn, "<module>", N)`. A stored version above yours raises `SchemaVersionError` (an older checkout never touches a newer db); bump N on incompatible DDL. One writer (module) per table. Declared today at v1: accounts, authz (also owns `account_instances`), lowering_library, job_library, supervisor. Announce new tables in the lane-split note. Resolve every path through `StatePaths`; nothing mutable under a repo worktree. Inspect: `authz.py schema --db <db>`. The review found `workspace_accounting.py` creates `workspace_*` tables through its own connect with no `schema_version`: do not point it at a live root until it uses the shared gate.

## 6. Federation (phase 1, loopback-tested only)

- A peer grant (`peer_grants` table, `authz.py`) names peer, prefix, direction (`host` = Hello role 1, `engine` = 2), allowed frame kinds (Hello is never grantable), expiry, revocation, and a local SERVICE account whose id is forced into Hello.id. Only a sha256 of the secret is stored. Capability `federate` belongs to admin, operator, service only.
- Issue (state-changing; the secret prints ONCE on the `SECRET` line; do not echo it into logs; use your dev db):

```powershell
& $Py "$Mk\authz.py" peer-grant <peer> --db "$Dev\sql\accounts.db" --prefix <prefix> --direction engine --ops <kinds> --days 7 --issued-by <you> --service-user <local service account>
& $Py "$Mk\authz.py" peer-revoke <grant_id> --db "$Dev\sql\accounts.db"
```

- Endpoint `/ws/contract/<prefix>` on a server started with `--federation-port <nodus contract port> --federation-canvas <canvas>`; without the flag every promise is refused with a reason. It sits behind the access tier (shader tier refused, `view_site` required).
- Client: `federation_peer.connect("ws://127.0.0.1:<p>/ws/contract/<prefix>", grant_id, secret, "engine", identity_digests(canvas, web_dir))`; `PeerSocket` plugs into `ServicePeer(..., announce=False)` or `EnginePeer`.
- Digest promise: only after the grant verifies, `graph_definition_digest` (sha256 of the canvas FILE bytes) and `api_digests` (sorted `module:entry` of every `*.api.yaml` under web_dir) are compared and mismatches named. A bad grant gets one reason, "grant refused".
- Not proven (STATE_AND_FEDERATION "Implemented (phase 1)"): any real Nodus host, cross-machine run, the Apache wstunnel path with TLS, NAT, QoS, replay, rate limits on failed grants. Do not claim it works live.

## 7. Lowering library

`lowering_library.py` CLI: `scan <root> --db`, `list --db`, `versions <core_id> <backend> --db`, `checkout --db --purpose family:name --caps a,b --account <user> --seconds N [--prefer]`, `checkin <lease_id> --db --account --generation`, `leases --db`. Tables `lowering_cores`, `lowerings`, `lowering_blobs`, `lowering_leases`. The live startup scan catalogued 0 products: retained products predate the `.api.yaml` writers, so live products need an explicit rebuild before they carry `.api.yaml`; that is an operator decision, not a restart side effect (LAUNCH "Console, viewer and the module-root fix"). Use your own db:

```powershell
& $Py "$Mk\lowering_library.py" scan $DevWeb --db "$Dev\sql\accounts.db"
& $Py "$Mk\lowering_library.py" list --db "$Dev\sql\accounts.db"
```

## 8. Coordination

- Lane-split note: `C:\dev\Powershell\speaktome\AGENTS\experience_reports\1794334300_DOC_Lane_Split_Apache_Service_Console_Federation.md`. The Apache/service/console lane owns `deploy/`, `server.py` and the `accounts.db` schema, `state_paths.py`, `schema_version.py`, `federation_peer.py`, the state-root cutover and federation. Your lane: disk accounting, worktree leases, merge tickets and the approval queue, temp/quarantine cleanup. Keep the account gate on `deploy/*.ps1`; never bypass it.
- Review `1794338100_REVIEW_Studio_Car_Game_Uncommitted_Changes.md`: your uncommitted studio edits are safe beside the live service. `tests/test_studio_game_entry.py` and `tools/stage_studio_game_layer.py` read frozen artifacts (`build/dot-newer-tire-dynamics-20261110/`, `build/dot-studio-game-transfer-20261110/`) and fail on a clean checkout. Hold `workspace_accounting.py` until it uses `schema_version`. `tools/publish_studio_game_layer.py` posts to live 8766 whenever invoked; do not invoke it casually.
- Uncommitted foreign edits in the live tree are never committed by the other lane, and you do not commit theirs.
- Read `HOW_TO_START.md` and meowdoku `AGENTS.md` before touching any running server: never kill a server to free a port, preserve sessions, coordinate with active workers.

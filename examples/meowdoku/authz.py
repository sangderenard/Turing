"""Meowdoku permissions: three roles (admin, member, guest) over the existing origin tiers.

ROLES.  ``guest`` is the absence of an account.  ``member`` is any account in ``accounts.db`` ``users`` without a row in
``account_roles``.  ``admin`` is an ``account_roles`` row; so is ``service`` (the supervisor's instance accounts: member caps +
library_manage, see DEFAULT_POLICY).  A request's PRINCIPAL is (role, user_id, username, origin, tier); origin and
tier come from access.py exactly as today.

CAPABILITIES (names are the whole vocabulary): view_site, play, dev_ui, owner_tools, library_manage, lifecycle_admin, studio_write,
photo_submit, share_live, account_manage, deploy_scripts, federate, compile_submit.  ``capabilities(principal, policy)`` is the one place they are computed:

    effective = tier_caps[tier] | role_caps[role]
    lifecycle_admin additionally requires origin == 'lan' (the operator's loopback/LAN rule stays)
    account_manage additionally requires role == 'admin'
    guest gets tier_caps only

Today's "LAN is edit" behaviour is preserved as POLICY (``tier_caps['edit']``, the default), not as code: a policy file can strip it.
The policy file ``authz_policy.json`` (beside accounts.db by default; re-read when its mtime changes, like access.py) holds
``{"tier_caps": {tier: [caps]}, "role_caps": {role: [caps]}}``; entries merge over the defaults.

BOOTSTRAP.  While ``account_roles`` is empty, ``grant_admin`` only accepts ``granted_by='operator:<admin_token_sha256_prefix>'``.  This
module does not check the token; the server's lifecycle.admin_authorize does, then calls here.  Afterwards the grantor is either an
operator string or an existing admin's username.

INTEGRATION SPEC (server.py is not in this branch).  Build the principal once in ``_classify``/``_job_identity``:
``principal_from(self._account(), self._origin, self._tier, db_path)``; replace each gate with ``authz.require(principal, cap)``
(AuthzError.status/.message go straight into ``_json``).  Keep token checks after the capability check.

    server.py line   current gate                                  purpose                                  capability
    ~495  _log_get          require_tier dev                       /api/log text ring                       dev_ui
    ~511  whoami            at_least dev / edit; tier == shader    page flags                               dev_ui / owner_tools; shader := not view_site
    ~515  do_HEAD           tier == 'shader'                       HEAD refused for shader tier             not view_site
    ~760  do_GET            tier == 'shader'                       shader-only redirect                     not view_site
    ~765  /api/tickets GET  require_tier dev                       ticket list                              dev_ui
    ~788  marble_photo/jobs require_tier edit                      photo job list                           photo_submit
    ~795  marble_photo/lenses require_tier edit                    lens list                                photo_submit
    ~803  marble_photo*     require_tier edit                      photo image/hdr/animation reads          photo_submit
    ~952  _ready            tier != 'shader'                       full readiness snapshot                  view_site
    ~987  loading anim upload  require_tier play|edit              temporary / permanent upload             play / owner_tools
    ~1048 owner_authorize   origin == 'lan' bypass                 loading-anim owner check                 library_manage (token check stays)
    ~1082 admin/lifecycle   require_tier edit                      lifecycle trigger                        lifecycle_admin (+ admin token)
    ~1132 do_POST           tier == 'shader'                       shader POST refusal                      not view_site
    ~1148 tickets POST      require_tier play | edit(/status)      file ticket / change status             play / owner_tools
    ~1181 /api/obj_assets   require_tier edit                      studio asset upload                      studio_write
    ~1198 /api/marble_objects require_tier edit                    studio object write                      studio_write
    ~1218 marble_camera/housing require_tier edit                  studio camera housing                    studio_write
    ~1240 marble_photo POST/cancel/live/retouch/film_preview       photo submit and live share             photo_submit (live: share_live)
    ~1343 loading_library activate/delete/graduate  edit           library manage (+ owner token)           library_manage
    ~1354 client_rays relay require_tier edit                      client ray relay                         share_live
    ~1416 /api/action batch EDIT_ACTIONS  require_tier edit        process-wide learner/Lichess/pilot       owner_tools
    ~1423 /api/action single EDIT_ACTIONS require_tier edit        same                                     owner_tools
    new   account admin endpoints (grant/revoke/list)              role management                          account_manage

COMPILE SUBMIT.  ``compile_submit`` (POST /api/mathlab/jobs: spend the single math-lab worker's compile time) is granted by NO tier, only by
the roles member, operator, service and admin, so a guest or an anonymous LAN visitor (tier edit) cannot submit.

SCRIPT ACCESS.  ``deploy_scripts`` is granted by no tier, only by role admin/operator.  deploy/*.ps1 run ``script-auth``: the account must
verify, be in ``script_access`` (explicit allow-list) and hold deploy_scripts; every attempt is a ``script_audit`` row (exit 3 refuses).
CLI: ``script-grant|script-revoke|script-list|script-auth``; ``deploy/operators.py`` bootstraps admin/codex/claude/guest_agent.

CLI: ``python authz.py grant-admin <user> --db accounts.db --granted-by operator:<prefix>``; ``revoke-admin``; ``show``; ``caps``.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from meowdoku import schema_version as _schema_version
except ImportError:      # loaded by path or run as a script: take the sibling file
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("meowdoku_schema_version", Path(__file__).with_name("schema_version.py"))
    _schema_version = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_schema_version)

CAPS = ("view_site", "play", "dev_ui", "owner_tools", "library_manage", "lifecycle_admin", "studio_write", "photo_submit",
        "share_live", "account_manage", "deploy_scripts", "federate", "compile_submit")
ROLES = ("admin", "member", "service", "operator", "guest")
ORIGINS = ("lan", "ip", "dns")
TIERS = ("shader", "play", "dev", "edit")
POLICY_FILE = "authz_policy.json"

DEFAULT_POLICY = {
    "tier_caps": {
        "shader": ["view_site"],
        "play": ["view_site", "play"],
        "dev": ["view_site", "play", "dev_ui"],
        "edit": [c for c in CAPS if c not in ("lifecycle_admin", "account_manage", "deploy_scripts", "federate", "compile_submit")],
    },
    "role_caps": {
        "member": ["studio_write", "photo_submit", "share_live", "compile_submit"],
        # service: a supervisor-owned instance account (deploy/apache_supervisor.py).  Member caps plus library_manage, so the
        # account that holds an instance's lowering leases may manage the library; no lifecycle_admin/account_manage.
        "service": ["studio_write", "photo_submit", "share_live", "library_manage", "federate", "compile_submit"],
        # operator: a member that may run the deploy scripts (see script_access) and the lifecycle (still LAN-only).
        "operator": ["studio_write", "photo_submit", "share_live", "deploy_scripts", "lifecycle_admin", "federate", "compile_submit"],
        "admin": list(CAPS),
        "guest": [],
    },
}

_DDL = """
CREATE TABLE IF NOT EXISTS account_roles (
    user_id INTEGER PRIMARY KEY,
    role TEXT NOT NULL CHECK(role IN ('admin','member','service','operator')),
    granted_by TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS script_access (
    username TEXT PRIMARY KEY,
    granted_by TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS script_audit (
    ts REAL,
    username TEXT,
    script TEXT,
    allowed INTEGER,
    host TEXT
);
CREATE TABLE IF NOT EXISTS account_instances (user_id INTEGER PRIMARY KEY, instance TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS peer_grants (
    grant_id INTEGER PRIMARY KEY AUTOINCREMENT,
    peer TEXT NOT NULL,
    prefix TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('host','engine')),
    allowed_ops TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    issued_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked_at REAL,
    service_user_id INTEGER NOT NULL
);
"""
_lock = threading.Lock()


class AuthzError(Exception):
    def __init__(self, message: str, status: int = 403):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Principal:
    role: str                 # admin | member | service | guest
    user_id: int | None
    username: str | None
    origin: str               # lan | ip | dns
    tier: str                 # the origin tier string as access.py gives it


# ---- policy -------------------------------------------------------------------------------------------------------------
def load_policy(path) -> dict:
    out = {k: {n: list(v) for n, v in m.items()} for k, m in DEFAULT_POLICY.items()}
    if path is None or not Path(path).exists():
        return out
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as error:
        print(f"[authz] {path}: unreadable ({error}); using the defaults", file=sys.stderr, flush=True)
        return out
    for key, names in (("tier_caps", TIERS), ("role_caps", ROLES)):
        for name, caps in (raw.get(key) or {}).items():
            if name in names and isinstance(caps, list) and all(c in CAPS for c in caps):
                out[key][name] = list(caps)
            else:
                print(f"[authz] ignoring {key} {name!r}: {caps!r}", file=sys.stderr, flush=True)
    return out


class Policy:
    """The policy file, re-read whenever its mtime changes."""

    def __init__(self, path):
        self.path = Path(path) if path is not None else None
        self._stamp, self._policy = object(), load_policy(None)
        self._lock = threading.Lock()

    @classmethod
    def beside(cls, db_path) -> "Policy":
        return cls(Path(db_path).parent / POLICY_FILE)

    def get(self) -> dict:
        try:
            stamp = self.path.stat().st_mtime_ns if self.path is not None else None
        except OSError:
            stamp = None
        with self._lock:
            if stamp != self._stamp:
                self._stamp, self._policy = stamp, load_policy(self.path)
            return self._policy


def capabilities(principal: Principal, policy: dict) -> frozenset:
    caps = set(policy["tier_caps"].get(principal.tier, ()))
    caps |= set(policy["role_caps"].get(principal.role, ()))
    if principal.origin != "lan":
        caps.discard("lifecycle_admin")
    if principal.role != "admin":
        caps.discard("account_manage")
    return frozenset(c for c in caps if c in CAPS)


def require(principal: Principal, cap: str, policy: dict | None = None) -> None:
    if cap not in capabilities(principal, policy or DEFAULT_POLICY):
        raise AuthzError(f"this needs {cap}", 403)


# ---- roles in accounts.db -------------------------------------------------------------------------------------------------
def _connect(db_path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(conn, "authz", 1)       # first: a newer database is refused before this module writes anything
    except _schema_version.SchemaVersionError:
        conn.close()
        raise
    conn.executescript(_DDL)
    sql = (conn.execute("SELECT sql FROM sqlite_master WHERE name = 'account_roles'").fetchone() or [""])[0] or ""
    if "'operator'" not in sql:     # a table created before the service/operator roles: rebuild it with the wider CHECK (rows kept)
        conn.executescript("BEGIN; ALTER TABLE account_roles RENAME TO account_roles_old;" + _DDL +
                           "INSERT INTO account_roles SELECT user_id, role, granted_by, created_at FROM account_roles_old;"
                           "DROP TABLE account_roles_old; COMMIT;")
    return conn


def role_of(db_path, user_id) -> str:
    if user_id is None:
        return "guest"
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT role FROM account_roles WHERE user_id = ?", (user_id,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else "member"


def _user_id(conn, username: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone()
    if row is None:
        raise AuthzError(f"no such account {username!r}", 404)
    return row[0]


def _check_grantor(conn, granted_by: str) -> None:
    text = str(granted_by)
    if text.startswith("operator:") and len(text) > len("operator:"):
        return
    empty = conn.execute("SELECT COUNT(*) FROM account_roles").fetchone()[0] == 0
    if empty:
        raise AuthzError("the first admin must be granted by operator:<admin_token_sha256_prefix>", 403)
    row = conn.execute("SELECT 1 FROM account_roles r JOIN users u ON u.id = r.user_id "
                       "WHERE r.role = 'admin' AND u.username = ? COLLATE NOCASE", (text,)).fetchone()
    if row is None:
        raise AuthzError("granted_by must be an operator string or an existing admin", 403)


def grant_admin(db_path, username: str, granted_by: str) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            _check_grantor(conn, granted_by)
            uid = _user_id(conn, username)
            conn.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?, 'admin', ?, ?)",
                         (uid, str(granted_by), time.time()))
            conn.commit()
            return uid
        finally:
            conn.close()


def revoke_admin(db_path, username: str, granted_by: str) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            _check_grantor(conn, granted_by)
            uid = _user_id(conn, username)
            conn.execute("DELETE FROM account_roles WHERE user_id = ?", (uid,))
            conn.commit()
            return uid
        finally:
            conn.close()


def principal_from(account, origin: str, tier: str, db_path) -> Principal:
    if account is None:
        return Principal("guest", None, None, origin, tier)
    return Principal(role_of(db_path, account["id"]), account["id"], account["username"], origin, tier)


def grant_role(db_path, username: str, role: str, granted_by: str) -> int:
    """Set a non-admin role row (operator, service, member) for an account; admin goes through grant_admin's grantor check."""
    if role == "admin":
        return grant_admin(db_path, username, granted_by)
    if role not in ROLES or role == "guest":
        raise AuthzError(f"cannot grant role {role!r}", 400)
    with _lock:
        conn = _connect(db_path)
        try:
            uid = _user_id(conn, username)
            conn.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?, ?, ?, ?)",
                         (uid, role, str(granted_by), time.time()))
            conn.commit()
            return uid
        finally:
            conn.close()


# ---- script access: who may run deploy/*.ps1 ------------------------------------------------------------------------------
def script_grant(db_path, username: str, granted_by: str) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            name = conn.execute("SELECT username FROM users WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone()
            if name is None:
                raise AuthzError(f"no such account {username!r}", 404)
            conn.execute("INSERT OR REPLACE INTO script_access (username, granted_by, created_at) VALUES (?, ?, ?)",
                         (name[0], str(granted_by), time.time()))
            conn.commit()
        finally:
            conn.close()


def script_revoke(db_path, username: str) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute("DELETE FROM script_access WHERE username = ? COLLATE NOCASE", (str(username),))
            conn.commit()
        finally:
            conn.close()


def script_list(db_path) -> list:
    conn = _connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT username FROM script_access ORDER BY username")]
    finally:
        conn.close()


def script_allowed(db_path, username: str) -> bool:
    """True only when the username is in script_access AND the account's capabilities include deploy_scripts."""
    conn = _connect(db_path)
    try:
        if conn.execute("SELECT 1 FROM script_access WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone() is None:
            return False
        row = conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    principal = Principal(role_of(db_path, row[0]), row[0], str(username), "lan", "shader")
    return "deploy_scripts" in capabilities(principal, Policy.beside(db_path).get())


def _accounts_module():
    try:
        from meowdoku import accounts
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import accounts
    return accounts


def script_auth(db_path, script: str, username: str, password_file=None, prompt: bool = False) -> tuple:
    """(allowed, reason).  Verifies the password with accounts' own verification, checks script_allowed, writes a script_audit row."""
    import getpass
    import socket
    accounts = _accounts_module()
    reason = ""
    password = ""
    try:
        if password_file:
            password = Path(password_file).read_text(encoding="utf-8").strip()
        elif prompt:
            # Only an explicit --prompt may block on the console: on Windows
            # getpass reads the console directly even when stdin is not a tty.
            password = getpass.getpass(f"password for {username}: ")
        else:
            reason = "no credential: pass --password-file, or --prompt for an interactive login"
    except OSError:
        reason = "password file unreadable"
    if not reason:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT password_hash FROM users WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone()
        finally:
            conn.close()
        if row is None or not accounts._verify_password(password, row[0]):
            reason = "wrong username or password"
        elif not script_allowed(db_path, username):
            reason = "account is not allowed to run the deploy scripts"
    allowed = not reason
    conn = _connect(db_path)
    try:
        conn.execute("INSERT INTO script_audit (ts, username, script, allowed, host) VALUES (?, ?, ?, ?, ?)",
                     (time.time(), str(username), str(script), int(allowed), socket.gethostname()))
        conn.commit()
    finally:
        conn.close()
    return allowed, reason


# ---- [FEDERATE] peer grants: the credential a remote Nodus instance presents at /ws/contract/<prefix> (server.py) -------------
FRAME_KINDS = {"offer": 1, "accept": 2, "result": 3, "receipt": 4, "hello": 5, "submit": 6, "admitted": 7, "route": 8, "decision": 9}
DIRECTION_ROLE = {"host": 1, "engine": 2}          # the Nodus Hello role the bridge sends for the peer
_PEER_COLUMNS = ("grant_id", "peer", "prefix", "direction", "allowed_ops", "issued_by", "created_at", "expires_at", "revoked_at",
                 "service_user_id")


def _peer_ops(allowed_ops) -> str:
    """'result,receipt' / '3,4' / [3, 'receipt'] -> '3,4' (sorted frame-kind numbers).  Hello (5) is never grantable: the bridge sends it."""
    items = allowed_ops.split(",") if isinstance(allowed_ops, str) else list(allowed_ops)
    kinds = set()
    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        kind = FRAME_KINDS.get(text) if not text.isdigit() else int(text)
        if kind is None or not 1 <= kind <= 9:
            raise AuthzError(f"unknown frame kind {item!r} (kinds 1-9 or {', '.join(FRAME_KINDS)})", 400)
        if kind == 5:
            raise AuthzError("hello cannot be granted: the bridge sends the Hello itself", 400)
        kinds.add(kind)
    if not kinds:
        raise AuthzError("a grant needs at least one allowed frame kind", 400)
    return ",".join(str(k) for k in sorted(kinds))


def _secret_hash(secret: str) -> str:
    """The secret is 256 random bits (token_urlsafe(32)), so a salted-scrypt stretch buys nothing; sha256 is compared in constant time."""
    return hashlib.sha256(str(secret).encode("utf-8")).hexdigest()


def _peer_row(row) -> dict:
    return dict(zip(_PEER_COLUMNS, row))


def grant_peer(db_path, peer: str, prefix: str, direction: str, allowed_ops, days: float, issued_by: str,
               service_user=None) -> tuple[int, str]:
    """Issue a federation grant -> (grant_id, secret).  The secret is returned once; only its hash is stored.
    ``service_user`` (username or id; default: the account named ``peer``) must be a SERVICE-role account; its id becomes the Hello id."""
    if direction not in DIRECTION_ROLE:
        raise AuthzError("direction is 'host' (participant) or 'engine' (producer)", 400)
    if not str(peer).strip() or not str(prefix).strip():
        raise AuthzError("peer and prefix are required", 400)
    if not float(days) > 0:
        raise AuthzError("days must be positive", 400)
    ops = _peer_ops(allowed_ops)
    conn = _connect(db_path)
    try:
        who = peer if service_user is None else service_user
        if isinstance(who, int) or str(who).isdigit():
            row = conn.execute("SELECT id FROM users WHERE id = ?", (int(who),)).fetchone()
            if row is None:
                raise AuthzError(f"no such account id {who!r}", 404)
            uid = row[0]
        else:
            uid = _user_id(conn, who)
        role = (conn.execute("SELECT role FROM account_roles WHERE user_id = ?", (uid,)).fetchone() or ["member"])[0]
        if role != "service":
            raise AuthzError(f"account {uid} has role {role!r}; a peer grant needs a service account", 400)
        secret, now = secrets.token_urlsafe(32), time.time()
        cur = conn.execute("INSERT INTO peer_grants (peer, prefix, direction, allowed_ops, secret_hash, issued_by, created_at, "
                           "expires_at, service_user_id) VALUES (?,?,?,?,?,?,?,?,?)",
                           (str(peer), str(prefix), direction, ops, _secret_hash(secret), str(issued_by), now,
                            now + float(days) * 86400.0, uid))
        conn.commit()
        return cur.lastrowid, secret
    finally:
        conn.close()


def revoke_peer(db_path, grant_id) -> bool:
    """Revoke a grant (idempotent: the first revocation time stays).  False when there is no such grant."""
    conn = _connect(db_path)
    try:
        cur = conn.execute("UPDATE peer_grants SET revoked_at = COALESCE(revoked_at, ?) WHERE grant_id = ?", (time.time(), int(grant_id)))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def verify_peer(db_path, grant_id, secret, prefix, direction) -> dict | None:
    """The grant row (no secret hash) when the secret matches and the grant is live, unrevoked and for exactly this prefix and
    direction; else None (callers give no reason, so a probe learns nothing about which check failed)."""
    try:
        gid = int(grant_id)
    except (TypeError, ValueError):
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT " + ", ".join(_PEER_COLUMNS) + ", secret_hash FROM peer_grants WHERE grant_id = ?", (gid,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    grant, stored = _peer_row(row[:-1]), row[-1]
    ok = hmac.compare_digest(_secret_hash(secret if isinstance(secret, str) else ""), stored)
    if not ok or grant["revoked_at"] is not None or grant["expires_at"] <= time.time():
        return None
    if grant["prefix"] != prefix or grant["direction"] != direction:
        return None
    return grant


def list_peers(db_path) -> list[dict]:
    conn = _connect(db_path)
    try:
        return [_peer_row(r) for r in conn.execute("SELECT " + ", ".join(_PEER_COLUMNS) + " FROM peer_grants ORDER BY grant_id")]
    finally:
        conn.close()


# ---- CLI ------------------------------------------------------------------------------------------------------------------
MIN_CLI_PASSWORD = 12


def _cli_password(args, username: str) -> str:
    """The password for set-password / create-user: --password-file, or two getpass prompts (--prompt).  Never echoed, never printed."""
    import getpass
    if args.password_file:
        password = Path(args.password_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        password = getpass.getpass(f"new password for {username}: ")
        if getpass.getpass("again: ") != password:
            raise AuthzError("the two entries differ", 400)
    else:
        raise AuthzError("no credential: pass --prompt or --password-file", 400)
    if len(password) < MIN_CLI_PASSWORD:
        raise AuthzError(f"password must be at least {MIN_CLI_PASSWORD} characters", 400)
    return password


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="authz.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("set-password", "create-user"):
        p = sub.add_parser(name)
        p.add_argument("username")
        p.add_argument("--db", required=True)
        p.add_argument("--prompt", action="store_true", help="read the password with getpass (twice)")
        p.add_argument("--password-file", default=None)
        if name == "create-user":
            p.add_argument("--role", choices=("member", "operator", "admin"), required=True)
            p.add_argument("--granted-by", required=True, help="operator:<name>")
    for name in ("grant-admin", "revoke-admin"):
        p = sub.add_parser(name)
        p.add_argument("username")
        p.add_argument("--db", required=True)
        p.add_argument("--granted-by", required=True)
    p = sub.add_parser("script-grant")
    p.add_argument("username")
    p.add_argument("--db", required=True)
    p.add_argument("--granted-by", required=True)
    p = sub.add_parser("script-revoke")
    p.add_argument("username")
    p.add_argument("--db", required=True)
    p = sub.add_parser("script-list")
    p.add_argument("--db", required=True)
    p = sub.add_parser("script-auth")
    p.add_argument("script_name")
    p.add_argument("--user", required=True)
    p.add_argument("--password-file", default=None)
    p.add_argument("--prompt", action="store_true", help="ask for the password on the console")
    p.add_argument("--db", required=True)
    p = sub.add_parser("peer-grant", help="issue a federation grant; the secret is printed once, on stdout, on the SECRET line")
    p.add_argument("peer")
    p.add_argument("--db", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--direction", choices=sorted(DIRECTION_ROLE), required=True)
    p.add_argument("--ops", required=True, help="allowed frame kinds, comma list of numbers or names (not hello)")
    p.add_argument("--days", type=float, required=True)
    p.add_argument("--issued-by", required=True)
    p.add_argument("--service-user", default=None, help="the local SERVICE account whose id is the Hello id (default: an account named PEER)")
    p = sub.add_parser("peer-revoke")
    p.add_argument("grant_id", type=int)
    p.add_argument("--db", required=True)
    p = sub.add_parser("peer-list")
    p.add_argument("--db", required=True)
    p = sub.add_parser("schema", help="print the schema_version rows (module, version, updated_at, writer)")
    p.add_argument("--db", required=True)
    p = sub.add_parser("show")
    p.add_argument("username")
    p.add_argument("--db", required=True)
    p = sub.add_parser("caps")
    p.add_argument("--role", choices=ROLES, required=True)
    p.add_argument("--tier", choices=TIERS + ORIGINS, required=True,
                   help="a tier, or an origin name meaning that origin with access.py's default tier (lan edit, ip dev, dns shader)")
    p.add_argument("--origin", choices=ORIGINS, default=None)
    p.add_argument("--policy", default=None)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "set-password":
            accounts = _accounts_module()
            password = _cli_password(args, args.username)
            acc = accounts.Accounts(Path(args.db))
            try:
                acc.set_password(args.username, password)
            except accounts.AccountError as error:
                raise AuthzError(str(error), error.status)
            print(f"OK {args.username}")
        elif args.cmd == "create-user":
            accounts = _accounts_module()
            password = _cli_password(args, args.username)
            acc = accounts.Accounts(Path(args.db))
            try:
                acc.ensure_user(args.username, password)
                acc.set_password(args.username, password)
            except accounts.AccountError as error:
                raise AuthzError(str(error), error.status)
            grant_role(args.db, args.username, args.role, args.granted_by)
            print(f"OK {args.username}")
        elif args.cmd == "grant-admin":
            print(f"admin granted to {args.username} (user {grant_admin(args.db, args.username, args.granted_by)})")
        elif args.cmd == "revoke-admin":
            print(f"admin revoked from {args.username} (user {revoke_admin(args.db, args.username, args.granted_by)})")
        elif args.cmd == "script-grant":
            script_grant(args.db, args.username, args.granted_by)
            print(f"script access granted to {args.username}")
        elif args.cmd == "script-revoke":
            script_revoke(args.db, args.username)
            print(f"script access revoked from {args.username}")
        elif args.cmd == "script-list":
            for name in script_list(args.db):
                print(name)
        elif args.cmd == "script-auth":
            allowed, reason = script_auth(args.db, args.script_name, args.user, args.password_file, prompt=args.prompt)
            if not allowed:
                print(f"REFUSED {reason}", file=sys.stderr)
                return 3
            print(f"OK {args.user}")
        elif args.cmd == "peer-grant":
            gid, secret = grant_peer(args.db, args.peer, args.prefix, args.direction, args.ops, args.days, args.issued_by,
                                     service_user=args.service_user)
            print(f"grant {gid} issued to {args.peer} ({args.direction} on {args.prefix}); the secret is shown ONCE and cannot be recovered:")
            print(f"SECRET {secret}")
        elif args.cmd == "peer-revoke":
            print(f"grant {args.grant_id} revoked" if revoke_peer(args.db, args.grant_id) else f"no grant {args.grant_id}")
        elif args.cmd == "peer-list":
            for g in list_peers(args.db):
                state = "revoked" if g["revoked_at"] else ("expired" if g["expires_at"] <= time.time() else "live")
                print(f"{g['grant_id']} {g['peer']} {g['direction']} {g['prefix']} ops={g['allowed_ops']} svc={g['service_user_id']} "
                      f"expires={time.strftime('%Y-%m-%d', time.localtime(g['expires_at']))} {state}")
        elif args.cmd == "schema":
            conn = sqlite3.connect(str(args.db), timeout=10)
            try:
                for module, version, at, writer in _schema_version.report(conn):
                    print(f"{module} version={version} writer={writer} updated_at={at:.0f}")
            finally:
                conn.close()
        elif args.cmd == "show":
            conn = _connect(args.db)
            try:
                uid = _user_id(conn, args.username)
            finally:
                conn.close()
            print(f"{args.username} user_id={uid} role={role_of(args.db, uid)}")
        else:
            uid = None if args.role == "guest" else 0
            default_tier = {"lan": "edit", "dns": "shader", "ip": "dev"}
            if args.tier in ORIGINS:
                origin, tier = args.tier, default_tier[args.tier]
            else:
                origin, tier = args.origin or "dns", args.tier
            pr = Principal(args.role, uid, None if uid is None else "cli", origin, tier)
            for name in sorted(capabilities(pr, load_policy(args.policy))):
                print(name)
    except AuthzError as error:
        print(f"error: {error.message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

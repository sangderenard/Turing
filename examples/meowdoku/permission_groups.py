"""Ranked account authority, separate from owner-created content sharing groups.

The highest accepted permission group implies every lower group.  Legacy
account_roles remain authoritative direct grants during migration; ordinary
account_content_groups never enter this calculation.
"""
import sqlite3
import time
from pathlib import Path

try:
    from meowdoku import agent_execution
except ImportError:  # authz.py also runs directly for deployment script authorization
    import importlib.util
    _spec = importlib.util.spec_from_file_location("meowdoku_agent_execution", Path(__file__).with_name("agent_execution.py"))
    agent_execution = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(agent_execution)


RANKS = {"member": 10, "dev": 20, "operator": 30, "admin": 40}
ORDER = tuple(RANKS)
# intern (a guest agent's kiosk account, authz.KIOSK_ROLES) is not a rung: it does not imply member, and no group is granted to it.
KIOSK = "intern"
DDL = """CREATE TABLE IF NOT EXISTS account_permission_grants (
    user_id INTEGER NOT NULL REFERENCES users(id),
    group_name TEXT NOT NULL CHECK(group_name IN ('dev','operator')),
    granted_by INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    PRIMARY KEY(user_id, group_name)
)"""
AUDIT_DDL = """CREATE TABLE IF NOT EXISTS account_permission_audit (
    id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
    group_name TEXT NOT NULL, enabled INTEGER NOT NULL, created_at REAL NOT NULL
)"""
MODIFIER_DDL = """CREATE TABLE IF NOT EXISTS account_capability_modifiers (
    user_id INTEGER NOT NULL REFERENCES users(id),
    capability TEXT NOT NULL CHECK(capability IN ('message')),
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    actor_id INTEGER NOT NULL REFERENCES users(id),
    updated_at REAL NOT NULL,
    PRIMARY KEY(user_id, capability)
)"""
MODIFIER_AUDIT_DDL = """CREATE TABLE IF NOT EXISTS account_capability_modifier_audit (
    id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
    capability TEXT NOT NULL, enabled INTEGER NOT NULL, created_at REAL NOT NULL
)"""
AGENT_EXEC_DDL = """CREATE TABLE IF NOT EXISTS account_agent_execution (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    actor_id INTEGER NOT NULL REFERENCES users(id),
    updated_at REAL NOT NULL
)"""
NODUS_DDL = """CREATE TABLE IF NOT EXISTS account_nodus_grants (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    granted_by INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL
)"""


def prepare(db):
    db.execute(DDL)
    db.execute(AUDIT_DDL)
    db.execute(NODUS_DDL)
    db.execute(MODIFIER_DDL)
    db.execute(MODIFIER_AUDIT_DDL)
    db.execute(AGENT_EXEC_DDL)


def authority(db, user_id):
    """Return direct groups, cumulative groups, and the highest ranked group.

    A service account keeps its separate service capability role and uses the
    server paper credential; it is not a rung in human account authority.
    An intern account is not a rung either: its highest group is intern, with
    no implied member group and no dev/operator/nodus grants.
    """
    if user_id is None:
        return {"highest": None, "direct": [], "groups": [], "service": False, "modifiers": {}}
    prepare(db)
    role_row = db.execute("SELECT role FROM account_roles WHERE user_id=?", (user_id,)).fetchone()
    role = role_row[0] if role_row else "member"
    modifiers = {cap: bool(enabled) for cap, enabled in db.execute(
        "SELECT capability, enabled FROM account_capability_modifiers WHERE user_id=?", (user_id,))}
    execution = db.execute("SELECT enabled FROM account_agent_execution WHERE user_id=?", (user_id,)).fetchone()
    if execution is not None:
        modifiers["execution"] = bool(execution[0])
    if role == KIOSK:
        return {"highest": KIOSK, "direct": [KIOSK], "groups": [KIOSK], "service": False, "modifiers": modifiers}
    direct = {"member"}
    if role in RANKS:
        direct.add(role)
    direct.update(row[0] for row in db.execute(
        "SELECT group_name FROM account_permission_grants WHERE user_id=?", (user_id,)))
    highest = max(direct, key=RANKS.__getitem__)
    groups = [name for name in ORDER if RANKS[name] <= RANKS[highest]]
    nodus = db.execute("SELECT 1 FROM account_nodus_grants WHERE user_id=?", (user_id,)).fetchone() is not None
    name = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()[0]
    db_path = db.execute("PRAGMA database_list").fetchone()[2]
    execution_group = agent_execution.server_owned(db_path, name) and modifiers.get("execution", True)
    return {"highest": highest, "direct": [*sorted(direct, key=RANKS.__getitem__), *(["nodus"] if nodus else []),
                                           *(["execution"] if execution_group else [])],
            "groups": [*groups, *(["nodus"] if nodus else []), *(["execution"] if execution_group else [])],
            "service": role == "service", "modifiers": modifiers}


def set_grant(db, actor_id, user_id, group_name, enabled):
    """Admin-only caller sets a group grant; admin uses grant_admin."""
    if group_name not in ("dev", "operator", "nodus", "message", "execution") or type(enabled) is not bool:
        raise ValueError("expected dev, operator, nodus, message or execution and boolean enabled")
    actor = authority(db, actor_id)
    if actor["highest"] != "admin":
        raise PermissionError("admin permission group required")
    if db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
        raise LookupError("account not found")
    if group_name == "message":
        return set_modifier(db, actor_id, user_id, group_name, enabled)
    if group_name == "execution":
        name = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()[0]
        db_path = db.execute("PRAGMA database_list").fetchone()[2]
        if not agent_execution.server_owned(db_path, name):
            raise PermissionError("execution is for installed server-owned agents except Llama")
        if authority(db, user_id)["highest"] == KIOSK:
            raise PermissionError("an intern account cannot join the execution group")
        db.execute("INSERT INTO account_agent_execution(user_id,enabled,actor_id,updated_at) VALUES(?,?,?,?) "
                   "ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,actor_id=excluded.actor_id,"
                   "updated_at=excluded.updated_at", (user_id, int(enabled), actor_id, time.time()))
        if not enabled and db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_execution_jobs'").fetchone():
            db.execute("UPDATE agent_execution_jobs SET status='failed',output='execution permission removed',finished=? "
                       "WHERE user_id=? AND status='queued'", (time.time(), user_id))
        db.execute("INSERT INTO account_capability_modifier_audit(actor_id,user_id,capability,enabled,created_at) "
                   "VALUES(?,?,?,?,?)", (actor_id, user_id, group_name, int(enabled), time.time()))
        return authority(db, user_id)
    if enabled and authority(db, user_id)["highest"] == KIOSK:
        raise PermissionError("an intern account takes no permission group; change its role instead")
    if group_name == "nodus":
        if enabled:
            db.execute("INSERT INTO account_nodus_grants(user_id,granted_by,created_at) VALUES(?,?,?) "
                       "ON CONFLICT(user_id) DO NOTHING", (user_id, actor_id, time.time()))
        else:
            db.execute("DELETE FROM account_nodus_grants WHERE user_id=?", (user_id,))
    elif enabled:
        db.execute("INSERT INTO account_permission_grants(user_id,group_name,granted_by,created_at) "
                   "VALUES(?,?,?,?) ON CONFLICT(user_id,group_name) DO NOTHING",
                   (user_id, group_name, actor_id, time.time()))
    else:
        db.execute("DELETE FROM account_permission_grants WHERE user_id=? AND group_name=?",
                   (user_id, group_name))
    db.execute("INSERT INTO account_permission_audit(actor_id,user_id,group_name,enabled,created_at) VALUES(?,?,?,?,?)",
               (actor_id, user_id, group_name, int(enabled), time.time()))
    return authority(db, user_id)


def set_modifier(db, actor_id, user_id, capability, enabled):
    """Admin-audited, account-specific capability override. Only DM for interns is grantable."""
    if capability != "message" or type(enabled) is not bool:
        raise ValueError("expected message and boolean enabled")
    if authority(db, actor_id)["highest"] != "admin":
        raise PermissionError("admin permission group required")
    if db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
        raise LookupError("account not found")
    target = authority(db, user_id)
    if target["highest"] != KIOSK:
        raise PermissionError("message modifier is reserved for intern accounts")
    db.execute("INSERT INTO account_capability_modifiers(user_id,capability,enabled,actor_id,updated_at) "
               "VALUES(?,?,?,?,?) ON CONFLICT(user_id,capability) DO UPDATE SET enabled=excluded.enabled, "
               "actor_id=excluded.actor_id, updated_at=excluded.updated_at",
               (user_id, capability, int(enabled), actor_id, time.time()))
    db.execute("INSERT INTO account_capability_modifier_audit(actor_id,user_id,capability,enabled,created_at) "
               "VALUES(?,?,?,?,?)", (actor_id, user_id, capability, int(enabled), time.time()))
    return authority(db, user_id)


def paper_credential_owner(db, user_id, shared_owner_id):
    """Resolve the paper key holder from permission authority, never a request body."""
    effective = authority(db, user_id)
    if effective["highest"] is None:
        raise PermissionError("sign in to use paper trading")
    if effective["highest"] == "admin" or effective["service"]:
        if type(shared_owner_id) is not int or shared_owner_id < 1:
            raise RuntimeError("shared paper account is not configured")
        return shared_owner_id
    return user_id

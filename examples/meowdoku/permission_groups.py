"""Ranked account authority, separate from owner-created content sharing groups.

The highest accepted permission group implies every lower group.  Legacy
account_roles remain authoritative direct grants during migration; ordinary
account_content_groups never enter this calculation.
"""
import sqlite3
import time


RANKS = {"member": 10, "dev": 20, "operator": 30, "admin": 40}
ORDER = tuple(RANKS)
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


def prepare(db):
    db.execute(DDL)
    db.execute(AUDIT_DDL)


def authority(db, user_id):
    """Return direct groups, cumulative groups, and the highest ranked group.

    A service account keeps its separate service capability role and uses the
    server paper credential; it is not a rung in human account authority.
    """
    if user_id is None:
        return {"highest": None, "direct": [], "groups": [], "service": False}
    prepare(db)
    role_row = db.execute("SELECT role FROM account_roles WHERE user_id=?", (user_id,)).fetchone()
    role = role_row[0] if role_row else "member"
    direct = {"member"}
    if role in RANKS:
        direct.add(role)
    direct.update(row[0] for row in db.execute(
        "SELECT group_name FROM account_permission_grants WHERE user_id=?", (user_id,)))
    highest = max(direct, key=RANKS.__getitem__)
    groups = [name for name in ORDER if RANKS[name] <= RANKS[highest]]
    return {"highest": highest, "direct": sorted(direct, key=RANKS.__getitem__),
            "groups": groups, "service": role == "service"}


def set_grant(db, actor_id, user_id, group_name, enabled):
    """Admin-only caller sets a dev/operator grant; admin uses grant_admin."""
    if group_name not in ("dev", "operator") or type(enabled) is not bool:
        raise ValueError("expected dev or operator and boolean enabled")
    actor = authority(db, actor_id)
    if actor["highest"] != "admin":
        raise PermissionError("admin permission group required")
    if db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
        raise LookupError("account not found")
    if enabled:
        db.execute("INSERT INTO account_permission_grants(user_id,group_name,granted_by,created_at) "
                   "VALUES(?,?,?,?) ON CONFLICT(user_id,group_name) DO NOTHING",
                   (user_id, group_name, actor_id, time.time()))
    else:
        db.execute("DELETE FROM account_permission_grants WHERE user_id=? AND group_name=?",
                   (user_id, group_name))
    db.execute("INSERT INTO account_permission_audit(actor_id,user_id,group_name,enabled,created_at) VALUES(?,?,?,?,?)",
               (actor_id, user_id, group_name, int(enabled), time.time()))
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

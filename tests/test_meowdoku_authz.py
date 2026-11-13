import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

AUTHZ = Path(__file__).resolve().parents[1] / "examples" / "meowdoku" / "authz.py"
spec = importlib.util.spec_from_file_location("meowdoku_authz", AUTHZ)
authz = importlib.util.module_from_spec(spec)
sys.modules["meowdoku_authz"] = authz
spec.loader.exec_module(authz)

USERS_DDL = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL
            );
"""
ALL = set(authz.CAPS) - {"compile_unstable", "nodus_edit"}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "accounts.db"
    conn = sqlite3.connect(path)
    conn.executescript(USERS_DDL)
    conn.executescript(authz._DDL)
    conn.execute("INSERT INTO users (username, password_hash, display_name, created_at) VALUES ('alice','x','alice',0)")
    conn.execute("INSERT INTO users (username, password_hash, display_name, created_at) VALUES ('bob','x','bob',0)")
    conn.commit()
    conn.close()
    return path


def P(role, origin="dns", tier="shader"):
    uid = None if role == "guest" else 1
    return authz.Principal(role, uid, None if uid is None else "u", origin, tier)


def test_guest_dns():
    assert authz.capabilities(P("guest"), authz.DEFAULT_POLICY) == {"view_site", "photo_read"}


def test_member_dns():
    assert authz.capabilities(P("member"), authz.DEFAULT_POLICY) == {"view_site", "photo_read", "studio_write", "photo_submit", "share_live", "compile_submit", "message"}


def test_admin_dns_no_lifecycle():
    assert authz.capabilities(P("admin"), authz.DEFAULT_POLICY) == ALL - {"lifecycle_admin"}


def test_admin_lan_everything():
    assert authz.capabilities(P("admin", "lan", "edit"), authz.DEFAULT_POLICY) == ALL


def test_member_lan_edit_has_no_admin_caps():
    caps = authz.capabilities(P("member", "lan", "edit"), authz.DEFAULT_POLICY)
    assert caps == ALL - {"lifecycle_admin", "account_manage", "deploy_scripts", "federate", "core_git", "paper_train", "mail_send"}


def test_higher_permission_group_inherits_lower_capabilities():
    dev = authz.capabilities(P("dev"), authz.DEFAULT_POLICY)
    operator = authz.capabilities(P("operator"), authz.DEFAULT_POLICY)
    admin = authz.capabilities(P("admin"), authz.DEFAULT_POLICY)
    assert authz.capabilities(P("member"), authz.DEFAULT_POLICY) <= dev <= operator <= admin
    assert "dev_ui" in operator


def test_policy_override_after_mtime_bump(tmp_path):
    f = tmp_path / "authz_policy.json"
    pol = authz.Policy(f)
    g = P("guest")
    assert authz.capabilities(g, pol.get()) == {"view_site", "photo_read"}
    f.write_text(json.dumps({"tier_caps": {"shader": ["view_site", "play"]}}), encoding="utf-8")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert authz.capabilities(g, pol.get()) == {"view_site", "play"}


def test_grant_revoke_roundtrip(db):
    uid = authz.grant_admin(db, "alice", "operator:abcd1234")
    assert authz.role_of(db, uid) == "admin"
    assert authz.role_of(db, None) == "guest"
    with pytest.raises(authz.AuthzError):
        authz.grant_admin(db, "bob", "mallory")
    authz.grant_admin(db, "bob", "alice")
    authz.revoke_admin(db, "bob", "alice")
    assert authz.role_of(db, 2) == "member"
    authz.revoke_admin(db, "alice", "operator:abcd1234")
    assert authz.role_of(db, uid) == "member"


def test_bootstrap_requires_operator(db):
    with pytest.raises(authz.AuthzError) as e:
        authz.grant_admin(db, "alice", "alice")
    assert e.value.status == 403


def test_principal_from(db):
    authz.grant_admin(db, "alice", "operator:abcd")
    assert authz.principal_from(None, "dns", "shader", db).role == "guest"
    pr = authz.principal_from({"id": 1, "username": "alice"}, "lan", "edit", db)
    assert (pr.role, pr.username) == ("admin", "alice")


def test_permission_group_rank_and_cumulative_membership(db):
    assert authz.role_of(db, 2) == "member"
    authz.grant_admin(db, "alice", "operator:abcd")
    with authz._connect(db) as conn:
        groups = authz._permission_groups
        dev = groups.set_grant(conn, 1, 2, "dev", True)
        assert dev["highest"] == "dev"
        assert dev["groups"] == ["member", "dev"]
        operator = groups.set_grant(conn, 1, 2, "operator", True)
        assert operator["highest"] == "operator"
        assert operator["groups"] == ["member", "dev", "operator"]
        assert conn.execute("SELECT COUNT(*) FROM account_permission_audit WHERE actor_id=1 AND user_id=2").fetchone()[0] == 2
    assert authz.role_of(db, 2) == "operator"
    with authz._connect(db) as conn:
        groups.set_grant(conn, 1, 2, "operator", False)
    assert authz.role_of(db, 2) == "dev"
    with authz._connect(db) as conn:
        groups.set_grant(conn, 1, 2, "dev", False)
    assert authz.role_of(db, 2) == "member"


def test_permission_groups_require_admin_and_ignore_flat_sharing_groups(db):
    with authz._connect(db) as conn:
        groups = authz._permission_groups
        conn.execute("CREATE TABLE account_content_groups (id TEXT, owner_id INTEGER, name TEXT)")
        conn.execute("INSERT INTO account_content_groups VALUES ('place', 2, 'admin')")
        assert groups.authority(conn, 2)["highest"] == "member"
        with pytest.raises(PermissionError):
            groups.set_grant(conn, 2, 1, "dev", True)
        with pytest.raises(ValueError):
            groups.set_grant(conn, 2, 1, "admin", True)


def test_service_is_separate_from_human_rank(db):
    authz.grant_role(db, "bob", "service", "supervisor")
    with authz._connect(db) as conn:
        effective = authz._permission_groups.authority(conn, 2)
    assert effective["service"] is True
    assert effective["highest"] == "member"
    assert authz.role_of(db, 2) == "service"


def test_paper_credential_boundary_uses_rank_and_service_status(db):
    groups = authz._permission_groups
    with authz._connect(db) as conn:
        assert groups.paper_credential_owner(conn, 2, None) == 2
        with pytest.raises(PermissionError):
            groups.paper_credential_owner(conn, None, 1)
    authz.grant_admin(db, "alice", "operator:abcd")
    with authz._connect(db) as conn:
        groups.set_grant(conn, 1, 2, "dev", True)
        assert groups.paper_credential_owner(conn, 2, 1) == 2
        assert groups.paper_credential_owner(conn, 1, 1) == 1
    authz.grant_role(db, "bob", "service", "supervisor")
    with authz._connect(db) as conn:
        assert groups.paper_credential_owner(conn, 2, 1) == 1
        with pytest.raises(RuntimeError):
            groups.paper_credential_owner(conn, 2, None)


def test_nodus_grant_is_admin_controlled_and_independent_of_rank(db):
    authz.grant_admin(db, "alice", "operator:abcd")
    with authz._connect(db) as conn:
        groups = authz._permission_groups
        with pytest.raises(PermissionError):
            groups.set_grant(conn, 2, 2, "nodus", True)
        granted = groups.set_grant(conn, 1, 2, "nodus", True)
        assert granted["highest"] == "member"
        assert granted["groups"] == ["member", "nodus"]
    member = authz.principal_from({"id": 2, "username": "bob"}, "dns", "shader", db)
    assert "nodus_edit" in authz.capabilities(member, authz.DEFAULT_POLICY)
    with authz._connect(db) as conn:
        groups.set_grant(conn, 1, 2, "nodus", False)
    member = authz.principal_from({"id": 2, "username": "bob"}, "dns", "shader", db)
    assert "nodus_edit" not in authz.capabilities(member, authz.DEFAULT_POLICY)


def test_require_raises_403_with_cap():
    with pytest.raises(authz.AuthzError) as e:
        authz.require(P("guest"), "studio_write")
    assert e.value.status == 403 and e.value.message == "this needs studio_write"
    authz.require(P("member"), "studio_write")


def test_cli_caps_sorted():
    out = subprocess.run([sys.executable, str(AUTHZ), "caps", "--role", "admin", "--tier", "dns"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == sorted(ALL - {"lifecycle_admin"})

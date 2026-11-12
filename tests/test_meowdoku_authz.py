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
ALL = set(authz.CAPS) - {"compile_unstable"}


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
    assert caps == ALL - {"lifecycle_admin", "account_manage", "deploy_scripts", "federate", "core_git"}


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


def test_require_raises_403_with_cap():
    with pytest.raises(authz.AuthzError) as e:
        authz.require(P("guest"), "studio_write")
    assert e.value.status == 403 and e.value.message == "this needs studio_write"
    authz.require(P("member"), "studio_write")


def test_cli_caps_sorted():
    out = subprocess.run([sys.executable, str(AUTHZ), "caps", "--role", "admin", "--tier", "dns"],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == sorted(ALL - {"lifecycle_admin"})

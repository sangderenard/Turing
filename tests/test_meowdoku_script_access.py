import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

MEOW = Path(__file__).resolve().parents[1] / "examples" / "meowdoku"
AUTHZ = MEOW / "authz.py"
OPERATORS = MEOW / "deploy" / "operators.py"
USERS = "admin,codex,claude"


def run(script, *args, **kw):
    env = dict(os.environ, PYTHONUTF8="1")
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, env=env, timeout=90, **kw)


@pytest.fixture
def boot(tmp_path):
    db, sec = tmp_path / "accounts.db", tmp_path / "creds"
    r = run(OPERATORS, "bootstrap", "--db", db, "--users", USERS, "--secrets-dir", sec)
    assert r.returncode == 0, r.stderr
    return db, sec, r


def auth(db, user, pwfile, script="Status.ps1"):
    return run(AUTHZ, "script-auth", script, "--user", user, "--password-file", pwfile, "--db", db)


def test_bootstrap_roles_access_secrets(boot):
    db, sec, r = boot
    conn = sqlite3.connect(db)
    roles = dict(conn.execute("SELECT u.username, r.role FROM account_roles r JOIN users u ON u.id = r.user_id"))
    assert roles == {"admin": "admin", "codex": "operator", "claude": "operator"}
    assert {x[0] for x in conn.execute("SELECT username FROM script_access")} == set(USERS.split(","))
    secrets_text = []
    for name in USERS.split(","):
        f = sec / f"{name}.secret"
        assert f.exists()
        secrets_text.append(f.read_text())
    assert all(t not in r.stdout + r.stderr for t in secrets_text)
    before = [(sec / f"{n}.secret").read_text() for n in USERS.split(",")]
    r2 = run(OPERATORS, "bootstrap", "--db", db, "--users", USERS, "--secrets-dir", sec)
    assert r2.returncode == 0, r2.stderr
    assert before == [(sec / f"{n}.secret").read_text() for n in USERS.split(",")]
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == len(USERS.split(","))


def test_auth_ok_and_audit(boot):
    db, sec, _ = boot
    for name in ("admin", "codex"):
        r = auth(db, name, sec / f"{name}.secret")
        assert r.returncode == 0 and r.stdout.strip() == f"OK {name}", r.stderr
    rows = sqlite3.connect(db).execute("SELECT username, script, allowed FROM script_audit ORDER BY ts").fetchall()
    assert rows == [("admin", "Status.ps1", 1), ("codex", "Status.ps1", 1)]


def test_refusals(boot):
    db, sec, _ = boot
    sys.path.insert(0, str(MEOW.parent))
    from meowdoku import accounts as accounts_mod
    accounts_mod.Accounts(db).ensure_user("plainmember", "member-password-1")
    pw = sec / "plainmember.secret"
    pw.write_text("member-password-1")
    r = auth(db, "plainmember", pw)                       # member, not in script_access
    assert r.returncode == 3 and r.stderr.startswith("REFUSED")
    run(AUTHZ, "script-grant", "plainmember", "--granted-by", "admin", "--db", db)
    r = auth(db, "plainmember", pw)                       # in script_access but role lacks deploy_scripts
    assert r.returncode == 3 and "not allowed" in r.stderr
    from meowdoku import authz
    authz.grant_role(db, "claude", "member", "admin")     # demote an operator that is still in script_access
    r = auth(db, "claude", sec / "claude.secret")
    assert r.returncode == 3 and "not allowed" in r.stderr
    bad = sec / "bad.secret"
    bad.write_text("not-the-password")
    r = auth(db, "codex", bad)                            # wrong password
    assert r.returncode == 3 and "wrong username or password" in r.stderr
    assert "not-the-password" not in r.stdout + r.stderr
    r = auth(db, "codex", sec / "missing.secret")
    assert r.returncode == 3
    allowed = [a for (a,) in sqlite3.connect(db).execute("SELECT allowed FROM script_audit")]
    assert allowed == [0, 0, 0, 0, 0]


def test_rotate_invalidates_old_secret(boot):
    db, sec, _ = boot
    old = (sec / "codex.secret").read_text()
    r = run(OPERATORS, "rotate", "codex", "--db", db, "--secrets-dir", sec)
    assert r.returncode == 0 and old not in r.stdout
    assert (sec / "codex.secret").read_text() != old
    assert auth(db, "codex", sec / "codex.secret").returncode == 0

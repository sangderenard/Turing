"""state_migrate: legacy build/meowdoku tree -> StatePaths root, copy / idempotence / verify.  Temp trees only."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import accounts, authz, job_library, state_migrate  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402


def _rows(path):
    db = sqlite3.connect(str(path))
    try:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        return {n: db.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        db.close()


@pytest.fixture
def legacy(tmp_path):
    base = tmp_path / "legacy"
    base.mkdir()
    acc = accounts.Accounts(base / "accounts.db")           # stays open: the WAL sidecars exist, as on the live db
    acc.ensure_user("alice", "pw-alice-123")
    authz._connect(base / "accounts.db").close()
    with job_library.connect(base / "marble_photos") as db:
        db.execute("INSERT INTO job_owners VALUES ('d1', 'alice', 1.0)")
    (base / "profiles.json").write_text('{"alice": {}}', encoding="utf-8")
    (base / "marble_photos" / "abc").mkdir(parents=True, exist_ok=True)
    (base / "marble_photos" / "abc" / "config.json").write_text("{}", encoding="utf-8")
    (base / "marble_photos" / "abc" / "status.json").write_text('{"s": 1}', encoding="utf-8")
    (base / "tls").mkdir()
    (base / "tls" / "cert.pem").write_text("dummy cert", encoding="utf-8")
    (base / "admin_token.txt").write_text("tok", encoding="utf-8")
    (base / "operator_credentials").mkdir()
    (base / "operator_credentials" / "x.secret").write_text("s3cret", encoding="utf-8")
    (base / "logs").mkdir()
    (base / "logs" / "a.log").write_text("line\n", encoding="utf-8")
    (base / "canvas").mkdir()
    (base / "canvas" / "m.canvas").write_text("CANVAS V4", encoding="utf-8")
    yield base, acc
    acc._conn.close()


def test_plan_copy_verify_idempotent_and_tamper(legacy, tmp_path):
    base, _acc = legacy
    root = tmp_path / "state"
    new = StatePaths(root)
    steps = state_migrate.plan(base, root)
    before = {p: p.stat().st_size for p in base.rglob("*") if p.is_file()}

    r = state_migrate.execute(steps, "copy-verify", root=root, restrict=False)
    assert r["verify_ok"] is True
    for dst in (new.accounts_db, new.job_library_db, new.profiles_json, new.marble_photos_dir / "abc" / "config.json",
                new.marble_photos_dir / "abc" / "status.json", new.tls_dir / "cert.pem", new.admin_token(),
                new.operator_credentials_dir / "x.secret", new.logs_dir / "a.log", new.canvas_dir / "m.canvas"):
        assert dst.is_file(), dst
    assert not (new.marble_photos_dir / "job-library.sqlite").exists()           # the db goes to sql/, not as a byte copy
    assert _rows(base / "accounts.db") == _rows(new.accounts_db) and _rows(new.accounts_db)["users"] >= 1
    assert "schema_version" in _rows(new.accounts_db)
    assert _rows(base / "marble_photos" / "job-library.sqlite") == _rows(new.job_library_db)

    receipt = json.loads(Path(r["receipt_path"]).read_text(encoding="utf-8"))
    assert Path(r["receipt_path"]).parent == new.run_dir
    assert receipt["count"] == len(steps) and len(receipt["files"]) == len(steps)
    assert all(len(f["sha256"]) == 64 and "size" in f and f["verify"]["ok"] for f in receipt["files"])
    assert {f["rel"] for f in receipt["files"]} >= {"sql/accounts.db", "sql/job-library.sqlite", "secrets/tls/cert.pem", "canvas/m.canvas"}
    assert before == {p: p.stat().st_size for p in base.rglob("*") if p.is_file() and p in before}     # sources untouched

    again = state_migrate.execute(state_migrate.plan(base, root), "copy", root=root, restrict=False)
    assert all(f["action"] == "skipped-identical" for f in again["files"])

    (new.profiles_json).write_text('{"alice": {"x": 1}}', encoding="utf-8")        # destination altered
    bad = state_migrate.execute(state_migrate.plan(base, root), "verify", root=root, restrict=False)
    assert bad["verify_ok"] is False
    assert [f["rel"] for f in bad["files"] if not f["verify"]["ok"]] == ["content/profiles.json"]
    with pytest.raises(state_migrate.MigrationError):                                # and copy refuses to overwrite it
        state_migrate.execute(state_migrate.plan(base, root), "copy", root=root, restrict=False)
    assert new.profiles_json.read_text(encoding="utf-8") == '{"alice": {"x": 1}}'

"""Targeted test: scan records and a dream work table sealed into an account
workspace through the service's real store, on a fixture accounts database."""

import importlib.util
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.compiler.dream_document import parse_dream_document
from src.compiler.repo_world_scan import scan_repository
from src.compiler.repo_world_sink import (
    bundle_revision, read_bundle, scan_bundle, submit, work_table_bundle,
)

_STORE = Path(__file__).resolve().parents[1] / "examples/meowdoku/workspace_accounting.py"
_spec = importlib.util.spec_from_file_location("workspace_accounting", _STORE)
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)

DREAM = (
    "/*@turing.segment.v1\nid = want\nlanguage = prose\nauthor = albert\n@end*/\n"
    "A function returning the larger of two numbers.\n/*@turing.end*/\n"
    "/*@turing.segment.v1\nid = got\nlanguage = python\nsatisfies = want\nauthor = claude\n@end*/\n"
    "def larger(a, b):\n    return a if a >= b else b\n/*@turing.end*/\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "accounts.db"
    conn = sqlite3.connect(db)
    with conn:
        conn.executescript(
            "CREATE TABLE users(id INTEGER PRIMARY KEY); INSERT INTO users VALUES(1),(2),(3);"
            "CREATE TABLE account_roles(user_id INTEGER PRIMARY KEY, role TEXT);"
            "INSERT INTO account_roles VALUES(3,'admin');"
        )
    conn.close()
    with patch.object(w.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(10 ** 8, 0, 10 ** 8)):
        yield w.Store(db, tmp_path, capacity_bytes=4_000_000, user_quota_bytes=2_000_000, free_floor_bytes=100)


def _fixture_scan(tmp_path):
    repo = tmp_path / "repo"
    (repo / "plans").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("import b\n", encoding="utf-8")
    (repo / "b.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "plans" / "larger.dream").write_bytes(DREAM.encode("utf-8"))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return scan_repository(repo)


def test_scan_summary_and_claims_seal_into_the_actors_workspace(store, tmp_path):
    scan = _fixture_scan(tmp_path)
    bundle = scan_bundle(scan)
    assert set(bundle) == {"summary.json", "claims.json"}

    receipt = submit(store, 1, bundle)

    lease = store.get(1, receipt.workspace_id)
    assert lease["state"] == "active" and lease["base_revision"] == bundle_revision(bundle)
    assert receipt.base_revision == bundle_revision(bundle)
    ticket = store.get_ticket(1, receipt.ticket_id)
    assert ticket["status"] == "open" and ticket["snapshot_bytes"] == receipt.snapshot_bytes
    assert receipt.files == ("claims.json", "summary.json")

    stored = read_bundle(store, 1, receipt.ticket_id)
    assert stored == bundle
    summary = json.loads(stored["summary.json"])
    assert summary["commit"] == scan.commit
    assert summary["counts"] == {
        "regions": 2, "rooms": 3, "portal_pairs": 1, "unresolved_references": 0, "claimed_rooms": 1,
    }
    (claim,) = json.loads(stored["claims.json"])
    assert claim["stand_in"] == "want" and claim["owner"] == "claude" and claim["state"] == "satisfied"

    # The same bundle again is the same ticket (the store's idempotency), and
    # the records belong to the actor alone.
    again = submit(store, 1, bundle, workspace_id=receipt.workspace_id)
    assert again.ticket_id == receipt.ticket_id
    with pytest.raises(w.NotFound):
        store.read_snapshot(2, receipt.ticket_id)
    assert store.list_owned(2) == []


def test_full_scan_is_stored_only_when_asked_and_only_if_it_fits(store, tmp_path):
    scan = _fixture_scan(tmp_path)
    full = scan_bundle(scan, full=True)
    assert set(full) == {"summary.json", "claims.json", "scan.json"}
    receipt = submit(store, 1, full)
    assert json.loads(read_bundle(store, 1, receipt.ticket_id)["scan.json"])["commit"] == scan.commit

    # Over the store's bundle cap: refused by the store before anything is kept.
    oversize = {"scan.json": b"x" * (w.MAX_SNAPSHOT_BYTES + 1)}
    with pytest.raises(ValueError, match="1 MiB"):
        submit(store, 1, oversize)
    assert all(lease["measured_bytes"] <= receipt.snapshot_bytes for lease in store.list_owned(1))


def test_work_table_bundle_and_store_refusals(store):
    document = parse_dream_document(DREAM)
    bundle = work_table_bundle(document, source="plans/larger.dream")
    receipt = submit(store, 1, bundle)
    table = json.loads(read_bundle(store, 1, receipt.ticket_id)["work_table.json"])
    assert table["source"] == "plans/larger.dream"
    assert table["stand_ins"][0]["owner"] == "claude"
    assert table["edges"] == [{"from": "want", "to": "got", "role": "satisfied-by", "author": "claude"}]

    with pytest.raises(w.Refused, match="authenticated account"):
        submit(store, 99, bundle)
    with pytest.raises(ValueError, match="empty bundle"):
        submit(store, 1, {})

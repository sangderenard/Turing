"""[BATCH] private_record notes are written in batches, not one connection and one commit per occurrence (2026-11-18).

Ring._load re-redacted every saved line and committed once per address found: 60 s on an idle disk, minutes in a restart.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import private_record as P  # noqa: E402


def _record(tmp_path):
    return P.Record(b"k" * 32, tmp_path / "private_record.db")


def _rows(tmp_path):
    with sqlite3.connect(tmp_path / "private_record.db") as conn:
        return {r[0]: (r[1], r[2]) for r in conn.execute("SELECT token, value, seen FROM private_record")}


def test_a_batch_is_one_transaction_and_counts_every_occurrence(tmp_path, monkeypatch):
    record = _record(tmp_path)
    commits = []
    real = record._connect

    def counting():
        conn = real()
        commits.append(conn)
        return conn
    monkeypatch.setattr(record, "_connect", counting)
    with record.batch():
        for _ in range(500):
            record.redact("from 10.1.2.3 and 10.9.9.9 and a@b.example")
    assert len(commits) == 1, "the whole reload is one connection and one transaction"
    rows = _rows(tmp_path)
    assert sorted(v[1] for v in rows.values()) == [500, 500, 500], "every occurrence still counted"
    assert {v[0] for v in rows.values()} == {"10.1.2.3", "10.9.9.9", "a@b.example"}, "the raw value is kept beside its token"


def test_a_new_token_is_written_at_once_and_repeats_wait_for_the_flush(tmp_path):
    record = _record(tmp_path)
    token = record.token("ip", "192.0.2.7")
    assert _rows(tmp_path)[token][1] == 1, "a new token reaches the record at once"
    for _ in range(9):
        record.token("ip", "192.0.2.7")
    assert _rows(tmp_path)[token][1] == 1, "repeats inside FLUSH_SECONDS are held in memory"
    assert record.resolve([token])[token]["seen"] == 10, "reading the record back flushes first"
    record.token("ip", "192.0.2.7")
    record.flush()
    assert _rows(tmp_path)[token][1] == 11


def test_without_a_database_nothing_is_recorded_and_tokens_still_replace_values(tmp_path):
    record = P.Record(b"k" * 32)
    with record.batch():
        text = record.redact("10.1.2.3")
    assert text.startswith("ip~") and "10.1.2.3" not in text
    assert not (tmp_path / "private_record.db").exists()

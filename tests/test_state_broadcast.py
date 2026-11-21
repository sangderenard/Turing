"""Regional state broadcast (examples/meowdoku/state_broadcast.py, ticket e2f19532): one writer, enrolled readers.

One writer region packs a generation of a small state root; an enrolled reader accepts it and can take over from it;
everything that must refuse does: an unenrolled region, a tampered chunk, a replayed generation, an unpinned or
impostor writer, a take-over without the writer stopped. The writer's own region keys never travel.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "meowdoku"))
pytest.importorskip("cryptography")

import state_broadcast as sb  # noqa: E402


def _state(root: Path) -> Path:
    (root / "sql").mkdir(parents=True)
    with sqlite3.connect(root / "sql" / "accounts.db") as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE)")
        conn.execute("INSERT INTO users (id, username) VALUES (17, 'operator-claude')")
        conn.execute("CREATE TABLE notes (body TEXT)")
        conn.executemany("INSERT INTO notes VALUES (?)", [(f"note {i}",) for i in range(200)])
    (root / "content").mkdir()
    (root / "content" / "page.txt").write_text("hello regions\n")
    (root / "secrets").mkdir()
    (root / "secrets" / "link.key").write_bytes(b"\x07" * 32)
    return root


@pytest.fixture
def regions(tmp_path):
    writer = _state(tmp_path / "home")
    reader = tmp_path / "oracle"
    stranger = tmp_path / "elsewhere"
    for root in (reader, stranger):
        root.mkdir()
    home_pub = sb.keygen(writer, "home")
    oracle_pub = sb.keygen(reader, "oracle")
    sb.keygen(stranger, "elsewhere")
    db = writer / "sql" / "accounts.db"
    sb.enroll(db, "oracle", oracle_pub["sign_pub"], oracle_pub["box_pub"], "operator-claude")
    recovery = sb.recovery_keygen(tmp_path / "owner-recovery.key")
    sb.enroll(db, "owner-recovery", "", recovery["box_pub"], "operator-claude", kind="recovery")
    for root in (reader, stranger):
        sb.pin_writer(root, "home", home_pub["sign_pub"])
    return {"tmp": tmp_path, "writer": writer, "reader": reader, "stranger": stranger, "db": db, "home": home_pub}


def _pack(regions, generation_out="out"):
    return sb.pack(regions["writer"], regions["db"], regions["tmp"] / "work", regions["tmp"] / generation_out)


def test_enrolled_reader_accepts_and_takes_over(regions):
    packed = _pack(regions)
    assert packed["generation"] == 1 and packed["recipients"] == ["oracle", "owner-recovery"]
    held = regions["tmp"] / "held"
    accepted = sb.accept(regions["reader"], packed["dir"], held)
    assert accepted["accepted"] and accepted["generation"] == 1
    # Nothing in the clear is kept: the held copy is the sealed records only.
    assert all(p.suffix == ".rec" for p in Path(accepted["held"]).iterdir())
    assert b"hello regions" not in b"".join(p.read_bytes() for p in Path(accepted["held"]).iterdir())

    target = regions["tmp"] / "regrown"
    with pytest.raises(sb.BroadcastError, match="writer is stopped"):
        sb.take_over(regions["reader"], accepted["held"], target, writer_stopped=False)
    took = sb.take_over(regions["reader"], accepted["held"], target, writer_stopped=True)
    assert took["generation"] == 1
    assert (target / "content" / "page.txt").read_text() == "hello regions\n"
    assert (target / "secrets" / "link.key").read_bytes() == b"\x07" * 32
    with sqlite3.connect(target / "sql" / "accounts.db") as conn:
        assert conn.execute("SELECT count(*) FROM notes").fetchone()[0] == 200
        assert conn.execute("SELECT region FROM region_enrollment ORDER BY region").fetchall() == [
            ("oracle",), ("owner-recovery",)]
    # The writer's own keys never travel.
    assert not (target / "secrets" / "region").exists()


def test_refusals(regions):
    packed = _pack(regions)
    held = regions["tmp"] / "held"
    with pytest.raises(sb.BroadcastError, match="not sealed to elsewhere"):
        sb.accept(regions["stranger"], packed["dir"], regions["tmp"] / "held-stranger")

    tampered = regions["tmp"] / "tampered"
    shutil.copytree(packed["dir"], tampered)
    chunk = tampered / "000001.rec"
    data = bytearray(chunk.read_bytes())
    data[len(data) // 2] ^= 1
    chunk.write_bytes(bytes(data))
    with pytest.raises(sb.BroadcastError, match="does not match its hash"):
        sb.accept(regions["reader"], tampered, held)

    sb.accept(regions["reader"], packed["dir"], held)
    with pytest.raises(sb.BroadcastError, match="not above 1"):
        sb.accept(regions["reader"], packed["dir"], held)

    # An impostor writer with its own keys, claiming to be home, is refused by the pinned key.
    impostor = _state(regions["tmp"] / "impostor")
    sb.keygen(impostor, "home")
    oracle_pub = sb.Region(regions["reader"]).public()
    sb.enroll(impostor / "sql" / "accounts.db", "oracle", oracle_pub["sign_pub"], oracle_pub["box_pub"], "operator-claude")
    forged = sb.pack(impostor, impostor / "sql" / "accounts.db", regions["tmp"] / "work2", regions["tmp"] / "forged")
    # The signature is checked before the generation number, so even a forged higher number is refused for its key.
    with pytest.raises(sb.BroadcastError, match="signature does not verify"):
        sb.accept(regions["reader"], forged["dir"], held)


def test_generations_rise_and_revocation_excludes(regions):
    first = _pack(regions)
    sb.revoke(regions["db"], "oracle")
    second = _pack(regions)
    assert second["generation"] == first["generation"] + 1
    assert second["recipients"] == ["owner-recovery"]
    with pytest.raises(sb.BroadcastError, match="not sealed to oracle"):
        sb.accept(regions["reader"], second["dir"], regions["tmp"] / "held")

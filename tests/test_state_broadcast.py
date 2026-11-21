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
    # The clear manifest names nothing private: no file names, no recipient regions (key fingerprints only).
    header = (Path(accepted["held"]) / "000000.rec").read_bytes()
    for private in (b"page.txt", b"link.key", b"accounts.db", b"oracle", b"owner-recovery"):
        assert private not in header, private
    # The writer's own keys never travel: what the regrown root holds is the taking-over region's identity, and it is a
    # working writer that continues the numbering.
    assert sb.Region(target).name == "oracle"
    assert sb.Region(target).key_id == sb.Region(regions["reader"]).key_id
    successor = sb.pack(target, target / "sql" / "accounts.db", regions["tmp"] / "work-next", regions["tmp"] / "out-next")
    assert successor["writer"] == "oracle" and successor["generation"] == 2


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


def _arrivals(records, spool):
    """The wire order: the sealed chunks, then the signed manifest (000000.rec), each one spooled arrival."""
    order = sorted(r for r in records if r.name != "000000.rec") + [r for r in records if r.name == "000000.rec"]
    spool.mkdir(exist_ok=True)
    for index, record in enumerate(order):
        shutil.copyfile(record, spool / f"{index + 1:08d}.arrival")
    return sorted(spool.glob("*.arrival"))


def test_assemble_rebuilds_a_generation_from_arrivals(regions):
    """The link carries one record per tensor; arrivals are spooled in order and assembled back into a generation."""
    packed = _pack(regions)
    arrivals = _arrivals(sorted(Path(packed["dir"]).glob("*.rec")), regions["tmp"] / "spool")
    generation, rest = sb.assemble(arrivals[:-1], regions["tmp"] / "incoming")     # the manifest has not arrived
    assert generation is None and len(rest) == len(arrivals) - 1
    extra = regions["tmp"] / "spool" / "99999999.arrival"
    extra.write_bytes(b"the next generation's first chunk")
    generation, rest = sb.assemble(arrivals + [extra], regions["tmp"] / "incoming")
    assert generation is not None and rest == [extra]
    assert sb.accept(regions["reader"], generation, regions["tmp"] / "held")["accepted"]


def test_many_chunks_stream_and_emit(regions, monkeypatch):
    """A small chunk size makes many sealed chunks: the last one alone says it is last, emit hands each record over as
    it is written and leaves none behind, and the reader still assembles, accepts and regrows the state."""
    monkeypatch.setattr(sb, "CHUNK_BYTES", 4096)
    (regions["writer"] / "content" / "noise.bin").write_bytes(__import__("os").urandom(50_000))
    handed = []
    spool = regions["tmp"] / "spool"
    spool.mkdir()

    def emit(path):
        handed.append(path.name)
        shutil.copyfile(path, spool / f"{len(handed):08d}.arrival")

    packed = sb.pack(regions["writer"], regions["db"], regions["tmp"] / "work", regions["tmp"] / "out", emit=emit)
    assert packed["chunks"] > 10 and handed[-1] == "000000.rec" and len(handed) == packed["chunks"] + 1
    assert list(Path(packed["dir"]).iterdir()) == []                                 # nothing left on disk
    generation, rest = sb.assemble(sorted(spool.glob("*.arrival")), regions["tmp"] / "incoming")
    assert rest == []
    held = sb.accept(regions["reader"], generation, regions["tmp"] / "held")
    target = regions["tmp"] / "regrown"
    sb.take_over(regions["reader"], held["held"], target, writer_stopped=True)
    assert (target / "content" / "noise.bin").read_bytes() == (regions["writer"] / "content" / "noise.bin").read_bytes()


def test_assemble_places_chunks_by_hash_in_any_order(regions, monkeypatch):
    """A link may deliver a generation out of order (a refused send is retried later, even after the manifest); every
    chunk is placed by its hash, so a shuffled arrival order still assembles and accepts."""
    import random
    monkeypatch.setattr(sb, "CHUNK_BYTES", 4096)
    (regions["writer"] / "content" / "noise.bin").write_bytes(__import__("os").urandom(40_000))
    packed = _pack(regions)
    records = sorted(Path(packed["dir"]).glob("*.rec"))
    random.Random(7).shuffle(records)
    spool = regions["tmp"] / "spool"
    spool.mkdir()
    for index, record in enumerate(records):
        shutil.copyfile(record, spool / f"{index + 1:08d}.arrival")
    generation, rest = sb.assemble(sorted(spool.glob("*.arrival")), regions["tmp"] / "incoming")
    assert generation is not None and rest == []
    assert sb.accept(regions["reader"], generation, regions["tmp"] / "held")["accepted"]


def test_assemble_never_follows_an_unverified_writer_out_of_incoming(regions):
    """assemble runs before any signature is checked, so a header naming a writer like ../outside, an absolute path or a
    non-integer generation is not a manifest: nothing outside the incoming directory is removed or written."""
    import json as _json
    import struct as _struct
    packed = _pack(regions)
    records = sorted(Path(packed["dir"]).glob("*.rec"))
    manifest, _body, _signature = sb.read_header(records[0].read_bytes())
    incoming = regions["tmp"] / "incoming"
    outside = regions["tmp"] / "outside-00000001"
    outside.mkdir()
    (outside / "keep.txt").write_text("must survive")
    for writer, generation in (("../outside", 1), (str(outside), 1), ("ok", "1"), ("a/b", 1), ("..", 1)):
        forged = dict(manifest, writer=writer, generation=generation)
        body = _json.dumps(forged).encode("utf-8")
        spool = regions["tmp"] / f"spool-{abs(hash((writer, str(generation))))}"
        spool.mkdir()
        header = spool / "00000001.arrival"
        header.write_bytes(sb.MAGIC + _struct.pack(">I", len(body)) + body + bytes(64))
        arrivals = [header]
        for index, record in enumerate(records[1:]):
            arrivals.append(spool / f"{index + 2:08d}.arrival")
            shutil.copyfile(record, arrivals[-1])
        generation_dir, rest = sb.assemble(arrivals, incoming)
        assert generation_dir is None and len(rest) == len(arrivals), (writer, generation)
    assert (outside / "keep.txt").read_text() == "must survive"
    assert not incoming.exists() or not any(incoming.iterdir())


def test_region_names_are_plain_components(tmp_path):
    for bad in ("../x", "a/b", "", ".hidden", "x" * 65):
        with pytest.raises(sb.BroadcastError):
            sb.keygen(tmp_path / bad.replace("/", "_").replace(".", "_") / "s", bad)

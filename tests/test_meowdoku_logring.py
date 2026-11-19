"""[LOG] The server log ring (examples/meowdoku/logring.py) and GET /api/log's tier gate."""
from __future__ import annotations

import http.client
import io
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import access as A  # noqa: E402
from meowdoku import logring as L  # noqa: E402
from meowdoku import server as srv  # noqa: E402


def test_cap_is_bytes_and_oldest_dropped():
    ring = L.Ring(cap_bytes=1000)
    for i in range(100):
        ring.add(f"line {i:03d} " + "x" * 40)
    assert ring.bytes <= 1000
    got = ring.fetch(0, 5000)
    assert got["last"] == 100 and got["entries"][-1]["text"].startswith("line 099")
    assert got["first"] == got["entries"][0]["seq"] > 1                     # the oldest were dropped
    assert [e["seq"] for e in got["entries"]] == list(range(got["first"], 101))
    ring.add("y" * 100000)                                                 # a huge line is cut, and cannot empty the ring of itself
    assert len(ring.fetch(0)["entries"][-1]["text"]) == L.LINE_CAP


def test_incremental_fetch_and_sources():
    ring = L.Ring(cap_bytes=10000)
    ring.add("[client-error] 1.2.3.4: TypeError | boom | x | s | ua")
    ring.add("[client-error] 1.2.3.4: fountain-diag | {} | x | | ua")
    ring.add("[meowdoku] serving")
    first = ring.fetch(0)
    assert [e["src"] for e in first["entries"]] == ["client-error", "fountain-diag", "server"]
    assert ring.fetch(first["last"])["entries"] == []
    ring.add("[idle-trainer] step")
    more = ring.fetch(first["last"])
    assert [e["text"] for e in more["entries"]] == ["[idle-trainer] step"]
    assert [e["seq"] for e in ring.fetch(0, limit=2)["entries"]] == [1, 2]
    assert [e["src"] for e in ring.fetch(0, sources={"server"})["entries"]] == ["server", "server"]


def test_login_and_activity_lines(monkeypatch):
    monkeypatch.setattr(L, "RING", L.Ring(cap_bytes=10000))
    monkeypatch.setattr(L, "_last_seen", {})
    L.note_login("fail", " Ad min\nX ", "1.2.3.4")                          # caller-typed name: no whitespace survives
    L.note_login("ok", "admin", "1.2.3.4")
    assert L.note_activity(7, "admin", "1.2.3.4", now=100.0) is not None     # first sight is a line
    assert L.note_activity(7, "admin", "1.2.3.4", now=100.0 + L.IDLE_SECONDS - 1) is None
    assert L.note_activity(7, "admin", "1.2.3.4", now=100.0 + 2 * L.IDLE_SECONDS) is not None   # quiet IDLE_SECONDS since the last request
    got = L.RING.fetch(0, sources={"login", "activity"})["entries"]
    ip = L.private_record.token("ip", "1.2.3.4")                         # [PRIVATE] the ring holds the address's token
    assert [(e["src"], e["text"]) for e in got] == [
        ("login", f"fail user=adminx client={ip}"), ("login", f"ok user=admin client={ip}"),
        ("activity", f"active user=admin client={ip}"), ("activity", f"active user=admin client={ip}")]
    assert "1.2.3.4" not in json.dumps(L.RING.fetch(0))


def test_login_endpoint_files_login_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RING", L.Ring(cap_bytes=10000))
    httpd = _server(tmp_path, {"lan": "dev"})
    try:
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        for pw in ("wrong-password", "correcthorse1"):
            c.request("POST", "/api/account/login", body=json.dumps({"username": "log_cat", "password": pw}),
                      headers={"Content-Type": "application/json"})
            c.getresponse().read()
        texts = [e["text"] for e in L.RING.fetch(0, sources={"login"})["entries"]]
        ip = L.private_record.token("ip", "127.0.0.1")
        assert texts == [f"fail user=log_cat client={ip}", f"ok user=log_cat client={ip}"]
    finally:
        httpd.shutdown(); httpd.server_close()


def test_persistence_survives_restart(tmp_path):
    p = tmp_path / "ring.jsonl"
    key = tmp_path / "secrets" / "records.key"
    a = L.Ring(cap_bytes=5000, path=p, key_path=key)
    for i in range(10):
        a.add(f"n{i}")
    assert p.read_bytes().startswith(b"enc:v1:")
    assert b'"text": "n0"' not in p.read_bytes()
    b = L.Ring(cap_bytes=5000, path=p, key_path=key)
    assert b.seq == 10 and b.add("after") == 11 and len(b.fetch(0)["entries"]) == 11


def test_bad_persisted_lines_are_skipped_not_fatal(tmp_path):
    """Ticket f62934c8: one cut-off or corrupt line must not stop the server from starting."""
    p = tmp_path / "ring.jsonl"
    key = tmp_path / "secrets" / "records.key"
    a = L.Ring(cap_bytes=5000, path=p, key_path=key)
    for i in range(5):
        a.add(f"n{i}")
    good = p.read_bytes()
    cut = a.cipher.seal(b'{"seq": 6, "ts": 1.0, "src": "server", "text": "n5"}')[:30]   # an append cut off by a kill
    p.write_bytes(good + b"enc:v1:not-a-token\n" + cut)
    found = p.read_bytes()
    b = L.Ring(cap_bytes=5000, path=p, key_path=key)
    texts = [e["text"] for e in b.fetch(0)["entries"]]
    assert texts[:5] == [f"n{i}" for i in range(5)]
    assert texts[5].startswith("[logring] skipped 2 unreadable line(s)") and b.seq == 6
    evidence = list(tmp_path.glob("ring.jsonl.bad-*"))
    assert len(evidence) == 1 and evidence[0].read_bytes() == found          # the file as found is kept
    assert not list(tmp_path.glob("ring.jsonl.rewrite-*"))                    # no temporary left behind
    c = L.Ring(cap_bytes=5000, path=p, key_path=key)                          # the rewritten ring reloads clean
    assert [e["text"] for e in c.fetch(0)["entries"]] == texts
    assert len(list(tmp_path.glob("ring.jsonl.bad-*"))) == 1


def test_failed_evidence_copy_leaves_the_file_as_found(tmp_path, monkeypatch):
    """Ticket f62934c8, review eab6c75f (codex): no rewrite may drop the unreadable lines before an evidence copy of them exists,
    and an append after a cut-off tail must not fuse with it into one unreadable line."""
    p = tmp_path / "ring.jsonl"
    key = tmp_path / "secrets" / "records.key"
    a = L.Ring(cap_bytes=5000, path=p, key_path=key)
    for i in range(5):
        a.add(f"n{i}")
    cut = a.cipher.seal(b'{"seq": 6, "ts": 1.0, "src": "server", "text": "n5"}')[:30]   # no newline: an append cut off by a kill
    p.write_bytes(p.read_bytes() + b"enc:v1:not-a-token\n" + cut)
    found = p.read_bytes()

    def failing_copy(src, dst, *args, **kwargs):
        dst.write(b"partial")                                                 # a copy that fails after writing into its file
        raise OSError("disk full")
    monkeypatch.setattr(L.shutil, "copyfileobj", failing_copy)
    b = L.Ring(cap_bytes=5000, path=p, key_path=key)
    texts = [e["text"] for e in b.fetch(0)["entries"]]
    assert texts[:5] == [f"n{i}" for i in range(5)]                           # startup still works from the readable lines
    assert texts[5].startswith("[logring] skipped 2 unreadable line(s)") and "could not be written" in texts[5]
    assert p.read_bytes().startswith(found)                                   # the original bytes are untouched
    assert not list(tmp_path.glob("ring.jsonl.bad-*"))                        # no partial evidence left behind
    b.add("after")
    c = L.Ring(cap_bytes=5000, path=p, key_path=key)                          # restart while the copy still fails
    texts = [e["text"] for e in c.fetch(0)["entries"]]
    assert texts[:7] == [f"n{i}" for i in range(5)] + [texts[5], "after"]    # the records appended after the cut tail survive
    assert texts[7].startswith("[logring] skipped 2 unreadable line(s)")      # still only the original bad lines
    assert p.read_bytes().startswith(found)

    c._written = 2 * c.cap + 1                                                # an oversize rewrite while the copy fails: appended
    c.add("still failing")
    assert p.read_bytes().startswith(found) and not list(tmp_path.glob("ring.jsonl.bad-*"))
    monkeypatch.undo()                                                        # the copy works again: evidence first, then the rewrite
    before = p.read_bytes()
    c.add("copy works")
    evidence = list(tmp_path.glob("ring.jsonl.bad-*"))
    assert len(evidence) == 1 and evidence[0].read_bytes() == before and before.startswith(found)
    assert not p.read_bytes().startswith(found)                               # rewritten from the ring
    d = L.Ring(cap_bytes=5000, path=p, key_path=key)
    assert [e["text"] for e in d.fetch(0)["entries"]] == [e["text"] for e in c.fetch(0)["entries"]]
    assert len(list(tmp_path.glob("ring.jsonl.bad-*"))) == 1                 # the rewritten ring reloads clean


def test_same_second_failed_copy_keeps_the_earlier_snapshot(tmp_path, monkeypatch):
    """Ticket f62934c8, review 0ea3e90b (codex): two captures in the same second must not share a destination; a failed later copy
    must neither truncate nor remove the earlier snapshot, and must leave the damaged ring as found (no rewrite)."""
    p = tmp_path / "ring.jsonl"
    key = tmp_path / "secrets" / "records.key"
    monkeypatch.setattr(L.time, "time", lambda: 1795000000.0)                # every capture below falls in the same second
    a = L.Ring(cap_bytes=5000, path=p, key_path=key)
    for i in range(5):
        a.add(f"n{i}")
    p.write_bytes(p.read_bytes() + b"enc:v1:not-a-token\n")
    found = p.read_bytes()
    L.Ring(cap_bytes=5000, path=p, key_path=key)                              # the first capture succeeds
    first = tmp_path / "ring.jsonl.bad-1795000000"
    assert first.read_bytes() == found

    p.write_bytes(p.read_bytes() + b"enc:v1:not-a-token-either\n")            # damaged again, same second
    found2 = p.read_bytes()
    real_copy = L.shutil.copyfileobj

    def failing_copy(src, dst, *args, **kwargs):
        dst.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(L.shutil, "copyfileobj", failing_copy)
    c = L.Ring(cap_bytes=5000, path=p, key_path=key)
    assert "could not be written" in c.fetch(0)["entries"][-1]["text"]
    assert first.read_bytes() == found                                        # the earlier snapshot is intact
    assert list(tmp_path.glob("ring.jsonl.bad-*")) == [first]                 # the failed attempt left nothing behind
    assert p.read_bytes().startswith(found2)                                  # the damaged ring is left as found

    monkeypatch.setattr(L.shutil, "copyfileobj", real_copy)                   # the copy works again, still the same second
    before = p.read_bytes()
    L.Ring(cap_bytes=5000, path=p, key_path=key)
    second = tmp_path / "ring.jsonl.bad-1795000000-1"
    assert first.read_bytes() == found and second.read_bytes() == before and before.startswith(found2)
    assert sorted(tmp_path.glob("ring.jsonl.bad-*")) == sorted([first, second])


def test_oversize_rewrite_is_atomic(tmp_path):
    p = tmp_path / "ring.jsonl"
    key = tmp_path / "secrets" / "records.key"
    a = L.Ring(cap_bytes=600, path=p, key_path=key)
    for i in range(200):                                                      # forces several rewrites (written > 2 * cap)
        a.add(f"line {i:03d} " + "x" * 20)
    assert not list(tmp_path.glob("ring.jsonl.rewrite-*"))
    b = L.Ring(cap_bytes=600, path=p, key_path=key)
    assert b.seq == 200 and b.fetch(0)["entries"][-1]["text"].startswith("line 199")
    assert not list(tmp_path.glob("ring.jsonl.bad-*"))


def test_tee_keeps_stdout_and_files_lines(monkeypatch):
    real = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real)
    monkeypatch.setattr(L, "RING", L.Ring(cap_bytes=10000))
    L.tee_stdout()
    print("[meowdoku] hello", flush=True)
    sys.stdout.write("part"); sys.stdout.write("ial\n")
    out = sys.stdout
    monkeypatch.setattr(sys, "stdout", real)                               # un-tee before asserting
    assert real.getvalue() == "[meowdoku] hello\npartial\n" and out is not real
    assert [e["text"] for e in L.RING.fetch(0)["entries"]] == ["[meowdoku] hello", "partial"]


def _server(tmp_path, policy):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.access = A.Access(None)
    sessions.access.config()["policy"].update(policy)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    # a visitor with no account is a guest and never holds dev_ui (owner, 2026-11-13): the tiers are tried as a signed-in member
    uid = sessions.accounts.ensure_user("log_cat", "correcthorse1")
    httpd.member = {"Cookie": f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(uid)}"}
    return httpd


def _get(httpd, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    return r.status, r.read()


def test_log_endpoint_tier_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "RING", L.Ring(cap_bytes=10000))
    L.RING.add("[meowdoku] one"); L.RING.add("[meowdoku] two")
    # loopback is "lan": tiers by origin; every tier is tried by pointing the lan origin at it
    for tier in ("edit", "dev", "play", "shader"):
        httpd = _server(tmp_path, {"lan": tier})
        try:
            assert _get(httpd, "/api/log?since=1&limit=10")[0] == 403                     # a guest, on any tier
            status, body = _get(httpd, "/api/log?since=1&limit=10", httpd.member)
            assert status == 403 and b"meowdoku" not in body
        finally:
            httpd.shutdown(); httpd.server_close()

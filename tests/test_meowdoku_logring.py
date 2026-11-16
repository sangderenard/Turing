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
    assert [(e["src"], e["text"]) for e in got] == [
        ("login", "fail user=adminx client=1.2.3.4"), ("login", "ok user=admin client=1.2.3.4"),
        ("activity", "active user=admin client=1.2.3.4"), ("activity", "active user=admin client=1.2.3.4")]


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
        assert texts == ["fail user=log_cat client=127.0.0.1", "ok user=log_cat client=127.0.0.1"]
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

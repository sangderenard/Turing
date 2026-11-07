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


def test_persistence_survives_restart(tmp_path):
    p = tmp_path / "ring.jsonl"
    a = L.Ring(cap_bytes=5000, path=p)
    for i in range(10):
        a.add(f"n{i}")
    b = L.Ring(cap_bytes=5000, path=p)
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
    for tier, ok in (("edit", True), ("dev", True), ("play", False), ("shader", False)):
        httpd = _server(tmp_path, {"lan": tier})
        try:
            status, body = _get(httpd, "/api/log?since=1&limit=10")
            assert (status == 200) == ok, (tier, status)
            if ok:
                assert [e["text"] for e in json.loads(body)["entries"]] == ["[meowdoku] two"]
            else:
                assert status == 403 and b"meowdoku" not in body
            if ok:
                status, _ = _get(httpd, "/api/log", {srv.share_mod.VIEWER_HEADER: "x"})      # a view-only link is refused even on a dev tier
                assert status == 403
        finally:
            httpd.shutdown(); httpd.server_close()

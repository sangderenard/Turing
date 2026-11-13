"""Nodus web service (examples/meowdoku/nodus_service.py, server.py [NODUS]) against a FAKE console on an ephemeral loopback port and temp roots."""
from __future__ import annotations

import http.client
import json
import logging
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import messaging  # noqa: E402
from meowdoku import nodus_service as ns  # noqa: E402
from meowdoku import server as srv  # noqa: E402

PASSWORD = "correct horse 12"
TOKEN = "tok-SECRET-0123456789abcdef"
OUTSIDE = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}
SHA = "ab12cd34ef56" + "0" * 52

DIGEST = {"protocol": 2, "canvas": "demo.nodus.txt", "revision": 7, "sha256": SHA, "bytes": 100, "modules": 2, "rows": 3, "edges": 1,
          "contract_ports": 1, "compile_hosts": 0, "tick": 42, "rate": 30.0, "started": 1.0}
GRAPH = {"revision": 7,
         "modules": [{"index": 0, "name": "src", "id": 10, "place": "cpu", "exec_mode": "seq",
                      "rows": [{"index": 0, "tool": "gen", "plugin_id": "p1"}, {"index": 1, "tool": "scale", "plugin_id": "p1"}],
                      "contacts": [{"index": 0, "name": "out"}]},
                     {"index": 1, "name": "sink", "id": 11, "place": "cpu", "exec_mode": "seq",
                      "rows": [{"index": 0, "tool": "show", "plugin_id": "p2"}], "contacts": [{"index": 0, "name": "in"}]}],
         "edges": [{"index": 0, "from": {"module": 0, "contact": 0}, "to": {"module": 1, "contact": 0}, "token_bytes": 8, "slots": 4}],
         "ports": [], "contract_ports": [{"prefix": "cp", "capacity": 4, "listen_port": 5555, "loan_ticks": 2, "edge": 0}]}
TOOLS = {"tools": [{"name": "gen", "plugin_id": "p1", "source": "s", "description": "makes"}]}
ACTIVITY = {"tick": 42, "edges": [{"index": 0, "fill": 2, "seq": 9}], "contract_ports": [], "compile_hosts": []}
REPLIES = {"DIGEST": DIGEST, "GRAPH": GRAPH, "TOOLS": TOOLS, "ACTIVITY": ACTIVITY}


class FakeConsole:
    def __init__(self, v2=True):
        self.v2, self.connections, self.verbs, self.seen_tokens = v2, 0, [], []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile("rb")
        try:
            hello = f.readline().decode().split()
            self.seen_tokens.append(hello[2] if len(hello) > 2 else "")
            if hello[:2] != ["HELLO", "nodus-console/1"] or hello[2] != TOKEN:
                conn.sendall(b"REFUSED bad token\n")
                return
            conn.sendall(b"OK console demo.nodus.txt read-only\n")
            for raw in f:
                verb = raw.decode().strip()
                self.verbs.append(verb)
                if self.v2 and verb in REPLIES:
                    data = json.dumps(REPLIES[verb]).encode()
                    conn.sendall(f"JSON {verb.lower()} {len(data)}\n".encode() + data)
                else:
                    conn.sendall(b"REFUSED read-only\n")
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self.sock.close()


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    if data:
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    raw, status, cookies = r.read(), r.status, r.msg.get_all("Set-Cookie") or []
    conn.close()
    try:
        return status, cookies, json.loads(raw)
    except ValueError:
        return status, cookies, raw.decode("utf-8", "replace")


@pytest.fixture()
def server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ns.CACHE._items.clear()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, tmp=Path(tmp_path))
    httpd.shutdown()
    httpd.server_close()
    ns.CACHE._items.clear()


def _register(server, port, name="demo", state="running"):
    """A nodus row in the supervisor's own table plus the per-run token file at the supervisor's own path."""
    sup = ns._supervisor()
    with sup.connect(server.sessions.accounts_db) as db:
        db.execute("INSERT OR REPLACE INTO service_instances (name, kind, port, route, account_user_id, state, console_port) VALUES (?,?,?,?,?,?,?)",
                   (name, "nodus", None, "", 0, state, port))
    path = sup.console_secret_path(server.sessions.state, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TOKEN, encoding="utf-8")


def _cookie(server, name, role):
    server.sessions.accounts.ensure_user(name, PASSWORD)
    if role != "member":
        authz.grant_role(server.sessions.accounts_db, name, role, "operator:test")
    status, cookies, _ = _call(server.port, "/api/account/login", "POST", None, {"username": name, "password": PASSWORD})
    assert status == 200
    return {"Cookie": next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))}


@pytest.fixture()
def fake():
    f = FakeConsole()
    yield f
    f.close()


def test_refused_without_dev_ui(server, fake):
    _register(server, fake.port)
    for path in ("instances", "digest", "graph", "tools", "activity", "abstract"):
        assert _call(server.port, "/api/nodus/" + path, headers=OUTSIDE)[0] == 403
    h = _cookie(server, "member1", "member")
    assert _call(server.port, "/api/nodus/digest?instance=demo", headers={**h, **OUTSIDE})[0] == 403
    assert fake.connections == 0


def test_instances_hide_token_and_path(server, fake):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    status, _, body = _call(server.port, "/api/nodus/instances", headers=h)
    assert status == 200 and body == {"instances": [{"name": "demo", "state": "running", "console": True}]}
    assert TOKEN not in json.dumps(body) and ".secret" not in json.dumps(body)


def test_passthrough_and_cache(server, fake, caplog):
    caplog.set_level(logging.DEBUG)
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    st, _, d = _call(server.port, "/api/nodus/digest?instance=demo", headers=h)
    assert st == 200 and d["instance"] == "demo" and d["revision"] == 7 and d["sha256"] == SHA
    _call(server.port, "/api/nodus/digest?instance=demo", headers=h)
    assert fake.connections == 1                                    # digest cached 1 s
    st, _, g = _call(server.port, "/api/nodus/graph?instance=demo", headers=h)
    assert st == 200 and g["instance"] == "demo" and g["modules"][0]["name"] == "src"
    assert fake.connections == 2                                    # the digest was cached; one new connection for GRAPH
    _call(server.port, "/api/nodus/graph?instance=demo", headers=h)
    assert fake.connections == 2                                    # graph cached by (instance, revision)
    st, _, t = _call(server.port, "/api/nodus/tools?instance=demo", headers=h)
    assert st == 200 and t["tools"][0]["name"] == "gen" and t["instance"] == "demo"
    st, _, a = _call(server.port, "/api/nodus/activity?instance=demo", headers=h)
    assert st == 200 and a["edges"][0]["fill"] == 2 and a["instance"] == "demo"
    assert set(fake.verbs) == {"DIGEST", "GRAPH", "TOOLS", "ACTIVITY"} and fake.verbs.count("GRAPH") == 1
    assert set(fake.seen_tokens) == {TOKEN}
    # no instance named, exactly one running: that one
    assert _call(server.port, "/api/nodus/digest", headers=h)[2]["instance"] == "demo"
    assert TOKEN not in caplog.text
    ns.CACHE._items.clear()
    n = fake.connections
    assert _call(server.port, "/api/nodus/digest?instance=demo", headers=h)[0] == 200 and fake.connections == n + 1


def test_unknown_instance_and_token_never_in_responses(server, fake):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    assert _call(server.port, "/api/nodus/digest?instance=nope", headers=h)[0] == 404
    for path in ("instances", "digest", "graph", "tools", "activity", "abstract"):
        status, _, body = _call(server.port, "/api/nodus/" + path, headers=h)
        assert TOKEN not in json.dumps(body) and ".secret" not in json.dumps(body)


def test_v1_host_is_501(server):
    old = FakeConsole(v2=False)
    try:
        _register(server, old.port)
        h = _cookie(server, "dev1", "operator")
        for path in ("digest", "graph", "tools", "activity", "abstract"):
            status, _, body = _call(server.port, "/api/nodus/%s?instance=demo" % path, headers=h)
            assert status == 501 and "not supported by this host" in body["error"], path
    finally:
        old.close()


def test_nothing_listening_is_503(server):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    _register(server, port)
    h = _cookie(server, "dev1", "operator")
    status, _, body = _call(server.port, "/api/nodus/digest?instance=demo", headers=h)
    assert status == 503 and TOKEN not in json.dumps(body)


def test_stopped_instance_is_503(server, fake):
    _register(server, fake.port, state="stopped")
    h = _cookie(server, "dev1", "operator")
    assert _call(server.port, "/api/nodus/digest?instance=demo", headers=h)[0] == 503
    assert _call(server.port, "/api/nodus/instances", headers=h)[2]["instances"][0]["console"] is False


# ---- generic graph ----
def test_generic_graph_shape():
    first = ns.as_generic_graph(GRAPH, "demo")
    second = ns.as_generic_graph(json.loads(json.dumps(GRAPH)), "demo")
    assert first == second and len({n["identity"] for n in first}) == len(first)
    kinds = {n["kind"] for n in first}
    assert kinds == {"canvas", "module", "row", "contact", "edge", "contract-port"}
    by_id = {n["identity"]: n for n in first}
    canvases = [n for n in first if n["kind"] == "canvas"]
    assert len(canvases) == 1 and canvases[0]["dependencies"] == []
    for n in first:
        assert set(n) >= {"identity", "kind", "name", "dependencies"}
        for d in n["dependencies"]:
            assert set(d) == {"relationship", "target"} and d["target"] in by_id
        if n["kind"] != "canvas":
            assert sum(d["relationship"] == "contained-by" for d in n["dependencies"]) == 1
        if n["kind"] == "edge":
            ends = {d["relationship"]: by_id[d["target"]] for d in n["dependencies"] if d["relationship"] != "contained-by"}
            assert set(ends) == {"from-contact", "to-contact"} and all(e["kind"] == "contact" for e in ends.values())
    assert [n["name"] for n in first if n["kind"] == "row"] == ["gen", "scale", "show"]
    port = next(n for n in first if n["kind"] == "contract-port")
    assert any(d["relationship"] == "bound-to" and by_id[d["target"]]["kind"] == "edge" for d in port["dependencies"])


def test_generic_graph_unlisted_contact_is_declared():
    g = {"modules": [{"index": 0, "name": "a"}, {"index": 1, "name": "b"}],
         "edges": [{"index": 0, "from": {"module": 0, "contact": 3}, "to": {"module": 1, "contact": 4}}]}
    nodes = ns.as_generic_graph(g, "i")
    ids = {n["identity"] for n in nodes}
    edge = next(n for n in nodes if n["kind"] == "edge")
    assert len([d for d in edge["dependencies"] if d["relationship"] != "contained-by"]) == 2
    assert all(d["target"] in ids for d in edge["dependencies"])


def test_abstract_endpoint(server, fake):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    status, _, body = _call(server.port, "/api/nodus/abstract?instance=demo", headers=h)
    assert status == 200 and body["revision"] == 7 and body["nodes"] == ns.as_generic_graph({**GRAPH, "canvas": "demo.nodus.txt"}, "demo")


def test_living_picture_console_wall_and_region_feed(server, fake):
    class UiFeed(BaseHTTPRequestHandler):
        requests = []

        def do_GET(self):
            self.__class__.requests.append(self.path)
            body = b"\x89PNG\r\n\x1a\nframe"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    ui = ThreadingHTTPServer(("127.0.0.1", 0), UiFeed)
    threading.Thread(target=ui.serve_forever, daemon=True).start()
    try:
        _register(server, fake.port)
        server.sessions.nodus_ui_ports["demo"] = ui.server_address[1]
        h = _cookie(server, "map_viewer", "operator")
        status, _, picture = _call(server.port, "/api/nodus/picture?instance=demo", headers=h)
        assert status == 200 and picture["map"]["schema"] == "abstract-ui-div-map-v0"
        assert picture["view"] == "/nodus.html?instance=demo#living-map"
        regions = picture["map"]["regions"]
        wall = next(r for r in regions if r["kind"] == "console-wall")
        console = next(r for r in regions if r["kind"] == "console")
        assert any(t == {"source": console["identity"], "target": wall["identity"], "relationship": "contained-by"}
                   for t in picture["map"]["tracks"])
        assert console["view"]["full"] == "/api/nodus/frame?instance=demo"
        status, _, dm = _call(server.port, "/api/paper/dm", "POST", {**h, **OUTSIDE},
                              {"verb": "look", "target": "picture", "input": {}})
        assert status == 200 and dm["map"] == picture["map"]
        conn = http.client.HTTPConnection("127.0.0.1", server.port)
        conn.request("GET", "/api/nodus/frame?instance=demo&x=2&y=3&width=4&height=5", headers=h)
        reply = conn.getresponse()
        assert reply.status == 200 and reply.getheader("Content-Type") == "image/png"
        assert reply.read().startswith(b"\x89PNG")
        conn.close()
        assert UiFeed.requests[-1] == "/frame?x=2&y=3&width=4&height=5" or set(UiFeed.requests[-1].split("?")[1].split("&")) == {"x=2", "y=3", "width=4", "height=5"}
        for frame in ("1", "2"):
            conn = http.client.HTTPConnection("127.0.0.1", server.port)
            conn.request("GET", f"/api/nodus/frame?instance=demo&frame={frame}", headers=h)
            reply = conn.getresponse()
            assert reply.status == 200 and reply.read().startswith(b"\x89PNG")
            conn.close()
            assert UiFeed.requests[-1] == "/frame"
        assert _call(server.port, "/api/nodus/frame?instance=demo&frame=bad", headers=h)[0] == 400
        assert _call(server.port, "/api/nodus/frame?instance=demo&x=2", headers=h)[0] == 400
    finally:
        ui.shutdown()
        ui.server_close()


# ---- share ----
def _uid(server, name):
    import sqlite3
    c = sqlite3.connect(str(server.sessions.accounts_db))
    try:
        return c.execute("SELECT id FROM users WHERE username=?", (name,)).fetchone()[0]
    finally:
        c.close()


def test_share_requires_both_capabilities(server, fake):
    _register(server, fake.port)
    body = {"instance": "demo", "to": "x", "note": "hi"}
    assert _call(server.port, "/api/nodus/share", "POST", OUTSIDE, body)[0] == 403
    h = _cookie(server, "member1", "member")                       # message but no dev_ui
    assert _call(server.port, "/api/nodus/share", "POST", {**h, **OUTSIDE}, body)[0] == 403
    assert fake.connections == 0


def test_share_sends_exactly_one_message(server, fake):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    server.sessions.accounts.ensure_user("peer1", PASSWORD)
    me, peer = _uid(server, "dev1"), _uid(server, "peer1")
    # not a contact: the messaging module's own refusal
    status, _, body = _call(server.port, "/api/nodus/share", "POST", h, {"instance": "demo", "to": "peer1", "note": "look"})
    assert status == 403 and body["error"] == "you cannot message this user"
    db = messaging.connect(server.sessions.accounts_db)
    try:
        messaging.request_contact(db, me, "peer1")
        messaging.respond_contact(db, peer, "dev1", True)
    finally:
        db.close()
    status, _, body = _call(server.port, "/api/nodus/share", "POST", h, {"instance": "demo", "to": "peer1", "note": "look at this"})
    assert status == 200 and body["ok"] is True
    db = messaging.connect(server.sessions.accounts_db)
    try:
        rows = db.execute("SELECT * FROM msg_messages").fetchall()
    finally:
        db.close()
    assert len(rows) == 1
    row = rows[0]
    assert (row["subject_kind"], row["subject_ref"], row["msg_type"]) == ("topic", "nodus:demo@7:ab12cd34ef56", "note")
    assert "look at this" in row["body"] and "rev 7" in row["body"] and len(row["body"]) < 1000
    assert TOKEN not in row["body"] and "http" not in row["body"]
    status, _, body = _call(server.port, "/api/nodus/share", "POST", h, {"instance": "demo", "to": "peer1", "note": "x" * 501})
    assert status == 400


def test_client_repr_hides_token():
    assert TOKEN not in repr(ns.ConsoleClient("127.0.0.1", 1, TOKEN))

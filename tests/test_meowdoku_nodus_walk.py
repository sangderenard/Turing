"""Walking the live Nodus canvas: /api/nodus/map -> the real Living Data cursor handlers, over real HTTP on an ephemeral port (fake console)."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "examples" / "meowdoku" / "deploy")]

from tests.test_meowdoku_nodus_service import (  # noqa: E402,F401  (fixtures are used by name)
    GRAPH, _call, _cookie, _register, fake, ns, server)
import agent_walker  # noqa: E402
from agent_messenger import Client, _NoRedirect  # noqa: E402

SESS = "/api/living-data/sessions"
ROOT_ID = "nodus:demo/demo.nodus.txt"
A, B = ROOT_ID + "/m0", ROOT_ID + "/m1"


def _map(server, h):
    status, _, body = _call(server.port, "/api/nodus/map?instance=demo", headers=h)
    assert status == 200, body
    return body


def _exits(state):
    return {(e["to"], e["relationship"], e["direction"]) for e in state["exits"]}


def _move(server, h, sid, state, to, rel, direction, rid, revision=None):
    body = {"to": to, "relationship": rel, "direction": direction,
            "expected_revision": state["revision"] if revision is None else revision, "request_id": rid}
    return _call(server.port, f"{SESS}/{sid}/move", "POST", h, body)


def test_map_requires_dev_ui_and_matches_generic_graph(server, fake):
    _register(server, fake.port)
    assert _call(server.port, "/api/nodus/map?instance=demo", headers={"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https",
                                                                      "Host": "cats.example.org"})[0] == 403
    h = _cookie(server, "dev1", "operator")
    body = _map(server, h)
    assert body["instance"] == "demo" and body["revision"] == 7 and body["at"] == ROOT_ID and body["map"]["identity"] == ROOT_ID
    generic = {n["identity"]: n for n in ns.as_generic_graph({**GRAPH, "canvas": "demo.nodus.txt"}, "demo")}
    living = {body["map"]["identity"]} | {r["identity"] for r in body["map"]["regions"]}
    walkable = {i for i, n in generic.items() if n["kind"] in ("canvas", "module", "row", "contract-port")}
    assert living == walkable
    assert all(r["name"] for r in body["map"]["regions"])
    assert {r["name"] for r in body["map"]["regions"]} >= {"src", "sink", "row 0: gen", "row 1: scale", "cp"}
    assert _call(server.port, "/api/nodus/map?instance=nope", headers=h)[0] == 404


def test_real_rest_walk_over_the_canvas(server, fake):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    other = _cookie(server, "dev2", "operator")
    m = _map(server, h)
    create = {"map": m["map"], "at": m["at"], "request_id": "enter-1"}
    status, _, state = _call(server.port, SESS, "POST", h, create)
    assert status == 201 and state["at"]["identity"] == ROOT_ID
    sid = state["session_id"]
    url = f"{SESS}/{sid}"
    assert {(A, "contained-by", "incoming"), (B, "contained-by", "incoming")} <= _exits(state)

    status, _, state = _move(server, h, sid, state, A, "contained-by", "incoming", "m-a")
    assert status == 200 and state["at"]["name"] == "src"
    got = _exits(state)
    assert {(A + "/r0", "contained-by", "incoming"), (A + "/r1", "contained-by", "incoming"), (B, "feeds", "outgoing"),
            (ROOT_ID + "/cp0", "bound-on", "incoming"), (ROOT_ID, "contained-by", "outgoing")} <= got
    # retry with the same request id replays; the cursor does not move twice
    replay = _move(server, h, sid, state, B, "feeds", "outgoing", "m-b", revision=1)
    assert replay[0] == 200 and replay[2]["at"]["identity"] == B and replay[2]["revision"] == 2
    again = _move(server, h, sid, state, B, "feeds", "outgoing", "m-b", revision=1)
    assert again[0] == 200 and again[2]["replayed"] is True
    where = _call(server.port, url, headers=h)[2]
    assert where["at"]["identity"] == B and where["revision"] == 2
    # the reverse of the edge leads back
    assert (A, "feeds", "incoming") in _exits(where)
    # a stale expected_revision is refused and nothing moves
    assert _move(server, h, sid, where, A, "feeds", "incoming", "m-stale", revision=1)[0] == 409
    assert _call(server.port, url, headers=h)[2]["revision"] == 2
    back = _move(server, h, sid, where, A, "feeds", "incoming", "m-back")
    assert back[0] == 200 and back[2]["at"]["identity"] == A
    # accounts own their cursors
    assert _call(server.port, url, headers=other)[0] == 404
    assert _call(server.port, url + "/close", "POST", h, {"expected_revision": 3})[0] == 200


def test_walker_cli_drives_a_walk(server, fake, tmp_path, capsys):
    _register(server, fake.port)
    h = _cookie(server, "dev1", "operator")
    client = Client.__new__(Client)                       # an injected, already signed-in client: this server is plain http on loopback
    client.base, client.user = f"http://127.0.0.1:{server.port}", "dev1"
    client.opener = urllib.request.build_opener(_NoRedirect())
    client.opener.addheaders = [("Cookie", h["Cookie"])]
    state = str(tmp_path / "walk.json")
    base = ["--base", client.base, "--user", "dev1", "--state-file", state]

    def run(*words):
        capsys.readouterr()
        code = agent_walker.main(base + list(words), client=client)
        out = capsys.readouterr()
        return code, out.out, out.err

    code, out, _ = run("enter", "--instance", "demo")
    assert code == 0 and "demo.nodus.txt" in out and "1. contained-by incoming ->" in out
    code, out, _ = run("enter")
    assert code == 1 and "leave" in out
    code, out, _ = run("where")
    assert code == 0 and "src [" + A + "]" in out
    code, out, _ = run("go", "1")
    assert code == 0 and "You are at src" in out and "feeds outgoing -> sink [" + B + "]" in out
    code, out, _ = run("look", "--radius", "1")
    assert code == 0 and "src" in out and "row 0: gen" in out
    code, out, _ = run("go", "--to", B, "--relationship", "feeds", "--direction", "outgoing")
    assert code == 0 and "You are at sink" in out
    code, out, _ = run("--json", "where")
    assert json.loads(out)["at"]["identity"] == B
    code, out, _ = run("leave")
    assert code == 0 and "closed" in out
    code, _, err = run("where")
    assert code == 1 and "enter" in err

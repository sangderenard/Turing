"""Meowdoku share links and the view-only live stream: the share store, the viewer tokens and their enforcement."""

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import share  # noqa: E402


# ---------------------------------------------------------------- the share store
def test_store_round_trips_and_ids_are_content_hashes(tmp_path):
    store = share.ShareStore(tmp_path / "shares")
    view = {"v": 1, "g": "fountain3d", "c": {"shape": "hexagon"}, "l": {"gscale": 0.5}, "ls": {"meowdoku.pstyle": "heat"}}
    sid = store.put(view)
    assert share.ID_RE.match(sid)
    assert store.get(sid) == view
    assert store.put(dict(reversed(list(view.items())))) == sid          # key order does not matter: content addressed
    assert store.put({**view, "g": "fountain"}) != sid
    assert len(list((tmp_path / "shares").glob("*.json"))) == 2


def test_store_scrubs_credentials_and_foreign_keys(tmp_path):
    store = share.ShareStore(tmp_path)
    sid = store.put({"ls": {"meowdoku.pstyle": "heat", "meowdoku_player": "alice", "meowdoku_geo": "{}",
                            "meowdoku.lichess_token": "x", "meowdoku.api_key": "x", "other": "x", "meowdoku.n": 5}})
    assert store.get(sid)["ls"] == {"meowdoku.pstyle": "heat"}


def test_store_size_cap_and_count_cap(tmp_path):
    store = share.ShareStore(tmp_path, max_bytes=200, max_files=2)
    with pytest.raises(share.ShareError) as big:
        store.put({"ls": {"meowdoku.x": "y" * 500}})
    assert big.value.status == 413
    store.put({"g": "a"}); store.put({"g": "b"})
    with pytest.raises(share.ShareError) as full:
        store.put({"g": "c"})
    assert full.value.status == 507
    store.put({"g": "a"})                                                # already stored: still fine at the cap


@pytest.mark.parametrize("bad", ["", "../../etc/passwd", "ZZ" * 10, "0" * 19, None])
def test_store_rejects_malformed_ids(tmp_path, bad):
    with pytest.raises(share.ShareError) as e:
        share.ShareStore(tmp_path).get(bad)
    assert e.value.status == 404


# ---------------------------------------------------------------- the spectate registry (a fake clock)
class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def _state():
    return {"game": "fountain", "n": 4}


def test_viewer_cap_activity_and_expiry():
    clock = Clock()
    sp = share.Spectate(cap=3, clock=clock)
    token = sp.start("owner")
    assert sp.start("owner") == token                                    # one share per session
    vids = [sp.watch(token, None, None, _state)["vid"] for _ in range(3)]
    with pytest.raises(share.ShareError) as full:
        sp.watch(token, None, None, _state)
    assert full.value.status == 429
    assert sp.watch(token, vids[0], None, _state)["vid"] == vids[0]      # a known viewer keeps its place at the cap
    clock.t += share.ACTIVE_SECONDS + 1                                  # nobody polled: all inactive, room again
    assert sp.watch(token, None, None, _state)["viewers"] == 1
    clock.t += share.SHARE_TTL_SECONDS + 1                               # the hard time limit
    with pytest.raises(share.ShareError) as gone:
        sp.watch(token, None, None, _state)
    assert gone.value.status == 410


def test_revoke_and_session_end_kill_the_token():
    sp = share.Spectate()
    token = sp.start("a")
    sp.watch(token, None, None, _state)
    assert sp.stop("a") and not sp.stop("a")
    with pytest.raises(share.ShareError) as e:
        sp.watch(token, None, None, _state)
    assert e.value.status == 410
    token2 = sp.start("a")
    assert token2 != token
    sp.drop_owner("a")                                                   # the session expired
    with pytest.raises(share.ShareError):
        sp.watch(token2, None, None, _state)
    token3 = sp.start("b")
    with pytest.raises(share.ShareError):                                # the owner is gone from the session table
        sp.watch(token3, None, None, _state, owner_alive=lambda o: False)


def test_tokens_are_unguessable_and_state_is_sent_only_when_changed():
    sp = share.Spectate()
    tokens = {sp.start(f"o{i}") for i in range(20)}
    assert len(tokens) == 20 and all(len(t) >= 32 for t in tokens)
    token = sp.start("x")
    first = sp.watch(token, None, None, _state)
    assert first["state"] == _state()
    again = sp.watch(token, first["vid"], first["state_rev"], _state)
    assert "state" not in again and again["state_rev"] == first["state_rev"]
    sp.push_live("x", {"l": {"gscale": 0.3}, "ls": {"meowdoku.api_key": "no", "meowdoku.pstyle": "heat"}})
    live = sp.watch(token, first["vid"], first["state_rev"], _state)
    assert live["live"]["l"] == {"gscale": 0.3} and live["live"]["ls"] == {"meowdoku.pstyle": "heat"}
    quiet = sp.watch(token, first["vid"], first["state_rev"], _state, have_live=live["live_rev"])
    assert "live" not in quiet and "state" not in quiet                 # an unchanged view costs a few dozen bytes
    with pytest.raises(share.ShareError):
        sp.push_live("nobody", {})


# ---------------------------------------------------------------- the real handler
@pytest.fixture()
def server(tmp_path, monkeypatch):
    from meowdoku import server as srv
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "index.html").write_text("<html></html>", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield srv, sessions, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


class Client:
    def __init__(self, port, headers=None):
        self.port, self.headers, self.cookie = port, dict(headers or {}), None

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {**self.headers, **(headers or {})}
        if self.cookie:
            h["Cookie"] = self.cookie
        data = None
        if body is not None:
            data = json.dumps(body); h["Content-Type"] = "application/json"
        conn.request(method, path, data, h)
        r = conn.getresponse()
        raw = r.read()
        for value in r.headers.get_all("Set-Cookie") or []:
            if value.startswith("meowdoku_session"):
                self.cookie = value.split(";")[0]
        conn.close()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw


def test_end_to_end_viewer_is_read_only(server):
    srv, sessions, port = server
    owner = Client(port)
    status, st = owner.call("POST", "/api/new", {"game": "fountain"})
    assert status == 200 and st["game"] == "fountain"
    status, started = owner.call("POST", "/api/spectate", {"op": "start"})
    token = started["token"]
    assert status == 200 and started["cap"] == 16 and len(token) >= 32

    viewer = Client(port, {share.VIEWER_HEADER: token})
    status, first = viewer.call("GET", "/api/spectate/watch?rev=")
    assert status == 200 and first["state"]["game"] == "fountain"
    assert not (set(first["state"]) & set(share.PRIVATE_STATE_KEYS))
    status, second = viewer.call("GET", f"/api/spectate/watch?vid={first['vid']}&rev={first['state_rev']}")
    assert status == 200 and "state" not in second and second["stats"]["bytes"] > 0
    print("viewer bytes/s after two polls:", second["stats"])

    # the owner's control change reaches the viewer
    owner.call("POST", "/api/action", {"type": "fountain_mode", "mode": 1})
    import time as _t; _t.sleep(share.STATE_CACHE_SECONDS + 0.05)
    status, third = viewer.call("GET", f"/api/spectate/watch?vid={first['vid']}&rev={first['state_rev']}")
    assert third["state"]["mode"] == 1

    # every input / control / owner endpoint refuses the viewer header
    before = owner.call("GET", "/api/state")[1]
    refused = [("POST", "/api/new", {"game": "fountain"}), ("POST", "/api/action", {"type": "fountain_mode", "mode": 2}),
               ("POST", "/api/action", {"type": "tick"}), ("POST", "/api/profile", {"name": "x"}),
               ("POST", "/api/share", {"view": {"g": "x"}}), ("POST", "/api/spectate", {"op": "stop"}),
               ("POST", "/api/spectate", {"op": "start"}), ("POST", "/api/spectate/live", {"live": {}}),
               ("GET", "/api/state", None), ("GET", "/api/profile", None), ("GET", "/ws", None)]
    for method, path, body in refused:
        status, payload = viewer.call(method, path, body)
        assert status == 403, (method, path, status, payload)
    assert owner.call("GET", "/api/state")[1] == before                   # nothing reached the owner's table

    # the token is not a session: presented as the cookie it reaches nothing of the owner's
    thief = Client(port); thief.cookie = f"{srv.COOKIE}={token}"
    status, payload = thief.call("POST", "/api/action", {"type": "fountain_mode", "mode": 3})
    assert status == 400 and "no game" in payload["error"]
    assert owner.call("GET", "/api/state")[1] == before
    status, _ = Client(port).call("POST", "/api/spectate/live", {"live": {}})           # no session of its own to push to
    assert status == 409

    # the owner stops: the token is dead
    owner.call("POST", "/api/spectate", {"op": "stop"})
    assert viewer.call("GET", "/api/spectate/watch?rev=")[0] == 410


def test_session_expiry_ends_the_stream(server, monkeypatch):
    srv, sessions, port = server
    owner = Client(port)
    owner.call("POST", "/api/new", {"game": "fountain"})
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    viewer = Client(port, {share.VIEWER_HEADER: token})
    assert viewer.call("GET", "/api/spectate/watch?rev=")[0] == 200
    monkeypatch.setattr(srv, "SESSION_IDLE_SECONDS", -1)                  # every session is now idle past its limit
    Client(port).call("GET", "/api/catalog")                              # any new session purges the table
    status, payload = viewer.call("GET", "/api/spectate/watch?rev=")
    assert status == 410, payload


def test_share_endpoints_store_and_fetch(server):
    srv, sessions, port = server
    c = Client(port)
    view = {"v": 1, "g": "fountain", "l": {"gscale": 0.4}, "ls": {"meowdoku.pstyle": "heat", "meowdoku.token": "no"}}
    status, made = c.call("POST", "/api/share", {"view": view})
    assert status == 200
    status, got = c.call("GET", f"/api/share/{made['id']}")
    assert status == 200 and got["ls"] == {"meowdoku.pstyle": "heat"} and got["l"] == {"gscale": 0.4}
    assert c.call("GET", "/api/share/" + "0" * 20)[0] == 404
    assert c.call("POST", "/api/share", {"view": {"ls": {"meowdoku.x": "y" * 70000}}})[0] == 413
    assert c.call("POST", "/api/share", {"view": [1, 2]})[0] == 400

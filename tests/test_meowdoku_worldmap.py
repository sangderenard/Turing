"""Repository world map endpoints (examples/meowdoku/worldmap.py, server.py [WORLDMAP]) over an ephemeral loopback port and temp roots."""
from __future__ import annotations

import gzip
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import server as srv  # noqa: E402
from meowdoku import worldmap as wm  # noqa: E402

PASSWORD = "correct horse 12"
OUTSIDE = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}
COMMIT = "a" * 40
DIR = "owner__name"
MAP = {"schema": "repo-world-map.v1", "repo": "owner/name", "commit": COMMIT, "regions": [{"id": 0, "path": "", "parent": -1, "files": 1, "total_files": 1, "bytes": 5}],
       "rooms": [{"id": 0, "region": 0, "name": "a.py", "language": "python", "bytes": 5, "lines": 1, "out": 0, "in": 0}], "kinds": [], "portals": [], "region_links": []}


def _raw(port, path, method="GET", headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, headers=dict(headers or {}))
    r = conn.getresponse()
    body = r.read()
    out = (r.status, dict((k.lower(), v) for k, v in r.getheaders()), body)
    conn.close()
    return out


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    if data:
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    out, status, cookies = r.read(), r.status, r.msg.get_all("Set-Cookie") or []
    conn.close()
    try:
        return status, cookies, json.loads(out)
    except ValueError:
        return status, cookies, out.decode("utf-8", "replace")


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
    base = Path(sessions.state.content_dir) / "worldmap"
    (base / DIR / COMMIT).mkdir(parents=True)
    (base / DIR / COMMIT / "map.json").write_text(json.dumps(MAP) * 1, encoding="utf-8")
    (base / "index.json").write_text(json.dumps([{"repo": "owner/name", "dir": DIR, "commit": COMMIT, "generated": 1.0}]), encoding="utf-8")   # bare list form
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, tmp=Path(tmp_path))
    httpd.shutdown()
    httpd.server_close()


def _cookie(server, name, role):
    server.sessions.accounts.ensure_user(name, PASSWORD)
    if role != "member":
        authz.grant_role(server.sessions.accounts_db, name, role, "operator:test")
    status, cookies, _ = _call(server.port, "/api/account/login", "POST", None, {"username": name, "password": PASSWORD})
    assert status == 200
    return {"Cookie": next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))}


def test_refused_without_dev_ui(server):
    for path in ("/api/worldmap/index", "/api/worldmap/map?repo=%s&commit=%s" % (DIR, COMMIT)):
        assert _raw(server.port, path, headers=OUTSIDE)[0] == 403
    h = _cookie(server, "member1", "member")
    assert _raw(server.port, "/api/worldmap/index", headers={**h, **OUTSIDE})[0] == 403


def test_dev_gets_index_and_map(server):
    h = _cookie(server, "dev1", "operator")
    status, headers, body = _raw(server.port, "/api/worldmap/index", headers=h)
    assert status == 200 and headers["cache-control"] == "no-store"
    assert json.loads(body)["repos"][0]["dir"] == DIR                      # a bare list is accepted and normalised
    status, headers, body = _raw(server.port, "/api/worldmap/map?repo=%s&commit=%s" % (DIR, COMMIT), headers=h)
    assert status == 200 and headers["cache-control"] == "no-store" and "content-encoding" not in headers
    assert json.loads(body)["schema"] == "repo-world-map.v1"


def test_gzip_honoured(server):
    h = {**_cookie(server, "dev2", "operator"), "Accept-Encoding": "gzip"}
    status, headers, body = _raw(server.port, "/api/worldmap/map?repo=%s&commit=%s" % (DIR, COMMIT), headers=h)
    assert status == 200 and headers["content-encoding"] == "gzip" and int(headers["content-length"]) == len(body)
    assert json.loads(gzip.decompress(body))["rooms"][0]["name"] == "a.py"


@pytest.mark.parametrize("repo,commit", [
    ("..%2F..%2Fetc", COMMIT), ("owner__name%2F..", COMMIT), ("noseparator", COMMIT), ("..__..", COMMIT), ("owner__name", "A" * 40),
    ("owner__name", "abc"), ("owner__name", COMMIT + "0"), ("owner__name", "..%2F" + "a" * 36), ("", ""), ("owner__na%5Cme", COMMIT)])
def test_malformed_refused(server, repo, commit):
    h = _cookie(server, "dev3", "operator")
    assert _raw(server.port, "/api/worldmap/map?repo=%s&commit=%s" % (repo, commit), headers=h)[0] == 400


def test_missing_is_404(server):
    h = _cookie(server, "dev4", "operator")
    assert _raw(server.port, "/api/worldmap/map?repo=other__repo&commit=%s" % COMMIT, headers=h)[0] == 404
    assert _raw(server.port, "/api/worldmap/nope", headers=h)[0] == 404


def test_module_rules(tmp_path):
    paths = SimpleNamespace(content_dir=tmp_path)
    assert wm.index(paths) == {"repos": []}
    with pytest.raises(wm.WorldMapError) as e:
        wm.map_path(paths, "a__b", "f" * 40)
    assert e.value.status == 404
    with pytest.raises(wm.WorldMapError) as e:
        wm.map_path(paths, "..__..", "f" * 40)
    assert e.value.status == 400

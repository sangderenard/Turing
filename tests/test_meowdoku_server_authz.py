"""Server integration of proxy_trust.py, authz.py and lowering_library.py (examples/meowdoku/server.py), over an ephemeral loopback port."""
from __future__ import annotations

import hashlib
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

from meowdoku import proxy_trust  # noqa: E402
from meowdoku import server as srv  # noqa: E402

TOKEN = "t" * 40
PUBLIC_XFF = {"X-Forwarded-For": "8.8.8.8", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}
WASM = b"\0asm\1\0\0\0hand-made"


def _web(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "shader.html").write_text("<html>SHADER-PAGE</html>", encoding="utf-8")
    return web


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    hdrs = dict(headers or {})
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    raw = r.read()
    cookies = r.msg.get_all("Set-Cookie") or []
    ctype = r.getheader("Content-Type")
    conn.close()
    try:
        return r.status, json.loads(raw), cookies, ctype
    except ValueError:
        return r.status, raw, cookies, ctype


@pytest.fixture()
def server(tmp_path):
    web = _web(tmp_path)
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token=TOKEN, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    # a dns visitor is a "play" tier here, so account endpoints are reachable from a public origin (the default dns tier is shader)
    (web.parent / "access.json").write_text(json.dumps({"policy": {"dns": "play"}, "public_ip": []}), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, web=web)
    httpd.shutdown()
    httpd.server_close()


def _signup(port, name):
    status, _, cookies, _ = _call(port, "/api/account/signup", "POST", body={"username": name, "password": "correcthorse1"})
    assert status == 201
    return next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE))


def test_direct_peer_with_spoofed_forwarding_header_keeps_todays_classification(server):
    saved = set(proxy_trust.TRUSTED_PROXIES)
    proxy_trust.TRUSTED_PROXIES.clear()                    # the loopback test peer is not a trusted proxy
    try:
        status, who, _, _ = _call(server.port, "/api/whoami", headers={"X-Forwarded-For": "8.8.8.8", "X-Forwarded-Proto": "https"})
    finally:
        proxy_trust.TRUSTED_PROXIES.update(saved)
    assert status == 200 and who["origin"] == "lan" and who["tier"] == "edit" and who["role"] == "guest"
    assert who["edit"] is True and "owner_tools" in who["capabilities"]
    assert who["capabilities"] == sorted(who["capabilities"]) and "account_manage" not in who["capabilities"]


def test_trusted_proxy_forwarding_a_public_client_is_dns_and_guest(server):
    status, who, _, _ = _call(server.port, "/api/whoami", headers=PUBLIC_XFF)
    assert status == 200 and who["origin"] == "dns" and who["role"] == "guest"
    assert who["tier"] == "play" and who["capabilities"] == ["play", "view_site"] and who["edit"] is False
    status, who, _, _ = _call(server.port, "/api/whoami", headers={"Host": "127.0.0.1"})        # no X-Forwarded-For: a local direct caller
    assert who["origin"] == "lan"


def test_secure_cookie_follows_the_trusted_forwarded_scheme(server):
    body = {"username": "cat1", "password": "correcthorse1"}
    cookies = _call(server.port, "/api/account/signup", "POST", PUBLIC_XFF, body)[2]
    assert any(c.startswith(srv.ACCOUNT_COOKIE) and "; Secure" in c for c in cookies)
    cookies = _call(server.port, "/api/account/signup", "POST", {"Host": "127.0.0.1"}, {**body, "username": "cat2"})[2]
    assert any(c.startswith(srv.ACCOUNT_COOKIE) and "Secure" not in c for c in cookies)


def test_admin_bootstrap_then_owner_tools_from_a_dns_origin(server):
    port = server.port
    boss, plain = _signup(port, "boss"), _signup(port, "plain")
    local = {"Host": "127.0.0.1"}
    # the operator path: loopback-direct + the admin token; refused without the token and when forwarded
    assert _call(port, "/api/admin/account/grant", "POST", local, {"username": "boss"})[0] == 401
    assert _call(port, "/api/admin/account/grant", "POST", {**local, "X-Meowdoku-Admin": TOKEN, "X-Forwarded-For": "8.8.8.8"},
                 {"username": "boss"})[0] == 403
    status, out, _, _ = _call(port, "/api/admin/account/grant", "POST", {**local, "X-Meowdoku-Admin": TOKEN}, {"username": "boss"})
    assert status == 200 and out["admins"][0]["username"] == "boss"
    assert out["admins"][0]["granted_by"] == "operator:" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12]
    # the admin, arriving from a public (dns) origin through the proxy
    who = _call(port, "/api/whoami", headers={**PUBLIC_XFF, "Cookie": boss})[1]
    assert who["origin"] == "dns" and who["role"] == "admin" and who["edit"] is True
    assert {"owner_tools", "account_manage"} <= set(who["capabilities"]) and "lifecycle_admin" not in who["capabilities"]
    assert _call(port, "/api/account/me", headers={**PUBLIC_XFF, "Cookie": boss})[1]["role"] == "admin"
    member = _call(port, "/api/whoami", headers={**PUBLIC_XFF, "Cookie": plain})[1]
    assert member["role"] == "member" and "owner_tools" not in member["capabilities"] and "studio_write" in member["capabilities"]
    # account management: members and guests are refused; the admin grants and revokes
    for cookie in (plain, ""):
        status, out, _, _ = _call(port, "/api/account/admin/list", headers={**PUBLIC_XFF, "Cookie": cookie})
        assert status == 403 and out == {"error": "this needs account_manage"}
    admin_headers = {**PUBLIC_XFF, "Cookie": boss}
    status, out, _, _ = _call(port, "/api/account/admin/grant", "POST", admin_headers, {"username": "plain"})
    assert status == 200 and {a["username"]: a["granted_by"] for a in out["admins"]} == {
        "boss": "operator:" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12], "plain": "boss"}
    assert _call(port, "/api/account/admin/revoke", "POST", admin_headers, {"username": "plain"})[0] == 200
    assert [a["username"] for a in _call(port, "/api/account/admin/list", headers=admin_headers)[1]["admins"]] == ["boss"]
    assert _call(port, "/api/account/admin/grant", "POST", admin_headers, {"username": "nobody"})[0] == 404


def test_route_flag_sets_the_route_cookie_on_cookie_setting_responses(server):
    server.sessions.route = "a"
    status, _, cookies, _ = _call(server.port, "/api/account/signup", "POST", body={"username": "route_one", "password": "correcthorse1"})
    assert status == 201 and "meowdoku_route=a; Path=/; HttpOnly; SameSite=Lax" in cookies
    assert _call(server.port, "/api/whoami")[2] == []                                        # no cookie set, no route cookie
    server.sessions.route = ""
    assert not any(c.startswith("meowdoku_route") for c in _call(server.port, "/api/account/signup", "POST",
                                                                  body={"username": "route_two", "password": "correcthorse1"})[2])


def test_route_flag_is_parsed_and_validated():
    with pytest.raises(SystemExit):
        srv.main(["--route", "bad route", "--web-dir", str(ROOT / "no-such-web-dir")])


def _hand_made_product(web):
    (web / "products").mkdir()
    (web / "products" / "hand.api.yaml").write_text(
        "schema: turing-compiled-program-api-v1\nmodule: hand\nlanguage: wasm\nentry: f\nentry_points: []\nmetadata: {}\n",
        encoding="utf-8")
    (web / "products" / "hand.wasm").write_bytes(WASM)


def test_lowerings_list_and_checkout_checkin_round_trip(server):
    from meowdoku import lowering_library as L
    _hand_made_product(server.web)
    assert len(L.scan(server.web, server.sessions.accounts_db)) == 1
    port, sha = server.port, hashlib.sha256(WASM).hexdigest()
    status, listing, _, _ = _call(port, "/api/lowerings")
    assert status == 200 and [c["name"] for c in listing["cores"]] == ["hand"]
    low = listing["cores"][0]["lowerings"]
    assert len(low) == 1 and low[0]["backend"] == "wasm" and low[0]["product_sha256"] == sha and "bytes" not in json.dumps(listing)
    assert _call(port, "/api/lowerings", headers=PUBLIC_XFF)[0] == 200                      # view_site is held by the play tier

    cookie = _signup(port, "worker")
    me = {"Cookie": cookie}
    # no lease yet: the blob is refused; a guest from a public origin cannot check out at all
    assert _call(port, f"/api/lowerings/blob/{sha}", headers=me)[0] == 403
    assert _call(port, "/api/lowerings/checkout", "POST", PUBLIC_XFF, {"purpose": ["api", "hand"], "capabilities": ["wasm"]})[0] == 403
    assert _call(port, "/api/lowerings/checkout", "POST", me, {"purpose": ["api", "hand"], "capabilities": ["c"]})[0] == 404
    assert _call(port, "/api/lowerings/checkout", "POST", me, {"purpose": "hand", "capabilities": ["wasm"]})[0] == 400

    status, lease, _, _ = _call(port, "/api/lowerings/checkout", "POST", me,
                                {"purpose": ["api", "hand"], "capabilities": ["wasm", "c"], "prefer": ["wasm"], "seconds": 30})
    assert status == 200 and lease["product_sha256"] == sha and lease["backend"] == "wasm" and lease["generation"] == 1
    assert lease["byte_count"] == len(WASM) and lease["kind"] == "wasm" and "account_key" not in lease
    status, data, _, ctype = _call(port, f"/api/lowerings/blob/{sha}", headers=me)
    assert status == 200 and data == WASM and ctype == "application/wasm"
    other = {"Cookie": _signup(port, "other")}                                              # another account holds no lease
    assert _call(port, f"/api/lowerings/blob/{sha}", headers=other)[0] == 403
    bad = {"lease_id": lease["lease_id"], "generation": lease["generation"] + 1}
    assert _call(port, "/api/lowerings/checkin", "POST", me, bad)[0] == 409
    assert _call(port, "/api/lowerings/checkin", "POST", other, {"lease_id": lease["lease_id"], "generation": lease["generation"]})[0] == 409
    good = {"lease_id": lease["lease_id"], "generation": lease["generation"]}
    assert _call(port, "/api/lowerings/checkin", "POST", me, good)[1] == {"ok": True}
    assert _call(port, "/api/lowerings/checkin", "POST", me, good)[0] == 409                # returned once only
    assert _call(port, f"/api/lowerings/blob/{sha}", headers=me)[0] == 403                  # the lease is closed

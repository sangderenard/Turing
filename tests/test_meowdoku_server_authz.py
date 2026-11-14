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
from meowdoku import nodus_radio_hls  # noqa: E402

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


def test_radio_hls_uses_existing_view_site_gate_and_fixed_file_names(server):
    config = server.sessions.state.radio_json
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({'enabled': True, 'source': 'file',
        'jukebox': {'native': {'hls': {'enabled': True}}}}), encoding='utf-8')
    directory = nodus_radio_hls.output_dir(server.sessions.state)
    directory.mkdir(parents=True)
    (directory / 'radio.m3u8').write_bytes(b'#EXTM3U\n')
    (directory / 'radio_001.ts').write_bytes(b'aac-segment')
    status, body, _, content_type = _call(server.port, '/api/radio/hls/radio.m3u8', headers=PUBLIC_XFF)
    assert (status, body, content_type) == (200, b'#EXTM3U\n', 'application/vnd.apple.mpegurl')
    status, body, _, content_type = _call(server.port, '/api/radio/hls/radio_001.ts', headers=PUBLIC_XFF)
    assert (status, body, content_type) == (200, b'aac-segment', 'video/mp2t')
    assert _call(server.port, '/api/radio/hls/secret.txt', headers=PUBLIC_XFF)[0] == 404
    (server.web.parent / 'access.json').write_text(json.dumps({'policy': {'dns': 'shader'}}), encoding='utf-8')
    assert _call(server.port, '/api/radio/hls/radio.m3u8', headers=PUBLIC_XFF)[0] != 200


def test_direct_peer_with_spoofed_forwarding_header_keeps_todays_classification(server):
    saved = set(proxy_trust.TRUSTED_PROXIES)
    proxy_trust.TRUSTED_PROXIES.clear()                    # the loopback test peer is not a trusted proxy
    try:
        status, who, _, _ = _call(server.port, "/api/whoami", headers={"X-Forwarded-For": "8.8.8.8", "X-Forwarded-Proto": "https"})
    finally:
        proxy_trust.TRUSTED_PROXIES.update(saved)
    assert status == 200 and who["origin"] == "lan" and who["tier"] == "edit" and who["role"] == "guest"
    # a guest's caps are capped by role_caps['guest'] (owner, 2026-11-13): the LAN edit tier no longer means edit for a guest
    assert who["edit"] is False and who["capabilities"] == ["photo_read", "play", "view_site"]


def test_trusted_proxy_forwarding_a_public_client_is_dns_and_guest(server):
    status, who, _, _ = _call(server.port, "/api/whoami", headers=PUBLIC_XFF)
    assert status == 200 and who["origin"] == "dns" and who["role"] == "guest"
    assert who["tier"] == "play" and who["capabilities"] == ["photo_read", "play", "view_site"] and who["edit"] is False
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
    for cookie in (plain, ""):           # the guest gate answers as the handler's capability check would
        status, out, _, _ = _call(port, "/api/account/admin/list", headers={**PUBLIC_XFF, "Cookie": cookie})
        assert status == 403 and out == {"error": "this needs account_manage"}
    admin_headers = {**PUBLIC_XFF, "Cookie": boss}
    status, out, _, _ = _call(port, "/api/account/admin/grant", "POST", admin_headers, {"username": "plain"})
    assert status == 200 and {a["username"]: a["granted_by"] for a in out["admins"]} == {
        "boss": "operator:" + hashlib.sha256(TOKEN.encode()).hexdigest()[:12], "plain": "boss"}
    assert _call(port, "/api/account/admin/revoke", "POST", admin_headers, {"username": "plain"})[0] == 200
    assert [a["username"] for a in _call(port, "/api/account/admin/list", headers=admin_headers)[1]["admins"]] == ["boss"]
    assert _call(port, "/api/account/admin/grant", "POST", admin_headers, {"username": "nobody"})[0] == 404


def test_ranked_authority_endpoint_and_admin_grant(server):
    port = server.port
    boss, member = _signup(port, "rank_boss"), _signup(port, "rank_member")
    local = {"Host": "127.0.0.1"}
    assert _call(port, "/api/account/authority")[0] == 401
    assert _call(port, "/api/admin/account/grant", "POST", {**local, "X-Meowdoku-Admin": TOKEN},
                 {"username": "rank_boss"})[0] == 200
    admin_headers = {**PUBLIC_XFF, "Cookie": boss}
    member_headers = {**PUBLIC_XFF, "Cookie": member}
    assert _call(port, "/api/account/authority", headers=member_headers)[1]["groups"] == ["member"]
    body = {"username": "rank_member", "group": "dev", "enabled": True}
    assert _call(port, "/api/account/authority/grant", "POST", member_headers, body)[0] == 403
    assert _call(port, "/api/account/authority/grant", "POST", {**admin_headers, "Origin": "https://elsewhere.example"}, body)[0] == 403
    status, result, _, _ = _call(port, "/api/account/authority/grant", "POST", admin_headers, body)
    assert status == 200 and result["authority"]["groups"] == ["member", "dev"]
    assert _call(port, "/api/account/authority", headers=member_headers)[1]["highest"] == "dev"
    assert _call(port, "/api/whoami", headers=member_headers)[1]["role"] == "dev"
    assert _call(port, "/api/account/authority/grant", "POST", admin_headers, {**body, "enabled": False})[0] == 200
    assert _call(port, "/api/account/authority", headers=member_headers)[1]["groups"] == ["member"]


def test_nodus_action_requires_admin_granted_group(server):
    port = server.port
    boss, member = _signup(port, "nodus_boss"), _signup(port, "nodus_member")
    assert _call(port, "/api/admin/account/grant", "POST", {"Host": "127.0.0.1", "X-Meowdoku-Admin": TOKEN},
                 {"username": "nodus_boss"})[0] == 200
    admin_headers = {**PUBLIC_XFF, "Cookie": boss}
    member_headers = {**PUBLIC_XFF, "Cookie": member}
    action = {"instance": "missing-instance", "action": {"op": "viewport", "expected_revision": 0, "width": 800, "height": 600}}
    assert _call(port, "/api/nodus/action", "POST", member_headers, action)[0] == 403
    grant = {"username": "nodus_member", "group": "nodus", "enabled": True}
    assert _call(port, "/api/account/authority/grant", "POST", admin_headers, grant)[0] == 200
    assert _call(port, "/api/account/authority", headers=member_headers)[1]["groups"] == ["member", "nodus"]
    assert _call(port, "/api/nodus/action", "POST", {**member_headers, "Origin": "https://elsewhere.example"}, action)[0] == 403
    assert _call(port, "/api/nodus/action", "POST", member_headers, action)[0] != 403
    assert _call(port, "/api/paper/dm", "POST", member_headers,
                 {"verb": "use", "target": "nodus_ui", "input": action})[0] != 403
    assert _call(port, "/api/account/authority/grant", "POST", admin_headers, {**grant, "enabled": False})[0] == 200
    assert _call(port, "/api/nodus/action", "POST", member_headers, action)[0] == 403


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


def test_an_intern_is_answered_only_on_the_kiosk_endpoints(server):
    port = server.port
    cookie = _signup(port, "kiosk_cat")
    srv.authz_mod.grant_role(server.sessions.accounts_db, "kiosk_cat", "intern", "operator:test")
    local = {"Host": "127.0.0.1", "Cookie": cookie}             # a LAN origin is the edit tier: it adds nothing to an intern
    public = {**PUBLIC_XFF, "Cookie": cookie}
    for headers in (local, public):
        who = _call(port, "/api/whoami", headers=headers)[1]
        assert who["role"] == "intern" and who["kiosk"] is True and who["capabilities"] == ["photo_read", "view_site"]
        # writes, messaging, lifecycle and handlers that ask for no capability: refused before any handler runs
        for method, path, body in (("POST", "/api/reports", {"request_id": "r1"}), ("POST", "/api/relay/send", {}),
                                   ("GET", "/api/relay/addresses", None), ("GET", "/api/account/groups", None),
                                   ("POST", "/api/account/groups", {"name": "g"}), ("POST", "/api/messages/send", {}),
                                   ("POST", "/api/tickets", {}), ("POST", "/api/admin/lifecycle", {}), ("GET", "/api/log", None),
                                   ("GET", "/api/paper/account", None), ("POST", "/api/account/signup", {"username": "x"}),
                                   # no personal storage: material is burned on delivery
                                   ("GET", "/api/user-files", None), ("POST", "/api/user-files", {}),
                                   ("POST", "/api/relay/open", {}),
                                   ("GET", "/", None)):
            status, out, _, _ = _call(port, path, method, headers, body)
            assert status == 403 and "kiosk" in out["error"], (method, path, status, out)
        assert _call(port, "/", "HEAD", headers)[0] == 403
    if getattr(srv.shared_catalog, "kiosk_endpoint", None) is None:
        pytest.skip("the loaded SpeakToMe catalog declares no kiosk entries (set SPEAKTOME_REPO to a checkout that does)")
    assert _call(port, "/api/account/me", headers=public)[0] == 200
    index = _call(port, "/api/index", headers=public)[1]
    kiosk = {e["name"] for e in srv.shared_catalog.ENDPOINTS if e["kiosk"]}
    assert index["kiosk"] is True and {e["name"] for e in index["endpoints"] if e["allowed"]} == kiosk
    member_index = _call(port, "/api/index", headers={**PUBLIC_XFF, "Cookie": _signup(port, "plain_cat")})[1]
    assert member_index["kiosk"] is False and "messages_send" in {e["name"] for e in member_index["endpoints"] if e["allowed"]}
    # a kiosk path still meets its handler's own gate: the system map needs dev_ui
    status, out, _, _ = _call(port, "/site/system/", headers=public)
    assert status == 403 and out == {"error": "this needs dev_ui"}
    assert _call(port, "/api/account/logout", "POST", public, {})[0] == 200
    assert _call(port, "/api/whoami", headers=public)[1]["role"] == "guest"


def test_a_guest_keeps_pages_game_and_machine_routes_and_nothing_else(server):
    port = server.port
    if getattr(srv.shared_catalog, "guest_endpoint", None) is None:
        pytest.skip("the loaded SpeakToMe catalog declares no guest entries (set SPEAKTOME_REPO to a checkout that does)")
    local = {"Host": "127.0.0.1"}                                 # an anonymous LAN visitor: once the edit tier, now a guest
    assert _call(port, "/api/whoami", headers=local)[1]["guest"] is True        # the foyer: speaktome_mcp reads it as a visitor
    status, page, _, ctype = _call(port, "/shader.html", headers=local)
    assert status == 200 and b"SHADER-PAGE" in page                # pages are answered by the static handler
    for path in ("/api/whoami", "/api/route", "/api/state", "/api/catalog"):
        assert _call(port, path, headers=local)[0] == 200, path
    # writes and everything outside the guest set: refused before any handler runs, from any origin
    for headers in (local, PUBLIC_XFF):
        for method, path, body in (("POST", "/api/reports", {"request_id": "r1"}), ("POST", "/api/relay/send", {}),
                                   ("POST", "/api/tickets", {}), ("POST", "/api/obj_assets", {}), ("POST", "/api/marble_photo", {}),
                                   ("POST", "/api/loading_library/activate", {}), ("POST", "/api/messages/send", {}),
                                   ("GET", "/api/log", None), ("GET", "/api/user-files", None), ("GET", "/api/relay/addresses", None)):
            status, out, _, _ = _call(port, path, method, headers, body)
            # answered as the handler would first: a capability the guest lacks (403), else sign in (401)
            assert (status, out["error"][:9]) in ((403, "this need"), (401, "sign in f")), (method, path, status, out)
        assert _call(port, "/api/state", "HEAD", headers)[0] == 403
        assert _call(port, "/api/relay/send", "POST", headers, {})[0] == 401
        assert _call(port, "/api/obj_assets", "POST", headers, {})[1] == {"error": "this needs studio_write"}
    # a machine route is answered by its own credential: the loopback admin token, not a tier capability
    status, out, _, _ = _call(port, "/api/admin/lifecycle", "POST", local, {"op": "status"})
    assert status == 401 and out == {"error": "bad admin token"}
    status, out, _, _ = _call(port, "/api/admin/lifecycle", "POST", {**local, "X-Meowdoku-Admin": TOKEN, "X-Forwarded-For": "8.8.8.8"},
                              {"op": "status"})
    assert status == 403 and "localhost only" in out["error"]
    # sockets are declared in access.py: the game's pass the gate, an undeclared one does not
    assert _call(port, "/ws/other", headers=local)[0] == 401
    status, out, _, _ = _call(port, "/ws/contract/abc", headers=local)        # no upgrade, no grant: the bridge itself refuses
    assert not (isinstance(out, dict) and str(out.get("error", "")).startswith("sign in for this"))
    index = _call(port, "/api/index", headers=local)[1]
    reachable = {e["name"] for e in srv.shared_catalog.ENDPOINTS if srv.shared_catalog.reachable_by_guest(e)}
    assert index["guest"] is True and {e["name"] for e in index["endpoints"] if e["allowed"]} <= reachable
    assert "play_action" in {e["name"] for e in index["endpoints"] if e["allowed"]}

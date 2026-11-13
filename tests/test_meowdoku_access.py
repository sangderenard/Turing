"""Access tiers (examples/meowdoku/access.py): origin classification, the policy file, and the server's enforcement."""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import access as A  # noqa: E402
from meowdoku import server as srv  # noqa: E402

PUBLIC = A.parse_ip("173.195.26.104")


# ---------------------------------------------------------------- classification
@pytest.mark.parametrize("peer,host,expected", [
    ("127.0.0.1", "173.195.26.104:8766", "lan"),            # loopback
    ("::1", "localhost:8766", "lan"),
    ("::ffff:127.0.0.1", "x.example", "lan"),               # IPv4-mapped loopback
    ("192.168.1.20", "173.195.26.104:8766", "lan"),         # RFC1918
    ("10.1.2.3", "x.example", "lan"),
    ("172.16.0.9", "x.example", "lan"),
    ("172.32.0.9", "173.195.26.104", "ip"),                 # just outside 172.16/12
    ("169.254.9.9", "x.example", "lan"),                    # link-local
    ("fe80::1%12", "x.example", "lan"),                     # link-local v6 with a zone id
    ("fd12:3456::1", "x.example", "lan"),                   # ULA
    ("173.195.26.104", "173.195.26.104:8766", "lan"),       # the hairpin: the server's own public IP
    ("::ffff:173.195.26.104", "x.example", "lan"),
    ("8.8.8.8", "173.195.26.104:8766", "ip"),               # public peer, IP literal host
    ("8.8.8.8", "[2001:db8::1]:8766", "ip"),                # IPv6 literal Host (bracketed, with port)
    ("8.8.8.8", "[2001:db8::1]", "ip"),
    ("8.8.8.8", "2001:db8::1", "ip"),                       # bare IPv6 literal
    ("8.8.8.8", "mycats.duckdns.org", "dns"),               # a hostname
    ("8.8.8.8", "MyCats.DuckDNS.org.:8766", "dns"),         # case, trailing dot, port
    ("8.8.8.8", "localhost", "dns"),
    ("8.8.8.8", "2130706433", "ip"),                        # a number is an address to a browser, not a name
    ("8.8.8.8", None, "ip"),                                # no Host
    ("2001:db8::5", "x.example", "dns"),                    # public v6 peer
])
def test_classify(peer, host, expected):
    assert A.classify(peer, host, (PUBLIC,)) == expected


def test_hairpin_needs_the_public_ip():
    assert A.classify("173.195.26.104", "173.195.26.104", ()) == "ip"      # not learned yet (fails soft)


def test_forwarding_headers_are_ignored():
    """From a peer that is not a trusted proxy, a spoofed X-Forwarded-For can neither grant nor remove LAN (proxy_trust.py)."""
    from meowdoku import proxy_trust
    saved = set(proxy_trust.TRUSTED_PROXIES)
    proxy_trust.TRUSTED_PROXIES.clear()                    # the loopback test peer is an ordinary, untrusted peer here
    sessions = srv.Sessions(_web(Path(__import__("tempfile").mkdtemp())), None)
    sessions.access = A.Access(None)
    sessions.access._found, sessions.access._found_at = PUBLIC, time.time()
    sessions.access.config()["policy"].update({"lan": "edit", "ip": "dev"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        # a remote-looking Host with a LAN-looking header from loopback: still the loopback peer -> lan
        who = _get(port, "/api/whoami", {"X-Forwarded-For": "8.8.8.8", "Forwarded": "for=8.8.8.8", "X-Real-IP": "8.8.8.8"})[1]
        assert who["origin"] == "lan"
        # and the signature of the function takes no headers at all
        assert A.classify("8.8.8.8", "1.2.3.4", (PUBLIC,)) == "ip"
    finally:
        proxy_trust.TRUSTED_PROXIES.update(saved)
        httpd.shutdown(); httpd.server_close()


# ---------------------------------------------------------------- the policy
def test_policy_defaults_and_overrides(tmp_path):
    assert A.load_config(None)["policy"] == {"lan": "edit", "ip": "dev", "dns": "shader"}
    path = tmp_path / "access.json"
    assert A.Access(path).config()["policy"] == A.DEFAULT_POLICY                       # missing file
    path.write_text(json.dumps({"policy": {"ip": "shader", "dns": "play", "lan": "bogus", "moon": "edit"}}), encoding="utf-8")
    cfg = A.load_config(path)
    assert cfg["policy"] == {"lan": "edit", "dns": "play", "ip": "shader"}              # bad entries ignored, the rest applied
    path.write_text("{not json", encoding="utf-8")
    assert A.load_config(path)["policy"] == A.DEFAULT_POLICY                           # unreadable -> defaults


def test_policy_file_is_reread_on_change(tmp_path):
    path = tmp_path / "access.json"
    access = A.Access(path)
    assert access.decide("8.8.8.8", "1.2.3.4") == ("ip", "dev")
    assert access.decide("8.8.8.8", "cats.example.org") == ("dns", "shader")
    assert access.decide("127.0.0.1", "cats.example.org") == ("lan", "edit")
    path.write_text(json.dumps({"policy": {"ip": "shader"}}), encoding="utf-8")
    assert access.decide("8.8.8.8", "1.2.3.4") == ("ip", "shader")


def test_public_ip_lookup_is_cached_and_fails_soft():
    calls = []
    now = [1000.0]

    def lookup(url):
        calls.append(url)
        return PUBLIC if len(calls) > 1 else None                                     # the first answer fails

    access = A.Access(None, lookup=lookup, clock=lambda: now[0])
    access.public_ips()
    for _ in range(100):
        if calls and not access._busy:
            break
        time.sleep(0.01)
    assert access.public_ips() == () and len(calls) == 1                              # failed: nothing guessed, not retried at once
    now[0] += A.RETRY_SECONDS + 1
    access.public_ips()
    for _ in range(100):
        if len(calls) > 1 and not access._busy:
            break
        time.sleep(0.01)
    assert access.public_ips() == (PUBLIC,) and calls[-1] == A.DEFAULT_SERVICE
    n = len(calls)
    now[0] += 10
    access.public_ips()
    assert len(calls) == n                                                            # cached inside the ttl


def test_configured_public_ip_skips_the_lookup(tmp_path):
    path = tmp_path / "access.json"
    path.write_text(json.dumps({"public_ip": "173.195.26.104"}), encoding="utf-8")
    access = A.Access(path, lookup=lambda url: pytest.fail("no lookup when configured"))
    assert access.public_ips() == (PUBLIC,)
    assert access.decide("173.195.26.104", "173.195.26.104")[0] == "lan"


# ---------------------------------------------------------------- the server
def _web(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "shader.html").write_text("<html>SHADER-PAGE</html>", encoding="utf-8")
    (web / "index.html").write_text("<html>GAME-PAGE</html>", encoding="utf-8")
    (web / "loading_panel.js").write_text("// panel", encoding="utf-8")
    (web / "secret.json").write_text("{}", encoding="utf-8")
    return web


def _get(port, path, headers=None, method="GET", body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request(method, path, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data.decode("utf-8", "replace")


@pytest.fixture()
def tiered(tmp_path):
    """A server on loopback whose policy maps the loopback caller (origin lan) to whatever the test sets."""
    web = _web(tmp_path)
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = None

    def set_tier(tier):
        (web.parent / "access.json").write_text(json.dumps({"policy": {"lan": tier}, "public_ip": []}), encoding="utf-8")
        sessions.access._stamp = object()                                              # force a re-read even within one mtime tick

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], set_tier, sessions
    httpd.shutdown()
    httpd.server_close()


def _member(sessions, name="tier_cat"):
    """A signed-in account's cookie: a visitor with no account is a guest, held to view_site/photo_read/play on every tier (owner, 2026-11-13)."""
    return {"Cookie": f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(sessions.accounts.ensure_user(name, 'correcthorse1'))}"}


def test_whoami_reports_the_tier_and_nothing_else(tiered):
    port, set_tier, sessions = tiered
    member = _member(sessions)
    for tier in ("edit", "dev", "play", "shader"):
        set_tier(tier)
        status, who = _get(port, "/api/whoami", {"X-Secret": "abc"})
        assert status == 200 and who["tier"] == tier and who["origin"] == "lan"
        assert set(who) == {"origin", "tier", "dev", "edit", "shader", "role", "capabilities", "instance", "signed_in", "username", "origin_tier",
                            "kiosk", "guest"}
        assert who["edit"] is False and who["dev"] is False and who["shader"] == (tier == "shader")      # a guest, on any tier
        if tier != "shader":                                                                     # an account on shader is lifted to play
            who = _get(port, "/api/whoami", member)[1]
            assert who["tier"] == tier and who["edit"] == (tier == "edit") and who["dev"] == (tier in ("edit", "dev"))


def test_shader_tier_gets_the_shader_page_or_403(tiered):
    port, set_tier, _ = tiered
    set_tier("shader")
    for path in ("/", "/index.html", "/app.js", "/boards.json", "/secret.json", "/whatever/else"):
        status, body = _get(port, path)
        assert status == 200 and "SHADER-PAGE" in body, path                              # the game page and its files are not served
    assert _get(port, "/loading_panel.js")[1] == "// panel"                              # its own asset is
    for path in ("/api/state", "/api/catalog", "/api/profile", "/api/lifecycle", "/api/spectate/watch", "/api/share/x", "/ws"):
        assert _get(port, path)[0] == 403, path
    for path in ("/api/new", "/api/action", "/api/admin/lifecycle", "/api/loading_anim", "/api/clientlog", "/api/share"):
        assert _get(port, path, method="POST", body=b"{}")[0] == 403, path
    assert _get(port, "/", method="HEAD")[0] == 403
    status, ready = _get(port, "/api/ready")
    assert status == 200 and set(ready) == {"loading_anim"}                              # no product/build state for this tier
    assert _get(port, "/api/loading_anim")[0] == 404                                      # (none recorded here) but reachable, not 403


def test_normal_tiers_serve_the_game(tiered):
    port, set_tier, _ = tiered
    for tier in ("edit", "dev", "play"):
        set_tier(tier)
        assert "GAME-PAGE" in _get(port, "/index.html")[1]
        assert _get(port, "/api/catalog")[0] == 200


OWNER_ACTIONS = ("demo_learning", "arena_save", "arena_stop", "arena_resume", "lichess_pause", "lichess_resume", "pilot_arch")


def test_edit_endpoints_are_refused_below_edit(tiered):
    port, set_tier, sessions = tiered
    for tier in ("play", "dev", "shader"):
        set_tier(tier)
        # the lifecycle is answered by its own loopback + admin token check (catalog credential admin_token), here with no lifecycle
        # configured (503); the shader tier refuses every POST before it
        assert _get(port, "/api/admin/lifecycle", method="POST", body=b"{}", headers={"X-Meowdoku-Admin": "x"})[0] == (
            403 if tier == "shader" else 503)
        assert _get(port, "/api/loading_anim", method="POST", body=b"xx", headers={"X-Meowdoku-Loading": "1"})[0] == 403
        if tier != "shader":
            for kind in OWNER_ACTIONS:
                status, body = _get(port, "/api/action", method="POST", body=json.dumps({"type": kind}).encode())
                assert status == 403 and body["error"] == "this needs owner_tools", (tier, kind)
    assert sessions.anim.info() is None


def test_edit_tier_keeps_the_token_checks(tiered):
    port, set_tier, sessions = tiered
    set_tier("edit")
    member = _member(sessions)
    # the tier gate passes, then the old checks still apply: the lifecycle admin token, the loading header
    sessions.lifecycle = type("L", (), {"token": "t" * 40})()
    assert _get(port, "/api/admin/lifecycle", method="POST", body=b"{}", headers={"Host": f"127.0.0.1:{port}"})[0] == 401
    assert _get(port, "/api/admin/lifecycle", method="POST", body=b"{}", headers={"Host": "evil.example", "X-Meowdoku-Admin": "t" * 40})[0] == 403
    assert _get(port, "/api/loading_anim", method="POST", body=b"xx", headers={**member, "X-Meowdoku-Loading": "0"})[0] == 403
    status, body = _get(port, "/api/action", method="POST", body=json.dumps({"type": "arena_save"}).encode(), headers=member)
    assert status == 400 and "no game" in body["error"]
    status, body = _get(port, "/api/action", method="POST", body=json.dumps({"type": "arena_save"}).encode())
    assert status == 403 and body["error"] == "this needs owner_tools"                       # a guest on the edit tier is not an owner                                   # past the tier gate: the action's own refusal


def test_require_tier_is_generic():
    assert A.at_least("edit", "edit") and A.at_least("edit", "dev") and A.at_least("dev", "play")
    assert not A.at_least("dev", "edit") and not A.at_least("play", "dev") and not A.at_least("shader", "play")

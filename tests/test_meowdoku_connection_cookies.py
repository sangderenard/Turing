"""One handler serves one connection, and behind Apache one backend connection carries many visitors' requests in turn:
nothing one request was going to set as a cookie may reach the next request's answer (server.py handle_one_request)."""
import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import server as srv  # noqa: E402
from test_meowdoku_access import _web  # noqa: E402


@pytest.fixture()
def served(tmp_path):
    sessions = srv.Sessions(_web(tmp_path), None)
    sessions.lifecycle = None
    sessions.accounts.ensure_user("agentbot", "correcthorse1")
    owner = sessions.accounts.ensure_user("ownercat", "correcthorse1")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(owner)}"
    httpd.shutdown()
    httpd.server_close()


def _cookies(response):
    return [v.split(";")[0] for k, v in response.getheaders() if k.lower() == "set-cookie"]


@pytest.mark.parametrize("path", ["/api/chat/rooms", "/api/chat/rooms/agents", "/api/whoami", "/api/newsletter", "/api/state"])
@pytest.mark.parametrize("visitor", ["guest", "owner"])
def test_a_sign_in_cookie_never_reaches_the_next_visitor_on_the_connection(served, path, visitor):
    # Measured 2026-11-15: the agent account's login, then the owner's GET /api/chat/rooms on the same Apache backend
    # connection, was answered with the agent's meowdoku_account cookie, and the owner's browser became the agent.
    port, owner_cookie = served
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", "/api/account/login", body=json.dumps({"username": "agentbot", "password": "correcthorse1"}),
                     headers={"Content-Type": "application/json"})
        signed_in = conn.getresponse()
        signed_in.read()
        agent = [c for c in _cookies(signed_in) if c.startswith(srv.ACCOUNT_COOKIE + "=")]
        assert signed_in.status == 200 and agent, "the agent's own answer carries its session"
        conn.request("GET", path, headers={"Cookie": owner_cookie} if visitor == "owner" else {})
        answer = conn.getresponse()
        answer.read()
        assert agent[0] not in _cookies(answer), f"{path} handed the agent's session to the {visitor}"
        assert not [c for c in _cookies(answer) if c.startswith(srv.ACCOUNT_COOKIE + "=")], "no account cookie is set here"
    finally:
        conn.close()

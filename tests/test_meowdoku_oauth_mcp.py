"""An OAuth code, bound to PKCE and /mcp, reaches the real MCP and REST handlers."""
import base64
import hashlib
import http.client
import json
import sys
import threading
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import server as srv  # noqa: E402
from test_meowdoku_access import _web  # noqa: E402


def test_oauth_code_pkce_and_http_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKTOME_PUBLIC_ORIGIN", "https://stichka.duckdns.org:8766")
    sessions = srv.Sessions(_web(tmp_path), None)
    sessions.accounts.ensure_user("agentbot", "correcthorse1")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=30)
    proxy = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https",
             "Host": "stichka.duckdns.org:8766"}

    def call(method, path, body=None, headers=None):
        conn.request(method, path, body=body, headers={**proxy, **(headers or {})})
        response = conn.getresponse()
        payload = response.read()
        return response, payload

    try:
        response, body = call("GET", "/.well-known/oauth-protected-resource/mcp")
        assert response.status == 200
        assert json.loads(body)["authorization_servers"] == ["https://stichka.duckdns.org:8766"]
        response, body = call("POST", "/oauth/register", json.dumps({"client_name": "Test client",
            "redirect_uris": ["http://127.0.0.1:9876/callback"]}), {"Content-Type": "application/json"})
        assert response.status == 201, body
        client = json.loads(body)["client_id"]
        verifier = "x" * 43
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        query = urllib.parse.urlencode({"client_id": client, "redirect_uri": "http://127.0.0.1:9876/callback",
            "response_type": "code", "state": "state-1", "scope": "mcp", "resource": sessions.oauth_mcp.resource,
            "code_challenge_method": "S256", "code_challenge": challenge})
        response, body = call("GET", "/oauth/authorize?" + query)
        assert response.status == 200, body
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        nonce = cookie.split("=", 1)[1]
        form = urllib.parse.urlencode({"nonce": nonce, "decision": "allow", "username": "agentbot",
                                       "password": "correcthorse1"})
        response, body = call("POST", "/oauth/authorize", form,
                              {"Content-Type": "application/x-www-form-urlencoded", "Cookie": cookie})
        assert response.status == 302, body
        location = response.getheader("Location")
        code = urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)["code"][0]
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)["state"] == ["state-1"]
        wrong = urllib.parse.urlencode({"grant_type": "authorization_code", "client_id": client, "code": code,
            "redirect_uri": "http://127.0.0.1:9876/callback", "resource": sessions.oauth_mcp.resource,
            "code_verifier": "y" * 43})
        response, _ = call("POST", "/oauth/token", wrong, {"Content-Type": "application/x-www-form-urlencoded"})
        assert response.status == 400
        right = wrong.replace("y" * 43, verifier)
        response, body = call("POST", "/oauth/token", right, {"Content-Type": "application/x-www-form-urlencoded"})
        assert response.status == 200, body
        token = json.loads(body)["access_token"]
        response, _ = call("POST", "/oauth/token", right, {"Content-Type": "application/x-www-form-urlencoded"})
        assert response.status == 400
        accept = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}})
        response, _ = call("POST", "/mcp", request, accept)
        assert response.status == 401
        assert "resource_metadata" in response.getheader("WWW-Authenticate")
        accept["Authorization"] = "Bearer " + token
        response, body = call("POST", "/mcp", request, accept)
        assert response.status == 200, body
        assert json.loads(body)["result"]["protocolVersion"] == "2025-11-25"
        session_id = response.getheader("MCP-Session-Id")
        assert session_id
        accept["MCP-Session-Id"] = session_id
        accept["MCP-Protocol-Version"] = "2025-11-25"
        request = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                              "params": {"name": "whoami", "arguments": {}}})
        response, _ = call("POST", "/mcp", request, {k: v for k, v in accept.items() if k != "MCP-Session-Id"})
        assert response.status == 400
        response, body = call("POST", "/mcp", request, accept)
        assert response.status == 200, body
        text = json.loads(body)["result"]["content"][0]["text"]
        identity = json.loads(text)
        assert identity["username"] == "agentbot"
        assert identity["origin"] == "dns" and identity["tier"] == "play"
        response, _ = call("POST", "/mcp", request, {**accept, "Origin": "https://evil.example"})
        assert response.status == 403
        response, _ = call("POST", "/mcp", request, {**accept, "MCP-Protocol-Version": "invalid"})
        assert response.status == 400
        response, _ = call("DELETE", "/mcp", headers=accept)
        assert response.status == 204
        response, _ = call("POST", "/mcp", request, accept)
        assert response.status == 404
    finally:
        conn.close()
        httpd.shutdown()
        httpd.server_close()

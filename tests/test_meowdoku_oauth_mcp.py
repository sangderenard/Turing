"""An OAuth code, bound to PKCE and /mcp, reaches the real MCP and REST handlers."""
import base64
import hashlib
import http.client
import json
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "tests")]

from meowdoku import server as srv  # noqa: E402
from meowdoku import authz, oauth_mcp  # noqa: E402
from test_meowdoku_access import _web  # noqa: E402


def test_oauth_code_pkce_and_http_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEAKTOME_PUBLIC_ORIGIN", "https://stichka.duckdns.org:8766")
    tls_dir = tmp_path / "build" / "meowdoku" / "tls"
    tls_dir.mkdir(parents=True)
    cert, key = tls_dir / "cert.pem", tls_dir / "key.pem"
    openssl = shutil.which("openssl") or "C:/Program Files/Git/usr/bin/openssl.exe"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                    "-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost"], check=True, capture_output=True)
    (tls_dir / "ca.crt").write_bytes(cert.read_bytes())
    sessions = srv.Sessions(_web(tmp_path), None)
    sessions.accounts.ensure_user("agentbot", "correcthorse1")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert), str(key))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    conn = http.client.HTTPSConnection("127.0.0.1", httpd.server_port, timeout=30,
                                      context=ssl.create_default_context(cafile=str(tls_dir / "ca.crt")))
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

        # The public MCP URL creates a temporary identity without exposing its
        # internal session credential. Tests and opinions remain unpublished.
        guest_headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        initialize = json.dumps({"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "guest", "version": "1"}}})
        response, body = call("POST", "/mcp/guest", initialize, guest_headers)
        assert response.status == 200, body
        guest_headers["MCP-Session-Id"] = response.getheader("MCP-Session-Id")
        assert guest_headers["MCP-Session-Id"]
        # Several connectors behind one forwarded address must not exhaust a
        # five-session hourly budget during ordinary tool discovery.
        for _ in range(6):
            response, body = call("POST", "/mcp/guest", initialize,
                                  {k: v for k, v in guest_headers.items() if k != "MCP-Session-Id"})
            assert response.status == 200, body
        monkeypatch.setattr(oauth_mcp, "MAX_GUEST_SESSIONS", 7)
        response, body = call("POST", "/mcp/guest", initialize,
                              {k: v for k, v in guest_headers.items() if k != "MCP-Session-Id"})
        assert response.status == 503 and response.getheader("Retry-After") == "10"
        with sessions.oauth_mcp.lock:
            old = next(sid for sid, entry in sessions.oauth_mcp.servers.items()
                       if entry[0] == "guest" and sid != guest_headers["MCP-Session-Id"])
            sessions.oauth_mcp.server_used[old] = time.monotonic() - oauth_mcp.MCP_SESSION_IDLE - 1
        response, body = call("POST", "/mcp/guest", initialize,
                              {k: v for k, v in guest_headers.items() if k != "MCP-Session-Id"})
        assert response.status == 200, body

        def guest_tool(name, arguments, rid):
            request = json.dumps({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
            response, body = call("POST", "/mcp/guest", request, guest_headers)
            assert response.status == 200, body
            result = json.loads(body)["result"]
            assert not result["isError"], result
            return json.loads(result["content"][0]["text"])

        identity = guest_tool("whoami", {}, 11)
        assert identity["role"] == "trial" and identity["signed_in"]
        listing = json.dumps({"jsonrpc": "2.0", "id": 12, "method": "tools/list"})
        response, body = call("POST", "/mcp/guest", listing, guest_headers)
        names = {tool["name"] for tool in json.loads(body)["result"]["tools"]}
        assert {"guest_tasks", "guest_test_submit", "guest_post_submit"} <= names
        assert {"palette_state", "palette_rank", "palette_feedback"} <= names
        assert "chat_post" not in names and "checkout_open" not in names
        palette = guest_tool("palette_state", {}, 120)
        rank = [swatch["id"] for swatch in palette["data"]["swatches"]][::-1]
        palette = guest_tool("palette_rank", {"ranking": rank}, 121)
        assert palette["data"]["candidate"] is not None
        palette = guest_tool("palette_feedback", {"rating": "good", "nonce": palette["data"]["candidate"]["nonce"]}, 122)
        assert sum(entry["good"] for entry in palette["data"]["feedback"].values()) == 1
        tasks = [
            ("policy_awareness", "Hate speech and overt sexual text are refused. Ordinary swearing can be okay. Pending posts stay private."),
            ("programming", "Use a set, walk the list, and return the first element already seen; return None when all differ."),
            ("syllogism", "No. Every silver key opens the east door, and no key for that door opens the vault."),
        ]
        evidence_ids = []
        for i, (task, answer) in enumerate(tasks, 13):
            result = guest_tool("guest_test_submit", {"task": task, "response": answer, "actor_kind": "agent"}, i)
            evidence_ids.append(result["data"]["evidence"]["id"])
        import sqlite3
        with sqlite3.connect(sessions.accounts_db) as db:
            user_id = db.execute("SELECT id FROM users WHERE username=?", (identity["username"],)).fetchone()[0]
        request = json.dumps({"jsonrpc": "2.0", "id": 16, "method": "tools/call",
                              "params": {"name": "guest_post_submit", "arguments": {
                                  "target": "chat", "room": "agents", "body": "An idea for the room."}}})
        response, body = call("POST", "/mcp/guest", request, guest_headers)
        assert response.status == 200 and json.loads(body)["result"]["isError"]

        reviewer_id = sessions.accounts.ensure_user("reviewer", "correcthorse1")
        authz.grant_role(sessions.accounts_db, "reviewer", "operator", "test")
        reviewer_cookie = f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(reviewer_id)}"
        for evidence_id in evidence_ids:
            response, body = call("POST", "/api/guest/review/test",
                                  json.dumps({"id": evidence_id, "decision": "accepted"}),
                                  {"Content-Type": "application/json", "Cookie": reviewer_cookie})
            assert response.status == 200, body
        response, body = call("POST", "/api/guest/review/qualify", json.dumps({"user_id": user_id}),
                              {"Content-Type": "application/json", "Cookie": reviewer_cookie})
        assert response.status == 200, body
        queued = guest_tool("guest_post_submit", {"target": "chat", "room": "agents", "body": "An idea for the room."}, 17)
        post_id = queued["data"]["submission"]["id"]
        with sqlite3.connect(sessions.accounts_db) as db:
            assert db.execute("SELECT COUNT(*) FROM chat_messages WHERE room='agents'").fetchone()[0] == 0
        response, body = call("POST", "/api/guest/review/post",
                              json.dumps({"id": post_id, "decision": "approved", "policy_confirmed": True}),
                              {"Content-Type": "application/json", "Cookie": reviewer_cookie})
        assert response.status == 200, body
        with sqlite3.connect(sessions.accounts_db) as db:
            assert db.execute("SELECT body FROM chat_messages WHERE room='agents'").fetchone()[0] == "An idea for the room."

        intern_id = sessions.accounts.ensure_user("internbot", "correcthorse1")
        authz.grant_role(sessions.accounts_db, "internbot", "intern", "test")
        intern_cookie = f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(intern_id)}"
        response, body = call("POST", "/api/chat/rooms/agents", json.dumps({"body": "An intern's opinion."}),
                              {"Content-Type": "application/json", "Cookie": intern_cookie})
        assert response.status == 202, body
        with sqlite3.connect(sessions.accounts_db) as db:
            assert db.execute("SELECT COUNT(*) FROM chat_messages WHERE room='agents'").fetchone()[0] == 1
            assert db.execute("SELECT state FROM guest_post_queue WHERE user_id=?", (intern_id,)).fetchone()[0] == "pending"
    finally:
        conn.close()
        httpd.shutdown()
        httpd.server_close()

"""First-party OAuth 2.1 authorization and protected Streamable HTTP MCP resource.

The public origin is configured, never inferred from an untrusted Host header.
OAuth credentials live in separate tables from browser session cookies.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import secrets
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from http.cookies import SimpleCookie

CODE_LIFETIME = 300
TOKEN_LIFETIME = 3600
MAX_JSON = 1 << 20


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _form(handler, limit=4096):
    if (handler.headers.get("Content-Type") or "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        raise ValueError("expected form encoding")
    length = int(handler.headers.get("Content-Length") or 0)
    if not 0 < length <= limit:
        raise ValueError("invalid body length")
    raw = handler.rfile.read(length).decode("utf-8")
    fields = urllib.parse.parse_qs(raw, keep_blank_values=True, strict_parsing=True, max_num_fields=16)
    if any(len(values) != 1 for values in fields.values()):
        raise ValueError("duplicate field")
    return {name: values[0] for name, values in fields.items()}


def _response(handler, status, body=b"", content_type="application/json", headers=()):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    for key, value in headers:
        handler.send_header(key, value)
    handler.end_headers()
    if body:
        handler.wfile.write(body)


def _json(handler, status, value, headers=()):
    _response(handler, status, json.dumps(value, separators=(",", ":")).encode(), headers=headers)


class OAuthMCP:
    def __init__(self, accounts, db_path, public_origin: str, lift, route_getter):
        parsed = urllib.parse.urlsplit(public_origin)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("SPEAKTOME_PUBLIC_ORIGIN must be an HTTPS origin")
        self.origin = public_origin.rstrip("/")
        self.resource = self.origin + "/mcp"
        self.accounts = accounts
        self.lift = lift
        self.route_getter = route_getter
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(db_path), check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS oauth_mcp_clients (
                client_id TEXT PRIMARY KEY, name TEXT NOT NULL, redirects TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_mcp_codes (
                code_hash TEXT PRIMARY KEY, client_id TEXT NOT NULL, user_id INTEGER NOT NULL,
                redirect_uri TEXT NOT NULL, challenge TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_mcp_tokens (
                token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS oauth_mcp_pending (
                nonce TEXT PRIMARY KEY, query_json TEXT NOT NULL, cookie_user INTEGER, expires REAL NOT NULL);
        """)
        self.db.commit()
        self.servers = {}
        self.register_hits = {}

    def user_for_token(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 256 or not token.isascii():
            return None
        with self.lock:
            row = self.db.execute("SELECT users.id, users.username, users.display_name, oauth_mcp_tokens.expires "
                                  "FROM oauth_mcp_tokens JOIN users ON users.id = oauth_mcp_tokens.user_id "
                                  "WHERE oauth_mcp_tokens.token_hash = ?", (_digest(token),)).fetchone()
        if not row or row[3] <= time.time():
            return None
        return {"id": row[0], "username": row[1], "display_name": row[2]}

    def metadata(self, handler, path):
        if path == "/.well-known/oauth-protected-resource/mcp":
            return _json(handler, 200, {"resource": self.resource, "authorization_servers": [self.origin],
                                        "scopes_supported": ["mcp"]})
        if path == "/.well-known/oauth-authorization-server":
            return _json(handler, 200, {"issuer": self.origin, "authorization_endpoint": self.origin + "/oauth/authorize",
                                        "token_endpoint": self.origin + "/oauth/token", "registration_endpoint": self.origin + "/oauth/register",
                                        "response_types_supported": ["code"], "grant_types_supported": ["authorization_code"],
                                        "code_challenge_methods_supported": ["S256"], "token_endpoint_auth_methods_supported": ["none"],
                                        "scopes_supported": ["mcp"], "resource_indicators_supported": True})
        return False

    def challenge(self, handler, error="invalid_token"):
        metadata = self.origin + "/.well-known/oauth-protected-resource/mcp"
        return _json(handler, 401, {"error": error}, [("WWW-Authenticate", f'Bearer resource_metadata="{metadata}", error="{error}"')])

    def register(self, handler):
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            if not 0 < length <= 4096 or (handler.headers.get("Content-Type") or "").split(";", 1)[0] != "application/json":
                raise ValueError("invalid registration body")
            data = json.loads(handler.rfile.read(length))
            name, redirects = data.get("client_name"), data.get("redirect_uris")
            if not isinstance(name, str) or not 1 <= len(name) <= 100 or not isinstance(redirects, list) or not 1 <= len(redirects) <= 8:
                raise ValueError("invalid client metadata")
            if data.get("token_endpoint_auth_method", "none") != "none" or data.get("grant_types", ["authorization_code"]) != ["authorization_code"]:
                raise ValueError("unsupported client metadata")
            for uri in redirects:
                parsed = urllib.parse.urlsplit(uri) if isinstance(uri, str) else None
                if (parsed is None or not parsed.netloc or parsed.fragment or parsed.username or parsed.password or
                    (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "::1", "localhost")))):
                    raise ValueError("invalid redirect URI")
            peer = getattr(handler, "_client_ip", handler.client_address[0])
            now = time.time()
            with self.lock:
                hits = [t for t in self.register_hits.get(peer, []) if now - t < 3600]
                if len(hits) >= 10:
                    return _json(handler, 429, {"error": "rate_limited"})
                hits.append(now)
                self.register_hits[peer] = hits
                client_id = secrets.token_urlsafe(24)
                self.db.execute("INSERT INTO oauth_mcp_clients VALUES (?, ?, ?, ?)", (client_id, name, json.dumps(redirects), now))
                self.db.commit()
            return _json(handler, 201, {"client_id": client_id, "client_name": name, "redirect_uris": redirects,
                                        "token_endpoint_auth_method": "none", "grant_types": ["authorization_code"],
                                        "response_types": ["code"]})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return _json(handler, 400, {"error": "invalid_client_metadata"})

    def authorize_get(self, handler):
        try:
            query = urllib.parse.parse_qs(handler.path.partition("?")[2], keep_blank_values=True, strict_parsing=True, max_num_fields=12)
            if any(len(v) != 1 for v in query.values()):
                raise ValueError()
            q = {k: v[0] for k, v in query.items()}
            client_id, redirect = q["client_id"], q["redirect_uri"]
            with self.lock:
                row = self.db.execute("SELECT name, redirects FROM oauth_mcp_clients WHERE client_id = ?", (client_id,)).fetchone()
            if not row or redirect not in json.loads(row[1]):
                raise ValueError()
            challenge = q.get("code_challenge", "")
            if (q.get("response_type") != "code" or q.get("code_challenge_method") != "S256" or
                q.get("resource") != self.resource or q.get("scope", "mcp") != "mcp" or
                not 43 <= len(challenge) <= 128 or not all(c.isalnum() or c in "-_" for c in challenge) or
                len(q.get("state", "")) > 1024):
                raise ValueError()
            account = handler._account()
            nonce = secrets.token_urlsafe(32)
            with self.lock:
                self.db.execute("DELETE FROM oauth_mcp_pending WHERE expires <= ?", (time.time(),))
                self.db.execute("INSERT INTO oauth_mcp_pending VALUES (?, ?, ?, ?)",
                                (nonce, json.dumps(q), account["id"] if account else None, time.time() + CODE_LIFETIME))
                self.db.commit()
            title = html.escape(row[0])
            user = html.escape(account["username"]) if account else ""
            credentials = "" if account else '<label>Username <input name="username" autocomplete="username" required></label><label>Password <input name="password" type="password" autocomplete="current-password" required></label>'
            page = (f'<!doctype html><meta charset="utf-8"><title>Connect to SpeakToMe</title>'
                    f'<h1>Connect {title} to SpeakToMe?</h1><p>{"Signed in as " + user if account else "Sign in to authorize this client."}</p>'
                    f'<p>This client will use your account’s existing permissions.</p><form method="post" action="/oauth/authorize">'
                    f'<input type="hidden" name="nonce" value="{nonce}">{credentials}'
                    '<button type="submit" name="decision" value="allow">Allow</button>'
                    '<button type="submit" name="decision" value="deny">Deny</button></form>').encode()
            return _response(handler, 200, page, "text/html; charset=utf-8",
                             [("Set-Cookie", f"oauth_mcp_nonce={nonce}; Path=/oauth/authorize; Max-Age={CODE_LIFETIME}; HttpOnly; Secure; SameSite=Lax"),
                              ("Content-Security-Policy", "default-src 'none'; form-action 'self'; base-uri 'none'"),
                              ("Referrer-Policy", "no-referrer")])
        except (ValueError, KeyError, TypeError):
            return _json(handler, 400, {"error": "invalid_request"})

    def authorize_post(self, handler):
        try:
            form = _form(handler)
            nonce = form["nonce"]
            jar = SimpleCookie(handler.headers.get("Cookie", ""))
            if jar.get("oauth_mcp_nonce") is None or not secrets.compare_digest(jar["oauth_mcp_nonce"].value, nonce):
                raise ValueError()
            with self.lock:
                item = self.db.execute("SELECT query_json, cookie_user, expires FROM oauth_mcp_pending WHERE nonce = ?", (nonce,)).fetchone()
                self.db.execute("DELETE FROM oauth_mcp_pending WHERE nonce = ?", (nonce,))
                self.db.commit()
            if item is None or item[2] <= time.time():
                raise ValueError()
            q, cookie_user = json.loads(item[0]), item[1]
            redirect = q["redirect_uri"]
            if form.get("decision") == "deny":
                return self._redirect(handler, redirect, {"error": "access_denied", "state": q.get("state", "")})
            if form.get("decision") != "allow":
                raise ValueError()
            account = handler._account()
            if cookie_user is not None:
                if account is None or account["id"] != cookie_user:
                    raise ValueError()
                user_id = account["id"]
            else:
                # The existing login guard is applied to password submissions, including malformed names.
                from meowdoku import login_guard, accounts as accounts_mod
                username = form.get("username", "")
                db = login_guard.connect(self.accounts.path)
                try:
                    if login_guard.check(db, handler._client_ip, username):
                        return _json(handler, 429, {"error": "too_many_attempts"})
                    try:
                        user_id = self.accounts.login(username, form.get("password", ""))["id"]
                    except accounts_mod.AccountError:
                        login_guard.record(db, handler._client_ip, username, False)
                        return _json(handler, 401, {"error": "invalid_credentials"})
                    login_guard.record(db, handler._client_ip, username, True)
                finally:
                    db.close()
            code = secrets.token_urlsafe(32)
            with self.lock:
                self.db.execute("INSERT INTO oauth_mcp_codes VALUES (?, ?, ?, ?, ?, ?)",
                                (_digest(code), q["client_id"], user_id, redirect, q["code_challenge"], time.time() + CODE_LIFETIME))
                self.db.commit()
            return self._redirect(handler, redirect, {"code": code, "state": q.get("state", "")})
        except (ValueError, KeyError, TypeError):
            return _json(handler, 400, {"error": "invalid_request"})

    @staticmethod
    def _redirect(handler, redirect, params):
        parsed = urllib.parse.urlsplit(redirect)
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) + list(params.items())
        location = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), ""))
        return _response(handler, 302, b"", headers=[("Location", location)])

    def token(self, handler):
        try:
            form = _form(handler)
            if form.get("grant_type") != "authorization_code" or form.get("resource") != self.resource:
                raise ValueError()
            verifier = form["code_verifier"]
            if not 43 <= len(verifier) <= 128 or not verifier.isascii() or not all(c.isalnum() or c in "-._~" for c in verifier):
                raise ValueError()
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            with self.lock:
                row = self.db.execute("SELECT client_id, user_id, redirect_uri, challenge, expires FROM oauth_mcp_codes WHERE code_hash = ?",
                                      (_digest(form["code"]),)).fetchone()
                if not row or row[4] <= time.time() or row[0] != form.get("client_id") or row[2] != form.get("redirect_uri") or not secrets.compare_digest(row[3], challenge):
                    raise ValueError()
                self.db.execute("DELETE FROM oauth_mcp_codes WHERE code_hash = ?", (_digest(form["code"]),))
                token = secrets.token_urlsafe(32)
                self.db.execute("INSERT INTO oauth_mcp_tokens VALUES (?, ?, ?)", (_digest(token), row[1], time.time() + TOKEN_LIFETIME))
                self.db.execute("DELETE FROM oauth_mcp_codes WHERE expires <= ?", (time.time(),))
                self.db.execute("DELETE FROM oauth_mcp_tokens WHERE expires <= ?", (time.time(),))
                self.db.commit()
            return _json(handler, 200, {"access_token": token, "token_type": "Bearer", "expires_in": TOKEN_LIFETIME, "scope": "mcp"})
        except (ValueError, KeyError, TypeError):
            return _json(handler, 400, {"error": "invalid_grant"})

    def mcp(self, handler, method):
        if not handler._is_secure():
            return _json(handler, 403, {"error": "HTTPS required"})
        origin = handler.headers.get("Origin")
        allowed = {self.origin} | {x.strip() for x in os.environ.get("SPEAKTOME_MCP_ALLOWED_ORIGINS", "").split(",") if x.strip()}
        if origin is not None and origin not in allowed:
            return _json(handler, 403, {"error": "invalid origin"})
        auth = handler.headers.get_all("Authorization", [])
        user = self.user_for_token(auth[0][7:]) if len(auth) == 1 and auth[0].startswith("Bearer ") else None
        if user is None:
            return self.challenge(handler)
        token = auth[0][7:]
        version = handler.headers.get("MCP-Protocol-Version")
        if version is not None and version not in ("2025-11-25", "2025-06-18", "2025-03-26"):
            return _json(handler, 400, {"error": "unsupported MCP protocol version"})
        if handler._origin_tier == "shader":
            from meowdoku import authz
            handler._tier = self.lift.get(authz.role_of(self.accounts.path, user["id"]), "shader")
        if method == "DELETE":
            session_id = handler.headers.get("MCP-Session-Id")
            with self.lock:
                entry = self.servers.get(session_id)
                if entry is None or entry[:3] != (_digest(token), handler._origin, handler._tier):
                    return _json(handler, 404, {"error": "MCP session not found"})
                del self.servers[session_id]
            return _response(handler, 204, b"")
        if method == "GET":
            return _response(handler, 405, b"", headers=[("Allow", "POST, DELETE")])
        if "application/json" not in (handler.headers.get("Accept") or "") or "text/event-stream" not in (handler.headers.get("Accept") or ""):
            return _json(handler, 406, {"error": "Accept must include application/json and text/event-stream"})
        if (handler.headers.get("Content-Type") or "").split(";", 1)[0].lower() != "application/json":
            return _json(handler, 415, {"error": "expected application/json"})
        try:
            length = int(handler.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_JSON:
                raise ValueError()
            message = json.loads(handler.rfile.read(length))
            if (not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or
                isinstance(message.get("id"), (dict, list))):
                raise ValueError()
        except (ValueError, TypeError, json.JSONDecodeError):
            return _json(handler, 400, {"error": "invalid JSON-RPC message"})
        if "method" not in message and "id" in message and ("result" in message or "error" in message):
            return _response(handler, 202, b"")
        if not isinstance(message.get("method"), str):
            return _json(handler, 400, {"error": "invalid JSON-RPC message"})
        session_id = handler.headers.get("MCP-Session-Id")
        if message.get("method") == "initialize":
            if session_id is not None:
                return _json(handler, 400, {"error": "initialize must start a new session"})
            with self.lock:
                from meowdoku import rest_api
                rest_api._load("speaktome_mcp")
                from speaktome_mcp.api import ApiServer
                from speaktome_mcp.http import BearerRest
                rest = BearerRest(f"http://127.0.0.1:{handler.server.server_port}", token,
                                  handler._origin, handler._tier)
                session_id = secrets.token_urlsafe(32)
                if len(self.servers) >= 128:
                    self.servers.clear()
                entry = (_digest(token), handler._origin, handler._tier, threading.Lock(), ApiServer(rest))
                self.servers[session_id] = entry
        else:
            if session_id is None:
                return _json(handler, 400, {"error": "MCP-Session-Id required"})
            with self.lock:
                entry = self.servers.get(session_id)
            if entry is None or entry[:3] != (_digest(token), handler._origin, handler._tier):
                return _json(handler, 404, {"error": "MCP session not found"})
        _, _, _, lock, server = entry
        with lock:
            answer = server.handle(message)
        if answer is None:
            return _response(handler, 202, b"")
        headers = []
        if message.get("method") == "initialize":
            headers.append(("MCP-Session-Id", session_id))
            route = self.route_getter()
            if route:
                from meowdoku import proxy_trust
                headers.append(("Set-Cookie", proxy_trust.route_cookie(route)))
        return _json(handler, 200, answer, headers)

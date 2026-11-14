"""Authenticated host boundary for the optional remote MCP client.

Provider metadata is discovered by the SDK, not synthesized here. Configuration
is operator-owned; browser requests can never select an endpoint or vault owner.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

_lock = threading.Lock()


def vault_callbacks(root, owner, provider):
    if type(owner) is not int or owner < 1 or provider != 'suno':
        raise PermissionError('invalid connection owner')
    target = Path(root) / 'mcp_credentials' / str(owner) / 'suno.json'
    def read():
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding='utf-8'))
    def write(record):
        # Dedicated restricted OAuth subtree; never widen access to Alpaca/other credentials.
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name('.' + secrets.token_hex(16) + '.tmp')
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(record, f, separators=(',', ':'))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    def clear():
        target.unlink(missing_ok=True)
    return read, write, clear


def service_for(sessions):
    with _lock:
        service = getattr(sessions, '_remote_mcp_connections', None)
        if service is not None:
            return service
        from speaktome_api.remote_mcp import ConnectionService, JsonVault, Provider, PublicHTTPTransport
        config = json.loads((sessions.state.run_dir / 'mcp_connections.json').read_text(encoding='utf-8'))
        c = config['suno']
        callback = c['callback_url']
        parsed = urlsplit(callback)
        if parsed.path != '/api/connections/suno/callback' or parsed.query or parsed.fragment:
            raise ValueError('invalid callback')
        provider = Provider('https://api.suno.com/mcp', callback, frozenset(c['allowed_origins']))
        service = ConnectionService({'suno': provider},
            lambda owner, name: JsonVault(*vault_callbacks(sessions.state.secrets_dir, owner, name)),
            lambda name: PublicHTTPTransport())
        sessions._remote_mcp_connections = service
        atexit.register(service.close)
        return service


def route(handler, sessions, path, method):
    handler._set_cookie = []
    if not handler._is_secure():
        return handler._json({'error': 'HTTPS required'}, 403)
    owner = handler._principal().user_id
    if owner is None:
        return handler._json({'error': 'sign in first'}, 401)
    action = path.removeprefix('/api/connections/suno/')
    if path != '/api/connections/suno/' + action or action not in {
            'status', 'connect', 'callback', 'tools', 'call', 'disconnect'}:
        return handler._json({'error': 'unknown connection action'}, 404)
    if method != ('GET' if action in {'status', 'tools', 'callback'} else 'POST'):
        return handler._json({'error': 'method not allowed'}, 405)
    body = {}
    if method == 'POST':
        if (handler.headers.get('Origin') != 'https://' + handler.headers.get('Host', '')
                or handler.headers.get('X-Meowdoku-Secret') != '1'
                or handler.headers.get('Sec-Fetch-Site') == 'cross-site'):
            return handler._json({'error': 'same-origin connection request required'}, 403)
        if not (handler.headers.get('Content-Type') or '').lower().startswith('application/json'):
            return handler._json({'error': 'JSON only'}, 415)
        body = handler._body_json(1 << 20)
        if body is None:
            return
        expected = {'name', 'arguments'} if action == 'call' else set()
        if not isinstance(body, dict) or set(body) != expected:
            return handler._json({'error': 'invalid connection request'}, 400)
        if action == 'call' and (not isinstance(body['name'], str) or not body['name']
                                or not isinstance(body['arguments'], dict)):
            return handler._json({'error': 'invalid tool request'}, 400)
    try:
        service = service_for(sessions)
    except Exception:
        # Import/configuration/storage failures are intentionally nonsecret.
        return handler._json({'status': 'unavailable', 'configured': False,
                              'error': 'connection service is not configured'}, 503)
    try:
        if action == 'callback':
            query = parse_qs(urlsplit(handler.path).query, keep_blank_values=True)
            if set(query) - {'code', 'state', 'iss'} or any(len(v) != 1 for v in query.values()):
                raise ValueError('invalid callback')
            service.callback(owner, 'suno', query.get('code', [''])[0], query.get('state', [''])[0])
            handler.send_response(303)
            handler.send_header('Location', '/connected-services.html')
            handler.send_header('Cache-Control', 'no-store')
            handler.send_header('Referrer-Policy', 'no-referrer')
            handler.send_header('Content-Length', '0')
            handler.end_headers()
            return
        if action == 'status': result = service.status(owner, 'suno')
        elif action == 'connect': result = service.begin(owner, 'suno')
        elif action == 'tools': result = {'tools': service.tools(owner, 'suno')}
        elif action == 'call': result = service.call(owner, 'suno', body['name'], body['arguments'])
        else: result = service.disconnect(owner, 'suno')
        return handler._json(result)
    except (PermissionError, ValueError, KeyError):
        return handler._json({'error': 'connection request refused'}, 400)
    except Exception:
        return handler._json({'error': 'provider operation unavailable'}, 502)


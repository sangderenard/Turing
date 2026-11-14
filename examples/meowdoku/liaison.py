"""Local authenticated boundary for the Windows Suno liaison service.

The web process supplies only the account ID established by its session. The
liaison owns the SDK connection state and the writable credential vault. Both
sides use bounded JSON over loopback with a shared HMAC key; neither serializes
Python objects or writes a credential into the web process's state tree.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import struct
import threading
import time
from urllib.parse import urlsplit


HOST = "127.0.0.1"
PORT = 48790
MAX_REQUEST = 1 << 20
MAX_RESPONSE = 32 << 20
ACTIONS = frozenset(("status", "begin", "callback", "tools", "call", "disconnect"))


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _mac(key: bytes, value: dict) -> str:
    return hmac.new(key, _canonical(value), hashlib.sha256).hexdigest()


def _key(path: Path) -> bytes:
    data = Path(path).read_bytes()
    if len(data) != 32:
        raise ValueError("liaison IPC key must be 32 bytes")
    return data


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            raise ConnectionError("liaison connection closed")
        chunks.extend(part)
    return bytes(chunks)


def _recv(sock: socket.socket, limit: int) -> dict:
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < size <= limit:
        raise ValueError("liaison message size refused")
    value = json.loads(_read_exact(sock, size))
    if not isinstance(value, dict):
        raise ValueError("liaison message must be an object")
    return value


def _send(sock: socket.socket, value: dict, limit: int) -> None:
    data = _canonical(value)
    if len(data) > limit:
        raise ValueError("liaison response too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


class LiaisonClient:
    def __init__(self, key_file: Path, *, host: str = HOST, port: int = PORT):
        if host != HOST or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("liaison must use a fixed loopback endpoint")
        self.key_file, self.host, self.port = Path(key_file), host, port

    def _request(self, action: str, owner: int, **body):
        if action not in ACTIONS or type(owner) is not int or owner < 1:
            raise ValueError("invalid liaison request")
        payload = {"action": action, "owner": owner, "body": body,
                   "nonce": secrets.token_hex(16), "time": int(time.time())}
        wire = {**payload, "mac": _mac(_key(self.key_file), payload)}
        try:
            with socket.create_connection((self.host, self.port), timeout=5) as sock:
                sock.settimeout(70)
                _send(sock, wire, MAX_REQUEST)
                reply = _recv(sock, MAX_RESPONSE)
        except (OSError, TimeoutError) as error:
            raise RuntimeError("liaison service unavailable") from error
        signature = reply.pop("mac", None)
        if reply.get("nonce") != payload["nonce"] or not isinstance(signature, str) or not hmac.compare_digest(
                signature, _mac(_key(self.key_file), reply)):
            raise RuntimeError("liaison response authentication failed")
        if reply.get("ok") is True:
            return reply["result"]
        kind = reply.get("error")
        if kind == "permission":
            raise PermissionError("liaison operation refused")
        if kind == "invalid":
            raise ValueError("invalid liaison operation")
        raise RuntimeError("liaison operation unavailable")

    def status(self, owner, provider): return self._request("status", owner, provider=provider)
    def begin(self, owner, provider): return self._request("begin", owner, provider=provider)
    def callback(self, owner, provider, code, state):
        return self._request("callback", owner, provider=provider, code=code, state=state)
    def tools(self, owner, provider): return self._request("tools", owner, provider=provider)
    def call(self, owner, provider, name, arguments):
        return self._request("call", owner, provider=provider, name=name, arguments=arguments)
    def disconnect(self, owner, provider): return self._request("disconnect", owner, provider=provider)
    def close(self): pass


def vault_callbacks(root: Path, owner: int, provider: str):
    if type(owner) is not int or owner < 1 or provider != "suno":
        raise PermissionError("invalid connection owner")
    target = Path(root) / str(owner) / "suno.json"

    def read():
        return json.loads(target.read_text(encoding="utf-8")) if target.exists() else None

    def write(record):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name("." + secrets.token_hex(16) + ".tmp")
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)

    def clear(): target.unlink(missing_ok=True)
    return read, write, clear


def connection_service(config_file: Path, vault_root: Path):
    from speaktome_api.remote_mcp import ConnectionService, JsonVault, Provider, PublicHTTPTransport
    config = json.loads(Path(config_file).read_text(encoding="utf-8"))
    suno = config["suno"]
    callback = suno["callback_url"]
    parsed = urlsplit(callback)
    if parsed.path != "/api/connections/suno/callback" or parsed.query or parsed.fragment:
        raise ValueError("invalid callback")
    return ConnectionService(
        {"suno": Provider("https://api.suno.com/mcp", callback, frozenset(suno["allowed_origins"]))},
        lambda owner, provider: JsonVault(*vault_callbacks(vault_root, owner, provider)),
        lambda provider: PublicHTTPTransport())


class LiaisonServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, key_file: Path, service, *, host: str = HOST, port: int = PORT):
        if host != HOST:
            raise ValueError("liaison must bind loopback")
        self.key, self.service = _key(key_file), service
        self._seen, self._seen_lock = {}, threading.Lock()
        self._slots = threading.BoundedSemaphore(32)
        super().__init__((host, port), _RequestHandler)

    def _verify(self, wire: dict) -> dict:
        signature = wire.pop("mac", None)
        if not isinstance(signature, str) or not hmac.compare_digest(signature, _mac(self.key, wire)):
            raise PermissionError("liaison authentication failed")
        nonce, stamp = wire.get("nonce"), wire.get("time")
        if not isinstance(nonce, str) or len(nonce) != 32 or type(stamp) is not int or abs(time.time() - stamp) > 30:
            raise PermissionError("liaison request expired")
        with self._seen_lock:
            now = time.time()
            self._seen = {key: ts for key, ts in self._seen.items() if now - ts <= 60}
            if nonce in self._seen:
                raise PermissionError("liaison request replayed")
            self._seen[nonce] = now
        return wire

    def _dispatch(self, wire: dict):
        owner, action, body = wire.get("owner"), wire.get("action"), wire.get("body")
        if type(owner) is not int or owner < 1 or action not in ACTIONS or not isinstance(body, dict):
            raise ValueError("invalid liaison request")
        if body.get("provider") != "suno":
            raise ValueError("unknown provider")
        service = self.service
        if action == "status": return service.status(owner, "suno")
        if action == "begin": return service.begin(owner, "suno")
        if action == "callback":
            code, state = body.get("code"), body.get("state")
            if not isinstance(code, str) or not isinstance(state, str) or len(code) > 16384 or len(state) > 1024:
                raise ValueError("invalid callback")
            return service.callback(owner, "suno", code, state)
        if action == "tools": return service.tools(owner, "suno")
        if action == "call":
            name, arguments = body.get("name"), body.get("arguments")
            if not isinstance(name, str) or not 0 < len(name) <= 200 or not isinstance(arguments, dict):
                raise ValueError("invalid tool call")
            return service.call(owner, "suno", name, arguments)
        return service.disconnect(owner, "suno")


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        if not self.server._slots.acquire(blocking=False):
            return
        sock = self.request
        sock.settimeout(5)
        try:
            request = self.server._verify(_recv(sock, MAX_REQUEST))
            try:
                result = self.server._dispatch(request)
                response = {"nonce": request["nonce"], "ok": True, "result": result}
            except (PermissionError, ValueError, KeyError) as error:
                response = {"nonce": request["nonce"], "ok": False,
                            "error": "permission" if isinstance(error, PermissionError) else "invalid"}
            except Exception:
                response = {"nonce": request["nonce"], "ok": False, "error": "unavailable"}
            response["mac"] = _mac(self.server.key, response)
            _send(sock, response, MAX_RESPONSE)
        except (OSError, ConnectionError, ValueError, PermissionError):
            pass  # Never log request bodies, OAuth codes, or bearer material.
        finally:
            self.server._slots.release()

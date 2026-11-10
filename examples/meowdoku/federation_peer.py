"""Federation peer client: the remote side of ``wss://<host>/ws/contract/<prefix>`` (server.py ``_contract_socket``).

Stdlib only (aiohttp is in requirements-service.txt but a blocking, socket-shaped object is what nodus/tools/contract_service_peer.py's
``ServicePeer``/``EnginePeer`` take, so a minimal RFC 6455 client is the smaller piece).

    digests = identity_digests(canvas_path, web_dir)             # what the owner will compare against its own
    peer = connect("ws://127.0.0.1:8768/ws/contract/lab", grant_id, secret, "engine", digests)
    peer.send_frame(frame72); frame = peer.recv_frame(timeout=2)
    # or hand it to the Nodus helper (it already knows how to drive a socket; NEVER announce -- the bridge sends the Hello):
    EnginePeer(peer, producer)            ServicePeer(peer, executor, dispatch, on_receipt, announce=False)

``PeerSocket`` offers the socket subset those helpers use: ``setblocking``, ``send``/``sendall`` (whole 72-byte frames go out as binary
WebSocket messages; partial bytes are buffered), ``recv(n)`` (bytes of received frames; BlockingIOError when non-blocking and empty,
b"" once closed), ``close``.  ``identity_digests`` is shared with the server so both sides compute the promise identically.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
FRAME_BYTES = 72
MAGIC = 0x31434E4E


class FederationRefused(Exception):
    """The owner refused the promise (``reason`` is what it said) or the handshake."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def identity_digests(canvas_path, web_dir) -> dict:
    """The promise's digests: sha256 of the canvas file's bytes, and the sorted ``module:entry`` pairs of every ``*.api.yaml`` under web_dir."""
    import yaml
    pairs = []
    for path in sorted(Path(web_dir).rglob("*.api.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pairs.append(f"{doc.get('module') or ''}:{doc.get('entry') or ''}")
    return {"graph_definition_digest": hashlib.sha256(Path(canvas_path).read_bytes()).hexdigest(), "api_digests": sorted(pairs)}


def _ws_frame(opcode: int, payload: bytes) -> bytes:
    mask = os.urandom(4)
    n = len(payload)
    if n < 126:
        head = bytes([0x80 | opcode, 0x80 | n])
    elif n < 65536:
        head = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack(">H", n)
    else:
        head = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack(">Q", n)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def _read_message(stream):
    """(opcode, payload) of one message; None when the stream ended.  Joins fragments; control frames come back as they are."""
    opcode, payload = None, b""
    while True:
        head = stream.read(2)
        if len(head) < 2:
            return None
        fin, op, length = head[0] & 0x80, head[0] & 0x0F, head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", stream.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", stream.read(8))[0]
        mask = stream.read(4) if head[1] & 0x80 else b""
        data = stream.read(length)
        if len(data) < length:
            return None
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        if op in (0x8, 0x9, 0xA):
            return op, data
        opcode = op if opcode is None else opcode
        payload += data
        if fin:
            return opcode, payload


class PeerSocket:
    """A connected, promise-accepted federation socket with a socket-like surface (see the module docstring)."""

    def __init__(self, sock: socket.socket, stream, accepted: dict):
        self._sock, self._stream, self.accepted = sock, stream, accepted
        self.hello_id = accepted.get("hello_id")
        self.close_code: int | None = None
        self._wlock, self._cond = threading.Lock(), threading.Condition()
        self._incoming, self._pending_out = bytearray(), bytearray()
        self._blocking, self.closed = True, False
        self._reader = threading.Thread(target=self._read_loop, name="federation-peer-read", daemon=True)
        self._reader.start()

    # -- frames
    def send_frame(self, frame72: bytes) -> None:
        if len(frame72) != FRAME_BYTES:
            raise ValueError("a contract frame is exactly 72 bytes")
        with self._wlock:
            self._sock.sendall(_ws_frame(0x2, bytes(frame72)))

    def recv_frame(self, timeout: float | None = None) -> bytes | None:
        """The next whole 72-byte frame, or None on timeout/close."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while len(self._incoming) < FRAME_BYTES:
                if self.closed:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            out = bytes(self._incoming[:FRAME_BYTES])
            del self._incoming[:FRAME_BYTES]
            return out

    # -- the socket subset ServicePeer / EnginePeer use
    def setblocking(self, flag: bool) -> None:
        self._blocking = bool(flag)

    def send(self, data) -> int:
        self._pending_out.extend(data)
        while len(self._pending_out) >= FRAME_BYTES:
            frame, self._pending_out = bytes(self._pending_out[:FRAME_BYTES]), self._pending_out[FRAME_BYTES:]
            self.send_frame(frame)
        return len(data)

    sendall = send

    def recv(self, n: int) -> bytes:
        with self._cond:
            while not self._incoming:
                if self.closed:
                    return b""
                if not self._blocking:
                    raise BlockingIOError()
                self._cond.wait()
            out = bytes(self._incoming[:n])
            del self._incoming[:n]
            return out

    def close(self, code: int = 1000) -> None:
        if not self.closed:
            try:
                with self._wlock:
                    self._sock.sendall(_ws_frame(0x8, struct.pack(">H", code)))
            except OSError:
                pass
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    # -- the reader
    def _read_loop(self) -> None:
        try:
            while True:
                message = _read_message(self._stream)
                if message is None:
                    break
                op, data = message
                if op == 0x8:
                    self.close_code = struct.unpack(">H", data[:2])[0] if len(data) >= 2 else 1005
                    break
                if op == 0x9:
                    with self._wlock:
                        self._sock.sendall(_ws_frame(0xA, data))
                elif op == 0x2:
                    with self._cond:
                        self._incoming.extend(data)
                        self._cond.notify_all()
        except (OSError, ValueError, struct.error):
            pass
        finally:
            with self._cond:
                self.closed = True
                self._cond.notify_all()


def connect(url: str, grant_id: int, secret: str, direction: str, digests: dict, timeout: float = 10.0,
            host_header: str | None = None, ssl_context: ssl.SSLContext | None = None) -> PeerSocket:
    """Open ``url`` (ws:// or wss://, path ``/ws/contract/<prefix>``), send the promise, and return the PeerSocket once the owner
    answered ``ok``.  Raises FederationRefused(reason) on a refusal.  ``digests`` is ``identity_digests(...)``'s mapping."""
    parts = urlsplit(url)
    secure = parts.scheme == "wss"
    sock = socket.create_connection((parts.hostname, parts.port or (443 if secure else 80)), timeout=timeout)
    try:
        if secure:
            sock = (ssl_context or ssl.create_default_context()).wrap_socket(sock, server_hostname=parts.hostname)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (f"GET {parts.path or '/'}{'?' + parts.query if parts.query else ''} HTTP/1.1\r\nHost: {host_header or parts.netloc}\r\n"
                   f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        sock.sendall(request.encode("ascii"))
        stream = sock.makefile("rb")
        status = stream.readline().decode("latin-1").strip()
        headers = {}
        while True:
            line = stream.readline().decode("latin-1")
            if line in ("\r\n", "\n", ""):
                break
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        if " 101 " not in status + " ":
            raise FederationRefused(f"handshake refused: {status or 'no response'}")
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raise FederationRefused("handshake refused: bad Sec-WebSocket-Accept")
        promise = {"grant_id": grant_id, "secret": secret, "direction": direction, **digests}
        sock.sendall(_ws_frame(0x1, json.dumps(promise).encode("utf-8")))
        message = _read_message(stream)
        if message is None or message[0] == 0x8:
            code = struct.unpack(">H", message[1][:2])[0] if message and len(message[1]) >= 2 else None
            raise FederationRefused(f"closed before the answer (code {code})")
        answer = json.loads(message[1].decode("utf-8"))
        if not answer.get("ok"):
            raise FederationRefused(str(answer.get("reason") or "refused"))
        sock.settimeout(None)
        return PeerSocket(sock, stream, answer)
    except BaseException:
        sock.close()
        raise

"""Nodus UDP broker: introduces two Nodus instances so they can talk directly over UDP (hole punching).

Research case a02ca4a5d8eb4dbb85e91a76daa0f53d (ticket d3da321de3234a3783e4201fec36909e). Owner, 2026-11-19: "okay we'll make
a page that brokers and then hands off to udp using the udp line tools".

Both Nodus instances make only outgoing requests:
  1. HTTPS  POST /api/link/rendezvous/<prefix> {grant_id, secret, direction}  (server.py; the peer grant decides who may)
            -> {link_id, key, role, expires_at, broker: {host, port}}: ``rendezvous(...)`` below.
  2. UDP    HELLO from the Nodus's own socket to the broker port. Its source address, as the Nodus's NAT translated it,
            is what the peer will send to.
  3. UDP    when a host and a dialer of the same prefix have both said hello, the broker sends each an INTRO: the other's
            public address and a fresh session key, sealed with that peer's rendezvous key. Then it steps out; link data
            never passes through the server.
The Nodus side (nodus ``include/nodus_link_udp.h``) punches through, then runs the ordinary Nodus link over UDP.

The broker is a server service like ``relay.py``: the native ``udp_router`` (nodus tools/udp_router) holds the UDP socket,
the broker reaches it over a local connection. No UDP socket is ever bound in Python.

Wire formats (little-endian; identical to nodus ``nodus_link_udp.h``):
  rendezvous key  K = HMAC-SHA256(server link key, "NLR1" | link id [16] | role u8 | expires_at u64 | prefix utf-8)
  HELLO  magic "NLR1" u32 | kind 1 u8 | role u8 (1 host, 2 dialer) | prefix length u16 | link id [16] | expires_at u64 |
         local ipv4 [4] | local port u16 | prefix | HMAC-SHA256(K, everything before) [32]
  INTRO  magic u32 | kind 2 u8 | 0 u8 | 0 u16 | link id [16] | nonce [12] | AES-256-GCM(K, plaintext, aad = first 24 bytes):
         plaintext = peer public ipv4 [4, network order] | peer public port u16 | peer role u8 | 0 u8 | pair id [16] |
                     session key [32] | peer local ipv4 [4] | peer local port u16
  Two candidates, as ICE gives host and server-reflexive ones: the public address the broker saw, and the address the peer
  has on its own network. A pair behind one NAT meets on the local one, two networks on the public one.

Usage:  python link_broker.py --state-dir <state> [--router 127.0.0.1:47699] [--port 47702]
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meowdoku import link_tickets  # noqa: E402
from meowdoku.relay import BATCH, ENTRY, FRAME, NOTE, OPEN, OPENED  # noqa: E402

MAGIC = 0x31524C4E                      # "NLR1"
HELLO, INTRO = 1, 2
HOST, DIALER = link_tickets.HOST, link_tickets.DIALER
ROLE_OF_DIRECTION = {"host": HOST, "engine": DIALER}
PORT = 47702                            # inside the router's forwarded range 47700-47763
LIFE = 300
HELLO_HEAD = struct.Struct("<IBBH16sQ4sH")
INTRO_HEAD = struct.Struct("<IBBH16s")
STATUS_FILE = "link_broker.json"        # <state>/run/: counts only, no addresses and no keys


def rendezvous_key(server_key: bytes, link_id: bytes, role: int, expires_at: int, prefix: str) -> bytes:
    return hmac.new(server_key, b"NLR1" + link_id + struct.pack("<BQ", role, expires_at) + prefix.encode("utf-8"),
                    hashlib.sha256).digest()


def rendezvous(server_key: bytes, prefix: str, direction: str, life: int = LIFE, now: float | None = None) -> dict:
    """What POST /api/link/rendezvous/<prefix> answers (the broker's host and port are added by the caller)."""
    if direction not in ROLE_OF_DIRECTION:
        raise ValueError("direction is 'host' or 'engine'")
    if not 0 < int(life) <= LIFE:
        raise ValueError(f"life is 1..{LIFE} seconds")
    role, link_id = ROLE_OF_DIRECTION[direction], os.urandom(16)
    expires = int(time.time() if now is None else now) + int(life)
    key = rendezvous_key(server_key, link_id, role, expires, prefix)
    return {"link_id": link_id.hex(), "key": key.hex(), "role": role, "expires_at": expires, "prefix": prefix}


def hello(key: bytes, link_id: bytes, role: int, expires_at: int, prefix: str, local_ip: bytes = b"\0\0\0\0",
          local_port: int = 0) -> bytes:
    """A HELLO datagram (what the Nodus side sends; used by tests)."""
    p = prefix.encode("utf-8")
    body = HELLO_HEAD.pack(MAGIC, HELLO, role, len(p), link_id, expires_at, local_ip, local_port) + p
    return body + hmac.new(key, body, hashlib.sha256).digest()


def parse_hello(server_key: bytes, data: bytes, now: float):
    """(link_id, role, expires_at, prefix, K, local ip, local port) of a valid, live HELLO; else None."""
    if len(data) < HELLO_HEAD.size + 32:
        return None
    magic, kind, role, plen, link_id, expires, local_ip, local_port = HELLO_HEAD.unpack_from(data, 0)
    if magic != MAGIC or kind != HELLO or role not in (HOST, DIALER) or not 0 < plen <= 120 \
            or len(data) != HELLO_HEAD.size + plen + 32 or expires < now or expires > now + LIFE + 5:
        return None
    try:
        prefix = data[HELLO_HEAD.size:HELLO_HEAD.size + plen].decode("utf-8")
    except UnicodeDecodeError:
        return None
    key = rendezvous_key(server_key, link_id, role, expires, prefix)
    if not hmac.compare_digest(hmac.new(key, data[:-32], hashlib.sha256).digest(), data[-32:]):
        return None
    return link_id, role, expires, prefix, key, local_ip, local_port


def intro(key: bytes, link_id: bytes, peer_ip: bytes, peer_port: int, peer_role: int, pair_id: bytes, session: bytes,
          peer_local_ip: bytes = b"\0\0\0\0", peer_local_port: int = 0) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    head, nonce = INTRO_HEAD.pack(MAGIC, INTRO, 0, 0, link_id), os.urandom(12)
    plain = (peer_ip + struct.pack("<HBB", peer_port, peer_role, 0) + pair_id + session + peer_local_ip
             + struct.pack("<H", peer_local_port))
    return head + nonce + AESGCM(key).encrypt(nonce, plain, head)


class Broker:
    """Pairs a host and a dialer of the same prefix. State is in memory: a rendezvous lives at most LIFE seconds."""

    def __init__(self, server_key: bytes):
        self.server_key = server_key
        self.waiting = {}       # prefix -> {role: (link_id, ip, port, key, expires)}
        self.paired = {}        # link_id -> (intro datagram, ip, port, expires)
        self.counts = {"hellos": 0, "refused": 0, "pairs": 0, "intros": 0}

    def receive(self, ip: bytes, port: int, data: bytes, now: float) -> list:
        """The datagrams to send, as (ip, port, payload), for one arrived datagram."""
        self.counts["hellos"] += 1
        parsed = parse_hello(self.server_key, data, now)
        if parsed is None:
            self.counts["refused"] += 1
            return []
        link_id, role, expires, prefix, key, local_ip, local_port = parsed
        done = self.paired.get(link_id)
        if done:                                    # a repeated HELLO: the INTRO may have been lost
            self.paired[link_id] = (done[0], ip, port, done[3])
            self.counts["intros"] += 1
            return [(ip, port, done[0])]
        slots = self.waiting.setdefault(prefix, {})
        slots[role] = (link_id, ip, port, key, expires, local_ip, local_port)
        other = slots.get(DIALER if role == HOST else HOST)
        if other is None or other[0] == link_id:
            return []
        del self.waiting[prefix]
        pair_id, session = os.urandom(16), os.urandom(32)
        me = (link_id, ip, port, key, expires, local_ip, local_port)
        out = []
        for a, b, b_role in ((me, other, DIALER if role == HOST else HOST), (other, me, role)):
            datagram = intro(a[3], a[0], b[1], b[2], b_role, pair_id, session, b[5], b[6])
            self.paired[a[0]] = (datagram, a[1], a[2], a[4])
            out.append((a[1], a[2], datagram))
        self.counts["pairs"] += 1
        self.counts["intros"] += 2
        return out

    def expire(self, now: float) -> None:
        for prefix in list(self.waiting):
            slots = {r: v for r, v in self.waiting[prefix].items() if v[4] >= now}
            if slots:
                self.waiting[prefix] = slots
            else:
                del self.waiting[prefix]
        self.paired = {k: v for k, v in self.paired.items() if v[3] >= now}

    def status(self) -> dict:
        return {"waiting": {p: sorted("host" if r == HOST else "engine" for r in s) for p, s in self.waiting.items()},
                "paired_links": len(self.paired), **self.counts, "at": time.time()}


def run(state_dir: Path, router: str, port: int) -> None:
    server_key = link_tickets.load_key(state_dir / "secrets")
    broker = Broker(server_key)
    host, _, rport = router.rpartition(":")
    conn = socket.create_connection((host, int(rport)), timeout=10)
    conn.settimeout(None)
    frames = queue.Queue()

    def read(stream):
        try:
            while True:
                head = stream.read(FRAME.size)
                if len(head) < FRAME.size:
                    break
                op, p, ip, peer_port, length = FRAME.unpack(head)
                frames.put((op, p, stream.read(length) if length else b""))
        except OSError:
            pass
        frames.put(None)

    threading.Thread(target=read, args=(conn.makefile("rb"),), daemon=True).start()
    conn.sendall(FRAME.pack(OPEN, port, b"\0\0\0\0", 0, 0))
    status_path = state_dir / "run" / STATUS_FILE
    print(f"[link-broker] UDP {port} through the router at {router}", flush=True)
    last_status = 0.0
    while True:
        try:
            frame = frames.get(timeout=1.0)
        except queue.Empty:
            frame = ()
        if frame is None:
            raise SystemExit("[link-broker] the UDP router closed the connection")
        now = time.time()
        if frame:
            op, p, payload = frame
            if op == OPENED:
                print(f"[link-broker] UDP {p} open", flush=True)
            elif op == NOTE:
                print(f"[link-broker] UDP {p}: {payload.decode('utf-8', 'replace')}", flush=True)
            elif op == BATCH and p == port:
                count, at = struct.unpack_from("<I", payload, 0)[0], 4
                for _ in range(count):
                    ip, peer_port, length = ENTRY.unpack_from(payload, at)
                    at += ENTRY.size
                    for to_ip, to_port, datagram in broker.receive(ip, peer_port, payload[at:at + length], now):
                        conn.sendall(FRAME.pack(3, port, to_ip, to_port, len(datagram)) + datagram)   # SEND
                    at += length
        if now - last_status >= 1.0:
            broker.expire(now)
            try:
                status_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(broker.status()), encoding="utf-8")
                os.replace(tmp, status_path)
            except OSError:
                pass
            last_status = now


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Nodus UDP broker (a local service of the UDP router)")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--router", default="127.0.0.1:47699")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    run(Path(args.state_dir), args.router, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

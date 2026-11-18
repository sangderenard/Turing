"""Nodus link tickets: what a peer presents on the two sockets of ``wss://<host>/ws/link/<prefix>/{control,bulk}``.

Research case 98d49cb8520e44d092c5d134c37c3c32 (ticket bc214e177933475e9f8ac3f5ed3e3762), stage 3.  The native Nodus listener
(nodus ``include/nodus_link.h`` ``parse_ticket`` / ``LinkListener``) checks every ticket; this module only issues them, after the
server has verified an existing peer grant (``authz.verify_peer``, the same check as ``/ws/contract``).  Python never carries link
traffic.

Layout (128 bytes, little-endian), identical to ``nodus::link::sign_ticket``:
    magic "NLT1" u32 | version 1 u16 | channel u8 (1 control, 2 bulk) | role u8 (1 host, 2 dialer) | link id [16] |
    sha256(prefix) [32] | account u64 | expires_at u64 | issued_at u64 | nonce [16] | HMAC-SHA256(key, bytes 0..95) [32]

The key is 32 random bytes in ``<secrets>/nodus_link.key``; the Nodus host that registers the listener reads the same file.
``python -m meowdoku.link_tickets key <secrets-dir>`` creates it once (refuses to overwrite).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import sys
import time
from pathlib import Path

MAGIC = 0x31544C4E
VERSION = 1
CONTROL, BULK = 1, 2
HOST, DIALER = 1, 2
TICKET_BYTES = 128
KEY_FILE = "nodus_link.key"
MAX_LIFE = 300          # the listener's default max_ticket_life
DEFAULT_LIFE = 120


class LinkKeyMissing(Exception):
    """This instance holds no link key, so it cannot issue tickets."""


def sign(key: bytes, channel: int, role: int, link_id: bytes, prefix: str, account: int, issued_at: int, expires_at: int,
         nonce: bytes) -> bytes:
    """One ticket, byte for byte what nodus::link::sign_ticket makes from the same fields."""
    if len(key) != 32 or len(link_id) != 16 or len(nonce) != 16:
        raise ValueError("key is 32 bytes, link id and nonce 16")
    if channel not in (CONTROL, BULK) or role not in (HOST, DIALER):
        raise ValueError("channel is 1 (control) or 2 (bulk); role is 1 (host) or 2 (dialer)")
    body = (struct.pack("<IHBB", MAGIC, VERSION, channel, role) + link_id + hashlib.sha256(prefix.encode("utf-8")).digest()
            + struct.pack("<QQQ", int(account), int(expires_at), int(issued_at)) + nonce)
    assert len(body) == 96
    return body + hmac.new(key, body, hashlib.sha256).digest()


def load_key(secrets_dir) -> bytes:
    path = Path(secrets_dir) / KEY_FILE
    try:
        key = path.read_bytes()
    except OSError as error:
        raise LinkKeyMissing(f"no link key at {path}") from error
    if len(key) != 32:
        raise LinkKeyMissing(f"{path} is not 32 bytes")
    return key


def create_key(secrets_dir) -> Path:
    """Write a new key; refuses when one exists (the listener and every live ticket depend on it)."""
    path = Path(secrets_dir) / KEY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:            # "x": never overwrite
        handle.write(secrets.token_bytes(32))
    return path


def issue(key: bytes, prefix: str, account: int, life: int = DEFAULT_LIFE, now: float | None = None) -> dict:
    """Both tickets of one new link for a dialer, as the REST answer carries them."""
    life = int(life)
    if not 0 < life <= MAX_LIFE:
        raise ValueError(f"life is 1..{MAX_LIFE} seconds")
    issued = int(time.time() if now is None else now)
    link_id = os.urandom(16)
    tickets = {name: sign(key, channel, DIALER, link_id, prefix, account, issued, issued + life, os.urandom(16))
               for name, channel in (("control", CONTROL), ("bulk", BULK))}
    return {"link_id": link_id.hex(), "account": int(account), "issued_at": issued, "expires_at": issued + life,
            "path": f"/ws/link/{prefix}/", "control_ticket": base64.b64encode(tickets["control"]).decode("ascii"),
            "bulk_ticket": base64.b64encode(tickets["bulk"]).decode("ascii")}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "key":
        try:
            print(create_key(sys.argv[2]))
        except FileExistsError:
            sys.exit(f"a link key already exists in {sys.argv[2]}; refusing to replace it")
    else:
        sys.exit("usage: python -m meowdoku.link_tickets key <secrets-dir>")

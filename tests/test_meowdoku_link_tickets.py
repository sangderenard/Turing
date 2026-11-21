"""Nodus link tickets (examples/meowdoku/link_tickets.py), research case 98d49cb8520e44d092c5d134c37c3c32.

The known answer is nodus ``tools/link_test`` ``link_test vector`` (nodus branch claude/nodus-link-stage1, 2026-11-18): the C++
signer and this one must agree byte for byte, or the native listener refuses every ticket the server issues."""
import base64
import hashlib
import hmac
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import link_tickets  # noqa: E402

CPP_VECTOR = ("4e4c543101000102101112131415161718191a1b1c1d1e1fa51146d38077ff992eba84cf1d5e9b249073503eb74868699016153b031a38f72a"
              "000000000000003887fd6a00000000c086fd6a00000000a0a1a2a3a4a5a6a7a8a9aaabacadaeafc5b6a7e3197bc41d0d3aad20e89bb5b0c3f4"
              "72162079a301da310a3eed71213c")


def test_matches_the_native_signer():
    key = bytes(range(32))
    ticket = link_tickets.sign(key, link_tickets.CONTROL, link_tickets.DIALER, bytes(range(0x10, 0x20)), "lab", 42,
                               1795000000, 1795000120, bytes(range(0xA0, 0xB0)))
    assert ticket.hex() == CPP_VECTOR


def test_issue_gives_two_tickets_of_one_link(tmp_path):
    path = link_tickets.create_key(tmp_path)
    key = link_tickets.load_key(tmp_path)
    answer = link_tickets.issue(key, "lab", 42, life=60, now=1795000000)
    control, bulk = base64.b64decode(answer["control_ticket"]), base64.b64decode(answer["bulk_ticket"])
    assert len(control) == len(bulk) == link_tickets.TICKET_BYTES
    assert control[8:24] == bulk[8:24] == bytes.fromhex(answer["link_id"])       # one link id
    assert control[6] == link_tickets.CONTROL and bulk[6] == link_tickets.BULK
    assert control[7] == bulk[7] == link_tickets.DIALER
    assert control[24:56] == hashlib.sha256(b"lab").digest()
    assert struct.unpack("<QQQ", control[56:80]) == (42, 1795000060, 1795000000)
    assert control[80:96] != bulk[80:96]                                        # fresh nonces
    for t in (control, bulk):
        assert hmac.compare_digest(t[96:], hmac.new(key, t[:96], hashlib.sha256).digest())
    assert answer["path"] == "/ws/link/lab/" and answer["expires_at"] == 1795000060
    with pytest.raises(FileExistsError):
        link_tickets.create_key(tmp_path)                                       # never replaced
    assert path.read_bytes() == key


def test_refusals(tmp_path):
    with pytest.raises(link_tickets.LinkKeyMissing):
        link_tickets.load_key(tmp_path)
    (tmp_path / link_tickets.KEY_FILE).write_bytes(b"short")
    with pytest.raises(link_tickets.LinkKeyMissing):
        link_tickets.load_key(tmp_path)
    with pytest.raises(ValueError):
        link_tickets.issue(bytes(32), "lab", 1, life=link_tickets.MAX_LIFE + 1)
    with pytest.raises(ValueError):
        link_tickets.sign(bytes(31), 1, 2, bytes(16), "lab", 1, 0, 1, bytes(16))

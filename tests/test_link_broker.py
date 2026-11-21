"""link_broker: a host and a dialer of one prefix are introduced to each other; nothing else is."""
import os
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402
from meowdoku import link_broker as lb  # noqa: E402

KEY = bytes(range(32))


def _side(direction, prefix="proof", now=1_000_000.0):
    r = lb.rendezvous(KEY, prefix, direction, now=now)
    link_id, key = bytes.fromhex(r["link_id"]), bytes.fromhex(r["key"])
    return r, link_id, key, lb.hello(key, link_id, r["role"], r["expires_at"], prefix)


def _open(key, datagram):
    head, nonce, sealed = datagram[:24], datagram[24:36], datagram[36:]
    plain = AESGCM(key).decrypt(nonce, sealed, head)
    ip, (port, role, _), pair, session = plain[:4], struct.unpack_from("<HBB", plain, 4), plain[8:24], plain[24:56]
    assert len(plain) == 62
    return socket.inet_ntoa(ip), port, role, pair, session


def test_pairs_host_and_dialer_and_seals_each_intro():
    now = 1_000_000.0
    broker = lb.Broker(KEY)
    _, hid, hkey, hhello = _side("host", now=now)
    _, did, dkey, dhello = _side("engine", now=now)
    assert broker.receive(socket.inet_aton("1.2.3.4"), 5000, hhello, now) == []
    out = broker.receive(socket.inet_aton("5.6.7.8"), 6000, dhello, now)
    assert sorted(port for _, port, _ in out) == [5000, 6000]
    to_host = next(d for _, port, d in out if port == 5000)
    to_dialer = next(d for _, port, d in out if port == 6000)
    hip, hport, hrole, hpair, hsession = _open(hkey, to_host)
    dip, dport, drole, dpair, dsession = _open(dkey, to_dialer)
    assert (hip, hport, hrole) == ("5.6.7.8", 6000, lb.DIALER)
    assert (dip, dport, drole) == ("1.2.3.4", 5000, lb.HOST)
    assert hpair == dpair and hsession == dsession
    # A repeated HELLO gets the same INTRO again (the first may have been lost), to its current address.
    again = broker.receive(socket.inet_aton("5.6.7.8"), 6001, dhello, now)
    assert again == [(socket.inet_aton("5.6.7.8"), 6001, to_dialer)]


def test_refuses_forged_expired_and_same_role():
    now = 1_000_000.0
    broker = lb.Broker(KEY)
    _, _, _, hello = _side("host", now=now)
    forged = hello[:-1] + bytes([hello[-1] ^ 1])
    assert broker.receive(b"\1\1\1\1", 1, forged, now) == []
    assert broker.receive(b"\1\1\1\1", 1, hello, now + lb.LIFE + 10) == []      # expired
    assert broker.counts["refused"] == 2
    _, _, _, h1 = _side("host", now=now)
    _, _, _, h2 = _side("host", now=now)
    assert broker.receive(b"\1\1\1\1", 1, h1, now) == []
    assert broker.receive(b"\2\2\2\2", 2, h2, now) == []                       # two hosts never pair
    assert broker.counts["pairs"] == 0
    assert broker.status()["waiting"] == {"proof": ["host"]}


def test_other_prefixes_do_not_meet():
    now = 1_000_000.0
    broker = lb.Broker(KEY)
    _, _, _, h = _side("host", prefix="a", now=now)
    _, _, _, d = _side("engine", prefix="b", now=now)
    broker.receive(b"\1\1\1\1", 1, h, now)
    assert broker.receive(b"\2\2\2\2", 2, d, now) == []

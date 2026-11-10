"""Federation phase 1: /ws/contract/<prefix> bridges a grant-authenticated peer to a (stubbed) loopback contract port.

No real Nodus host: a stub TCP server on an ephemeral 127.0.0.1 port records the Hello it receives and answers a Result with a Receipt.
Everything is on ephemeral ports and a temp state root."""
from __future__ import annotations

import json
import socket
import sqlite3
import struct
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import federation_peer as fp  # noqa: E402
from meowdoku import server as srv  # noqa: E402

FRAME = struct.Struct("<II8Q")
MAGIC = 0x31434E4E
OFFER, ACCEPT, RESULT, RECEIPT, HELLO = 1, 2, 3, 4, 5


def frame(kind, *values):
    return FRAME.pack(MAGIC, kind, *values, *([0] * (8 - len(values))))


class StubContractPort:
    """Stands in for the Nodus contract port: records every frame, answers each Result with a Receipt."""

    def __init__(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        self.port = self.listener.getsockname()[1]
        self.frames: list[bytes] = []
        self.connections = 0
        self.closed = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            while True:
                data = b""
                while len(data) < 72:
                    chunk = conn.recv(72 - len(data))
                    if not chunk:
                        return
                    data += chunk
                self.frames.append(data)
                _, kind, *values = FRAME.unpack(data)
                if kind == RESULT:
                    conn.sendall(frame(RECEIPT, *values[:7]))
        except OSError:
            pass
        finally:
            conn.close()
            self.closed.set()

    def wait_frames(self, n, timeout=5.0):
        end = time.time() + timeout
        while len(self.frames) < n and time.time() < end:
            time.sleep(0.01)
        return list(self.frames)


@pytest.fixture()
def fed(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "lowerings").mkdir()
    (web / "lowerings" / "demo.api.yaml").write_text("module: demo\nentry: run\nlanguage: wasm\n", encoding="utf-8")
    canvas = tmp_path / "instance.canvas"
    canvas.write_bytes(b"canvas-bytes-v1")
    stub = StubContractPort()
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    sessions.federation_port, sessions.federation_canvas = stub.port, str(canvas)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    uid = sessions.accounts.ensure_user("lab-peer", "correcthorse1")
    conn = sqlite3.connect(sessions.accounts_db)
    conn.executescript(authz._DDL)
    conn.execute("INSERT OR REPLACE INTO account_roles (user_id, role, granted_by, created_at) VALUES (?, 'service', 'test', 0)", (uid,))
    conn.commit()
    conn.close()
    env = SimpleNamespace(port=httpd.server_address[1], sessions=sessions, stub=stub, uid=uid, web=web, canvas=canvas,
                          digests=fp.identity_digests(canvas, web), db=sessions.accounts_db)
    env.url = f"ws://127.0.0.1:{env.port}/ws/contract/lab"
    yield env
    httpd.shutdown()
    httpd.server_close()
    stub.listener.close()


def _grant(env, direction="engine", ops="result,decision", days=1):
    return authz.grant_peer(env.db, "lab-peer", "lab", direction, ops, days, "test")


def test_capability_federate_roles():
    policy = authz.DEFAULT_POLICY["role_caps"]
    assert all("federate" in policy[r] for r in ("admin", "operator", "service"))
    assert "federate" not in policy["member"] and "federate" not in authz.DEFAULT_POLICY["tier_caps"]["edit"]


def test_grant_stores_only_a_hash_and_verifies(fed):
    gid, secret = _grant(fed)
    raw = sqlite3.connect(fed.db).execute("SELECT secret_hash FROM peer_grants").fetchone()[0]
    assert secret not in raw and len(raw) == 64
    assert authz.verify_peer(fed.db, gid, secret, "lab", "engine")["service_user_id"] == fed.uid
    assert authz.verify_peer(fed.db, gid, secret, "other", "engine") is None
    assert authz.verify_peer(fed.db, gid, secret, "lab", "host") is None
    assert authz.verify_peer(fed.db, gid, secret + "x", "lab", "engine") is None
    assert [g["grant_id"] for g in authz.list_peers(fed.db)] == [gid] and "secret_hash" not in authz.list_peers(fed.db)[0]
    with pytest.raises(authz.AuthzError):
        authz.grant_peer(fed.db, "lab-peer", "lab", "engine", "hello", 1, "test")          # hello is the bridge's


def test_expired_grant_refused(fed):
    gid, secret = _grant(fed, days=1)
    conn = sqlite3.connect(fed.db)
    conn.execute("UPDATE peer_grants SET expires_at = 1")
    conn.commit()
    conn.close()
    assert authz.verify_peer(fed.db, gid, secret, "lab", "engine") is None


def test_bridge_hello_forwarding_and_receipt(fed):
    gid, secret = _grant(fed, "engine", "result,decision")
    peer = fp.connect(fed.url, gid, secret, "engine", fed.digests)
    try:
        assert peer.hello_id == fed.uid
        peer.send_frame(frame(RESULT, 7, 1, 2, 3, 4, 5, 6, 99))
        got = fed.stub.wait_frames(2)
        hello, result = FRAME.unpack(got[0]), FRAME.unpack(got[1])
        assert hello[1] == HELLO and hello[2:5] == (2, fed.uid, 2)                 # role engine, id = service account, version 2
        assert result[1] == RESULT and result[2] == 7
        receipt = peer.recv_frame(timeout=5)
        assert receipt is not None and FRAME.unpack(receipt)[1] == RECEIPT and FRAME.unpack(receipt)[2] == 7
    finally:
        peer.close()


def test_host_direction_hello_role(fed):
    gid, secret = _grant(fed, "host", "result")
    peer = fp.connect(fed.url, gid, secret, "host", fed.digests)
    try:
        assert FRAME.unpack(fed.stub.wait_frames(1)[0])[2:5] == (1, fed.uid, 2)
    finally:
        peer.close()


def test_peer_socket_adapter_drives_nodus_peer_helper(fed):
    sys.path.insert(0, str(ROOT.parent / "nodus" / "tools"))
    try:
        import contract_service_peer as csp
    except ImportError:
        pytest.skip("nodus/tools/contract_service_peer.py not importable")
    finally:
        sys.path.pop(0)
    gid, secret = _grant(fed, "engine", "submit,decision")
    peer = fp.connect(fed.url, gid, secret, "engine", fed.digests)
    try:
        engine = csp.EnginePeer(peer, SimpleNamespace(request=lambda: None, route=lambda r: 1, admitted=lambda i: None))
        assert engine.submit((11, 0, 0, 0))
        engine.poll()
        got = fed.stub.wait_frames(2)
        assert FRAME.unpack(got[1])[1] == 6 and FRAME.unpack(got[1])[2] == 11
    finally:
        peer.close()


def test_wrong_secret_refused(fed):
    gid, secret = _grant(fed)
    with pytest.raises(fp.FederationRefused) as refused:
        fp.connect(fed.url, gid, secret + "x", "engine", fed.digests)
    assert refused.value.reason == "grant refused" and fed.stub.connections == 0


def test_revoked_grant_refused(fed):
    gid, secret = _grant(fed)
    assert authz.revoke_peer(fed.db, gid)
    with pytest.raises(fp.FederationRefused) as refused:
        fp.connect(fed.url, gid, secret, "engine", fed.digests)
    assert refused.value.reason == "grant refused" and fed.stub.connections == 0


def test_wrong_prefix_refused(fed):
    gid, secret = _grant(fed)
    with pytest.raises(fp.FederationRefused):
        fp.connect(fed.url.replace("/lab", "/elsewhere"), gid, secret, "engine", fed.digests)


def test_digest_mismatch_refused_with_reason(fed):
    gid, secret = _grant(fed)
    bad = {**fed.digests, "graph_definition_digest": "0" * 64}
    with pytest.raises(fp.FederationRefused) as refused:
        fp.connect(fed.url, gid, secret, "engine", bad)
    assert "graph_definition_digest" in refused.value.reason and "api_digests" not in refused.value.reason
    bad = {**fed.digests, "api_digests": ["demo:other"]}
    with pytest.raises(fp.FederationRefused) as refused:
        fp.connect(fed.url, gid, secret, "engine", bad)
    assert "api_digests" in refused.value.reason and fed.stub.connections == 0


def test_no_nodus_instance_refused(fed):
    gid, secret = _grant(fed)
    fed.sessions.federation_port = None
    with pytest.raises(fp.FederationRefused) as refused:
        fp.connect(fed.url, gid, secret, "engine", fed.digests)
    assert "no Nodus instance" in refused.value.reason


def test_kind_outside_allowed_ops_closes_1008(fed):
    gid, secret = _grant(fed, "engine", "result")
    peer = fp.connect(fed.url, gid, secret, "engine", fed.digests)
    try:
        peer.send_frame(frame(6, 1, 0, 0, 0))                                      # Submit is not granted
        assert peer.recv_frame(timeout=5) is None
        assert peer.close_code == 1008
        assert len(fed.stub.frames) == 1                                           # only the bridge's Hello ever reached the port
    finally:
        peer.close()


def test_peer_hello_is_never_forwarded(fed):
    gid, secret = _grant(fed, "engine", "result")
    peer = fp.connect(fed.url, gid, secret, "engine", fed.digests)
    try:
        peer.send_frame(frame(HELLO, 1, 999, 2))                                   # a peer claiming another id
        assert peer.recv_frame(timeout=5) is None and peer.close_code == 1008
        assert len(fed.stub.frames) == 1
    finally:
        peer.close()


def test_peer_closing_closes_the_upstream_socket(fed):
    gid, secret = _grant(fed, "engine", "result")
    peer = fp.connect(fed.url, gid, secret, "engine", fed.digests)
    fed.stub.wait_frames(1)
    peer.close()
    assert fed.stub.closed.wait(5)

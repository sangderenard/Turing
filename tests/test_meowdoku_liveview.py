"""[LIVEVIEW] The viewer's continuous, read-only feed: board games by poll, real-time games by a push socket."""

import base64
import json
import os
import socket
import struct
import time

from tests.test_meowdoku_share import Client, server  # noqa: F401,E402  (the real handler fixture; also sets sys.path)
from meowdoku import share  # noqa: E402


class WS:
    """A bare WebSocket client (masked frames up, unmasked frames down)."""

    def __init__(self, port, path, cookie=None, protocol=None, timeout=5.0):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        extra = (f"Cookie: {cookie}\r\n" if cookie else "") + (f"Sec-WebSocket-Protocol: {protocol}\r\n" if protocol else "")
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n{extra}\r\n").encode())
        self.rf = self.s.makefile("rb")
        self.status = int(self.rf.readline().split()[1])
        self.headers = {}
        while True:
            line = self.rf.readline()
            if line in (b"\r\n", b""):
                break
            k, _, v = line.decode().partition(":")
            self.headers[k.strip().lower()] = v.strip()

    def send(self, text):
        p = text.encode(); mask = os.urandom(4)
        head = bytes([0x81, 0x80 | len(p)]) if len(p) < 126 else bytes([0x81, 0xFE]) + struct.pack(">H", len(p))
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(p)))

    def recv(self):
        h = self.rf.read(2)
        if len(h) < 2:
            return None
        n = h[1] & 127
        if n == 126:
            n = struct.unpack(">H", self.rf.read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self.rf.read(8))[0]
        return h[0] & 15, self.rf.read(n)

    def snapshot(self):
        """The next binary frame (text meta frames skipped); the close frame, or None, when the socket ended."""
        while True:
            m = self.recv()
            if m is None or m[0] == 8 or m[0] == 2:
                return m

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def _owner_ticker(port, cookie):
    ws = WS(port, "/ws", cookie=cookie)
    assert ws.status == 101

    def tick():
        ws.send(json.dumps({"inputs": []}))
        m = ws.snapshot()
        assert m and m[0] == 2
    return ws, tick


def _close_code(frame):
    return struct.unpack(">H", frame[1][:2])[0]


def test_viewer_socket_streams_the_owner_snapshots_and_takes_no_input(server):  # noqa: F811
    srv, sessions, port = server
    owner = Client(port)
    assert owner.call("POST", "/api/new", {"game": "space_invaders"})[0] == 200
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    ows, tick = _owner_ticker(port, owner.cookie)
    try:
        viewer = WS(port, "/ws/spectate", protocol=f"mdv.{token}")
        assert viewer.status == 101 and viewer.headers["sec-websocket-protocol"] == f"mdv.{token}"
        tick()
        first = viewer.snapshot()                                # a late viewer starts on a keyframe
        assert first and first[0] == 2
        sizes, t0 = [], time.perf_counter()
        for _ in range(12):                                      # N successive owner ticks reach the viewer
            tick()
            got = viewer.snapshot()
            assert got and got[0] == 2
            sizes.append(len(got[1]))
        print("viewer: 12 owner ticks -> 12 snapshots in %.2fs, mean %d bytes" % (time.perf_counter() - t0, sum(sizes) / len(sizes)))
        assert owner.call("POST", "/api/spectate", {"op": "status"})[1]["viewers"] == 1
        # inputs: any data frame ends the socket (1008), unread; nothing reached the owner's game
        before = owner.call("GET", "/api/state")[1]
        viewer.send(json.dumps({"inputs": [{"type": "fire"}], "move_axis": 1}))
        end = viewer.snapshot()
        assert end is not None and end[0] == 8 and _close_code(end) == 1008
        assert owner.call("GET", "/api/state")[1]["you"] == before["you"]
        viewer.close()
    finally:
        ows.close()


def test_viewer_socket_needs_a_live_token_and_the_owner_socket_refuses_viewers(server):  # noqa: F811
    srv, sessions, port = server
    owner = Client(port)
    owner.call("POST", "/api/new", {"game": "space_invaders"})
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    assert WS(port, "/ws/spectate", protocol="mdv.nope").status == 410
    assert WS(port, "/ws/spectate").status == 410
    # the owner's input socket never opens for the viewer's token, even carrying the owner's cookie
    assert WS(port, "/ws", cookie=owner.cookie, protocol=f"mdv.{token}").status == 403
    sessions.spectate.cap = 2                                     # the cap counts sockets
    a, b = WS(port, "/ws/spectate", protocol=f"mdv.{token}"), WS(port, "/ws/spectate", protocol=f"mdv.{token}")
    assert a.status == b.status == 101
    assert WS(port, "/ws/spectate", protocol=f"mdv.{token}").status == 429
    a.close(); b.close()


def test_revoke_stops_the_stream(server):  # noqa: F811
    srv, sessions, port = server
    owner = Client(port)
    owner.call("POST", "/api/new", {"game": "space_invaders"})
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    ows, tick = _owner_ticker(port, owner.cookie)
    try:
        viewer = WS(port, "/ws/spectate", protocol=f"mdv.{token}")
        tick()
        assert viewer.snapshot()[0] == 2
        owner.call("POST", "/api/spectate", {"op": "stop"})
        tick()
        end = viewer.snapshot()
        assert end is not None and end[0] == 8 and _close_code(end) == 4410
        viewer.close()
    finally:
        ows.close()


def test_board_game_updates_reach_a_polling_viewer_within_half_a_second(server):  # noqa: F811
    srv, sessions, port = server
    owner = Client(port)
    assert owner.call("POST", "/api/new", {"game": "wordle"})[0] == 200
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    viewer = Client(port, {share.VIEWER_HEADER: token})
    first = viewer.call("GET", "/api/spectate/watch?rev=")[1]
    rev, vid, seen, lat = first["state_rev"], first["vid"], 0, []
    for n in range(1, 6):
        owner.call("POST", "/api/action", {"type": "text", "text": "CRANE"[:n]})
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            d = viewer.call("GET", f"/api/spectate/watch?vid={vid}&rev={rev}")[1]
            if "state" in d:
                rev = d["state_rev"]; seen += 1; lat.append(time.perf_counter() - t0)
                break
            time.sleep(0.05)
    print("board game: update latency per move (poll every 50 ms in this test):", [round(x, 3) for x in lat])
    assert seen == 5 and max(lat) < 0.5


def test_fountain_keyframe_travels_through_the_octet_codec_to_the_viewer(server):  # noqa: F811
    import base64
    from meowdoku import loading_anim
    srv, sessions, port = server
    owner = Client(port)
    owner.call("POST", "/api/new", {"game": "fountain"})
    token = owner.call("POST", "/api/spectate", {"op": "start"})[1]["token"]
    total = 4096
    head = b"MFK1" + bytes([1, 0]) + struct.pack("<HIIH", 1, 777, total, total) + struct.pack("<8f", -3.3, 3.3, 0.0, 0.3, -1, 1, -1, 1)
    assert len(head) == share.KEY_HEADER_BYTES
    body = bytes((i * 7) % 251 if i % 5 else 0 for i in range(8 * total))
    raw = head + body
    status, made = owner.call("POST", "/api/spectate/key", {"key": base64.b64encode(raw).decode()})
    assert status == 200 and made["seq"] == 1
    assert owner.call("POST", "/api/spectate/key", {"key": base64.b64encode(b"junk").decode()})[0] == 400      # not an MFK1 stream
    assert owner.call("POST", "/api/spectate/key", {"key": "!!!"})[0] == 400
    viewer = Client(port, {share.VIEWER_HEADER: token})
    first = viewer.call("GET", "/api/spectate/watch?rev=")[1]
    assert first["key_seq"] == 1 and loading_anim.unpack(base64.b64decode(first["key"])) == raw     # the octet packet, lossless
    again = viewer.call("GET", f"/api/spectate/watch?vid={first['vid']}&rev={first['state_rev']}&kseq=1")[1]
    assert "key" not in again                                                                     # sent once per keyframe
    print("fountain keyframe: %d raw bytes -> %d packed bytes" % (len(raw), len(base64.b64decode(first["key"]))))
    # a viewer cannot post a keyframe
    assert viewer.call("POST", "/api/spectate/key", {"key": base64.b64encode(raw).decode()})[0] == 403


def test_keyframe_codec_in_node():
    import shutil
    import subprocess
    from pathlib import Path
    import pytest
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(["node", "--experimental-default-type=module", str(root / "tests" / "meowdoku_liveview_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr

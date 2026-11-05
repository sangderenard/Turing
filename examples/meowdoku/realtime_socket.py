"""One WebSocket per real-time game: input intervals up, binary deltas down.

The HTTP API stays for everything that is not a tick (new game, reset,
undo, layout, profiles).  A real-time page opens ``/ws`` on the same origin
-- the session cookie rides on the upgrade, so the socket drives the same
``Table`` -- and then, instead of POSTing ticks, sends one small JSON text
frame per tick: ``{"inputs": [...], <held controls>}``.  The server lands
them (``EntityGame.tick``) and answers with one binary frame: a SNAPSHOT
DELTA, little-endian:

    u8   kind = 1
    u8   flags        bit0 keyframe (every lane sent), bit1 clear (forget
                      every entity: a new game, a reset, a rewind)
    i32  ack          last input interval landed (-1: none)
    f32  x, y, vx, vy the player's ship (``you``), u8 alive, f32 shield charge,
                      f32 sync hint (the longest a faithful page may wait to check in),
                      f32 z, vz (the ship height above the board; 0 without a third axis),
                      f32 health (fraction; 0 for a game without health points)
    u8   names        new appearance names this connection has not seen,
                      each u8 length + utf-8; indices count up from 0
    u16  gone         lanes whose entity disappeared, u16 each
    u16  spawned      new entities: u16 lane, u8 mesh, u8 color,
                      u16 size*1000, i16 roll*1000, i16 spin*1000, u8 flags (bit0 bob, bit1 face its velocity),
                      i16 lean*1000, u16 owner (65535: none -- the lane that
                      fired a projectile), f32 age (world seconds since it
                      spawned), f32 x, y, vx, vy, z, vz
    u16  updated      changed entities: u16 lane, u8 mesh, u8 color,
                      f32 x, y, vx, vy, z, vz

An entity is UPDATED only when the page's extrapolation of what it was last
sent (position + velocity x world time) would be off by more than
``DRIFT``, or its velocity or look changed; a KEYFRAME (every lane) goes
every ``KEYFRAME_S`` of world time.  Constant-velocity bullets therefore
cost their spawn record and nothing until they die.  Everything else in the
game state (status, palette, motion constants) goes as a JSON text frame
``{"meta": ...}`` at most ``META_HZ`` times a second -- only the fields that
changed since the last one (the page merges), everything when the game is
cleared.

Frames are RFC 6455: client frames masked, server frames not; text, binary,
ping/pong and close are handled; nothing else is offered.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import time
from typing import Any

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
DRIFT = 0.02
KEYFRAME_S = 0.5
META_HZ = 4.0


# ---------------------------------------------------------------- RFC 6455 frames

def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")


def read_frame(stream) -> tuple[int, bytes] | None:
    """One complete message (opcode, payload); None when the peer is gone.
    Fragmented messages are joined."""
    opcode, payload = None, b""
    while True:
        head = stream.read(2)
        if len(head) < 2:
            return None
        fin, op = head[0] & 0x80, head[0] & 0x0F
        masked, length = head[1] & 0x80, head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", stream.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", stream.read(8))[0]
        mask = stream.read(4) if masked else b""
        data = stream.read(length)
        if len(data) < length:
            return None
        if masked:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data)) if length < 64 else _unmask(data, mask)
        if op in (0x8, 0x9, 0xA):          # control frames are never fragmented
            return op, data
        opcode = op if opcode is None else opcode
        payload += data
        if fin:
            return opcode, payload


def _unmask(data: bytes, mask: bytes) -> bytes:
    n = len(data)
    key = int.from_bytes((mask * (n // 4 + 1))[:n], "little")
    return (int.from_bytes(data, "little") ^ key).to_bytes(n, "little")


def frame(opcode: int, payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        head = struct.pack(">BB", 0x80 | opcode, n)
    elif n < 65536:
        head = struct.pack(">BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack(">BBQ", 0x80 | opcode, 127, n)
    return head + payload


# ---------------------------------------------------------------- snapshot deltas

class SnapshotEncoder:
    """What one connection's page knows, and the delta that brings it up to date."""

    def __init__(self):
        self.names: dict[str, int] = {}
        self.known: dict[int, dict[str, Any]] = {}     # lane -> what the page was last told
        self.game = None
        self.epoch = None
        self.keyframe_at = None
        self.meta_at = 0.0
        self.meta_sent: dict[str, Any] = {}
        self.bytes_sent = 0
        self.frames_sent = 0

    def _name(self, name: str, fresh: list[str]) -> int:
        if name not in self.names:
            self.names[name] = len(self.names)
            fresh.append(name)
        return self.names[name]

    def encode(self, game, state: dict[str, Any]) -> tuple[bytes, dict[str, Any] | None]:
        """(binary delta, meta or None) for ``state`` (the game's full state)."""
        clock = float(state.get("world_clock") or 0.0)
        epoch = getattr(game, "epoch", 0)
        clear = game is not self.game or epoch != self.epoch
        if clear:
            self.game, self.epoch, self.known, self.keyframe_at = game, epoch, {}, None
        keyframe = clear or self.keyframe_at is None or clock - self.keyframe_at >= KEYFRAME_S or clock < self.keyframe_at
        if keyframe:
            self.keyframe_at = clock
        fresh: list[str] = []
        current: dict[int, dict[str, Any]] = {}
        spawned, updated = [], []
        for entity in state.get("entities") or ():
            lane_text, born = str(entity["id"]).split(":", 1)
            lane = int(lane_text)
            record = {"born": born, "x": entity["x"], "y": entity["y"], "vx": entity["vx"], "vy": entity["vy"],
                      "z": float(entity.get("z", 0.0)), "vz": float(entity.get("vz", 0.0)),
                      "mesh": self._name(entity["mesh"], fresh), "color": self._name(entity["color"], fresh),
                      "clock": clock}
            current[lane] = record
            before = self.known.get(lane)
            if before is None or before["born"] != born:
                spawned.append((lane, record, entity))
                continue
            dt = clock - before["clock"]
            drift = max(abs(before["x"] + before["vx"] * dt - record["x"]), abs(before["y"] + before["vy"] * dt - record["y"]),
                        abs(before["z"] + before["vz"] * dt - record["z"]))
            if (keyframe or drift > DRIFT or before["vx"] != record["vx"] or before["vy"] != record["vy"]
                    or before["vz"] != record["vz"]
                    or before["mesh"] != record["mesh"] or before["color"] != record["color"]):
                updated.append((lane, record))
            else:
                current[lane] = before                 # the page still extrapolates from that
        # A lane whose entity died is gone; one respawned under the same lane
        # arrives as a spawn, which replaces what the page had there.
        gone = [lane for lane in self.known if lane not in current]
        self.known = current

        you = state.get("you") or {}
        ack = state.get("ack")
        parts = [struct.pack("<BBi4fBfffff", 1, (1 if keyframe else 0) | (2 if clear else 0), -1 if ack is None else int(ack),
                             float(you.get("x", 0.0)), float(you.get("y", 0.0)), float(you.get("vx", 0.0)),
                             float(you.get("vy", 0.0)), 1 if you.get("alive") else 0, float(you.get("shield", 0.0)),
                             float(state.get("sync_hint") or 0.03),
                             float(you.get("z", 0.0)), float(you.get("vz", 0.0)), float(you.get("health", 0.0)))]
        parts.append(struct.pack("<B", len(fresh)))
        for name in fresh:
            raw = name.encode("utf-8")
            parts.append(struct.pack("<B", len(raw)) + raw)
        parts.append(struct.pack("<H", len(gone)) + b"".join(struct.pack("<H", lane) for lane in gone))
        parts.append(struct.pack("<H", len(spawned)))
        for lane, record, entity in spawned:
            owner = int(entity.get("owner", -1))
            parts.append(struct.pack("<HBBHhhBhHf6f", lane, record["mesh"], record["color"],
                                     int(round(entity["size"] * 1000)), int(round(entity["roll"] * 1000)),
                                     int(round(entity["spin"] * 1000)),
                                     (1 if entity.get("bob") else 0) | (2 if entity.get("face") else 0),
                                     int(round(entity.get("lean", 0.55) * 1000)), 65535 if owner < 0 else owner,
                                     float(entity.get("age", 0.0)),
                                     record["x"], record["y"], record["vx"], record["vy"], record["z"], record["vz"]))
        parts.append(struct.pack("<H", len(updated)))
        for lane, record in updated:
            parts.append(struct.pack("<HBB6f", lane, record["mesh"], record["color"],
                                     record["x"], record["y"], record["vx"], record["vy"], record["z"], record["vz"]))
        binary = b"".join(parts)

        meta = None
        now = time.perf_counter()
        if clear or now - self.meta_at >= 1.0 / META_HZ:
            # Only what changed since the last meta (the page merges it into
            # the state it holds); everything on a clear.
            self.meta_at = now
            full = {key: value for key, value in state.items() if key != "entities"}
            meta = full if clear else {key: value for key, value in full.items()
                                       if self.meta_sent.get(key) != value}
            meta["game"] = full.get("game")
            self.meta_sent = full
        self.bytes_sent += len(binary) + (len(json.dumps(meta)) if meta else 0)
        self.frames_sent += 1
        return binary, meta

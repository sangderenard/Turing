"""One WebSocket per real-time game: input intervals up, binary deltas down.

The HTTP API stays for everything that is not a tick (new game, reset,
undo, layout, profiles).  A real-time page opens ``/ws`` on the same origin
-- the session cookie rides on the upgrade, so the socket drives the same
``Table`` -- and then, instead of POSTing ticks, sends one small JSON text
frame per tick: ``{"inputs": [...], <held controls>}``.  The server lands
them (``EntityGame.tick``) and answers with one binary frame: a SNAPSHOT
DELTA, little-endian:

    u8   kind = PROTOCOL (the frame layout's version: a page reading another
                      layout must not misparse it -- both servers serve one page)
    u8   flags        bit0 keyframe (every lane sent), bit1 clear (forget
                      every entity: a new game, a reset, a rewind), bit2 bodies
                      (a spherical game: every spawn and update record ends
                      with the body's flight state, below)
    i32  ack          last input interval landed (-1: none)
    f32  x, y, vx, vy the player's ship (``you``), u8 alive, f32 shield charge,
                      f32 sync hint (the longest a faithful page may wait to check in),
                      f32 z, vz (the ship height above the board; 0 without a third axis),
                      f32 health (fraction; 0 for a game without health points),
                      f32 yaw, vyaw, spd, vspd (a spherical game's turn and
                      throttle axes; 0 elsewhere),
                      f32 map_u, map_v, map_gx, map_gy (the land under its
                      ship: entity_motion.ground_travel's state),
                      f32 bank, vbank (its roll axis), beam (its laser's length; 0 off),
                      f32 ux, uy, uz, hx, hy, hz (its place and heading on the sphere),
                      f32 x5 the demo pilot's controls (roll, throttle, pitch, rudder,
                      fire; 0 unless the autopilot flies)
    u8   names        new appearance names this connection has not seen,
                      each u8 length + utf-8; indices count up from 0
    u16  gone         lanes whose entity disappeared, u16 each
    u16  spawned      new entities: u16 lane, u8 mesh, u8 color,
                      u16 size*1000, i16 roll*1000, i16 spin*1000, u8 flags (bit0 bob, bit1 face its velocity, bit2 translucent: drawn with the glass,
                      bit3 ribbon: a lightning end, drawn as the bolt between a strike's two ends,
                      bit4 ground: sits on the land's relief, wherever it stands,
                      bit5 glow: a city light -- a glowing point, not a mesh,
                      bit6 orient: drawn turned by its heading in the patch, which the page reads from the engine's sphere_heading),
                      i16 lean*1000, u16 owner (65535: none -- the lane that
                      fired a projectile), f32 age (world seconds since it
                      spawned), f32 x, y, vx, vy, z, vz [bodies: u8 flight, f32 ux, uy, uz,
                      hx, hy, hz, spd, turn]
    u16  updated      changed entities: u16 lane, u8 mesh, u8 color, i16 roll*1000,
                      f32 x, y, vx, vy, z, vz [bodies: u8 flight, f32 ux, uy, uz,
                      hx, hy, hz, spd, turn]

BODIES (a spherical game).  The page never moves a body with arithmetic of its
own: it carries each on from the state it was last sent -- place and heading on
the sphere (u, h), ground speed, the rate its heading was last turned at
(``turn``, rad/s), height and climb rate (z, vz), and the ``flight`` that says
which engine laws fly it (scroller.FLY_*) -- by the very laws the server's step
flew it with (``sphere_motion.carry``: sphere_turn, sphere_advance, air_gain,
climb, the trajectory step; the page runs their WebAssembly cores) and projects
it into the patch about its own ship with ``sphere_project``.  The encoder runs
that same text to know where the page has the body, so a body is UPDATED only
when it is not where the page puts it, or its flight, speed, turn or climb
changed.

Otherwise (a flat game) an entity is UPDATED only when the page's linear
extrapolation of what it was last sent (position + velocity x world time)
would be off by more than ``DRIFT``, or its velocity or look changed.  A
KEYFRAME (every lane) goes every ``KEYFRAME_S`` of world time.  Constant-velocity
bullets therefore cost their spawn record and nothing until they die.  Everything else in the
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


#: The snapshot frame layout's version, sent as the frame's first byte.  Bump
#: it whenever the layout changes; the page refuses any other version.
#:   3 = header + z, vz, health; spawn/update records carry z, vz; spawn
#:       flags bit1 = face its velocity.
#:   9 = frame flags bit2 (bodies): spawn and update records end with the
#:       body's flight state (u8 flight, f32 u, h, spd, turn).
PROTOCOL = 9

def _pack_body(body: dict[str, Any]) -> bytes:
    """A body's flight state: u8 flight, f32 u (3), h (3), spd, turn."""
    return struct.pack("<B8f", int(body["flight"]), *(float(v) for v in body["u"]), *(float(v) for v in body["h"]),
                       float(body["spd"]), float(body["turn"]))


class SnapshotEncoder:
    """What one connection's page knows, and the delta that brings it up to date."""

    def __init__(self):
        self.names: dict[str, int] = {}
        self.known: dict[int, dict[str, Any]] = {}     # lane -> what the page was last told
        self.sim: dict[int, dict[str, Any]] = {}       # lane -> where the page has the body now (its flight state)
        self.sim_clock = None
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

    @staticmethod
    def _flight_state(record: dict[str, Any]) -> dict[str, Any]:
        """The body as the page holds it after a record: what the flight laws carry on."""
        b = record["body"]
        return {"u": list(b["u"]), "h": list(b["h"]), "spd": b["spd"], "turn": b["turn"], "flight": b["flight"],
                "z": record["z"], "vz": record["vz"]}

    def _carry_bodies(self, clock: float, you: dict[str, Any], predict: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
        """Where the page has every body it is carrying now: each carried on
        from the state it holds by the flight laws (``sphere_motion.carry``,
        the text the page's cores are lowered from) for the world time since
        the last frame, in sub-steps no longer than the server's step cap, and
        projected into the patch about the ship (``sphere_project``).  Returns
        lane -> (x, y, z) in the patch's coordinates, and advances ``self.sim``."""
        import numpy as np
        from meowdoku import sphere_motion as sm
        from src.common.tensors import AbstractTensor

        lanes = list(self.sim)
        if not lanes or self.sim_clock is None:
            return {}
        cfg, sphere, patch = predict["bodies"], predict["sphere"], predict["patch"]
        T = lambda rows: AbstractTensor.get_tensor(np.asarray(rows, dtype=np.float64))  # noqa: E731
        bodies = [self.sim[lane] for lane in lanes]
        u = tuple(T([b["u"][i] for b in bodies]) for i in range(3))
        h = tuple(T([b["h"][i] for b in bodies]) for i in range(3))
        spd, z, vz = T([b["spd"] for b in bodies]), T([b["z"] for b in bodies]), T([b["vz"] for b in bodies])
        turn, flight = T([b["turn"] for b in bodies]), T([float(b["flight"]) for b in bodies])
        gap = max(0.0, clock - self.sim_clock)
        steps = 0 if gap <= 0.0 else max(1, min(60, int(np.ceil(gap / cfg["step"] - 1e-9))))
        for _ in range(steps):
            u, h, spd, z, vz = sm.carry(u, h, spd, turn, z, vz, flight, gap / steps, sphere["radius"], sphere["air"],
                                        cfg["ballistic"], cfg["flight"])
        un = [np.asarray(c.numpy(), dtype=np.float64) for c in u]
        hn = [np.asarray(c.numpy(), dtype=np.float64) for c in h]
        spd_n, z_n, vz_n = (np.asarray(c.numpy(), dtype=np.float64) for c in (spd, z, vz))
        for i, lane in enumerate(lanes):
            b = self.sim[lane]
            b["u"], b["h"] = [float(un[k][i]) for k in range(3)], [float(hn[k][i]) for k in range(3)]
            b["spd"], b["z"], b["vz"] = float(spd_n[i]), float(z_n[i]), float(vz_n[i])
        centre = tuple(T(float(you.get(n, 0.0))) for n in ("ux", "uy", "uz"))
        heading = tuple(T(float(you.get(n, 0.0))) for n in ("hx", "hy", "hz"))
        X, Y = sm.project(u, centre, heading, sphere["radius"])
        x = patch["cx"] + np.asarray(X.numpy(), dtype=np.float64)
        y = patch["cy"] - np.asarray(Y.numpy(), dtype=np.float64)
        return {lane: (float(x[i]), float(y[i]), float(z_n[i])) for i, lane in enumerate(lanes)}

    def encode(self, game, state: dict[str, Any]) -> tuple[bytes, dict[str, Any] | None]:
        """(binary delta, meta or None) for ``state`` (the game's full state)."""
        clock = float(state.get("world_clock") or 0.0)
        epoch = getattr(game, "epoch", 0)
        clear = game is not self.game or epoch != self.epoch
        if clear:
            self.game, self.epoch, self.known, self.keyframe_at = game, epoch, {}, None
            self.sim, self.sim_clock = {}, None
        keyframe = clear or self.keyframe_at is None or clock - self.keyframe_at >= KEYFRAME_S or clock < self.keyframe_at
        if keyframe:
            self.keyframe_at = clock
        fresh: list[str] = []
        current: dict[int, dict[str, Any]] = {}
        spawned, updated = [], []
        you = state.get("you") or {}
        predict = state.get("predict") or {}
        carrying = bool(predict.get("bodies") and predict.get("sphere") and predict.get("patch"))
        where_page_has = self._carry_bodies(clock, you, predict) if carrying else {}
        sim: dict[int, dict[str, Any]] = {}
        for entity in state.get("entities") or ():
            lane_text, born = str(entity["id"]).split(":", 1)
            lane = int(lane_text)
            record = {"born": born, "x": entity["x"], "y": entity["y"], "vx": entity["vx"], "vy": entity["vy"],
                      "z": float(entity.get("z", 0.0)), "vz": float(entity.get("vz", 0.0)),
                      "mesh": self._name(entity["mesh"], fresh), "color": self._name(entity["color"], fresh),
                      "clock": clock, "roll": round(float(entity.get("roll", 0.0)) * 20.0) / 20.0}
            body = entity.get("body") if carrying else None
            if carrying:
                record["body"] = body = body or {"u": [0.0, 0.0, 1.0], "h": [0.0, 1.0, 0.0], "spd": 0.0, "turn": 0.0, "flight": 1}
            current[lane] = record
            before = self.known.get(lane)
            if before is None or before["born"] != born:
                spawned.append((lane, record, entity))
                if carrying:
                    sim[lane] = self._flight_state(record)
                continue
            dt = clock - before["clock"]
            if carrying:
                # The page carries the body on by the flight laws from the state
                # it was told: is it where the server has it, and does it still
                # fly the way it was told?  (A ballistic body's speed and climb
                # are the law's to evolve, a body on its heading holds them.)
                at = where_page_has.get(lane)
                drift = (max(abs(at[0] - record["x"]), abs(at[1] - record["y"]), abs(at[2] - record["z"]))
                         if at is not None else float("inf"))
                b0, flies = before["body"], body["flight"]
                ballistic = flies in (predict["bodies"]["flight"]["missile"], predict["bodies"]["flight"]["bomb"],
                                      predict["bodies"]["flight"]["canopy"])
                moved = (b0["flight"] != flies or b0["turn"] != body["turn"]
                         or (not ballistic and (b0["spd"] != body["spd"] or before["vz"] != record["vz"])))
                told = keyframe or drift > DRIFT or moved
            else:
                drift = max(abs(before["x"] + before["vx"] * dt - record["x"]), abs(before["y"] + before["vy"] * dt - record["y"]),
                            abs(before["z"] + before["vz"] * dt - record["z"]))
                told = (keyframe or drift > DRIFT or before["vx"] != record["vx"] or before["vy"] != record["vy"]
                        or before["vz"] != record["vz"])
            if (told or before["mesh"] != record["mesh"] or before["color"] != record["color"]
                    or before["roll"] != record["roll"]):
                updated.append((lane, record))
                if carrying:
                    sim[lane] = self._flight_state(record)
            else:
                current[lane] = before                 # the page still carries on from that
                if carrying:
                    sim[lane] = self.sim.get(lane) or self._flight_state(before)
        # A lane whose entity died is gone; one respawned under the same lane
        # arrives as a spawn, which replaces what the page had there.
        gone = [lane for lane in self.known if lane not in current]
        self.known = current
        self.sim, self.sim_clock = sim, clock

        ack = state.get("ack")
        parts = [struct.pack("<BBi4fBfffffffffffffffffffffffffff", PROTOCOL, (1 if keyframe else 0) | (2 if clear else 0) | (4 if carrying else 0), -1 if ack is None else int(ack),
                             float(you.get("x", 0.0)), float(you.get("y", 0.0)), float(you.get("vx", 0.0)),
                             float(you.get("vy", 0.0)), 1 if you.get("alive") else 0, float(you.get("shield", 0.0)),
                             float(state.get("sync_hint") or 0.03),
                             float(you.get("z", 0.0)), float(you.get("vz", 0.0)), float(you.get("health", 0.0)),
                             float(you.get("yaw", 0.0)), float(you.get("vyaw", 0.0)), float(you.get("spd", 0.0)),
                             float(you.get("vspd", 0.0)),
                             float(you.get("map_u", 0.0)), float(you.get("map_v", 0.0)),
                             float(you.get("map_gx", 0.0)), float(you.get("map_gy", -1.0)),
                             float(you.get("bank", 0.0)), float(you.get("vbank", 0.0)), float(you.get("beam", 0.0)),
                             *(float(you.get(n, 0.0)) for n in ("ux", "uy", "uz", "hx", "hy", "hz")),
                             *(float(v) for v in (you.get("pilot") or [0.0] * 5)))]
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
                                     (1 if entity.get("bob") else 0) | (2 if entity.get("face") else 0)
                                     | (4 if entity.get("translucent") else 0) | (8 if entity.get("ribbon") else 0)
                                     | (16 if entity.get("ground") else 0) | (32 if entity.get("glow") else 0)
                                     | (64 if entity.get("orient") else 0),
                                     int(round(entity.get("lean", 0.55) * 1000)), 65535 if owner < 0 else owner,
                                     float(entity.get("age", 0.0)),
                                     record["x"], record["y"], record["vx"], record["vy"], record["z"], record["vz"]))
            if carrying:
                parts.append(_pack_body(record["body"]))
        parts.append(struct.pack("<H", len(updated)))
        for lane, record in updated:
            parts.append(struct.pack("<HBBh6f", lane, record["mesh"], record["color"], int(round(record["roll"] * 1000)),
                                     record["x"], record["y"], record["vx"], record["vy"], record["z"], record["vz"]))
            if carrying:
                parts.append(_pack_body(record["body"]))
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

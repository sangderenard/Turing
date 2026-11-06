"""Delve: a room-graph platformer ("metroidvania") as one dt-system entity law.

An original game of the room-graph dungeon genre.  The world is a big map of
ROOMS on a coarse grid (``delve_world.py``); a room is one or more screens
(wide rooms scroll sideways, shafts scroll up and down, big ones both) joined
to its neighbours by DOORS: openings in the side walls, and ladder shafts
through floors and ceilings.  Inside a room it is a platformer; between rooms
it is a tile map you walk across, the difference being gravity.

ONE law, whatever the room.  The room is DATA: the law is shaped by a fixed
room grid (``MAXW x MAXH`` tiles, one lane per tile, as Tile Quest's map), the
player and a handful of enemies, and a room transition only rewrites columns
(``tile``, the enemies' places, ``room_w`` / ``room_h``) -- the compiled law is
the same, never rebuilt for a room.  The page predicts the player's sideways
motion with the engine's ``damped_axis`` exactly as the climber does; the
server is the authority over everything else.

Lanes (contiguous blocks):
    0            the player (a frog); feet at ``y + ry``
    1..4         walkers (reverse at walls and ledges; a stomp kills them)
    5..6         flyers (bat: patrol sideways about their home, bobbing)
    7..          the room: one lane per tile, row-major in a ``MAXW`` stride
                 (``tile`` holds the kind, see ``delve_rules``; a gem or an
                 ability orb is picked up by the law, which clears the tile)

Movement, in the order the law steps it (``delve_world.Mirror`` is the same
arithmetic over a batch of hypothetical players -- that is how the generator
proves a door reachable with THIS jump):

    sideways       the shared damped motion law, then the leading edge tested
                   against solid tiles at three heights
    jump           ``fire`` (or Up off a ladder): on the ground, within
                   ``COYOTE`` of leaving it, from a ladder, or buffered for
                   ``BUFFER``; released early the rise is cut (variable height);
                   the DOUBLE JUMP (``ability``, an orb) once per flight
    ladders        in a ladder tile with a vertical stick the player climbs
                   (no gravity); jump leaves the ladder
    vertical       gravity, then landing on a solid top or a one-way top
                   (from above only), or bumping the head
    hazards        spikes and touching an enemy cost a heart (an invulnerable
                   second follows); landing on an enemy from above kills it
    leaving        the body's centre crossing the room's edge latches ``exit``
                   (1 left, 2 right, 3 up, 4 down) and freezes the bodies until
                   the game, which owns the room graph, changes the room.

Columns: lane slot rx ry pace home_x home_y (static); alive x y vx vy lives hurt
buf coy air pf climb exit ability score gems kills clock tile (state; ``lives``,
``exit``, ``ability``, ``score``, ``gems``, ``kills``, ``clock`` are the same
on every lane); room_w room_h (settings); move_axis steer_x fire move_axis_y
(controls).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from meowdoku import walkers
from meowdoku.delve_rules import (
    BUFFER, CLIMB, COYOTE, ENT, EMPTY, ENEMY_POINTS, FALL_MAX, FLY_BOB, FLY_RANGE, FLYER, FLYER_LANES, FLYER_RX,
    FLYER_RY, FOOT, GEM, GEM_POINTS, GRAVITY, HANG_SCALE, HANG_V, HURT_T, JUMP, JUMP2, JUMP_CUT, KNOCK, LADDER, LANES, MAP0, MAP_SLOT,
    MAXH, MAXW, ONEWAY, ORB, PLAYER_HP, PLAYER_RX, PLAYER_RY, PLAYER_SLOT, REACH, SOLID, SPEED, SPIKE,
    STOMP_BOUNCE, TAU, VIEW_H, WALKER, WALKER_LANES, WALKER_RX, WALKER_RY, WALKER_SPEED, FLYER_SPEED, camera_axis)
from meowdoku.entities import EntityGame, LawPiece, as_float
from meowdoku.entity_motion import damped_axis, vertical_axis
from src.common.tensors import AbstractTensor

where = AbstractTensor.where

CONTROLS = {"move_axis": "move-axis-x", "move_axis_y": "move-axis-y", "steer_x": "pointer-board-x", "fire": "primary-action"}
STATIC = ("lane", "slot", "rx", "ry", "pace", "home_x", "home_y")
STATE = ("alive", "x", "y", "vx", "vy", "lives", "hurt", "buf", "coy", "air", "pf", "climb", "exit", "ability",
         "score", "gems", "kills", "clock", "tile")
SETTINGS = ("room_w", "room_h")
ARGUMENTS = (*STATIC, *STATE, *SETTINGS, *CONTROLS, "dt")


def delve_step(*columns):
    """Advance every lane by ``dt``.  Every operand is a (LANES,) span (``dt`` a scalar)."""
    c = dict(zip(ARGUMENTS, columns))
    f = as_float
    n = ENT
    dt = (c["x"] * 0.0 + c["dt"])[0:n]

    def e(name):                     # the entity lanes of a column
        return c[name][0:n]

    slot, rx, ry, pace, home_x, home_y = e("slot"), e("rx"), e("ry"), e("pace"), e("home_x"), e("home_y")
    alive, x, y, vx, vy = e("alive"), e("x"), e("y"), e("vx"), e("vy")
    hurt, buf, coy, air, pf, climb = e("hurt"), e("buf"), e("coy"), e("air"), e("pf"), e("climb")
    lives, clock = e("lives"), e("clock")
    ability = e("ability")
    rw, rh = e("room_w"), e("room_h")
    axis, steer, fire, axy = e("move_axis"), e("steer_x"), e("fire"), e("move_axis_y")
    zero = x * 0.0
    is_p, is_w, is_f = slot == PLAYER_SLOT, slot == WALKER, slot == FLYER
    live = alive > 0.0
    playing = lives > 0.0
    me = is_p & live

    # ------------------------------------------------------------ the room
    tile = c["tile"]
    solid_t, oneway_t = f(tile == SOLID), f(tile == ONEWAY)
    ladder_t, spike_t = f(tile == LADDER), f(tile == SPIKE)

    def cell(px, py):
        cx = px.floor().clip(0.0, 1.0e6)
        cx = where(cx > rw - 1.0, rw - 1.0, cx)
        cy = py.floor().clip(0.0, 1.0e6)
        cy = where(cy > rh - 1.0, rh - 1.0, cy)
        return cy * float(MAXW) + cx + float(MAP0)

    def at(table, px, py):
        return walkers.gather(table, cell(px, py))

    exit_prev = (f(is_p) * e("exit")).sum()
    ready = live & playing & (exit_prev <= 0.0)

    # ------------------------------------------------------------ senses
    feet = y + ry
    sl, sr = x - rx * FOOT, x + rx * FOOT
    below = feet + 0.05
    row_b = below.floor()
    on_solid = (at(solid_t, sl, below) + at(solid_t, sr, below)) > 0.5
    on_plank = ((at(oneway_t, sl, below) + at(oneway_t, sr, below)) > 0.5) & (feet <= row_b + 0.06)
    grounded = (on_solid | on_plank) & (vy >= -0.05)
    in_ladder = at(ladder_t, x, y) > 0.5

    # ------------------------------------------------------------ the player's intent
    press = ((fire > 0.0) | ((axy < -0.5) & ~in_ladder)) & is_p
    press_edge = press & (pf <= 0.0)
    want_climb = is_p & in_ladder & ((axy.abs() > 0.2) | (climb > 0.5)) & (fire <= 0.0)
    stands = grounded | (climb > 0.5)
    dbl_jump = ready & is_p & press_edge & ~(stands | (coy > 0.0)) & (air > 0.5) & (ability > 0.5)
    climbing = want_climb & ready

    # ------------------------------------------------------------ vertical speed
    # entity_motion.vertical_axis: the one vertical law (buffered press, coyote time, launch, variable height,
    # apex hang, terminal velocity, exact integration); a double jump launches first, the ladder overrides.
    y_free, v_free, coy_f, buf_f, jumped = vertical_axis(
        y, where(dbl_jump, -JUMP2, vy), f(press), pf, f(stands), coy, buf, dt, GRAVITY, JUMP, JUMP_CUT, FALL_MAX,
        HANG_V, HANG_SCALE, COYOTE, BUFFER)
    do_jump = ready & is_p & (jumped > 0.5)
    climbing = climbing & ~do_jump
    air_n = where(stands, 1.0, where(dbl_jump, 0.0, air))
    coy_n = where(dbl_jump, 0.0, coy_f)
    fall = where(is_f, zero, where(climbing, axy.clip(-1.0, 1.0) * CLIMB, v_free))

    # ------------------------------------------------------------ sideways
    ship_x, ship_vx = damped_axis(x, vx, axis, steer, zero - 3.0, rw + 3.0, SPEED, TAU, REACH, dt)
    heading = where(vx >= 0.0, 1.0, -1.0)
    walk_v = heading * pace
    vxn = where(is_p, ship_vx, walk_v)
    nx = where(is_p, ship_x, x + walk_v * dt)
    lead = nx + where(vxn >= 0.0, rx, -rx)
    hit_x = ((at(solid_t, lead, y - ry + 0.08) + at(solid_t, lead, y) + at(solid_t, lead, y + ry - 0.08)) > 0.5)
    ahead_x = nx + where(vxn >= 0.0, rx + 0.15, -(rx + 0.15))
    ahead_y = y + ry + 0.15
    floor_ahead = (at(solid_t, ahead_x, ahead_y) + at(oneway_t, ahead_x, ahead_y)) > 0.5
    far = (nx - home_x).abs() > FLY_RANGE
    stop = where(is_p, hit_x, hit_x | (is_w & grounded & ~floor_ahead) | (is_f & far))
    x1 = where(ready & ~stop, nx, x)
    vx1 = where(ready, where(is_p, where(stop, 0.0, vxn), where(stop, -walk_v, walk_v)), where(is_p, zero, vx))

    # ------------------------------------------------------------ vertical: land or bump
    ny = where(is_f, y, where(climbing, y + fall * dt, y_free))
    bottom = ny + ry
    row_land = bottom.floor()
    s_hit = (at(solid_t, x1 - rx * FOOT, bottom) + at(solid_t, x1 + rx * FOOT, bottom)) > 0.5
    o_hit = (((at(oneway_t, x1 - rx * FOOT, bottom) + at(oneway_t, x1 + rx * FOOT, bottom)) > 0.5)
             & (feet <= row_land + 0.02) & ~climbing)
    land = ready & ~is_f & (fall > 0.0) & (s_hit | o_hit)
    top = ny - ry
    bump = ready & ~is_f & (fall < 0.0) & ((at(solid_t, x1 - rx * FOOT, top) + at(solid_t, x1 + rx * FOOT, top)) > 0.5)
    bob = home_y + FLY_BOB * (clock * 2.0 + c["lane"][0:n] * 1.3).sin()
    y_move = where(is_f, bob, where(land, row_land - ry - 1.0e-3, where(bump, top.floor() + 1.0 + ry + 1.0e-3, ny)))
    y1 = where(ready, y_move, y)
    vy1 = where(ready, where(land | bump | is_f, 0.0, fall), where(is_p, zero, vy))

    # ------------------------------------------------------------ hazards, enemies, stomps
    px = (f(me) * x1).sum()
    py = (f(me) * y1).sum()
    pfall = (f(me) * fall).sum()
    py_prev = (f(me) * y).sum()
    has_me = f(me).sum() > 0.0
    hurt_p = (f(me) * hurt).sum()
    foes = (is_w | is_f) & live & playing & has_me
    stomped = (foes & (pfall > 0.0) & (py_prev + PLAYER_RY <= (y1 - ry) + 0.25)
               & ((x1 - px).abs() < rx + PLAYER_RX) & ((py + PLAYER_RY) >= (y1 - ry)))
    touching = (foes & ~stomped & ((x1 - px).abs() < rx + 0.8 * PLAYER_RX) & ((y1 - py).abs() < ry + 0.8 * PLAYER_RY))
    spiked = me & ((at(spike_t, x1, y1 + ry - 0.12) + at(spike_t, x1 - rx * FOOT, y1 + ry - 0.12)
                    + at(spike_t, x1 + rx * FOOT, y1 + ry - 0.12) + at(spike_t, x1, y1)) > 0.5)
    stomp_n = f(stomped).sum()
    hurtful = (f(touching).sum() + f(spiked).sum()) > 0.0
    harmed = hurtful & (hurt_p <= 0.0) & (f(ready & is_p).sum() > 0.0)
    stomp_any = stomp_n > 0.0
    vy_player = where(harmed, -KNOCK, where(stomp_any, -STOMP_BOUNCE, vy1))
    vy_f = where(is_p, vy_player, vy1)
    hurt_f = where(is_p, where(harmed, HURT_T, (hurt - dt).clip(0.0, 9.0)), zero)
    alive_f = where(is_p, f(live), f(live & ~stomped))

    # ------------------------------------------------------------ the room's pickups (every tile lane)
    k = c["lane"] - float(MAP0)
    cell_y = ((k + 0.5) * (1.0 / MAXW)).floor()
    cell_x = k - cell_y * float(MAXW)
    nearby = (((cell_x + 0.5) - px).abs() < 0.5 + PLAYER_RX) & (((cell_y + 0.5) - py).abs() < 0.5 + PLAYER_RY)
    alive_p = has_me & (f(is_p & playing).sum() > 0.0)
    got_gem = (tile == GEM) & nearby & alive_p
    got_orb = (tile == ORB) & nearby & alive_p
    gems_d = f(got_gem).sum()
    tile_n = where(got_gem | got_orb, float(EMPTY), tile)
    ability_n = where(f(got_orb).sum() > 0.0, 1.0, c["ability"])

    # ------------------------------------------------------------ leaving the room
    out_code = where(x1 < 0.0, 1.0, where(x1 > rw, 2.0, where(y1 < 0.0, 3.0, where(y1 > rh, 4.0, 0.0))))
    left = (f(me & ready) * out_code).sum()
    exit_n = where((left > 0.0) & (exit_prev <= 0.0), left, c["exit"])

    # ------------------------------------------------------------ the next state
    lives_n = c["lives"] - where(harmed, 1.0, 0.0)
    score_n = c["score"] + GEM_POINTS * gems_d + ENEMY_POINTS * stomp_n
    gems_n = c["gems"] + gems_d
    kills_n = c["kills"] + stomp_n
    live_n = f(alive_f)
    vmax = where(live, fall.abs() + vxn.abs(), zero).max()
    dt_limit = (0.4 / (vmax + 1.0e-6)).clip(0.0, 0.5 * TAU)

    def wide(name, head):            # entity lanes' new values, then the room's lanes (unchanged)
        return walkers.widen(head, c[name][n:])

    full_x = c["x"]
    return (
        wide("alive", live_n), wide("x", x1), wide("y", y1), wide("vx", vx1), wide("vy", vy_f),
        lives_n, wide("hurt", hurt_f), wide("buf", where(is_p, buf_f, zero)), wide("coy", where(is_p, coy_n, zero)),
        wide("air", where(is_p, air_n, air)), wide("pf", where(is_p, f(press), zero)),
        wide("climb", where(is_p, f(climbing), zero)), exit_n, ability_n, score_n, gems_n, kills_n,
        c["clock"] + c["dt"], tile_n,
        f(live_n).sum() + full_x * 0.0,
        dt_limit + full_x * 0.0,
    )


class DelvePiece(LawPiece):
    """The law as a dt-system piece; compiled by ``LawPiece.compile``."""

    entry = "delve_step"
    law = staticmethod(delve_step)
    argument_names = ARGUMENTS
    output_names = (*(f"{name}_next" for name in STATE), "residual", "dt_limit")
    source_modules = ("meowdoku.walkers", "meowdoku.delve_rules")

    def __init__(self, batch: int = LANES):
        super().__init__(batch)


# ------------------------------------------------------------------ room -> columns

def blank_columns() -> dict[str, np.ndarray]:
    """Every column with the lanes' fixed rows (kinds, sizes, speeds) and no room loaded."""
    slot = np.full(LANES, MAP_SLOT)
    slot[0] = PLAYER_SLOT
    slot[slice(*WALKER_LANES)] = WALKER
    slot[slice(*FLYER_LANES)] = FLYER
    rx, ry, pace = np.zeros(LANES), np.zeros(LANES), np.zeros(LANES)
    rx[0], ry[0] = PLAYER_RX, PLAYER_RY
    rx[slice(*WALKER_LANES)], ry[slice(*WALKER_LANES)], pace[slice(*WALKER_LANES)] = WALKER_RX, WALKER_RY, WALKER_SPEED
    rx[slice(*FLYER_LANES)], ry[slice(*FLYER_LANES)], pace[slice(*FLYER_LANES)] = FLYER_RX, FLYER_RY, FLYER_SPEED
    zeros = {name: np.zeros(LANES) for name in (*STATE, "home_x", "home_y")}
    columns = {"lane": np.arange(LANES, dtype=np.float64), "slot": slot, "rx": rx, "ry": ry, "pace": pace, **zeros,
               "room_w": np.zeros(LANES), "room_h": np.zeros(LANES),
               "move_axis": np.zeros(LANES), "move_axis_y": np.zeros(LANES), "steer_x": np.full(LANES, -1.0),
               "fire": np.zeros(LANES)}
    columns["lives"][:] = PLAYER_HP
    columns["air"][:] = 1.0
    return columns


def place_room(columns: dict, grid: np.ndarray, walkers_at=(), flyers_at=(), player=None) -> None:
    """Write a room into ``columns`` (a dict of arrays, or the dt system's state
    object's attributes through ``columns[name]``): its tiles, bounds, enemies
    and (when given) the player's place ``(x, y, climbing)``."""
    h, w = grid.shape
    cells = np.full((MAXH, MAXW), float(SOLID))
    cells[:h, :w] = grid
    columns["tile"][MAP0:] = cells.reshape(-1)
    columns["room_w"][...] = float(w)
    columns["room_h"][...] = float(h)
    for lanes, spots, kind in ((WALKER_LANES, walkers_at, WALKER), (FLYER_LANES, flyers_at, FLYER)):
        for i, lane in enumerate(range(*lanes)):
            if i < len(spots):
                sx, sy = spots[i]
                columns["alive"][lane], columns["x"][lane], columns["y"][lane] = 1.0, sx, sy
                columns["home_x"][lane], columns["home_y"][lane] = sx, sy
                columns["vx"][lane] = columns["pace"][lane] * (1.0 if i % 2 == 0 else -1.0)
            else:
                columns["alive"][lane] = 0.0
            columns["vy"][lane] = 0.0
    if player is not None:
        px, py, climbing = player
        columns["x"][0], columns["y"][0], columns["vx"][0], columns["vy"][0] = px, py, 0.0, 0.0
        columns["alive"][0] = 1.0
        columns["climb"][0] = 1.0 if climbing else 0.0
        columns["buf"][0] = columns["coy"][0] = columns["pf"][0] = columns["hurt"][0] = 0.0
        columns["air"][0] = 1.0
        columns["exit"][...] = 0.0


# ------------------------------------------------------------------ the game

#: Mesh vocabulary beyond ``invaders.EMOJI`` (frog, brick, star), the side
#: scroller's (ground, snail) and Tile Quest's (door): name -> emoji, exported
#: by ``web_export``.
EMOJI = {"ladder": "\U0001FA9C", "cactus": "\U0001F335", "gem": "\U0001F48E", "orb": "\U0001F52E", "bat": "\U0001F987"}
EMOJI_SUBDIVISIONS: dict[str, int] = {}

#: tile kind -> (mesh, colour, half size); a solid with air above is the floor's face.
TILE_LOOK = {SOLID: ("brick", "wall", 0.5), ONEWAY: ("ground", "plank", 0.5), LADDER: ("ladder", "ladder", 0.5),
             SPIKE: ("cactus", "spike", 0.42), GEM: ("gem", "gem", 0.32), ORB: ("orb", "orb", 0.4)}


class Delve(EntityGame):
    game = "delve"
    title = "Delve (room dungeon)"
    height = VIEW_H
    CONTROLS = tuple(CONTROLS)
    PLAYER_LANE = 0
    HEALTH_MAX = PLAYER_HP
    #: Tiles stop the server's player where the page's tile-free prediction
    #: walks on; a gap beyond this keeps the server's word.
    ADOPT_BOUND = 0.25
    #: ``stage``: tiles and bodies all stand on one backdrop plane leaning back by this many radians (the page's
    #: placeEntity: the tiles are the scenery layer, the bodies a layer in front).
    SCENE = {"floor": True, "glass": False, "arcade": False, "stage": 0.2}
    HOSTILE_LANES = ((1, ENT),)
    #: Tiles beyond the view's edge still sent (the page predicts a little ahead of us).
    TILE_MARGIN = 4

    def __init__(self, board_id: str = "", aspect: float | None = None, seed: int | None = None,
                 size: str = "small", gated: bool = True):
        from meowdoku import delve_world
        self.world_seed = int(time.time_ns() % 100000) if seed is None else int(seed)
        self.quiet = False                               # a probe's switch: no enemies in any room
        self.size, self.gated = size, bool(gated)
        self.width = self.window(aspect)
        self.map_open = False
        self._build_world(delve_world)
        super().__init__(board_id)

    # -------------------------------------------------------------- the world
    @staticmethod
    def window(aspect: float | None) -> float:
        """The view window's width in tiles for a screen's shape (its height is ``VIEW_H``)."""
        if not aspect or aspect <= 0.0:
            return 16.0
        return float(np.clip(round(VIEW_H * aspect), 9, 18))

    def _build_world(self, delve_world) -> None:
        self.world = delve_world.generate(self.world_seed, size=self.size, gated=self.gated)
        self.room = self.world.start_room
        self.visited = {self.room}
        self.saved: dict[int, np.ndarray] = {}          # a room's tiles as the player left them (picked-up gems stay gone)
        self.serial = 0                                  # room changes so far (the page fades on it)
        self.hist = []                                   # the rooms walked through, in order
        self.arrival_door = -1                           # the door the player came in by (-1: the spawn)

    def start_columns(self) -> dict[str, np.ndarray]:
        columns = blank_columns()
        if self.gated is False:
            columns["ability"][:] = 1.0
        self.load_room(columns, self.room, self.world.rooms[self.room].spawn_arrival())
        return columns

    def initial_columns(self) -> dict[str, np.ndarray]:
        return self.start_columns()

    def make_piece(self):
        return DelvePiece()

    def load_room(self, columns, room_id: int, arrival: tuple[float, float, bool]) -> None:
        room = self.world.rooms[room_id]
        grid = self.saved.get(room_id, room.grid)
        place_room(columns, grid, [] if self.quiet else room.walkers, [] if self.quiet else room.flyers, arrival)

    def writable(self) -> dict[str, np.ndarray]:
        """The columns a room change writes: the dt system's live state, or round zero's."""
        s = self.session
        names = ("tile", "room_w", "room_h", "alive", "x", "y", "vx", "vy", "home_x", "home_y", "pace", "climb", "buf",
                 "coy", "pf", "hurt", "air", "exit")
        return {name: (getattr(s.state, name) if s.state is not None else s.columns[name]) for name in names}

    # -------------------------------------------------------------- room transitions
    def _land(self, seconds: float, held: dict) -> None:
        super()._land(seconds, held)
        self.check_exit()

    def check_exit(self) -> None:
        s = self.session
        if s.state is None:
            return
        code = int(round(float(s.column("exit")[0])))
        if code > 0:
            self.transition(code)

    def transition(self, code: int) -> None:
        """The player left the room through side ``code`` (1 left, 2 right, 3 up, 4 down):
        keep the room as left, load the one the door opens on, place the player at the
        matching door."""
        s = self.session
        room = self.world.rooms[self.room]
        px, py = float(s.column("x")[0]), float(s.column("y")[0])
        side = "LRUD"[code - 1]
        door = room.door_for(side, px, py)
        if door is None:                                         # no door there: a wall holds the player in
            s.state.exit[...] = 0.0
            return
        other_id, other_door = self.world.link(self.room, door)
        self.saved[self.room] = np.asarray(s.column("tile")[MAP0:]).reshape(MAXH, MAXW)[:room.h, :room.w].copy()
        keep = {name: float(s.column(name)[0]) for name in ("hurt",)}
        self.room = other_id
        self.visited.add(other_id)
        self.serial += 1
        self.hist.append(other_id)
        self.arrival_door = other_door
        arrival = self.world.rooms[other_id].arrival_at(other_door)
        self.load_room(self.writable(), other_id, arrival)
        s.state.hurt[0] = keep["hurt"]
        self.epoch += 1

    # -------------------------------------------------------------- the page's view
    def motion(self) -> dict[str, Any]:
        room = self.world.rooms[self.room]
        return {"speed": SPEED, "tau": TAU, "reach": REACH, "x": [-3.0, room.w + 3.0],
                "vertical": {"gravity": GRAVITY, "jump": JUMP, "cut": JUMP_CUT, "fall_max": FALL_MAX, "hang_v": HANG_V,
                             "hang_scale": HANG_SCALE, "coyote": COYOTE, "buffer": BUFFER, "press": ["fire"],
                             # contact (the page's prediction): the tile meshes landed on (solid: struck from below too;
                             # oneway: a plank, landed on from above only), the body's half height and its feet's reach
                             "ry": PLAYER_RY, "foot": PLAYER_RX * FOOT, "solid": [TILE_LOOK[SOLID][0]],
                             "oneway": [TILE_LOOK[ONEWAY][0]], "bump": 0.0}}

    def layout(self, aspect: float) -> None:
        width = self.window(aspect)
        if width != self.width:
            self.width = width
            self.epoch += 1

    def reset(self) -> None:
        from meowdoku import delve_world
        self.world_seed = (self.world_seed + 1) % 100000          # a new world each run
        self._build_world(delve_world)
        super().reset()

    def palette(self) -> dict[str, list[float]]:
        return {"floor": [0.07, 0.08, 0.13], "wall": [0.46, 0.44, 0.56], "plank": [0.70, 0.50, 0.28],
                "ladder": [0.86, 0.72, 0.40], "spike": [0.30, 0.78, 0.38], "gem": [0.35, 0.85, 0.95],
                "orb": [0.78, 0.45, 0.98], "frog": [0.45, 0.9, 0.35], "snail": [0.92, 0.62, 0.35],
                "bat": [0.62, 0.40, 0.78], "top": [0.40, 0.58, 0.36]}

    def controls_ui(self) -> list[dict[str, Any]]:
        return [
            {"id": "map", "kind": "button", "label": "Map (M)", "action": "delve_map"},
            {"id": "new", "kind": "button", "label": "New world", "action": "delve_new"},
            {"id": "size", "kind": "select", "label": "world", "value": self.size, "action": "delve_size",
             "field": "size", "options": [["small", "small"], ["large", "large"]]},
            {"id": "gated", "kind": "select", "label": "double jump", "value": "gated" if self.gated else "unlocked",
             "action": "delve_gated", "field": "gated", "options": [["gated", "find the orb"], ["unlocked", "have it"]]},
        ]

    def do(self, payload: dict) -> None:
        """The game's own actions (``delve_*``)."""
        kind = str(payload.get("type"))
        if kind == "delve_map":
            self.map_open = not self.map_open
        elif kind == "delve_new":
            self.reset()
        elif kind == "delve_size":
            self.size = "large" if str(payload.get("size")) == "large" else "small"
            self.reset()
        elif kind == "delve_gated":
            self.gated = str(payload.get("gated")) != "unlocked"
            self.reset()
        else:
            raise ValueError(f"unknown delve action {kind!r}")

    # -------------------------------------------------------------- what the page draws
    def view_origin(self) -> tuple[float, float]:
        s, room = self.session, self.world.rooms[self.room]
        if s.state is None:
            return 0.0, 0.0
        return (camera_axis(float(s.column("x")[0]), self.width, float(room.w)),
                camera_axis(float(s.column("y")[0]), self.height, float(room.h)))

    def tiles(self) -> list[dict[str, Any]]:
        """The tiles near the view as flat entities (ids past the law's lanes, per room)."""
        s, room = self.session, self.world.rooms[self.room]
        grid = np.asarray(s.column("tile")[MAP0:]).reshape(MAXH, MAXW)[:room.h, :room.w]
        vx0, vy0 = self.view_origin()
        m = self.TILE_MARGIN
        x0, x1 = max(0, int(np.floor(vx0)) - m), min(room.w, int(np.ceil(vx0 + self.width)) + m)
        y0, y1 = max(0, int(np.floor(vy0)) - m), min(room.h, int(np.ceil(vy0 + self.height)) + m)
        out = []
        for cy in range(y0, y1):
            for cx in range(x0, x1):
                kind = int(grid[cy, cx])
                if kind == EMPTY:
                    continue
                mesh, color, size = TILE_LOOK[kind]
                if kind == SOLID and cy > 0 and int(grid[cy - 1, cx]) in (EMPTY, LADDER, SPIKE, GEM, ORB, ONEWAY):
                    color = "top"                                   # a surface you can stand on
                out.append({"id": f"{10000 + cy * MAXW + cx}:{self.room}", "x": cx + 0.5, "y": cy + 0.5, "vx": 0.0, "vy": 0.0,
                            "mesh": mesh, "color": color, "size": size, "roll": 0.0, "lean": 0.0, "owner": -1, "scenery": True,
                            "age": 99.0, "spin": 1.2 if kind in (GEM, ORB) else 0.0, "bob": 0.0})
        return out

    def entities(self) -> list[dict[str, Any]]:
        names = ("slot", "alive", "x", "y", "vx", "vy", "hurt")
        col = {name: self.session.column(name)[:ENT] for name in names}
        clock = float(self.session.column("clock")[0])
        out = self.tiles()
        for k in np.flatnonzero(col["alive"] > 0.0):
            slot = col["slot"][k]
            mesh, size = (("frog", PLAYER_RY) if slot == PLAYER_SLOT else ("snail", WALKER_RY) if slot == WALKER else ("bat", 0.36))
            blink = k == 0 and col["hurt"][0] > 0.0 and int(clock * 12) % 2 == 0
            out.append({"id": f"{k}:{self.room}", "x": float(col["x"][k]), "y": float(col["y"][k]),
                        "vx": float(col["vx"][k]), "vy": float(col["vy"][k]) if k != 0 else 0.0,
                        "mesh": mesh, "color": mesh, "size": size * (0.6 if blink else 1.0), "roll": 0.0,
                        "lean": 0.25 if k == 0 else 0.45, "owner": -1, "age": 99.0, "spin": 0.0,
                        "bob": 1.0 if slot == FLYER else 0.0})
        return out

    def state(self) -> dict[str, Any]:
        st = super().state()
        room = self.world.rooms[self.room]
        st["roomview"] = {"w": room.w, "h": room.h, "serial": self.serial, "id": self.room}
        st["room"] = {"id": self.room, "name": room.name, "kind": room.kind, "w": room.w, "h": room.h,
                      "gx": room.gx, "gy": room.gy, "cw": room.cw, "ch": room.ch}
        st["world"] = self.world.overview(self.visited, self.room, self.map_open)
        st["controls_ui"] = self.controls_ui()
        return st

    def result(self) -> dict[str, Any] | None:
        s = self.session
        if s.state is None or s.column("lives")[0] > 0.0:
            return None
        return {"score": float(s.column("score")[0]), "detail": f"{len(self.visited)} rooms, {int(s.column('gems')[0])} gems"}

    def describe(self) -> str:
        s = self.session
        seconds, calls = s.last_round
        room = self.world.rooms[self.room]
        head = "GAME OVER · R for a new world" if s.column("lives")[0] <= 0 else f"hp {s.column('lives')[0]:.0f}"
        ability = "double jump" if s.column("ability")[0] > 0.5 else "no ability yet"
        return (f"{head} · {room.name} · rooms {len(self.visited)}/{len(self.world.rooms)} · gems {int(s.column('gems')[0])} · "
                f"{ability} · tick {seconds * 1000:.0f} ms = {calls} law steps")

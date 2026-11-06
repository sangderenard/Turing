"""Delve's rules: the constants that BOTH the law (``delve.py``) and the world
generator's traversal (``delve_world.py``) read.

One source: the jump the generator proves a door reachable with is the jump
the law integrates, because both import it from here.  Tiles are one unit; a
room is a rectangle of tiles (``y`` down, as everywhere in the engine).
"""

from __future__ import annotations

# ---------------------------------------------------------------- tile vocabulary
EMPTY, SOLID, ONEWAY, LADDER, SPIKE, GEM, ORB = range(7)
TILE_NAMES = {EMPTY: "empty", SOLID: "solid", ONEWAY: "oneway", LADDER: "ladder", SPIKE: "spike", GEM: "gem", ORB: "orb"}

# ---------------------------------------------------------------- the world's units
SCREEN_W, SCREEN_H = 16, 10           # one screen of tiles; a room is 1..3 screens either way
MAX_SCREENS = 3
MAXW, MAXH = SCREEN_W * MAX_SCREENS, SCREEN_H * MAX_SCREENS      # the room grid every law call is shaped by
DOOR_H = 3                            # rows of a side door's opening
DOOR_W = 2                            # columns of a floor/ceiling opening (the first holds a ladder)

# ---------------------------------------------------------------- movement (the law's, the planner's)
GRAVITY = 30.0
JUMP = 12.0                           # apex JUMP**2 / (2 GRAVITY) = 2.4 above the feet
JUMP2 = 11.0                          # the double jump (an ability)
JUMP_CUT = 3.5                        # rising faster than this with the button released: cut to it
FALL_MAX = 18.0
SPEED = 6.0
TAU = 0.06
REACH = 0.2
CLIMB = 3.5
COYOTE = 0.09
BUFFER = 0.11
PLAYER_RX, PLAYER_RY = 0.3, 0.4
FOOT = 0.9                            # the feet sense at x +/- FOOT * rx
STOMP_BOUNCE = 8.0
KNOCK = 9.0
HURT_T = 1.0
PLAYER_HP = 3.0

# ---------------------------------------------------------------- enemies and pickups
WALKER, FLYER = 1.0, 2.0
PLAYER_SLOT, MAP_SLOT = 0.0, 3.0
N_WALKERS, N_FLYERS = 4, 2
WALKER_RX, WALKER_RY = 0.35, 0.35
FLYER_RX, FLYER_RY = 0.3, 0.3
WALKER_SPEED, FLYER_SPEED = 1.6, 2.2
FLY_RANGE, FLY_BOB = 3.0, 0.5
GEM_POINTS, ENEMY_POINTS = 100.0, 200.0

# ---------------------------------------------------------------- lanes
ENT = 1 + N_WALKERS + N_FLYERS        # the player, then the walkers, then the flyers
MAP0 = ENT
MAPN = MAXW * MAXH
LANES = ENT + MAPN
WALKER_LANES = (1, 1 + N_WALKERS)
FLYER_LANES = (1 + N_WALKERS, ENT)

# ---------------------------------------------------------------- the camera (page and server share the formula)
VIEW_H = 10.0


def camera_axis(pos: float, window: float, room: float) -> float:
    """The view's origin along one axis: centred on ``pos``, clamped to the
    room; a room smaller than the window is centred in it."""
    if room <= window:
        return (room - window) / 2.0
    return min(max(pos - window / 2.0, 0.0), room - window)

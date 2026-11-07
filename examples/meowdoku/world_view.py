"""The two render-side dials of a real-time scrolling game: its CLOCK SPEED and its WORLD ZOOM.

Neither is law.  The law's step (``dt``) and its text are the same on the server and the page at any value of either.

CLOCK SPEED scales how much law time a wall second lands: ``EntityGame`` lands ``speed * wall`` seconds of play per
tick (the dt controller cuts them into the same fixed steps it always chose), and the page lands ``speed * frame``
seconds in its own prediction.  The server's wall clock for the timing exchange is the *virtual* wall
(``EntityGame.virtual``): the wall integrated by the speed in force, so the law clock and ``law_wall`` still advance
together (``web/pilot_clock.js`` reads them unchanged, in virtual time) and a handoff at 0.1x is as smooth as at 1x.

WORLD ZOOM is the page's scene camera (``web/world_zoom.js``).  Zooming out packs more world into the same screen, so
the server's VIEW WINDOW (which bodies it sends) widens with it, up to what the game can supply: a game declares the
``reach`` of world it has beyond its field edges, and the zoom floor is the zoom at which the window fills it.  The
observations the autopilot reads never see the zoom.
"""

from __future__ import annotations

from typing import Any

#: Game clock speed, a multiple of real time.  The page's slider is logarithmic over this range.
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.05, 2.0, 0.4
#: World zoom (1 = the field fills the screen; below 1 shows more world).
ZOOM_MAX, ZOOM_DEFAULT = 2.5, 1.0
#: Games with a spherical patch send bodies out to at most this arc distance from the player (the streaming budget),
#: whatever the zoom; the autopilot's own observation radius (``scroller.PATCH_R``) is a different, fixed number.
SEND_R_MAX = 22.0

NO_REACH = {"left": 0.0, "right": 0.0, "up": 0.0, "down": 0.0}


def clamp_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return SPEED_DEFAULT
    if speed != speed:
        return SPEED_DEFAULT
    return min(SPEED_MAX, max(SPEED_MIN, speed))


def zoom_floor(width: float, height: float, reach: dict[str, float] | None) -> float:
    """The least zoom at which the view window still lies inside the world the game supplies."""
    r = {**NO_REACH, **(reach or {})}
    return min(1.0, max(width / (width + max(0.0, r["left"]) + max(0.0, r["right"])),
                        height / (height + max(0.0, r["up"]) + max(0.0, r["down"]))))


def clamp_zoom(value: Any, width: float, height: float, reach: dict[str, float] | None) -> float:
    try:
        zoom = float(value)
    except (TypeError, ValueError):
        return ZOOM_DEFAULT
    if zoom != zoom:
        return ZOOM_DEFAULT
    return min(ZOOM_MAX, max(zoom_floor(width, height, reach), zoom))


def view_window(width: float, height: float, zoom: float, reach: dict[str, float] | None, limit: bool = True) -> dict[str, float]:
    """The world rectangle (screen frame: x right from the field's left edge, y down from its top) that a zoom shows.

    The field is centred on the screen; at ``zoom`` the window is ``1/zoom`` as large about the same centre, then
    slid to stay inside the world the game supplies (``reach`` beyond each edge) -- a side scroller has more world
    ahead than behind, so zooming out pans the window ahead.  Zooming in never leaves the field."""
    r = {**NO_REACH, **(reach or {})}
    zoom = clamp_zoom(zoom, width, height, reach) if limit else float(zoom)
    w, h = width / zoom, height / zoom
    x0, y0 = (width - w) / 2.0, (height - h) / 2.0
    lo_x, hi_x = -max(0.0, r["left"]), width + max(0.0, r["right"])
    lo_y, hi_y = -max(0.0, r["up"]), height + max(0.0, r["down"])
    if zoom < 1.0:                       # (only a widened window is slid; a narrowed one stays centred)
        x0 = min(max(x0, lo_x), hi_x - w)
        y0 = min(max(y0, lo_y), hi_y - h)
    return {"x0": x0, "x1": x0 + w, "y0": y0, "y1": y0 + h, "zoom": zoom}


def zoom_state(width: float, height: float, zoom: float, reach: dict[str, float] | None) -> dict[str, Any]:
    """What the server publishes of the zoom: its limits, value, reach and the window it sends."""
    zoom = clamp_zoom(zoom, width, height, reach)
    return {"min": zoom_floor(width, height, reach), "max": ZOOM_MAX, "value": zoom,
            "reach": {**NO_REACH, **(reach or {})}, "window": view_window(width, height, zoom, reach)}


def speed_state(speed: float, gen: int, adjustable: bool = True) -> dict[str, Any]:
    return {"value": speed, "min": SPEED_MIN, "max": SPEED_MAX, "default": SPEED_DEFAULT, "gen": gen, "adjustable": adjustable}

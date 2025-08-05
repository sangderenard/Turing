"""Utility mathematics for reel-to-reel and cassette demo."""

from __future__ import annotations

import math
from typing import Tuple


def tape_radius(spool_radius: float, max_radius: float, tape_on_spool: float, total_tape: float) -> float:
    """Linearly interpolate tape radius based on remaining tape length.

    Parameters
    ----------
    spool_radius:
        The radius of the bare spool (no tape).
    max_radius:
        Radius when the spool is fully wound.
    tape_on_spool:
        Current tape length on the spool.
    total_tape:
        Total tape length across both spools.
    """
    if total_tape <= 0:
        raise ValueError("total_tape must be positive")
    ratio = max(0.0, min(1.0, tape_on_spool / total_tape))
    return spool_radius + ratio * (max_radius - spool_radius)


def tangent_points(cx: float, cy: float, r: float, px: float, py: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return tangent points on circle from an external point.

    The two returned points lie on the circle of radius ``r`` centered at
    ``(cx, cy)`` and form tangents that pass through ``(px, py)``.
    """
    dx, dy = px - cx, py - cy
    dist = math.hypot(dx, dy)
    if dist <= r:
        # Degenerate; return point directly to maintain continuity.
        angle = math.atan2(dy, dx)
        point = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        return point, point

    angle_to_p = math.atan2(dy, dx)
    offset = math.acos(r / dist)
    t1 = (cx + r * math.cos(angle_to_p + offset), cy + r * math.sin(angle_to_p + offset))
    t2 = (cx + r * math.cos(angle_to_p - offset), cy + r * math.sin(angle_to_p - offset))
    return t1, t2


"""The oil & glycerine toy, ONE description for the 2-D and the 3-D fountain.

``toy_parts(w, c)`` returns the toy as pure geometry in the plane (x across, y up from the floor y = 0): the hourglass vessel, the three ramps,
the filtering peg field, the collection pool under it, the small free wheel, and the designed height.  ``fountain.sdf_preset`` turns it into
2-D slots; ``fountain3d.sdf_preset`` into 3-D slots, the same parts EXTRUDED through the depth (the capsule ramps and the peg cylinders span it, the
hourglass is the planar profile, the wheel's hinge axis is z).  Nothing here knows the law: it is host-side data, so an edit to it never changes a
law digest (see the ``# <toy>`` regions of ``fountain.py`` / ``fountain_build.py`` / ``fountain3d.py`` / ``fountain3d_build.py``).

Bottom up, as the toy is built (every height is what the part needs, not a fraction of a window):

* the POOL: an open basin ``POOL_W`` half-widths deep between the floor and the lowest peg, where what passes the pegs collects;
* the PEG FIELD: ``PEG_ROWS`` staggered rows of pegs of radius ``PEG_R`` cells.  The pegs are the filter: the clear gap between two pegs of a row
  is ``FILTER_FRACTION`` of the largest grain's diameter (``c``) -- between the species' diameters, so the larger species cannot pass a row and the
  smaller ones can.  The pitch is a whole number of times across the vessel and the outermost pegs sit ON the walls, so no wider path goes around;
* 3 cells of clearance, then the FUNNEL (slope ``SLOPE``) up to the throat, and the upper funnel;
* the WHEEL: a small free-spinning paddle wheel just above the throat's funnel, under the low end of the last ramp, so the stream turns it while the
  hourglass empties underneath;
* the RAMPS (three capsules, zig-zag: left-down, right-down, left-down), stacked so each pours onto the next; their high ends stand ``OPEN_CELLS``
  cells off the case walls -- a gap liquid passes through, which is what lets a turned-over toy pour along the same cascade upside down.  The last ramp
  is SHORT (it ends over the wheel);
* the inlet ``0.12`` above the highest ramp end, the ceiling ``0.12`` above that.
"""
from __future__ import annotations

import math

#: the toy's tunables (cells are the grain cell ``c``: twice the largest grain radius)
PEG_FILTER = False              # True: the pegs are a size filter (gap between the species' diameters); False: the earlier plinko field
PEG_ROWS = 6                    # staggered rows of pegs (the filter field has PEG_ROWS_FILTER)
PEG_ROWS_FILTER = 8
PEG_R = 0.9                     # peg radius, cells
POOL_W = 0.35                   # the collection pool's depth, half-widths (at least ``POOL_MIN_CELLS``)
POOL_MIN_CELLS = 8.0
SLOPE = 1.7                     # the funnel's slope (dx / dy)
RAMP_R = 0.9                    # ramp capsule radius, cells
OPEN_CELLS = 3.5                # clear gap between a ramp's high end and the case wall, cells
TAN = 0.42                      # the ramps' slope (dy / dx)
HOP = 0.12                      # a ramp's low end stands this many half-widths (at least 6 cells) over the ramp it pours onto
DRIFT = 0.20                    # how far (half-widths) a stream leaving a ramp's low end travels sideways before it lands
LOW_X = (-0.10, 0.30, 0.30)     # the ramps' low ends, half-widths from the centre (ramp 2, the last, is shortened: it ends over the wheel)
WHEEL_REACH = 0.15              # the wheel's reach, half-widths (at least ``WHEEL_MIN_CELLS``)
WHEEL_MIN_CELLS = 3.0
WHEEL_PADDLES = 8
WHEEL_CLEAR = 4.0               # cells between the wheel's tip and the ramp over it / the funnel under it
#: the species the filter sits between: it passes the smaller half and holds the larger half of ``radii`` (sorted): the gap is the geometric-mean-ish middle
#: of the two middle species' diameters, as a fraction of the largest diameter
def filter_fraction(radii) -> float:
    r = sorted(float(v) for v in radii)
    return math.sqrt(r[1] * r[2]) / r[-1]


def toy_parts(w: float, c: float, radii=(0.0090, 0.0105, 0.0120, 0.0140)) -> dict:
    """The toy for half-width ``w`` and grain cell ``c`` (see the module doc).  Lengths in scene units."""
    rr = RAMP_R * c
    rp = PEG_R * c
    throat = max(0.09 * w, 3.5 * c)
    delta = (w - throat) / SLOPE
    # ---- the pegs: a clear gap g between pegs of a row, a whole number K of pitches across the half width (the outer pegs are centred ON the walls)
    gap = filter_fraction(radii) * c
    if PEG_FILTER:
        k = max(1, round(w / (gap + 2.0 * rp)))
        px = w / k
        rp = 0.5 * (px - gap)                                                     # the peg radius takes up the rounding: the gap is exactly the design's
        rows = PEG_ROWS_FILTER
    else:
        px = 4.6 * c
        k = int(w // px)
        rows = PEG_ROWS
    py = px
    pool = max(POOL_W * w, POOL_MIN_CELLS * c)
    y_lo = pool + rp
    pegs_top = y_lo + (rows - 1) * py + rp
    cols = 2 * k + 1
    pegs = {"px": px, "py": py, "r": rp, "cols": cols, "rows": rows, "cy": y_lo + 0.5 * (rows - 1) * py, "cx": 0.0,
            "gap": px - 2.0 * rp, "y_lo": y_lo, "top": pegs_top}
    yc = pegs_top + 3.0 * c + delta                                               # the throat
    # ---- the wheel and the last ramp over it
    reach = max(WHEEL_REACH * w, WHEEL_MIN_CELLS * c)
    edge = w - OPEN_CELLS * c - rr                                                # the ramps' high ends: |x| of the capsule's end point
    lx = [v * w for v in LOW_X]
    hub_x = lx[2] - DRIFT * w - 0.6 * reach                                       # the stream lands 0.6 reach right of the hinge
    funnel_at = lambda x: yc + max(0.0, (abs(x) - throat) / SLOPE)               # the funnel wall's height at |x|
    hub_y = max(funnel_at(hub_x - reach), funnel_at(hub_x + reach)) + reach + WHEEL_CLEAR * c
    # ---- the ramps, bottom up.  ramp k: high end at the side ``side[k]`` (+1: the right wall), low end ``lx[k]``
    side = (1.0, -1.0, 1.0)
    low_y = [0.0, 0.0, hub_y + reach + WHEEL_CLEAR * c + rr]
    hop = max(HOP * w, 6.0 * c)

    def surface(k: int, x: float) -> float:
        """The top surface of ramp ``k`` at ``x`` (the capsule axis; the capsule is ``rr`` thick)."""
        return low_y[k] + TAN * side[k] * (x - lx[k])
    for k in (1, 0):
        land = lx[k] - side[k] * DRIFT * w                                          # ramp k runs toward -side: its stream leaves, and lands, that way
        low_y[k] = surface(k + 1, land) + hop
    ramps = []
    for k in range(3):
        high = (side[k] * edge, low_y[k] + TAN * abs(side[k] * edge - lx[k]))
        ramps.append({"high": high, "low": (lx[k], low_y[k]), "r": rr})
    y0 = ramps[0]["high"][1]
    nozzle_y = y0 + 0.12
    top = nozzle_y + 0.12
    return {"w": w, "c": c, "top": top, "nozzle_y": nozzle_y, "throat": throat, "slope": SLOPE, "yc": yc, "delta": delta,
            "pegs": pegs, "pool": pool, "ramps": ramps, "edge": edge, "open": w - edge - rr,
            "wheel": {"x": hub_x, "y": hub_y, "reach": reach, "paddles": WHEEL_PADDLES}}

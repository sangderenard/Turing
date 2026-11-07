"""Swept (continuous) contact tests for the entity laws: ``block_hits`` over the step."""
from src.common.tensors import AbstractTensor
from meowdoku.entities import as_float


def swept_block_hits(start, end, rx, ry, live, rows: tuple[int, int], cols: tuple[int, int], rz):
    """``block_hits`` over the STEP, not at its end: the hit counts of the lanes
    in ``rows`` by the lanes in ``cols`` if their boxes overlapped at any
    instant of the step (zero outside ``rows``).

    ``start`` and ``end`` are ``(x, y, z)`` spans: where each lane was when the
    step began (after any spawn) and where it ended, in the SAME frame (the
    end-of-step patch), the lane moving in a straight line between them.
    Relative to a ``cols`` lane, a ``rows`` lane then moves on the segment
    ``d0 + t * dv`` for t in [0, 1], and per axis the boxes overlap while
    |d0 + t dv| < r: an interval of t, ends ``(+-r - d0) / dv``.  The boxes
    overlap at some instant when the three intervals and [0, 1] share a point:
    the exact swept test of the box the law already uses (its closest
    approach, axis by axis).  A step no longer needs to be short enough that
    nothing crosses what it can hit: a pair that crossed within it still
    touches.  A pair that did not move is the old static test, exactly.
    Branch-free; the same lane-by-lane block as ``block_hits``.
    """
    where = AbstractTensor.where
    r0, r1 = rows
    c0, c1 = cols
    n = int(rx.shape[0])
    a, b = r1 - r0, c1 - c0
    lo_all = None
    hi_all = None
    for s, e, r in zip(start, end, (rx, ry, rz)):
        d0 = s[r0:r1].reshape((a, 1)) - s[c0:c1].reshape((1, b))
        dv = (e[r0:r1].reshape((a, 1)) - s[r0:r1].reshape((a, 1))) - (e[c0:c1].reshape((1, b)) - s[c0:c1].reshape((1, b)))
        reach = r[r0:r1].reshape((a, 1)) + r[c0:c1].reshape((1, b))
        # a pair that does not close on this axis has the whole step or none of
        # it: a tiny dv sends the interval ends far outside [0, 1] either way
        safe = where(dv.abs() < 1e-9, 1e-9, dv)
        t_a, t_b = (reach * -1.0 - d0) / safe, (reach - d0) / safe
        enter, leave = where(t_a < t_b, t_a, t_b), where(t_a < t_b, t_b, t_a)
        lo_all = enter if lo_all is None else where(lo_all > enter, lo_all, enter)
        hi_all = leave if hi_all is None else where(hi_all < leave, hi_all, leave)
    touch = ((lo_all.clip(0.0, 1.0e9) <= hi_all.clip(-1.0e9, 1.0))
             & live[r0:r1].reshape((a, 1)) & live[c0:c1].reshape((1, b)))
    counts = as_float(touch).sum(dim=1)
    parts = [rx[:r0] * 0.0] if r0 > 0 else []
    parts.append(counts)
    if r1 < n:
        parts.append(rx[r1:] * 0.0)
    return AbstractTensor.cat(parts, dim=0) if len(parts) > 1 else counts

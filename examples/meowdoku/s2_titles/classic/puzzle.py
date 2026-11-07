"""Meowdoku puzzle setup: a unique cat arrangement and the regions around it.

This is setup, not a step law.  It runs once in Python and produces the
static columns the game law reads every round: ``row``, ``col``, ``region``
and the hidden ``solution`` mask, one cell per lane in row-major order
(``cell = row * n + col``).

Rules the arrangement satisfies (the published game's three rules):
one cat per row, one per column, one per region, and no two cats touch,
diagonals included.  Regions are grown outward from the cats, and a board is
accepted only when a backtracking count finds exactly one arrangement that
satisfies all three rules -- the game promises a single solution.  Above
n=8 that random search is hopeless, so ``generate_batch`` builds the
uniqueness in, batched over puzzles as AbstractTensor spans (see its
docstring); the count remains the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Puzzle:
    n: int
    row: np.ndarray        # (n*n,) float64, row index of each cell
    col: np.ndarray        # (n*n,) float64, column index of each cell
    region: np.ndarray     # (n*n,) float64, region id of each cell
    solution: np.ndarray   # (n*n,) float64, 1.0 where a cat belongs

    @property
    def cells(self) -> int:
        return self.n * self.n

    def columns(self) -> dict[str, np.ndarray]:
        """The static state columns, each a contiguous (n*n,) float64 span."""
        return {
            "row": self.row.copy(), "col": self.col.copy(),
            "region": self.region.copy(), "solution": self.solution.copy(),
        }

    def region_grid(self) -> np.ndarray:
        return self.region.reshape(self.n, self.n).astype(int)

    def solution_columns(self) -> tuple[int, ...]:
        """Solution as the cat's column per row."""
        grid = self.solution.reshape(self.n, self.n)
        return tuple(int(np.argmax(grid[r])) for r in range(self.n))


def _touching(a: int, b: int) -> bool:
    return abs(a - b) <= 1


def cat_arrangement(n: int, rng: random.Random) -> tuple[int, ...]:
    """A column per row: a permutation where consecutive rows never touch.

    Rows two or more apart cannot touch, so consecutive-row separation is the
    whole no-touch rule for a permutation.
    """
    while True:
        columns = list(range(n))
        rng.shuffle(columns)
        if all(not _touching(columns[i], columns[i + 1]) for i in range(n - 1)):
            return tuple(columns)


def grow_regions(n: int, cats: tuple[int, ...], rng: random.Random) -> np.ndarray:
    """Grow ``n`` regions outward from the cats until every cell is owned."""
    grid = -np.ones((n, n), dtype=int)
    for r, c in enumerate(cats):
        grid[r, c] = r
    unassigned = n * n - n
    while unassigned:
        owned = [(r, c) for r in range(n) for c in range(n) if grid[r, c] >= 0]
        r, c = owned[rng.randrange(len(owned))]
        free = [
            (rr, cc) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= rr < n and 0 <= cc < n and grid[rr, cc] < 0
        ]
        if not free:
            continue
        rr, cc = free[rng.randrange(len(free))]
        grid[rr, cc] = grid[r, c]
        unassigned -= 1
    return grid


def count_arrangements(region_grid: np.ndarray, limit: int = 2) -> int:
    """Count arrangements satisfying all three rules, stopping at ``limit``."""
    n = region_grid.shape[0]
    found = 0

    def place(r: int, used_cols: set[int], used_regions: set[int], prev: int) -> None:
        nonlocal found
        if found >= limit:
            return
        if r == n:
            found += 1
            return
        for c in range(n):
            if c in used_cols or (prev >= 0 and _touching(prev, c)):
                continue
            region = int(region_grid[r, c])
            if region in used_regions:
                continue
            place(r + 1, used_cols | {c}, used_regions | {region}, c)

    place(0, set(), set(), -1)
    return found


@dataclass(frozen=True)
class PuzzleBatch:
    """``batch`` puzzles of one size, every column a ``(batch, n*n)`` span."""

    n: int
    row: np.ndarray
    col: np.ndarray
    region: np.ndarray
    solution: np.ndarray

    @property
    def batch(self) -> int:
        return int(self.solution.shape[0])

    def puzzle(self, index: int) -> Puzzle:
        return Puzzle(self.n, self.row[index].copy(), self.col[index].copy(),
                      self.region[index].copy(), self.solution[index].copy())


class _Draws:
    """Uniform draws from one seeded repository PRNG stream (Xoroshiro128**),
    taken from a buffer refilled in blocks, so a draw of ``count`` values
    costs ``count`` values of the stream, no more, and the sequence is a
    function of the seed alone."""

    def __init__(self, seed: int, block: int = 1 << 14):
        from src.common.tensors.abstraction_methods.random import PRNG_ALGO, RANDOM_KIND, random_generator
        self._stream = random_generator(kind=RANDOM_KIND.PRNG, algo=PRNG_ALGO.XOROSHIRO128SS,
                                        seed=int(seed), dtype="float", batch_size=int(block))
        self._buffer = np.empty(0, dtype=np.float64)

    def take(self, count: int) -> np.ndarray:
        pieces, have = [], 0
        if self._buffer.size:
            pieces.append(self._buffer); have = self._buffer.size
        while have < count:
            block = np.asarray(next(self._stream), dtype=np.float64).reshape(-1)
            pieces.append(block); have += block.size
        values = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        self._buffer = values[count:]
        return values[:count]

    def uniform(self, AT, *shape):
        count = int(np.prod(shape))
        return AT.tensor(np.ascontiguousarray(self.take(count)).reshape(shape))


def _f(mask):
    """A comparison's bool mask as a float 0/1 tensor: the form every mask
    is carried in here.  AbstractTensor's &, | and ~ take a per-element
    Python path on every backend (measured: 7.9 of 9.2 s of a 20-board
    run); products, maxima, one-minus and sums are backend-vectorised."""
    return mask.to_dtype("float")


def _any(mask_f, dim=None):
    """Float-mask ANY as a bool (one vectorised comparison)."""
    return (mask_f.sum(dim=dim) if dim is not None else mask_f.sum()) > 0.0


def _rank_of(keys):
    """Each key's rank among its row: a random permutation when the keys are
    random.  Spelled as a comparison count, not a sort: ``rank_i = #{j :
    key_j < key_i}``, one masked span reduction over (batch, n, n)."""
    n = int(keys.shape[-1])
    return (keys.reshape((-1, 1, n)) < keys.reshape((-1, n, 1))).to_dtype("float").sum(dim=2)


def _shifted(AT, tensor, dr, dc, fill):
    """``tensor`` (b, n, n) moved by (dr, dc) in {-1, 0, 1}, ``fill`` (a
    same-shaped tensor) entering at the exposed edge."""
    if dr == 1:
        tensor = AT.cat([fill[:, :1, :], tensor[:, :-1, :]], dim=1)
    elif dr == -1:
        tensor = AT.cat([tensor[:, 1:, :], fill[:, :1, :]], dim=1)
    if dc == 1:
        tensor = AT.cat([fill[:, :, :1], tensor[:, :, :-1]], dim=2)
    elif dc == -1:
        tensor = AT.cat([tensor[:, :, 1:], fill[:, :, :1]], dim=2)
    return tensor


_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT = _FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1))

#: Every batched routine runs its AbstractTensor work here unless told
#: otherwise: the torch backend on the GPU.
DEFAULT_BACKEND = "torch"
DEFAULT_DEVICE = "cuda"


def singles_solve(region, solution, *, max_rounds: int | None = None):
    """Solve ``(b, n, n)`` boards by singles alone, batched.

    Each round places every cat that is forced: a row, a column or a region
    with exactly one unexcluded cell.  Each placement excludes its row,
    column, region and eight neighbours.  Every deduction is sound, so a
    board that is fully placed has exactly ONE arrangement (and it is
    ``solution``, which is asserted).  Returns ``(solved, rounds)`` per
    board: ``solved`` is the uniqueness certificate, ``rounds`` is how many
    passes of singles it took, a difficulty grade.

    This is the uniqueness gate for anything that is not unique by
    construction, and the same arithmetic as the game law's own checks.
    Masks are float 0/1 tensors throughout (see ``_f``).
    """
    from src.common.tensors import AbstractTensor as AT

    b, n = int(region.shape[0]), int(region.shape[1])
    index_n = AT.arange(n).to_dtype("float")
    one_hot = _f(region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n)))
    zero = region * 0.0
    placed = zero
    excluded = zero
    rounds = zero[:, 0, 0] * 0.0
    for _round in range(max_rounds or n):
        cand = (1.0 - excluded) * (1.0 - placed)
        row_single = _f(cand.sum(dim=2).reshape((b, n, 1)) == 1.0) + zero
        col_single = _f(cand.sum(dim=1).reshape((b, 1, n)) == 1.0) + zero
        per_region = (one_hot * cand.reshape((b, n, n, 1))).sum(dim=(1, 2))      # (b, n)
        region_single = (one_hot * _f(per_region == 1.0).reshape((b, 1, 1, n))).sum(dim=3)
        new = cand * row_single.maximum(col_single).maximum(region_single)
        if not bool(_any(new)):
            break
        rounds = rounds + _f(_any(new, dim=(1, 2)))
        placed = placed.maximum(new)
        row_done = placed.sum(dim=2).reshape((b, n, 1)) + zero
        col_done = placed.sum(dim=1).reshape((b, 1, n)) + zero
        cats_per_region = (one_hot * placed.reshape((b, n, n, 1))).sum(dim=(1, 2))   # (b, n)
        region_done = (one_hot * cats_per_region.reshape((b, 1, 1, n))).sum(dim=3)
        near = zero
        for dr, dc in _EIGHT:
            near = near + _shifted(AT, placed, dr, dc, zero)
        excluded = _f((excluded + row_done + col_done + region_done + near) > 0.0)
    solved = placed.sum(dim=(1, 2)) == float(n)
    off = (placed - solution).abs().sum(dim=(1, 2)) > 0.0
    if bool(_any(_f(off) * _f(solved))):
        raise AssertionError("singles placed a cat off the constructed solution: a deduction is unsound")
    return solved, rounds


def next_deduction(region, cats, solution, mice=None):
    """The next move a player can deduce from the visible rules alone.

    ``region`` is the ``(n, n)`` region map, ``cats`` the ``(n, n)`` 0/1 cats
    already home (the player's X marks are notes and never premises, so the
    answer is always sound), ``solution`` the leaf the game scores against
    (used only as the last-resort reveal).  The same arithmetic as
    ``singles_solve`` -- a placed cat excludes its row, column, region and
    eight neighbours; a row, column or region left with one open cell forces
    a cat there -- read from the player's own board instead of an empty one,
    and extended by the locked-line eliminations.  Returns
    ``{"kind": "cat" | "mice", "cells": [flat index, ...], "why": str}``;
    ``cells`` is empty only when the board is complete.
    """
    region = np.asarray(region, dtype=int)
    cats = np.asarray(cats, dtype=int)
    n = region.shape[0]
    marked = np.zeros(n * n, dtype=bool) if mice is None else np.asarray(mice).reshape(-1) > 0
    rr, cc = np.indices((n, n))
    excluded = np.zeros((n, n), dtype=bool)
    reasons: dict[int, str] = {}
    for r, c in zip(*np.nonzero(cats)):
        near = (np.abs(rr - r) <= 1) & (np.abs(cc - c) <= 1)
        for mask, why in ((rr == r, f"row {r + 1}"), (cc == c, f"column {c + 1}"),
                          (region == region[r, c], "its region"), (near, "touching")):
            for i in np.flatnonzero(mask.reshape(-1) & ~excluded.reshape(-1) & (cats.reshape(-1) == 0)):
                reasons[int(i)] = f"the cat at row {r + 1}, column {c + 1} covers it ({why})"
            excluded |= mask
    open_ = ~excluded & (cats == 0)
    lines = [(f"row {r + 1}", (rr == r)) for r in range(n) if not cats[r].any()]
    lines += [(f"column {c + 1}", (cc == c)) for c in range(n) if not cats[:, c].any()]
    lines += [("a coloured region", (region == k)) for k in range(n) if not cats[region == k].any()]
    for name, mask in lines:                                   # singles: one open cell forces a cat
        cells = np.flatnonzero((mask & open_).reshape(-1))
        if len(cells) == 1:
            return {"kind": "cat", "cells": [int(cells[0])], "why": f"{name} has only one open cell left"}
    # locked lines: a region whose open cells share one row (column) leaves
    # no room for other cats in that row (column), and a row (column) whose
    # open cells share one region claims that region.
    for k in range(n):
        if cats[region == k].any():
            continue
        mask = region == k
        for axis, label, index in ((rr, "row", rr), (cc, "column", cc)):
            where = np.unique(axis[mask & open_])
            if len(where) == 1:
                other = (index == where[0]) & ~mask & open_ & ~marked.reshape(n, n)
                if other.any():
                    return {"kind": "mice", "cells": [int(i) for i in np.flatnonzero(other.reshape(-1))],
                            "why": f"a coloured region's cat must be in {label} {int(where[0]) + 1}, so the rest of it is mice"}
    for label, axis in (("row", rr), ("column", cc)):
        for v in range(n):
            if (cats[axis == v]).any():
                continue
            line = (axis == v) & open_
            regions = np.unique(region[line])
            if len(regions) == 1:
                other = (region == regions[0]) & (axis != v) & open_ & ~marked.reshape(n, n)
                if other.any():
                    return {"kind": "mice", "cells": [int(i) for i in np.flatnonzero(other.reshape(-1))],
                            "why": f"{label} {v + 1}'s cat must be in one region, so that region's other cells are mice"}
    if reasons and not marked[list(reasons)].all():            # plain exclusions by a placed cat, richest first
        by_cat: dict[str, list[int]] = {}
        for i, why in reasons.items():
            if marked[i]:
                continue
            by_cat.setdefault(why.split(" covers")[0], []).append(i)
        key = max(by_cat, key=lambda k: len(by_cat[k]))
        return {"kind": "mice", "cells": sorted(by_cat[key]), "why": f"{key} rules these cells out"}
    todo = np.flatnonzero((np.asarray(solution).reshape(-1) > 0) & (cats.reshape(-1) == 0))
    if len(todo):
        return {"kind": "cat", "cells": [int(todo[0])], "why": "no pure deduction left: here is a cat"}
    return {"kind": "cat", "cells": [], "why": "the board is complete"}


_CYCLE = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))   # R UR U UL L DL D DR


def _simple_points(AT, region):
    """Cells whose removal keeps their region 4-connected: Yokoi's
    connectivity number for 4-connectivity equals one.  ``x_k`` is whether
    the k-th 8-neighbour (cyclic from the right) is in the same region;
    ``N4 = sum over k in {R, U, L, D} of x_k - x_k x_{k+1} x_{k+2}``.  Checked
    against a flood fill in the tests."""
    fill = region * 0.0 - 2.0
    x = [(_shifted(AT, region, -dr, -dc, fill) == region).to_dtype("float") for dr, dc in _CYCLE]
    total = region * 0.0
    for k in (0, 2, 4, 6):
        total = total + x[k] - x[k] * x[(k + 1) % 8] * x[(k + 2) % 8]
    return total == 1.0


def generate_batch(n: int, batch: int, seed: int = 0, *, style: str = "relaxed",
                   rounds: int | None = None, jitter: float = 0.35,
                   chunk_elements: int = 32_000_000, backend: str = DEFAULT_BACKEND,
                   device=DEFAULT_DEVICE) -> PuzzleBatch:
    """``batch`` contiguous puzzles of size ``n`` from first principles,
    batched: AbstractTensor arithmetic over ``(batch, n, n)`` spans and
    ``(batch, n, n, n)`` one-hots, never a loop over cells or boards.

    Two styles.  ``"relaxed"`` (steps 2 to 4 below) is unique by
    construction.  ``"free"`` keeps step 1 and grows the regions as plain
    wavefronts from every cat, nearest jittered centre first, with no
    solving-order constraint: balanced shapes, any number of valid
    arrangements.  The solution column is the cat map either way: the
    arrangement the game scores against, chosen ahead of time; ``tree.py``
    types a board by its solution tree and can draw another leaf.

    1. CAT MAP first.  The arrangement is a random permutation ``sigma``
       (column per row), the rank of random keys; the no-touch rule
       ``|sigma[r] - sigma[r+1]| >= 2`` is a batched rejection.
    2. A solving order ``rank`` over the cats.  A non-cat cell ``(r, c)`` is
       eliminated the moment the cat of row ``r`` or the cat in column ``c``
       is placed: ``eliminated_at = min(rank[r], rank[rowcat[c]])``.  A cell
       is ELIGIBLE for region ``k`` when ``rank[k] > eliminated_at``.  Any
       partition in which every cell is eligible for its region is solved
       by singles in that order, so it has exactly one arrangement.
    3. The valid start: the last-solved region owns every non-cat cell
       (eligible everywhere), every other region is its cat.  Valid,
       contiguous, unique, and not yet a puzzle.
    4. RELAXATION.  Each round, each parity class ``(r % 2, c % 2)`` in
       turn: every non-cat cell that is a 4-simple point of its region
       (``_simple_points``) and has an eligible 4-neighbour region at least
       two cells smaller moves to the smallest such neighbour.  Cells of
       one parity class are never 8-adjacent, so simultaneous moves are
       independent and every region stays connected; the donor keeps its
       cat; the receiver gains an adjacent cell.  Then one PIECE HANDOVER
       per board: a non-simple boundary cell moves together with the piece
       it cuts off from its donor's cat, when that whole piece is eligible
       for the receiver (a flood from the cats finds the piece).  Validity
       therefore holds after every round, and the rounds stop when nothing
       can move.

    What comes out: every region contiguous, every board unique by
    construction (``count_arrangements`` is the oracle in the tests at the
    sizes it finishes), sizes as balanced as eligibility allows.  The order
    constraint is real: an early-solved region can own only cells in rows
    and columns of earlier cats, so the last region stays the largest
    (about half the board) and a few early ones are single cells.  Those
    are the puzzle's openings.

    Chunks keep the ``(b, n, n, n)`` one-hot under ``chunk_elements``
    cells; the chunk loop and the relaxation rounds are the only Python
    loops.
    """
    from src.common.tensors import AbstractTensor as AT

    if n < 4:
        raise ValueError("Meowdoku needs at least a 4x4 board for a no-touch arrangement")
    if batch < 1:
        raise ValueError("batch must be at least one puzzle")
    if style not in {"relaxed", "free"}:
        raise ValueError(f"style={style!r}: relaxed or free")
    per_chunk = max(1, int(chunk_elements) // (n * n * n))
    regions, solutions = [], []
    # One seeded stream of the repository's own PRNG (Xoroshiro128**) feeds
    # every draw in order, so a seed names the whole batch exactly.
    # (``AbstractTensor.randn`` is not reached by any seed; measured.)
    # ``backend``/``device`` scope every tensor of the build: ``"torch"``
    # with ``device="cuda"`` puts the whole batch on the GPU; the draws come
    # from the host stream, buffered, and are moved once per draw.
    draws = _Draws(seed)

    def uniform(*shape):
        return draws.uniform(AT, *shape)

    with AT.use_backend(backend, device):
        index_n = AT.arange(n).to_dtype("float")
        rr = index_n.reshape((1, n, 1))
        cc = index_n.reshape((1, 1, n))
        columns = index_n.reshape((1, 1, n))
        regions_k = index_n.reshape((1, 1, 1, n))
        huge = float(n * n + 1)
        done = 0
        while done < batch:
            b = min(per_chunk, batch - done)
            # 1. cat map: ranks of random keys, redrawn until no two
            #    consecutive rows touch.
            sigma = _rank_of(uniform(b, n))
            touches = _f((sigma[:, 1:] - sigma[:, :-1]).abs() < 2.0).sum(dim=1)     # (b,) count
            while bool(_any(touches)):
                redraw = _rank_of(uniform(b, n))
                sigma = AT.where((touches == 0.0).reshape((b, 1)), sigma, redraw)
                touches = _f((sigma[:, 1:] - sigma[:, :-1]).abs() < 2.0).sum(dim=1)
            is_cat = (sigma.reshape((b, n, 1)) == columns).to_dtype("float")   # (b, n, n)
            zero = is_cat * 0.0
            if style == "free":
                # Wavefronts from every cat; each round every unassigned
                # cell joins the 4-neighbour whose region centre (jittered
                # per cat) is nearest.  Contiguous by construction.
                centre_r = index_n.reshape((1, n)) + (uniform(b, n) - 0.5) * (float(jitter) * n)
                centre_c = sigma + (uniform(b, n) - 0.5) * (float(jitter) * n)
                region = AT.where(is_cat > 0.5, rr + zero, zero - 1.0)
                own_cr = AT.where(is_cat > 0.5, centre_r.reshape((b, n, 1)) + zero, zero)
                own_cc = AT.where(is_cat > 0.5, centre_c.reshape((b, n, 1)) + zero, zero)
                cell_r, cell_c = rr + zero, cc + zero
                for _round in range(2 * n + 2):
                    unassigned = _f(region < 0.0)
                    if not bool(_any(unassigned)):
                        break
                    best_score = zero + float(4 * n * n + 1)
                    best = [region, own_cr, own_cc]
                    for dr, dc in _FOUR:
                        nb = [_shifted(AT, m, dr, dc, zero - 1.0) for m in (region, own_cr, own_cc)]
                        score = (cell_r - nb[1]) ** 2 + (cell_c - nb[2]) ** 2
                        take = (unassigned * _f(nb[0] >= 0.0) * _f(score < best_score)) > 0.5
                        best_score = AT.where(take, score, best_score)
                        best = [AT.where(take, new, old) for new, old in zip(nb, best)]
                    region, own_cr, own_cc = best
                regions.append(np.asarray(region.numpy(), dtype=np.float64).reshape(b, n * n))
                solutions.append(np.asarray(is_cat.numpy(), dtype=np.float64).reshape(b, n * n))
                done += b
                continue
            # 2. solving order and each cell's elimination moment.
            rank = _rank_of(uniform(b, n))                                     # (b, n)
            rank_of_column_cat = (is_cat * rank.reshape((b, n, 1))).sum(dim=1)   # (b, n)
            eliminated_at = rank.reshape((b, n, 1)).minimum(rank_of_column_cat.reshape((b, 1, n)))
            # 3. the valid start: everything to the last-solved region.
            last = ((rank == float(n - 1)).to_dtype("float") * index_n.reshape((1, n))).sum(dim=1)   # (b,)
            region = AT.where(is_cat > 0.5, rr + zero, last.reshape((b, 1, 1)) + zero)
            own_rank = AT.where(is_cat > 0.5, rank.reshape((b, n, 1)) + zero, float(n - 1) + zero)
            parity_r, parity_c = (rr % 2.0) + zero, (cc % 2.0) + zero
            # 4. relaxation rounds: simple-point moves per parity class, then
            #    one piece handover per board.
            cell_index = AT.arange(n * n).to_dtype("float").reshape((1, n * n))
            for _round in range(rounds or 8 * n):
                moved = False
                for pr, pc in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
                    one_hot = (region.reshape((b, n, n, 1)) == regions_k).to_dtype("float")
                    size = one_hot.sum(dim=(1, 2))                                    # (b, n)
                    own_size = (one_hot * size.reshape((b, 1, 1, n))).sum(dim=3)      # (b, n, n)
                    may_move = (_f(_simple_points(AT, region)) * (1.0 - is_cat)
                                * _f(parity_r == pr) * _f(parity_c == pc))
                    best_size = zero + huge
                    best_region, best_rank = region, own_rank
                    for dr, dc in _FOUR:
                        nb_region = _shifted(AT, region, dr, dc, zero - 1.0)
                        nb_rank = _shifted(AT, own_rank, dr, dc, zero - 1.0)
                        nb_size = _shifted(AT, own_size, dr, dc, zero + huge)
                        take = (may_move * _f(nb_region >= 0.0) * _f(nb_region != region)
                                * _f(nb_rank > eliminated_at) * _f(nb_size + 1.0 < own_size)
                                * _f(nb_size < best_size)) > 0.5
                        best_size = AT.where(take, nb_size, best_size)
                        best_region = AT.where(take, nb_region, best_region)
                        best_rank = AT.where(take, nb_rank, best_rank)
                    if bool(_any((best_region - region).abs())):
                        moved = True
                        region, own_rank = best_region, best_rank
                if moved:
                    # Handovers only when the simple-point moves have
                    # stalled: each needs a flood, and most rounds do not.
                    continue
                # Piece handover.  A cell that is NOT a simple point of its
                # donor may still move: the piece it cuts off from the
                # donor's cat goes with it, when every cell of that piece is
                # eligible for the receiver.  The piece hangs off the moved
                # cell and the moved cell off the receiver, so both regions
                # stay connected.  One handover per board per round, the one
                # with the largest size difference; the flood from the cats
                # finds the piece.  (Measured: without this, half the
                # largest region's boundary was eligible but not simple.)
                one_hot = (region.reshape((b, n, n, 1)) == regions_k).to_dtype("float")
                size = one_hot.sum(dim=(1, 2))
                own_size = (one_hot * size.reshape((b, 1, 1, n))).sum(dim=3)
                not_simple = (1.0 - _f(_simple_points(AT, region))) * (1.0 - is_cat)
                best_gain = zero - 1.0
                best_region, best_rank = region, own_rank
                for dr, dc in _FOUR:
                    nb_region = _shifted(AT, region, dr, dc, zero - 1.0)
                    nb_rank = _shifted(AT, own_rank, dr, dc, zero - 1.0)
                    nb_size = _shifted(AT, own_size, dr, dc, zero + huge)
                    gain = own_size - nb_size
                    take = (not_simple * _f(nb_region >= 0.0) * _f(nb_region != region)
                            * _f(nb_rank > eliminated_at) * _f(nb_size + 1.0 < own_size)
                            * _f(gain > best_gain)) > 0.5
                    best_gain = AT.where(take, gain, best_gain)
                    best_region = AT.where(take, nb_region, best_region)
                    best_rank = AT.where(take, nb_rank, best_rank)
                flat_gain = best_gain.reshape((b, n * n))
                has = _f(_any(_f(flat_gain > 0.0), dim=1))                      # (b,) float
                if not bool(_any(has)):
                    if not moved:
                        break
                    continue
                pick = (flat_gain * -1.0).argmin(dim=1).to_dtype("float").reshape((b, 1))   # first best per board
                pick_mask = _f(cell_index == pick).reshape((b, n, n)) * has.reshape((b, 1, 1))
                trial = AT.where(pick_mask > 0.5, best_region, region)
                trial_rank = AT.where(pick_mask > 0.5, best_rank, own_rank)
                receiver = (pick_mask * best_region).sum(dim=(1, 2)).reshape((b, 1, 1))
                receiver_rank = (pick_mask * best_rank).sum(dim=(1, 2)).reshape((b, 1, 1))
                reached = is_cat
                for _flood in range(n * n):
                    grow = reached
                    for dr, dc in _FOUR:
                        same = _f(_shifted(AT, trial, dr, dc, zero - 2.0) == trial)
                        grow = grow.maximum(_shifted(AT, reached, dr, dc, zero) * same)
                    if not bool(_any(grow - reached)):
                        break
                    reached = grow
                stranded = 1.0 - reached
                ineligible = (stranded * _f(receiver_rank <= eliminated_at)).sum(dim=(1, 2))   # (b,)
                ok_board = (has * _f(ineligible == 0.0)).reshape((b, 1, 1))
                handed = AT.where(stranded > 0.5, receiver + zero, trial)
                handed_rank = AT.where(stranded > 0.5, receiver_rank + zero, trial_rank)
                if bool(_any(ok_board)):
                    moved = True
                    region = AT.where(ok_board > 0.5, handed, region)
                    own_rank = AT.where(ok_board > 0.5, handed_rank, own_rank)
                if not moved:
                    break
            regions.append(np.asarray(region.numpy(), dtype=np.float64).reshape(b, n * n))
            solutions.append(np.asarray(is_cat.numpy(), dtype=np.float64).reshape(b, n * n))
            done += b
    cells = np.arange(n * n)
    row = np.broadcast_to((cells // n).astype(np.float64), (batch, n * n)).copy()
    col = np.broadcast_to((cells % n).astype(np.float64), (batch, n * n)).copy()
    return PuzzleBatch(n, row, col, np.concatenate(regions), np.concatenate(solutions))


def contiguous_regions(puzzles: "PuzzleBatch", *, backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE) -> np.ndarray:
    """Per puzzle: every region is one 4-connected piece.  A batched flood
    from each cat over its own region; a region is contiguous when the
    flood covers it.  Float masks."""
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend(backend, device):
        region = AT.tensor(puzzles.region.reshape(b, n, n))
        reached = AT.tensor(puzzles.solution.reshape(b, n, n))
        zero = region * 0.0
        for _round in range(n * n):
            grow = reached
            for dr, dc in _FOUR:
                same = _f(_shifted(AT, region, dr, dc, zero - 1.0) == region)
                grow = grow.maximum(_shifted(AT, reached, dr, dc, zero) * same)
            if not bool(_any(grow - reached)):
                break
            reached = grow
        covered = (1.0 - reached).sum(dim=(1, 2)) == 0.0
        return np.asarray(covered.numpy(), dtype=bool).reshape(b)


def verify_batch(puzzles: PuzzleBatch, *, backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE) -> np.ndarray:
    """Per puzzle: the solution obeys all three rules and every region holds
    exactly one cat.  Batched AbstractTensor checks, the same arithmetic the
    game law uses, not a Python walk.  Float masks."""
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend(backend, device):
        solution = AT.tensor(puzzles.solution.reshape(b, n, n))
        region = AT.tensor(puzzles.region.reshape(b, n, n))
        index_n = AT.arange(n).to_dtype("float")
        rows_off = _f(solution.sum(dim=2) != 1.0).sum(dim=1)
        cols_off = _f(solution.sum(dim=1) != 1.0).sum(dim=1)
        sigma = (solution * index_n.reshape((1, 1, n))).sum(dim=2)
        touches = _f((sigma[:, 1:] - sigma[:, :-1]).abs() < 2.0).sum(dim=1)
        one_hot = _f(region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n)))
        cats_per_region = (one_hot * solution.reshape((b, n, n, 1))).sum(dim=(1, 2))
        regions_off = _f(cats_per_region != 1.0).sum(dim=1)
        unused = _f(one_hot.sum(dim=(1, 2)) == 0.0).sum(dim=1)
        ok = (rows_off + cols_off + touches + regions_off + unused) == 0.0
        return np.asarray(ok.numpy(), dtype=bool).reshape(b)


def generate(n: int, seed: int = 0, method: str = "auto") -> Puzzle:
    """A puzzle with exactly one arrangement, deterministic per seed.

    ``method``: ``"random"`` grows regions at random and keeps a board only
    when the backtracking count finds one arrangement (the original
    generator; fast to n=8 and hopeless past 9, kept as the oracle);
    ``"batched"`` is ``generate_batch`` (any n, AbstractTensor, unique by
    construction); ``"auto"`` keeps the random boards for n <= 8, so
    existing seeds stay what they were, and uses the batched generator
    above.
    """
    if n < 4:
        raise ValueError("Meowdoku needs at least a 4x4 board for a no-touch arrangement")
    if method not in {"auto", "random", "batched"}:
        raise ValueError(f"method={method!r}: auto, random or batched")
    if method == "auto":
        method = "random" if n <= 8 else "batched"
    if method == "batched":
        return generate_batch(n, 1, seed).puzzle(0)
    rng = random.Random(seed)
    while True:
        cats = cat_arrangement(n, rng)
        regions = grow_regions(n, cats, rng)
        if count_arrangements(regions) == 1:
            break
    cells = np.arange(n * n)
    solution = np.zeros(n * n, dtype=np.float64)
    for r, c in enumerate(cats):
        solution[r * n + c] = 1.0
    return Puzzle(
        n=n,
        row=(cells // n).astype(np.float64),
        col=(cells % n).astype(np.float64),
        region=regions.reshape(-1).astype(np.float64),
        solution=solution,
    )

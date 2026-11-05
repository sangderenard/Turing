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
    """
    from src.common.tensors import AbstractTensor as AT

    b, n = int(region.shape[0]), int(region.shape[1])
    index_n = AT.arange(n).to_dtype("float")
    one_hot = (region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n))).to_dtype("float")
    zero = region * 0.0
    placed = zero > 1.0
    excluded = zero > 1.0
    rounds = zero[:, 0, 0] * 0.0
    for _round in range(max_rounds or n):
        candidate = ~(excluded | placed)
        cand = candidate.to_dtype("float")
        row_single = cand.sum(dim=2).reshape((b, n, 1)) == 1.0
        col_single = cand.sum(dim=1).reshape((b, 1, n)) == 1.0
        per_region = (one_hot * cand.reshape((b, n, n, 1))).sum(dim=(1, 2))      # (b, n)
        region_single = (one_hot * per_region.reshape((b, 1, 1, n))).sum(dim=3) == 1.0
        new = candidate & (row_single | col_single | region_single)
        if not bool(new.any()):
            break
        rounds = rounds + new.any(dim=(1, 2)).to_dtype("float")
        placed = placed | new
        got = placed.to_dtype("float")
        row_done = got.sum(dim=2).reshape((b, n, 1)) > 0.0
        col_done = got.sum(dim=1).reshape((b, 1, n)) > 0.0
        cats_per_region = (one_hot * got.reshape((b, n, n, 1))).sum(dim=(1, 2))   # (b, n)
        region_done = (one_hot * cats_per_region.reshape((b, 1, 1, n))).sum(dim=3) > 0.0
        near = zero
        for dr, dc in _EIGHT:
            near = near + _shifted(AT, got, dr, dc, zero)
        excluded = excluded | row_done | col_done | region_done | (near > 0.0)
    solved = placed.to_dtype("float").sum(dim=(1, 2)) == float(n)
    agrees = ((placed.to_dtype("float") - solution).abs().sum(dim=(1, 2)) == 0.0) | ~solved
    if not bool(agrees.all()):
        raise AssertionError("singles placed a cat off the constructed solution: a deduction is unsound")
    return solved, rounds


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
            valid = ((sigma[:, 1:] - sigma[:, :-1]).abs() >= 2.0).all(dim=1)
            while not bool(valid.all()):
                redraw = _rank_of(uniform(b, n))
                sigma = AT.where(valid.reshape((b, 1)), sigma, redraw)
                valid = ((sigma[:, 1:] - sigma[:, :-1]).abs() >= 2.0).all(dim=1)
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
                    unassigned = region < 0.0
                    if not bool(unassigned.any()):
                        break
                    best_score = zero + float(4 * n * n + 1)
                    best = [region, own_cr, own_cc]
                    for dr, dc in _FOUR:
                        nb = [_shifted(AT, m, dr, dc, zero - 1.0) for m in (region, own_cr, own_cc)]
                        score = (cell_r - nb[1]) ** 2 + (cell_c - nb[2]) ** 2
                        take = unassigned & (nb[0] >= 0.0) & (score < best_score)
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
                    may_move = _simple_points(AT, region) & (is_cat < 0.5) & (parity_r == pr) & (parity_c == pc)
                    best_size = zero + huge
                    best_region, best_rank = region, own_rank
                    for dr, dc in _FOUR:
                        nb_region = _shifted(AT, region, dr, dc, zero - 1.0)
                        nb_rank = _shifted(AT, own_rank, dr, dc, zero - 1.0)
                        nb_size = _shifted(AT, own_size, dr, dc, zero + huge)
                        take = (may_move & (nb_region >= 0.0) & (nb_region != region)
                                & (nb_rank > eliminated_at) & (nb_size + 1.0 < own_size) & (nb_size < best_size))
                        best_size = AT.where(take, nb_size, best_size)
                        best_region = AT.where(take, nb_region, best_region)
                        best_rank = AT.where(take, nb_rank, best_rank)
                    if bool((best_region != region).any()):
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
                not_simple = ~_simple_points(AT, region) & (is_cat < 0.5)
                best_gain = zero - 1.0
                best_region, best_rank = region, own_rank
                for dr, dc in _FOUR:
                    nb_region = _shifted(AT, region, dr, dc, zero - 1.0)
                    nb_rank = _shifted(AT, own_rank, dr, dc, zero - 1.0)
                    nb_size = _shifted(AT, own_size, dr, dc, zero + huge)
                    gain = own_size - nb_size
                    take = (not_simple & (nb_region >= 0.0) & (nb_region != region)
                            & (nb_rank > eliminated_at) & (nb_size + 1.0 < own_size) & (gain > best_gain))
                    best_gain = AT.where(take, gain, best_gain)
                    best_region = AT.where(take, nb_region, best_region)
                    best_rank = AT.where(take, nb_rank, best_rank)
                flat_gain = best_gain.reshape((b, n * n))
                has = flat_gain.max(dim=1) > 0.0 if hasattr(flat_gain, "max") else None
                if has is None or not bool(has.any()):
                    if not moved:
                        break
                    continue
                pick = (flat_gain * -1.0).argmin(dim=1).to_dtype("float").reshape((b, 1))   # first best per board
                pick_mask = (cell_index == pick).reshape((b, n, n)) & has.reshape((b, 1, 1))
                trial = AT.where(pick_mask, best_region, region)
                trial_rank = AT.where(pick_mask, best_rank, own_rank)
                receiver = (pick_mask.to_dtype("float") * best_region).sum(dim=(1, 2)).reshape((b, 1, 1))
                receiver_rank = (pick_mask.to_dtype("float") * best_rank).sum(dim=(1, 2)).reshape((b, 1, 1))
                reached = is_cat > 0.5
                for _flood in range(n * n):
                    grow = reached
                    for dr, dc in _FOUR:
                        same = _shifted(AT, trial, dr, dc, zero - 2.0) == trial
                        grow = grow | (_shifted(AT, reached.to_dtype("float"), dr, dc, zero) > 0.0) & same
                    if not bool((grow & ~reached).any()):
                        break
                    reached = grow
                stranded = ~reached
                ok_board = has.reshape((b, 1, 1)) & ~(stranded & ~(receiver_rank > eliminated_at)).any(dim=(1, 2)).reshape((b, 1, 1))
                handed = AT.where(stranded, receiver + zero, trial)
                handed_rank = AT.where(stranded, receiver_rank + zero, trial_rank)
                if bool(ok_board.any()):
                    moved = True
                    region = AT.where(ok_board, handed, region)
                    own_rank = AT.where(ok_board, handed_rank, own_rank)
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
    flood covers it."""
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend(backend, device):
        region = AT.tensor(puzzles.region.reshape(b, n, n))
        reached = AT.tensor(puzzles.solution.reshape(b, n, n)) > 0.5
        zero = region * 0.0
        for _round in range(n * n):
            grow = reached
            got = reached.to_dtype("float")
            for dr, dc in _FOUR:
                same = _shifted(AT, region, dr, dc, zero - 1.0) == region
                grow = grow | ((_shifted(AT, got, dr, dc, zero) > 0.0) & same)
            if not bool((grow & ~reached).any()):
                break
            reached = grow
        return np.asarray(reached.all(dim=(1, 2)).numpy(), dtype=bool).reshape(b)


def verify_batch(puzzles: PuzzleBatch, *, backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE) -> np.ndarray:
    """Per puzzle: the solution obeys all three rules and every region holds
    exactly one cat.  Batched AbstractTensor checks, the same arithmetic the
    game law uses, not a Python walk."""
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend(backend, device):
        solution = AT.tensor(puzzles.solution.reshape(b, n, n))
        region = AT.tensor(puzzles.region.reshape(b, n, n))
        index_n = AT.arange(n).to_dtype("float")
        one_per_row = (solution.sum(dim=2) == 1.0).all(dim=1)
        one_per_col = (solution.sum(dim=1) == 1.0).all(dim=1)
        sigma = (solution * index_n.reshape((1, 1, n))).sum(dim=2)
        no_touch = ((sigma[:, 1:] - sigma[:, :-1]).abs() >= 2.0).all(dim=1)
        one_hot = (region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n))).to_dtype("float")
        cats_per_region = (one_hot * solution.reshape((b, n, n, 1))).sum(dim=(1, 2))
        one_per_region = (cats_per_region == 1.0).all(dim=1)
        every_region_used = (one_hot.sum(dim=(1, 2)) > 0.0).all(dim=1)
        ok = one_per_row & one_per_col & no_touch & one_per_region & every_region_used
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

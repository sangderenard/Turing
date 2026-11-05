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


def _rank_of(keys):
    """Each key's rank among its row: a random permutation when the keys are
    random.  Spelled as a comparison count, not a sort: ``rank_i = #{j :
    key_j < key_i}``, one masked span reduction over (batch, n, n)."""
    n = int(keys.shape[-1])
    return (keys.reshape((-1, 1, n)) < keys.reshape((-1, n, 1))).to_dtype("float").sum(dim=2)


def generate_batch(n: int, batch: int, seed: int = 0, *, jitter: float = 0.35,
                   chunk_elements: int = 32_000_000) -> PuzzleBatch:
    """``batch`` unique puzzles of size ``n`` from first principles, batched.

    Everything is AbstractTensor arithmetic over ``(batch, n, n)`` spans (and
    one ``(batch, n, n, n)`` region score), no loop over cells:

    1. The arrangement is a random permutation ``sigma`` (column per row),
       taken as the rank of random keys; the no-touch rule
       ``|sigma[r] - sigma[r+1]| >= 2`` is a batched rejection: boards that
       fail are redrawn until every board passes.
    2. A random solving ORDER ``rank`` over the rows.  A non-cat cell ``(r,
       c)`` is eliminated the moment the cat of row ``r`` or the cat standing
       in column ``c`` is placed: ``eliminated_at = min(rank[r],
       rank[rowcat[c]])``.
    3. A cell may belong to any region solved after that moment
       (``rank[k] > eliminated_at``).  Among the eligible regions it takes the
       nearest cat by jittered squared distance (Voronoi-like, so regions
       are compact), by one masked argmin.  Cat cells own their own region.

    Solving in that order, every region in turn holds exactly one unexcluded
    cell, its cat: singles alone solve the board, so the arrangement is
    unique by construction.  ``count_arrangements`` is the oracle this is
    checked against in the tests at the sizes the oracle still finishes.

    Chunks of the batch keep the ``(b, n, n, n)`` score under
    ``chunk_elements`` cells; the chunk loop is the only Python loop.
    """
    from src.common.tensors import AbstractTensor as AT
    from src.common.tensors.abstraction_methods.random import (
        PRNG_ALGO, RANDOM_KIND, random_generator,
    )

    if n < 4:
        raise ValueError("Meowdoku needs at least a 4x4 board for a no-touch arrangement")
    if batch < 1:
        raise ValueError("batch must be at least one puzzle")
    per_chunk = max(1, int(chunk_elements) // (n * n * n))
    regions = []
    solutions = []
    # One seeded stream of the repository's own PRNG (Xoroshiro128**) feeds
    # every draw in order, so a seed names the whole batch exactly.
    # (``AbstractTensor.randn`` is not reached by any seed; measured.)
    stream = random_generator(kind=RANDOM_KIND.PRNG, algo=PRNG_ALGO.XOROSHIRO128SS,
                              seed=int(seed), dtype="float", batch_size=1)

    def uniform(*shape):
        count = int(np.prod(shape))
        values = np.fromiter((next(stream) for _ in range(count)), dtype=np.float64, count=count)
        return AT.tensor(values.reshape(shape))

    with AT.use_backend("numpy"):
        index_n = AT.arange(n).to_dtype("float")
        rr = index_n.reshape((1, n, 1, 1))
        cc = index_n.reshape((1, 1, n, 1))
        kk = index_n.reshape((1, 1, 1, n))
        columns = index_n.reshape((1, 1, n))
        big = float(4 * n * n + 1)
        done = 0
        while done < batch:
            b = min(per_chunk, batch - done)
            # 1. arrangement: ranks of random keys, redrawn until no two
            #    consecutive rows touch.
            sigma = _rank_of(uniform(b, n))
            valid = ((sigma[:, 1:] - sigma[:, :-1]).abs() >= 2.0).all(dim=1)
            while not bool(valid.all()):
                redraw = _rank_of(uniform(b, n))
                sigma = AT.where(valid.reshape((b, 1)), sigma, redraw)
                valid = ((sigma[:, 1:] - sigma[:, :-1]).abs() >= 2.0).all(dim=1)
            # 2. solving order and each cell's elimination moment.
            rank = _rank_of(uniform(b, n))                                 # (b, n)
            is_cat = (sigma.reshape((b, n, 1)) == columns).to_dtype("float")   # (b, n, n)
            rank_of_column_cat = (is_cat * rank.reshape((b, n, 1))).sum(dim=1)   # (b, n)
            eliminated_at = rank.reshape((b, n, 1)).minimum(rank_of_column_cat.reshape((b, 1, n)))
            # 3. nearest eligible cat, by one masked argmin over regions k.
            eligible = rank.reshape((b, 1, 1, n)) > eliminated_at.reshape((b, n, n, 1))
            #    Each cat's centre is jittered so region shapes vary between
            #    boards; the jitter is per cat, not per cell, so it stays a
            #    Voronoi partition and costs 2n draws per board.
            centre_r = kk + (uniform(b, 1, 1, n) - 0.5) * (float(jitter) * n)
            centre_c = sigma.reshape((b, 1, 1, n)) + (uniform(b, 1, 1, n) - 0.5) * (float(jitter) * n)
            distance = (rr - centre_r) * (rr - centre_r) + (cc - centre_c) * (cc - centre_c)
            score = AT.where(eligible, distance, big)
            region = score.argmin(dim=3).to_dtype("float")                 # (b, n, n)
            region = AT.where(is_cat > 0.5, rr.reshape((1, n, 1)) + region * 0.0, region)
            regions.append(np.asarray(region.numpy(), dtype=np.float64).reshape(b, n * n))
            solutions.append(np.asarray(is_cat.numpy(), dtype=np.float64).reshape(b, n * n))
            done += b
    cells = np.arange(n * n)
    row = np.broadcast_to((cells // n).astype(np.float64), (batch, n * n)).copy()
    col = np.broadcast_to((cells % n).astype(np.float64), (batch, n * n)).copy()
    return PuzzleBatch(n, row, col, np.concatenate(regions), np.concatenate(solutions))


def verify_batch(puzzles: PuzzleBatch) -> np.ndarray:
    """Per puzzle: the solution obeys all three rules and every region holds
    exactly one cat.  Batched AbstractTensor checks, the same arithmetic the
    game law uses, not a Python walk."""
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend("numpy"):
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

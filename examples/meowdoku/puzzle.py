"""Meowdoku puzzle setup: a unique cat arrangement and the regions around it.

This is setup, not a step law.  It runs once in Python and produces the
static columns the game law reads every round: ``row``, ``col``, ``region``
and the hidden ``solution`` mask, one cell per lane in row-major order
(``cell = row * n + col``).

Rules the arrangement satisfies (the published game's three rules):
one cat per row, one per column, one per region, and no two cats touch,
diagonals included.  Regions are grown outward from the cats, and a board is
accepted only when a backtracking count finds exactly one arrangement that
satisfies all three rules -- the game promises a single solution.
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


def generate(n: int, seed: int = 0) -> Puzzle:
    """A puzzle with exactly one arrangement, deterministic per seed."""
    if n < 4:
        raise ValueError("Meowdoku needs at least a 4x4 board for a no-touch arrangement")
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

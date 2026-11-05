"""The solution tree of a Meowdoku region map, resolved on AbstractTensor.

A region map may admit many cat arrangements.  Solving by deduction
(singles: a row, column or region with one candidate; each placement
excluding its row, column, region and eight neighbours) finishes, dies in
a contradiction, or STALLS.  At a stall the solver must guess: the region
with the fewest candidates is the decision node, each candidate a branch.
The tree of those decisions down to the valid arrangements is the board's
solution tree: its leaves are the arrangements, its decision nodes the
guesses a solver cannot avoid.

Everything vectorised here is AbstractTensor, batched over boards, on the
torch backend on CUDA by default:

- ``propagate``: singles to a fixed point for a whole batch at once;
- ``sample_solutions``: a random walk down every board's tree at once,
  restarting a board from the top when it hits a contradiction, which
  resolves a batch of maps to one leaf each and counts the decisions taken;
- ``solution_tree``: the full tree of one board (recursive over branches,
  AbstractTensor for every mask), enumerated while it fits ``limit_leaves``;
- ``catalog`` / ``type_table`` / ``select``: the typed records a database
  is built from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_BACKEND = "torch"
DEFAULT_DEVICE = "cuda"

_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT = _FOUR + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _shifted(AT, tensor, dr, dc, fill):
    """``tensor`` (b, n, n) moved by (dr, dc) in {-1, 0, 1}; ``fill`` enters."""
    if dr == 1:
        tensor = AT.cat([fill[:, :1, :], tensor[:, :-1, :]], dim=1)
    elif dr == -1:
        tensor = AT.cat([tensor[:, 1:, :], fill[:, :1, :]], dim=1)
    if dc == 1:
        tensor = AT.cat([fill[:, :, :1], tensor[:, :, :-1]], dim=2)
    elif dc == -1:
        tensor = AT.cat([tensor[:, :, 1:], fill[:, :, :1]], dim=2)
    return tensor


def _f(mask):
    return mask.to_dtype("float")


def exclusions(AT, region, one_hot, placed):
    """Cells excluded by the placed cats: their rows, columns, regions and
    eight neighbours.  ``placed`` (b, n, n) float 0/1."""
    b, n = int(region.shape[0]), int(region.shape[1])
    zero = region * 0.0
    row_done = placed.sum(dim=2).reshape((b, n, 1)) + zero
    col_done = placed.sum(dim=1).reshape((b, 1, n)) + zero
    cats_per_region = (one_hot * placed.reshape((b, n, n, 1))).sum(dim=(1, 2))        # (b, n)
    region_done = (one_hot * cats_per_region.reshape((b, 1, 1, n))).sum(dim=3)
    near = zero
    for dr, dc in _EIGHT:
        near = near + _shifted(AT, placed, dr, dc, zero)
    return (row_done + col_done + region_done + near) > 0.0


def propagate(AT, region, one_hot, placed, excluded, *, max_rounds: int | None = None):
    """Singles to a fixed point, batched.  Returns ``(placed, excluded,
    solved, dead, rounds)``: ``solved`` and ``dead`` are per-board booleans
    (a board that is neither is stalled), ``rounds`` the singles passes.

    Every deduction is sound, so a solved board's placement is a valid
    arrangement.  A contradiction is a row, column or region with no cat
    and no candidate, or forced placements that conflict with each other
    (same row, column or region, or touching): each of them is forced, and
    they cannot all stand."""
    b, n = int(region.shape[0]), int(region.shape[1])
    zero = region * 0.0
    rounds = zero[:, 0, 0] * 0.0
    dead = rounds > 1.0
    excluded = excluded | exclusions(AT, region, one_hot, placed)
    for _round in range(max_rounds or 2 * n):
        candidate = ~(excluded | (placed > 0.5))
        cand = _f(candidate)
        row_count = cand.sum(dim=2)                                                    # (b, n)
        col_count = cand.sum(dim=1)
        region_count = (one_hot * cand.reshape((b, n, n, 1))).sum(dim=(1, 2))          # (b, n)
        row_has = placed.sum(dim=2) > 0.5
        col_has = placed.sum(dim=1) > 0.5
        region_has = (one_hot * placed.reshape((b, n, n, 1))).sum(dim=(1, 2)) > 0.5
        starved = ((~row_has & (row_count == 0.0)).any(dim=1) | (~col_has & (col_count == 0.0)).any(dim=1)
                   | (~region_has & (region_count == 0.0)).any(dim=1))
        dead = dead | starved
        row_single = (row_count == 1.0).reshape((b, n, 1)) + zero > 0.5
        col_single = (col_count == 1.0).reshape((b, 1, n)) + zero > 0.5
        region_single = (one_hot * _f(region_count == 1.0).reshape((b, 1, 1, n))).sum(dim=3) > 0.5
        new = candidate & (row_single | col_single | region_single)
        if not bool(new.any()):
            break
        rounds = rounds + _f(new.any(dim=(1, 2)))
        together = placed + _f(new)
        near_new = zero
        for dr, dc in _EIGHT:
            near_new = near_new + _shifted(AT, together, dr, dc, zero)
        conflict = ((together.sum(dim=2) > 1.0).any(dim=1) | (together.sum(dim=1) > 1.0).any(dim=1)
                    | ((one_hot * together.reshape((b, n, n, 1))).sum(dim=(1, 2)) > 1.0).any(dim=1)
                    | ((near_new > 0.0) & (together > 0.5) & (_f(new) > 0.5)).any(dim=(1, 2)))
        dead = dead | conflict
        placed = together.minimum(zero + 1.0)
        excluded = excluded | exclusions(AT, region, one_hot, placed)
    solved = (placed.sum(dim=(1, 2)) == float(n)) & ~dead
    return placed, excluded, solved, dead, rounds


def _uniform_stream(seed: int):
    from src.common.tensors.abstraction_methods.random import PRNG_ALGO, RANDOM_KIND, random_generator
    return random_generator(kind=RANDOM_KIND.PRNG, algo=PRNG_ALGO.XOROSHIRO128SS,
                            seed=int(seed), dtype="float", batch_size=1 << 16)


def _uniform(AT, stream, *shape):
    count = int(np.prod(shape))
    pieces, have = [], 0
    while have < count:
        block = np.asarray(next(stream), dtype=np.float64).reshape(-1)
        pieces.append(block)
        have += block.size
    values = np.concatenate(pieces)[:count] if len(pieces) > 1 else pieces[0][:count]
    return AT.tensor(np.ascontiguousarray(values).reshape(shape))


def sample_solutions(region_batch, seed: int = 0, *, max_steps: int | None = None,
                     backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE):
    """One random leaf per board, batched: a random walk down every
    board's solution tree at once.

    Each step propagates singles; stalled boards place a random candidate
    of their tightest open region (a decision when that region had more
    than one candidate); dead boards restart from the top with fresh
    draws.  Returns ``(solution, decisions, restarts, solved)`` as host
    arrays: ``solution`` (b, n*n) 0/1, ``decisions`` the guesses on the
    final path, ``restarts`` the contradictions met, ``solved`` whether the
    walk finished within ``max_steps``."""
    from src.common.tensors import AbstractTensor as AT

    stream = _uniform_stream(seed)
    region_np = np.asarray(region_batch, dtype=np.float64)
    b = region_np.shape[0]
    n = int(round(region_np.shape[-1] ** 0.5)) if region_np.ndim == 2 else region_np.shape[1]
    with AT.use_backend(backend, device):
        region = AT.tensor(region_np.reshape(b, n, n))
        index_n = AT.arange(n).to_dtype("float")
        one_hot = _f(region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n)))
        zero = region * 0.0
        cell_index = AT.arange(n * n).to_dtype("float").reshape((1, n * n))
        placed = zero
        excluded = zero > 1.0
        decisions = zero[:, 0, 0] * 0.0
        restarts = decisions * 0.0
        solved = decisions > 1.0
        huge = float(n * n + 1)
        for _step in range(max_steps or 4 * n * n):
            placed, excluded, solved_now, dead, _ = propagate(AT, region, one_hot, placed, excluded)
            solved = solved | solved_now
            if bool(solved.all()):
                break
            # Restart the dead boards from the top.
            dead3 = dead.reshape((b, 1, 1))
            placed = AT.where(dead3, zero, placed)
            excluded = AT.where(dead3, zero > 1.0, excluded)
            decisions = AT.where(dead, decisions * 0.0, decisions)
            restarts = restarts + _f(dead)
            # Stalled boards guess in their tightest open region.
            candidate = ~(excluded | (placed > 0.5))
            cand = _f(candidate)
            region_count = (one_hot * cand.reshape((b, n, n, 1))).sum(dim=(1, 2))
            region_has = (one_hot * placed.reshape((b, n, n, 1))).sum(dim=(1, 2)) > 0.5
            tightest = AT.where(region_has | (region_count == 0.0), region_count * 0.0 + huge, region_count).argmin(dim=1)
            tightest = tightest.to_dtype("float").reshape((b, 1, 1))
            in_region = candidate & (region == tightest)
            draw = _uniform(AT, stream, b, n, n)
            score = AT.where(in_region, draw + 1.0, zero)
            pick = (score.reshape((b, n * n)) * -1.0).argmin(dim=1).to_dtype("float").reshape((b, 1))
            pick_mask = (cell_index == pick).reshape((b, n, n)) & in_region
            stalled = ~solved & ~dead & in_region.any(dim=(1, 2))
            tight_count = (region_count * _f(index_n.reshape((1, n)) == tightest.reshape((b, 1)))).sum(dim=1)
            decisions = decisions + _f(stalled & (tight_count > 1.0))
            placed = AT.where(stalled.reshape((b, 1, 1)) & pick_mask, zero + 1.0, placed)
        return (np.asarray(placed.numpy(), dtype=np.float64).reshape(b, n * n),
                np.asarray(decisions.numpy(), dtype=np.float64).reshape(b),
                np.asarray(restarts.numpy(), dtype=np.float64).reshape(b),
                np.asarray(solved.numpy(), dtype=bool).reshape(b))


@dataclass
class TreeStats:
    leaves: int = 0
    decisions: int = 0
    dead_ends: int = 0
    depth: int = 0
    branch_factors: list[int] = field(default_factory=list)
    truncated: bool = False


def solution_tree(region_grid, *, limit_leaves: int = 512,
                  backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE) -> tuple[dict[str, Any], TreeStats]:
    """The full tree of forced guesses of ONE board down to every valid
    arrangement.  Recursive over branches; every mask is AbstractTensor.
    A node is ``{"leaf": columns}``, ``{"dead": True}`` or ``{"region": k,
    "branches": [((r, c), child), ...]}``; dead branches are dropped and
    counted.  Enumeration stops at ``limit_leaves`` (``stats.truncated``)."""
    from src.common.tensors import AbstractTensor as AT

    grid = np.asarray(region_grid, dtype=np.float64)
    n = grid.shape[0]
    stats = TreeStats()
    with AT.use_backend(backend, device):
        region = AT.tensor(grid.reshape(1, n, n))
        index_n = AT.arange(n).to_dtype("float")
        one_hot = _f(region.reshape((1, n, n, 1)) == index_n.reshape((1, 1, 1, n)))
        zero = region * 0.0
        cell_index = AT.arange(n * n).to_dtype("float").reshape((1, n, n))

        def descend(placed, excluded, depth):
            placed, excluded, solved, dead, _ = propagate(AT, region, one_hot, placed, excluded)
            stats.depth = max(stats.depth, depth)
            if bool(solved.all()):
                stats.leaves += 1
                columns = (placed * index_n.reshape((1, 1, n))).sum(dim=2)
                return {"leaf": tuple(int(v) for v in np.asarray(columns.numpy()).reshape(-1))}
            if bool(dead.all()):
                stats.dead_ends += 1
                return {"dead": True}
            candidate = ~(excluded | (placed > 0.5))
            cand = _f(candidate)
            region_count = (one_hot * cand.reshape((1, n, n, 1))).sum(dim=(1, 2))
            region_has = (one_hot * placed.reshape((1, n, n, 1))).sum(dim=(1, 2)) > 0.5
            masked = AT.where(region_has | (region_count == 0.0), region_count * 0.0 + float(n * n + 1), region_count)
            k = int(np.asarray(masked.argmin(dim=1).numpy()).reshape(-1)[0])
            cells = np.argwhere(np.asarray((candidate & (region == float(k))).numpy()).reshape(n, n))
            branches = []
            for r, c in cells:
                if stats.leaves >= limit_leaves:
                    stats.truncated = True
                    break
                guess = _f(cell_index == float(r * n + c))
                child = descend(placed.maximum(guess), excluded, depth + 1)
                if "dead" not in child:
                    branches.append(((int(r), int(c)), child))
            live = [child for _, child in branches]
            if not live:
                stats.dead_ends += 1
                return {"dead": True}
            if len(live) == 1:
                # One live branch is a forced step the deduction set missed,
                # not a guess a solver can get wrong.
                return live[0]
            stats.decisions += 1
            stats.branch_factors.append(len(live))
            return {"region": k, "branches": branches}

        tree = descend(zero, zero > 1.0, 0)
    return tree, stats


def tree_type(stats: TreeStats) -> str:
    """``forced`` for one leaf and no decisions, else ``<decisions>d/<leaves>l``
    (``+`` when the enumeration was truncated)."""
    if stats.decisions == 0 and stats.leaves == 1:
        return "forced"
    return f"{stats.decisions}d/{stats.leaves}l" + ("+" if stats.truncated else "")


def catalog(puzzles, *, seed: int = 0, limit_leaves: int = 512, full_tree_max_n: int = 11,
            backend: str = DEFAULT_BACKEND, device=DEFAULT_DEVICE) -> list[dict[str, Any]]:
    """One typed record per board of a ``PuzzleBatch``: the sampled walk
    (batched) for every board, the full tree up to ``full_tree_max_n``,
    region-size facts, the batched singles facts and contiguity."""
    from meowdoku.puzzle import contiguous_regions, singles_solve
    from src.common.tensors import AbstractTensor as AT

    n, b = puzzles.n, puzzles.batch
    with AT.use_backend(backend, device):
        solved, rounds = singles_solve(AT.tensor(puzzles.region.reshape(b, n, n)),
                                       AT.tensor(puzzles.solution.reshape(b, n, n)))
        solved = np.asarray(solved.numpy(), bool).reshape(b)
        rounds = np.asarray(rounds.numpy()).reshape(b)
        region = AT.tensor(puzzles.region.reshape(b, n, n))
        index_n = AT.arange(n).to_dtype("float")
        sizes = _f(region.reshape((b, n, n, 1)) == index_n.reshape((1, 1, 1, n))).sum(dim=(1, 2))
        sizes = np.asarray(sizes.numpy()).reshape(b, n)
    contiguous = contiguous_regions(puzzles, backend=backend, device=device)
    _, walk_decisions, restarts, walk_solved = sample_solutions(puzzles.region, seed, backend=backend, device=device)
    records = []
    for index in range(b):
        record = {
            "index": index, "n": n,
            "singles_solved": bool(solved[index]), "singles_rounds": int(rounds[index]),
            "contiguous": bool(contiguous[index]),
            "largest_region": int(sizes[index].max()), "smallest_region": int(sizes[index].min()),
            "singletons": int((sizes[index] == 1).sum()),
            "walk_decisions": int(walk_decisions[index]), "walk_restarts": int(restarts[index]),
            "walk_solved": bool(walk_solved[index]),
        }
        if n <= full_tree_max_n:
            _, stats = solution_tree(puzzles.region[index].reshape(n, n), limit_leaves=limit_leaves,
                                     backend=backend, device=device)
            record.update(type=tree_type(stats), decisions=stats.decisions, leaves=stats.leaves,
                          depth=stats.depth, dead_ends=stats.dead_ends,
                          branch_factors=stats.branch_factors, truncated=stats.truncated)
        else:
            record.update(type=("forced" if walk_decisions[index] == 0 else f"~{int(walk_decisions[index])}d"),
                          decisions=int(walk_decisions[index]), leaves=None, depth=None,
                          dead_ends=int(restarts[index]), branch_factors=[], truncated=True)
        records.append(record)
    return records


def type_table(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Counts per type, most common first."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record["type"]] = counts.get(record["type"], 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def select(records: list[dict[str, Any]], *, decisions: int | None = None,
           max_largest_region: int | None = None, contiguous: bool = True) -> list[int]:
    """Indices of the boards matching a wanted type: a guess load
    (``decisions``), a size cap on the largest region, contiguity."""
    picked = []
    for record in records:
        if contiguous and not record["contiguous"]:
            continue
        if decisions is not None and record["decisions"] != decisions:
            continue
        if max_largest_region is not None and record["largest_region"] > max_largest_region:
            continue
        picked.append(record["index"])
    return picked

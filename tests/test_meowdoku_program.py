"""Meowdoku's step law, played eagerly through its dt-system piece shape."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku.program import (  # noqa: E402
    CONTROLS, HEARTS, MeowdokuControls, MeowdokuPiece, initial_columns,
    place_controls, toggle_controls,
)

PLACE_CAT, TOGGLE_X = "place", "toggle"
from meowdoku.puzzle import count_arrangements, generate  # noqa: E402


class Game:
    """Drive the piece the way the dt system does: columns in, _next landed."""

    def __init__(self, n, seed):
        self.puzzle = generate(n, seed)
        self.columns = initial_columns(self.puzzle)
        self.dt = np.ones(self.puzzle.cells)
        self.piece = MeowdokuPiece(self.puzzle.cells)
        spans = {name: self.columns[name] for name in self.piece.argument_names if name != "dt"}
        spans["dt"] = self.dt
        self.piece.instantiate(spans)
        self.residual = None

    def move(self, r, c, action):
        # The Python game hands a control record; its fields are columns.
        controls = (place_controls if action == PLACE_CAT else toggle_controls)(r, c)
        for name, value in controls.columns(self.puzzle.cells).items():
            self.columns[name][...] = value
        args = [self.columns[name] if name != "dt" else self.dt
                for name in self.piece.argument_names]
        out = dict(zip(self.piece.output_names, self.piece(*args)))
        for name, value in out.items():
            if name.endswith("_next"):
                self.columns[name[:-5]][...] = value
        self.residual = float(out["residual"][0])
        assert float(out["dt_limit"][0]) == 1.0
        return out

    @property
    def hearts(self):
        return float(self.columns["hearts"][0])

    def cat(self, r, c):
        return float(self.columns["cat"][r * self.puzzle.n + c])

    def x(self, r, c):
        return float(self.columns["x_mark"][r * self.puzzle.n + c])


def test_generated_puzzle_has_one_arrangement_and_valid_cats():
    puzzle = generate(6, seed=3)
    cols = puzzle.solution_columns()
    assert sorted(cols) == list(range(6))
    assert all(abs(cols[i] - cols[i + 1]) >= 2 for i in range(5))
    grid = puzzle.region_grid()
    assert len({int(grid[r, cols[r]]) for r in range(6)}) == 6
    assert count_arrangements(grid) == 1


def test_play_to_a_win_with_marks_and_two_mistakes():
    game = Game(5, seed=1)
    cols = game.puzzle.solution_columns()
    n = game.puzzle.n

    game.move(0, cols[0], PLACE_CAT)
    assert game.cat(0, cols[0]) == 1.0 and game.hearts == HEARTS and game.residual == 4

    game.move(2, 2, TOGGLE_X)
    assert game.x(2, 2) == 1.0
    game.move(2, 2, TOGGLE_X)
    assert game.x(2, 2) == 0.0

    # Wrong by the hidden solution: not a cat cell, and not touching row 0.
    wrong = next((r, c) for r in range(n) for c in range(n)
                 if cols[r] != c and r > 1)
    game.move(*wrong, PLACE_CAT)
    assert game.cat(*wrong) == 0.0 and game.hearts == HEARTS - 1

    # Wrong by the visible rules: touching the first cat diagonally.
    game.move(1, cols[0] + 1, PLACE_CAT)
    assert game.cat(1, cols[0] + 1) == 0.0 and game.hearts == HEARTS - 2

    # An X under a correct placement is cleared by it.
    game.move(1, cols[1], TOGGLE_X)
    game.move(1, cols[1], PLACE_CAT)
    assert game.cat(1, cols[1]) == 1.0 and game.x(1, cols[1]) == 0.0

    for r in range(2, n):
        game.move(r, cols[r], PLACE_CAT)
    assert game.residual == 0.0 and game.hearts == HEARTS - 2
    np.testing.assert_array_equal(game.columns["cat"], game.puzzle.solution)

    # Solved: the board is inert, a wrong move costs nothing.
    game.move(*wrong, PLACE_CAT)
    assert game.hearts == HEARTS - 2


def test_three_mistakes_lose_and_freeze_the_board():
    game = Game(5, seed=1)
    cols = game.puzzle.solution_columns()
    wrongs = [(r, c) for r in range(5) for c in range(5) if cols[r] != c][:3]
    for cell in wrongs:
        game.move(*cell, PLACE_CAT)
    assert game.hearts == 0.0 and game.residual == 5
    game.move(0, cols[0], PLACE_CAT)
    assert game.cat(0, cols[0]) == 0.0 and game.residual == 5


def test_piece_is_admitted_by_the_dt_system():
    from llvm_dt_system import require_piece, column_names_of, owned_columns

    piece = require_piece(MeowdokuPiece(25), "meowdoku")
    names = column_names_of([piece])
    assert "dt" not in names and "cat" in names and "hearts" in names
    assert set(CONTROLS) <= set(names)            # controls are state columns
    assert owned_columns([piece]) == ("cat", "x_mark", "hearts")


def test_controls_are_declared_on_the_dt_contract():
    """The control record reaches the ProgramABI: each control column's
    PieceState field carries the semantic action a shell fills it from."""
    from llvm_dt_system import column_names_of, dt_system_contract
    import pytest

    piece = MeowdokuPiece(25)
    contract = dt_system_contract("dt_system_over", column_names_of([piece]), 25,
                                  controls=CONTROLS)
    fields = contract.program_abi.receipt()["records"]["PieceState"]["fields"]
    for name, action in CONTROLS.items():
        assert fields[name]["control"] == action
        assert fields[name]["storage"] == "span" and fields[name]["shape"] == [25]
    assert "control" not in fields["cat"]
    with pytest.raises(ValueError, match="not state columns"):
        dt_system_contract("dt_system_over", column_names_of([piece]), 25,
                           controls={"throttle": "move-forward"})


def test_the_record_is_what_the_python_game_hands_over():
    controls = MeowdokuControls(2.0, 3.0, 1.0, 0.0)
    columns = controls.columns(25)
    assert set(columns) == set(CONTROLS)
    assert columns["select_row"][7] == 2.0 and columns["primary_action"].sum() == 25.0
    assert MeowdokuControls().select_row == -1.0      # no cell: never selects


def test_batched_generator_is_unique_contiguous_and_scales():
    import time
    from meowdoku.puzzle import contiguous_regions, generate as make, generate_batch, verify_batch

    for n in (9, 10, 11):
        puzzles = generate_batch(n, 12, seed=n)
        assert verify_batch(puzzles).all() and contiguous_regions(puzzles).all()
        for index in range(puzzles.batch):
            assert count_arrangements(puzzles.puzzle(index).region_grid(), limit=2) == 1, (n, index)
    started = time.perf_counter()
    big = generate_batch(32, 64, seed=0)               # 64 boards of 1024 cells
    elapsed = time.perf_counter() - started
    assert verify_batch(big).all() and contiguous_regions(big).all() and elapsed < 60.0, elapsed
    huge = generate_batch(128, 1, seed=1)              # 16,384 cells
    assert verify_batch(huge).all() and contiguous_regions(huge).all()
    # Deterministic per seed, different across seeds.
    assert np.array_equal(generate_batch(16, 3, seed=5).region, generate_batch(16, 3, seed=5).region)
    assert not np.array_equal(generate_batch(16, 3, seed=5).region, generate_batch(16, 3, seed=6).region)
    # Existing small boards are unchanged: auto still uses the random oracle.
    assert np.array_equal(make(6, seed=3).region, make(6, seed=3, method="random").region)
    assert make(32, seed=0).cells == 1024


def test_simple_points_agree_with_a_flood_fill():
    """Yokoi's 4-connectivity number against the definition: a cell is simple
    when removing it leaves its region 4-connected."""
    from collections import deque
    from meowdoku.puzzle import _simple_points
    from src.common.tensors import AbstractTensor as AT

    rng = np.random.default_rng(0)
    n = 7
    for _trial in range(20):
        # Random partition into 3 labels with some structure (row-major runs).
        region = rng.integers(0, 3, size=(n, n)).astype(np.float64)
        with AT.use_backend("numpy"):
            simple = np.asarray(_simple_points(AT, AT.tensor(region.reshape(1, n, n))).numpy()).reshape(n, n)
        for r in range(n):
            for c in range(n):
                label = region[r, c]
                cells = {(i, j) for i in range(n) for j in range(n) if region[i, j] == label and (i, j) != (r, c)}
                mine = {(i, j) for i in range(n) for j in range(n) if region[i, j] == label}
                # Connected component count of the cell's own 4-component after removal.
                def components(points):
                    seen = set(); count = 0
                    for start in points:
                        if start in seen:
                            continue
                        count += 1; queue = deque([start]); seen.add(start)
                        while queue:
                            i, j = queue.popleft()
                            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                q = (i + di, j + dj)
                                if q in points and q not in seen:
                                    seen.add(q); queue.append(q)
                    return count
                # Local definition: among the 8-neighbours in the same region,
                # the 4-components that touch the cell's 4-neighbours count 1.
                nb = {(r + di, c + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)} & mine
                local = components(nb)
                four = {(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)} & mine
                # Components of nb not touching any 4-neighbour do not count.
                def touching(points):
                    seen = set(); count = 0
                    for start in points:
                        if start in seen:
                            continue
                        comp = {start}; queue = deque([start]); seen.add(start)
                        while queue:
                            i, j = queue.popleft()
                            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                q = (i + di, j + dj)
                                if q in points and q not in seen:
                                    seen.add(q); comp.add(q); queue.append(q)
                        count += bool(comp & four)
                    return count
                assert bool(simple[r, c]) == (touching(nb) == 1), (r, c, region)

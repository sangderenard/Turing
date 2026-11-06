"""Headless behaviour probe for Purrline (no server, no page).

A seeded game advanced through the dt system exactly as the server does
(``session.advance``), the pointer scripted as the page sends it (``fire`` with
``steer_x/y`` on the board point, held for a tick, then released).  Scenarios:

    swap      a scripted swap that makes a match: moves spent, score awarded,
              the board settles with no standing match and a legal move left;
              a swap that makes none is refused (nothing changes, no move spent)
    powers    each power piece forms where it should (a 4-run -> pounce, a 5-run
              -> catnip, an L/T crossing -> hairball) and fires correctly
              (pounce: its row and column; hairball: its 3 x 3; catnip: every
              tile of the swapped kind)
    cascades  a board set to fall into a second match scores the second step at
              the higher multiplier
    play      hundreds of random legal swaps; after every one the board settles
              with NO standing match and at least one legal move; a dead board is
              reshuffled
    levels    reaching the score target starts the next level; running out of
              moves ends the game with the score as its result

    python examples/meowdoku/purrline_probe.py [seed]

The lane is MEOWDOKU_LAW as everywhere.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import purrline as pl  # noqa: E402

STEP = 0.05
W, H, N = pl.W, pl.H, pl.N


def fresh(seed: int) -> pl.Purrline:
    game = pl.Purrline(seed=seed)
    game.session.control({})
    game.session.advance(1.0 / 60.0)             # round zero instantiates the state
    settle(game)
    return game


def board(game: pl.Purrline) -> np.ndarray:
    return game.session.column("kind").astype(int).copy()


def specs(game: pl.Purrline) -> np.ndarray:
    return game.session.column("spec").astype(int).copy()


def put(game: pl.Purrline, rows: list[list[int]], spec: dict[tuple[int, int], int] | None = None, idle: bool = True) -> None:
    """Set the board (rows of kinds); every tile gets a fresh identity; the game is then let to settle
    (the law checks the board once: no match, a legal swap, the player's turn)."""
    st = game.session.state
    st.kind[:] = np.asarray(rows, dtype=float).reshape(-1)
    st.spec[:] = 0.0
    for (r, c), v in (spec or {}).items():
        st.spec[r * W + c] = float(v)
    for name in ("ox", "oy", "flag", "act", "wait", "sel", "combo", "quiet"):
        getattr(st, name)[:] = 0.0 if name != "sel" else -1.0
    st.mode[:] = float(pl.RESOLVE)
    st.tkind[:] = st.swa[:] = st.swb[:] = -1.0
    st.tid[:] = 100.0 + np.arange(N) + st.counter[0]
    st.counter[:] = st.counter[0] + 100.0
    st.moves[:] = float(pl.MOVES_PER_LEVEL)
    st.over[:] = 0.0
    if idle:
        settle(game)


def busy(game: pl.Purrline) -> bool:
    s = game.session
    return bool(np.abs(s.column("ox")).max() > 1e-9 or np.abs(s.column("oy")).max() > 1e-9 or s.column("wait")[0] > 1e-9)


def settle(game: pl.Purrline, limit: int = 2000) -> int:
    """Land animation time until the board is idle again (or the game is over)."""
    s = game.session
    s.control({"fire": 0.0, "steer_x": -1.0, "steer_y": -1.0})
    for i in range(limit):
        s.advance(STEP)
        if (s.column("mode")[0] == pl.IDLE or s.column("over")[0] > 0.5) and not busy(game):
            return i
    raise AssertionError("the board never settled")


def tap(game: pl.Purrline, cell: int) -> None:
    s = game.session
    s.control({"fire": 1.0, "steer_x": cell % W + 0.5, "steer_y": cell // W + 0.5})
    s.advance(0.03)
    s.control({"fire": 0.0, "steer_x": -1.0, "steer_y": -1.0})
    s.advance(0.03)


def swap(game: pl.Purrline, a: int, b: int, drag: bool = False) -> None:
    """The player's swap: tap a then b (or press a and drag to b); then let it resolve."""
    s = game.session
    if drag:
        s.control({"fire": 1.0, "steer_x": a % W + 0.5, "steer_y": a // W + 0.5})
        s.advance(0.03)
        s.control({"fire": 1.0, "steer_x": b % W + 0.5, "steer_y": b // W + 0.5})
        s.advance(0.03)
        s.control({"fire": 0.0, "steer_x": -1.0, "steer_y": -1.0})
        s.advance(0.03)
    else:
        tap(game, a)
        tap(game, b)
    settle(game)


def step_until(game: pl.Purrline, predicate, limit: int = 400) -> bool:
    s = game.session
    s.control({"fire": 0.0, "steer_x": -1.0, "steer_y": -1.0})
    for _ in range(limit):
        if predicate():
            return True
        s.advance(0.02)
    return False


def standing_matches(game: pl.Purrline) -> bool:
    return bool(pl.runs_of(board(game), specs(game)).any())


def a_legal_swap(game: pl.Purrline, rng: np.random.Generator) -> tuple[int, int]:
    moves = pl.legal_swaps(board(game), specs(game))
    return moves[int(rng.integers(0, len(moves)))]


# -------------------------------------------------------------- scenarios

def scenario_swap(seed: int) -> dict:
    game = fresh(seed)
    s = game.session
    kinds0 = board(game)
    assert not standing_matches(game)
    moves = pl.legal_swaps(kinds0)
    a, b = moves[0]
    score0, left0 = float(s.column("score")[0]), float(s.column("moves")[0])
    swap(game, a, b)
    out = {"swap": (a, b), "score_gain": float(s.column("score")[0]) - score0,
           "moves_spent": left0 - float(s.column("moves")[0]), "standing": standing_matches(game),
           "legal_left": len(pl.legal_swaps(board(game), specs(game)))}
    # a refused swap
    game = fresh(seed)
    s = game.session
    kinds0 = board(game)
    legal = {tuple(sorted(m)) for m in pl.legal_swaps(kinds0)}
    refused = next((a, a + 1) for a in range(N) if a % W < W - 1 and (a, a + 1) not in legal)
    score0 = float(s.column("score")[0])
    swap(game, *refused)
    out["refused"] = refused
    out["refused_unchanged"] = bool((board(game) == kinds0).all())
    out["refused_moves_spent"] = float(pl.MOVES_PER_LEVEL) - float(s.column("moves")[0])
    out["refused_score"] = float(s.column("score")[0]) - score0
    # the same swap by drag
    game = fresh(seed)
    a, b = pl.legal_swaps(board(game))[0]
    swap(game, a, b, drag=True)
    out["drag_moves_spent"] = float(pl.MOVES_PER_LEVEL) - float(game.session.column("moves")[0])
    return out


def scenario_powers() -> dict:
    """Constructed boards: kinds 0..5, a tile of kind 9 is never used (rows are filled with a
    no-match background of kinds 3, 4, 5 in a diagonal pattern)."""
    def background():
        return [[3 + (r + 2 * c) % 3 for c in range(W)] for r in range(H)]

    result = {}
    # --- pounce: row 4 reads 0 0 . 0 with a 0 below the gap: swap pulls it in -> a 4-run
    rows = background()
    for c in (0, 1, 3):
        rows[4][c] = 0
    rows[5][2] = 0
    game = fresh(1)
    put(game, rows)
    s = game.session
    # run the swap but stop at the flash, where the new piece stands
    tap(game, 5 * W + 2)
    tap(game, 4 * W + 2)
    step_until(game, lambda: s.column("mode")[0] == pl.POP)
    sp = specs(game).reshape(H, W)
    result["pounce_formed_at"] = [tuple(int(v) for v in np.argwhere(sp == pl.PAW)[0])] if (sp == pl.PAW).any() else None
    result["pounce_swapped_cell"] = (4, 2)
    result["pounce_count"] = int((sp == pl.PAW).sum())
    result["pounce_flashing"] = int(s.column("flag").sum())
    settle(game)
    result["pounce_kept"] = int((specs(game) == pl.PAW).sum())
    # --- hairball: a T: row 4 has 0 0 . 0, column 2 has 0 at rows 5, 6 (swap (4,2) <-> (3,2) pulls a 0 down
    rows = background()
    for c in (0, 1, 3):
        rows[4][c] = 0
    rows[3][2] = 0
    rows[5][2] = 0
    rows[6][2] = 0
    # (4,2) is a gap: after pulling the 0 from (3,2) down, row 4 = 0 0 0 0 and column 2 = rows 4,5,6 -> crossing
    game = fresh(1)
    s = game.session
    put(game, rows)
    tap(game, 3 * W + 2)
    tap(game, 4 * W + 2)
    step_until(game, lambda: s.column("mode")[0] == pl.POP)
    sp = specs(game).reshape(H, W)
    result["hairball_formed"] = [tuple(int(v) for v in x) for x in np.argwhere(sp == pl.BALL)]
    result["hairball_all_specs"] = {int(k): int((sp == k).sum()) for k in (1, 2, 3)}
    settle(game)
    # --- catnip: a 5-run
    rows = background()
    for c in (0, 1, 3, 4):
        rows[4][c] = 0
    rows[5][2] = 0
    game = fresh(1)
    s = game.session
    put(game, rows)
    tap(game, 5 * W + 2)
    tap(game, 4 * W + 2)
    step_until(game, lambda: s.column("mode")[0] == pl.POP)
    sp = specs(game).reshape(H, W)
    result["catnip_formed"] = [tuple(int(v) for v in x) for x in np.argwhere(sp == pl.NIP)]
    settle(game)
    return result


def fires(kind_rows, spec, trigger_swap):
    """Put a board with power pieces, swap, and report which cells were cleared at the first flash."""
    game = fresh(1)
    put(game, kind_rows, spec)
    s = game.session
    tap(game, trigger_swap[0])
    tap(game, trigger_swap[1])
    step_until(game, lambda: s.column("mode")[0] == pl.POP)
    flagged = s.column("flag").reshape(H, W) > 0.5
    score = float(s.column("score")[0])
    settle(game)
    return flagged, score, game


def scenario_fires() -> dict:
    def background():
        return [[3 + (r + 2 * c) % 3 for c in range(W)] for r in range(H)]

    out = {}
    # pounce at (3,3) of kind 0, in a vertical match of column 3 rows 1..3 after pulling a 0 from (2,4) into (2,3)... use a
    # horizontal match: row 5 = 0 0 _ with 0 at (4,2): swap (4,2)<->(5,2) -> run (5,0..2) holding the pounce at (5,0)
    rows = background()
    rows[5][0] = rows[5][1] = 0
    rows[4][2] = 0
    flagged, score, _ = fires(rows, {(5, 0): pl.PAW}, (4 * W + 2, 5 * W + 2))
    expect = np.zeros((H, W), dtype=bool)
    expect[5, :] = True
    expect[:, 0] = True
    out["pounce_row_and_column"] = bool((flagged == expect).all())
    out["pounce_cleared"] = int(flagged.sum())
    # hairball at (5,1) in the same kind of match: bursts rows 4..6 x columns 0..2
    flagged, score, _ = fires(rows, {(5, 1): pl.BALL}, (4 * W + 2, 5 * W + 2))
    expect = np.zeros((H, W), dtype=bool)
    expect[4:7, 0:3] = True
    out["hairball_3x3"] = bool((flagged == expect).all())
    out["hairball_cleared"] = int(flagged.sum())
    # catnip swapped with a tile of kind 4: every 4 goes, and the catnip
    rows = background()
    four = int((np.asarray(rows) == 4).sum())
    flagged, score, game = fires(rows, {(2, 2): pl.NIP}, (2 * W + 2, 2 * W + 3))
    kind = np.asarray(rows)
    target = kind[2, 3]
    expect = (kind == target)
    # the swap puts the kind-5 tile on (2,2) and the catnip on (2,3): every kind-5 tile and the catnip clear
    expect[2, 2] = True
    expect[2, 3] = True
    out["catnip_target_kind"] = int(target)
    out["catnip_clears_kind"] = bool((flagged == expect).all())
    out["catnip_cleared"] = int(flagged.sum())
    # two catnips: the board
    flagged, score, game = fires(background(), {(2, 2): pl.NIP, (2, 3): pl.NIP}, (2 * W + 2, 2 * W + 3))
    out["double_catnip_clears_all"] = bool(flagged.all())
    return out


def scenario_cascade() -> dict:
    """Clear a 3-run whose removal drops two 0s onto a third: a second step at the higher multiplier."""
    def background():
        return [[3 + (r + 2 * c) % 3 for c in range(W)] for r in range(H)]
    rows = background()
    rows[6][0] = rows[6][1] = 1
    rows[7][2] = 1                        # swap (6,2) <-> (7,2): row 6 = 1 1 1
    rows[4][0] = rows[5][0] = rows[7][0] = 0   # when row 6 goes, column 0 falls: 0 0 0 on rows 5..7
    assert not pl.runs_of(np.asarray(rows)).any()
    game = fresh(1)
    s = game.session
    put(game, rows)
    score0 = float(s.column("score")[0])
    tap(game, 6 * W + 2)
    tap(game, 7 * W + 2)
    step_until(game, lambda: s.column("mode")[0] == pl.POP)
    after_first = float(s.column("score")[0]) - score0
    combo1 = float(s.column("combo")[0])
    settle(game)
    return {"first_step_points": after_first, "combo_after_first": combo1, "total": float(s.column("score")[0]) - score0,
            "cascades_seen": float(s.column("bestcombo")[0]), "standing": standing_matches(game)}


def scenario_play(seed: int, moves: int = 300) -> dict:
    game = fresh(seed)
    s = game.session
    rng = np.random.default_rng(seed)
    standing, no_legal, made, deepest, clashes = 0, 0, 0, 0, 0
    levels = 1
    for i in range(moves):
        if s.column("over")[0] > 0.5:
            break
        a, b = a_legal_swap(game, rng)
        before = float(s.column("score")[0])
        swap(game, a, b)
        made += 1
        deepest = max(deepest, int(s.column("bestcombo")[0]))
        if standing_matches(game):
            standing += 1
        if len(set(s.column("tid").tolist())) != N:
            clashes += 1
        if s.column("over")[0] < 0.5 and not pl.legal_swaps(board(game), specs(game)) and not (specs(game) == pl.NIP).any():
            no_legal += 1
        levels = int(s.column("level")[0])
    return {"swaps_made": made, "tile_identity_clashes": clashes, "standing_matches_after_settle": standing, "settled_without_legal_move": no_legal,
            "score": float(s.column("score")[0]), "level": levels, "deepest_cascade": deepest,
            "shuffles": float(s.column("shuffles")[0]), "over": float(s.column("over")[0]),
            "tiles_cleared": float(s.column("cleared")[0]), "powers_fired": float(s.column("fired")[0])}


def dead_board() -> list[list[int]]:
    """A board with no legal swap: kinds laid out so no swap can line up three."""
    for a in range(1, 6):
        for b in range(1, 6):
            rows = [[(a * r + b * c) % pl.KINDS for c in range(W)] for r in range(H)]
            flat = np.asarray(rows).reshape(-1)
            if not pl.runs_of(flat).any() and not pl.legal_swaps(flat):
                return rows
    raise AssertionError("no dead pattern found")


def scenario_shuffle() -> dict:
    game = fresh(5)
    s = game.session
    rows = dead_board()
    n0 = float(s.column("shuffles")[0])
    put(game, rows)
    return {"shuffles": float(s.column("shuffles")[0]) - n0, "legal_after": len(pl.legal_swaps(board(game), specs(game))),
            "standing": standing_matches(game), "board_changed": bool((board(game) != np.asarray(rows).reshape(-1)).any())}


def scenario_scan(count: int = 40) -> dict:
    """The law's match scan against the plain reference on random (unsettled) boards: the cells it
    flashes, plus the cells that turn into power pieces, are exactly the cells in a run of three."""
    rng = np.random.default_rng(11)
    wrong, with_runs = 0, 0
    for i in range(count):
        game = fresh(7)
        s = game.session
        rows = rng.integers(0, 4, size=(H, W))                 # four kinds: plenty of runs
        put(game, rows.tolist(), idle=False)
        flat = rows.reshape(-1)
        expect = pl.runs_of(flat)
        if not expect.any():
            continue
        with_runs += 1
        step_until(game, lambda: s.column("mode")[0] == pl.POP)
        flagged = s.column("flag") > 0.5
        made = specs(game) > 0
        if not ((flagged | made) == expect).all() or (flagged & made).any():
            wrong += 1
    return {"boards_with_runs": with_runs, "scan_disagreements": wrong}


def scenario_noshuffle(count: int = 30) -> dict:
    """Freshly dealt boards hold a legal swap: the law must not reshuffle them (its legal-move tables
    against the brute-force check)."""
    spurious = 0
    for seed in range(count):
        game = fresh(seed)
        before = board(game).copy()
        if float(game.session.column("shuffles")[0]) > 0 or not (board(game) == before).all():
            spurious += 1
    return {"boards": count, "spurious_shuffles": spurious}


def scenario_deterministic(seed: int = 4, moves: int = 12) -> dict:
    """Refill is deterministic from the seed: the same script on the same deal gives the same game."""
    runs = []
    for _ in range(2):
        game = fresh(seed)
        rng = np.random.default_rng(0)
        for _ in range(moves):
            if game.session.column("over")[0] > 0.5:
                break
            a, b = a_legal_swap(game, rng)
            swap(game, a, b)
        runs.append((board(game).tolist(), float(game.session.column("score")[0])))
    return {"same_board": runs[0][0] == runs[1][0], "same_score": runs[0][1] == runs[1][1], "score": runs[0][1]}


def scenario_levels() -> dict:
    game = fresh(2)
    s = game.session
    s.state.lscore[:] = s.state.target[0] - 5.0          # one match short of the target
    rng = np.random.default_rng(0)
    level0, moves0 = float(s.column("level")[0]), float(s.column("moves")[0])
    a, b = a_legal_swap(game, rng)
    swap(game, a, b)
    out = {"level": (level0, float(s.column("level")[0])), "moves_reset": float(s.column("moves")[0]),
           "lscore_reset_below": float(s.column("lscore")[0]), "target": float(s.column("target")[0]),
           "result_while_playing": game.result()}
    # run out of moves
    game = fresh(2)
    s = game.session
    s.state.moves[:] = 1.0
    s.state.target[:] = 1.0e9
    a, b = a_legal_swap(game, rng)
    swap(game, a, b)
    out["over"] = float(s.column("over")[0])
    out["result"] = game.result()
    st = game.state()
    out["state_over"] = st["over"]
    return out


def main(argv: list[str]) -> int:
    seed = int(argv[1]) if len(argv) > 1 else 3
    failed = []

    def run(name, fn, ok):
        started = time.perf_counter()
        result = fn()
        print(f"{name:9s} {result}  [{time.perf_counter() - started:.1f}s wall]", flush=True)
        good = bool(ok(result))
        print(f"{name:9s} {'PASS' if good else 'FAIL'}", flush=True)
        if not good:
            failed.append(name)

    run("swap", lambda: scenario_swap(seed),
        lambda r: r["score_gain"] > 0 and r["moves_spent"] == 1 and not r["standing"] and r["legal_left"] > 0
        and r["refused_unchanged"] and r["refused_moves_spent"] == 0 and r["refused_score"] == 0 and r["drag_moves_spent"] == 1)
    run("powers", scenario_powers,
        lambda r: r["pounce_formed_at"] == [(4, 2)] and r["pounce_count"] == 1 and r["hairball_formed"] == [(4, 2)]
        and r["catnip_formed"] == [(4, 2)] and r["pounce_kept"] >= 1)
    run("fires", scenario_fires,
        lambda r: r["pounce_row_and_column"] and r["hairball_3x3"] and r["catnip_clears_kind"] and r["double_catnip_clears_all"])
    run("cascade", scenario_cascade,
        lambda r: r["cascades_seen"] >= 2 and not r["standing"] and r["total"] > r["first_step_points"])
    run("shuffle", scenario_shuffle,
        lambda r: r["shuffles"] >= 1 and r["legal_after"] > 0 and not r["standing"] and r["board_changed"])
    run("levels", scenario_levels,
        lambda r: r["level"] == (1.0, 2.0) and r["moves_reset"] == pl.MOVES_PER_LEVEL and r["over"] == 1.0
        and r["result"] is not None and r["result_while_playing"] is None and r["state_over"])
    run("scan", scenario_scan, lambda r: r["boards_with_runs"] > 20 and r["scan_disagreements"] == 0)
    run("noshuffle", scenario_noshuffle, lambda r: r["spurious_shuffles"] == 0)
    run("determinism", scenario_deterministic, lambda r: r["same_board"] and r["same_score"] and r["score"] > 0)
    run("play", lambda: scenario_play(seed),
        lambda r: r["standing_matches_after_settle"] == 0 and r["settled_without_legal_move"] == 0 and r["deepest_cascade"] >= 2
        and r["tile_identity_clashes"] == 0
        and r["powers_fired"] > 0)
    print(f"purrline {'FAIL: ' + ', '.join(failed) if failed else 'PASS'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""Purrline: the board, the match-three law, the power pieces and the game's contract.

The law is run through the NumPy product (the translatable lane); the native
lane is checked against it, bit for bit, by the parity script (see the
examples/meowdoku/purrline_probe.py header) and the scenarios are the probe's.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

from meowdoku import purrline as pl  # noqa: E402
from meowdoku import purrline_probe as probe  # noqa: E402
from meowdoku.realtime_socket import SnapshotEncoder  # noqa: E402


@pytest.mark.parametrize("seed", range(12))
def test_a_deal_is_settled_and_playable(seed):
    kind = pl.deal(seed)
    assert kind.shape == (pl.N,) and kind.min() >= 0 and kind.max() < pl.KINDS
    assert not pl.runs_of(kind).any()
    assert pl.legal_swaps(kind)


def test_the_reference_sees_runs_and_swaps():
    rows = np.array([[3 + (r + 2 * c) % 3 for c in range(pl.W)] for r in range(pl.H)])
    assert not pl.runs_of(rows).any()
    rows[4, 0] = rows[4, 1] = rows[4, 3] = 0
    rows[5, 2] = 0
    assert [(34, 42)] == [m for m in pl.legal_swaps(rows) if m == (34, 42)]       # pulling (5,2) up makes four
    rows[4, 2] = 0
    assert pl.runs_of(rows).reshape(pl.H, pl.W)[4, :4].all()


def test_a_swap_that_matches_scores_and_a_swap_that_does_not_is_refused():
    r = probe.scenario_swap(3)
    assert r["score_gain"] > 0 and r["moves_spent"] == 1 and not r["standing"] and r["legal_left"] > 0
    assert r["refused_unchanged"] and r["refused_moves_spent"] == 0 and r["refused_score"] == 0
    assert r["drag_moves_spent"] == 1                       # a drag from one tile onto its neighbour swaps too


def test_power_pieces_form_where_the_match_was_made():
    r = probe.scenario_powers()
    assert r["pounce_formed_at"] == [(4, 2)] and r["pounce_count"] == 1       # a 4-run: on the swapped tile
    assert r["hairball_formed"] == [(4, 2)]                                   # a crossing of two runs
    assert r["catnip_formed"] == [(4, 2)]                                     # a 5-run
    assert r["pounce_kept"] >= 1                                              # the piece survives the clear


def test_power_pieces_fire():
    r = probe.scenario_fires()
    assert r["pounce_row_and_column"] and r["pounce_cleared"] == 15            # row 5 and column 0
    assert r["hairball_3x3"] and r["hairball_cleared"] == 9
    assert r["catnip_clears_kind"]                                            # every tile of the swapped kind
    assert r["double_catnip_clears_all"]


def test_cascades_score_at_a_rising_multiplier():
    r = probe.scenario_cascade()
    assert r["cascades_seen"] >= 2 and not r["standing"]
    assert r["first_step_points"] == 30.0 and r["total"] == 30.0 + 60.0       # 3 tiles x 10, then 3 x 10 x 2


def test_the_scan_agrees_with_the_plain_reference_on_random_boards():
    r = probe.scenario_scan(20)
    assert r["boards_with_runs"] >= 15 and r["scan_disagreements"] == 0


def test_a_board_with_no_legal_swap_is_reshuffled_and_a_playable_one_is_not():
    r = probe.scenario_shuffle()
    assert r["shuffles"] >= 1 and r["legal_after"] > 0 and not r["standing"] and r["board_changed"]
    assert probe.scenario_noshuffle(10)["spurious_shuffles"] == 0


def test_refill_is_deterministic_from_the_seed():
    r = probe.scenario_deterministic(moves=6)
    assert r["same_board"] and r["same_score"]


def test_levels_and_the_end_of_the_game():
    r = probe.scenario_levels()
    assert r["level"] == (1.0, 2.0) and r["moves_reset"] == pl.MOVES_PER_LEVEL and r["target"] > pl.TARGET0
    assert r["result_while_playing"] is None
    assert r["over"] == 1.0 and r["result"]["score"] > 0 and r["state_over"]


def test_no_board_ever_settles_with_a_standing_match_or_without_a_move():
    r = probe.scenario_play(5, moves=60)
    assert r["swaps_made"] > 10
    assert r["standing_matches_after_settle"] == 0 and r["settled_without_legal_move"] == 0
    assert r["tile_identity_clashes"] == 0 and r["powers_fired"] >= 0


def test_the_game_is_an_entity_game_the_page_and_socket_can_carry():
    game = pl.Purrline(seed=2)
    game.control({})
    game.tick()
    st = game.state()
    assert st["board"] == "free" and st["controls"] == ["steer_x", "steer_y", "fire"]
    assert st["width"] == pl.W and st["height"] == pl.H
    ids = [e["id"] for e in st["entities"]]
    assert len(ids) == len(set(ids)) == 2 * pl.N            # the checkerboard and one entity per tile
    lanes = [int(i.split(":")[0]) for i in ids]
    assert max(lanes) < 65535
    meshes = {e["mesh"] for e in st["entities"]}
    assert meshes <= set(pl.EMOJI) | {"ground"}
    for e in st["entities"]:
        assert e["color"] in st["palette"]
    encoder = SnapshotEncoder()
    binary, meta = encoder.encode(game, st)
    assert len(binary) > 0 and meta is not None
    # selecting a tile shows the cursor
    game.control({"fire": 1.0, "steer_x": 2.5, "steer_y": 3.5})
    probe.settle(game)
    game.session.control({"fire": 1.0, "steer_x": 2.5, "steer_y": 3.5})
    game.session.advance(0.03)
    assert game.session.column("sel")[0] == 3 * pl.W + 2
    assert any(e["mesh"] == "pl_cursor" for e in game.state()["entities"])


def test_the_page_sends_the_pointer_on_both_axes_for_a_game_that_declares_it():
    js = (ROOT / "examples" / "meowdoku" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'includes("steer_y")' in js


# ------------------------------------------------------------------ sand mode (finite sky stock)

def _sand(seed=3, stock=pl.STOCK0):
    game = pl.Purrline(seed=seed, sand=True, stock=stock)
    game.session.control({})
    game.session.advance(1.0 / 60.0)
    probe.settle(game)
    return game


def _press(game, a, b):
    """The player's tap a then tap b, without letting the board settle."""
    s = game.session
    for cell in (a, b):
        s.control({"fire": 1.0, "steer_x": cell % pl.W + 0.5, "steer_y": cell // pl.W + 0.5})
        s.advance(0.03)
        s.control({"fire": 0.0, "steer_x": -1.0, "steer_y": -1.0})
        s.advance(0.03)


def _holes_only_above(game):
    kind = game.session.column("kind").reshape(pl.H, pl.W)
    for c in range(pl.W):
        seen_hole = False
        for r in range(pl.H - 1, -1, -1):             # bottom up: once a hole, never a tile again
            if kind[r, c] < 0:
                seen_hole = True
            else:
                assert not seen_hole, f"tile above a hole in column {c}"


def test_default_mode_is_unchanged_by_the_sand_columns():
    game = pl.Purrline(seed=2)
    game.control({})
    game.tick()
    assert not game.modes()["sand"] and game.state()["purrline"]["sand"] is False
    assert game.session.column("stock")[0] == 0.0 and game.session.column("finite")[0] == 0.0
    assert game.session.column("sky_on").max() == 0.0 and (game.session.column("kind") >= 0).all()


def test_the_finite_stock_is_consumed_and_the_rest_stay_holes():
    game = _sand(stock=2)
    rng = np.random.default_rng(1)
    s = game.session
    for _ in range(40):
        if s.column("stock")[0] == 0.0 or s.column("over")[0] > 0.5:
            break
        a, b = probe.a_legal_swap(game, rng)
        probe.swap(game, a, b)
    assert s.column("stock")[0] == 0.0                          # both items were spent, none refilled after
    cleared = int(s.column("cleared")[0])
    holes = int((s.column("kind") < 0).sum())
    assert holes == cleared - 2 and holes >= 1                  # every cleared tile but the two replaced is a hole
    assert (s.column("sky_on") < 0.5).all()                     # nothing left in the air
    _holes_only_above(game)
    # more play spends nothing: the holes only grow
    if s.column("over")[0] < 0.5:
        a, b = probe.a_legal_swap(game, rng)
        probe.swap(game, a, b)
    assert s.column("stock")[0] == 0.0 and int((s.column("kind") < 0).sum()) >= holes


def test_stock_accounting_while_it_lasts():
    game = _sand(stock=pl.STOCK0)
    rng = np.random.default_rng(2)
    s = game.session
    for _ in range(3):
        a, b = probe.a_legal_swap(game, rng)
        probe.swap(game, a, b)
        spent = pl.STOCK0 - int(s.column("stock")[0])
        assert spent == int(s.column("cleared")[0])              # while stock lasts every cleared tile is replaced
        assert (s.column("kind") >= 0).all()                     # ... and no hole is left


def test_a_falling_item_lands_in_its_column_at_a_deterministic_time():
    def run():
        game = _sand(seed=5)
        s = game.session
        a, b = probe.a_legal_swap(game, np.random.default_rng(3))
        _press(game, a, b)
        born, landed = {}, {}
        for _ in range(3000):
            on = s.column("sky_on") > 0.5
            clock = float(s.column("clock")[0])
            for k in np.flatnonzero(on):
                born.setdefault(int(k), (clock, float(s.column("sky_y")[k]), float(s.column("sky_kind")[k]),
                                         float(s.column("sky_tid")[k])))
            for k in born:
                if k not in landed and not on[k]:
                    landed[k] = (clock, float(s.column("kind")[k]), float(s.column("tid")[k]))
            if born and not on.any():
                break
            s.advance(0.01)
        return born, landed, s.column("kind").copy()

    born, landed, board = run()
    assert born and set(born) == set(landed)
    for k, (t0, y0, kind0, tid0) in born.items():
        t1, kind1, tid1 = landed[k]
        assert (kind1, tid1) == (kind0, tid0)                    # it became the tile of ITS cell (its column)
        expected = (2.0 * ((k // pl.W + 0.5) - y0) / pl.SKY_G) ** 0.5
        assert abs((t1 - t0) - expected) <= 0.04, (k, t1 - t0, expected)
    born2, landed2, board2 = run()
    assert born == born2 and landed == landed2 and (board == board2).all()      # deterministic


def test_the_occupancy_mask_and_the_falling_items_are_in_the_state():
    game = _sand(stock=3)
    rng = np.random.default_rng(4)
    s = game.session
    seen_falling = False
    for _ in range(6):
        if s.column("over")[0] > 0.5:
            break
        a, b = probe.a_legal_swap(game, rng)
        _press(game, a, b)
        for _ in range(2000):
            st = game.state()["purrline"]
            kind = s.column("kind")
            assert st["occupancy"] == [1 if v >= 0 else 0 for v in kind]             # the mask is the board
            assert (st["rows"], st["cols"], st["cell"], st["origin"]) == (pl.H, pl.W, 1.0, [0.0, 0.0])
            ents = {e["id"]: e for e in game.state()["entities"]}
            for item in st["falling"]:
                seen_falling = True
                assert st["occupancy"][item["row"] * pl.W + item["col"]] == 0        # its cell is a hole until it lands
                assert item["y"] < item["row"] + 0.5 and item["vy"] >= 0.0 and item["plane_z"] == pl.PLANE_Z
                assert item["x"] == item["col"] + 0.5 and item["size"] == pl.TILE_SIZE
                assert ents[item["id"]]["z"] == pl.PLANE_Z and ents[item["id"]]["y"] == item["y"]
            if not st["falling"] and not probe.busy(game) and s.column("mode")[0] == pl.IDLE:
                break
            s.advance(0.02)
    assert seen_falling
    binary, meta = SnapshotEncoder().encode(game, game.state())
    assert len(binary) > 0 and meta["purrline"]["stock"] == int(s.column("stock")[0])
    assert all(e["mesh"] in set(pl.EMOJI) | {"ground"} for e in game.state()["entities"])


def test_the_sand_game_ends_with_a_score():
    game = _sand(seed=7, stock=12)
    rng = np.random.default_rng(5)
    s = game.session
    for _ in range(300):
        if s.column("over")[0] > 0.5:
            break
        a, b = probe.a_legal_swap(game, rng)
        probe.swap(game, a, b)
    assert s.column("over")[0] == 1.0
    assert s.column("stock")[0] == 0.0
    assert not pl.legal_swaps(s.column("kind").astype(int), s.column("spec").astype(int))     # truly no move left
    result = game.result()
    assert result is not None and result["score"] > 0 and game.state()["over"]
    assert "NO MOVES" in game.describe()


def test_the_sand_mode_toggle_deals_a_new_game():
    game = pl.Purrline(seed=2)
    game.control({})
    game.tick()
    game.set_mode("sand", True)
    game.control({})
    game.tick()
    st = game.state()
    assert st["modes"]["on"] == {"autoshuffle": True, "sand": True} and st["purrline"]["stock"] == pl.STOCK0
    with pytest.raises(ValueError):
        game.set_mode("nope", True)


# ------------------------------------------------------------------ auto shuffle

def test_a_dead_board_is_reshuffled_by_permuting_its_kinds_to_a_playable_board():
    game = probe.fresh(5)
    s = game.session
    assert game.modes()["autoshuffle"] is True                        # on by default
    rows = np.asarray(probe.dead_board()).reshape(-1)
    probe.put(game, rows.reshape(pl.H, pl.W).tolist())
    after = probe.board(game)
    assert s.column("shuffles")[0] >= 1 and s.column("over")[0] == 0.0
    assert sorted(after) == sorted(rows) and (after != rows).any()    # the same items, permuted
    assert pl.legal_swaps(after, probe.specs(game)) and not probe.standing_matches(game)
    # deterministic from the seed and the shuffle count: the same game reshuffles the same way
    other = probe.fresh(5)
    probe.put(other, rows.reshape(pl.H, pl.W).tolist())
    assert (probe.board(other) == after).all()


def test_with_auto_shuffle_off_a_dead_board_ends_the_game():
    game = probe.fresh(5)
    game.set_mode("autoshuffle", False)
    s = game.session
    assert game.modes()["autoshuffle"] is False and game.state()["modes"]["on"]["autoshuffle"] is False
    rows = probe.dead_board()
    probe.put(game, rows, idle=False)
    probe.settle(game)
    assert s.column("over")[0] == 1.0 and s.column("shuffles")[0] == 0.0
    assert (probe.board(game) == np.asarray(rows).reshape(-1)).all()
    assert game.result() is not None


def test_sand_mode_shuffle_keeps_holes_and_spends_no_stock():
    game = _sand(seed=5, stock=7)
    s = game.session
    rows = probe.dead_board()
    holes = (0, 1, 2)
    for c in holes:
        rows[0][c] = -1
    flat = np.asarray(rows).reshape(-1)
    if pl.legal_swaps(flat) or pl.runs_of(flat).any():
        pytest.skip("this dead pattern is not dead with holes")
    probe.put(game, rows)
    after = probe.board(game)
    assert s.column("shuffles")[0] >= 1 and s.column("stock")[0] == 7.0
    assert [int(after[c]) for c in holes] == [-1, -1, -1] and (after < 0).sum() == 3
    assert sorted(after) == sorted(flat) and pl.legal_swaps(after, probe.specs(game))
    assert not pl.runs_of(after).any()


def test_the_page_toggles_are_declared():
    game = pl.Purrline(seed=2)
    game.control({})
    game.tick()
    assert game.state()["modes"]["labels"] == {"autoshuffle": "Auto shuffle", "sand": "Finite sky stock"}

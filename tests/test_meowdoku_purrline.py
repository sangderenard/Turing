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

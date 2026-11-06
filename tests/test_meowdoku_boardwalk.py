"""Boardwalk: the track, the kinds' choice networks and the turn law.

The law is run through the NumPy product (the translatable lane); the native
lane is checked against it by the parity script and examples/meowdoku/boardwalk_probe.py.
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

from meowdoku import boardwalk as bw  # noqa: E402
from meowdoku import boardwalk_probe as probe  # noqa: E402
from meowdoku import walkers  # noqa: E402


@pytest.mark.parametrize("seed", range(8))
def test_the_track_is_one_connected_loop_with_two_forks(seed):
    b = bw.make_board(seed)
    assert b.dist[0].max() < bw.BIG and b.dist[:, 0].max() < bw.BIG          # every tile reachable from every other
    forks = np.flatnonzero(b.na != b.nb)
    assert forks.tolist() == [3, 8]
    for field in b.fields:
        assert field.max() < bw.BIG
    assert len(set(b.pos)) == bw.N                                            # no two tiles share a cell
    for kind in (bw.MARKET, bw.TREASURE, bw.CAMP, bw.DEN, bw.GATE, bw.BOG):
        assert (b.kinds == kind).any()
    # the shortcut is shorter than the ring round the east end, and costs the bog
    assert b.dist[3, 25] < b.dist[4, 25] + 1 and b.dist[3, 25] <= 6
    assert b.cost[35] == 2.0 and b.cost[37] == 2.0


def test_kinds_choose_their_turns():
    peddler, brigand, rival, pilgrim = (bw.KINDS[k] for k in (bw.PEDDLER, bw.BRIGAND, bw.RIVAL, bw.PILGRIM))
    assert walkers.explain(pilgrim, health=1.0, goal_far=0.5)[0] == "follow"
    assert walkers.explain(pilgrim, health=1.0, need=0.9)[0] == "rest"
    assert walkers.explain(peddler, health=1.0, goal_far=0.5)[0] == "follow"
    assert walkers.explain(peddler, health=1.0, goal_far=0.0)[0] == "rest"           # arrived at the market: trade
    assert walkers.explain(peddler, health=1.0, goal_far=0.5, foe_prox=0.8, foe_ahead=-0.8)[0] == "flee"
    assert walkers.explain(brigand, health=1.0)[0] == "wander"
    assert walkers.explain(brigand, health=1.0, foe_prox=0.7, foe_ahead=0.7)[0] == "seek"      # prey ahead
    assert walkers.explain(brigand, health=1.0, foe_prox=0.8, foe_close=0.9, foe_ahead=0.8)[0] == "attack"
    assert walkers.explain(brigand, health=1.0, foe_prox=0.7, foe_ahead=-0.7)[0] == "wander"   # behind: not prey
    assert walkers.explain(brigand, health=0.3, refuge_near=0.9)[0] == "rest"
    assert walkers.explain(rival, health=1.0, goal_far=0.5)[0] == "follow"
    assert walkers.explain(rival, health=1.0, foe_prox=0.7, foe_ahead=-0.7)[0] == "flee"       # sprint from a pursuer
    assert walkers.explain(rival, health=1.0, need=0.9)[0] == "rest"


def test_turns_follow_the_active_mask_and_the_rules_hold():
    game = probe.fresh(3)
    s = game.session
    assert int(s.column("active")[0]) == 0
    s.control({"fire": 0.0})
    for _ in range(12):                          # the player's turn waits for its confirm
        s.advance(0.1)
    assert int(s.column("turns")[0]) == 0 and int(s.column("active")[0]) == 0
    records = probe.play(game, 18, log=False)
    assert [r["lane"] for r in records] == list(range(bw.ENT)) * 2         # one turn each, in lane order
    for r in records:
        assert r["moved"] <= r["allow"] <= bw.STEPS
        assert r["choice"] != "rest" or (r["allow"] == 0 and r["moved"] == 0)
        assert r["laps"] >= 0
    assert np.isfinite(s.column("x")).all() and np.isfinite(s.column("energy")).all()
    node = s.column("node")[:bw.ENT].astype(int)
    for lane in range(bw.ENT):                    # every entity stands on its tile's cell
        assert int(s.column("x")[lane]) == game.board.pos[node[lane]][0]
        assert int(s.column("y")[lane]) == game.board.pos[node[lane]][1]
    st = game.state()
    assert st["board"] == "free" and st["controls"] == ["move_axis", "move_axis_y", "fire"]


def test_the_players_pad_picks_the_fork_and_the_pace():
    outcomes = {}
    for label, controls in (("fork-A", {"move_axis": -1.0}), ("fork-B", {"move_axis": 1.0}),
                            ("run", {"move_axis_y": -1.0}), ("rest", {"move_axis_y": 1.0}), ("network", {})):
        game = probe.fresh(3)
        probe.put_player(game, 2)
        first = [r for r in probe.play(game, 1, controls, log=False)][0]
        outcomes[label] = (first, int(game.session.column("forkb")[0]))
    assert outcomes["fork-A"][1] == 0 and outcomes["fork-B"][1] == 1
    assert outcomes["fork-A"][0]["to"] != outcomes["fork-B"][0]["to"]
    assert outcomes["rest"][0]["moved"] == 0 and outcomes["rest"][0]["allow"] == 0
    assert outcomes["run"][0]["allow"] > outcomes["network"][0]["allow"] or outcomes["network"][0]["choice"] == "run"
    # the bog costs double: the shortcut's first step leaves too little for the second
    assert outcomes["fork-B"][0]["moved"] < outcomes["fork-B"][0]["allow"]

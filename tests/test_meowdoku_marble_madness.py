"""The marble course consumes the existing 3-D SDF product without changing its law."""
import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku.marble_madness import MarbleMadness
from meowdoku import fountain3d as F


def test_marble_course_uses_one_emitted_id_and_sdf_toy():
    game = MarbleMadness()
    state = game.state()
    uniforms = state["compute"]["uniforms"]
    assert state["game"] == "marble_madness"
    assert state["compute"]["manifest"] == "fountain3d_toy.manifest.json"
    assert state["compute"]["body"] is not None
    assert state["compute"]["size"] == "4k"
    assert uniforms["sdf_on"] == 1
    assert uniforms["emit_rate"] == MarbleMadness.EMIT_INTERVAL
    assert uniforms["wt0"] == 1 and all(uniforms[f"wt{s}"] == 0 for s in (1, 2, 3))
    assert len(state["compute"]["chamber_mesh"]["v"]) > 0
    assert [pid for pid in range(game.n) if pid * uniforms["emit_rate"] < 3600] == [0]


def test_goal_is_inside_law_chamber_and_finish_requires_it():
    game = MarbleMadness()
    goal = game.goal()
    u = {k: float(v) for k, v in game.uniforms().items()}
    assert float(F.chamber_distance(u, 0, goal["y"], 0)[0].item()) > 0
    with pytest.raises(ValueError, match="not reached"):
        game.do({"type": "marble_finish", "x": 0, "y": u["nozzle_y"], "z": 0, "seconds": 9})
    game.do({"type": "marble_finish", "x": 0, "y": goal["y"], "z": 0, "seconds": 9})
    assert game.state()["won"] is True
    assert game.result()["score"] == round(100000 / 9)
    epoch = game.epoch
    game.do({"type": "marble_restart"})
    assert game.epoch == epoch + 1 and not game.state()["won"] and game.result() is None


def test_server_catalog_and_action_route(tmp_path):
    from meowdoku.server import Table

    (tmp_path / "boards.json").write_text(json.dumps({"boards": [], "levels": []}), encoding="utf-8")
    table = Table(tmp_path)
    assert "marble_madness" in table.catalog()["games"]
    state = table.new({"game": "marble_madness"})
    assert state["marble"]["active_ids"] == 1
    goal = state["marble"]["goal"]
    won = table.action({"type": "marble_finish", "x": 0, "y": goal["y"], "z": 0, "seconds": 12})
    assert won["won"] is True
    restarted = table.action({"type": "marble_restart"})
    assert restarted["won"] is False
    tuned = table.action({"type": "fountain_param", "name": "mu0", "value": 0.42})
    assert tuned["compute"]["uniforms"]["mu0"] == pytest.approx(0.42)

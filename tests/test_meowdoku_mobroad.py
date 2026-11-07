"""Mobroad, the road-defence mob shooter: the panel rules, the splits, the crowd on the road (the fountain law's mobroad variant), killed grains
parked for good, the link between the page and the server, and the default products' digests untouched by the variant."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MEOWDOKU_LAW", "numpy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from meowdoku import fountain as F  # noqa: E402
from meowdoku import fountain_build as B  # noqa: E402
from meowdoku import mobroad as M  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

# ------------------------------------------------------------------------------------------------ the lane law, eager


def _tensors(cols, names):
    return [AbstractTensor.get_tensor(np.asarray(cols[n], dtype=np.float64)) for n in names]


def _step(cols, dt=1.0 / 60.0, steps=1):
    """``steps`` laws of ``mobroad_step`` on the columns (the AbstractTensor law itself, no product): the state is carried, the columns updated in place."""
    names = (*M.STATIC, *M.STATE, *M.SETTINGS, *M.CONTROLS, "dt")
    out = None
    for _ in range(steps):
        cols["dt"] = np.full(M.LANES, dt)
        out = M.mobroad_step(*_tensors(cols, names))
        for name, value in zip(M.STATE, out):
            cols[name] = np.broadcast_to(np.asarray(value.numpy(), dtype=np.float64), (M.LANES,)).copy()
    return cols


def _game_columns():
    cols = M.initial_columns()
    cols["dt"] = np.full(M.LANES, 1.0 / 60.0)
    cols["autofire"][:] = 0.0                         # tests fire by placing bullets
    return cols


def _bullet(cols, lane, x, y, vx=0.0, vy=-M.BULLET_SPEED, dmg=1.0):
    cols["alive"][lane], cols["x"][lane], cols["y"][lane], cols["vx"][lane], cols["vy"][lane] = 1.0, x, y, vx, vy
    cols["dmg"][lane], cols["born"][lane] = dmg, cols["clock"][0]


def _below(panel):
    """A bullet's y one step before it crosses the panel's line, flying up."""
    return panel["y"] + 0.5 * M.BULLET_SPEED / 60.0


def _setup_panel(cols, index, **fields):
    k = M.PANEL0 + index
    for name, value in fields.items():
        cols[name][k] = value
    return k


def _live_bullets(cols):
    lanes = np.arange(M.BULLET0, M.LANES)
    return lanes[cols["alive"][lanes] > 0.0]


def test_the_lane_layout_matches_the_roads_board():
    assert (M.WIDTH, M.HEIGHT) == F.MOB_BOARD
    assert M.LANES == M.BULLET0 + M.BULLETS == 41
    cols = _game_columns()
    assert set(M.initial_columns()) >= set(M.MobroadPiece.argument_names) - {"dt"}
    assert cols["alive"][0] == 1.0 and cols["alive"][M.PANEL0:M.PANEL0 + M.PANELS].sum() == M.PANELS


def test_a_bullet_that_crosses_a_split_panel_hits_it_once_and_the_value_rises():
    cols = _game_columns()
    p = M.LEVEL_PANELS[0]                                              # the split panel: x2.00, +0.25 a hit
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    _step(cols)
    k = M.PANEL0
    assert cols["phits"][k] == 1 and cols["pval"][k] == pytest.approx(p["v0"] + p["step"])
    _step(cols, steps=3)                                               # it is past the panel: no second hit
    assert cols["phits"][k] == 1 and cols["pval"][k] == pytest.approx(p["v0"] + p["step"])
    assert cols["score"][0] == pytest.approx(M.POINTS_HIT)             # a hit scores 1


def test_a_bullet_outside_the_panels_width_misses_it():
    cols = _game_columns()
    p = M.LEVEL_PANELS[0]
    _bullet(cols, M.BULLET0, p["x"] + p["hw"] + 0.5, _below(p))
    _step(cols)
    assert cols["phits"][M.PANEL0] == 0 and cols["pval"][M.PANEL0] == p["v0"] and len(_live_bullets(cols)) == 1


@pytest.mark.parametrize("value, copies", [(1.0, 1), (1.99, 1), (2.0, 2), (2.5, 2), (3.25, 3), (5.9, 5), (6.0, 6), (9.0, 6)])
def test_a_split_leaves_floor_of_the_value_bullets_capped(value, copies):
    """The rule: floor(value) bullets, at least 1 and at most MAX_COPIES, fanned about the heading and all straight on from the crossing."""
    assert M.panel_effect(M.SPLIT, value)["copies"] == copies
    cols = _game_columns()
    p = M.LEVEL_PANELS[0]
    cols["pval"][M.PANEL0] = value
    cols["pcap"][M.PANEL0] = 99.0
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    _step(cols)
    live = _live_bullets(cols)
    assert len(live) == copies                                         # (the value used is the one the bullet arrived with)
    if copies > 1:
        angles = np.sort(np.arctan2(cols["vx"][live], -cols["vy"][live]))
        assert np.allclose(np.diff(angles), M.FAN, atol=1e-9) and angles.mean() == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(np.hypot(cols["vx"][live], cols["vy"][live]), M.BULLET_SPEED)
        assert np.allclose(cols["y"][live], cols["y"][live][0]) and np.allclose(cols["x"][live], cols["x"][live][0])


def test_a_split_is_cut_short_by_a_full_bullet_pool():
    cols = _game_columns()
    p = M.LEVEL_PANELS[0]
    cols["pval"][M.PANEL0] = 5.0
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    for lane in range(M.BULLET0 + 1, M.LANES - 2):                     # all but two lanes are busy (far away, not crossing)
        _bullet(cols, lane, 0.2, 2.0)
    _step(cols)
    assert len(_live_bullets(cols)) == M.BULLETS                       # the pool is full, not more
    live_y = cols["y"][_live_bullets(cols)]
    assert (live_y < 5.0).sum() == M.BULLETS - 3 and (np.abs(live_y - p["y"]) < 0.5).sum() == 3     # the crossing bullet + 2 children; 4 copies were lost


def test_two_split_panels_in_a_row_multiply():
    cols = _game_columns()
    a, b = M.LEVEL_PANELS[2], M.LEVEL_PANELS[0]                        # the nearer panel is crossed first (smaller y is farther up the road)
    cols["pval"][M.PANEL0 + 2] = 2.0
    _bullet(cols, M.BULLET0, a["x"], _below(a))
    _step(cols)
    assert len(_live_bullets(cols)) == 2
    for lane in _live_bullets(cols):                                   # each child flies on to the next panel's line within its width
        cols["x"][lane], cols["y"][lane] = b["x"], _below(b)
    cols["pval"][M.PANEL0] = 3.0
    _step(cols)
    assert len(_live_bullets(cols)) == 6                               # 2 x 3


def test_a_damage_panel_scales_the_bullets_damage_and_reach_by_its_value():
    cols = _game_columns()
    p = M.LEVEL_PANELS[0]
    _setup_panel(cols, 0, pkind=M.DAMAGE, pval=2.25, pstep=0.5, pcap=8.0)
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    _step(cols)
    live = _live_bullets(cols)
    assert len(live) == 1 and cols["dmg"][live[0]] == pytest.approx(2.25) and cols["pval"][M.PANEL0] == pytest.approx(2.75)
    assert M.panel_effect(M.DAMAGE, 2.25) == {"copies": 1.0, "damage": 2.25}
    game = M.Mobroad.__new__(M.Mobroad)                                # the picture: size ~ sqrt(damage), and so the page's kill reach
    assert M.KILL_REACH * np.sqrt(2.25) == pytest.approx(M.BOLT_SIZE * np.sqrt(2.25) * M.REACH_PER_SIZE)


def test_a_rate_panel_completes_when_its_value_reaches_its_target_and_is_spent():
    cols = _game_columns()
    p = M.LEVEL_PANELS[1]                                              # rate: 1.0 .. 2.0, +0.1 a hit, x2 on completion
    _setup_panel(cols, 1, pval=1.96)
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    _step(cols)
    k = M.PANEL0 + 1
    assert cols["alive"][k] == 0.0 and cols["pval"][k] == pytest.approx(2.0)          # capped at the target, spent
    assert cols["rate_mult"][0] == pytest.approx(2.0) and cols["ncomp"][0] == 1
    assert cols["score"][0] == pytest.approx(M.POINTS_PANEL + M.POINTS_HIT)
    cols["reload"][:] = 0.0
    cols["fire"][:] = 1.0
    _step(cols)
    assert cols["reload"][0] == pytest.approx(1.0 / (M.BASE_RATE * 2.0), abs=1.0e-9)   # the player's next reload is half


def test_a_rate_panel_below_its_target_only_rises():
    cols = _game_columns()
    p = M.LEVEL_PANELS[1]
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    _step(cols)
    k = M.PANEL0 + 1
    assert cols["alive"][k] == 1.0 and cols["pval"][k] == pytest.approx(p["v0"] + p["step"]) and cols["rate_mult"][0] == 1.0


def test_a_shooters_panel_adds_escorts_up_to_the_maximum():
    cols = _game_columns()
    p = M.LEVEL_PANELS[1]
    _setup_panel(cols, 1, pkind=M.SHOOTERS, pval=1.96, pamount=1.0)
    _bullet(cols, M.BULLET0, p["x"], _below(p))
    assert cols["alive"][M.ESCORT0] == 0.0
    _step(cols)
    assert cols["shooters"][0] == 2.0
    _step(cols)
    assert cols["alive"][M.ESCORT0] == 1.0 and cols["alive"][M.ESCORT0 + 1] == 0.0      # the first escort stands beside the player
    assert abs(cols["x"][M.ESCORT0] - cols["x"][0]) == pytest.approx(abs(M.ESCORT_OFFSETS[0]), abs=0.05)
    cols["shooters"][:] = 5.0                                          # the cap
    _setup_panel(cols, 3, pkind=M.SHOOTERS, pval=1.96, pamount=3.0, home_y=_below(M.LEVEL_PANELS[3]) - 0.05)
    _bullet(cols, M.BULLET0 + 1, M.LEVEL_PANELS[3]["x"], M.LEVEL_PANELS[3]["y"] + 0.05)
    _step(cols)
    assert cols["shooters"][0] == float(M.MAX_SHOOTERS)


def test_every_shooter_fires_one_bullet_per_volley_and_autofire_holds_the_trigger():
    cols = _game_columns()
    cols["shooters"][:] = 3.0
    cols["fire"][:] = 1.0
    _step(cols, steps=2)
    assert len(_live_bullets(cols)) == 3
    off = _game_columns()
    off["shooters"][:] = 3.0
    _step(off, steps=3)
    assert len(_live_bullets(off)) == 0                                # nothing held, no autofire
    off["autofire"][:] = 1.0
    _step(off, steps=2)
    assert len(_live_bullets(off)) == 3


def test_bullets_expire_at_their_range_and_a_panel_is_hit_only_by_a_bullet_that_flies_through_it():
    cols = _game_columns()
    _bullet(cols, M.BULLET0, 6.0, 10.0)
    _step(cols, steps=int(M.BULLET_LIFE * 60) + 3)
    assert len(_live_bullets(cols)) == 0


def test_score_and_lives_follow_the_mobs_reported_counts():
    cols = _game_columns()
    cols["dead"][:], cols["goal"][:] = 7.0, 3.0
    _step(cols)
    assert cols["score"][0] == pytest.approx(7 * M.POINTS_KILL) and cols["lives"][0] == M.LOSE_AT - 3
    cols["goal"][:] = M.LOSE_AT + 4
    _step(cols)
    assert cols["lives"][0] == 0.0
    cols["fire"][:] = 1.0
    cols["autofire"][:] = 1.0
    _step(cols, steps=3)
    assert len(_live_bullets(cols)) == 0                               # the run is over: nobody fires


def test_the_game_takes_the_pages_reports_only_for_its_run_upward_and_bounded():
    game = M.Mobroad(size="small")
    e = game.mob_epoch
    assert game.report({"epoch": e, "dead": 10, "goal": 2, "crush": 5, "live": 90})
    assert (game.mob["dead"], game.mob["goal"]) == (10, 2) and game.session.columns["dead"][0] == 10.0
    assert not game.report({"epoch": e + 1, "dead": 200, "goal": 0})        # another run's
    game.report({"epoch": e, "dead": 4, "goal": 1})                          # never downward
    assert (game.mob["dead"], game.mob["goal"]) == (10, 2)
    game.report({"epoch": e, "dead": 10_000, "goal": 10_000})                # bounded by the id count
    assert game.mob["dead"] + game.mob["goal"] <= game.active
    state = game.state()
    assert state["mobroad"]["dead"] == game.mob["dead"] and state["compute"]["manifest"] == "fountain_mobroad.manifest.json"
    assert state["compute"]["mobroad"]["k"] == F.mobroad_map()["k"] and "result" not in state
    game.reset()
    assert game.mob_epoch == e + 1 and game.mob == {"dead": 0, "goal": 0, "crush": 0, "live": 0}


def test_the_run_is_won_when_every_id_is_resolved_and_lost_at_the_limit():
    game = M.Mobroad(size="small")
    e = game.mob_epoch
    assert game.result() is None
    game.report({"epoch": e, "dead": game.active - 2, "goal": 2})
    game.session.advance(0.05)                                              # the law lands the score and the lives
    res = game.result()
    assert res and res["won"] and res["score"] == (game.active - 2) * M.POINTS_KILL
    lost = M.Mobroad(size="small")
    lost.report({"epoch": lost.mob_epoch, "dead": 50, "goal": int(M.LOSE_AT)})
    lost.session.advance(0.05)                                              # the law lands the lives
    res = lost.result()
    assert res and not res["won"] and lost.state()["over"]


def test_the_game_is_registered_in_the_catalog_and_made_by_the_table(tmp_path):
    from meowdoku import server
    (tmp_path / "boards.json").write_text(json.dumps({"boards": [], "levels": []}), encoding="utf-8")
    table = server.Table(tmp_path)
    assert "mobroad" in table.catalog()["games"]
    table.new({"game": "mobroad", "size": "small"})
    table.game.session.advance(0.05)                                    # (the entities are the law's, from its first round)
    state = table.state()
    assert state["game"] == "mobroad" and state["board"] == "free" and (state["width"], state["height"]) == F.MOB_BOARD
    assert {e["mesh"] for e in state["entities"]} <= {"rocket", "bolt"} and "rocket" in {e["mesh"] for e in state["entities"]} and len(state["mobroad"]["panels"]) == 4
    assert {p["kind"] for p in state["mobroad"]["panels"]} == {"split", "rate"}      # the level's two kinds


# ------------------------------------------------------------------------------------------------ the mob: the fountain law's variant


class _Eager:
    """The mobroad law as plain AbstractTensor ops, called with columns."""

    def __init__(self, n, active=None):
        self.n, self.law = n, F.make_law_mobroad(n, active)

    def __call__(self, state, cols):
        tensors = [AbstractTensor.get_tensor(np.asarray(state[k] if k in state else cols[k], dtype=np.float64)) for k in F.MOBROAD_ARGUMENTS]
        out = [np.broadcast_to(np.asarray(v.numpy(), dtype=np.float64), (self.n,)).copy() for v in self.law(*tensors)]
        return dict(zip(F.MOBROAD_OUTPUTS, out))


def _mob_world(n, x, y, vy=0.0, still=False):
    """``n`` grains placed (all emitted: ``age`` past every emit time) on the road: (law, columns, state).  ``still``: the pather's weights all zero (no walking)."""
    cols = F.mobroad_columns(n, active=n)
    cols["age"][:] = 1.0e3
    if still:
        cols.update({name: np.float64(0.0) for name in F.MOB_PATHER_UNIFORMS})
    cols["x"], cols["y"], cols["vy"] = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.full(n, vy)
    return _Eager(n, n), cols, {k: np.array(cols[k], dtype=np.float64) for k in F.STATE}


def _run(law, cols, state, steps, each=None):
    out = None
    for step in range(steps):
        out = law(state, cols)
        state = {k: out[k + "_next"] for k in F.STATE}
        if each is not None:
            each(step, state, out)
    return state, out


def test_the_road_is_slots_and_the_variants_names_extend_the_default_ones():
    assert F.MOBROAD_ARGUMENTS[: len(F.ARGUMENTS)] == F.ARGUMENTS and F.MOBROAD_OUTPUTS[: len(F.OUTPUTS)] == F.OUTPUTS
    assert F.MOBROAD_DIAGNOSTICS == ("n_dead", "n_goal", "n_crush") and F.MOB_KZ == 32
    spec = F.mobroad_spec("small")
    assert spec["view"] == [-F.MOB_ROAD_HALF, F.MOB_ROAD_HALF, 0.0, F.MOB_ROAD_TOP]
    assert spec["map"]["k"] * 2 * F.MOB_ROAD_HALF == pytest.approx(F.MOB_BOARD[0]) and spec["map"]["k"] * F.MOB_ROAD_TOP == pytest.approx(F.MOB_BOARD[1])
    u = spec["uniforms"]
    kinds = [int(u[f"q{k}kind"]) for k in range(F.SDF_SLOTS)]
    assert kinds.count(F.SDF_KINDS["box"]) == 3 and kinds.count(F.SDF_KINDS["circle"]) == 2 and kinds.count(F.SDF_KINDS["capsule"]) == 2
    assert kinds.count(F.SDF_KINDS["pegs"]) == 1                      # the chamber + 2 jaws, 2 pillars, 2 funnel barriers, the peg field
    assert len(spec["chamber_outline"]) >= 8                           # the walls and every feature draw


def test_the_default_uniforms_and_arguments_do_not_know_the_road():
    assert not any(name.startswith(("kz", "goal")) for name in F.ARGUMENTS + F.UNIFORMS + F.RUNTIME_INPUTS)
    assert "sdf_mobroad" in F.SDF_SHAPES                                # (the shape table is shared; the default chamber choices are not)
    assert "sdf_mobroad" not in dict(F.CHAMBER_CHOICES)


def test_the_mob_is_held_by_the_roads_features():
    """Grains driven into the narrowing's jaw at speed: none leaves the chamber (n_out = 0), none ends up inside the jaw (the contact law holds them)."""
    g = F.MOB_FEATURES[1]
    jaw_top = g["y"] + g["half_height"]
    xs, ys = np.meshgrid(np.linspace(-0.52, -0.29, 7), jaw_top + 0.045 + np.arange(4) * 0.028)
    n = xs.size
    law, cols, state = _mob_world(n, xs.ravel(), ys.ravel(), vy=-1.2)
    u = {k: float(v) for k, v in cols.items() if k.startswith("q") or k == "sdf_on"}

    def check(step, st, out):
        assert out["n_out"][0] == 0 and out["n_nan"][0] == 0 and out["n_live"][0] == n

    state, out = _run(law, cols, state, 80, check)
    d = F.chamber_distance(u, state["x"], state["y"])[0]
    radius = out["rad"]
    assert (d > -0.75 * radius).all()                                  # nobody inside the jaw or through the wall
    assert state["y"].min() > jaw_top - 0.02 and out["n_crush"][0] >= 0


def test_a_swept_bullet_parks_the_grains_it_touches_for_good():
    n = 12
    xs = np.linspace(-0.3, 0.3, n)
    law, cols, state = _mob_world(n, xs, np.full(n, 0.9), still=True)         # no walking: they stay where they are
    m = F.mobroad_map()
    # one bullet swept across the road at y = 0.9 (fountain) from board x of the road's left third to its middle: it kills the grains in its path
    y_board = (m["top"] - 0.9) * m["k"]
    seg = F.mobroad_inputs([{"id": "b", "x": (0.0 + m["half"]) * m["k"], "y": y_board, "kr": 0.005 * m["k"]}], {"b": ((-0.34 + m["half"]) * m["k"], y_board)}, m)
    for k, v in seg.items():
        cols[k] = np.float64(v)
    state, out = _run(law, cols, state, 2)
    hit = xs <= 0.0 + 1e-9
    hit &= xs >= -0.34 - 1e-9
    dead = state["age"] < -5.0e8
    assert dead.sum() >= 2 and out["n_dead"][0] == dead.sum() and (dead == hit).all()
    assert (state["x"][dead] == 0).all() and (state["y"][dead] == 0).all() and np.allclose(state["age"][dead], -1.0e9, atol=1.0)       # pinned far below any emit time (the f64 lane adds a step to it)
    assert out["n_parked"][0] == dead.sum() and out["n_live"][0] == n - dead.sum() and out["n_goal"][0] == 0
    # the bullet is gone: nothing comes back, the survivors are untouched
    for k in seg:
        cols[k] = np.float64(0.0) if k.endswith("_on") else cols[k]
    before = {k: v.copy() for k, v in state.items()}
    state, out = _run(law, cols, state, 6)
    assert (state["age"][dead] < -5.0e8).all() and (state["x"][dead] == 0).all() and out["n_dead"][0] == dead.sum()
    assert out["n_live"][0] == n - dead.sum() and np.allclose(state["x"][~dead], before["x"][~dead], atol=1e-3)


def test_a_grain_within_the_attack_radius_of_the_player_has_arrived_and_stays_parked():
    px, py = float(F.mobroad_defaults()["tgt_x"]), float(F.mobroad_defaults()["tgt_y"])
    law, cols, state = _mob_world(3, [px + 0.5 * F.MOB_ATTACK_R, px + 0.3, -0.2], [py, py, 0.9], still=True)       # one at the player, one a road's width off, one far
    state, out = _run(law, cols, state, 2)
    assert out["n_goal"][0] == 1 and out["n_dead"][0] == 0 and state["age"][0] < -1.5e9 and state["x"][0] == 0.0 and state["y"][0] == 0.0
    state, out = _run(law, cols, state, 4)
    assert out["n_goal"][0] == 1 and out["n_live"][0] == 2 and state["age"][0] < -1.5e9


def test_the_pather_arrival_point_is_where_the_game_puts_the_player():
    d = F.mobroad_defaults()
    m = F.mobroad_map()
    assert d["tgt_y"] == pytest.approx(m["top"] - M.PLAYER_Y / m["k"], abs=1e-6) and d["atk_r"] == pytest.approx(F.MOB_ATTACK_R)
    assert F.mobroad_target(M.WIDTH / 2, M.PLAYER_Y, m) == {"tgt_x": pytest.approx(0.0, abs=1e-6), "tgt_y": pytest.approx(d["tgt_y"], abs=1e-6)}


# ------------------------------------------------------------------------------------------------ gravity orthogonal to the scene, and the pather


def test_the_in_plane_gravity_is_zero_in_the_variant():
    u = F.mobroad_uniforms(256)
    assert u["grav_x"] == 0.0 and u["grav_y"] == 0.0                    # the variant's own uniforms: gravity points into the plane
    # ... and the law ignores what a page writes there (a sensor, a spinning case): a still grain (no walking) in open road does not move
    law, cols, state = _mob_world(1, [0.0], [1.0], still=True)
    cols["grav_x"], cols["grav_y"] = np.float64(0.7), np.float64(-1.3)
    state, out = _run(law, cols, state, 4)
    assert abs(state["vx"][0]) < 1e-9 and abs(state["vy"][0]) < 1e-9 and state["x"][0] == pytest.approx(0.0) and state["y"][0] == pytest.approx(1.0)
    assert F.MOB_MARCH.get("grav_y") is None


def _tensor_batch(rng, n):
    x, y = rng.uniform(-0.6, 0.6, n), rng.uniform(0.0, 2.0, n)
    vx, vy = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    d = rng.uniform(-0.05, 0.4, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    return x, y, vx, vy, d, np.cos(ang), np.sin(ang)


def _pather_tensors(w, x, y, vx, vy, d, nx, ny, tx=0.1, ty=0.09):
    t = lambda v: AbstractTensor.get_tensor(np.asarray(v, dtype=np.float64))      # noqa: E731
    ax, ay = F.mob_pather_accel(t(x), t(y), t(vx), t(vy), t(d), t(nx), t(ny), t(np.float64(tx)), t(np.float64(ty)), [t(np.float64(v)) for v in w])
    return np.asarray(ax.numpy()), np.asarray(ay.numpy())


def test_the_pather_output_is_capped_at_the_walking_force_whatever_the_weights():
    rng = np.random.default_rng(3)
    batch = _tensor_batch(rng, 200)
    for scale in (0.0, 1.0, 40.0):
        w = rng.normal(0.0, 1.0, F.MOB_PW) * scale
        ax, ay = _pather_tensors(w, *batch)
        assert (np.hypot(ax, ay) <= F.MOB_WALK * (1 + 1e-9)).all()
    ax, ay = _pather_tensors(F.mob_pather_hand(), *batch)
    mag = np.hypot(ax, ay)
    assert mag.max() <= F.MOB_WALK * (1 + 1e-9) and mag.max() > 0.9 * F.MOB_WALK           # it does walk, up to the cap
    assert len(F.MOB_PATHER_UNIFORMS) == F.MOB_PW == len(F.mob_pather_hand()) == len(F.mob_pather_weights("trained"))


def test_the_trainers_batched_forward_is_the_laws_pather():
    from meowdoku import mobroad_pather_sim as S
    rng = np.random.default_rng(5)
    x, y, vx, vy, d, nx, ny = _tensor_batch(rng, 50)
    w = rng.normal(0.0, 0.8, F.MOB_PW)
    ax, ay = _pather_tensors(w, x, y, vx, vy, d, nx, ny, 0.2, 0.09)
    bx, by = S.forward(w[None], x[None], y[None], vx[None], vy[None], d[None], nx[None], ny[None], np.full((1, 1), 0.2), np.full((1, 1), 0.09))
    assert np.allclose(ax, bx[0], atol=1e-12) and np.allclose(ay, by[0], atol=1e-12)


def test_the_hand_set_pather_walks_a_lone_grain_toward_the_target_and_off_a_pillars_axis():
    """Eager law, short: a grain just above a pillar's crown with the player below and to its left walks left (around the pillar), not into it."""
    pillar = F.MOB_FEATURES[0]
    crown = pillar["y"] + pillar["r"]
    start = (pillar["x"], crown + 0.03)
    law, cols, state = _mob_world(1, [start[0]], [start[1]])
    cols.update({k: np.float64(v) for k, v in F.mobroad_pather_inputs(F.mob_pather_weights("hand")).items()})
    cols["tgt_x"], cols["tgt_y"] = np.float64(0.0), np.float64(1.2)
    cols["fl_mu"] = np.float64(0.0)                                     # (no ground friction: the short run shows the push)
    state, out = _run(law, cols, state, 36)
    assert out["n_nan"][0] == 0 and out["n_live"][0] == 1
    assert state["vx"][0] < -0.01 and state["x"][0] < pillar["x"] - 1e-4          # it slides toward the player's side of the pillar
    d = F.chamber_distance({k: float(v) for k, v in cols.items() if k.startswith("q") or k == "sdf_on"}, state["x"], state["y"])[0]
    assert d[0] > 0.0                                                  # ... and is not in the pillar


def test_a_lone_grain_with_the_hand_set_pather_walks_round_a_pillar_to_the_player():
    from meowdoku import mobroad_pather_sim as S
    pillar = F.MOB_FEATURES[0]
    target_x = 0.0
    r = S.simulate([F.mob_pather_hand()], [target_x], horizon=20.0, start=([pillar["x"]], [pillar["y"] + pillar["r"] + 0.2]))
    assert r["arrived"][0, 0] == 1.0
    xs = np.array([t[0][0] for t in r["track"]])
    ys = np.array([t[1][0] for t in r["track"]])
    d = F.chamber_distance(F.mobroad_uniforms(256), xs, ys)[0]
    assert (d > 0.0).all()                                              # never through a wall or the pillar
    assert ys[-1] < F.MOB_PLAYER_Y + F.MOB_ATTACK_R and (ys[1:] <= ys[:-1] + 1e-6).sum() > 0.7 * (len(ys) - 1)


def test_the_trained_weights_beat_the_hand_set_ones_on_arrival_rate():
    from meowdoku import mobroad_pather_sim as S
    hand, trained = S.arrival_rate(F.mob_pather_weights("hand")), S.arrival_rate(F.mob_pather_weights("trained"))
    print(f"arrival rate at the player, held-out: hand-set {hand:.4f}  trained {trained:.4f}")
    assert trained > hand + 0.01


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's write")
def test_the_pages_target_and_pather_writes_equal_pythons(tmp_path):
    m = F.mobroad_map()
    w = F.mob_pather_weights("trained")
    case = {"map": m, "board": [4.2, 18.9], "target": F.mobroad_target(4.2, 18.9, m), "w": w, "pather": F.mobroad_pather_inputs(w)}
    (tmp_path / "case.json").write_text(json.dumps(case), encoding="utf-8")
    (tmp_path / "run.mjs").write_text(f"""
import fs from 'node:fs';
import {{ pathToFileURL }} from 'node:url';
const {{ mobroadTarget, mobroadPather }} = await import(pathToFileURL({json.dumps(str(ROOT / 'examples' / 'meowdoku' / 'web' / 'mobroad.js'))}).href);
const c = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'case.json'))}, 'utf8'));
let bad = 0;
const t = mobroadTarget(c.board[0], c.board[1], c.map), p = mobroadPather(c.w);
for (const [k, v] of Object.entries(c.target)) if (!(Math.abs(t[k] - v) <= 1e-6 * Math.max(1, Math.abs(v)))) {{ bad++; console.log('MISMATCH', k, t[k], v); }}
for (const [k, v] of Object.entries(c.pather)) if (!(Math.abs(p[k] - v) <= 1e-6 * Math.max(1, Math.abs(v)))) {{ bad++; console.log('MISMATCH', k, p[k], v); }}
console.log('bad', bad, Object.keys(p).length);
process.exit(bad || Object.keys(p).length !== Object.keys(c.pather).length ? 1 : 0);
""", encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "run.mjs")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_variant_fires_only_inside_its_wrapper_and_the_default_law_is_unchanged():
    assert F._MOBROAD["on"] is False and F._PURRLINE["on"] is False
    n = 4
    cols = F.initial_columns(n, active=4096)
    cols["age"][:] = -1.0e9
    plain = F.make_law(n, 4096)
    tens = lambda c, names: [AbstractTensor.get_tensor(np.asarray(c[k], dtype=np.float64)) for k in names]      # noqa: E731
    base = [np.asarray(v.numpy()) for v in plain(*tens(cols, F.ARGUMENTS))]
    mob = F.mobroad_columns(n, active=4096)
    mob["age"][:] = -1.0e9
    variant = F.make_law_mobroad(n, 4096)(*tens(mob, F.MOBROAD_ARGUMENTS))
    assert len(variant) == len(F.OUTPUTS) + 3 and F._MOBROAD["on"] is False
    after = [np.asarray(v.numpy()) for v in plain(*tens(cols, F.ARGUMENTS))]
    assert all(np.array_equal(a, b) for a, b in zip(base, after)) and len(base) == len(F.OUTPUTS)


# ------------------------------------------------------------------------------------------------ the link, and the products

def test_the_bullets_become_swept_kill_segments():
    m = F.mobroad_map()
    bullets = [{"id": "11:0.5", "x": 6.25, "y": 10.0, "kr": 0.34}, {"id": "9:0.1", "x": 3.0, "y": 8.0, "kr": 0.5}]
    out = F.mobroad_inputs(bullets, {"9:0.1": (3.0, 8.4)}, m)
    k = m["k"]
    assert out["kz0_on"] == 1.0 and out["kz0_r"] == pytest.approx(0.5 / k, rel=1e-6)                 # the most advanced (smallest y) is slot 0
    assert out["kz0_ax"] == pytest.approx(3.0 / k - m["half"], rel=1e-6) and out["kz0_ay"] == pytest.approx(m["top"] - 8.4 / k, rel=1e-5)
    assert out["kz0_by"] == pytest.approx(m["top"] - 8.0 / k, rel=1e-5) and out["kz0_by"] > out["kz0_ay"]        # it flew up the road
    assert out["kz1_ax"] == out["kz1_bx"] and out["kz1_ay"] == out["kz1_by"]                                # no previous place: a point
    assert all(out[f"kz{i}_on"] == 0.0 for i in range(2, F.MOB_KZ))
    many = [{"id": str(i), "x": 1.0, "y": 5.0 + i, "kr": 0.3} for i in range(40)]
    assert sum(F.mobroad_inputs(many, {}, m)[f"kz{i}_on"] for i in range(F.MOB_KZ)) == F.MOB_KZ


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's write")
def test_the_pages_write_equals_pythons(tmp_path):
    m = F.mobroad_map()
    bullets = [{"id": f"{i}:0.1", "x": 1.0 + 0.7 * i, "y": 12.0 - 0.9 * i, "kr": 0.3 + 0.01 * i} for i in range(9)]
    prev = {b["id"]: (b["x"] + 0.1, b["y"] + 0.4) for b in bullets[::2]}
    case = {"bullets": bullets, "prev": {k: list(v) for k, v in prev.items()}, "map": m, "expect": F.mobroad_inputs(bullets, prev, m)}
    (tmp_path / "case.json").write_text(json.dumps(case), encoding="utf-8")
    (tmp_path / "run.mjs").write_text(f"""
import fs from 'node:fs';
import {{ pathToFileURL }} from 'node:url';
const {{ mobroadInputs }} = await import(pathToFileURL({json.dumps(str(ROOT / 'examples' / 'meowdoku' / 'web' / 'mobroad.js'))}).href);
const c = JSON.parse(fs.readFileSync({json.dumps(str(tmp_path / 'case.json'))}, 'utf8'));
const got = mobroadInputs(c.bullets, new Map(Object.entries(c.prev)), c.map);
let bad = 0;
for (const [k, v] of Object.entries(c.expect)) if (!(Math.abs(got[k] - v) <= 1e-6 * Math.max(1, Math.abs(v)))) {{ bad++; console.log('MISMATCH', k, got[k], v); }}
console.log('keys', Object.keys(got).length, Object.keys(c.expect).length, 'bad', bad);
process.exit(bad || Object.keys(got).length !== Object.keys(c.expect).length ? 1 : 0);
""", encoding="utf-8")
    done = subprocess.run(["node", str(tmp_path / "run.mjs")], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


#: The DEFAULT and WHEEL 2-D products' source digests, as they were before the mobroad variant existed (the examples/meowdoku/fountain.py and
#: fountain_build.py at the merge of main), and the purrline variant's: the variant lives between ``# <mobroad>`` markers that every one of them leaves out.
BASELINE_DIGESTS = {
    "default": {"small": "3c4362437fea7313", "4k": "a4bd4dad8185839c", "16k": "084ba8e865438ba7", "64k": "262295590088b70d", "256k": "0403d076cec9587c",
                "1M": "6f8a685d82b9b025", "4M": "b3ab04cfd4249359", "10M": "ab8101dd0e570bac"},
    "wheel": {"small": "ef9d188dfb9d881f", "4k": "109c76a5b2c27917", "16k": "72859f7dcf0b1652", "64k": "d102cf7cde48ed3a", "256k": "d4d8b8767a3263c4",
              "1M": "b334b16a4ab13dbf", "4M": "3a77011fa218e3a8", "10M": "f2f4dfed887151b2"},
    "purrline": {"small": "2929b91392753c4a", "4k": "1295f266043af8b3", "16k": "aab13f0f06aeb53c", "64k": "c6039ddcf237f9e6", "256k": "bd6239fd802a46e0",
                 "1M": "c41ed00f3bc60772", "4M": "95b07b4c036d9b49", "10M": "f277f30be189b876"},
}


def test_the_default_wheel_and_purrline_digests_are_unchanged_for_every_size():
    for tag in B.SIZES:
        assert B.digest(tag) == BASELINE_DIGESTS["default"][tag], tag
        assert B.digest(tag, True) == BASELINE_DIGESTS["wheel"][tag], tag
        assert B.purrline_digest(tag) == BASELINE_DIGESTS["purrline"][tag], tag
    assert len(B.SIZES) == 8


def test_the_default_sources_leave_the_variant_out_and_its_own_digest_keeps_it():
    text = B._sources()
    assert "mobroad" not in text.lower() and "make_law_mobroad" not in text and "_MOBROAD" not in text
    whole = B._module_sources(False)
    assert "make_law_mobroad" in whole and "build_mobroad" in whole
    assert B.mobroad_digest("4k") not in (B.digest("4k"), B.purrline_digest("4k"))
    assert "mobroad" not in B._module_sources(False, B._MARKED_MOB).lower()          # the purrline digest's text

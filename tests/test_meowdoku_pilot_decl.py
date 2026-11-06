"""The declared autopilot (pilot_decl), its two networks, span harvesting and the idle trainer.

Everything runs headless on the NumPy lane in a temporary weights directory.
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

from meowdoku import autopilot as ap  # noqa: E402
from meowdoku import idle_trainer as it  # noqa: E402
from meowdoku import pilot_decl as pd  # noqa: E402
from meowdoku import pilot_train as pt  # noqa: E402

GAMES = ("delve", "platformer", "side_scroller", "tilequest")


@pytest.fixture
def build(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "BUILD", tmp_path)
    monkeypatch.setattr(ap, "IDLE_TRAINER", object())          # recorded play is left to the idle trainer
    monkeypatch.setattr(ap, "_PILOT_WEIGHTS", {})
    monkeypatch.setattr(ap, "_HUMAN_GAMES", {})
    monkeypatch.setattr(ap, "SPAN_ROWS", 12)
    monkeypatch.setattr(ap, "FLUSH_ROWS", 36)
    return tmp_path


def teacher_play(game, name, ticks, seq0=1):
    """Drive a game the way a player does: input intervals carrying the held controls."""
    decl = pd.DECLS[name]
    teacher = decl.make_teacher(game)
    for i in range(ticks):
        controls = decl.controls_from(game, teacher.act(game))
        game.control({"inputs": [{"seq": seq0 + i, "dur": decl.step, **controls}]})
        game.tick()


# ------------------------------------------------------------------ the declaration
@pytest.mark.parametrize("name", GAMES)
def test_declaration_observes_and_maps_controls(name):
    decl = pd.DECLS[name]
    game = decl.make(7)
    game.control({}); game.tick()
    joint = decl.observe(game, "joint")
    assert joint.shape == (decl.widths("joint")[0],)
    assert decl.observe(game, "token").shape == (decl.widths("token")[0],)
    assert decl.observe(game, "graph").shape == (decl.widths("graph")[0],)
    assert np.array_equal(decl.slice("token", joint), decl.observe(game, "token"))
    assert np.array_equal(decl.slice("graph", joint), decl.observe(game, "graph"))
    view = decl.read(game)
    assert view.patch.shape == (pd.NCH, decl.patch[1], decl.patch[0])
    assert view.patch.sum() > 0                                           # the map patch sees tiles
    out = np.array([0.5 if c.kind == "axis" else 0.8 for c in decl.controls])
    controls = decl.controls_from(game, out)
    assert set(controls) == set(game.CONTROLS)
    back = decl.target_from(game, controls)
    assert back.shape == (decl.outputs,) and np.allclose(back[[c.kind == "axis" for c in decl.controls]], 0.5)
    assert all(abs(v) <= 1.0 for v in controls.values())


def test_graph_network_is_sparse_and_masks_stay_fixed():
    decl = pd.DECLS["delve"]
    params = decl.init_params("graph")
    assert ap.arch_of(params) == "graph" and ap.widths_of(params) == decl.widths("graph")
    assert 0.1 < params["GM1"].mean() < 0.7
    assert "GM1" not in ap.PolicyTrainer(params).trained


# ------------------------------------------------------------------ a trained policy per game
@pytest.mark.parametrize("name", GAMES)
@pytest.mark.parametrize("arch", pd.ARCHS)
def test_trained_policy_loads_and_acts(build, monkeypatch, name, arch):
    decl = pd.DECLS[name]
    result = pt.train(name, (arch,), episodes=3, rounds=0, epochs=2, seconds=3.0, log=lambda *_: None)
    assert (build / f"{name}.{arch}.npz").exists() and result[arch]["loss_last"] == result[arch]["loss_last"]
    monkeypatch.setenv(f"MEOWDOKU_PILOT_ARCH_{name.upper()}", arch)
    game = decl.make(5)
    game.control({}); game.tick()
    game.set_autopilot(True)                                           # loads <game>.<arch>.npz
    assert game.pilot.lane == "numpy" and ap.arch_of(game.pilot.params) == arch
    assert game.PILOT_STEP == decl.step
    for i in range(6):
        game.control({"inputs": [{"seq": i + 1, "dur": decl.step}]}); game.tick()
    assert set(game.held) == set(game.CONTROLS)
    assert game.pilot.controls(game) == ap.controls_from(game, game.pilot.run(game.pilot.features(game))[0])


# ------------------------------------------------------------------ spans: recorded without the game ending
def test_spans_of_an_unfinished_game_are_recorded_scored_and_trained_on(build):
    name = "platformer"
    game = pd.DECLS[name].make(1001)
    game.learns = True
    game.control({}); game.tick()
    teacher_play(game, name, 450)
    assert game.result() is None and not getattr(game, "_ended", False)       # the game has NOT ended
    rec = ap.load_records(name)
    assert rec is not None and len(rec["x"]) >= ap.FLUSH_ROWS
    assert len(rec["episode_score"]) == 0                                     # no finished game anywhere
    assert (rec["tags"] & ap.TAG_SCORE).any()                                 # outcome-linked spans (the climb scored)
    assert len(np.unique(rec["span"])) >= 3
    assert np.isnan(rec["loss"]).all()                                        # not yet scored
    trainer = it.IdleTrainer(remote=False, steps_per_pass=12, chunk_steps=4, log=lambda *_: None)
    params = trainer._params(name)
    data = trainer.build(name, params)
    assert data["report"]["human_spans"] >= 3 and data["report"]["scored"] >= 3
    scored = ap.load_records(name)
    assert not np.isnan(scored["loss"]).any()                                 # every span now has its loss
    assert len(data["val_x"]) > 0 and abs(data["p"].sum() - 1.0) < 1e-9
    record = trainer.train_game(name)
    assert record["status"] == "adopted" and record["val_after"] <= record["val_before"]
    assert record["human_spans"] >= 3
    assert (build / f"{name}.token.npz").exists()


def test_high_loss_and_outcome_spans_are_replayed_more():
    # prioritised replay: weight grows with loss; an outcome tag adds worth; damage has little
    base = (0.5 + it.EPS) ** it.ALPHA
    assert (0.5 + it.EPS) ** it.ALPHA > (0.01 + it.EPS) ** it.ALPHA * 3
    assert ap.tag_weight(ap.TAG_ROOM) > ap.tag_weight(0) > ap.tag_weight(ap.TAG_DAMAGE)
    assert base > 0
    assert it.IdleTrainer(remote=False).human_weight > it.IdleTrainer(remote=False).self_weight


# ------------------------------------------------------------------ the idle trainer
def test_idle_trainer_pauses_on_a_tick_resumes_and_hot_swaps_weights(build):
    name = "side_scroller"            # imitable from a short sample (the platformer overfits one this small)
    game = pd.DECLS[name].make(1002)
    game.learns = True
    game.control({}); game.tick()
    teacher_play(game, name, 140)
    # the weights to start from: a fresh network, flying live
    decl = pd.DECLS[name]
    first = decl.init_params("token", seed=3)
    ap.save_params(first, ap.decl_weights_path(name, "token"))
    live = ap.live_params(name)
    pilot = ap.Pilot(name, live)
    before = {n: live[n].copy() for n in ap.param_names(live)}
    now = [100.0]

    class Trainer(it.IdleTrainer):
        chunks = 0

        def _submit(self, *args):
            out = super()._submit(*args)
            Trainer.chunks += 1
            if Trainer.chunks == 2:
                self.touch()                                  # a tick arrives in the middle of the pass
            return out

    trainer = Trainer(remote=False, steps_per_pass=40, chunk_steps=4, idle_after=5.0, clock=lambda: now[0], log=lambda *_: None, lr=1e-2)
    trainer.touch()
    assert not trainer.idle() and trainer.cycle() == 0                          # play is on: nothing trains
    now[0] += 6.0
    assert trainer.idle()
    record = trainer.train_game(name)
    assert record["status"] == "paused" and record["steps"] == 8 and trainer.paused_count == 1
    assert name in trainer.passes                                              # the pass is kept
    assert all(np.array_equal(live[n], before[n]) for n in before)             # nothing adopted yet
    assert trainer.cycle() == 0                                                # still not idle
    now[0] += 6.0
    record = trainer.train_game(name)                                          # resumes where it paused
    assert record["status"] in ("adopted", "kept") and record["steps"] == 40 and Trainer.chunks == 10
    assert record["val_after"] <= record["val_before"] or record["status"] == "kept"
    assert record["status"] == "adopted"
    assert any(not np.array_equal(live[n], before[n]) for n in before)         # the LIVE dict changed in place
    assert pilot.params is live and ap.WEIGHT_VERSION[name] >= 1               # the live pilot reads the new weights
    assert ap.decl_weights_path(name, "token").with_suffix(".prev.npz").exists()   # the previous weights are kept
    # a pass whose result is worse is not adopted
    held = {n: live[n].copy() for n in before}
    trainer.headless_guard = lambda *_: False
    teacher_play(game, name, 60, seq0=1000)
    trainer.touch(); now[0] += 6.0
    record = trainer.train_game(name)
    assert record["status"] == "kept" and all(np.array_equal(live[n], held[n]) for n in held)


# ------------------------------------------------------------------ the autopilot's own play
def test_selfplay_is_recorded_weighted_and_does_not_regress(build, monkeypatch):
    name = "platformer"
    decl = pd.DECLS[name]
    pt.train(name, ("token",), episodes=6, rounds=0, epochs=6, seconds=10.0, log=lambda *_: None)
    monkeypatch.setenv(f"MEOWDOKU_PILOT_ARCH_{name.upper()}", "token")
    game = decl.make(2001)
    game.learns = True
    game.control({}); game.tick()
    game.set_autopilot(True)                                          # the demo plays; inference play is harvested like a human's
    for i in range(400):
        if game.result() is not None:                                 # a weak pilot falls: the next run starts
            game.reset(); game.control({}); game.tick()
        game.control({"inputs": [{"seq": i + 1, "dur": 0.05}]}); game.tick()
    own = ap.load_records(name, "selfplay")
    assert own is not None and len(own["x"]) >= ap.FLUSH_ROWS and not np.isnan(own["ret"]).any()
    assert ap.load_records(name, "human") is None                     # none of it was recorded as a human's
    adv = it.advantages(own)
    assert len(adv) >= 3 and abs(np.mean(list(adv.values()))) < 5
    ordered = sorted(adv.values())
    assert it.outcome_weight(ordered[-1]) >= it.outcome_weight(ordered[0])    # a better outcome weighs at least as much
    assert it.outcome_weight(-10.0) < 0.05 < it.outcome_weight(0.0)           # a bad span all but vanishes
    # train on it (outcome-weighted), guarded by a short headless run: the policy scores at least as well
    seeds, seconds = (1000, 1001), 15.0

    def score(params):
        runs = [pt.fly(name, s, pt.student(name, params), seconds, record=False)["score"] for s in seeds]
        return float(np.mean(runs))

    live = ap.live_params(name)
    old = {n: v.copy() for n, v in live.items()}
    before = score(live)
    trainer = it.IdleTrainer(remote=False, steps_per_pass=24, chunk_steps=8, log=lambda *_: None, self_weight=1.0,
                             headless_guard=lambda g, o, n: score(n) >= score(o))
    record = trainer.train_game(name)
    assert record["self_spans"] >= 3 and record["status"] in ("adopted", "kept")
    after = score(ap.live_params(name))
    assert after >= before, (before, after, record)


def _trace_rewards(name, seed, ticks, drive, monkeypatch):
    monkeypatch.setattr(ap, "SPAN_ROWS", 12)
    monkeypatch.setattr(ap, "STALL_PENALTY", 1e6)           # unmistakable in the span rewards
    decl = pd.DECLS[name]
    game = decl.make(seed)
    game.control({}); game.tick()
    teacher = decl.make_teacher(game)
    trace = ap.PlayTrace()
    for i in range(ticks):
        controls = decl.controls_from(game, teacher.act(game)) if drive else {}
        game.control({"inputs": [{"seq": 1 + i, "dur": decl.step, **controls}]})
        game.tick()
        trace.record(game, controls, demo=True)
    return [r for (_x, _y, _tag, r) in trace.pending]


def test_loitering_in_place_is_penalised_and_travel_is_not(monkeypatch):
    still = _trace_rewards("side_scroller", 1003, 200, False, monkeypatch)
    late = still[len(still) // 2:]                                    # after STALL_SECONDS of history
    assert late and all(r < -5e5 for r in late)                       # standing still: charged
    moving = _trace_rewards("side_scroller", 1003, 200, True, monkeypatch)
    assert len(moving) > 4
    assert all(r > -5e5 for r in moving)                              # running the course: never charged


def test_stall_penalty_compares_against_recent_average():
    hist = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0)]
    assert ap.stall_penalty(hist[:2], (0.5, 0.0), 3) == 0.0           # too little history
    assert ap.stall_penalty(hist, (0.6, 0.2), 3) == ap.STALL_PENALTY  # pacing around the same spot
    assert ap.stall_penalty(hist, (5.0, 0.0), 3) == 0.0               # moved on


def test_time_trickle_charges_at_the_usual_score_rate(monkeypatch):
    monkeypatch.setattr(ap, "SCORE_RATE", {})
    assert ap.time_trickle("g", 20.0, 2.0) == 20.0                  # first span: the usual rate is its own
    for _ in range(200):
        ap.time_trickle("g", 20.0, 2.0)                             # settles at 10 points / s
    fast = 40.0 - ap.time_trickle("g", 40.0, 2.0)                   # twice the usual rate
    slow = 0.0 - ap.time_trickle("g", 0.0, 2.0)                     # nothing scored: pays for the time
    assert fast > 15.0 and slow < -15.0

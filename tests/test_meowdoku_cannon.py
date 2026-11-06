"""The tracking cannon: the lock-on, the gun mount and the shots that leave along it.

* ``entity_motion.aim_solution`` and the extended ``shot_spawn`` are ONE text for the server's spans and the page's
  WebAssembly cores; the cores agree with the span laws to the host's libm (``atan2``) and the shots are exact.
* ``scroller_step`` locks the best live enemy in the cone with hysteresis, slews the gun mount on ``damped_axis``
  within its rate, converges it on the aim solution, and fires along it.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

import src.common.tensors.numpy_backend  # noqa: E402,F401  (the span laws run on the NumPy backend)
from meowdoku import entity_motion as em  # noqa: E402
from meowdoku import scroller as sc  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

SHELL = ROOT / "examples/mobile_reference_app/web/turing-wasm-shell.js"
CORES = ROOT / "build" / "meowdoku_page_cores"
STEMS = {stem: (fn, names, outs) for fn, names, outs, stem in em.PREDICTORS}
R = sc.RADIUS

PARITY_JS = """
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const [dir, shell, casesPath] = process.argv.slice(2);
const { TuringWasmProgram } = await import(pathToFileURL(shell).href);
const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
const report = {};
for (const [stem, list] of Object.entries(cases)) {
  const program = await TuringWasmProgram.fromBytes(fs.readFileSync(`${dir}/${stem}.wasm`), JSON.parse(fs.readFileSync(`${dir}/${stem}.abi.json`, 'utf8')));
  let worst = 0, exact = 0, total = 0;
  for (const [inputs, expected] of list) {
    const got = program.invoke(inputs);
    for (const [name, value] of Object.entries(expected)) { const d = Math.abs(got[name] - value); total++; if (d === 0) exact++; worst = Math.max(worst, d); }
  }
  report[stem] = { total, exact, worst };
}
console.log(JSON.stringify(report));
"""


def _T(values):
    return AbstractTensor.get_tensor(np.asarray(values, dtype=np.float64))


def _cases(stem, feeds):
    fn, names, outs = STEMS[stem]
    result = fn(*[_T(feeds[n]) for n in names])
    result = result if isinstance(result, tuple) else (result,)
    got = {o: np.asarray(r.numpy(), dtype=np.float64).reshape(-1) for o, r in zip(outs, result)}
    n = len(next(iter(feeds.values())))
    return [({k: float(v[i]) for k, v in feeds.items()}, {k: float(v[i]) for k, v in got.items()}) for i in range(n)]


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's cores")
def test_the_aim_and_shot_cores_are_the_span_laws(tmp_path):
    CORES.mkdir(parents=True, exist_ok=True)
    for stem in ("aim", "shot"):
        fn, names, outs = STEMS[stem]
        em._build_one(CORES, fn, names, outs, stem)
    rng = np.random.default_rng(11)
    n = 400
    heading = rng.uniform(0, 2 * np.pi, n)
    X, Y = rng.uniform(-6, 6, n), rng.uniform(-1, 12, n)
    aim = {"X": X, "Y": Y, "X2": X + 0.1 * np.sin(heading), "Y2": Y + 0.1 * np.cos(heading), "step": np.full(n, sc.AIM_STEP),
           "tspd": rng.uniform(0, 30, n), "shot_spd": sc.SHOT_SPEED + rng.uniform(2.5, 7, n), "dz": rng.uniform(-3.5, 3.5, n),
           "tvz": rng.uniform(-1, 1, n), "slope": np.full(n, sc.AIM_SLOPE), "pitch": rng.uniform(-1, 1, n),
           "yaw_half": np.full(n, sc.MOUNT_TRAVEL[0]), "pitch_half": np.full(n, sc.MOUNT_TRAVEL[1])}
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    t = rng.normal(size=(n, 3)); t -= (t * u).sum(axis=1, keepdims=True) * u; t /= np.linalg.norm(t, axis=1, keepdims=True)
    shot = {"ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2], "hx": t[:, 0], "hy": t[:, 1], "hz": t[:, 2],
            "c": rng.uniform(-2.5, 2.5, n), "fan": rng.integers(0, 4, n).astype(float), "laser": rng.choice([0.0, 1.0], n),
            "ship_spd": rng.uniform(2.5, 7, n), "pitch": rng.uniform(-1, 1, n), "ahead": np.full(n, sc.MUZZLE),
            "spacing": np.full(n, sc.MUZZLE_SPACING), "fan_step": np.full(n, sc.FAN_STEP), "speed": np.full(n, sc.SHOT_SPEED),
            "laser_mult": np.full(n, sc.LASER_SPEED), "slope": np.full(n, sc.AIM_SLOPE), "radius": np.full(n, R),
            "mount_yaw": rng.uniform(-0.6, 0.6, n), "mount_pitch": rng.uniform(-1, 1, n)}
    (tmp_path / "cases.json").write_text(json.dumps({"aim": _cases("aim", aim), "shot": _cases("shot", shot)}), encoding="utf-8")
    script = tmp_path / "check.mjs"
    script.write_text(PARITY_JS, encoding="utf-8")
    run = subprocess.run(["node", str(script), str(CORES), str(SHELL), str(tmp_path / "cases.json")], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["shot"]["exact"] == report["shot"]["total"], report          # arithmetic only: bit for bit
    assert report["aim"]["worst"] < 1e-12, report                              # (the host's atan2 and sqrt)


def test_the_mount_is_the_spreads_rotation_and_the_pitch_commands_climb():
    one = lambda v: _T([v])  # noqa: E731
    base = [one(v) for v in (0.0, 0.0, 1.0, 0.0, 1.0, 0.0)]
    args = lambda c, fan, mount, mpitch: (*base, one(c), one(fan), one(0.0), one(4.0), one(0.0),   # noqa: E731
                                          one(sc.MUZZLE), one(0.0), one(sc.FAN_STEP), one(sc.SHOT_SPEED), one(2.0),
                                          one(sc.AIM_SLOPE), one(R), one(mount), one(mpitch))
    free = em.shot_spawn(*args(0.0, 0.0, 0.0, 0.0))
    # a mount of y is the rotation a spread of y makes (no gun spacing here: the same rotation)
    spread = em.shot_spawn(*base, one(1.0), one(1.0), one(0.0), one(4.0), one(0.0), one(sc.MUZZLE), one(0.0), one(0.37),
                           one(sc.SHOT_SPEED), one(2.0), one(sc.AIM_SLOPE), one(R), one(0.0), one(0.0))
    mounted = em.shot_spawn(*base, one(0.0), one(0.0), one(0.0), one(4.0), one(0.0), one(sc.MUZZLE), one(0.0), one(sc.FAN_STEP),
                            one(sc.SHOT_SPEED), one(2.0), one(sc.AIM_SLOPE), one(R), one(0.37), one(0.0))
    for a, b in zip(spread[:6], mounted[:6]):
        assert float(a.numpy().reshape(-1)[0]) == float(b.numpy().reshape(-1)[0])
    # the shot's yaw off the nose is the mount's angle (right positive): h = (0,1,0) at the pole, right = +x
    hx, hy = float(mounted[3].numpy().reshape(-1)[0]), float(mounted[4].numpy().reshape(-1)[0])
    assert math.atan2(hx, hy) == pytest.approx(2.0 * math.atan(0.37 / 2.0), abs=2e-4)      # (the spread's Cayley rotation)
    # the climb is the slope per unit of (pitch command + mount pitch)
    pitched = em.shot_spawn(*args(0.0, 0.0, 0.0, 0.5))
    assert float(pitched[7].numpy().reshape(-1)[0]) == pytest.approx(float(pitched[6].numpy().reshape(-1)[0]) * sc.AIM_SLOPE * 0.5)
    assert float(free[7].numpy().reshape(-1)[0]) == 0.0


# ------------------------------------------------------------------ the law, stepped on controlled columns


@pytest.fixture(scope="module")
def law():
    """The span law (NumPy) and a copy of a running game's columns (weights and all) to build scenes from."""
    for parent in ROOT.parents:                   # engine_toy (the trajectory law) sits beside the repository
        if (parent / "engine_toy").exists() and str(parent / "engine_toy") not in sys.path:
            sys.path.append(str(parent / "engine_toy"))
            break
    game = sc.BulletScroller("cannon")
    game.learns = False
    game.control({})
    game.tick()
    piece = game.session.pieces[0]
    names = sc.ScrollerPiece.argument_names
    columns = {n: np.array(game.session.column(n), dtype=np.float64) for n in names if n != "dt"}
    return piece, names, columns


def _scene(columns):
    """A copy with no enemies and no chance of any returning, the player healthy and the pilot off."""
    c = {k: v.copy() for k, v in columns.items()}
    lo, hi = sc.ENEMIES
    c["alive"][lo:hi] = 0.0
    c["age"][lo:hi] = -1.0e9
    c["lives"][:] = 1.0e6
    c["energy"][:] = 1.0
    c["fire"][:] = 0.0
    c["net_pilot"][:] = 0.0
    c["spd"][0] = 4.0
    c["lock"][:] = 0.0
    return c


def _frame(c):
    u0 = np.array([c["ux"][0], c["uy"][0], c["uz"][0]])
    h0 = np.array([c["hx"][0], c["hy"][0], c["hz"][0]])
    return u0, h0, np.cross(h0, u0)


def _put(c, lane, X, Y, z, heading=0.0, spd=0.0):
    """Enemy ``lane`` live at patch point (X, Y) about the player, flying ``heading`` off the player's nose (right positive)."""
    u0, h0, r0 = _frame(c)
    d = math.hypot(X, Y)
    a = d / R
    direction = (X * r0 + Y * h0) / d
    u = math.cos(a) * u0 + math.sin(a) * direction
    g = math.sin(heading) * r0 + math.cos(heading) * h0
    g = g - (g @ u) * u
    g /= np.linalg.norm(g)
    for name, v in zip(("ux", "uy", "uz"), u):
        c[name][lane] = v
    for name, v in zip(("hx", "hy", "hz"), g):
        c[name][lane] = v
    c["alive"][lane], c["z"][lane], c["spd"][lane], c["vz"][lane], c["hp"][lane] = 1.0, z, spd, 0.0, 1.0e6
    c["kind"][lane] = 0.0


def _step(law, c, dt=1.0 / 60.0):
    piece, names, _ = law
    out = piece.product(*[c[n] for n in names if n != "dt"], np.asarray(dt))
    nxt = {k: v.copy() for k, v in c.items()}
    for name, value in zip(piece.output_names, out):
        if name.endswith("_next") and name[:-5] in nxt:
            nxt[name[:-5]] = np.array(np.broadcast_to(np.asarray(value, dtype=np.float64), (sc.LANES,)))
    return nxt


def _locked(c):
    return int(round(c["lock"][0]))


A, B = sc.ENEMIES[0], sc.ENEMIES[0] + 1
PZ = sc.DEPTH / 2.0


def test_the_lock_takes_the_best_enemy_in_the_cone_and_only_in_front(law):
    c = _scene(law[2])
    _put(c, A, 0.0, 8.0, c["z"][0])
    assert _locked(_step(law, c)) == A
    # behind, beyond the range, and outside the cone (a yaw past the half-angle) are never locked
    for X, Y in ((0.0, -6.0), (0.0, sc.LOCK_RANGE + 1.0), (8.0 * math.tan(sc.LOCK_CONE[0] + 0.12), 8.0), (-6.0, 4.0)):
        c = _scene(law[2])
        _put(c, A, X, Y, c["z"][0])
        assert _locked(_step(law, c)) == 0, (X, Y)
    # inside the cone just short of the edge: locked; the height difference counts as pitch against the cone's height
    c = _scene(law[2])
    _put(c, A, 8.0 * math.tan(sc.LOCK_CONE[0] - 0.1), 8.0, c["z"][0])
    assert _locked(_step(law, c)) == A
    c = _scene(law[2])
    c["z"][0] = 0.3
    _put(c, A, 0.0, 2.0, 0.3 + 0.99 * sc.LOCK_CONE[1] * 2.0 * sc.AIM_SLOPE)             # a pitch just inside the cone's height
    assert _locked(_step(law, c)) == A
    _put(c, A, 0.0, 2.0, 0.3 + 1.05 * sc.LOCK_CONE[1] * 2.0 * sc.AIM_SLOPE)
    assert _locked(_step(law, c)) == 0
    # the nearest, most central wins
    c = _scene(law[2])
    _put(c, A, 0.0, 9.0, c["z"][0])
    _put(c, B, 4.0, 5.0, c["z"][0])
    q = (math.atan2(4.0, 5.0) / sc.LOCK_CONE[0]) ** 2
    expected = A if 9.0 < math.hypot(4.0, 5.0) * (1 + sc.LOCK_WEIGHT * q) else B
    assert _locked(_step(law, c)) == expected


def test_the_lock_holds_until_another_is_clearly_better_or_it_leaves_or_dies(law):
    c = _scene(law[2])
    z = c["z"][0]
    _put(c, A, 0.0, 8.0, z)
    c = _step(law, c)
    assert _locked(c) == A
    # another a little better (score 6 against 8: above the switch fraction): the lock holds
    assert 6.0 / 8.0 > sc.LOCK_SWITCH
    c = _scene(law[2]); c["lock"][0] = A
    _put(c, A, 0.0, 8.0, z); _put(c, B, 0.0, 6.0, z)
    assert _locked(_step(law, c)) == A
    # clearly better (4.5 / 8 under it): the lock switches
    assert 4.5 / 8.0 < sc.LOCK_SWITCH
    _put(c, B, 0.0, 4.5, z)
    assert _locked(_step(law, c)) == B
    # the held lock may stray a little past the cone's edge (LOCK_KEEP) ...
    edge = 8.0 * math.tan(sc.LOCK_CONE[0] * 1.15)
    c = _scene(law[2]); c["lock"][0] = A
    _put(c, A, edge, 8.0, z)
    assert _locked(_step(law, c)) == A
    # ... but a fresh target there is not locked (the same place, no held lock), and the held lock far outside drops
    c["lock"][0] = 0.0
    assert _locked(_step(law, c)) == 0
    c["lock"][0] = A
    _put(c, A, 8.0 * math.tan(sc.LOCK_CONE[0] * 1.6), 8.0, z)
    assert _locked(_step(law, c)) == 0
    # a dead lock drops (to the next best if there is one)
    c = _scene(law[2]); c["lock"][0] = A
    _put(c, B, 0.0, 9.0, z)
    assert _locked(_step(law, c)) == B
    c["alive"][B] = 0.0
    assert _locked(_step(law, c)) == 0
    # no lock without the player alive
    c = _scene(law[2]); _put(c, A, 0.0, 8.0, z); c["alive"][0] = 0.0
    assert _locked(_step(law, c)) == 0


def test_the_mount_slews_within_its_rate_and_converges_on_the_lead_solution(law):
    c = _scene(law[2])
    z = float(c["z"][0])
    X, Y, heading, spd = 4.0, 9.0, math.pi / 2, 3.0           # a crossing enemy: to the right, flying right across the nose
    _put(c, A, X, Y, z, heading, spd)
    c["lock"][0] = A
    dt = 1.0 / 60.0
    yaws, pitches = [], []
    for _ in range(120):
        # the puppet: the same relative place, heading and speed every step (a steady crossing target)
        _put(c, A, X, Y, z + 1.0, heading, spd)
        c = _step(law, c, dt)
        yaws.append(float(c["mount_yaw"][0])); pitches.append(float(c["mount_pitch"][0]))
        assert _locked(c) == A
    steps = np.abs(np.diff([0.0] + yaws)) / dt
    assert steps.max() <= sc.MOUNT_RATE[0] * (1 + 1e-9), steps.max()                 # rate-limited, never a snap
    assert steps.max() > 0.5 * sc.MOUNT_RATE[0]                                      # ... and it does slew at speed
    psteps = np.abs(np.diff([0.0] + pitches)) / dt
    assert psteps.max() <= sc.MOUNT_RATE[1] * (1 + 1e-9)
    assert abs(yaws[-1]) <= sc.MOUNT_TRAVEL[0] and abs(pitches[-1]) <= sc.MOUNT_TRAVEL[1]
    # converged on the aim solution of the state it ended in (the law's own, one text)
    _put(c, A, X, Y, z + 1.0, heading, spd)
    u0, h0, _ = _frame(c)
    tu = np.array([c["ux"][A], c["uy"][A], c["uz"][A]]); th = np.array([c["hx"][A], c["hy"][A], c["hz"][A]])
    project = lambda p: [float(v.numpy().reshape(-1)[0]) for v in em.sphere_project(  # noqa: E731
        *[_T([x]) for x in p], *[_T([x]) for x in u0], *[_T([x]) for x in h0], _T([R]))]
    t2 = em.sphere_advance(*[_T([x]) for x in tu], *[_T([x]) for x in th], _T([sc.AIM_STEP]), _T([R]))
    X1, Y1 = project(tu)
    X2, Y2 = project([float(v.numpy().reshape(-1)[0]) for v in t2[:3]])
    air = float(em.air_gain(_T([z + 1.0]), _T([sc.AIR_FALL]), _T([sc.AIR_GAIN]))[0].numpy().reshape(-1)[0])
    solution = em.aim_solution(*[_T([v]) for v in (X1, Y1, X2, Y2, sc.AIM_STEP, spd * air, sc.SHOT_SPEED + c["spd"][0], 1.0, 0.0,
                                                    sc.AIM_SLOPE, 0.0, sc.MOUNT_TRAVEL[0], sc.MOUNT_TRAVEL[1])])
    yaw_aim = float(solution[0].numpy().reshape(-1)[0]) - sc.MOUNT_TRAVEL[0]
    assert yaws[-1] == pytest.approx(yaw_aim, abs=2e-3)
    assert yaws[-1] > 0.0 and 2.0 * math.atan(yaw_aim / 2.0) > math.atan2(X, Y)      # led: past the target, towards where it goes
    # and with no lock the mount comes home to the nose
    c["lock"][0] = 0.0
    c["alive"][A] = 0.0
    for _ in range(90):
        c = _step(law, c, dt)
    assert abs(c["mount_yaw"][0]) < 1e-3 and abs(c["mount_pitch"][0]) < 1e-3


def test_the_shots_leave_along_the_mount(law):
    c = _scene(law[2])
    c["mount_yaw"][0], c["mount_pitch"][0] = 0.3, 0.4
    c["fire"][0], c["fire_mode"][:] = 1.0, sc.AUTO
    c["mult"][0] = 0.0
    u0, h0, r0 = _frame(c)
    n = _step(law, c)
    lo, hi = sc.SHOTS
    born = np.flatnonzero((n["alive"][lo:hi] > 0) & (c["alive"][lo:hi] == 0)) + lo
    assert len(born) >= 1
    for k in born:
        h = np.array([n["hx"][k], n["hy"][k], n["hz"][k]])
        u = np.array([n["ux"][k], n["uy"][k], n["uz"][k]])
        # (the first sub-step has also flown the shot a hair: measure the heading's angle off the ship's in the shot's own frame)
        right = np.cross(h0, u)
        assert math.atan2(h @ right, h @ h0) == pytest.approx(2.0 * math.atan(0.3 / 2.0), abs=5e-3)
        assert n["vz"][k] / n["spd"][k] == pytest.approx(sc.AIM_SLOPE * 0.4, abs=1e-9)

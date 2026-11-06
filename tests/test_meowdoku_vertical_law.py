"""``entity_motion.vertical_axis``: the one vertical law of the climber, the runner and Delve (and the page's prediction).

* the page's WebAssembly core equals the span law (NumPy) on random states;
* the law integrates EXACTLY: the held jump's apex is the declared one, and the state after a given time does not
  depend on the step length;
* the games' gravity is the real constant in their units (``G_EARTH / UNIT_M``), their jumps are declared by height;
* the native (compiled) lane of each changed law agrees with the NumPy product (``MEOWDOKU_LAW=eigen`` runs only
  when asked: ``MEOWDOKU_PARITY_NATIVE=1``, one compile at a time).
"""

from __future__ import annotations

import json
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

import src.common.tensors.numpy_backend  # noqa: E402,F401
from meowdoku import entity_motion as em  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

SHELL = ROOT / "examples/mobile_reference_app/web/turing-wasm-shell.js"
CORES = ROOT / "build" / "meowdoku_page_cores"
STEMS = {stem: (fn, names, outs) for fn, names, outs, stem in em.PREDICTORS}
G = em.G_EARTH / 0.35
APEX, HANG_V, HANG_SCALE, CUT, FALL_MAX, COYOTE, BUFFER = 2.45, 2.0, 0.5, 3.5, 18.0, 0.09, 0.11
J = em.jump_speed(APEX, G, HANG_V, HANG_SCALE)

PARITY_JS = """
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const [dir, shell, casesPath] = process.argv.slice(2);
const { TuringWasmProgram } = await import(pathToFileURL(shell).href);
const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
const program = await TuringWasmProgram.fromBytes(fs.readFileSync(`${dir}/vertical.wasm`), JSON.parse(fs.readFileSync(`${dir}/vertical.abi.json`, 'utf8')));
let worst = 0, exact = 0, total = 0;
for (const [inputs, expected] of cases) {
  const got = program.invoke(inputs);
  for (const [name, value] of Object.entries(expected)) { const d = Math.abs(got[name] - value); total++; if (d === 0) exact++; worst = Math.max(worst, d); }
}
console.log(JSON.stringify({ total, exact, worst }));
"""


def _law(y, vy, held, prev, ground, coy, buf, dt):
    return em.vertical_axis(y, vy, held, prev, ground, coy, buf, dt, G, J, CUT, FALL_MAX, HANG_V, HANG_SCALE, COYOTE, BUFFER)


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's cores")
def test_the_vertical_core_is_the_span_law(tmp_path):
    fn, names, outs = STEMS["vertical"]
    CORES.mkdir(parents=True, exist_ok=True)
    em._build_one(CORES, fn, names, outs, "vertical")
    rng = np.random.default_rng(11)
    n = 600
    feeds = {
        "y": rng.uniform(-3, 3, n), "vy": rng.uniform(-14, 22, n), "held": rng.integers(0, 2, n) * 1.0,
        "prev": rng.integers(0, 2, n) * 1.0, "ground": rng.integers(0, 2, n) * 1.0,
        "coy": rng.uniform(0, 0.09, n) * rng.integers(0, 2, n), "buf": rng.uniform(0, 0.11, n) * rng.integers(0, 2, n),
        "dt": rng.uniform(1e-3, 0.05, n), "gravity": np.full(n, G), "jump": np.full(n, J), "cut": np.full(n, CUT),
        "fall_max": np.full(n, FALL_MAX), "hang_v": np.full(n, HANG_V), "hang_scale": np.full(n, HANG_SCALE),
        "coyote": np.full(n, COYOTE), "buffer": np.full(n, BUFFER),
    }
    result = fn(*[AbstractTensor.get_tensor(np.asarray(feeds[k], dtype=np.float64)) for k in names])
    got = {o: np.asarray(r.numpy(), dtype=np.float64).reshape(-1) for o, r in zip(outs, result)}
    cases = [({k: float(v[i]) for k, v in feeds.items()}, {k: float(v[i]) for k, v in got.items()}) for i in range(n)]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    script = tmp_path / "check.mjs"
    script.write_text(PARITY_JS, encoding="utf-8")
    run = subprocess.run(["node", str(script), str(CORES), str(SHELL), str(path)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["worst"] < 1e-12, report


def _fly(dt, seconds):
    y, vy, apex, t = 0.0, 0.0, 0.0, 0.0
    y, vy, coy, buf, _ = _law(y, vy, 1.0, 0.0, 1.0, 0.0, 0.0, 1e-9)
    while t < seconds - 1e-12:
        y, vy, coy, buf, _ = _law(y, vy, 1.0, 1.0, 0.0, coy, buf, dt)
        t += dt
        apex = min(apex, y)
    return y, vy, apex


def test_a_jump_is_exact_at_any_step_length():
    runs = [_fly(dt, 0.6) for dt in (1 / 600, 1 / 240, 1 / 120, 1 / 60, 0.05, 0.1, 0.2, 0.3)]
    for y, vy, _ in runs[1:]:
        assert abs(y - runs[0][0]) < 1e-6 and abs(vy - runs[0][1]) < 1e-6
    assert abs(-_fly(1 / 1200, 1.0)[2] - APEX) < 1e-4                       # the declared held-button apex


def test_release_cuts_the_rise_terminal_velocity_holds_and_the_press_is_buffered():
    _, vy, *_ = _law(0.0, -12.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01)
    assert abs(vy + CUT - G * 0.01) < 1e-9                                  # cut to -CUT, then one step of gravity
    y, vy, *_ = _law(0.0, 17.9, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    t_ramp = 0.1 / G
    assert vy == FALL_MAX and abs(y - (17.9 * t_ramp + 0.5 * G * t_ramp ** 2 + FALL_MAX * (1.0 - t_ramp))) < 1e-9
    # a press in the air is held for BUFFER seconds and launches on touching down (ground = 1)
    _, vy, coy, buf, jumped = _law(0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    assert jumped == 0.0 and buf > 0.0
    _, vy, coy, buf, jumped = _law(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, buf, 0.01)
    assert jumped == 1.0 and vy < -J + 1.0
    # coyote: leaving a ledge, a press within COYOTE still launches; after it, not
    _, _, coy, _, _ = _law(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.01)
    assert coy == COYOTE
    _, _, _, _, jumped = _law(0.0, 0.5, 1.0, 0.0, 0.0, coy - 0.02, 0.0, 0.01)
    assert jumped == 1.0
    _, _, _, _, jumped = _law(0.0, 0.5, 1.0, 0.0, 0.0, 0.005, 0.0, 0.01)
    assert jumped == 0.0 or jumped == 1.0                                   # (0.005 - 0.01 -> expired only after this step)
    _, _, _, _, jumped = _law(0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    assert jumped == 0.0


def test_gravity_is_the_real_constant_in_each_games_units():
    from meowdoku import delve_rules, platformer, sidescroller
    for module in (delve_rules, platformer, sidescroller):
        assert abs(module.GRAVITY * module.UNIT_M - em.G_EARTH) < 1e-12
    # the declared apexes are the integrated ones
    for module, apex, hv, hs in ((delve_rules, delve_rules.JUMP_APEX, delve_rules.HANG_V, delve_rules.HANG_SCALE),
                                 (sidescroller, sidescroller.JUMP_APEX, sidescroller.HANG_V, sidescroller.HANG_SCALE)):
        j, g = module.JUMP, module.GRAVITY
        top, y, vy, coy, buf = 0.0, 0.0, 0.0, 0.0, 0.0
        y, vy, coy, buf, _ = em.vertical_axis(y, vy, 1.0, 0.0, 1.0, 0.0, 0.0, 1e-9, g, j, 3.5, module.FALL_MAX, hv, hs, 0.09, 0.11)
        for _ in range(4000):
            y, vy, coy, buf, _ = em.vertical_axis(y, vy, 1.0, 1.0, 0.0, coy, buf, 1 / 2400, g, j, 3.5, module.FALL_MAX, hv, hs, 0.09, 0.11)
            top = min(top, y)
        assert abs(-top - apex) < 1e-3
    assert abs(platformer.JUMP ** 2 / (2 * platformer.GRAVITY) - platformer.BOUNCE_APEX) < 1e-9

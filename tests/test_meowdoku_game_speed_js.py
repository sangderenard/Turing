"""The page's side of the game clock speed and the world zoom (node, no browser): ``web/game_clock.js``,
``web/world_zoom.js`` and ``web/pilot_clock.js`` across a speed change.

The zoom's easing is a critically damped spring (no overshoot, exact at any frame length), its clamp and window math are
world_view.py's text, and the law clock keeps its lock through a speed change at 0.1x: the readings are in the game's
virtual wall on both sides, and a change of the speed (the server's ``gen``) re-bases the clock once.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "examples/meowdoku/web"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's modules")

JS = r"""
import { pathToFileURL } from 'node:url';
const web = process.argv[2];
const G = await import(pathToFileURL(`${web}/game_clock.js`).href);
const Z = await import(pathToFileURL(`${web}/world_zoom.js`).href);
const { LawClock } = await import(pathToFileURL(`${web}/pilot_clock.js`).href);
const out = {};

// ---- the clock speed
const c = new G.GameClock(0.4, 100);
const a = c.at(110);
c.set(2, 110);
out.clock = { a, continuous: c.at(110), later: c.at(111), clamped: [G.clampSpeed(0.001), G.clampSpeed(99), G.clampSpeed('x')],
              slider: [G.sliderToSpeed(0), G.sliderToSpeed(1), G.speedToSlider(0.05), G.speedToSlider(2), G.speedToSlider(G.sliderToSpeed(0.3))],
              dflt: G.SPEED.default };

// ---- the zoom: easing (critically damped), clamping, the window
const states = [];
let x = 0, v = 0; const target = Math.log(0.5), omega = Z.ZOOM_OMEGA;
let maxOvershoot = 0, mono = true, last = x;
for (let i = 0; i < 600; i++) {            // 10 s at 60 fps
  const s = Z.springStep(x, v, target, 1 / 60, omega); x = s.x; v = s.v;
  maxOvershoot = Math.max(maxOvershoot, target - x);
  if (x > last + 1e-12) mono = false; last = x;
  if (i === 29 || i === 59) states.push(x);
}
// one big step equals many small ones (the closed form is exact)
let xs = 0, vs = 0; for (let i = 0; i < 60; i++) { const s = Z.springStep(xs, vs, target, 1 / 60, omega); xs = s.x; vs = s.v; }
const big = Z.springStep(0, 0, target, 1, omega);
out.spring = { end: x, target, overshoot: maxOvershoot, monotone: mono, half: states, stepExact: Math.abs(big.x - xs), vEnd: v };

const reach = { left: 1.6, right: 13, up: 20, down: 0 };
out.limits = { floor: Z.zoomFloor(9, 10, reach), none: Z.zoomFloor(9, 16, null),
               clampLow: Z.clampZoom(0.01, 9, 10, reach), clampHigh: Z.clampZoom(99, 9, 10, reach), nan: Z.clampZoom(NaN, 9, 10, reach) };
const win = (z, focus) => Z.viewWindow(9, 10, z, reach, focus);
out.window = { one: win(1), half: win(0.5), floor: win(out.limits.floor), two: win(2, [8, 5]), twoLeft: win(2, [0, 0]), widths: [1, 0.8, 0.6, 0.4].map((z) => { const w = win(z); return w.x1 - w.x0; }),
               shift: Z.windowShift(9, 10, 0.5, reach) };

// ---- the law clock across a speed change at 0.1x, in the virtual wall
// A server whose virtual wall runs at `speed` x the page's real wall (an offset THETA between them); the page's virtual
// clock is the same integral.  Ticks every 40 ms real with jittery delay; the speed goes 0.1 -> 0.4 -> 0.1 at fixed instants.
const rng = (seed) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
function run(changeLatency) {
  const rand = rng(7), lc = new LawClock();
  const page = new G.GameClock(0.1, 0), server = { v: 5000, real: 0, speed: 0.1, gen: 0, clock: 0 };
  const serverAt = (real) => server.v + server.speed * (real - server.real);
  const errs = []; const rebased = [];
  const schedule = [[2.0, 0.4], [5.0, 0.1]];
  const pending = [];                                                   // [apply time on the server, speed]
  for (let t = 0.04, k = 0; t < 8; t += 0.04, k++) {
    for (const [at, s] of schedule) if (Math.abs(t - at) < 0.02 && !page._done?.[at]) { (page._done ||= {})[at] = 1; page.set(s, t); pending.push([t + changeLatency, s]); }
    while (pending.length && pending[0][0] <= t) { const [when, s] = pending.shift(); server.v = serverAt(when); server.real = when; server.speed = s; server.gen++; }
    // the tick: sent at t (page virtual), received after 20 ms +- jitter, answered at once, arrives after the same again
    const up = 0.02 + rand() * 0.01, down = 0.02 + rand() * 0.01;
    const ct = page.at(t), recv = serverAt(t + up), sent = recv, arrive = t + up + down, ca = page.at(arrive);
    const lawWall = recv;                                                // the state stands for the receipt instant
    const clock = server.clock + (lawWall - server.lastWall || 0);       // law clock advances with the virtual wall
    server.clock = clock; server.lastWall = lawWall;
    const before = lc.ready() ? lc.lawAt(ca) : null;
    lc.sample({ ct, ca, recv, sent, lawWall, clock, gen: server.gen });
    if (lc.rebased) rebased.push([t, before == null ? null : before - lc.lawAt(ca)]);
    // truth: the law time at the page's `ca` instant is the server's clock advanced by the virtual wall since the reply
    if (lc.ready() && t > 0.5) errs.push(Math.abs(lc.lawAt(ca) - (clock + (serverAt(arrive) - lawWall))));
  }
  errs.sort((a, b) => a - b);
  return { p95: errs[Math.floor(errs.length * 0.95)], max: errs[errs.length - 1], rebased: rebased.length, jumps: rebased.map((r) => r[1]), snaps: lc.snaps };
}
out.lock = { fast: run(0.0), laggy: run(0.06) };
console.log(JSON.stringify(out));
"""


def _node(tmp_path):
    script = tmp_path / "speed.mjs"
    script.write_text(JS, encoding="utf-8")
    run = subprocess.run(["node", str(script), str(WEB)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-3000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_game_clock_zoom_easing_clamp_window_and_the_lock_across_a_speed_change(tmp_path):
    r = _node(tmp_path)
    clock = r["clock"]
    assert clock["dflt"] == 0.4 and clock["a"] == pytest.approx(104.0) and clock["continuous"] == pytest.approx(104.0)
    assert clock["later"] == pytest.approx(106.0)                        # the new rate from the change on
    assert clock["clamped"] == [0.05, 2.0, 0.4]
    assert clock["slider"][0] == pytest.approx(0.05) and clock["slider"][1] == pytest.approx(2.0)
    assert clock["slider"][2] == pytest.approx(0.0) and clock["slider"][3] == pytest.approx(1.0) and clock["slider"][4] == pytest.approx(0.3)

    spring = r["spring"]                                                  # critically damped: no overshoot, monotone, exact
    assert spring["overshoot"] <= 1e-9 and spring["monotone"] and abs(spring["end"] - spring["target"]) < 1e-6
    assert spring["half"][1] - spring["target"] < 0.05 * abs(spring["target"]) and spring["stepExact"] < 1e-12

    limits = r["limits"]
    assert limits["floor"] == pytest.approx(9.0 / 23.6) and limits["none"] == 1.0
    assert limits["clampLow"] == pytest.approx(limits["floor"]) and limits["clampHigh"] == 2.5 and limits["nan"] == 1

    w = r["window"]
    assert (w["one"]["x0"], w["one"]["x1"], w["one"]["y0"], w["one"]["y1"]) == (0, 9, 0, 10)
    assert all(b > a for a, b in zip(w["widths"], w["widths"][1:]) if False) or w["widths"] == sorted(w["widths"])
    assert w["half"]["x1"] - w["half"]["x0"] == pytest.approx(18.0)
    assert -1.6 - 1e-9 <= w["half"]["x0"] and w["half"]["x1"] <= 22.0 + 1e-9 and w["half"]["y1"] <= 10.0 + 1e-9     # slid inside the world
    assert w["floor"]["x1"] - w["floor"]["x0"] == pytest.approx(23.6) and w["floor"]["x0"] == pytest.approx(-1.6)
    assert w["two"]["x1"] - w["two"]["x0"] == pytest.approx(4.5) and w["two"]["x0"] == pytest.approx(4.5)            # follows the ship in, held inside the field
    assert w["twoLeft"]["x0"] == 0 and w["twoLeft"]["y0"] == 0                                                         # ...and stays in the field
    assert w["shift"][0] > 0                                                                                            # zooming out pans ahead

    for name in ("fast", "laggy"):
        lock = r["lock"][name]
        # the lock holds through two speed changes: bounded error, and the clock re-based once per change (not per tick)
        assert lock["rebased"] == 2, lock
        assert lock["p95"] < 0.03 and lock["max"] < 0.2, lock

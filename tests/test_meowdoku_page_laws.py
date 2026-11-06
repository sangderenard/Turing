"""The page runs the server's laws, not laws of its own (bullet hell, World Walk).

* ``entity_motion.sphere_project`` is ONE text: the server's batched law runs it on spans, the page's WebAssembly
  core is lowered from it (the natural source -- a quotient of an ``atan2`` -- lowers: the planner types a
  ``math`` float function's result).
* ``entity_motion.fire_gate`` is WHEN a volley may leave: ``scroller_step`` calls it, the page runs its core on the
  server's step grid; in each fire mode the volleys are the same ones.
* The snapshot (``realtime_socket``) carries each body's flight state; the page's decoder and flight code
  (``web/bodies.js``, the very module ``app.js`` imports) carry every body on by the engine laws' cores and put it
  where the server has it.

The cores are built into ``build/meowdoku_page_cores`` (rebuilt only when a law's text changes) and run in node.
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

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's cores")

import src.common.tensors.numpy_backend  # noqa: E402,F401  (the span laws run on the NumPy backend)
from meowdoku import entity_motion as em  # noqa: E402
from meowdoku import realtime_socket as rs  # noqa: E402
from meowdoku import scroller as sc  # noqa: E402
from src.common.tensors import AbstractTensor  # noqa: E402

SHELL = ROOT / "examples/mobile_reference_app/web/turing-wasm-shell.js"
BODIES = ROOT / "examples/meowdoku/web/bodies.js"
CORES = ROOT / "build" / "meowdoku_page_cores"
STEMS = {stem: (fn, names, outs) for fn, names, outs, stem in em.PREDICTORS}

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

BODIES_JS = """
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const [dir, shell, bodiesJs, tracePath] = process.argv.slice(2);
const { TuringWasmProgram } = await import(pathToFileURL(shell).href);
const { decodeSnapshot, BodyFlight, BODY_CORES } = await import(pathToFileURL(bodiesJs).href);
const load = async (s) => TuringWasmProgram.fromBytes(fs.readFileSync(`${dir}/${s}.wasm`), JSON.parse(fs.readFileSync(`${dir}/${s}.abi.json`, 'utf8')));
const flight = new BodyFlight(Object.fromEntries(await Promise.all(BODY_CORES.map(async (s) => [s, await load(s)]))));
const trace = JSON.parse(fs.readFileSync(tracePath, 'utf8'));
const { bodies, sphere, patch } = trace.predict;
const names = [], pieces = new Map();
let worst = 0, worstZ = 0, compared = 0, records = 0;
for (const f of trace.frames) {
  const bytes = Uint8Array.from(Buffer.from(f.hex, 'hex'));
  const d = decodeSnapshot(new DataView(bytes.buffer), names);
  if (d.flags & 2) pieces.clear();
  for (const lane of d.gone) pieces.delete(lane);
  for (const r of d.spawned) { pieces.set(r.lane, BodyFlight.held(r, f.clock)); records++; }
  for (const r of d.updated) { if (pieces.has(r.lane)) { pieces.set(r.lane, BodyFlight.held(r, f.clock)); records++; } }
  for (const [lane, b] of pieces) {
    const truth = f.truth[lane];
    if (!truth || lane === 0) continue;
    flight.carry(b, f.clock - b.at, bodies, sphere); b.at = f.clock;
    const at = flight.patch(b, f.you, patch, sphere.radius);
    worst = Math.max(worst, Math.hypot(at.x - truth[0], at.y - truth[1])); worstZ = Math.max(worstZ, Math.abs(at.z - truth[2])); compared++;
  }
}
console.log(JSON.stringify({ worst, worstZ, compared, records, frames: trace.frames.length }));
"""


def _core(*stems):
    """Build (or reuse) the page's cores for ``stems`` in the persistent directory."""
    CORES.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        fn, names, outs = STEMS[stem]
        em._build_one(CORES, fn, names, outs, stem)


def _node(script, tmp_path, *args):
    path = tmp_path / "check.mjs"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(["node", str(path), *map(str, args)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def _T(values):
    return AbstractTensor.get_tensor(np.asarray(values, dtype=np.float64))


def _cases(stem, feeds):
    fn, names, outs = STEMS[stem]
    result = fn(*[_T(feeds[n]) for n in names])
    result = result if isinstance(result, tuple) else (result,)
    got = {o: np.asarray(r.numpy(), dtype=np.float64).reshape(-1) for o, r in zip(outs, result)}
    n = len(next(iter(feeds.values())))
    return [({k: float(v[i]) for k, v in feeds.items()}, {k: float(v[i]) for k, v in got.items()}) for i in range(n)]


def test_the_projection_is_one_text_for_spans_and_the_page_core(tmp_path):
    _core("project", "heading")
    rng = np.random.default_rng(3)
    n = 300

    def frames(count):
        u = rng.normal(size=(count, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        t = rng.normal(size=(count, 3)); t -= (t * u).sum(axis=1, keepdims=True) * u
        return u, t / np.linalg.norm(t, axis=1, keepdims=True)

    c, ch = frames(n)
    p = c + 0.6 * rng.normal(size=(n, 3)) * rng.uniform(0, 1.5, (n, 1))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    h, _ = frames(n)
    cases = {
        "project": _cases("project", {"ux": p[:, 0], "uy": p[:, 1], "uz": p[:, 2], "cx": c[:, 0], "cy": c[:, 1], "cz": c[:, 2],
                                      "hx": ch[:, 0], "hy": ch[:, 1], "hz": ch[:, 2], "radius": np.full(n, 30.0)}),
        "heading": _cases("heading", {"hx": h[:, 0], "hy": h[:, 1], "hz": h[:, 2], "cx": c[:, 0], "cy": c[:, 1], "cz": c[:, 2],
                                      "gx": ch[:, 0], "gy": ch[:, 1], "gz": ch[:, 2]}),
    }
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    report = _node(PARITY_JS, tmp_path, CORES, SHELL, tmp_path / "cases.json")
    # (the host's atan2 is the browser's libm, NumPy's the C runtime's: they may differ in the last place)
    assert report["project"]["worst"] < 1e-12 and report["heading"]["worst"] < 1e-12, report
    # the server's projection IS the shared law
    from meowdoku import sphere_motion as sm
    u = tuple(_T(p[:, k]) for k in range(3))
    centre, heading = tuple(_T(c[:, k]) for k in range(3)), tuple(_T(ch[:, k]) for k in range(3))
    X, Y = sm.project(u, centre, heading, 30.0)
    X2, Y2 = em.sphere_project(*u, *centre, *heading, 30.0)
    assert np.array_equal(np.asarray(X.numpy()), np.asarray(X2.numpy())) and np.array_equal(np.asarray(Y.numpy()), np.asarray(Y2.numpy()))


def test_the_fire_gate_core_is_the_span_law_and_the_servers_volleys_are_the_gates(tmp_path):
    _core("gate")
    rng = np.random.default_rng(5)
    n = 400
    feeds = {"fire_mode": rng.choice([0.0, 1.0, 2.0, 3.0, 0.4, 1.6, 2.4, 3.2, -1.0], n), "held": rng.choice([0.0, 1.0], n),
             "prev": rng.choice([0.0, 1.0], n), "timer": rng.choice([0.0, -0.01, 0.05, 0.3], n),
             "burst_left": rng.choice([0.0, 1.0, 2.0, 3.0], n), "ready": rng.choice([0.0, 1.0], n),
             "laser": rng.choice([0.0, 1.0], n, p=[0.8, 0.2]), "missile": rng.choice([0.0, 1.0], n, p=[0.8, 0.2]),
             "shots_n": rng.integers(1, 6, n).astype(float), "free": rng.integers(0, 25, n).astype(float),
             "dt": rng.uniform(0.001, 0.05, n), "reload": np.full(n, sc.RELOAD), "missile_reload": np.full(n, sc.MISSILE_RELOAD),
             "burst_count": np.full(n, sc.BURST_N), "cont_factor": np.full(n, sc.CONT_FACTOR), "burst_factor": np.full(n, sc.BURST_FACTOR)}
    (tmp_path / "cases.json").write_text(json.dumps({"gate": _cases("gate", feeds)}), encoding="utf-8")
    report = _node(PARITY_JS, tmp_path, CORES, SHELL, tmp_path / "cases.json")
    assert report["gate"]["exact"] == report["gate"]["total"], report          # bit for bit: arithmetic only

    # The real game, fire held then released, in each mode: the gate -- the page's text, on plain numbers --
    # stepped as the server steps reproduces the volleys scroller_step spawned.
    holds, dt = [(0.0, 0.5), (0.7, 0.75), (0.9, 1.6)], 1.0 / 60.0
    for mode in (sc.CONTINUOUS, sc.AUTO, sc.BURST, sc.SINGLE):
        game = sc.BulletScroller("p")
        game.control({})
        game.tick()
        col = game.session.column
        col("mult")[0], col("fan")[0], col("lives")[:], col("energy")[:] = 1.0, 0.0, 1e6, 1.0
        col("alive")[sc.ENEMIES[0]:sc.ENEMIES[1]] = 0.0
        col("age")[sc.ENEMIES[0]:sc.ENEMIES[1]] = -1e9
        controls = {"move_axis": 0.0, "move_axis_y": 0.0, "move_axis_z": 0.0, "steer_x": -1.0, "steer_y": -1.0, "alt_fire": 0.0,
                    "bomb_style": 0.0, "move_axis_r": 0.0, "fire_mode": mode, "primary_slot": 0.0}
        timer, burst, prev, page, server, seen = 0.0, 0.0, 0.0, [], [], set()
        for i in range(int(2.0 / dt)):
            held = 1.0 if any(a <= i * dt < b for a, b in holds) else 0.0
            game.control({"inputs": [{"seq": i + 1, "dur": dt, "fire": held, **controls}]})
            game.tick()
            live = set(int(k) for k in np.flatnonzero((col("slot") == sc.SHOT) & (col("alive") > 0)))
            if live - seen:
                server.append((i, len(live - seen)))
            seen = live
            volley, burst, timer, count = em.fire_gate(mode, held, prev, timer, burst, 1.0, 0.0, 0.0, 2.0, 24.0 - len(live), dt,
                                                       sc.RELOAD, sc.MISSILE_RELOAD, sc.BURST_N, sc.CONT_FACTOR, sc.BURST_FACTOR)
            prev = held
            if volley > 0.5:
                page.append((i, int(count)))
        assert page == server and server, (mode, page, server)


@pytest.fixture(scope="module")
def trace():
    """A real run of the game, every tick encoded as the socket does, with the server's own bodies."""
    for parent in ROOT.parents:                   # engine_toy (the trajectory law) sits beside the repository
        if (parent / "engine_toy").exists() and str(parent / "engine_toy") not in sys.path:
            sys.path.append(str(parent / "engine_toy"))
            break
    game = sc.BulletScroller("p")
    game.control({})
    game.tick()
    s = game.session
    s.column("owned_m")[:] = 1.0
    s.column("mult")[0], s.column("fan")[0] = 2.0, 1.0
    encoder, frames = rs.SnapshotEncoder(), []
    base = {"move_axis": 0.0, "move_axis_y": 0.0, "move_axis_z": 0.0, "steer_x": -1.0, "steer_y": -1.0, "fire": 0.0, "alt_fire": 0.0,
            "bomb_style": 0.0, "move_axis_r": 0.0, "fire_mode": 1.0, "primary_slot": 0.0}
    for i in range(240):
        t = i / 60.0
        c = dict(base, move_axis_r=0.5 * np.sin(t * 0.8), move_axis=0.3 * np.sin(t * 0.5), move_axis_y=-0.6,
                 fire=1.0, primary_slot=2.0 if 100 <= i < 160 else 0.0, alt_fire=1.0 if i % 40 < 3 else 0.0,
                 bomb_style=float((i // 80) % 3))
        game.control({"inputs": [{"seq": i + 1, "dur": 1.0 / 60.0, **c}]})
        game.tick()
        st = game.state()
        binary, _meta = encoder.encode(game, st)
        you = st["you"]
        truth = {int(str(e["id"]).split(":")[0]): [e["x"], e["y"], e["z"]] for e in st["entities"]}
        frames.append({"clock": st["world_clock"], "hex": binary.hex(), "truth": truth,
                       "you": {k: you[k] for k in ("ux", "uy", "uz", "hx", "hy", "hz")}})
    return {"frames": frames, "predict": game.state()["predict"]}


def test_the_page_carries_every_server_body_to_where_the_server_has_it(trace, tmp_path):
    _core("turn", "advance", "air", "climb", "ballistic", "project", "heading")
    assert all(f["hex"][:2] == format(rs.PROTOCOL, "02x") for f in trace["frames"])
    (tmp_path / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
    report = _node(BODIES_JS, tmp_path, CORES, SHELL, BODIES, tmp_path / "trace.json")
    assert report["compared"] > 1000 and report["records"] > 100, report
    # every body, every frame, through the real frames and the compiled cores: where the server has it
    assert report["worst"] < 1e-4 and report["worstZ"] < 1e-4, report


def test_the_walk_cores_are_the_span_laws_the_server_walks_by(tmp_path):
    """World Walk's walk-speed, pace and route-turn laws: the page's WebAssembly cores give what the server's spans
    (and, for the speed, the NumPy pathing costs) give."""
    _core("walkspeed", "walkdrive", "walkroute")
    rng = np.random.default_rng(11)
    n = 400
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    h = rng.normal(size=(n, 3)); h -= (h * u).sum(axis=1, keepdims=True) * u; h /= np.linalg.norm(h, axis=1, keepdims=True)
    w = u + 0.05 * rng.normal(size=(n, 3)) * rng.uniform(0, 3.0, (n, 1)); w /= np.linalg.norm(w, axis=1, keepdims=True)
    cases = {
        "walkspeed": _cases("walkspeed", {"base": rng.uniform(0.5, 3.5, n), "mat": rng.integers(0, 6, n).astype(float),
                                          "slope": np.round(rng.uniform(0.0, 0.5, n) * 100) / 100}),
        "walkdrive": _cases("walkdrive", {"spd": rng.uniform(-2, 4, n), "top": rng.uniform(0, 4, n), "fwd": rng.uniform(-1, 1, n),
                                          "route": rng.integers(0, 2, n).astype(float), "align": rng.uniform(0, 1, n),
                                          "dt": np.full(n, 1 / 240), "tau": np.full(n, 0.15)}),
        "walkroute": _cases("walkroute", {"ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2], "hx": h[:, 0], "hy": h[:, 1], "hz": h[:, 2],
                                          "wx": w[:, 0], "wy": w[:, 1], "wz": w[:, 2], "route": rng.integers(0, 2, n).astype(float),
                                          "turn_in": rng.uniform(-1, 1, n), "rate": np.full(n, 2.4), "route_rate": np.full(n, 5.0),
                                          "reach": np.full(n, 0.6), "dt": np.full(n, 1 / 240), "radius": np.full(n, 30.0)}),
    }
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    report = _node(PARITY_JS, tmp_path, CORES, SHELL, tmp_path / "cases.json")
    # (the arithmetic laws agree to the bit; the route law's atan2 is the browser's libm against NumPy's)
    assert report["walkspeed"]["worst"] == 0.0 and report["walkdrive"]["worst"] == 0.0, report
    assert report["walkroute"]["worst"] < 1e-12, report
    # the speeds are the declared table, and the NumPy pathing costs run the same text on arrays
    for name, (index, factor) in em.WALK_MATERIALS.items():
        assert em.walk_speed(2.0, np.asarray([float(index)]), np.asarray([0.0]))[0][0] == pytest.approx(2.0 * factor), name
    arr = em.walk_speed(1.0, np.asarray([0.0, 1.0, 4.0]), np.asarray([0.0, 0.0, 0.0]))[0]
    spans = em.walk_speed(_T([1.0] * 3), _T([0.0, 1.0, 4.0]), _T([0.0] * 3))[0]
    assert np.array_equal(arr, np.asarray(spans.numpy()))


WALKER_JS = """
import fs from 'node:fs';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';
const [dir, shell, appPath, casePath] = process.argv.slice(2);
const { TuringWasmProgram } = await import(pathToFileURL(shell).href);
const load = async (s) => TuringWasmProgram.fromBytes(fs.readFileSync(`${dir}/${s}.wasm`), JSON.parse(fs.readFileSync(`${dir}/${s}.abi.json`, 'utf8')));
const pred = { walkSpeed: await load('walkspeed'), walkDrive: await load('walkdrive'), walkRoute: await load('walkroute'),
               turn: await load('turn'), advance: await load('advance'), travel: await load('travel'), ship: null };
// the page's own walker code, cut out of app.js between its two markers and run with the page's globals stubbed
const text = fs.readFileSync(appPath, 'utf8');
const a = text.indexOf('// ---------------------------------------------------------------- the walker (World Walk)');
const b = text.indexOf('// The click path as the page draws it');
if (a < 0 || b < 0) throw new Error('the walker section of app.js was not found');
const job = JSON.parse(fs.readFileSync(casePath, 'utf8'));
const state = { game: { avatar: 'walker', walk: job.walk, walking: job.walking } };
const context = { pred, state, PRED_STEP: 1 / 240, Math, Float64Array, atob, Number, Infinity };
vm.createContext(context);
vm.runInContext(text.slice(a, b) + '\\n;globalThis.__walk = walk; globalThis.__sync = syncWalk; globalThis.__step = stepWalker;', context);
context.__sync(state.game);
const ship = job.ship;
for (const interval of job.intervals) context.__step(ship, interval.controls, interval.dur, job.motion);
console.log(JSON.stringify(ship));
"""


def test_the_page_walks_the_servers_walk(tmp_path):
    """The page's walker prediction (the stepWalker in app.js, run in node on the compiled cores) against the server's
    law over the same script: a click path across the roads, then free walking, a turn and a release."""
    _core("walkspeed", "walkdrive", "walkroute", "turn", "advance", "travel")
    from meowdoku import worldwalk as ww
    w = ww.world()
    game = ww.WorldWalk(seed=0)
    s = game.session
    s.control({})
    s.advance(1.0 / 60.0)
    towns = [w.site_nodes[i] for i in w.towns]
    path = min((p for a in towns for b in towns if a < b for p in [game.ground.path(a, b)] if p), key=len)
    p, g = w.nodes[path[0]], w.nodes[path[1]]
    h = g - (g @ p) * p
    h /= np.linalg.norm(h)
    for name, v in zip(("ux", "uy", "uz"), p):
        getattr(s.state, name)[0] = v
    for name, v in zip(("hx", "hy", "hz"), h):
        getattr(s.state, name)[0] = v
    s.state.node[0] = ww.MAP0 + path[0]
    game.goto(node=path[-1])
    st = game.state()
    names = ("ux", "uy", "uz", "hx", "hy", "hz", "spd", "yaw", "map_u", "map_v", "map_gx", "map_gy", "wp", "route")
    ship = {n: float(s.column(n)[0]) for n in names}
    script = [(5.0, {}), (1.0, {"move_axis_y": -1.0}), (0.7, {"move_axis": 1.0, "move_axis_y": -0.6}), (1.0, {})]
    intervals = []
    for seconds, controls in script:
        s.control({"move_axis": 0.0, "move_axis_y": 0.0, **controls})
        for _ in range(int(round(seconds * 30))):
            s.advance(1.0 / 30.0)
        intervals.append({"dur": seconds, "controls": {"move_axis": 0.0, "move_axis_y": 0.0, **controls}})
    case = {"walk": st["walk"], "walking": st["walking"] | {"active": True, "id": st["walk"]["route"]["id"]},
            "motion": st["predict"], "ship": ship, "intervals": intervals}
    (tmp_path / "walker.json").write_text(json.dumps(case), encoding="utf-8")
    page = _node(WALKER_JS, tmp_path, CORES, SHELL, ROOT / "examples/meowdoku/web/app.js", tmp_path / "walker.json")
    server = {n: float(s.column(n)[0]) for n in names}
    place = ww.RADIUS * float(np.arccos(np.clip(np.array([page[n] for n in ("ux", "uy", "uz")]) @ np.array([server[n] for n in ("ux", "uy", "uz")]), -1, 1)))
    # (two integrators -- the page's fixed 1/240 step, the dt system's own -- of one law: they agree to a fraction of a unit)
    print(f"[walker parity] page vs server after 7.7 s: place {place:.4f} units, spd {abs(page['spd'] - server['spd']):.4f}, yaw {abs(page['yaw'] - server['yaw']):.4f}")
    assert place < 0.15, (place, page, server)
    assert abs(page["spd"] - server["spd"]) < 0.05 and abs(page["yaw"] - server["yaw"]) < 0.05
    assert page["route"] == server["route"] == 0.0                     # the stick cancelled the path on both

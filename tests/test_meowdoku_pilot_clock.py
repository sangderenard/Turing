"""The page's law time is locked to the server's (bullet hell's pilot).

The pilot's ship is a function of LAW time, so the picture is smooth exactly when the law time the page shows is:
``web/pilot_clock.js`` reads the timing exchange every tick carries (the page's send time echoed with the server's
receipt and send instants, ``realtime_socket`` PROTOCOL 13) the way NTP does, and ``app.js`` shows the pilot's ship at
the law time of the page's now.  Before, the law time of each snapshot was taken from its ARRIVAL (``clock`` plus the
half round trip), so every reply's queueing and the server's compute time (which grows with the window it landed) moved
the picture's law time and the ship was pulled back each time (a correction of tens of milliseconds' flight, up to a
fifth of a unit a snapshot).

Headless: the page's own modules (``pilot_timeline.js`` stepping the real compiled cores, ``pilot_clock.js``) against a
simulated server on jittery links -- the server lands the wall time since its previous tick began, as ``EntityGame.tick``
does, in law steps cut where a decision takes effect, and answers with the snapshot's exchange.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
os.environ.setdefault("MEOWDOKU_LAW", "numpy")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node runs the page's cores")

import src.common.tensors.numpy_backend  # noqa: E402,F401  (the span laws run on the NumPy backend)
from meowdoku import entities as ent  # noqa: E402
from meowdoku import entity_motion as em  # noqa: E402
from meowdoku import realtime_socket as rs  # noqa: E402
from meowdoku import scroller as sc  # noqa: E402

WEB = ROOT / "examples/meowdoku/web"
SHELL = ROOT / "examples/mobile_reference_app/web/turing-wasm-shell.js"
CORES = ROOT / "build" / "meowdoku_page_cores"
STEMS = {stem: (fn, names, outs) for fn, names, outs, stem in em.PREDICTORS}

SIM_JS = r"""
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const [dir, shell, webDir, casePath] = process.argv.slice(2);
const { TuringWasmProgram } = await import(pathToFileURL(shell).href);
const { shipStepper, PilotTimeline, pilotControls } = await import(pathToFileURL(`${webDir}/pilot_timeline.js`).href);
const { LawClock } = await import(pathToFileURL(`${webDir}/pilot_clock.js`).href);
const load = async (stem) => TuringWasmProgram.fromBytes(fs.readFileSync(`${dir}/${stem}.wasm`), JSON.parse(fs.readFileSync(`${dir}/${stem}.abi.json`, 'utf8')));
const pred = { program: await load('motion'), bankTurn: await load('bank'), turn: await load('turn'), advance: await load('advance'),
               air: await load('air'), travel: await load('travel') };
const stepShip = shipStepper(pred, () => ({ x: 0 }));
const { motion, you0 } = JSON.parse(fs.readFileSync(casePath, 'utf8'));
const R = motion.sphere.radius;
const STEP_CAP = 1 / 30, INTERVAL = motion.pilot.interval, PREDELAY = motion.pilot.predelay;

const rng = (seed) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const dist = (a, b) => R * Math.hypot(a.ux - b.ux, a.uy - b.uy, a.uz - b.uz);
const q = (xs, p) => { const s = xs.filter(Number.isFinite).sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : NaN; };
const rest = pilotControls([0, 0, 0, 0, 0]);
const THETA = 987.654321;        // the server's wall minus the page's

// ONE RUN: the server answers the page's ticks over jittery links; the page shows the pilot's ship by `scheme`.
//   'old': the law time of a snapshot is its arrival, plus half the smoothed round trip (the page before the lock)
//   'lock': the law time of now is the locked clock's
function run(scheme, { seed, seconds, cadence, drop }) {
  const rand = rng(seed), ctl = rng(seed + 1000);
  const exp = (mean) => -Math.log(1 - rand()) * mean;
  // ---- the server
  const server = { ship: { ...you0 }, lawT: 0, last: null, next: 0, seq: 0, items: [], lost: 0, sentSeq: 0 };
  const decide = () => ({ seq: ++server.seq, start: server.lawT + PREDELAY, length: INTERVAL,
                          controls: [ctl() * 2 - 1, ctl() * 2 - 1, ctl() < 0.5 ? 1 : 0, ctl() * 2 - 1, 0] });
  function landOnServer(span) {
    let left = span;
    while (left > 1e-12) {
      if (server.lawT >= server.next - 1e-9) { server.items.push(decide()); server.next += INTERVAL; }
      const nxt = server.items.find((i) => i.start > server.lawT + 1e-12);
      const step = Math.min(STEP_CAP, left, nxt ? nxt.start - server.lawT : Infinity);
      const now = server.items.filter((i) => i.start <= server.lawT + 1e-12).pop();
      stepShip(server.ship, now ? pilotControls(now.controls) : rest, step, motion, 1);
      server.lawT += step; left -= step;
    }
  }
  // ---- the page
  const line = new PilotTimeline(), clock = new LawClock();
  let base = null, baseT = 0, at0 = 0, arrivedAt = 0, window_ = STEP_CAP, rtt = null, seen = -Infinity;
  const showAt = (t) => {            // the pilot's ship the page draws at page time t
    const lawNow = scheme === 'lock' ? clock.lawAt(t) : at0 + (t - arrivedAt);
    const end = Math.min(Math.max(lawNow, baseT), baseT + (scheme === 'lock' ? PREDELAY : 0.5));    // (the pilot's controls are known predelay past a snapshot)
    const ship = { ...base };
    line.integrate(stepShip, ship, baseT, end, motion, rest, window_);
    return { ship, lawNow: end };
  };
  const corrections = [], clockErr = [], frameStep = [], clockMoves = [];
  const events = [];
  let sendAt = 0;
  for (let j = 0; sendAt < seconds; j++) {
    sendAt = j * cadence + rand() * 0.004;                                              // the page's send time (its clock)
    const up = 0.002 + exp(0.004) + (rand() < 0.04 ? 0.05 : 0);
    const recv = Math.max(sendAt + THETA + up, (server.last ?? -Infinity) + 1e-4);      // the server's receipt (its clock), in order
    const raw = server.last == null ? 0 : recv - server.last;
    let span = Math.min(0.5, raw);
    if (drop && sendAt >= drop.at && !drop.done) { drop.done = true; span -= drop.lost; server.lost += drop.lost; }   // (a tick the cap cut short)
    server.lost += raw - Math.min(0.5, raw);
    if (server.epoch0 == null) server.epoch0 = recv;
    server.last = recv;
    landOnServer(span);
    const compute = 0.01 + 0.2 * span + exp(0.004);                                     // compute grows with the window landed
    const sent = recv + compute;
    const arrive = Math.max(sent - THETA + 0.002 + exp(0.004) + (rand() < 0.04 ? 0.05 : 0), (server.arrived ?? -Infinity) + 1e-4);   // (a link keeps its order)
    server.arrived = arrive;
    events.push({ arrive, ct: sendAt, recv, sent, lawWall: recv, clock: server.lawT, ship: { ...server.ship },
                  items: server.items.filter((i) => i.seq > server.sentSeq).map((i) => ({ ...i })) });
    server.sentSeq = server.seq;
  }
  events.sort((a, b) => a.arrive - b.arrive);
  // the true law time at page time t: the server's wall (t + THETA) less where law time zero stood on it
  const lawTrue = (t) => t + THETA - server.epoch0 - server.lostAt(t);
  server.lostAt = (t) => (drop && t >= drop.at ? drop.lost : 0);
  let frameAt = 0, prev = null, lastShown = null;
  const FRAME = 1 / 120;
  const frames = (until) => {
    for (; frameAt < until; frameAt += FRAME) {
      if (!base) continue;
      const { ship } = showAt(frameAt);
      if (lastShown && frameAt > 2.0) frameStep.push(dist(ship, lastShown) / FRAME);
      if (clock.ready() && scheme === 'lock' && frameAt > 6) clockErr.push(clock.lawAt(frameAt) - lawTrue(frameAt));
      lastShown = ship;
    }
  };
  for (const e of events) {
    frames(e.arrive);
    // the correction a reply makes: the ship shown just before it against the ship it shows
    const before = base ? showAt(e.arrive).ship : null;
    if (scheme === 'lock') { const was = clock.offset; clock.sample({ ct: e.ct, ca: e.arrive, recv: e.recv, sent: e.sent, lawWall: e.lawWall, clock: e.clock });
      if (was != null && clock.snaps === 1) clockMoves.push(Math.abs(clock.offset - was) / Math.max(1e-9, e.arrive - (prev ?? e.arrive))); }
    const measured = e.arrive - e.ct; rtt = rtt == null ? measured : rtt * 0.85 + measured * 0.15;
    line.accept(e.items, e.clock);
    if (e.clock > seen && seen > -Infinity) window_ = Math.min(STEP_CAP, Math.max(1e-3, e.clock - seen));
    seen = e.clock;
    base = { ...e.ship }; baseT = e.clock; arrivedAt = e.arrive; at0 = e.clock + Math.min(0.3, 0.5 * rtt);
    prev = e.arrive;
    if (before && e.arrive > 2.0) corrections.push(dist(before, showAt(e.arrive).ship));
  }
  frames(seconds);
  return { n: events.length, corr: { p50: q(corrections, 0.5), p95: q(corrections, 0.95), max: q(corrections, 1) },
           clockErr: { p95: q(clockErr.map(Math.abs), 0.95), max: q(clockErr.map(Math.abs), 1) },
           clockMoves: { max: q(clockMoves, 1) }, snaps: clock.snaps,
           frameSpeed: { p05: q(frameStep, 0.05), p95: q(frameStep, 0.95) } };
}

const out = {};
for (const seed of [1, 2, 3]) {
  out[`old${seed}`] = run('old', { seed, seconds: 40, cadence: 0.25 });
  out[`lock${seed}`] = run('lock', { seed, seconds: 40, cadence: 0.1 });
}
// a fast cadence (a hostile close: the page checks in at 30 ms)
out.oldFast = run('old', { seed: 4, seconds: 20, cadence: 0.03 });
out.lockFast = run('lock', { seed: 4, seconds: 20, cadence: 0.03 });
// the server's law falls 40 ms behind its wall (a tick over the cap): the lock follows at its slew, never jumps
out.lockDrop = run('lock', { seed: 5, seconds: 60, cadence: 0.1, drop: { at: 20, lost: 0.04 } });
// the hand-over takes the newest reading whole (the picture is being re-based anyway)
{
  const c = new LawClock();
  const m = (epoch) => ({ ct: 10, ca: 10.02, recv: 1010.005, sent: 1010.015, lawWall: 1010.005 + epoch, clock: 3 });
  c.sample(m(0)); c.sample({ ...m(0), ct: 10.1, ca: 10.12, recv: 1010.105, sent: 1010.115, lawWall: 1010.105 + 0.2, clock: 3.1 });
  const slewed = c.offset; c.takeNewest();
  out.take = { slewed, taken: c.offset, want: c.want.offset, step: c.offset - slewed };
}
console.log(JSON.stringify(out));
"""


def _node(script, tmp_path, *args):
    path = tmp_path / "check.mjs"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(["node", str(path), *map(str, args)], capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


def _core(*stems):
    CORES.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        fn, names, outs = STEMS[stem]
        em._build_one(CORES, fn, names, outs, stem)


def test_the_pilot_ship_is_shown_at_a_locked_law_time_and_no_reply_moves_it(tmp_path):
    _core("motion", "bank", "turn", "advance", "air", "travel")
    game = sc.BulletScroller("p")
    game.control({})
    game.tick()
    state = game.state()
    you0 = dict(state["you"])
    you0.update(lock=0.0, mount_yaw=0.0, mount_vyaw=0.0, mount_pitch=0.0, mount_vpitch=0.0)
    case = {"motion": state["predict"], "you0": you0}
    (tmp_path / "case.json").write_text(json.dumps(case), encoding="utf-8")
    r = _node(SIM_JS, tmp_path, CORES, SHELL, WEB, tmp_path / "case.json")
    for seed in (1, 2, 3):
        old, lock = r[f"old{seed}"], r[f"lock{seed}"]
        # the lock: the law time shown stays within a few ms of the true law time, with one reading taken whole (the first)
        assert lock["clockErr"]["p95"] < 0.004 and lock["clockErr"]["max"] < 0.012, lock
        assert lock["snaps"] == 1, lock
        assert lock["clockMoves"]["max"] <= 0.004 * 1.001 + 1e-9, lock          # the offset moves no faster than its slew
        # a reply's correction of the picture: under a centimetre (the law's own step discretisation), where the arrival
        # clock pulled the ship back by tens of milliseconds of flight
        assert lock["corr"]["p50"] < 0.001 and lock["corr"]["p95"] < 0.01, lock
        assert old["corr"]["p95"] > 10.0 * max(lock["corr"]["p95"], 1e-4), (old, lock)
    assert r["lockFast"]["corr"]["p95"] < 0.01 and r["oldFast"]["corr"]["p95"] > 10.0 * max(r["lockFast"]["corr"]["p95"], 1e-4), (r["oldFast"], r["lockFast"])
    assert r["lockFast"]["clockErr"]["max"] < 0.012 and r["lockFast"]["snaps"] == 1, r["lockFast"]
    # the hand-over from a person takes the newest reading whole: the epoch it carries (0.2 s on) is followed at its bounded
    # rate between hand-overs and taken at the hand-over
    take = r["take"]
    assert abs(take["taken"] - take["want"]) < 1e-12 and abs(take["step"]) > 0.1 and abs(take["slewed"] - take["want"]) > 0.1, take
    # a drop of law time on the server (a tick over the cap) moves law time zero on the wall: the clock follows it as a
    # time warp of at most EPOCH_SLEW (a quarter), never taking it as a jump, and has caught up within a second
    drop = r["lockDrop"]
    assert drop["snaps"] == 1 and drop["clockMoves"]["max"] <= (0.25 + 0.004) * 1.001 + 1e-9 and drop["corr"]["p95"] < 0.01, drop
    assert drop["clockErr"]["max"] < 0.041 and drop["clockErr"]["p95"] < 0.004, drop      # (never further out than the time dropped)


def test_the_snapshot_carries_the_timing_exchange_and_the_pilot_flag(tmp_path):
    """PROTOCOL 13: the law clock is followed by the server wall instant it stands for, the page's echoed send time, the
    server's receipt and send instants; the frame flag says the pilot flies."""
    game = sc.BulletScroller("p")
    game.control({})
    game.tick()
    state = game.state()
    state["piloted"] = True
    encoder = rs.SnapshotEncoder()
    binary, meta = encoder.encode(game, state, echo=123.25, received=456.5)
    assert binary[0] == rs.PROTOCOL == 13 and binary[1] & 8
    assert meta is not None and "you" not in meta and "ack" not in meta          # the ship rides the binary frame alone
    js = r"""
import { pathToFileURL } from 'node:url';
const [bodies, hex] = process.argv.slice(2);
const { decodeSnapshot } = await import(pathToFileURL(bodies).href);
const b = Buffer.from(hex, 'hex');
const d = decodeSnapshot(new DataView(b.buffer, b.byteOffset, b.byteLength), []);
console.log(JSON.stringify({ clock: d.you.clock, lawWall: d.you.lawWall, echo: d.you.echo, recv: d.you.recv, sent: d.you.sent, piloted: d.you.piloted,
                             intervals: d.you.intervals.length, flags: d.flags }));
"""
    got = _node(js, tmp_path, WEB / "bodies.js", binary.hex())
    assert got["echo"] == 123.25 and got["recv"] == 456.5 and got["piloted"] is True
    assert got["clock"] == state["world_clock"] and got["lawWall"] == state["law_wall"] and got["sent"] > 0.0


def test_the_law_clock_follows_the_wall_through_a_hand_over_and_a_slow_tick():
    """Server side of the lock: a tick lands the wall time since the previous one BEGAN (so the law clock and
    ``law_wall`` advance together, whichever mode landed the previous tick), a gap over ``max_frame`` costs law time
    (law time zero moves later on the wall by what was dropped), and ``max_frame`` clears the longest relaxed check-in."""
    assert ent.EntityGame.max_frame >= 2.0 * ent.EntityGame.SYNC_MAX
    game = sc.BulletScroller("p")
    game.control({})
    game.tick()                                                   # round zero
    seen = []
    for gap in (0.05, 0.12, 0.25, 0.04):
        time.sleep(gap)
        game.control({})
        game.tick()
        st = game.state()
        seen.append(st["law_wall"] - st["world_clock"])
    # law_wall - clock (where law time zero stands on the server's wall) holds while nothing was dropped
    assert max(seen) - min(seen) < 1e-6, seen
    time.sleep(game.max_frame + 0.3)                              # a stall the cap cuts short
    previous = game.last_tick                                     # (the gap is counted from the previous tick's start)
    game.control({})
    game.tick()
    st = game.state()
    dropped = (st["law_wall"] - previous) - game.max_frame
    assert dropped >= 0.3
    # the dropped time moves law time zero later on the wall by exactly what was dropped
    assert (st["law_wall"] - st["world_clock"]) - seen[-1] == pytest.approx(dropped, abs=1e-6)

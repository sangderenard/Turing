// [LOADING] Making the loading animation from the page's own fountain snapshots.  The state memory (compute.tape,
// recording.js) is the source when it fits; larger runs sample its packed snapshots into a bounded visual-only tape.
// This drives the existing fountain on a 2-D chamber (the hourglass by default), TURNS THE CASE OVER (compute.setCaseTilt,
// the same input the "turn over" control writes -- logged at its step) each time the average kinetic energy
// has fallen low enough that the grains have settled, a few rounds; then FILTERS the retained snapshots
// (loading_anim.js `filterTape`: positions only, frames and grains decimated, quantized) and uploads the stream.  The server validates it,
// packs it with the octet codec and keeps it in the library (loading_anim.py); the first one becomes the one players get.
//
// Run by hand: open the page with ?record_loading=1 (the defaults below; skipped when the server already has an animation).
// [STUDIO] The loading studio (studio.html) runs the same path in an iframe with `&studio=1&rp=<json>` (loading_params.js): the
// parameters are the chamber, the sim size, the species mix, the grain size, the grains kept, the flip rule, fps / bits, the duration cap
// and the colour source; it records even when an animation exists (the save goes into the library) and reports progress to its parent.
// (Never automatic: it takes the page's fountain over for about a minute.)
import { filterTape, encodeStream, VisualTape } from "./loading_anim.js";
import { fromQuery, attempts, metadata, RAW_CAP } from "./loading_params.js";

const SETTLE_SECONDS = 0.6;       // simulated seconds the average energy must stay below the threshold
const WALL_CAP_MS = 10 * 60 * 1000;

export async function record({ state, newGame, act, compute }) {
  const { studio, params: P } = fromQuery(location.search);
  const log = (...a) => console.log("[loading-record]", ...a);
  const report = (msg) => { if (studio && window.parent !== window) { try { window.parent.postMessage({ type: "loading-record", ...msg }, location.origin); } catch { /* no parent */ } } };
  const say = (t, send = true) => { const el = document.getElementById("status"); if (el) el.textContent = `loading animation: ${t}`; if (send) report({ stage: "progress", text: t }); };
  const fail = (text, extra = {}) => { say(text); report({ stage: "error", text, ...extra }); return null; };
  const have = await (await fetch("/api/ready", { cache: "no-store" })).json();
  if (!studio && have.loading_anim && have.loading_anim.present) { log("the server already has a loading animation"); return null; }

  // the existing fountain game on the chosen chamber at the chosen size
  const aspect = 1;
  report({ stage: "start", params: P });
  await newGame({ game: "fountain", aspect });
  await act({ type: "fountain_size", size: P.size, aspect });
  await act({ type: "fountain_choice", name: "shape", value: P.chamber });
  if (P.grainScale !== 1) {                                       // grain size: every species' radius times the scale (the controls hold the defaults)
    for (const c of (state.game && state.game.controls_ui) || []) {
      if (/^r[0-3]$/.test(c.id)) await act({ type: "fountain_param", name: c.id, value: c.value * P.grainScale, aspect });
    }
  }
  await act({ type: "fountain_mode", mode: P.colour });
  await act({ type: "fountain_speed", speed: P.speed });          // 0: the run follows the newest frame, the pump runs as fast as the device allows
  await act({ type: "fountain_start", aspect, ...(P.mix === null ? {} : { mix: P.mix }) });
  for (let i = 0; i < 600 && !(compute.ready && compute.tape && compute.manifest && compute.rec && compute.rec.tag === P.size); i++) await new Promise((r) => setTimeout(r, 100));
  if (!compute.ready) return fail("the fountain's compute product did not come up");
  compute.setSpin(0);
  compute.setGravity(null);
  compute.setCaseTilt(0);
  const n = compute.rec.n, dt = compute.manifest.law.dt, settle = compute.manifest.settle || {};
  // A complete state memory retains every packed point at every compute snapshot.  If the requested duration cannot
  // fit that window, keep a bounded visual projection of those same snapshots and leave physics replay switched off.
  const visualOnly = Math.ceil(P.cap / (dt * compute.snap)) * compute.product.frameBytes > compute.capBytes;
  compute.setRecording(!visualOnly);
  let visual = null, visualError = null, visualStart = null;
  if (visualOnly) {
    const stride = Math.max(1, Math.round(1 / (P.fps * dt * compute.snap)));
    visual = new VisualTape({ sourceSnap: compute.snap, stride, sourceCount: n, maxGrains: Math.min(4096, P.grains * 2) });
    compute.onSnapshot = (k, data) => {
      if (visualStart === null || (k - visualStart) % stride !== 0 || visualError) return;
      try { visual.push(k, data); } catch (error) { visualError = error; compute.onSnapshot = null; }
    };
    report({ stage: "progress", text: `visual-only capture: ${n.toLocaleString()} simulated ids; no physics replay kept` });
  }
  const threshold = (settle.ke_per_id || 1e-7) * P.settle;       // average kinetic energy per id below which the grains count as settled
  const outline = (state.game && state.game.compute && state.game.compute.chamber_outline) || [];
  log("recording", { n, dt, threshold, route: compute.route, params: P });

  const out = { from: null, to: null, turns: 0, tilt: 0 };
  let calmSince = null, turnEndsAt = -Infinity, lastEven = null, capped = false, longestFlip = 0;
  const wall0 = performance.now();
  try { await new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      try {
        if (visualError) throw visualError;
        const d = compute.diag, steps = compute.steps, t = steps * dt;
        if (!d || !compute.tape || compute.tape.hi < compute.tape.lo) return;
        const average = d.ke / Math.max(d.n_live, 1);
        const poured = d.n_live >= 0.97 * n;
        const finish = (to, why) => { out.to = to; capped = why; clearInterval(timer); resolve(); };
        if (performance.now() - wall0 > WALL_CAP_MS && out.from === null) throw new Error("the grains never settled within the time limit");
        if (out.from !== null && (t > P.cap || performance.now() - wall0 > WALL_CAP_MS)) { finish(lastEven !== null && lastEven > out.from ? lastEven : compute.tape.hi, true); return; }
        const calm = poured && average < threshold && t > turnEndsAt + 0.2;
        report({ stage: "tick", turns: out.turns, rounds: P.turns, energy: average, threshold, t, cap: P.cap, frame: compute.tape.hi, poured });
        if (!calm) { calmSince = null; say(`${out.turns}/${P.turns} turns, energy ${average.toExponential(1)}`, false); return; }       // (the tick report above carries it)
        if (calmSince === null) calmSince = t;
        if (t - calmSince < SETTLE_SECONDS) return;
        // settled: the first time starts the loop, each time after that is a round's end
        if (out.from === null) {
          out.from = compute.tape.hi;
          if (visual) { visualStart = out.from; visual.push(out.from, compute.tape.get(out.from));
            visual.inputs.push({ step: compute.tape.stepOf(out.from), values: { grav_x: 0, grav_y: -1 } }); }
        }
        if (out.turns % 2 === 0) lastEven = compute.tape.hi;
        if (out.turns >= P.turns) { finish(compute.tape.hi, false); return; }
        const motion = compute.startCaseFlip(P.flipSeconds, P.flipDirection);
        if (!motion) throw new Error("the case has no live grains or supported turn axis yet");
        out.tilt += P.flipDirection * Math.PI; out.turns++; turnEndsAt = t + motion.seconds; calmSince = null;
        longestFlip = Math.max(longestFlip, motion.seconds);
        report({ stage: "progress", text: `turning the case over in ${motion.seconds.toFixed(2)} simulated seconds (spring contact limit)` });
        log(`turn ${out.turns} at step ${steps}, frame ${compute.tape.hi}`);
      } catch (error) { clearInterval(timer); reject(error); }
    }, 100);
  }); } catch (error) {
    compute.onSnapshot = null;
    compute.setCaseTilt(0);
    return fail(`recording failed: ${error.message || error}`);
  }
  compute.onSnapshot = null;
  if (!visual && compute.tape.lo > out.from) { compute.setCaseTilt(0); return fail("the state memory's window slid past the start of the loop: lower the duration cap or the turns"); }
  if (capped) log("the duration cap ended the recording at frame", out.to);

  // Filter the chosen snapshot source; with `fit`, tighten it until the stream fits the budget.
  report({ stage: "progress", text: "filtering captured frames" });
  let anim = null, bytes = null, tried = [];
  const source = visual || compute.tape;
  try {
    for (const [fps, grains, bits] of attempts(P)) {             // the panel interpolates between frames
      anim = filterTape(source, { from: visual ? source.lo : out.from, to: visual ? source.hi : out.to, maxGrains: grains, fps, bits, dt, outline, angleAtStep: (step) => compute.caseAngleAt(step) });
      bytes = encodeStream(anim);
      tried.push({ fps, grains, bits, raw: bytes.length });
      log(`filtered ${anim.x.length} frames x ${anim.grains} grains at ${fps} fps, ${bits} bits: ${bytes.length} raw bytes`);
      if (bytes.length <= (P.fit ? 900 * 1024 : RAW_CAP)) break;
    }
  } catch (error) { compute.setCaseTilt(0); return fail(`filtering failed: ${error.message || error}`); }
  compute.setCaseTilt(0);
  if (bytes.length > RAW_CAP) return fail(`the raw stream is ${bytes.length} bytes, over the ${RAW_CAP}-byte cap: lower the grains, fps, bits or the duration`, { raw: bytes.length, tried });
  const meta = metadata(P, { kept_fps: anim.fps, kept_grains: anim.grains, frames: anim.x.length, capped, sim_n: n, visual_only: visualOnly, effective_flip_seconds: longestFlip });
  report({ stage: "progress", text: "uploading", raw: bytes.length });
  const r = await fetch("/api/loading_anim", { method: "POST", headers: { "X-Meowdoku-Loading": "1", "X-Meowdoku-Params": encodeURIComponent(JSON.stringify(meta)), "Content-Type": "application/octet-stream" }, body: bytes });
  const reply = await r.json().catch(() => ({}));
  log("upload", r.status, reply);
  say(r.ok ? `stored (${reply.stored && reply.stored.bytes} bytes through the octet codec)` : `not stored: ${reply.error || r.status}`);
  report({ stage: r.ok ? "done" : "error", status: r.status, text: r.ok ? "stored" : (reply.error || `HTTP ${r.status}`), stored: reply.stored, raw: bytes.length, tried, capped, sim_n: n, visual_only: visualOnly, effective_flip_seconds: longestFlip });
  return { status: r.status, reply, raw: bytes.length };
}

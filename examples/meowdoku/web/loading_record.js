// [LOADING] Making the loading animation.  NOT a capture path: the page's existing state memory (compute.tape, recording.js) is the
// source.  This drives the existing fountain on the small hourglass chamber, TURNS THE CASE OVER (compute.setCaseTilt, the same input
// the "turn over" control writes -- logged in the memory's input log with its step) each time the average kinetic energy has fallen low
// enough that the grains have settled, a few rounds, with the memory's recording on; then FILTERS the memory (loading_anim.js
// `filterTape`: positions only, frames and grains decimated, quantized) and uploads the stream.  The server validates it, packs it with
// the octet codec and keeps it (loading_anim.py); it accepts one, ever, so every later visit just plays it.
//
// Run by hand: open the page with ?record_loading=1.  (Never automatic: it takes the page's fountain over for about a minute.)
import { filterTape, encodeStream } from "./loading_anim.js";

const ROUNDS = 4;                 // case turns (an even number: the animation ends in the pose it began in, so it loops)
const SETTLE_SECONDS = 0.6;       // simulated seconds the average energy must stay below the threshold
const TURN_SECONDS = 0.4;         // the swift turn, in simulated seconds, as the view eases it
const BUDGET = 900 * 1024;        // the raw stream's size goal (the server caps uploads at 1 MiB)

export async function record({ state, newGame, act, compute }) {
  const log = (...a) => console.log("[loading-record]", ...a);
  const say = (t) => { const el = document.getElementById("status"); if (el) el.textContent = `loading animation: ${t}`; };
  const have = await (await fetch("/api/ready", { cache: "no-store" })).json();
  if (have.loading_anim && have.loading_anim.present) { log("the server already has a loading animation"); return null; }

  // the existing fountain game on the hourglass chamber at the small size
  const aspect = 1;
  await newGame({ game: "fountain", aspect });
  await act({ type: "fountain_size", size: "small", aspect });
  await act({ type: "fountain_choice", name: "shape", value: "sdf_hourglass" });
  await act({ type: "fountain_start", aspect });
  for (let i = 0; i < 600 && !(compute.ready && compute.tape && compute.manifest && compute.rec && compute.rec.tag === "small"); i++) await new Promise((r) => setTimeout(r, 100));
  if (!compute.ready) throw new Error("the fountain's compute product did not come up");
  compute.setRecording(true);                                    // the existing state memory, kept up to its cap
  const n = compute.rec.n, dt = compute.manifest.law.dt, settle = compute.manifest.settle || {};
  const threshold = (settle.ke_per_id || 1e-7) * 40;             // average kinetic energy per id below which the grains count as settled
  const outline = (state.game && state.game.compute && state.game.compute.chamber_outline) || [];
  log("recording", { n, dt, threshold, route: compute.route });

  const out = { from: null, to: null, turns: 0, tilt: 0 };
  let calmSince = null, turnedAt = -Infinity;
  await new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      try {
        const d = compute.diag, steps = compute.steps, t = steps * dt;
        if (!d || !compute.tape || compute.tape.hi < compute.tape.lo) return;
        const average = d.ke / Math.max(d.n_live, 1);
        const poured = d.n_live >= 0.97 * n;
        const calm = poured && average < threshold && t - turnedAt > TURN_SECONDS + 0.2;
        if (!calm) { calmSince = null; say(`${out.turns}/${ROUNDS} turns, energy ${average.toExponential(1)}`); return; }
        if (calmSince === null) calmSince = t;
        if (t - calmSince < SETTLE_SECONDS) return;
        // settled: the first time starts the loop, each time after that is a round's end
        if (out.from === null) out.from = compute.tape.hi;
        if (out.turns >= ROUNDS) { out.to = compute.tape.hi; clearInterval(timer); resolve(); return; }
        out.tilt += Math.PI; out.turns++; turnedAt = t; calmSince = null;
        compute.setCaseTilt(out.tilt);                           // the case is turned over: gravity in its frame turns with it
        log(`turn ${out.turns} at step ${steps}, frame ${compute.tape.hi}`);
      } catch (error) { clearInterval(timer); reject(error); }
    }, 100);
  });
  compute.setCaseTilt(0);

  // the FILTER of the state memory; tighten it until the stream fits the budget
  let anim = null, bytes = null;
  for (const [fps, grains] of [[24, 400], [20, 300], [15, 256], [12, 200]]) {
    anim = filterTape(compute.tape, { from: out.from, to: out.to, maxGrains: grains, fps, dt, outline, turnSeconds: TURN_SECONDS });
    bytes = encodeStream(anim);
    log(`filtered ${anim.x.length} frames x ${anim.grains} grains at ${fps} fps: ${bytes.length} raw bytes`);
    if (bytes.length <= BUDGET) break;
  }
  const r = await fetch("/api/loading_anim", { method: "POST", headers: { "X-Meowdoku-Loading": "1", "Content-Type": "application/octet-stream" }, body: bytes });
  const reply = await r.json().catch(() => ({}));
  log("upload", r.status, reply);
  say(r.ok ? `stored (${reply.stored && reply.stored.bytes} bytes through the octet codec)` : `not stored: ${reply.error || r.status}`);
  return { status: r.status, reply, raw: bytes.length };
}

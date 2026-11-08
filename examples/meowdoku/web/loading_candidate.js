// A Fountain control captures the same packed compute snapshots as Studio's
// loading recorder, then uses its existing visual filter and MLA1 codec.
import { VisualTape, filterTape, encodeStream } from "./loading_anim.js";

const FPS = 12, BITS = 10, SECONDS = 30;

export function beginCandidate({ compute, game, onStatus, onDone }) {
  if (!compute.ready || !compute.rec || !compute.manifest || !game?.compute)
    throw new Error("start the Fountain simulation before recording");
  if (compute.onSnapshot) throw new Error("another snapshot capture is in progress");
  const product = compute.product, dt = compute.manifest.law.dt;
  const stride = Math.max(1, Math.round(1 / (FPS * dt * compute.snap)));
  const tape = new VisualTape({ sourceSnap: compute.snap, stride, sourceCount: compute.rec.n });
  let first = null, finished = false;
  const sourceGame = game.game, sourceSize = game.compute.size;
  const detach = () => { if (compute.onSnapshot === snapshot) compute.onSnapshot = null; };
  const finish = async (cancel = false, reason = "capture discarded") => {
    if (finished) return;
    finished = true; detach();
    if (cancel) { onStatus(reason); onDone(); return; }
    try {
      if (tape.hi < 2) throw new Error("wait for at least two captured frames");
      onStatus("filtering compact visual frames…");
      const anim = filterTape(tape, { from: tape.lo, to: tape.hi,
        fps: FPS, bits: BITS, dt, view: game.compute.view,
        outline: game.compute.chamber_outline || [], seamSeconds: 0,
        angleAtStep: (step) => compute.caseAngleAt(step) });
      const bytes = encodeStream(anim);
      const params = { source: sourceGame, size: sourceSize, visual_only: true,
        projected_3d: sourceGame === "fountain3d", fps: FPS, grains: anim.grains,
        bits: BITS, frames: anim.x.length };
      onStatus(`uploading ${Math.round(bytes.length / 1024)} KB…`);
      const response = await fetch("/api/loading_candidate", { method: "POST", headers: {
        "X-Meowdoku-Loading": "1", "X-Meowdoku-Params": encodeURIComponent(JSON.stringify(params)),
        "Content-Type": "application/octet-stream" }, body: bytes });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
      onStatus(`submitted ${anim.x.length} frames · ${anim.grains} grains · expires ${body.stored.expires_at || "after graduation"}`);
    } catch (error) { onStatus(`capture failed: ${error.message || error}`); }
    onDone();
  };
  const snapshot = (k, data) => {
    if (finished) return;
    if (compute.product !== product) { void finish(true); return; }
    if (first === null) first = k;
    if ((k - first) % stride !== 0) return;
    try { tape.push(k, data); } catch (error) { void finish(true, `capture failed: ${error.message || error}`); return; }
    const seconds = (k - first) * compute.snap * dt;
    onStatus(`capturing ${seconds.toFixed(1)} / ${SECONDS} sim s · ${tape.hi} frames`);
    if (seconds >= SECONDS) void finish();
  };
  compute.onSnapshot = snapshot;
  onStatus(`capturing compact visual frames for up to ${SECONDS} simulated seconds…`);
  return { stop: () => finish(), cancel: () => finish(true) };
}

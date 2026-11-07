// Upload the exact saved Studio triangles to the Spectral thin-lens worker.
// The server verifies this mesh against the saved object's geometry receipt.
const encoder = new TextEncoder();
const hexDigest = /^[0-9a-f]{64}$/;

export function packMarblePhoto(geometry, objectDigest, options = {}) {
  if (!hexDigest.test(objectDigest)) throw new Error("save or load this design before exposing a photo");
  const faces = [...geometry.interior, ...geometry.glass];
  const config = {
    object_digest: objectDigest,
    exposure_seconds: options.seconds ?? 600,
    transport: options.transport ?? "fixed:3",
    scale_m: 0.025,
    object_center_m: [0, 0, 0.18],
    camera: {
      resolution: options.resolution ?? 256,
      eye_m: [0, 0, 0.07], target_m: [0, 0, 0.18],
      fov_degrees: 36, aperture_radius_m: 0.002, focus_distance_m: 0.11,
    },
    checker: { center_m: [0, 0, 0.30], width_m: 0.15, checks: 4 },
  };
  const json = encoder.encode(JSON.stringify(config));
  const n = faces.length;
  const buffer = new ArrayBuffer(16 + json.length + 76 * n);
  const view = new DataView(buffer);
  for (const [i, byte] of [...encoder.encode("MRP1")].entries()) view.setUint8(i, byte);
  view.setUint32(4, json.length, true);
  view.setUint32(8, n, true);
  view.setUint32(12, 0, true);
  new Uint8Array(buffer, 16, json.length).set(json);
  let at = 16 + json.length;
  for (const [a, b, c] of faces) for (const point of [a, b, c]) for (const x of point) {
    if (!Number.isFinite(x)) throw new Error("nonfinite photo triangle");
    view.setFloat32(at, x, true); at += 4;
  }
  for (const [, , , , normals] of faces) for (const normal of normals) for (const x of normal) {
    if (!Number.isFinite(x)) throw new Error("nonfinite photo normal");
    view.setFloat32(at, x, true); at += 4;
  }
  for (const [, , , id] of faces) {
    if (!Number.isInteger(id) || id < 0) throw new Error("invalid photo material ID");
    view.setUint32(at, id, true); at += 4;
  }
  return buffer;
}

export function createMarblePhoto(doc = document) {
  const $ = (id) => doc.getElementById(id);
  const state = { objectDigest: null, jobDigest: null, timer: null, generation: 0 };
  const status = (value) => { $("photo-status").textContent = value; };
  const stopPolling = () => { if (state.timer) clearTimeout(state.timer); state.timer = null; };
  const setButtons = () => {
    $("photo-start").disabled = !state.objectDigest;
    $("photo-cancel").disabled = !state.jobDigest;
  };
  const cancel = async () => {
    const digest = state.jobDigest;
    state.jobDigest = null; state.generation++; stopPolling(); setButtons();
    if (digest) {
      $("photo-hdr").hidden = true;
      $("photo-full").hidden = true;
      await fetch("/api/marble_photo/cancel", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest }) });
      status("Exposure canceled. The Phong preview remains interactive.");
    }
  };
  const stale = () => {
    if (!state.objectDigest) return;
    state.objectDigest = null;
    const digest = state.jobDigest;
    state.jobDigest = null; state.generation++; stopPolling(); setButtons();
    if (digest) fetch("/api/marble_photo/cancel", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest }) }).catch(() => {});
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    if (!$("photo-panel").hidden) $("photo-caption").textContent = "Photo of the previous saved design";
    status("Design changed. Save it again before exposing another photo.");
  };
  const saved = (digest) => {
    if (!hexDigest.test(digest)) throw new Error("invalid saved object digest");
    if (state.objectDigest !== digest && state.jobDigest) {
      const prior = state.jobDigest;
      state.jobDigest = null; state.generation++; stopPolling();
      fetch("/api/marble_photo/cancel", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest: prior }) }).catch(() => {});
    }
    state.objectDigest = digest; setButtons();
    status(`Saved design ${digest.slice(0, 12)} is ready for a thin-lens photograph.`);
  };
  const poll = async (digest, generation) => {
    if (state.jobDigest !== digest || generation !== state.generation) return;
    try {
      const response = await fetch(`/api/marble_photo?digest=${digest}`, { cache: "no-store" });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `photo status HTTP ${response.status}`);
      if (state.jobDigest !== digest || generation !== state.generation) return;
      if (value.image) {
        $("photo-panel").hidden = false;
        const imageUrl = `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(value.image)}`;
        $("photo-image").src = imageUrl;
        $("photo-full").href = imageUrl;
        $("photo-full").hidden = false;
        $("photo-caption").textContent = `${value.transport || "Fixed-line"} · ${Math.round(value.elapsed_seconds || 0)} s exposure`;
      }
      if (value.hdr) {
        const link = $("photo-hdr");
        link.href = `/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(value.hdr)}`;
        link.download = value.hdr;
        link.hidden = false;
      }
      status(value.state === "complete" ? `Photo complete · ${Math.round(value.elapsed_seconds || 0)} s exposure` :
        value.state === "error" ? `Photo failed: ${value.error || "worker error"}` :
          value.state === "preparing" ? "Preparing the saved mesh and thin-lens scene…" :
            `Exposing on GPU · ${Math.round(value.elapsed_seconds || 0)} s · ${(value.forward_ray_jobs || 0).toLocaleString()} ray jobs`);
      if (value.state === "complete" || value.state === "error" || value.state === "canceled") {
        state.jobDigest = null; setButtons(); return;
      }
    } catch (error) {
      status(`Photo status unavailable: ${error.message || error}`);
    }
    state.timer = setTimeout(() => poll(digest, generation), 2000);
  };
  const start = async (geometry) => {
    if (!state.objectDigest) throw new Error("save or load this design before exposing a photo");
    if (state.jobDigest) await cancel();
    const body = packMarblePhoto(geometry, state.objectDigest, {
      resolution: +$("photo-resolution").value, seconds: +$("photo-seconds").value,
      transport: $("photo-transport").value,
    });
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    status(`Uploading ${geometry.triangleCount.toLocaleString()} saved triangles…`);
    const response = await fetch("/api/marble_photo", { method: "POST",
      headers: { "Content-Type": "application/octet-stream" }, body });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `photo HTTP ${response.status}`);
    if (!hexDigest.test(value.digest)) throw new Error("photo server returned no digest");
    state.jobDigest = value.digest; const generation = ++state.generation;
    $("photo-panel").hidden = false;
    $("photo-caption").textContent = "Preparing thin-lens photograph…";
    setButtons(); poll(value.digest, generation);
  };
  $("photo-cancel").addEventListener("click", () => cancel().catch((error) => status(error.message || String(error))));
  setButtons();
  return { start, stale, saved, status };
}

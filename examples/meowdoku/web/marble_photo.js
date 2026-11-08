// Upload the exact saved Studio triangles to the Spectral thin-lens worker.
// The server verifies this mesh against the saved object's geometry receipt.
const encoder = new TextEncoder();
const hexDigest = /^[0-9a-f]{64}$/;
const OBJECT_CENTER = [0, 0, 0.18];

// Studio's orbit is T(-distance) * Rx(pitch) * Ry(yaw). Its inverse places
// the eye in the saved mesh's frame. The photo's focal distance is scaled by
// the same zoom ratio. The worker's checker is an XY plane, so place it on
// the far Z side rather than moving its center into the marble at side views.
export function photoCameraFromStudio(yaw, pitch, distance) {
  if (![yaw, pitch, distance].every(Number.isFinite) || distance <= 0) throw new RangeError("invalid Studio camera");
  const d = 0.11 * distance / 3.45;
  const direction = [-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  return {
    camera: { eye_m: OBJECT_CENTER.map((v, i) => v + d * direction[i]), target_m: OBJECT_CENTER,
      fov_degrees: 18, aperture_radius_m: 0.002, focus_distance_m: d },
    checker: { center_m: [0, 0, OBJECT_CENTER[2] - 0.12 * (direction[2] >= 0 ? 1 : -1)],
      width_m: 0.15, checks: 4, normal_sign: direction[2] >= 0 ? 1 : -1 },
  };
}

// Any Studio view-space position (not just the unit view direction above)
// maps into the saved mesh's frame the same way: invert Rx(pitch)*Ry(yaw),
// the same formula photoCameraFromStudio already uses for its direction
// vector (verified identical for the (0,0,1) case), then apply the same
// zoom-proportional scale and recenter on the marble.
function studioPointToWorld(viewPos, yaw, pitch, distance) {
  const scale = 0.11 / 3.45;
  const [x, y, z] = viewPos;
  const y1 = y * Math.cos(pitch) + z * Math.sin(pitch);
  const z1 = -y * Math.sin(pitch) + z * Math.cos(pitch);
  const xw = x * Math.cos(yaw) - z1 * Math.sin(yaw);
  const zw = x * Math.sin(yaw) + z1 * Math.cos(yaw);
  return OBJECT_CENTER.map((v, i) => v + scale * [xw, y1, zw][i]);
}

// The ray-traced photo's "Current Studio orbit and zoom" camera option only
// matched the camera pose before this -- the lighting was still the single
// default stage light, a different scene from what the Phong preview shows.
// This rebuilds the exact same five-role rig (background/2 kickers/key/fill)
// marble_studio.js's frame() renders, in world space, so selecting "studio"
// for the photo really is the same scene, not just the same camera angle.
export function photoLightsFromStudio(yaw, pitch, distance, rig) {
  const aimAt = (pos) => {
    const d = OBJECT_CENTER.map((v, i) => v - pos[i]);
    const n = Math.max(1e-6, Math.hypot(...d));
    return d.map((v) => v / n);
  };
  const toLight = (viewPos, power, label, shape, filterRgb) => {
    const pos = studioPointToWorld(viewPos, yaw, pitch, distance);
    return { pos, dir: aimAt(pos), power, label, shape, filter_rgb: filterRgb ?? null };
  };
  const key = [rig.keyX, rig.keyY, rig.keyZ];
  const background = [0, distance * 0.20, -distance * 1.65];
  const kickerL = [-distance * 0.95, distance * 0.30, -distance * 1.05];
  const kickerR = [ distance * 0.95, distance * 0.30, -distance * 1.05];
  const fill = [0, -distance * 0.95, distance * 0.30];
  return {
    lights: [
      toLight(key, rig.keyStrength, "key", rig.keyShape, rig.keyFilterRgb),
      toLight(background, rig.bgStrength, "background", "bare"),
      toLight(kickerL, rig.kickerStrength, "kicker-left", "bare"),
      toLight(kickerR, rig.kickerStrength, "kicker-right", "bare"),
      toLight(fill, rig.fillStrength, "fill", "scrim"),
    ],
  };
}

export function packMarblePhoto(geometry, objectDigest, options = {}) {
  if (!hexDigest.test(objectDigest)) throw new Error("save or load this design before exposing a photo");
  const faces = [...geometry.interior, ...geometry.glass];
  const config = {
    object_digest: objectDigest,
    exposure_seconds: options.seconds ?? 600,
    orbit_frames: options.orbitFrames ?? 1,
    transport: options.transport ?? "fixed:3",
    gamma: options.gamma ?? 1.0,
    scale_m: 0.025,
    object_center_m: OBJECT_CENTER,
    camera: {
      resolution: options.resolution ?? 256,
      eye_m: [0, 0, 0.07], target_m: [0, 0, 0.18],
      fov_degrees: 18, aperture_radius_m: 0.002, focus_distance_m: 0.11,
      ...options.camera,
    },
    checker: options.checker ?? { center_m: [0, 0, 0.30], width_m: 0.15, checks: 4 },
  };
  if (options.lights) config.lights = options.lights;
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

export function createMarblePhoto(doc = document, currentCamera = null) {
  const $ = (id) => doc.getElementById(id);
  const state = { objectDigest: null, jobDigest: null, lastDigest: null, timer: null, generation: 0,
    frames: [], selectedFrame: null, playTimer: null, frameImages: new Map(), displayToken: 0 };
  const status = (value) => { $("photo-status").textContent = value; };
  const stopPolling = () => { if (state.timer) clearTimeout(state.timer); state.timer = null; };
  const stopPlayback = () => {
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
    $("photo-play").textContent = "Play orbit";
  };
  const clearFrames = () => {
    stopPlayback(); state.frames = []; state.selectedFrame = null;
    state.frameImages.clear(); state.displayToken++; state.previewUrl = null;
    $("photo-frames").replaceChildren(); $("photo-frames").hidden = true;
    $("photo-play").hidden = true; $("photo-animation").hidden = true;
  };
  const drawImage = (image) => {
    const canvas = $("photo-image");
    if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    }
    canvas.getContext("2d").drawImage(image, 0, 0);
  };
  const frameImage = (url) => {
    let cached = state.frameImages.get(url);
    if (!cached) {
      const image = new Image();
      cached = { image, ready: false, failed: false };
      cached.promise = new Promise((resolve, reject) => {
        image.onload = async () => {
          try { await image.decode(); cached.ready = true; resolve(image); }
          catch (error) { cached.failed = true; reject(error); }
        };
        image.onerror = () => { cached.failed = true; reject(new Error("photo frame could not load")); };
      });
      image.src = url;
      state.frameImages.set(url, cached);
    }
    return cached;
  };
  const drawWhenReady = (url, caption) => {
    const token = ++state.displayToken;
    frameImage(url).promise.then((image) => {
      if (token !== state.displayToken) return;
      drawImage(image);
      $("photo-caption").textContent = caption;
    }).catch((error) => {
      if (token === state.displayToken) status(`Photo frame failed: ${error.message || error}`);
    });
  };
  const showFrame = (frame, digest) => {
    state.selectedFrame = frame.index;
    const imageUrl = `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(frame.image)}`;
    drawWhenReady(imageUrl, `Orbit frame ${frame.index + 1} of ${state.frames.length}`);
    $("photo-full").href = imageUrl; $("photo-full").hidden = false;
    const hdr = $("photo-hdr");
    hdr.hidden = !frame.hdr;
    if (frame.hdr) { hdr.href = `/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(frame.hdr)}`; hdr.download = frame.hdr; }
  };
  const renderFrames = (frames, digest) => {
    if (frames.length === state.frames.length) return;
    state.frames = frames;
    const strip = $("photo-frames"); strip.replaceChildren(); strip.hidden = frames.length < 2;
    $("photo-play").hidden = frames.length < 2;
    for (const frame of frames) {
      frameImage(`/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(frame.image)}`)
        .promise.catch(() => {});
      const button = doc.createElement("button"); button.type = "button";
      button.textContent = String(frame.index + 1); button.title = `View orbit frame ${frame.index + 1}`;
      button.addEventListener("click", () => { stopPlayback(); showFrame(frame, digest); });
      strip.append(button);
    }
  };
  const setButtons = () => {
    $("photo-start").disabled = !state.objectDigest;
    $("photo-cancel").disabled = !state.jobDigest;
  };
  const cancel = async () => {
    const digest = state.jobDigest;
    state.jobDigest = null; state.generation++; stopPolling(); stopPlayback(); setButtons();
    if (digest) {
      $("photo-hdr").hidden = true;
      $("photo-full").hidden = true;
      clearFrames();
      await fetch("/api/marble_photo/cancel", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest }) });
      status("Exposure canceled. The Phong preview remains interactive.");
    }
  };
  const stale = () => {
    if (!state.objectDigest) return;
    state.objectDigest = null;
    const digest = state.jobDigest;
    state.jobDigest = null; state.generation++; stopPolling(); stopPlayback(); setButtons();
    if (digest) fetch("/api/marble_photo/cancel", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest }) }).catch(() => {});
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    clearFrames();
    if (!$("photo-panel").hidden) $("photo-caption").textContent = "Photo of the previous saved design";
    status("Design changed. Save it again before exposing another photo.");
  };
  const saved = (digest) => {
    if (!hexDigest.test(digest)) throw new Error("invalid saved object digest");
    if (state.objectDigest !== digest && state.jobDigest) {
      const prior = state.jobDigest;
      state.jobDigest = null; state.generation++; stopPolling(); stopPlayback();
      fetch("/api/marble_photo/cancel", { method: "POST",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ digest: prior }) }).catch(() => {});
    }
    state.objectDigest = digest; clearFrames(); setButtons();
    status(`Saved design ${digest.slice(0, 12)} is ready for a thin-lens photograph.`);
  };
  const poll = async (digest, generation) => {
    if (state.jobDigest !== digest || generation !== state.generation) return;
    try {
      const response = await fetch(`/api/marble_photo?digest=${digest}`, { cache: "no-store" });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `photo status HTTP ${response.status}`);
      if (state.jobDigest !== digest || generation !== state.generation) return;
      renderFrames(value.frames || [], digest);
      if (value.image) {
        $("photo-panel").hidden = false;
        const imageUrl = `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(value.image)}`;
        if (state.selectedFrame === null) {
          if (state.previewUrl !== imageUrl) {
            if (state.previewUrl && !state.frames.some((frame) => state.previewUrl.endsWith(encodeURIComponent(frame.image)))) {
              state.frameImages.delete(state.previewUrl);
            }
            state.previewUrl = imageUrl;
            drawWhenReady(imageUrl, `${value.transport || "Fixed-line"} · frame ${(value.frame_index || 0) + 1}/${value.total_frames || 1} · ${Math.round(value.frame_elapsed_seconds || value.elapsed_seconds || 0)} s`);
          }
          $("photo-full").href = imageUrl;
          $("photo-full").hidden = false;
        }
      }
      if (value.hdr && state.selectedFrame === null) {
        const link = $("photo-hdr");
        link.href = `/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(value.hdr)}`;
        link.download = value.hdr;
        link.hidden = false;
      }
      if (value.animation) {
        const link = $("photo-animation");
        link.href = `/api/marble_photo/animation?digest=${digest}`;
        link.title = `Available until ${new Date(value.animation_expires_at * 1000).toLocaleString()}`;
        link.hidden = false;
      }
      status(value.state === "complete" ? `Photo complete · ${(value.frames || []).length || 1} frame(s) · ${value.frame_elapsed_seconds ?? $("photo-seconds").value} s final exposure` :
        value.state === "error" ? `Photo failed: ${value.error || "worker error"}` :
          value.state === "preparing" ? "Preparing the saved mesh and thin-lens scene…" :
            value.state === "encoding" ? `Encoding the orbit with Turing's JPEG AVI writer · ${value.encoding_frames || 0}/${value.total_frames || 1} frames${value.encoding_bytes ? ` · ${(value.encoding_bytes / 1048576).toFixed(1)} MiB` : ""}…` :
              `Exposing frame ${(value.frame_index || 0) + 1}/${value.total_frames || 1} on GPU · ${Math.round(value.frame_elapsed_seconds || 0)} s · ${(value.forward_ray_jobs || 0).toLocaleString()} ray jobs`);
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
      orbitFrames: +$("photo-orbit-frames").value,
      transport: $("photo-transport").value,
      gamma: +$("photo-gamma").value / 100,
      ...($("photo-camera").value === "studio" ? currentCamera?.() : {}),
    });
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    clearFrames();
    status(`Uploading ${geometry.triangleCount.toLocaleString()} saved triangles…`);
    const response = await fetch("/api/marble_photo", { method: "POST",
      headers: { "Content-Type": "application/octet-stream" }, body });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `photo HTTP ${response.status}`);
    if (!hexDigest.test(value.digest)) throw new Error("photo server returned no digest");
    state.jobDigest = state.lastDigest = value.digest; const generation = ++state.generation;
    $("photo-panel").hidden = false;
    $("photo-caption").textContent = "Preparing thin-lens photograph…";
    setButtons(); poll(value.digest, generation);
  };
  $("photo-cancel").addEventListener("click", () => cancel().catch((error) => status(error.message || String(error))));
  $("photo-gamma").addEventListener("input", () => {
    $("photo-gamma-out").textContent = (+$("photo-gamma").value / 100).toFixed(2);
  });
  $("photo-play").addEventListener("click", () => {
    if (state.playTimer) { stopPlayback(); return; }
    if (state.frames.length < 2) return;
    $("photo-play").textContent = "Pause orbit";
    let index = state.selectedFrame == null ? -1 : state.selectedFrame;
    state.playTimer = setInterval(() => {
      const next = (index + 1) % state.frames.length;
      const frame = state.frames[next];
      const url = `/api/marble_photo/image?digest=${state.jobDigest || state.lastDigest}&name=${encodeURIComponent(frame.image)}`;
      if (!frameImage(url).ready) return;
      index = next;
      showFrame(frame, state.jobDigest || state.lastDigest);
    }, 125);
  });
  setButtons();
  const resumeDigest = new URLSearchParams(doc.defaultView?.location.search || "").get("photo");
  if (hexDigest.test(resumeDigest || "")) {
    state.jobDigest = state.lastDigest = resumeDigest;
    const generation = ++state.generation;
    setButtons();
    poll(resumeDigest, generation);
  }
  return { start, stale, saved, status };
}

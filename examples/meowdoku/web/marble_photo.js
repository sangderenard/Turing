import {cameraWithBack} from "./marble_camera_back.js";
import {FOCAL_MAX_M, fovFromFocal} from "./marble_lens_controls.js";
// Upload the exact saved Studio triangles to the Spectral thin-lens worker.
// The server verifies this mesh against the saved object's geometry receipt.
import {createFilmResponseControls, normalizeFilmResponse} from "./marble_film_response.js";
import {geometryPolicy, photoGeometryPolicy, createGeometryPolicyControls, showGeometryDiagnostic} from "./marble_geometry_policy.js";
import {createRodBundleWarning} from "./marble_optical_diagnostic.js";
import {createStudioLensControls, FOCUS_MIN_M, FOCUS_MAX_M} from "./marble_lens_controls.js";
import {createPhotoParticipation, clientIntersectionEligible} from "./marble_photo_participation.js";
const encoder = new TextEncoder();
const hexDigest = /^[0-9a-f]{64}$/;
export const OBJECT_CENTER = [0, 0, 0.18];

function bounceCount(value) {
  const count = Number(value);
  if (!Number.isInteger(count) || count < 1 || count > 200) throw new RangeError("Light bounces must be a whole number from 1 to 200.");
  return count;
}

// The original field keeps actual counts, including restored legacy values.
// Only the separate range uses a quadratic position, so photo and live never
// accidentally exponentiate an actual saved count such as the default four.
export function photoBounceCount(doc = document) {
  return bounceCount(doc.getElementById("photo-bounces").value);
}

export function photoBounceMode(doc = document) {
  const mode = doc.getElementById("photo-bounce-mode")?.value ?? "manual";
  if (mode !== "auto" && mode !== "manual") throw new RangeError("Choose Auto or Manual light bounces.");
  return mode;
}

export function showBounceDiagnostic(doc, value) {
  const output = doc.getElementById("photo-bounce-diagnostic");
  if (!output || !value?.bounce_diagnostic) return;
  const probe = value.bounce_diagnostic;
  output.title = "";
  if (probe.mode === "manual") { output.textContent = `Manual: ${probe.chosen_depth} bounces.`; return; }
  if (probe.state === "probing") {
    output.textContent = `Auto: probing to 1000 bounces, batch ${probe.batch + 1}/${probe.batches}. Render cap: 200.`;
    return;
  }
  const coverage = probe.coverage_hits ? ` Hits per 64-ray batch: ${probe.coverage_hits.join(", ")}.` : "";
  const survivors = Number.isFinite(probe.paths_at_probe_limit) ?
    ` ${probe.paths_at_render_cap}/256 paths reach bounce 201; ${probe.paths_at_probe_limit}/256 still carry energy after 1000.` : "";
  const tail = Number.isFinite(probe.tail_fraction_95_batch_upper) ?
    ` Observed tail: ${(100 * probe.tail_fraction).toFixed(2)}%; 95% batch-spread upper: ${(100 * probe.tail_fraction_95_batch_upper).toFixed(2)}%.` : "";
  const beyondCap = Number.isInteger(probe.estimated_required_depth) && probe.estimated_required_depth > 200 ?
    ` Pilot threshold reached only near ${probe.estimated_required_depth} bounces, beyond the render cap.` : "";
  output.textContent = `Auto: rendering ${probe.chosen_depth} bounces (cap 200). Probe depth 1000.${survivors}${tail}${coverage}${beyondCap} ${probe.confidence || "Inconclusive pilot"}. Sparse estimate, not an image-error guarantee.${probe.warning ? ` ${probe.warning}` : ""}`;
  output.title = probe.limitations || "";
}

export function bounceSliderPosition(count) {
  return 100 * Math.sqrt((bounceCount(count) - 1) / 199);
}

export function bouncesFromSlider(value) {
  const position = Number(value);
  if (!Number.isFinite(position) || position < 0 || position > 100) throw new RangeError("Invalid bounce slider position.");
  return Math.round(1 + 199 * (position / 100) ** 2);
}

export function createPhotoBounceControls(doc = document) {
  const input = doc.getElementById("photo-bounces"), slider = doc.getElementById("photo-bounces-slider");
  const output = doc.getElementById("photo-bounces-out"), mode = doc.getElementById("photo-bounce-mode");
  const sync = () => {
    let count;
    try { count = photoBounceCount(doc); }
    catch (error) { input.setCustomValidity?.(error.message); return; }
    input.setCustomValidity?.("");
    const automatic = photoBounceMode(doc) === "auto";
    input.disabled = automatic;
    if (slider) slider.disabled = automatic;
    if (output) output.textContent = automatic ? "Auto, cap 200" : String(count);
    if (slider) {
      slider.value = String(bounceSliderPosition(count));
      slider.setAttribute?.("aria-valuetext", `${count} ${count === 1 ? "bounce" : "bounces"}`);
    }
  };
  mode?.addEventListener("change", () => {
    sync();
    const diagnostic = doc.getElementById("photo-bounce-diagnostic");
    if (diagnostic) diagnostic.textContent = photoBounceMode(doc) === "auto" ?
      "The next photo or live scene will probe sparse room paths to depth 1000; rendering stays capped at 200." :
      `Manual: ${photoBounceCount(doc)} bounces. The saved manual count is preserved when you use Auto.`;
  });
  input.addEventListener("input", sync);
  input.addEventListener("change", sync);
  slider?.addEventListener("input", () => { input.value = String(bouncesFromSlider(slider.value)); sync(); });
  slider?.addEventListener("keydown", event => {
    const steps = {ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10};
    if (!(event.key in steps) && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const count = bouncesFromSlider(slider.value);
    slider.value = String(bounceSliderPosition(event.key === "Home" ? 1 : event.key === "End" ? 200 :
      Math.max(1, Math.min(200, count + steps[event.key]))));
    slider.dispatchEvent(new doc.defaultView.Event("input", {bubbles: true}));
  });
  sync();
}

export function validatePhotoFocus(camera) {
  if (!Number.isFinite(camera?.focus_distance_m) || camera.focus_distance_m < FOCUS_MIN_M || camera.focus_distance_m > FOCUS_MAX_M)
    throw new RangeError("Analytical camera focus must be between 1 mm and 10 m. This is a numerical range, not a physical lens close-focus guarantee.");
  return camera;
}

// Studio's orbit is T(-distance) * Rx(pitch) * Ry(yaw). Its inverse places
// the eye in the saved mesh's frame. The photo's focal distance is scaled by
// the same zoom ratio. The worker's checker is an XY plane, so place it on
// the far Z side rather than moving its center into the marble at side views.
export function photoCameraFromStudio(yaw, pitch, distance, scale = 0.11 / 3.45, fovDegrees = 18, target = [0,0,0]) {
  if (![yaw, pitch, distance].every(Number.isFinite) || distance <= 0) throw new RangeError("invalid Studio camera");
  if (!Number.isFinite(scale) || scale <= 0) throw new RangeError("invalid Studio world scale");
  if (!Number.isFinite(fovDegrees) || fovDegrees < fovFromFocal(FOCAL_MAX_M) || fovDegrees > 100) throw new RangeError("invalid Studio field of view");
  if (!Array.isArray(target) || target.length !== 3 || !target.every(Number.isFinite)) throw new RangeError("invalid Studio camera target");
  const d = scale * distance;
  const direction = [-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const targetM = OBJECT_CENTER.map((v,i) => v + scale * target[i]);
  return {
    camera: { eye_m: targetM.map((v, i) => v + d * direction[i]), target_m: targetM,
      fov_degrees: fovDegrees, aperture_radius_m: 0.002, focus_distance_m: d },
    checker: { center_m: [0, 0, OBJECT_CENTER[2] - 0.12 * (direction[2] >= 0 ? 1 : -1)],
      width_m: 0.15, checks: 4, normal_sign: direction[2] >= 0 ? 1 : -1 },
  };
}

// Any Studio view-space position (not just the unit view direction above)
// maps into the saved mesh's frame the same way: invert Rx(pitch)*Ry(yaw),
// the same formula photoCameraFromStudio already uses for its direction
// vector (verified identical for the (0,0,1) case), then apply the same
// zoom-proportional scale and recenter on the marble.
export function studioPointToWorld(viewPos, yaw, pitch, distance) {
  const scale = 0.11 / 3.45;
  const [x, y, z] = viewPos;
  const y1 = y * Math.cos(pitch) + z * Math.sin(pitch);
  const z1 = -y * Math.sin(pitch) + z * Math.cos(pitch);
  const xw = x * Math.cos(yaw) - z1 * Math.sin(yaw);
  const zw = x * Math.sin(yaw) + z1 * Math.cos(yaw);
  return OBJECT_CENTER.map((v, i) => v + scale * [xw, y1, zw][i]);
}

export function packMarblePhoto(geometry, objectDigest, options = {}) {
  if (!options.objectSnapshot && !hexDigest.test(objectDigest)) throw new Error("save or load this design before exposing a photo");
  const policy = geometryPolicy(options);
  const faces = [...geometry.interior, ...geometry.glass];
  const config = {
    object_digest: objectDigest,
    exposure_seconds: options.seconds ?? 30,
    oversampling: options.oversampling ?? 1,
    orbit_frames: options.orbitFrames ?? 1,
    transport: options.transport ?? "fixed:3",
    gamma: options.gamma ?? 1.0,
    max_bounces: options.maxBounces ?? 4,
    bounce_mode: options.bounceMode ?? (options.maxBounces === undefined ? "auto" : "manual"),
    triangle_budget: policy.triangleBudget,
    sphere_coverage: policy.sphereCoverage,
    scale_m: 0.025,
    object_center_m: options.object_center_m ?? OBJECT_CENTER,
    camera: {
      resolution: options.resolution ?? 256,
      eye_m: [0, 0, 0.07], target_m: [0, 0, 0.18],
      fov_degrees: 18, aperture_radius_m: 0.002, focus_distance_m: 0.11,
      ...options.camera,
    },
    checker: options.checker ?? { center_m: [0, 0, 0.30], width_m: 0.15, checks: 4 },
  };
  if (options.width !== undefined || options.height !== undefined) {
    if (![options.width, options.height].every(v => Number.isInteger(v) && v >= 128 && v <= 32768))
      throw new RangeError("Custom photo width and height must be integers from 128 to 32768.");
    config.camera.width = options.width; config.camera.height = options.height;
  }
  if (!Number.isInteger(config.oversampling) || config.oversampling < 1 || config.oversampling > 4)
    throw new RangeError("Oversampling must be an integer multiplier from 1 to 4.");
  if (options.samplesPerOutputPixel !== undefined) {
    if (!Number.isInteger(options.samplesPerOutputPixel) || options.samplesPerOutputPixel < 1 || options.samplesPerOutputPixel > 65536)
      throw new RangeError("Samples per output pixel must be an integer from 1 to 65536.");
    config.samples_per_output_pixel = options.samplesPerOutputPixel;
  }
  const tiled = config.oversampling > 1 || options.samplesPerOutputPixel !== undefined || options.tiled !== undefined || config.camera.width !== undefined || config.camera.height !== undefined ||
    ![128,256,512,768,1024,2048,4096].includes(config.camera.resolution);
  if (tiled) {
    if (config.transport !== "fixed:3" || config.orbit_frames !== 1 || options.live || options.clientIntersections)
      throw new RangeError("Device intersections require ordinary RGB still photos; custom, large and oversampled photos use server tiles.");
    config.tiled = options.tiled ?? {tile_size: 1024};
  }
  if (options.cameraPoses !== undefined) {
    if (!Array.isArray(options.cameraPoses) || options.cameraPoses.length < 1 || options.cameraPoses.length > 32)
      throw new RangeError("Camera poses must contain 1..32 camera overrides.");
    if (config.orbit_frames !== 1 || tiled || options.live || options.clientIntersections)
      throw new RangeError("Camera poses cannot be combined with orbit, tiles, live view or device intersections.");
    // The server resolves each override through the existing camera/lens/back
    // authorities before admitting the job; keep input order unchanged.
    config.camera_poses = structuredClone(options.cameraPoses);
  }
  if (options.objectSnapshot) { delete config.object_digest; config.object_snapshot = options.objectSnapshot; }
  if (options.live) config.live = options.live;
  if(options.surfaceTransport!==undefined) {
    if(options.surfaceTransport!=='studio-pbr-mixture-v1')throw new RangeError('Unsupported surface transport.');
    if(config.transport!=='fixed:3')throw new RangeError('Corrected opaque PBR transport requires RGB rays.');
    config.surface_transport=options.surfaceTransport;
  }
  if(options.emitterSampling!==undefined) {
    const sampling=options.emitterSampling;
    if(!sampling||sampling.version!==1||Object.keys(sampling).some(k=>!['version','ray_budget','seed','refresh_each_sweep'].includes(k))||
        !Number.isSafeInteger(sampling.ray_budget)||sampling.ray_budget<128||sampling.ray_budget>65536||
        !Number.isSafeInteger(sampling.seed)||sampling.seed<0||sampling.seed>2147483647||typeof sampling.refresh_each_sweep!=='boolean')
      throw new RangeError('Invalid canonical emitter sampling settings.');
    if(sampling.refresh_each_sweep&&(!options.live||!options.receiveLightField||config.transport!=='fixed:3'))
      throw new RangeError('Fresh source populations require Live RGB field reception.');
    config.emitter_sampling=structuredClone(sampling);
  }
  if(options.lightFieldInternalMedia!==undefined) {
    if(typeof options.lightFieldInternalMedia!=='boolean')throw new TypeError('lightFieldInternalMedia must be a boolean.');
    if(options.lightFieldInternalMedia&&(!options.live||!options.receiveLightField||!options.emitterSampling?.refresh_each_sweep))
      throw new RangeError('Internal medium field recording requires fresh Live field populations.');
    if(options.lightFieldInternalMedia)config.light_field_internal_media=true;
  }
  if (options.receiveLightField !== undefined && typeof options.receiveLightField !== "boolean")
    throw new TypeError("receiveLightField must be a boolean.");
  if (options.receiveLightField) {
    if (config.transport !== "fixed:3" || config.orbit_frames !== 1 || tiled || options.cameraPoses)
      throw new RangeError("Forward-field reception requires one ordinary RGB still or Live camera.");
    config.receive_light_field = true;
    const stencil=options.lightFieldStencil??12;
    if(!Number.isInteger(stencil)||![6,12].includes(stencil))throw new RangeError('Light field stencil must be 6 or 12.');
    config.light_field_stencil=stencil;
    const grid=options.lightFieldGridSize??64;
    if(!Number.isInteger(grid)||![32,64].includes(grid))throw new RangeError('Light field grid must be 32 or 64.');
    config.light_field_grid_size=grid;
    config.light_field_estimator='track-length-root-sweep-v1';
    if(options.lightFieldRegion!==undefined) {
      if(options.lightFieldRegion!=='subject')throw new RangeError('Only bounded subject recording is supported.');
      const pitch=options.lightFieldCellSizeM??.0005,margin=options.lightFieldMarginM??.001;
      if(typeof pitch!=='number'||!Number.isFinite(pitch)||pitch<.00005||pitch>.01)throw new RangeError('Subject cell size must be between 0.05 and 10 mm.');
      if(typeof margin!=='number'||!Number.isFinite(margin)||margin<0||margin>.032)throw new RangeError('Subject margin must be between 0 and 32 mm.');
      config.light_field_region='subject';config.light_field_cell_size_m=pitch;config.light_field_margin_m=margin;
      delete config.light_field_grid_size; // Exact pitch and subject bounds determine the smallest supported cube.
    } else if(options.lightFieldCellSizeM!==undefined||options.lightFieldMarginM!==undefined)
      throw new RangeError('Physical cell size and margin require bounded subject recording.');
  }
  else if(options.lightFieldStencil!==undefined||options.lightFieldGridSize!==undefined||options.lightFieldRegion!==undefined||options.lightFieldCellSizeM!==undefined||options.lightFieldMarginM!==undefined)
    throw new RangeError('Light field options require field reception.');
  if(config.camera.camera_back){
    let aspect=(config.camera.width??config.camera.resolution)/(config.camera.height??config.camera.resolution);
    // Live's existing shader uses the viewport aspect; its integer pixel grid
    // is only a sampling lattice, displayed back into that same viewport.
    if(options.live)aspect=options.live.aspect??1;
    config.camera=cameraWithBack(config.camera,config.camera.camera_back,aspect);
  }
  if (options.clientIntersections) {
    if (!clientIntersectionEligible({transport:config.transport, orbitFrames:config.orbit_frames, resolution:config.camera.resolution, live:options.live})) throw new Error("Device intersections require RGB transport, one frame, and a listed resolution.");
    config.client_intersections = true;
  }
  validatePhotoFocus(config.camera);
  if (options.lights) config.lights = options.lights;
  if (config.transport === "fixed:3") config.film_response = normalizeFilmResponse(options.filmResponse);
  else if (options.filmResponse !== undefined) throw new Error("Custom RGB film response requires RGB transport.");
  // Real emissive fixture geometry (the box/tube housings + glowing panel
  // faces), not a synthetic light-list approximation of them -- the
  // worker appends this to the scene the same way it appends the saved
  // marble mesh, so Spectral's own emissive-material detection lights the
  // photo from the actual fixtures, not a stand-in.
  if (options.fixtures) config.fixtures = options.fixtures;
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

export function createMarblePhoto(doc = document, currentCamera = null, prepareExposure = null, fieldView = null) {
  const $ = (id) => doc.getElementById(id);
  const lensControls = createStudioLensControls(doc);
  const participation = createPhotoParticipation(doc);
  createPhotoBounceControls(doc);
  createGeometryPolicyControls(doc);
  const filmResponse = $("photo-film-response") ? createFilmResponseControls($("photo-film-response"), $("photo-transport")) : null;
  const state = { objectDigest: null, jobDigest: null, lastDigest: null, timer: null, generation: 0,
    frames: [], selectedFrame: null, playTimer: null, frameImages: new Map(), displayToken: 0,
    retouchDigest: null, displayGamma: 1, previewHdr: null, photoComplete: false, hasColorProfile: false,
    retouchRevision: 0, retouchDebounce: null, retouchPending: null, retouchInFlight: null,
    retouchError: null, statusRequest: 0, appliedStatusRequest: 0, nonSpecularTone: "exponential",
    displayedPhoto: null, displayWhite: null,
    retouchParams: { exposure: 0, gamma: 1, red: 1, green: 1, blue: 1, whitebalance: 0, tint: 0, intensity: 1,
      display_look: "reference", normalization: "auto99", normalization_white: 1, normalization_percentile: 99, tone_curve: "exponential",
      saturation: 1, vibrance: 0, black_point: 0,
      firefly_filter: false, firefly_threshold: 8, firefly_strength: 1 } };
  const snapshotToggle = $("photo-auto-snapshot");
  const autoSnapshot = () => typeof prepareExposure === "function" && snapshotToggle?.checked;
  let starting = false;
  const hdrObservers = new Set();
  const completedHdr = () => {
    const finalSelected = state.selectedFrame === null || state.frames.some(frame =>
      frame.index === state.selectedFrame && frame.hdr && frame.hdr === state.previewHdr);
    const shown = state.displayedPhoto;
    if (starting || state.jobDigest || state.playTimer || !state.photoComplete || !state.hasColorProfile ||
        !state.previewHdr || !finalSelected || !state.retouchDigest || state.retouchDigest !== state.lastDigest ||
        state.retouchDebounce || state.retouchPending || state.retouchInFlight || state.retouchError ||
        !shown || shown.digest !== state.lastDigest || shown.hdr !== state.previewHdr ||
        shown.selectedFrame !== state.selectedFrame || $("photo-panel").hidden) return null;
    return {id:`${state.lastDigest}:${state.previewHdr}`,
      url:`/api/marble_photo/hdr?digest=${state.lastDigest}&name=${encodeURIComponent(state.previewHdr)}`,
      revision:JSON.stringify([state.generation,state.displayToken,state.selectedFrame,state.retouchRevision]),
      complete:true,profiled:true,width:$("photo-image").width,height:$("photo-image").height,params:{...state.retouchParams},
      whiteReference:state.retouchParams.normalization === "manual" ? state.retouchParams.normalization_white : state.displayWhite};
  };
  const notifyHdr = () => {for (const observer of hdrObservers) observer(completedHdr());};
  const status = (value) => { $("photo-status").textContent = value; };
  const rodWarning=createRodBundleWarning(doc,'photo-optical-warning');
  const syncRetouchProfile = (value = null) => {
    const set = (id, text) => { const node = $(id); if (node) node.textContent = text; };
    const profile = value?.color_profile;
    const hasProfile = profile && typeof profile === "object" && !Array.isArray(profile);
    state.hasColorProfile = !!hasProfile;
    const number = (x) => typeof x === "number" && Number.isFinite(x) ? Number(x.toPrecision(6)).toString() : "not recorded";
    set("retouch-profile-name", !value ? "Sensor profile: waiting for exposure metadata." :
      hasProfile ? `Sensor profile: ${profile.id || "unnamed saved profile"}. Output: ${profile.output_space || "not recorded"}.` :
        "Legacy / unprofiled HDR: no sensor-to-display color profile was recorded.");
    set("retouch-profile-display", !value ? "" : hasProfile ?
      `Recorded display: ${profile.display_transform || "not recorded"}. Recorded normalization: ${profile.exposure_reference || "not recorded"}. White: ${number(value.display_white)}.` :
      `Legacy display path. White: ${number(value.display_white)}. No sensor conversion is invented for this exposure.`);
    const rendered = value?.retouch_display_transform || value?.display_transform;
    state.displayWhite = rendered?.effective_white ?? value?.display_white ?? null;
    set("retouch-profile-rendered", !value ? "" : !hasProfile || rendered?.legacy ?
      `Last rendered: legacy display path. ${rendered?.warning || "No profiled normalization or tone selection."}` :
      rendered && typeof rendered === "object" ?
        `Last rendered: ${rendered.normalization || "unrecorded"} normalization; selected white ${number(rendered.normalization_white)}; effective white ${number(rendered.effective_white)}. Effective percentile: ${rendered.effective_percentile === null ? "manual white" : number(rendered.effective_percentile)}. ${rendered.highlight_policy || ""} Tone: ${rendered.tone_curve || "unrecorded"} (${rendered.tone_formula || "formula not recorded"}). Saturation: ${number(rendered.saturation)}; vibrance: ${number(rendered.vibrance)}; black point: ${number(rendered.black_point)} of normalized white. ${rendered.black_point_formula || ""} ${rendered.chroma_formula || ""} ${rendered.creative_controls || ""} Isolated-highlight filter: ${rendered.firefly_filter === true ? "on" : rendered.firefly_filter === false ? "off" : "not recorded"}; threshold ${number(rendered.firefly_threshold)}×; strength ${number(rendered.firefly_strength)}. ${rendered.firefly_method || ""} ${rendered.firefly_warning || ""} ${rendered.gamut_mapping || ""}; ${rendered.transfer || "transfer not recorded"}.` :
        "Last rendered normalization and tone choices were not recorded separately.");
    set("retouch-profile-reference", hasProfile ? `Reference white: ${profile.reference_white || "not recorded"}.` : "Reference white: not recorded.");
    set("retouch-profile-condition", hasProfile ?
      `Response condition: ${number(profile.condition_number)}${profile.neutral_fallback ? "; neutral fallback active" : ""}.` : "Response condition: not recorded.");
    const matrix = profile?.sensor_to_linear_srgb;
    const validMatrix = Array.isArray(matrix) && matrix.length === 3 && matrix.every(row =>
      Array.isArray(row) && row.length === 3 && row.every(x => typeof x === "number" && Number.isFinite(x)));
    set("retouch-profile-matrix", validMatrix ? "Sensor to linear sRGB:\n" +
      matrix.map(row => row.map(number).join("  ")).join("\n") : "Sensor matrix: not recorded.");
    set("retouch-profile-limits", hasProfile ? profile.limits || "" : "");
    const warning = $("retouch-profile-warning");
    if (warning) {
      warning.textContent = String(profile?.warning || value?.color_profile_warning || "");
      warning.hidden = !warning.textContent;
    }
    syncRetouchDisplayControls();
  };
  // Live retouch is processed server-side, by the worker's own snapshot
  // loop, straight off the linear radiance it already holds in GPU memory
  // every ~2s snapshot -- not a client-side re-tonemap of a downloaded HDR
  // file. The browser only ever sends the small parameter set and displays
  // whatever PNG the worker wrote back (value.retouched_image), so this
  // works identically mid-exposure or after completion.
  const retouchFields = ["exposure", "gamma", "red", "green", "blue", "whitebalance", "tint", "intensity"];
  const colorFields = {saturation: [0, 2], vibrance: [-1, 1], black_point: [0, .5]};
  const filterFields = {firefly_threshold: [2, 64, 1], firefly_strength: [0, 1, 100]};
  const displayFields = ["display_look", "normalization", "normalization_white", "normalization_percentile", "tone_curve", ...Object.keys(colorFields),
    "firefly_filter", ...Object.keys(filterFields)];
  const displayControl = (field) => $(`retouch-${field.replaceAll("_", "-")}`);
  const syncRetouchDisplayControls = (values = false) => {
    for (const field of displayFields) {
      const node = displayControl(field);
      if (!node) continue;
      if (values) {
        if (field === "firefly_filter") node.checked = state.retouchParams[field];
        else node.value = String(field in colorFields ? Math.round(state.retouchParams[field] * 100) :
          field in filterFields ? state.retouchParams[field] * filterFields[field][2] : state.retouchParams[field]);
      }
      if (field in colorFields) {
        const output = $(`${node.id}-out`);
        if (output) output.textContent = state.retouchParams[field].toFixed(2);
      }
      if (field in filterFields) {
        const output = $(`${node.id}-out`);
        if (output) output.textContent = field === "firefly_threshold" ? `${state.retouchParams[field]}×` : state.retouchParams[field].toFixed(2);
      }
      if (field === "normalization_percentile") {
        const output = $(`${node.id}-out`);
        if (output) output.textContent = `${state.retouchParams[field]}%`;
      }
      node.disabled = !state.hasColorProfile ||
        (field === "normalization_white" && state.retouchParams.normalization !== "manual") ||
        (field === "normalization_percentile" && !["percentile", "specular"].includes(state.retouchParams.normalization)) ||
        (["tone_curve", "display_look"].includes(field) && state.retouchParams.normalization === "specular") ||
        (field in filterFields && !state.retouchParams.firefly_filter);
    }
    const summary = $("retouch-display-summary");
    if (summary) summary.textContent = !state.hasColorProfile ?
      "Normalization, tone, artistic colour and isolated-highlight controls require a recorded sensor profile. Legacy RGB corrections remain available." :
      `Display look: ${state.retouchParams.display_look === "color_priority" ? "Color priority (custom, preserves highlight color)" : "Reference (luminance priority)"}. Selected: ${state.retouchParams.normalization === "manual" ? `manual scene-linear white ${state.retouchParams.normalization_white}` :
        state.retouchParams.normalization === "auto99" ? "automatic 99th-percentile white" :
        `${state.retouchParams.normalization === "specular" ? "specular shot: " : ""}${state.retouchParams.normalization_percentile}th-percentile white`}; ${state.retouchParams.tone_curve === "linear" ? "linear luminance" :
        "exponential luminance"}${state.retouchParams.normalization === "specular" ? "; highlights above the selected white clip in the display" : ""}. Black point ${state.retouchParams.black_point.toFixed(2)} of normalized white before tone; saturation ${state.retouchParams.saturation.toFixed(2)}; vibrance ${state.retouchParams.vibrance.toFixed(2)}. Isolated-highlight filter ${state.retouchParams.firefly_filter ? `on: threshold ${state.retouchParams.firefly_threshold}×; strength ${state.retouchParams.firefly_strength.toFixed(2)}` : "off"}. Artistic linear-sRGB controls, not a calibrated or perceptually uniform colour adjustment. Then gamut compression and one sRGB transfer. Raw sensor values stay unchanged.`;
  };
  const readRetouchDisplayControls = () => {
    if (!state.hasColorProfile) return;
    const normalization = displayControl("normalization")?.value ?? state.retouchParams.normalization;
    let tone = displayControl("tone_curve")?.value ?? state.retouchParams.tone_curve;
    const look=normalization==="specular"?"reference":(displayControl("display_look")?.value||"reference");
    if(!["reference","color_priority"].includes(look))throw new Error("Choose Reference or Color priority display.");
    if (!["auto99", "manual", "percentile", "specular"].includes(normalization)) throw new Error("Choose automatic, percentile, specular shot, or manual normalization.");
    if (!["exponential", "linear"].includes(tone)) throw new Error("Choose exponential or linear tone mapping.");
    if (normalization === "specular") tone = "linear";
    const white = normalization === "manual" ? +(displayControl("normalization_white")?.value ?? state.retouchParams.normalization_white) :
      state.retouchParams.normalization_white;
    if (!Number.isFinite(white) || white < 1e-6 || white > 1e12) {
      throw new Error("Manual white must be a finite scene-linear value from 0.000001 to 1000000000000.");
    }
    const percentileInput = displayControl("normalization_percentile");
    const percentile = ["percentile", "specular"].includes(normalization) && percentileInput ?
      (percentileInput.value.trim() === "" ? NaN : Number(percentileInput.value)) : state.retouchParams.normalization_percentile;
    if (!Number.isFinite(percentile) || percentile < 1 || percentile > 100) {
      throw new Error("Normalization percentile must be a finite value from 1 to 100.");
    }
    const colors = {};
    for (const [field, [min, max]] of Object.entries(colorFields)) {
      const input = displayControl(field);
      const value = input ? (input.value.trim() === "" ? NaN : Number(input.value) / 100) : state.retouchParams[field];
      if (!Number.isFinite(value) || value < min || value > max) {
        throw new Error(`${field.replaceAll("_", " ")} must be a finite value from ${min} to ${max}.`);
      }
      colors[field] = value;
    }
    const filter = {firefly_filter: displayControl("firefly_filter")?.checked ?? state.retouchParams.firefly_filter};
    for (const [field, [min, max, scale]] of Object.entries(filterFields)) {
      const input = displayControl(field);
      const value = input ? (input.value.trim() === "" ? NaN : Number(input.value) / scale) : state.retouchParams[field];
      if (!Number.isFinite(value) || value < min || value > max) {
        throw new Error(`${field.replaceAll("_", " ")} must be a finite value from ${min} to ${max}.`);
      }
      filter[field] = value;
    }
    Object.assign(state.retouchParams, {display_look:look, normalization, normalization_white: white, normalization_percentile: percentile, tone_curve: tone, ...colors, ...filter});
    if (normalization !== "specular") state.nonSpecularTone = tone;
    if (normalization === "specular" && displayControl("tone_curve")) displayControl("tone_curve").value = tone;
  };
  const syncRetouchSliders = () => {
    for (const field of retouchFields) {
      const value = state.retouchParams[field];
      $(`retouch-${field}`).value = String(Math.round(value * 100));
      $(`retouch-${field}-out`).textContent = value.toFixed(2);
    }
    syncRetouchDisplayControls(true);
    const EventType = doc.defaultView?.Event;
    if (EventType) $("photo-retouch").dispatchEvent(new EventType("retouch-controls-change", { bubbles: true }));
  };
  const resetRetouchParams = (displayGamma = 1) => {
    state.nonSpecularTone = "exponential";
    state.retouchParams = { exposure: 0, gamma: displayGamma || 1, red: 1, green: 1, blue: 1,
      whitebalance: 0, tint: 0, intensity: 1, display_look: "reference", normalization: "auto99", normalization_white: 1, normalization_percentile: 99, tone_curve: "exponential",
      saturation: 1, vibrance: 0, black_point: 0,
      firefly_filter: false, firefly_threshold: 8, firefly_strength: 1 };
  };
  const invalidateRetouch = () => {
    state.retouchRevision++;
    if (state.retouchDebounce) clearTimeout(state.retouchDebounce);
    state.retouchDebounce = null;
    state.retouchPending = null;
    state.retouchError = null;
    // Also invalidate an image that finished downloading after the next edit.
    state.displayToken++;
    notifyHdr();
  };
  const currentRetouch = (request) => request.digest === state.retouchDigest &&
    request.generation === state.generation && request.revision === state.retouchRevision &&
    request.selectedFrame === state.selectedFrame;
  const postRetouch = async () => {
    if (state.retouchInFlight) return;
    // A finished HDR is converted synchronously by the server. Serializing
    // writes prevents a slower old edit from replacing the newest settings;
    // slider bursts retain only one pending request, not a growing POST queue.
    while (state.retouchPending) {
      const request = state.retouchPending;
      state.retouchPending = null;
      if (!currentRetouch(request)) continue;
      state.retouchInFlight = request;
      try {
        const response = await fetch("/api/marble_photo/retouch", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ digest: request.digest, ...request.params }),
        });
        const value = await response.json();
        if (!response.ok || value.ok !== true) throw new Error(value.error || `retouch HTTP ${response.status}`);
        if (currentRetouch(request)) await poll(request.digest, request.generation, request);
      } catch (error) {
        if (currentRetouch(request)) {
          state.retouchError = `Retouch failed: ${error.message || error}`;
          status(state.retouchError);
        }
      } finally {
        state.retouchInFlight = null;
        notifyHdr();
      }
    }
  };
  const queueRetouch = (immediate = false) => {
    invalidateRetouch();
    try { readRetouchDisplayControls(); }
    catch (error) {
      state.retouchError = `Retouch failed: ${error.message || error}`;
      status(state.retouchError);
      return;
    }
    syncRetouchDisplayControls();
    const digest = state.retouchDigest;
    if (!digest) {
      if (immediate) status(state.selectedFrame === null ? "No HDR photo available to retouch." :
        "Retouch is available for the final exposure only. Select the final orbit frame.");
      return;
    }
    const request = { digest, generation: state.generation, revision: state.retouchRevision,
      selectedFrame: state.selectedFrame, params: { ...state.retouchParams } };
    if (!state.hasColorProfile) for (const field of displayFields) delete request.params[field];
    const send = () => {
      state.retouchDebounce = null;
      if (!currentRetouch(request)) return;
      state.retouchPending = request;
      postRetouch();
    };
    if (immediate) send();
    else state.retouchDebounce = setTimeout(send, 150);
    notifyHdr();
  };
  const activateRetouch = (digest, displayGamma) => {
    if (displayGamma != null) state.displayGamma = displayGamma;
    if (state.retouchDigest === digest) return;
    state.retouchDigest = digest;
    resetRetouchParams(state.displayGamma);
    syncRetouchSliders();
    $("photo-retouch").hidden = false;
  };
  const deactivateRetouch = () => {
    invalidateRetouch();
    state.retouchDigest = null; $("photo-retouch").hidden = true;
  };
  const stopPolling = () => { if (state.timer) clearTimeout(state.timer); state.timer = null; };
  const stopPlayback = () => {
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = null;
    $("photo-play").textContent = "Play orbit";
    notifyHdr();
  };
  const clearFrames = () => {
    stopPlayback(); state.frames = []; state.selectedFrame = null;
    state.frameImages.clear(); state.displayToken++; state.previewUrl = null; state.previewHdr = null; state.photoComplete = false;
    state.displayedPhoto = null;
    $("photo-frames").replaceChildren(); $("photo-frames").hidden = true;
    $("photo-play").hidden = true; $("photo-animation").hidden = true;
    deactivateRetouch();
    syncRetouchProfile();
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
  const drawWhenReady = (url, caption, opticalStatus=null) => {
    const token = ++state.displayToken;
    const identity = {digest:state.lastDigest,hdr:state.previewHdr,selectedFrame:state.selectedFrame};
    frameImage(url).promise.then((image) => {
      if (token !== state.displayToken) return;
      drawImage(image);
      state.displayedPhoto = identity;
      if(opticalStatus)rodWarning.displayed(opticalStatus.value,opticalStatus.key);
      $("photo-caption").textContent = caption;
      notifyHdr();
    }).catch((error) => {
      if (token === state.displayToken) status(`Photo frame failed: ${error.message || error}`);
    });
  };
  const showFrame = (frame, digest) => {
    invalidateRetouch();
    state.selectedFrame = frame.index;
    const imageUrl = `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(frame.image)}`;
    const hdr = $("photo-hdr");
    hdr.hidden = !frame.hdr;
    if (frame.hdr) {
      hdr.href = `/api/marble_photo/hdr?digest=${digest}&name=${encodeURIComponent(frame.hdr)}`;
      hdr.download = frame.hdr;
      if (state.photoComplete && frame.hdr === state.previewHdr) activateRetouch(digest, state.displayGamma);
      else {
        deactivateRetouch();
        status("Retouch is available for the final exposure only. Select the final orbit frame.");
      }
    } else {
      deactivateRetouch();
    }
    drawWhenReady(imageUrl, `Orbit frame ${frame.index + 1} of ${state.frames.length}`);
    $("photo-full").href = imageUrl; $("photo-full").hidden = false;
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
    $("photo-start").disabled = starting || (!state.objectDigest && !autoSnapshot()) || !!state.jobDigest;
    $("photo-cancel").disabled = !state.jobDigest;
  };
  snapshotToggle?.addEventListener("change", setButtons);
  const cancel = async () => {
    participation.stop();
    const digest = state.jobDigest;
    fieldView?.stop(digest);
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
  const stale = (invalidateObject=true) => {
    fieldView?.invalidate();
    if (!state.objectDigest && !state.jobDigest && !state.retouchDigest) return;
    if(invalidateObject)state.objectDigest = null;
    // Editing a recipe must not cancel its already captured still exposure.
    if(state.jobDigest) {
      setButtons(); status(invalidateObject?"Design changed. The existing exposure continues; save these changes before the next photo.":"Scene changed. The existing exposure continues with its original scene.");
      return;
    }
    state.generation++; stopPolling(); stopPlayback(); setButtons();
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    clearFrames();
    if (!$("photo-panel").hidden) $("photo-caption").textContent = invalidateObject?"Photo of the previous saved design":"Photo of the previous scene";
    status(invalidateObject?"Design changed. Save it again before exposing another photo.":"Scene changed. The saved object is ready for a new photograph.");
  };
  const saved = (digest) => {
    if (!hexDigest.test(digest)) throw new Error("invalid saved object digest");
    if(state.jobDigest) {
      state.objectDigest=digest; setButtons();
      status("Design saved. The existing exposure continues with its original saved design.");
      return;
    }
    state.objectDigest = digest; clearFrames(); setButtons();
    status(`Saved design ${digest.slice(0, 12)} is ready for a thin-lens photograph.`);
  };
  const poll = async (digest, generation, retouchRequest = null) => {
    // Completion ends exposure polling, not the lifetime of its saved HDR.
    // A successful retouch gets one status read even after jobDigest is cleared.
    const current = () => generation === state.generation && digest === state.lastDigest &&
      (retouchRequest ? currentRetouch(retouchRequest) : state.jobDigest === digest);
    if (!current()) return;
    const requestNumber = ++state.statusRequest;
    const revision = state.retouchRevision;
    try {
      const response = await fetch(`/api/marble_photo?digest=${digest}`, { cache: "no-store" });
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `photo status HTTP ${response.status}`);
      if (!current() || requestNumber < state.appliedStatusRequest) return;
      state.appliedStatusRequest = requestNumber;
      void fieldView?.observe(value,digest);
      const opticalImageKey=`${digest}/${value.image??''}/${value.live_revision??''}`;
      rodWarning.observe(value,opticalImageKey);
      syncRetouchProfile(value);
      state.previewHdr = value.hdr || null;
      state.photoComplete = value.state === "complete";
      if (value.display_gamma != null) state.displayGamma = value.display_gamma;
      renderFrames(value.frames || [], digest);
      const selectedIsFinal = state.photoComplete && state.selectedFrame !== null && state.frames.some(frame =>
        frame.index === state.selectedFrame && frame.hdr && frame.hdr === value.hdr);
      const displayingExposure = state.selectedFrame === null || selectedIsFinal;
      if (value.hdr && displayingExposure) {
        activateRetouch(digest, value.display_gamma);
      } else if (state.retouchDigest) {
        deactivateRetouch();
      }
      const canDisplay = displayingExposure && revision === state.retouchRevision &&
        !state.retouchDebounce && !state.retouchPending && (!state.retouchInFlight || retouchRequest ||
          state.retouchInFlight.digest !== digest || state.retouchInFlight.generation !== generation);
      const usingRetouch = !!value.retouched_image && displayingExposure;
      const displayName = usingRetouch ? value.retouched_image : value.image;
      if (displayName) {
        $("photo-panel").hidden = false;
        // Retouches normally receive a new filename. Keep epoch in the URL
        // for live workers that still publish successive snapshots by name.
        const imageUrl = usingRetouch
          ? `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(displayName)}&epoch=${value.epoch}`
          : `/api/marble_photo/image?digest=${digest}&name=${encodeURIComponent(displayName)}`;
        if (canDisplay) {
          if (state.previewUrl !== imageUrl || state.displayedPhoto?.hdr !== state.previewHdr) {
            if (state.previewUrl === imageUrl) state.frameImages.delete(imageUrl);
            if (state.previewUrl && !state.frames.some((frame) => state.previewUrl.endsWith(encodeURIComponent(frame.image)))) {
              state.frameImages.delete(state.previewUrl);
            }
            state.previewUrl = imageUrl;
            drawWhenReady(imageUrl, selectedIsFinal ? `Orbit frame ${state.selectedFrame + 1} of ${state.frames.length}` :
              `${value.transport || "Fixed-line"} · frame ${(value.frame_index || 0) + 1}/${value.total_frames || 1} · ${Math.round(value.frame_elapsed_seconds || value.elapsed_seconds || 0)} s`,{value,key:opticalImageKey});
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
      showBounceDiagnostic(doc, value);
      showGeometryDiagnostic(doc, value);
      status(state.retouchError || (value.tiled ? (value.state === "error" ? `Photo failed: ${value.error || "worker error"}` :
        value.state === "complete" ? `Tiled photo complete · ${value.tiled.output_width} × ${value.tiled.output_height} · ${value.tiled.completed_tiles} tiles` :
        value.state === "assembling" ? `Assembling ${value.tiled.output_width} × ${value.tiled.output_height} photo from linear HDR tiles…` :
        `Tiled photo · ${value.tiled.completed_tiles}/${value.tiled.total_tiles} tiles complete · ${value.tiled.target_samples_per_output_pixel ? `${value.tiled.target_samples_per_output_pixel} samples per output pixel (requested ${value.tiled.requested_samples_per_output_pixel})` : `${value.tiled.exposure_seconds_per_tile} s per tile`}`) : value.state === "complete" ? `Photo complete · ${(value.frames || []).length || 1} frame(s) · ${value.frame_elapsed_seconds ?? $("photo-seconds").value} s final exposure` :
        value.state === "error" ? `Photo failed: ${value.error || "worker error"}` :
          value.state === "preparing" ? "Preparing the saved mesh and thin-lens scene…" :
            value.state === "encoding" ? `Encoding the orbit with Turing's JPEG AVI writer · ${value.encoding_frames || 0}/${value.total_frames || 1} frames${value.encoding_bytes ? ` · ${(value.encoding_bytes / 1048576).toFixed(1)} MiB` : ""}…` :
              `Exposing frame ${(value.frame_index || 0) + 1}/${value.total_frames || 1} on GPU · ${Math.round(value.frame_elapsed_seconds || 0)} s · ${(value.forward_ray_jobs || 0).toLocaleString()} ray jobs`));
      if (value.state === "complete" || value.state === "error" || value.state === "canceled") {
        state.jobDigest = null; stopPolling(); participation.stop(); setButtons(); notifyHdr(); return;
      }
    } catch (error) {
      if (!current()) return;
      if (retouchRequest) throw error;
      status(state.retouchError || `Photo status unavailable: ${error.message || error}`);
    } finally {
      // An older live read may lose to the one-shot retouch read. It still
      // owns the next polling tick while the exposure remains active.
      if (!retouchRequest && current()) state.timer = setTimeout(() => poll(digest, generation), 2000);
    }
  };
  const start = async (geometry) => {
    if (starting) throw new Error("An exposure is already preparing.");
    const snapshotEnabled = autoSnapshot();
    if (!state.objectDigest && !snapshotEnabled) throw new Error("Save this design or enable Snapshot before exposure.");
    starting = true; setButtons(); notifyHdr();
    try {
    // Validate before canceling anything: editing response controls alone never
    // alters an exposure already running, and invalid inputs cannot cancel it.
    const responseForExposure = filmResponse?.forTransport();
    const maxBounces = photoBounceCount(doc), bounceMode = photoBounceMode(doc);
    const policy = photoGeometryPolicy(doc);
    // Camera framing does not choose the scene: close-up exposures need the
    // same authored room and light fixtures, with their exposure-only gains.
    const scene = currentCamera?.(Boolean(snapshotEnabled));
    const subjectCenter = scene?.object_center_m ?? OBJECT_CENTER;
    const sceneOptions = $("photo-camera").value === "studio" ? scene :
      {fixtures: scene?.fixtures, lights: scene?.lights, object_center_m: subjectCenter,
        camera: {eye_m: subjectCenter.map((v,i)=>[0,0,.07][i]+(v-OBJECT_CENTER[i])), target_m: subjectCenter}};
    if (state.jobDigest) throw new Error("An exposure is already running. Wait for it to finish or use Cancel exposure.");
    const objectDigest = state.objectDigest;
    const fieldTicket = fieldView?.capture({saved:!snapshotEnabled});
    const options = {
      ...($('ray-surface-transport')?.checked?{surfaceTransport:'studio-pbr-mixture-v1'}:{}),
      receiveLightField:!!fieldTicket,
      ...(fieldTicket?{lightFieldStencil:fieldTicket.stencil,lightFieldGridSize:fieldTicket.gridSize}:{}),
      ...(fieldTicket?.region==='subject'?{lightFieldRegion:'subject',lightFieldCellSizeM:fieldTicket.cellSizeM,lightFieldMarginM:fieldTicket.marginM}:{}),
      resolution: $("photo-resolution").value === "custom" ? 256 : +$("photo-resolution").value,
      ...($("photo-resolution").value === "custom" ? {width:+$("photo-width").value, height:+$("photo-height").value} : {}),
      seconds: +$("photo-seconds").value,
      oversampling: +($("photo-oversampling")?.value || 1),
      ...($("photo-tile-size")?.value && $("photo-tile-size").value !== "auto" ? {tiled:{tile_size:+$("photo-tile-size").value}} : {}),
      orbitFrames: +$("photo-orbit-frames").value,
      transport: $("photo-transport").value,
      filmResponse: responseForExposure,
      clientIntersections: participation.requested(),
      gamma: +$("photo-gamma").value / 100,
      maxBounces,
      bounceMode,
      ...policy,
      ...sceneOptions,
    };
    // Freeze the selected lens at the click, before asynchronous snapshot hashing.
    options.camera = lensControls.applyCamera({fov_degrees:18, aperture_radius_m:.002,
      focus_distance_m:.11, ...options.camera});
    // Capture the detached object synchronously at this click. Preparation can
    // hash asynchronously; later edits do not change this exposure's source.
    if (snapshotEnabled) {
      status("Preparing an exposure-only snapshot (library and autosave unchanged)…");
      const prepared = await prepareExposure();
      geometry = prepared.geometry; options.objectSnapshot = prepared.value;
    }
    const body = packMarblePhoto(geometry, objectDigest, options);
    $("photo-hdr").hidden = true;
    $("photo-full").hidden = true;
    clearFrames();
    showGeometryDiagnostic(doc, null);
    status(`Uploading ${geometry.triangleCount.toLocaleString()} captured triangles…`);
    const response = await fetch("/api/marble_photo", { method: "POST",
      headers: { "Content-Type": "application/octet-stream" }, body });
    const value = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(value.error || `photo HTTP ${response.status}`);
    if (!hexDigest.test(value.digest)) throw new Error("photo server returned no digest");
    state.jobDigest = state.lastDigest = value.digest; const generation = ++state.generation;
    fieldView?.connect(value.digest,fieldTicket);
    $("photo-panel").hidden = false;
    $("photo-caption").textContent = "Preparing thin-lens photograph…";
    setButtons(); poll(value.digest, generation);
    try { participation.connect(value, options.clientIntersections); }
    catch (error) { status(`Exposure created, but device participation could not start: ${error.message}`); }
    } finally { starting = false; setButtons(); notifyHdr(); }
  };
  $("photo-cancel").addEventListener("click", () => cancel().catch((error) => status(error.message || String(error))));
  $("photo-gamma").addEventListener("input", () => {
    $("photo-gamma-out").textContent = (+$("photo-gamma").value / 100).toFixed(2);
  });
  for (const field of retouchFields) {
    $(`retouch-${field}`).addEventListener("input", () => {
      const value = +$(`retouch-${field}`).value / 100;
      state.retouchParams[field] = value;
      $(`retouch-${field}-out`).textContent = value.toFixed(2);
      queueRetouch();
    });
  }
  for (const field of displayFields) {
    displayControl(field)?.addEventListener(["normalization_white", "normalization_percentile"].includes(field) || field in colorFields || field in filterFields ? "input" : "change", () => {
      // Entering this display preset sets its white percentile; subsequent
      // slider edits stay explicit and never alter the saved linear HDR.
      if (field === "normalization" && displayControl(field).value === "specular" && state.retouchParams.normalization !== "specular") {
        if (displayControl("normalization_percentile")) displayControl("normalization_percentile").value = "95";
        if (displayControl("tone_curve")) displayControl("tone_curve").value = "linear";
      } else if (field === "normalization" && displayControl(field).value !== "specular" && state.retouchParams.normalization === "specular") {
        if (displayControl("tone_curve")) displayControl("tone_curve").value = state.nonSpecularTone;
      }
      queueRetouch();
    });
  }
  $("retouch-apply").addEventListener("click", () => {
    queueRetouch(true);
  });
  $("retouch-reset").addEventListener("click", () => {
    resetRetouchParams(state.displayGamma);
    syncRetouchSliders();
    queueRetouch(true);
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
    notifyHdr();
  });
  setButtons();
  const resumeDigest = new URLSearchParams(doc.defaultView?.location.search || "").get("photo");
  if (hexDigest.test(resumeDigest || "")) {
    state.jobDigest = state.lastDigest = resumeDigest;
    const generation = ++state.generation;
    setButtons();
    poll(resumeDigest, generation);
  }
  return { start, stale, saved, status, sceneChanged:()=>stale(false),
    completedHdr,
    subscribeCompletedHdr(observer) {hdrObservers.add(observer);observer(completedHdr());return ()=>hdrObservers.delete(observer);},
    applySpotAdjustment(proposal, {sourceId,sourceRevision} = {}) {
      const source = completedHdr();
      if (!source || source.id !== sourceId || source.revision !== sourceRevision)
        throw new Error("The photo or retouch settings changed. Sample the point again before applying.");
      const limits = {exposure:[-6,6],red:[0,3],green:[0,3],blue:[0,3],whitebalance:[-3,3],tint:[-3,3]};
      const entries = Object.entries(proposal?.params ?? {});
      if (!entries.length || entries.some(([key,value]) => !Object.hasOwn(limits,key) || typeof value !== "number" ||
          !Number.isFinite(value) || value < limits[key][0] || value > limits[key][1]))
        throw new Error("Invalid bounded spot adjustment.");
      Object.assign(state.retouchParams, proposal.params);
      syncRetouchSliders();queueRetouch(true);
    },
    applyLensCamera: lensControls.applyCamera, lensFov: lensControls.fov,
    lensMinimumFocus: lensControls.minimumFocus, focusLensAt: lensControls.focusAt,
    filmResponse: () => filmResponse?.getState() ?? normalizeFilmResponse() };
}

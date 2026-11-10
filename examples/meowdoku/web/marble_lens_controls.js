// Controls for the existing analytical camera, never a compound-renderer switch.
export const FOCUS_MIN_M = .001;
export const FOCUS_MAX_M = 10;
export const SENSOR_HEIGHT_M = .024;
export const FOCAL_MAX_M = .5;
export const PUPIL_RADIUS_MAX_M = .03125;
export const SELECTION_SCHEMA = "studio-analytical-lens-selection-v1";
const finite = (x, lo, hi, name) => {
  if (typeof x !== "number" || !Number.isFinite(x) || x < lo || x > hi)
    throw new RangeError(`${name} must be between ${lo} and ${hi}.`);
  return x;
};
export const focalFromFov = degrees => SENSOR_HEIGHT_M / (2 * Math.tan(degrees * Math.PI / 360));
export const fovFromFocal = metres => (2 * Math.atan(SENSOR_HEIGHT_M / (2 * metres))) * (180 / Math.PI);

// This compares the selected focal plane with the camera's aim point, not a
// measured surface or the apparent image of an object behind refracting glass.
export function cameraFocusDiagnostic(camera) {
  if (!camera?.eye_m?.every(Number.isFinite) || !camera?.target_m?.every(Number.isFinite)) return null;
  const distanceM = Math.hypot(...camera.eye_m.map((v,i)=>v-camera.target_m[i]));
  const focusM = camera.focus_distance_m, pupilM = camera.aperture_radius_m;
  if (!(distanceM > 0 && focusM > 0 && Number.isFinite(focusM))) return null;
  const mismatch = Math.abs(focusM-distanceM) / distanceM;
  const blurFraction = pupilM * Math.abs(1-distanceM/focusM) /
    (distanceM * Math.tan(camera.fov_degrees*Math.PI/360));
  return {distanceM, focusM, mismatch, blurFraction:Number.isFinite(blurFraction)?blurFraction:null,
    warning:mismatch > .25 && pupilM > 0};
}

export function cameraWithLens(camera, settings, profiles = []) {
  if (!settings || settings.key === "unchanged") return {...camera, lens_selection: null};
  const contract = settings.key === "custom" ? null : profiles.find(p => p.key === settings.key);
  if (settings.key !== "custom" && !contract) throw new Error("Lens collection unavailable; reload it or choose Custom analytical.");
  const f = finite(settings.focalMm / 1000, focalFromFov(100), FOCAL_MAX_M, "Focal length (m)");
  const n = finite(settings.fNumber, 1, 64, "F-number");
  const focus = finite(settings.focusMm / 1000, FOCUS_MIN_M, FOCUS_MAX_M, "Axial focus (m)");
  if (contract) {
    const c = contract.controls;
    finite(f, c.focal_length.minimum, c.focal_length.maximum, "Profile focal length (m)");
    finite(focus, c.focus_distance.minimum ?? FOCUS_MIN_M, FOCUS_MAX_M, "Profile focus (m)");
    finite(n, c.aperture.minimum, c.aperture.maximum ?? 64, "Profile f-number");
  }
  const radius = f / (2 * n);
  finite(radius, .00001, PUPIL_RADIUS_MAX_M, "Pupil radius (m); increase f-number to reduce opening");
  return {...camera, fov_degrees: fovFromFocal(f), aperture_radius_m: radius,
    focus_distance_m: focus,
    lens_selection: {schema: SELECTION_SCHEMA, key: contract?.key ?? null,
      source_sha256: contract?.source_sha256 ?? null, focal_length_m: f,
      f_number: n, sensor_height_m: SENSOR_HEIGHT_M}};
}

export function createStudioLensControls(doc, {fetcher = (...args) => fetch(...args)} = {}) {
  const $ = id => doc.getElementById(id), panel = $("photo-lens-controls");
  if (!panel) return {applyCamera: camera => camera, fov: value => value, minimumFocus:()=>FOCUS_MIN_M,
    focusAt:()=>{}, ready: Promise.resolve()};
  const select = $("photo-lens"), status = $("photo-lens-status");
  const focus = $("photo-focus-mm"), focal = $("photo-focal-mm"), aperture = $("photo-f-number");
  const focusSlider = $("photo-focus-slider"), focalSlider = $("photo-focal-slider"), apertureSlider = $("photo-aperture-slider");
  let profiles = [], failure = "", lastValidFov = null;
  const settings = () => ({key: select.value, focusMm: +focus.value, focalMm: +focal.value, fNumber: +aperture.value});
  const minimumFocus = () => profiles.find(p=>p.key===select.value)?.controls.focus_distance.minimum ?? FOCUS_MIN_M;
  const applyCamera = camera => {
    const result = cameraWithLens(camera, settings(), profiles), diagnostic = $("photo-focus-diagnostic");
    const d = cameraFocusDiagnostic(result);
    if (diagnostic && d) {
      const text = `Camera target ${(d.distanceM*1000).toFixed(2)} mm away; axial focus ${(d.focusM*1000).toFixed(2)} mm. ` +
        (d.warning ? "Focus differs substantially from the camera target. A source lens may not focus this close: use Custom analytical explicitly for a mathematical close-up. " : "") +
        "Target distance is not a surface measurement or a refractive autofocus result. Glass interiors may need a different focus.";
      if (diagnostic.textContent !== text) diagnostic.textContent = text;
      if (diagnostic.dataset.warning !== String(d.warning)) diagnostic.dataset.warning = String(d.warning);
      if (panel.dataset.focusWarning !== String(d.warning)) panel.dataset.focusWarning = String(d.warning);
    }
    return result;
  };
  const sync = () => {
    const off = select.value === "unchanged", contract = profiles.find(p => p.key === select.value);
    const fMin = contract?.controls.focal_length.minimum * 1000 || focalFromFov(100) * 1000;
    const fMax = contract?.controls.focal_length.maximum * 1000 || FOCAL_MAX_M * 1000;
    const focusMin = contract?.controls.focus_distance.minimum * 1000 || 1;
    const nMin = Math.max(contract?.controls.aperture.minimum || 1, +focal.value / (2 * PUPIL_RADIUS_MAX_M * 1000));
    for (const el of [focus, focal, aperture, focusSlider, focalSlider, apertureSlider]) el.disabled = off;
    focal.disabled = focalSlider.disabled = off || fMin === fMax;
    const nMax = contract?.controls.aperture.maximum ?? 64;
    aperture.disabled = apertureSlider.disabled = off || nMin === nMax;
    aperture.max = String(nMax); apertureSlider.max = String(Math.log2(nMax));
    focal.min = focalSlider.min = String(fMin); focal.max = focalSlider.max = String(fMax);
    focus.min = String(focusMin); focusSlider.min = String(Math.log10(focusMin));
    aperture.min = String(nMin); apertureSlider.min = String(Math.log2(nMin));
    focusSlider.value = String(Math.log10(+focus.value)); focalSlider.value = focal.value;
    apertureSlider.value = String(Math.log2(+aperture.value));
    try {
      const result = applyCamera({focus_distance_m: .11, fov_degrees: 18});
      lastValidFov = off ? null : result.fov_degrees;
      status.textContent = off ? "Original framing, focus and pupil are unchanged. Choose Custom analytical or a source profile to enable controls." :
        `${contract ? "Authored profile approximation" : "Custom analytical camera"}: ${(+focal.value).toFixed(2)} mm, f/${(+aperture.value).toFixed(2)}, axial focus ${(+focus.value).toFixed(2)} mm. FOV ${result.fov_degrees.toFixed(2)}°; entrance-pupil radius ${(result.aperture_radius_m * 1000).toFixed(2)} mm. ` +
        (contract?.controls.focus_distance.minimum != null ? `Source minimum focus ${focusMin} mm; its original distance datum is unspecified. ` : "1 mm is a numerical limit, not a real lens close-focus promise. ") +
        "24 mm reference sensor height; width follows image aspect. No calibrated distortion, vignetting or lens spectral transmission is applied. " +
        (contract && nMin > contract.controls.aperture.minimum ? `The source opens to f/${contract.controls.aperture.minimum}; this renderer's 31.25 mm pupil-radius limit requires f/${nMin.toFixed(2)} or higher. ` : "") +
        (contract?.source_profile?.specs_only ? "Specs-only mirror lens: minimum focus is unknown; central obstruction and annular bokeh are not modeled. " : "") +
        "10 m focus and f/64 are application travel limits; a source fixed aperture remains fixed.";
      status.dataset.valid = "true";
    } catch (error) { status.textContent = error.message; status.dataset.valid = "false"; }
    if (failure) status.textContent += " " + failure;
    doc.dispatchEvent?.(new (doc.defaultView?.Event ?? Event)("studio-lens-change"));
  };
  select.addEventListener("change", () => {
    const c = profiles.find(p => p.key === select.value);
    if (c) {
      focal.value = String(c.controls.focal_length.default * 1000);
      focus.value = String(Math.max(+focus.value, c.controls.focus_distance.minimum * 1000));
      aperture.value = String(Math.min(c.controls.aperture.maximum ?? 64, Math.max(+aperture.value, c.controls.aperture.minimum, +focal.value / (2 * PUPIL_RADIUS_MAX_M * 1000))));
    }
    sync();
  });
  for (const input of [focus, focal, aperture]) input.addEventListener("input", sync);
  focusSlider.addEventListener("input", () => {focus.value = String(10 ** +focusSlider.value); sync();});
  focalSlider.addEventListener("input", () => {focal.value = focalSlider.value; aperture.value = String(Math.max(+aperture.value, +focal.value / (2 * PUPIL_RADIUS_MAX_M * 1000))); sync();});
  apertureSlider.addEventListener("input", () => {aperture.value = String(2 ** +apertureSlider.value); sync();});
  $("photo-frame-focus")?.addEventListener("click", () => $("frame-marble")?.click());
  sync();
  const ready = (async () => {
    try {
      const response = await fetcher("/api/marble_photo/lenses");
      const catalog = await response.json();
      if (!response.ok || catalog.schema !== "studio-lens-catalog-v1") throw new Error(catalog.error || "Lens collection could not load.");
      profiles = catalog.profiles;
      for (const p of profiles) { const option = doc.createElement("option"); option.value = p.key; option.textContent = `${p.key} (authored approximation)`; select.append(option); }
      for (const p of catalog.references ?? []) {const option = doc.createElement("option"); option.disabled = true; option.value = p.key; option.textContent = `${p.key.split(":").at(-1)} (reference only)`; select.append(option);}
      $("photo-lens-inventory").textContent = `${profiles.length} source profiles available; ${(catalog.references ?? []).length} compound reference states await their shader adapter and calibration. The original compound prescriptions remain untouched.`;
    } catch (error) { failure = `${error.message} Custom analytical controls remain available.`; }
    sync();
  })();
  // An incomplete number edit must not stop the animation loop. Exposures and
  // live camera writes still fail explicitly through applyCamera.
  return {applyCamera, ready, minimumFocus, fov: value => lastValidFov ?? value,
    focusAt(distanceM, {camera, custom = false} = {}) {
      if (custom) {
        finite(distanceM, FOCUS_MIN_M, FOCUS_MAX_M, "Axial focus (m)");
        if (!camera) throw new Error("The actual camera is required to preserve framing and pupil.");
        const f = focalFromFov(camera.fov_degrees), n = f/(2*camera.aperture_radius_m);
        const next = {key:"custom",focusMm:distanceM*1000,focalMm:f*1000,fNumber:n};
        cameraWithLens(camera,next,profiles); // Validate before changing any UI state.
        select.value="custom";focal.value=String(next.focalMm);aperture.value=String(n);
        focus.value=String(next.focusMm);sync();return;
      }
      finite(distanceM, minimumFocus(), FOCUS_MAX_M, "Axial focus (m)");
      if(select.value!=="unchanged"){
        cameraWithLens({}, {...settings(),focusMm:distanceM*1000}, profiles);
        focus.value=String(distanceM*1000);sync();
      }
    }};
}

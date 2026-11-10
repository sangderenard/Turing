// Live client-side re-tonemap of a saved linear HDR exposure (the
// Spectral thin-lens tracer's own exposure-NNNNNN.npz, raw radiance before
// any display mapping). Lets exposure/gamma/white-balance/intensity be
// retried instantly against the one GPU pass already paid for, instead of
// re-running the path tracer for every adjustment.
//
// Reproduces the worker's own filmic formula (see
// camera_designer/marble_preview_worker.py::_snapshot, transport
// "fixed:3"): mapped = 1 - exp(-linear * 2/white), then the standard sRGB
// transfer function, then **gamma. "white" there is a fixed 99th-percentile
// of the raw linear luma, computed once per loaded exposure; the Exposure
// slider is a stop multiplier on top of it, so turning it to 0 reproduces
// the worker's own picture exactly (same formula, same percentile) and the
// user can push it either way if that auto white point picked a glare spot.
import { parseNpz } from "./npz.js";

const LUMA = [0.2126, 0.7152, 0.0722];

function percentile(sortedAscending, p) {
  if (!sortedAscending.length) return 0;
  const index = (sortedAscending.length - 1) * (p / 100);
  const lower = Math.floor(index), upper = Math.ceil(index);
  if (lower === upper) return sortedAscending[lower];
  const frac = index - lower;
  return sortedAscending[lower] * (1 - frac) + sortedAscending[upper] * frac;
}

function srgbEncode(linear) {
  return linear <= 0.0031308 ? 12.92 * linear : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
}

export function createHdrRetouch(canvas) {
  let hdr = null; // { width, height, data: Float32Array HxWx3, baseWhite }
  let frame = 0;

  const params = { exposure: 0, gamma: 1, red: 1, green: 1, blue: 1, intensity: 1 };

  async function load(url) {
    const buffer = await (await fetch(url, { cache: "force-cache" })).arrayBuffer();
    const arrays = await parseNpz(buffer);
    const exposure = arrays.get("linear_rgb_exposure");
    if (!exposure) throw new Error("exposure NPZ has no linear_rgb_exposure array");
    const [height, width] = exposure.shape;
    const data = exposure.data instanceof Float32Array ? exposure.data : Float32Array.from(exposure.data);
    const luma = new Float32Array(width * height);
    for (let p = 0, o = 0; p < luma.length; p++, o += 3) {
      luma[p] = data[o] * LUMA[0] + data[o + 1] * LUMA[1] + data[o + 2] * LUMA[2];
    }
    const nonzero = Array.from(luma).filter((v) => v > 0).sort((a, b) => a - b);
    const baseWhite = Math.max(percentile(nonzero, 99), 1e-6);
    hdr = { width, height, data, baseWhite };
    return hdr;
  }

  function unload() { hdr = null; }
  function isLoaded() { return hdr !== null; }

  function setParams(next) { Object.assign(params, next); }

  function reset(displayGamma = 1) {
    Object.assign(params, { exposure: 0, gamma: displayGamma || 1, red: 1, green: 1, blue: 1,
      whitebalance: 0, tint: 0, intensity: 1 });
  }

  function render() {
    if (!hdr) return false;
    const { width, height, data, baseWhite } = hdr;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(width, height);
    const out = image.data;
    const expMul = Math.pow(2, params.exposure);
    const invGamma = 1 / Math.max(1e-3, params.gamma);
    const intensity = Math.max(0, params.intensity);
    // White balance: warm (+) boosts red / cuts blue, cool (-) the reverse.
    // Tint: magenta (+) cuts green, green (-) boosts green -- the standard
    // two-axis photographic correction, applied as linear-space gain before
    // the tonemap curve, same place Red/Green/Blue already act.
    const wb = Math.pow(2, params.whitebalance || 0);
    const tn = Math.pow(2, params.tint || 0);
    const gain = [params.red * expMul * wb, params.green * expMul / tn, params.blue * expMul / wb];
    const k = 2 / baseWhite;
    for (let p = 0, o = 0; p < data.length; p += 3, o += 4) {
      for (let c = 0; c < 3; c++) {
        const mapped = 1 - Math.exp(-Math.max(data[p + c] * gain[c], 0) * k);
        let v = Math.min(1, Math.max(0, srgbEncode(mapped))) ** invGamma;
        v = Math.min(1, Math.max(0, v * intensity));
        out[o + c] = Math.round(v * 255);
      }
      out[o + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return true;
  }

  function scheduleRender() {
    const token = ++frame;
    requestAnimationFrame(() => { if (token === frame) render(); });
  }

  return { load, unload, isLoaded, setParams, reset, render, scheduleRender, params };
}

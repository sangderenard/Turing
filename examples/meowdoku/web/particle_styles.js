// Particle-field RENDER STYLES for the compute games' point cloud (compute.js: PointCloud).
//
// Render only: a style reads the packed points PointCloud already uploaded ([x, y, rgba8, meta] per displayed id; meta =
// 100 (moving) or 200 (at rest) + radius * 1000, below 50 = not alive) and draws them; it never touches the law, the
// recording or the data.  The packed points carry NO velocity (only the moving / at-rest flag the pack pass derives from
// it), so "heat" colours by the local density and by the share of moving grains -- not by speed.
//
// Styles
//   points  today's look (PointCloud's own passes: sub-pixel accumulation, true-radius discs).  The default.
//   liquid  screen-space fluid: depth + thickness sphere splats, a narrow-range bilateral blur of the depth, normals rebuilt
//           from it, shaded with Fresnel reflection and a thickness-tinted refraction of the page background, lit by the
//           scene's key / fill light.  Needs float (or half-float) render targets.
//   sand    shaded sphere impostors, per-id colour, a cheap ambient-occlusion darkening from the local grain density.
//   glow    additive soft sprites (sparkler / plasma).
//   heat    density / motion mapped through a heat ramp instead of the hue.  Needs half-float render targets.
//   material  (page-side: PAGE_STYLES) every grain a real instanced icosphere through the scene's own Phong material shader
//           and lights, with real depth against the scene's objects; no sprites, no quads.  Falls back to sand when too many.
//   stamp   a persistent texture over the board that every frame's points stamp sprites into and that decays slowly
//           (trails; the start of "sprites stamped into memory").  8-bit, so it needs no float targets.  The current
//           points are drawn over it by PointCloud's own passes.
//
// The pure part (the registry, what a device can run, which style is used instead) is at the top and is node-tested.

import { sizeDepth } from "./world_curve.js";                               // [CURVE] a grain's size depth: the eye depth, or 1 under an orthographic projection
const PERSPECTIVE_W = (() => { const m = new Array(16).fill(0); m[11] = -1; return m; })();       // (no projection given: a perspective one)

export const STYLES = [
  { id: "points", label: "points", needs: null },
  { id: "liquid", label: "liquid", needs: "float" },
  { id: "sand", label: "sand", needs: null, wants: "float" },     // without float targets: no ambient occlusion, still sand
  { id: "glow", label: "glow", needs: null },
  { id: "heat", label: "heat", needs: "float" },
  { id: "stamp", label: "stamp (memory)", needs: null },
];
// STYLES is the server's declared list (fountain.py PARTICLE_STYLES; a test holds the two equal).  The styles the page adds
// itself live in PAGE_STYLES and reach the select through app.js's pageSpecs, so the server's controls (and the compiled
// products keyed on them) are untouched.
export const PAGE_STYLES = [
  { id: "material", label: "material (scene spheres)", needs: null, scene: true, fallback: "sand" },
  // material-mode glass: the scene shader's own material (its alpha, translucence, enamel, profiles), composited order-independently (OIT)
  { id: "stained", label: "stained glass (scene glass, OIT)", needs: null, scene: true, fallback: "sand", budgetScale: 0.6 },   // both faces accumulate: fewer grains
  // the explicit fast mode: plain alpha blending in draw order, back faces then front faces (the look before OIT)
  { id: "stained_fast", label: "stained glass (fast: alpha blend)", needs: null, scene: true, fallback: "sand", budgetScale: 0.6 },
  // every grain an animal emoji (see "emoji style" below): (a) sprites textured from an atlas baked at export, alpha-tested and
  // depth-written per sprite; (b) instanced extruded emoji meshes through the scene's material shader, which falls back to (a)
  { id: "emoji", label: "emoji (image sprites)", needs: "float", sprite: true },
  { id: "emoji_mesh", label: "emoji (3-D meshes)", needs: "float", scene: true, emojiMesh: true, fallback: "emoji" },
  { id: "designed", label: "saved marble / die (full mesh)", needs: null, scene: true, designed: true },
];
export const ALL_STYLES = [...STYLES, ...PAGE_STYLES];
export const STYLE_IDS = ALL_STYLES.map((s) => s.id);
export const DEFAULT_STYLE = "points";
const MIN_POINT_SIZE = 8;     // a sprite style needs point sprites at least this wide

// caps: { floatRender: "float" | "half" | null, maxPointSize: number } (see capabilitiesOf); null = no GL at all
export function styleAvailability(id, caps) {
  const style = ALL_STYLES.find((s) => s.id === id);
  if (!style) return { ok: false, note: `unknown style '${id}'` };
  if (id === DEFAULT_STYLE) return { ok: true, note: "" };
  if (!caps) return { ok: false, note: "no WebGL context" };
  if (!style.scene && !(caps.maxPointSize >= MIN_POINT_SIZE)) return { ok: false, note: `this device draws points only up to ${caps.maxPointSize || 0} px` };
  if (style.needs === "float" && !caps.floatRender) return { ok: false, note: "needs float or half-float render targets (EXT_color_buffer_float), which this device lacks" };
  if (style.wants === "float" && !caps.floatRender) return { ok: true, note: "no float render targets here: without the ambient-occlusion darkening" };
  if (style.needs === "float" && caps.floatRender === "half") return { ok: true, note: "half-float render targets (less depth precision)" };
  return { ok: true, note: "" };
}

// The style to draw: the requested one when the device can run it (and it has not failed at run time -- `broken` maps id -> reason),
// else its fallback ("sand" for material, "points" for the rest), with the reason (shown in the control's title).
// `scene` (optional) is what the material style needs to know: { ready, count, radiusPx } (see materialPlan).
export function chooseStyle(requested, caps, broken = {}, scene = null) {
  const id = STYLE_IDS.includes(requested) ? requested : DEFAULT_STYLE;
  if (id === DEFAULT_STYLE) return { style: id, requested: id, reason: "" };
  const spec = ALL_STYLES.find((s) => s.id === id), fallback = spec.fallback || DEFAULT_STYLE;
  const instead = (why) => {
    const next = fallback === DEFAULT_STYLE ? { style: DEFAULT_STYLE } : chooseStyle(fallback, caps, broken, scene);
    return { style: next.style, requested: id, reason: `${id} unavailable (${why}): showing ${next.style}` };
  };
  if (broken[id]) return instead(broken[id]);
  const av = styleAvailability(id, caps);
  if (!av.ok) return instead(av.note);
  if (spec.scene && scene) {
    const budget = { ...MATERIAL_BUDGET, maxInstances: MATERIAL_BUDGET.maxInstances * (spec.budgetScale || 1) };
    if (spec.designed) {
      const plan = designedMeshPlan(scene);
      if (!plan.ok) return instead(plan.reason);
      return { style: id, requested: id, reason: plan.note, lod: plan.lod };
    }
    if (spec.emojiMesh) {
      const plan = emojiMeshPlan({ ...scene, budget, force: EMOJI_OPTIONS.forceMesh });
      if (!plan.ok) return instead(plan.reason);
      return { style: id, requested: id, reason: plan.note, lod: plan.lod, tris: plan.tris };
    }
    const plan = materialPlan({ ...scene, budget });
    if (!plan.ok) return instead(plan.reason);
    return { style: id, requested: id, reason: plan.note, lod: plan.lod, tris: plan.tris };
  }
  if (spec.sprite && scene) {
    const plan = emojiSpritePlan({ ...scene, force: EMOJI_OPTIONS.forceSprite });
    if (!plan.ok) return instead(plan.reason);
  }
  return { style: id, requested: id, reason: av.note };
}

// ---------------------------------------------------------------- glass compositing (pure)
// How a glass grain (a material with an opacity below OPAQUE_ABOVE, or the stained style) is composited:
//   "oit"   weighted blended order-independent transparency (McGuire and Bavoil 2013): every glass fragment is accumulated
//           with a depth weight into an RGBA16F target (premultiplied colour * w, alpha * w) and the optical depth tau = -ln(1 - a)
//           into a second one (the sum of tau is the product of (1 - a), exactly the "revealage"; the sum form needs no
//           per-target blend function, which WebGL2 lacks), then ONE composite pass over the scene.  Needs float or half-float
//           render targets.  The fragment colour and alpha are the scene shader's own (base_material).
//   "alpha" plain alpha blending in draw order (the stained_fast style, and the fallback without float render targets).
export const OPAQUE_ABOVE = 0.95;
export const OIT_WEIGHT_MAX = 100;       // (the sum of w * colour must stay inside half-float range with many overlapping faces)
// the depth weight of a fragment at eye distance `z` (McGuire and Bavoil's eye-space weight 10 / (1e-5 + (z/5)^2 + (z/200)^6): the
// window-depth form saturates at its cap for a perspective depth buffer; OIT_FRAG_WRITE is the GLSL twin).  The coverage `a` is
// already in the accumulated colour (rgb * a) and alpha.
export function oitWeight(z) {
  const w = 10 / (1e-5 + Math.pow(Math.abs(z) / 5, 2) + Math.pow(Math.abs(z) / 200, 6));
  return Math.min(OIT_WEIGHT_MAX, Math.max(1e-2, w));
}
export const oitTau = (a) => -Math.log(1 - Math.min(Math.max(a, 0), 0.9999));
// The light that gets through a stack of fragments of coverages `alphas`: the product of (1 - a), however they are ordered.
export const transmittance = (alphas) => alphas.reduce((t, a) => t * (1 - Math.min(Math.max(a, 0), 0.9999)), 1);
// The composite pass: the accumulated (rgb * w, a * w) and the summed tau -> { rgb, alpha } to blend over the scene with
// (SRC_ALPHA, ONE_MINUS_SRC_ALPHA).
export function oitComposite(accum, tau) {
  const reveal = Math.exp(-tau), w = Math.max(accum[3], 1e-5);
  return { rgb: [accum[0] / w, accum[1] / w, accum[2] / w], alpha: 1 - reveal, reveal };
}
// Which compositing a glass draw uses, and why (the reason is shown in the control's title).  `fast`: the explicit fast style.
// `broken`: why OIT failed at run time here ("" when it has not).
export function glassMode({ fast = false, caps = null, broken = "" } = {}) {
  if (fast) return { mode: "alpha", reason: "glass: plain alpha blending in draw order (fast mode; not order-independent)" };
  if (!caps || !caps.floatRender) return { mode: "alpha", reason: "glass: alpha blending in draw order, not OIT: OIT needs float or half-float render targets (EXT_color_buffer_float / EXT_color_buffer_half_float), which this device lacks" };
  if (broken) return { mode: "alpha", reason: `glass: alpha blending in draw order, not OIT: ${broken}` };
  return { mode: "oit", reason: `glass: weighted blended OIT (order-independent), ${caps.floatRender === "float" ? "RGBA16F" : "half-float"} accumulation and optical-depth targets` };
}
// The depth formats an OIT target may take (a depth blit needs the scene framebuffer's own format), in the order tried.
export const OIT_DEPTH_FORMATS = {
  d24s8: { internal: "DEPTH24_STENCIL8", attach: "DEPTH_STENCIL_ATTACHMENT" },
  d24: { internal: "DEPTH_COMPONENT24", attach: "DEPTH_ATTACHMENT" },
  d16: { internal: "DEPTH_COMPONENT16", attach: "DEPTH_ATTACHMENT" },
  d32f: { internal: "DEPTH_COMPONENT32F", attach: "DEPTH_ATTACHMENT" },
};
export const OIT_DEPTH_ORDER = ["d24s8", "d24", "d16", "d32f"];
// Is a material with this pbr opacity glass (composited, not drawn opaque)?
export const isGlass = (opacity) => opacity < OPAQUE_ABOVE;
// The fragment stage's addition for OIT (extendForOit): the accumulation write beside the original colour write.
export const OIT_FRAG_WRITE = `
    if (uOit) {
        float oa = clamp(alpha, 0.0, 0.9999);
        float oz = abs(vPosV.z);
        float ow = clamp(10.0 / (1e-5 + pow(oz / 5.0, 2.0) + pow(oz / 200.0, 6.0)), 1e-2, ${OIT_WEIGHT_MAX}.0);
        FragColor = vec4(srgb * oa, oa) * ow;
        oitTau = vec4(-log(1.0 - oa), 0.0, 0.0, 0.0);
    } else {
        FragColor = vec4(srgb, alpha);
        oitTau = vec4(0.0);
    }`;
// -> the fragment source with the OIT outputs added (the original colour write is kept, unchanged, for uOit = false);
// throws an Error naming what the source did not have.
export function extendForOit(frag) {
  const out = /\bout\s+vec4\s+FragColor\s*;/.exec(frag);
  if (!out) throw new Error("the fragment shader has no `out vec4 FragColor;`");
  const write = /FragColor\s*=\s*vec4\s*\(\s*srgb\s*,\s*alpha\s*\)\s*;/.exec(frag);
  if (!write || write.index < out.index) throw new Error("the fragment shader's final colour write was not found");
  return frag.slice(0, out.index) + "layout(location = 0) out vec4 FragColor;\nlayout(location = 1) out vec4 oitTau;\nuniform bool uOit;"
    + frag.slice(out.index + out[0].length, write.index) + OIT_FRAG_WRITE.trim() + frag.slice(write.index + write[0].length);
}

// ---------------------------------------------------------------- colour source (pure)
// Where a style's colour comes from.  "simulation": the per-id colour the packed points carry (today's).  "solid": one colour
// from a picker.  "material": the colour (for the material and stained styles, the whole material) of one of the scene's
// declared materials.  "size": a ramp over the grain's true radius.  "species": one colour per species.  A PALETTE maps the
// size ramp, the species colours and (as a hue lookup) the simulation's own colours; "native" leaves them as they are.
// Every style takes the result as `tint` = [r, g, b, use] (display colour, use 1 = this colour instead of the grain's) and
// `grain` = { mode, lut, bands, range } (see GRAIN_COLOR_GLSL).
export const COLOR_SOURCES = [["simulation", "simulation (per grain)"], ["solid", "solid colour"], ["material", "scene material"],
                              ["size", "grain size"], ["species", "species"]];
export const DEFAULT_COLOR = { source: "simulation", solid: "#d9a441", material: "", palette: "native" };
export const PALETTES = {
  native: null,
  sea: [[0.02, 0.05, 0.16], [0.04, 0.22, 0.58], [0.04, 0.52, 0.62], [0.32, 0.86, 0.84], [0.93, 0.99, 0.96]],        // deep navy, blue, teal, aqua, foam
  ember: [[0.07, 0.02, 0.03], [0.5, 0.06, 0.04], [0.92, 0.36, 0.05], [1.0, 0.76, 0.2], [1.0, 0.97, 0.8]],
  meadow: [[0.04, 0.15, 0.08], [0.18, 0.44, 0.12], [0.56, 0.72, 0.2], [0.96, 0.86, 0.36], [0.9, 0.46, 0.62]],
};
export const PALETTE_IDS = Object.keys(PALETTES);
const NATIVE_RAMP = [[0.25, 0.12, 0.6], [0.1, 0.45, 0.92], [0.1, 0.82, 0.6], [0.92, 0.86, 0.2], [0.95, 0.3, 0.15]];     // the size ramp without a palette
const NATIVE_SPECIES = [[0.95, 0.35, 0.3], [0.3, 0.7, 0.95], [0.95, 0.8, 0.25], [0.5, 0.85, 0.45], [0.75, 0.5, 0.9], [0.95, 0.55, 0.2], [0.3, 0.85, 0.8], [0.9, 0.5, 0.7]];
export const LUT_STOPS = 16, MAX_SPECIES = 8;
const sample = (stops, u) => {            // piecewise-linear colour at u in [0, 1]
  const f = Math.min(1, Math.max(0, u)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), k = f - i;
  return [0, 1, 2].map((c) => stops[i][c] + (stops[i + 1][c] - stops[i][c]) * k);
};
// the colours a mode looks up: LUT_STOPS smooth stops (size, simulation-through-palette) or one entry per species
export function paletteLut(name, species = 0) {
  const lut = new Float32Array(LUT_STOPS * 3), stops = PALETTES[name] || null;
  for (let i = 0; i < LUT_STOPS; i++) {
    const c = species ? (stops ? sample(stops, (i + 0.5) / species) : NATIVE_SPECIES[i % NATIVE_SPECIES.length]) : sample(stops || NATIVE_RAMP, i / (LUT_STOPS - 1));
    lut.set(c, i * 3);
  }
  return lut;
}
// SPECIES from the packed radius, in ONE place.  The law does not export a species column yet, so a grain's species is the
// nearest of the manifest's species radii (manifest.species rows [name, radius, ...]); an exported species column makes this
// function read it instead (and GLSL grainColor's band count become that column).
export const DEFAULT_SPECIES = [["grain", 0.009], ["bead", 0.0105], ["dust", 0.012], ["pearl", 0.014]];
export function speciesTable(rows) {
  const t = (rows && rows.length ? rows : DEFAULT_SPECIES).slice(0, MAX_SPECIES).map((r) => ({ name: r[0], radius: r[1] }));
  return t.sort((a, b) => a.radius - b.radius);
}
export function speciesBands(table) {       // the radii between neighbouring species (unused edges 1e9: a grain never exceeds them)
  const bands = new Float32Array(MAX_SPECIES - 1).fill(1e9);
  for (let i = 0; i + 1 < table.length; i++) bands[i] = (table[i].radius + table[i + 1].radius) / 2;
  return bands;
}
export function speciesOf(radius, table) {
  let s = 0;
  for (let i = 0; i + 1 < table.length; i++) if (radius > (table[i].radius + table[i + 1].radius) / 2) s = i + 1;
  return s;
}
export function parseHexColor(text) {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(String(text || "").trim());
  return m ? [1, 2, 3].map((i) => parseInt(m[i], 16) / 255) : null;
}
const toDisplay = (v) => Math.pow(Math.min(1, Math.max(0, v)), 1 / 2.2);     // a material's albedo is linear; the sprite styles shade display colours
const GRAIN_MODE = { own: 0, size: 2, species: 3, palette: 4 };
// pick = { source, solid, material, palette }; albedoOf(name) -> the material's linear [r, g, b] or null; speciesRows = manifest.species
export function resolveColorSource(pick, albedoOf = () => null, speciesRows = null) {
  const source = COLOR_SOURCES.some(([id]) => id === (pick && pick.source)) ? pick.source : "simulation";
  const palette = PALETTE_IDS.includes(pick && pick.palette) ? pick.palette : "native";
  const table = speciesTable(speciesRows);
  const radii = table.map((s) => s.radius), lo = Math.min(...radii) * 0.9, hi = Math.max(...radii) * 1.05;
  const grain = (mode, lut) => ({ mode, lut, bands: speciesBands(table), range: [lo, 1 / Math.max(1e-6, hi - lo)] });
  const none = { mode: "simulation", tint: [0, 0, 0, 0], material: null, reason: "", species: table, grain: grain(0, paletteLut("native")) };
  if (source === "solid") {
    const rgb = parseHexColor(pick.solid);
    return rgb ? { ...none, mode: "solid", tint: [...rgb, 1] } : { ...none, reason: `'${pick.solid}' is not a colour: using the simulation's` };
  }
  if (source === "material") {
    const name = pick.material, albedo = name ? albedoOf(name) : null;
    return albedo ? { ...none, mode: "material", tint: [...albedo.map(toDisplay), 1], material: name }
                  : { ...none, reason: name ? `the scene has no material '${name}': using the simulation's colour` : "no material chosen: using the simulation's colour" };
  }
  if (source === "size") return { ...none, mode: "size", grain: grain(GRAIN_MODE.size, paletteLut(palette)) };
  if (source === "species") return { ...none, mode: "species", grain: grain(GRAIN_MODE.species, paletteLut(palette, table.length)) };
  return palette === "native" ? none : { ...none, grain: grain(GRAIN_MODE.palette, paletteLut(palette)) };
}
// Sets the colour-source uniforms of a program that carries GRAIN_COLOR_GLSL; `loc(name)` -> its uniform location.
export function grainUniforms(gl, loc, tint, grain) {
  gl.uniform4fv(loc("uTint"), tint || NO_TINT);
  const g = grain || null;
  gl.uniform4f(loc("uGrain"), g ? g.mode : 0, g ? g.range[0] : 0, g ? g.range[1] : 1, 0);
  if (g) { gl.uniform3fv(loc("uLut"), g.lut); gl.uniform1fv(loc("uBands"), g.bands); }
}
// The colour source on the GPU: one function every particle vertex stage calls.  `own` is the grain's own colour, `meta` the
// packed metadata (the true radius is decoded here).  Declared before main().
export const GRAIN_COLOR_GLSL = `
uniform vec4 uTint;        // solid / material colour: rgb, and 1 = use it instead of the grain's own
uniform vec4 uGrain;       // x: 0 own colour, 2 size ramp, 3 species, 4 own colour's hue through the palette; y, z: size ramp low radius, 1 / span
uniform vec3 uLut[${LUT_STOPS}];
uniform float uBands[${MAX_SPECIES - 1}];
vec3 lutAt(float u) { float f = clamp(u, 0.0, 1.0) * ${LUT_STOPS - 1}.0; int i = int(floor(f)); return mix(uLut[i], uLut[min(i + 1, ${LUT_STOPS - 1})], f - float(i)); }
vec4 grainColor(vec4 own, float meta) {
  if (uTint.w > 0.5) return vec4(uTint.rgb, own.a);
  float m = uGrain.x;
  if (m < 0.5) return own;
  float r = (meta - floor(meta / 100.0 + 0.001) * 100.0) * 0.001;
  if (m < 2.5) return vec4(lutAt((r - uGrain.y) * uGrain.z), own.a);
  if (m < 3.5) { int s = 0; for (int i = 0; i < ${MAX_SPECIES - 1}; i++) if (r > uBands[i]) s++; return vec4(uLut[s], own.a); }
  vec3 c = own.rgb; float mx = max(c.r, max(c.g, c.b)), d = mx - min(c.r, min(c.g, c.b));
  float h = d < 1e-4 ? 0.0 : (mx == c.r ? mod((c.g - c.b) / d, 6.0) : mx == c.g ? (c.b - c.r) / d + 2.0 : (c.r - c.g) / d + 4.0) / 6.0;
  return vec4(lutAt(h), own.a);
}
`;

// ---------------------------------------------------------------- heat tone mapping (pure)
// The heat style maps ONE scalar field through a log-like curve, x = ln(1 + HEAT_K q), normalises x between a running low and
// high percentile of the current field (so a burst does not clip the top and a resting pile does not flatten the bottom),
// then applies a gamma.  The quantity q is pluggable: GLSL `heatQuantity` (below) is the only place that says what q is --
// today the local density weighted by the moving share, later a per-grain kinetic energy / temperature channel.
export const HEAT_K = 50;
export const HEAT_X_MAX = Math.log(1 + HEAT_K * 20);       // the histogram covers q in [0, 20]
export const HEAT_PERCENTILES = [0.02, 0.98];
export const HEAT_TAU = 1.0;                                  // seconds: the range adapts to the field in about this long
export const HEAT_MIN_SPAN = 0.6;                             // the narrowest range in x (a flat field is not stretched into noise)
export const HEAT_FIXED = [0.2, Math.log(1 + HEAT_K * 3)];   // the manual "fixed range" (x)
export const HEAT_HIST_BINS = 256;
export const quantityToX = (q) => Math.log(1 + HEAT_K * Math.max(0, q));
export const xToQuantity = (x) => (Math.exp(x) - 1) / HEAT_K;
// counts[bin] over HEAT_HIST_BINS bins of x in [0, HEAT_X_MAX]; p in [0, 1] -> x (bin centre), or null with no samples
export function histogramPercentile(counts, p) {
  let total = 0;
  for (const c of counts) total += c;
  if (!total) return null;
  const want = p * total;
  let run = 0;
  for (let i = 0; i < counts.length; i++) { run += counts[i]; if (run >= want && counts[i] > 0) return ((i + 0.5) / counts.length) * HEAT_X_MAX; }
  return HEAT_X_MAX;
}
// the target [lo, hi] (x) from a histogram; null when too few samples
export function heatTarget(counts, minSamples = 16) {
  let total = 0;
  for (const c of counts) total += c;
  if (total < minSamples) return null;
  const lo = histogramPercentile(counts, HEAT_PERCENTILES[0]), hi = histogramPercentile(counts, HEAT_PERCENTILES[1]);
  return [lo, Math.max(hi, lo + HEAT_MIN_SPAN)];
}
// one step of the exposure's adaptation: exponential approach to `target` over about tau seconds (prev null = take it)
export function adaptRange(prev, target, dt, tau = HEAT_TAU) {
  if (!target) return prev;
  if (!prev) return [target[0], target[1]];
  const k = 1 - Math.exp(-Math.max(0, dt) / tau);
  return [prev[0] + (target[0] - prev[0]) * k, prev[1] + (target[1] - prev[1]) * k];
}
// the mapped value in [0, 1] (the shader's twin)
export function heatMap(q, range, gamma = 1) {
  const t = Math.min(1, Math.max(0, (quantityToX(q) - range[0]) / Math.max(1e-6, range[1] - range[0])));
  return Math.pow(t, gamma);
}
// the GLSL every heat pass shares: what the quantity IS (the one pluggable place), and its log curve
const HEAT_GLSL = `
float heatQuantity(vec4 a) {            // a = the density target: r the area fraction covered, g the moving part of it
  float d = a.r, m = a.g / max(d, 1e-3);
  return d * (0.3 + 0.7 * m);           // PLUG HERE: a per-grain kinetic energy / temperature channel replaces this
}
float heatX(float q) { return log(1.0 + ${HEAT_K.toFixed(1)} * max(q, 0.0)); }
`;

// ---------------------------------------------------------------- material style (pure)
// Every grain is an instanced icosphere: 20 triangles (12 vertices), 80 (42) or 320 (162).  The detail follows the grain's
// size on screen and the grain count; above a count the GPU cannot hold a frame rate for (MATERIAL_BUDGET, measured; see the
// commit message) the style is "sand" and says so.
export const SPHERE_TRIS = [20, 80, 320];
export const MATERIAL_BUDGET = {
  maxInstances: 130000,          // above this many displayed grains the material style is not drawn (GTX 1660 SUPER, 641x207: 65k 4.8 ms, 262k 16.9 ms, 1M 53 ms)
  maxTriangles: 2.6e6,           // instances x triangles per frame the detail is chosen to stay under
  minRadiusPx: 0.4,              // a grain under this (screen radius, px) is a speck: sand draws it with coverage
  detailAtPx: [1.5, 4],          // below the first: 20 triangles; below the second: 80; else 320 (finer than the silhouette needs: a highlight is a small part of a sphere)
};
export function materialPlan({ ready = true, count = 0, radiusPx = 4, budget = MATERIAL_BUDGET } = {}) {
  if (!ready) return { ok: false, reason: "the scene's material shader is not in use here (flat view, or it could not be extended for instancing)" };
  if (count > budget.maxInstances) return { ok: false, reason: `${count.toLocaleString("en-US")} grains is more than the ${budget.maxInstances.toLocaleString("en-US")} meshes hold a frame rate for here` };
  if (radiusPx < budget.minRadiusPx) return { ok: false, reason: `grains are ${radiusPx.toFixed(2)} px in radius at this zoom: too small to mesh` };
  const bySize = radiusPx < budget.detailAtPx[0] ? 0 : radiusPx < budget.detailAtPx[1] ? 1 : 2;
  let byCount = 2;
  while (byCount > 0 && count * SPHERE_TRIS[byCount] > budget.maxTriangles) byCount--;
  const lod = Math.min(bySize, byCount);
  return { ok: true, lod, tris: SPHERE_TRIS[lod], note: `${count.toLocaleString("en-US")} grains as ${SPHERE_TRIS[lod]}-triangle spheres` };
}

// The triangle budget chooses an authored mesh detail level; it never replaces
// a selected object with points because the instanced triangle count is high.
export function designedMeshPlan({ ready = false, designedMesh = null, count = 0 } = {}) {
  if (!ready) return { ok: false, reason: "the scene's instanced Phong shader is unavailable" };
  if (!designedMesh || !designedMesh.count) return { ok: false, reason: "select a saved marble or die first" };
  let lod = "full", mesh = designedMesh;
  for (const detail of ["medium", "low"]) {
    if (mesh.count / 3 * count <= MATERIAL_BUDGET.maxTriangles) break;
    if (designedMesh.lods?.[detail]) { lod = detail; mesh = designedMesh.lods[detail]; }
  }
  const triangles = mesh.count / 3 * count;
  return { ok: true, lod, note: `${count.toLocaleString("en-US")} modeled objects · ${lod} detail · ${Math.round(triangles).toLocaleString("en-US")} instanced triangles` };
}

// ---------------------------------------------------------------- emoji style (pure)
// Every grain an animal emoji.  Two looks, one assignment: which emoji a grain is depends only on its displayed slot (the
// packed point's index, stable from frame to frame), its true radius and the assignment the page picked, so the image variant
// (GLSL, per vertex) and the mesh variant (JS, one instanced batch per kind) show the SAME animal on the same grain.
// The kinds, their cells and colours are web_export.py's (emoji_atlas.json: {cell, cols, rows, ink, kinds: [{name, mesh,
// color}]}); this file holds no list of its own.  The packing carries no spin, so a mesh's orientation is a fixed hash of the slot.
export const EMOJI_ASSIGN = [["id", "by id (hashed mix)"], ["species", "by species"], ["size", "by size"]];
export const EMOJI_ASSIGN_IDS = EMOJI_ASSIGN.map(([id]) => id);
export const EMOJI_OPTIONS = { assign: "id", forceMesh: false, forceSprite: false };     // the page's setting (the force flags: measurement only, they ignore the budget / the too-small check)
export const MAX_EMOJI = 16;                      // the atlas holds at most this many kinds (the shader's uniform array)
export const EMOJI_MIN_PX = 2;                    // a sprite grain smaller than this (screen radius, px) is a speck: points
export const EMOJI_MESH_TRIS = [850, 250];        // triangles per instance of the full mesh and of its front-cap LOD (the export's mean; the draw counts the real ones)
export const EMOJI_MESH_FULL_AT_PX = 4;           // the full mesh from this screen radius up (a coarser grain shows only its front)
// the integer hash both sides use (Chris Wellons' lowbias32); the GLSL twin is EMOJI_HASH_GLSL
export function emojiHash(i) {
  let x = i >>> 0;
  x = (x ^ (x >>> 16)) >>> 0; x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0; x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}
// the species count a band table describes (unused edges are 1e9)
export const speciesCountOf = (bands) => 1 + Array.from(bands).filter((b) => b < 1e8).length;
// The kind (index into the atlas's kinds) of one grain.  assign "id": a stable hashed mix of the slot; "species": the grain's
// species (the nearest species radius, see speciesOf) spread evenly over the kinds; "size": the radius's place in the size
// ramp `range` = [low radius, 1 / span] spread evenly over the kinds.  GLSL twin: emojiKindOf.
export function emojiKind(assign, slot, radius, K, bands, range) {
  if (!(K > 0)) return 0;
  if (assign === "species") {
    let s = 0;
    for (const b of bands) if (radius > b) s++;
    return Math.min(K - 1, Math.floor((s + 0.5) * K / speciesCountOf(bands)));
  }
  if (assign === "size") return Math.min(K - 1, Math.max(0, Math.floor((radius - range[0]) * range[1] * K)));
  return emojiHash(slot) % K;
}
// the cell of kind `index` as [u0, v0, du, dv] (the texture's row 0 is the image's top, v grows downward like gl_PointCoord)
export function atlasRect(index, meta) {
  const c = index % meta.cols, r = Math.floor(index / meta.cols);
  return [c / meta.cols, r / meta.rows, 1 / meta.cols, 1 / meta.rows];
}
export function atlasRects(meta) {
  const out = new Float32Array(MAX_EMOJI * 4);
  for (let i = 0; i < Math.min(meta.kinds.length, MAX_EMOJI); i++) out.set(atlasRect(i, meta), i * 4);
  return out;
}
// the atlas meta as the page may use it, or the reason it may not
export function checkAtlasMeta(meta) {
  if (!meta || !Array.isArray(meta.kinds) || !meta.kinds.length) return "the emoji atlas lists no kinds";
  if (meta.kinds.length > MAX_EMOJI) return `the emoji atlas lists ${meta.kinds.length} kinds (the shader holds ${MAX_EMOJI})`;
  if (!(meta.cols > 0 && meta.rows > 0 && meta.cols * meta.rows >= meta.kinds.length)) return "the emoji atlas's grid does not hold its kinds";
  return "";
}
// 8-bit rgba words (little-endian rgba8, the packing's colour) of the kinds' native colours (display values 0..1)
export const packRgba8 = (rgb) => ((255 << 24) | (Math.round(rgb[2] * 255) << 16) | (Math.round(rgb[1] * 255) << 8) | Math.round(rgb[0] * 255)) >>> 0;
// "native": the emoji's own colours (no source or palette recolours the grains)
export const emojiNative = (tint, grain) => !(tint && tint[3] > 0.5) && !(grain && grain.mode > 0);
// Sprite variant: drawn unless the grains are specks on the screen (then points, which do accumulate sub-pixel coverage).
export function emojiSpritePlan({ radiusPx = 4, force = false } = {}) {
  if (!force && radiusPx < EMOJI_MIN_PX) return { ok: false, reason: `grains are ${radiusPx.toFixed(2)} px in radius at this zoom: too small to read as emoji` };
  return { ok: true, note: "" };
}
// Mesh variant: the scene shader instancing it needs (ready), and the material style's budget in instances and triangles per
// frame.  The full mesh from EMOJI_MESH_FULL_AT_PX up while it fits; else its front cap (a quarter of the triangles); else
// not drawn (the image variant is).  `tris` = [full, front cap] per instance (the loaded meshes' mean; else EMOJI_MESH_TRIS);
// `force`: ignore the triangle and instance budget (measurement only).
export function emojiMeshPlan({ ready = true, count = 0, radiusPx = 4, budget = MATERIAL_BUDGET, tris = EMOJI_MESH_TRIS, force = false } = {}) {
  if (!ready) return { ok: false, reason: "the scene's material shader is not in use here (flat view, or it could not be extended for instancing)" };
  if (radiusPx < budget.minRadiusPx) return { ok: false, reason: `grains are ${radiusPx.toFixed(2)} px in radius at this zoom: too small to mesh` };
  const fits = (lod) => force || (count <= budget.maxInstances && count * tris[lod] <= budget.maxTriangles);
  const lod = radiusPx >= EMOJI_MESH_FULL_AT_PX && fits(0) ? 0 : fits(1) ? 1 : -1;
  if (lod < 0) {
    return { ok: false, reason: count > budget.maxInstances ? `${count.toLocaleString("en-US")} grains is more than the ${budget.maxInstances.toLocaleString("en-US")} meshes hold a frame rate for here`
      : `${count.toLocaleString("en-US")} grains of about ${Math.round(tris[1])} triangles is more than the ${budget.maxTriangles.toLocaleString("en-US")} a frame holds here` };
  }
  return { ok: true, lod, tris: tris[lod], note: `${count.toLocaleString("en-US")} grains as ${Math.round(tris[lod])}-triangle ${lod ? "front-cap" : "extruded"} emoji` };
}
// GLSL twin of emojiHash and emojiKind (a vertex stage declares the uniforms below)
export const EMOJI_HASH_GLSL = `
uint emojiHash(uint x) { x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
`;
export const EMOJI_KIND_GLSL = EMOJI_HASH_GLSL + `
uniform vec4 uEmoji;           // x: assignment (0 id, 1 species, 2 size), y: the kinds' count, z: the size ramp's low radius, w: 1 / its span
uniform float uEmojiBands[${MAX_SPECIES - 1}];
int emojiKindOf(int slot, float radius) {
  int K = int(uEmoji.y);
  if (uEmoji.x > 1.5) return clamp(int(floor((radius - uEmoji.z) * uEmoji.w * float(K))), 0, K - 1);
  if (uEmoji.x > 0.5) {
    int s = 0, n = 1;
    for (int i = 0; i < ${MAX_SPECIES - 1}; i++) { if (radius > uEmojiBands[i]) s++; if (uEmojiBands[i] < 1e8) n++; }
    return min(K - 1, int(floor((float(s) + 0.5) * float(K) / float(n))));
  }
  return int(emojiHash(uint(slot)) % uint(K));
}
`;

// An icosphere of unit radius: subdivision 0 / 1 / 2 gives 12 / 42 / 162 vertices and 20 / 80 / 320 triangles.  Indexed,
// outward (counter-clockwise) winding; the vertex position doubles as the smooth normal.
export function icosphere(level) {
  const t = (1 + Math.sqrt(5)) / 2;
  const v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    .map((p) => { const l = Math.hypot(...p); return p.map((c) => c / l); });
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
           [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let k = 0; k < level; k++) {
    const mid = new Map(), next = [];
    const half = (a, b) => {
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!mid.has(key)) { const p = [0, 1, 2].map((i) => v[a][i] + v[b][i]), l = Math.hypot(...p); v.push(p.map((c) => c / l)); mid.set(key, v.length - 1); }
      return mid.get(key);
    };
    for (const [a, b, c] of f) { const ab = half(a, b), bc = half(b, c), ca = half(c, a); next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    f = next;
  }
  return { positions: new Float32Array(v.flat()), indices: new Uint16Array(f.flat()), triangles: f.length };
}

// ---------------------------------------------------------------- the scene's material shader, extended for instancing (pure)
// The scene's Phong program is exported from Pluck's base_material shaders (the generated source the page fetches).  Its
// vertex stage takes one model matrix per draw, so instancing it is one addition to that source, made here until the export's
// own source (spectral-analyzer csrc/shaders/base_material.vert.glsl) carries it: the original `main` is kept as `main_scene`
// and runs first, unchanged; when `uInstanced` is set (only the material particle style sets it) a thin wrapper then places
// the vertex at an instance's centre and radius and hands the fragment stage the grain's colour.  The fragment stage gets one
// line: a grain that carries a colour overrides the material's albedo; nothing else of the material (roughness, metal, ior,
// emission, enamel, lights, shadows) is touched.  Pieces never set the flag, so they run the original code path.
//
// The instance attributes are read in ONE place in the GPU (instanceAlive / instanceRadius / instanceCenter below) and ONE
// place in JS (PACKING, with readPoint / pointCount / decodeMeta): a z column added to the packing is a change to PACKING and
// to instanceCenter.
export const PACKING = {                         // the packed point: [x, y, z, rgba8, meta], 20 bytes (compute.js PointCloud's layout)
  stride: 20,
  position: { size: 3, offset: 0 },              // floats: x, y on the board's plane, z the slab's depth toward the camera (0 for a 2-D product)
  color: { offset: 12 },                         // 4 normalised bytes
  meta: { offset: 16 },                          // float: 100 (moving) / 200 (at rest) + radius * 1000; below 50 = not alive
};
export const INSTANCE_LOCATIONS = { position: 7, color: 8, meta: 9, id: 10, quaternion: 11 };
// A partitioned batch (the emoji meshes: one batch per kind) carries the grain's displayed slot as a sixth word, so its fixed
// rotation does not depend on where the grain sits in the batch.
export const BATCH_PACKING = { stride: 24, id: { offset: 20 } };
// JS twin of the metadata decode: meta -> { alive, moving, radius }
export function decodeMeta(meta) {
  if (!(meta >= 50)) return { alive: false, moving: false, radius: 0 };
  const flag = Math.floor(meta / 100 + 0.001);
  return { alive: true, moving: flag < 1.5, radius: (meta - flag * 100) * 0.001 };
}
// One packed point as an object (every JS read of the packing goes through here, and through PACKING for the GL layout).
export function readPoint(data, i) {
  const k = i * (PACKING.stride / 4);
  return { x: data[k], y: data[k + 1], z: data[k + 2], ...decodeMeta(data[k + 4]) };
}
export function pointCount(data) { return Math.floor(data.length / (PACKING.stride / 4)); }
// the largest true radius among the first live points (board units)
export function typicalRadius(data, sample = 4096) {
  let r = 0;
  for (let i = 0, k = Math.min(pointCount(data), sample); i < k; i++) r = Math.max(r, readPoint(data, i).radius);
  return r || 0.02;
}

const INSTANCE_VERT_DECLS = GRAIN_COLOR_GLSL + `
// ---- instancing for the material particle style (particle_styles.js)
layout(location = ${INSTANCE_LOCATIONS.position}) in vec3 aInstPos;
layout(location = ${INSTANCE_LOCATIONS.color}) in vec4 aInstColor;
layout(location = ${INSTANCE_LOCATIONS.meta}) in float aInstMeta;
layout(location = ${INSTANCE_LOCATIONS.id}) in float aInstId;     // the displayed slot (emoji batches only; the instance's own rotation key)
layout(location = ${INSTANCE_LOCATIONS.quaternion}) in vec4 aInstQuat; // physical orientation in the law's xyz frame
uniform bool uInstanced;
uniform float uInstRot;     // 1: emoji batch slot; 2: direct instanced mesh slot
uniform bool uInstSpin;     // emoji batch supplies the law's integrated orientation
uniform vec4 uInstMesh;     // the mesh's centre (xyz) and 1 / its larger half-width in the glyph's plane
uint instHash(uint x) { x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
mat3 instanceRotation() {   // the glyph lies in its xy plane facing +z: stand it on the board facing up, +y toward the board's top, then yaw it round and tilt it a little, all fixed by the slot
  uint h = instHash((uInstRot > 1.5 ? uint(gl_InstanceID) : uint(aInstId)) ^ 0x9e3779b9u);
  float yaw = float(h & 0xffffu) * (6.2831853 / 65536.0);
  float a = (float((h >> 16) & 0xffu) / 255.0 - 0.5) * 0.9, b = (float((h >> 24) & 0xffu) / 255.0 - 0.5) * 0.9;
  float ca = cos(a), sa = sin(a), cb = cos(b), sb = sin(b), cy = cos(yaw), sy = sin(yaw);
  mat3 base = mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), vec3(0.0, 1.0, 0.0));
  mat3 tx = mat3(1.0, 0.0, 0.0, 0.0, ca, sa, 0.0, -sa, ca), tz = mat3(cb, sb, 0.0, -sb, cb, 0.0, 0.0, 0.0, 1.0), ry = mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
  return ry * tx * tz * base;
}
vec3 physicalSpin(vec3 v) {
  // Board/world basis is (law x, law z, -law y). Rotate in law coordinates,
  // then return to world coordinates so a law-axis roll tumbles the mesh.
  vec3 law = vec3(v.x, -v.z, v.y);
  vec3 t = 2.0 * cross(aInstQuat.xyz, law);
  law += aInstQuat.w * t + cross(aInstQuat.xyz, t);
  return vec3(law.x, law.z, -law.y);
}
uniform vec4 uInstMap;      // board window centre x, y and the board -> world scale k
uniform vec3 uInstBoard;    // board width, height, and the plane's height
uniform float uInstOwn;     // 1: the material's own colour (colour source "material"); else the grain's colour from grainColor()
flat out float vInstUse;
flat out vec3 vInstColor;
bool instanceAlive() { return aInstMeta >= 50.0; }
float instanceRadius() { float flag = floor(aInstMeta / 100.0 + 0.001); return (aInstMeta - flag * 100.0) * 0.001 * uInstMap.z; }
vec3 instanceCenter() {     // the grain's centre in world space (the packed x, y on the board's plane, resting on it)
  return vec3(uInstBoard.x * 0.5 + (aInstPos.x - uInstMap.x) * uInstMap.z, uInstBoard.z + instanceRadius() + aInstPos.z * uInstMap.z, uInstBoard.y * 0.5 - (aInstPos.y - uInstMap.y) * uInstMap.z);
}
`;
const INSTANCE_VERT_MAIN = `
void main() {
  main_scene();
  vInstUse = 0.0; vInstColor = vec3(0.0);
  if (!uInstanced) return;
  if (!instanceAlive()) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec3 lp = aPos, nrm = aNorm;
  if (uInstRot > 0.5) { mat3 R = instanceRotation(); lp = R * ((aPos - uInstMesh.xyz) * uInstMesh.w); nrm = R * aNorm; }
  if (uInstSpin) { lp = physicalSpin(lp); nrm = physicalSpin(nrm); }
  vec3 world = instanceCenter() + lp * instanceRadius();
  vec4 posV = uMV * vec4(world, 1.0);
  vPosV = posV.xyz; vNormV = mat3(uMV) * nrm; vPosObj = world;
  gl_Position = uMVP * vec4(world, 1.0);
  vInstUse = uInstOwn > 0.5 ? 0.0 : 1.0;
  vInstColor = pow(grainColor(aInstColor, aInstMeta).rgb, vec3(2.2));             // display colour -> the shader's linear
}
`;
const INSTANCE_FRAG_DECLS = `
flat in float vInstUse;
flat in vec3 vInstColor;
`;
// -> { vert, frag } extended, or throws an Error naming what the source did not have where this expects it
export function extendForInstancing(vert, frag) {
  const mainRe = /void\s+main\s*\(\s*\)\s*\{/;
  const mainV = mainRe.exec(vert), mainF = mainRe.exec(frag);
  if (!mainV) throw new Error("the vertex shader has no main()");
  if (!mainF) throw new Error("the fragment shader has no main()");
  const albedo = /vec3\s+albedo\s*=\s*mat_albedo\s*\(\s*id\s*\)\s*;/.exec(frag);
  if (!albedo || albedo.index < mainF.index) throw new Error("the fragment shader's albedo line was not found in main()");
  const inner = /vec3\s+inner\s*=\s*ph_inner\s*\(\s*id\s*\)\s*;/.exec(frag);     // the colour a back face shows (stained glass draws them)
  if (!inner || inner.index < albedo.index) throw new Error("the fragment shader's inner-colour line was not found after the albedo line");
  for (const name of ["aPos", "aNorm", "uMV", "uMVP", "vPosV", "vNormV", "vPosObj"]) if (!new RegExp(`\\b${name}\\b`).test(vert)) throw new Error(`the vertex shader has no ${name}`);
  const v = vert.slice(0, mainV.index) + INSTANCE_VERT_DECLS + "void main_scene() {" + vert.slice(mainV.index + mainV[0].length) + INSTANCE_VERT_MAIN;
  const cut = albedo.index + albedo[0].length, cut2 = inner.index + inner[0].length;
  const f = frag.slice(0, mainF.index) + INSTANCE_FRAG_DECLS + frag.slice(mainF.index, cut) + "\n    if (vInstUse > 0.5) albedo = vInstColor;"
    + frag.slice(cut, cut2) + "\n    if (vInstUse > 0.5) inner = vInstColor;" + frag.slice(cut2);
  return { vert: v, frag: f };
}

export function capabilitiesOf(gl) {
  const full = !!gl.getExtension("EXT_color_buffer_float"), half = !!gl.getExtension("EXT_color_buffer_half_float");
  const pointRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
  return { floatRender: full ? "float" : half ? "half" : null, maxPointSize: pointRange?.[1] ?? 1 };
}

// ---------------------------------------------------------------- shaders
// One sprite vertex stage for every style: the same board -> world -> clip placement as compute.js's POINT_VS, over the same
// vertex layout (location 0 position, 1 colour, 2 meta), with the sprite sized in TARGET pixels (uFocal is already scaled).
const SPRITE_VS = `#version 300 es
${GRAIN_COLOR_GLSL}
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform mat4 uViewProj; uniform mat4 uView; uniform vec4 uMap; uniform vec3 uBoard;
uniform float uFocal; uniform float uMax; uniform float uStride;
uniform float uGrow;      // the sprite's radius as a multiple of the true radius
uniform float uMinPx;     // the least sprite diameter, target pixels
uniform float uPad;       // extra pixels around it
uniform float uFull;      // 1: the sprite's whole square is the shape (glow)
out vec4 vColor; out float vRho; out float vRw; out float vEye; out float vMoving; out float vArea; out float vREff; out float vRpx;
void main() {
  vColor = grainColor(aColor, aMeta); vRho = 1.0; vRw = 0.0; vEye = 0.0; vMoving = 0.0; vArea = 0.0; vREff = 1.0; vRpx = 0.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
  if (aMeta < 50.0) return;
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  vec3 pos = vec3(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.z + aPos.z * uMap.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z);   // z: the slab's depth, toward the camera
  vec4 clip = uViewProj * vec4(pos, 1.0);
  if (clip.w <= 0.0) return;
  float eye = -(uView * vec4(pos, 1.0)).z;
  float rpx = radius * uMap.z * uFocal / clip.w;                      // the true radius, target pixels
  float rEff = max(rpx * uGrow, 0.5 * uMinPx);
  float size = clamp(2.0 * rEff + uPad, 1.0, uMax);
  gl_Position = clip; gl_PointSize = size;
  vRho = uFull > 0.5 ? 1.0 : clamp(rEff / (0.5 * size), 0.01, 1.0);
  vREff = min(rEff, 0.5 * size);
  vRw = vREff * clip.w / uFocal;                                       // the sprite's radius in world units
  vEye = eye; vMoving = flag < 1.5 ? 1.0 : 0.0; vRpx = rpx;
  vArea = 3.14159265 * rpx * rpx * uStride;                            // the share of a pixel's area the true disc covers, over the stride
}`;
const FS_HEAD = `#version 300 es
precision highp float; precision highp int;
in vec4 vColor; in float vRho; in float vRw; in float vEye; in float vMoving; in float vArea; in float vREff; in float vRpx;
uniform vec4 uProjZ;      // proj[10], proj[14], proj[11], proj[15]
vec2 sprite() { vec2 q = (gl_PointCoord - 0.5) * 2.0; q.y = -q.y; return q / vRho; }   // in sphere radii, y up
float eyeToDepth(float e) { float w = uProjZ.z * (-e) + uProjZ.w; return ((uProjZ.x * (-e) + uProjZ.y) / w) * 0.5 + 0.5; }
`;
const DEPTH_FS = FS_HEAD + `
out vec4 FragColor;
void main() {
  vec2 q = sprite(); float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float e = vEye - sqrt(1.0 - r2) * vRw;
  FragColor = vec4(e, 0.0, 0.0, 1.0);
  gl_FragDepth = eyeToDepth(e);
}`;
const THICK_FS = FS_HEAD + `
uniform float uThickGain;
out vec4 FragColor;
void main() {
  vec2 q = sprite(); float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float t = 2.0 * sqrt(1.0 - r2) * vRw * min(1.0, vArea / (3.14159265 * vREff * vREff) + 0.35) * uThickGain;
  FragColor = vec4(vColor.rgb * t, t);
}`;
// the OIT composite: the accumulated (rgb * w, a * w) and the optical depth tau (sum of -ln(1 - a)) -> the averaged colour and 1 - exp(-tau)
const OIT_COMPOSITE_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uAccum;
uniform sampler2D uTau;
uniform ivec2 uOrigin;
out vec4 FragColor;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) - uOrigin;
  vec4 a = texelFetch(uAccum, p, 0);
  float tau = texelFetch(uTau, p, 0).r;
  if (tau <= 0.0) discard;
  FragColor = vec4(a.rgb / max(a.a, 1e-5), 1.0 - exp(-tau));
}`;
export const QUAD_VS = `#version 300 es
void main() { vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)); gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0); }`;
const BLUR_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uD; uniform ivec2 uDir; uniform float uStep; uniform float uRange;
out vec4 FragColor;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); ivec2 sz = textureSize(uD, 0);
  float c = texelFetch(uD, p, 0).r;
  if (c <= 0.0) { FragColor = vec4(0.0); return; }
  float sum = c, wsum = 1.0;
  for (int i = 1; i <= 7; i++) {
    float ws = exp(-float(i * i) / 18.0);
    for (int s = -1; s <= 1; s += 2) {
      ivec2 q = p + uDir * s * int(floor(float(i) * uStep + 0.5));
      if (q.x < 0 || q.y < 0 || q.x >= sz.x || q.y >= sz.y) continue;
      float d = texelFetch(uD, q, 0).r;
      if (d <= 0.0) continue;
      float dd = (d - c) / uRange;
      float w = ws * exp(-0.5 * dd * dd);                          // narrow range: far-off depths barely count
      sum += d * w; wsum += w;
    }
  }
  FragColor = vec4(sum / wsum, 0.0, 0.0, 1.0);
}`;
const LIQUID_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uD; uniform sampler2D uT;
uniform vec2 uProj0; uniform vec2 uProj5;   // proj[0], proj[8] ; proj[5], proj[9]
uniform vec4 uProjZ; uniform float uTexScale; uniform vec2 uTexSize;
uniform vec3 uL; uniform vec3 uKey; uniform vec3 uFill; uniform vec3 uBg; uniform float uAbsorb; uniform float uLod;
out vec4 FragColor;
vec3 eyePos(ivec2 p, float depth) {
  vec2 ndc = (vec2(p) + 0.5) / uTexSize * 2.0 - 1.0;
  float ze = -depth, w = uProjZ.z * ze + uProjZ.w;
  return vec3((ndc.x * w - uProj0.y * ze) / uProj0.x, (ndc.y * w - uProj5.y * ze) / uProj5.x, ze);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy * uTexScale);
  float d0 = texelFetch(uD, p, 0).r;
  vec4 th = textureLod(uT, gl_FragCoord.xy * uTexScale / uTexSize, uLod);   // the mip chain smooths the thickness and its colour
  if (d0 <= 0.0 || th.a <= 1e-4) discard;
  vec3 P = eyePos(p, d0);
  float dr = texelFetch(uD, p + ivec2(1, 0), 0).r, dl = texelFetch(uD, p - ivec2(1, 0), 0).r;
  float du = texelFetch(uD, p + ivec2(0, 1), 0).r, dd = texelFetch(uD, p - ivec2(0, 1), 0).r;
  vec3 ex = vec3(0.0), ey = vec3(0.0);
  bool useR = dr > 0.0 && (dl <= 0.0 || abs(dr - d0) < abs(d0 - dl));
  if (useR) ex = eyePos(p + ivec2(1, 0), dr) - P; else if (dl > 0.0) ex = P - eyePos(p - ivec2(1, 0), dl);
  bool useU = du > 0.0 && (dd <= 0.0 || abs(du - d0) < abs(d0 - dd));
  if (useU) ey = eyePos(p + ivec2(0, 1), du) - P; else if (dd > 0.0) ey = P - eyePos(p - ivec2(0, 1), dd);
  vec3 n = cross(ex, ey);
  n = dot(n, n) > 1e-18 ? normalize(n) : vec3(0.0, 0.0, 1.0);
  if (n.z < 0.0) n = -n;
  vec3 V = normalize(-P);
  float cosv = max(dot(n, V), 0.0);
  float fres = 0.03 + 0.97 * pow(1.0 - cosv, 5.0);
  float thick = th.a * uAbsorb;
  vec3 tint = th.rgb / th.a;                                           // the thickness-weighted grain colour
  tint = mix(tint, tint / max(max(tint.r, tint.g), max(tint.b, 1e-3)), 0.75);   // averaging random hues greys them: bring the colour back
  vec3 absorb = exp(-thick * (vec3(1.0) - tint) * 1.6 - thick * 0.12);   // Beer-Lambert through the body
  vec3 refr = uBg * absorb;                                            // the page background seen through it
  float body = 1.0 - exp(-thick * 1.2);
  vec3 scatter = tint * (uFill * 0.9 + uKey * max(dot(n, uL), 0.0) * 0.9) * body;
  vec3 H = normalize(uL + V);
  vec3 spec = uKey * pow(max(dot(n, H), 0.0), 120.0) * 1.4;
  vec3 refl = mix(uFill, uKey, 0.3) * (0.55 + 0.45 * n.y);             // the sky the surface mirrors
  vec3 col = mix(refr + scatter, refl, fres) + spec;
  float alpha = clamp(max(1.0 - exp(-thick * 3.0), fres) * smoothstep(0.0, 0.08, th.a * uAbsorb), 0.0, 1.0);
  FragColor = vec4(col, alpha);
  float ze = -d0;
  gl_FragDepth = ((uProjZ.x * ze + uProjZ.y) / (uProjZ.z * ze + uProjZ.w)) * 0.5 + 0.5;
}`;
const SAND_FS = FS_HEAD + `
uniform sampler2D uOcc; uniform vec2 uViewport; uniform vec3 uL; uniform vec3 uKey; uniform vec3 uFill; uniform float uAO; uniform vec3 uBg;
out vec4 FragColor;
void main() {
  vec2 q = sprite(); float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float nz = sqrt(1.0 - r2);
  vec3 n = vec3(q, nz);
  float occ = uAO > 0.0 ? texture(uOcc, gl_FragCoord.xy / uViewport).r : 0.0;
  float ao = 1.0 - uAO * smoothstep(0.0, 1.0, occ);
  float diff = max(dot(n, uL), 0.0);
  vec3 H = normalize(uL + vec3(0.0, 0.0, 1.0));
  vec3 col = vColor.rgb * (uFill * (0.55 + 0.45 * n.y) * ao + uKey * diff * (0.45 + 0.55 * ao)) * (0.75 + 0.25 * nz);
  col = col * 1.7 + uKey * pow(max(dot(n, H), 0.0), 28.0) * 0.12 * ao;      // exposure: the scene's light is dim and the grains are small
  float rim = clamp((1.0 - sqrt(r2)) * vREff + 0.5, 0.0, 1.0);          // antialiased edge, pixels
  float alpha = vREff > 1.0 ? rim : min(1.0, vArea) * (1.0 - 0.5 * r2);
  alpha = vRpx >= 0.5 ? rim : alpha;
  col = mix(uBg, col, alpha);
  FragColor = vec4(col, alpha);
  gl_FragDepth = vRpx >= 0.5 ? eyeToDepth(vEye - nz * vRw) : gl_FragCoord.z;
}`;
const DENSITY_FS = FS_HEAD + `
out vec4 FragColor;
void main() {
  vec2 q = sprite(); float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float w = exp(-3.0 * r2) * vArea * 3.0 / (3.14159265 * vREff * vREff);   // the local area fraction the grains cover
  FragColor = vec4(w, w * vMoving, 0.0, 0.0);
}`;
const GLOW_FS = FS_HEAD + `
uniform float uGain;
out vec4 FragColor;
void main() {
  vec2 q = sprite(); float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  float g = exp(-3.5 * r2) * (1.0 - r2);
  float k = uGain * min(1.0, 0.25 + vArea);
  FragColor = vec4(vColor.rgb * g * 0.16 * k + mix(vColor.rgb, vec3(1.0), 0.5) * pow(g, 6.0) * 0.3 * k, 1.0);   // a soft halo and a hot core
}`;
// ---- emoji, image variant: a point sprite per grain, sized to the grain's true radius (the glyph's ink fills `ink` of its cell),
// showing its cell of the atlas; alpha-tested and depth-written per sprite (the sprite's centre, z included: the slab's depth
// is part of the position), so it sorts against the scene and against other grains exactly like the sand impostors
const EMOJI_VS = `#version 300 es
${GRAIN_COLOR_GLSL}${EMOJI_KIND_GLSL}
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform mat4 uViewProj; uniform mat4 uView; uniform vec4 uMap; uniform vec3 uBoard;
uniform float uFocal; uniform float uMax;
uniform float uMinPx;       // the least sprite diameter, target pixels
uniform float uInk;         // the glyph's longer side as a share of its cell
uniform vec4 uEmojiRect[${MAX_EMOJI}];
out vec4 vTint; flat out vec4 vCell; out float vEye;
void main() {
  vTint = grainColor(aColor, aMeta); vCell = vec4(0.0); vEye = 0.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
  if (aMeta < 50.0) return;
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  vec3 pos = vec3(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.z + aPos.z * uMap.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z);
  vec4 clip = uViewProj * vec4(pos, 1.0);
  if (clip.w <= 0.0) return;
  float rpx = radius * uMap.z * uFocal / clip.w;                      // the true radius, target pixels
  gl_Position = clip;
  gl_PointSize = clamp(max(2.0 * rpx / uInk, uMinPx), 1.0, uMax);
  vEye = -(uView * vec4(pos, 1.0)).z;
  vCell = uEmojiRect[emojiKindOf(gl_VertexID, radius)];
}`;
const EMOJI_FS = `#version 300 es
precision highp float; precision highp int;
in vec4 vTint; flat in vec4 vCell; in float vEye;
uniform vec4 uProjZ;        // proj[10], proj[14], proj[11], proj[15]
uniform sampler2D uAtlas;
uniform float uNative;      // 1: the emoji's own colours; else the colour source tints them (keeping their shading)
out vec4 FragColor;
float eyeToDepth(float e) { float w = uProjZ.z * (-e) + uProjZ.w; return ((uProjZ.x * (-e) + uProjZ.y) / w) * 0.5 + 0.5; }
void main() {
  vec4 t = texture(uAtlas, vCell.xy + gl_PointCoord * vCell.zw);
  if (t.a < 0.5) discard;
  vec3 c = t.rgb;
  if (uNative < 0.5) c = clamp(vTint.rgb * (0.3 + 0.95 * dot(t.rgb, vec3(0.299, 0.587, 0.114))), 0.0, 1.0);
  FragColor = vec4(c, 1.0);
  gl_FragDepth = eyeToDepth(vEye);
}`;
// a quad over the board's plane (the board's rectangle, plus a margin), like compute.js's RESOLVE_VS
export const PLANE_VS = `#version 300 es
uniform mat4 uViewProj; uniform vec3 uBoard; uniform float uMargin; out vec2 vUV;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  vec2 u = c * (1.0 + 2.0 * uMargin) - uMargin;
  vUV = u;
  gl_Position = uViewProj * vec4(uBoard.x * u.x, uBoard.z, uBoard.y * u.y, 1.0);
}`;
const GLOW_RESOLVE_FS = `#version 300 es
precision highp float;
uniform sampler2D uDens; uniform vec2 uViewport; out vec4 FragColor;
void main() {
  vec3 x = texture(uDens, gl_FragCoord.xy / uViewport).rgb;
  if (max(x.r, max(x.g, x.b)) < 0.004) discard;
  FragColor = vec4(1.0 - exp(-x), 1.0);                                  // tone-mapped: dense places go to white smoothly, never clip
}`;
const HIST_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uDens; uniform vec2 uSize; uniform float uLod; out vec4 FragColor;
${HEAT_GLSL}void main() {
  vec4 a = textureLod(uDens, gl_FragCoord.xy / uSize, uLod);
  if (a.r < 0.01) { FragColor = vec4(0.0); return; }                   // empty board does not count
  FragColor = vec4(clamp(heatX(heatQuantity(a)) / ${HEAT_X_MAX.toFixed(6)}, 0.0, 1.0), 0.0, 0.0, 1.0);
}`;
const HEAT_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uDens; uniform vec2 uViewport; uniform vec4 uTint; uniform vec2 uRange; uniform float uGamma; out vec4 FragColor;
${HEAT_GLSL}vec3 ramp(float t) {
  if (t < 0.25) return mix(vec3(0.03, 0.0, 0.18), vec3(0.38, 0.0, 0.48), t / 0.25);
  if (t < 0.5) return mix(vec3(0.38, 0.0, 0.48), vec3(0.86, 0.14, 0.14), (t - 0.25) / 0.25);
  if (t < 0.75) return mix(vec3(0.86, 0.14, 0.14), vec3(1.0, 0.72, 0.08), (t - 0.5) / 0.25);
  return mix(vec3(1.0, 0.72, 0.08), vec3(1.0, 1.0, 0.86), (t - 0.75) / 0.25);
}
void main() {
  vec4 a = texture(uDens, gl_FragCoord.xy / uViewport);
  float d = a.r;
  if (d < 0.01) discard;
  float t = pow(clamp((heatX(heatQuantity(a)) - uRange.x) / max(uRange.y - uRange.x, 1e-4), 0.0, 1.0), uGamma);
  // a chosen colour (solid / material) replaces the ramp's hues, keeping its dark-to-hot ordering
  vec3 c = uTint.w > 0.5 ? mix(uTint.rgb * 0.12, uTint.rgb, smoothstep(0.0, 0.7, t)) + smoothstep(0.75, 1.0, t) * 0.5 : ramp(t);
  FragColor = vec4(c, smoothstep(0.01, 0.3, d) * 0.94);
}`;
// the memory: stamps are drawn in the board's own coordinates (the texture covers the board), so the camera may move
const STAMP_VS = `#version 300 es
${GRAIN_COLOR_GLSL}
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform vec4 uMap; uniform vec3 uBoard; uniform float uTexel; uniform float uMax; uniform float uGrow;
out vec4 vColor;
void main() {
  vColor = grainColor(aColor, aMeta); gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
  if (aMeta < 50.0) return;
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  vec2 b = vec2(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z);
  gl_Position = vec4(b / uBoard.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = clamp(max(2.0 * radius * uMap.z * uTexel * uGrow, 2.5), 1.0, uMax);
}`;
const STAMP_FS = `#version 300 es
precision highp float;
in vec4 vColor; out vec4 FragColor;
void main() { float r = length(gl_PointCoord - 0.5) * 2.0; if (r > 1.0) discard; FragColor = vec4(vColor.rgb, smoothstep(1.0, 0.35, r)); }`;
const FADE_FS = `#version 300 es
precision highp float; uniform float uFade; out vec4 FragColor;
void main() { FragColor = vec4(0.0, 0.0, 0.0, uFade); }`;
const MEMORY_FS = `#version 300 es
precision highp float; uniform sampler2D uMem; uniform float uOpacity; in vec2 vUV; out vec4 FragColor;
void main() {
  if (vUV.x < 0.0 || vUV.y < 0.0 || vUV.x > 1.0 || vUV.y > 1.0) discard;
  vec4 m = texture(uMem, vUV);
  if (m.a <= 0.002) discard;
  FragColor = vec4(m.rgb, m.a * uOpacity);
}`;

const NO_TINT = new Float32Array(4);
class Unavailable extends Error {}

// Save a texture unit's 2D binding before using it, put it back after: the scene's passes share the units (a float texture
// left on unit 0 made the pbr material table read garbage and the sun and the bulbs rendered black).
export class UnitGuard {
  constructor(gl) { this.gl = gl; this.active = gl.getParameter(gl.ACTIVE_TEXTURE); this.saved = new Map(); }
  bind(unit, tex) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    if (!this.saved.has(unit)) this.saved.set(unit, gl.getParameter(gl.TEXTURE_BINDING_2D));
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }
  restore() {
    const gl = this.gl;
    for (const [unit, tex] of this.saved) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); }
    gl.activeTexture(this.active);
  }
}

export class ParticleStyles {
  constructor(gl, link, compile) {
    this.gl = gl; this.link = link; this.compile = compile;
    this.caps = capabilitiesOf(gl);
    this.programs = new Map(); this.targets = new Map();
    this.broken = {};                    // id -> why it failed here (never retried)
    this.emptyVao = gl.createVertexArray();
    this.fallback = "";                  // why the last request was drawn as points ("" when it was not)
    this.lastMs = 0; this.stamp = null; this.stampAt = 0; this.fadeCarry = 0;
  }

  // Draw `style`; true when the frame is complete, false when PointCloud should go on to draw its own points (the points
  // style, a style this device cannot run, and the stamp style, whose memory lies under the current points).
  draw(style, ctx) {
    const gl = this.gl;
    const emojiStyle = style === "emoji" || style === "emoji_mesh";
    if (emojiStyle) this.#emojiLoad();
    const choice = chooseStyle(style, this.caps, this.broken, (style === "material" || style === "stained" || style === "stained_fast" || style === "designed" || emojiStyle) ? this.#sceneOf(ctx) : null);
    this.fallback = choice.style === style ? "" : choice.reason;
    this.note = choice.reason;
    this.heatInfo = choice.style === "heat" ? this.heatInfo : "";
    this.glassInfo = "";               // how glass grains were composited in the last draw (set by #material)
    if (choice.style === DEFAULT_STYLE || !ctx.camera || !ctx.camera.proj || !ctx.camera.view) return false;
    if (choice.style === "emoji" || choice.style === "emoji_mesh") {      // the atlas is a page asset fetched once: until it is here the grains are points
      const wait = this.#emojiWait(choice.style);
      if (wait) { this.fallback = this.note = `${style} unavailable (${wait}): showing points`; return false; }
    }
    const guard = new UnitGuard(gl);
    const program = gl.getParameter(gl.CURRENT_PROGRAM);
    const keep = { cull: gl.isEnabled(gl.CULL_FACE), scissor: gl.isEnabled(gl.SCISSOR_TEST) };
    gl.disable(gl.CULL_FACE); gl.disable(gl.SCISSOR_TEST);
    let done = true, retry = false;
    try {
      gl.bindVertexArray(ctx.vao);
      if (choice.style === "liquid") this.#liquid(ctx, guard);
      else if (choice.style === "sand") this.#sand(ctx, guard);
      else if (choice.style === "glow") this.#glow(ctx, guard);
      else if (choice.style === "heat") this.#heat(ctx, guard);
      else if (choice.style === "stamp") { this.#stamp(ctx, guard); done = false; }
      else if (choice.style === "material" || choice.style === "stained" || choice.style === "stained_fast") this.#material(ctx, choice, choice.style);
      else if (choice.style === "emoji") this.#emoji(ctx, guard);
      else if (choice.style === "emoji_mesh") this.#emojiMesh(ctx, choice);
      else if (choice.style === "designed") this.#designedMesh(ctx, choice);
    } catch (error) {
      if (!(error instanceof Unavailable)) throw error;
      this.broken[choice.style] = error.message; this.fallback = `${choice.style} unavailable (${error.message}): showing points`;
      done = false;
      retry = choice.style === "emoji_mesh";           // (the image variant is its fallback: draw it this frame)
    } finally {
      guard.restore();
      gl.useProgram(program);
      gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
      gl.blendEquation(gl.FUNC_ADD); gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true);
      if (keep.cull) gl.enable(gl.CULL_FACE);
      if (keep.scissor) gl.enable(gl.SCISSOR_TEST);
      gl.bindVertexArray(done ? null : ctx.vao);
    }
    return retry ? this.draw(style, ctx) : done;
  }

  // ---- plumbing
  #program(name, vs, fs) {
    let p = this.programs.get(name);
    if (p) return p;
    const gl = this.gl;
    const program = this.link(this.compile(gl.VERTEX_SHADER, vs), this.compile(gl.FRAGMENT_SHADER, fs));
    const u = {};
    for (const m of (vs + fs).matchAll(/uniform\s+(?:highp\s+|mediump\s+)?\w+\s+(\w+)/g)) u[m[1]] = gl.getUniformLocation(program, m[1]);
    p = { program, u }; this.programs.set(name, p);
    return p;
  }
  #target(key, w, h, { fmt = "rgba16f", depth = false, linear = false, mips = false } = {}) {
    const gl = this.gl;
    let t = this.targets.get(key);
    if (t && t.w === w && t.h === h && t.fmt === fmt) return t;
    if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); if (t.rb) gl.deleteRenderbuffer(t.rb); this.targets.delete(key); }
    const unit = gl.getParameter(gl.ACTIVE_TEXTURE), held = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, mips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1, { rgba16f: gl.RGBA16F, rgba32f: gl.RGBA32F, rgba8: gl.RGBA8 }[fmt], w, h);
    const filter = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips ? gl.LINEAR_MIPMAP_LINEAR : filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    let rb = null;
    if (depth) {
      rb = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
    }
    gl.bindTexture(gl.TEXTURE_2D, held); gl.activeTexture(unit);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteTexture(tex); gl.deleteFramebuffer(fbo); if (rb) gl.deleteRenderbuffer(rb);
      throw new Unavailable(`a ${fmt} render target is incomplete here (0x${status.toString(16)})`);
    }
    t = { w, h, fmt, tex, fbo, rb };
    this.targets.set(key, t);
    return t;
  }
  #depthFormat() { return this.caps.floatRender === "float" ? "rgba32f" : "rgba16f"; }
  // the uniforms every sprite pass shares; `scale` is the target's size over the screen's
  #sprite(p, ctx, scale, { grow = 1, minPx = 1, pad = 0, full = 0 } = {}) {
    const gl = this.gl, u = p.u, c = ctx.camera;
    gl.uniformMatrix4fv(u.uViewProj, false, ctx.viewProj); gl.uniformMatrix4fv(u.uView, false, c.view);
    gl.uniform4f(u.uMap, ctx.map[0], ctx.map[1], ctx.map[2], 0);
    gl.uniform3f(u.uBoard, ctx.board[0], ctx.board[1], ctx.y);
    gl.uniform1f(u.uFocal, ctx.focal * ctx.H / 2 * scale); gl.uniform1f(u.uMax, this.caps.maxPointSize); gl.uniform1f(u.uStride, ctx.stride || 1);
    grainUniforms(gl, (n) => u[n], ctx.tint, ctx.grain);
    gl.uniform1f(u.uGrow, grow); gl.uniform1f(u.uMinPx, minPx); gl.uniform1f(u.uPad, pad); gl.uniform1f(u.uFull, full);
    const P = c.proj;
    gl.uniform4f(u.uProjZ, P[10], P[14], P[11], P[15]);
  }
  #lighting(p, ctx) {
    const gl = this.gl, u = p.u, V = ctx.camera.view, l = ctx.lighting || {};
    const d = l.dir || [0.3, 0.8, 0.5], key = l.key || [0.8, 0.78, 0.72], fill = l.fill || [0.55, 0.6, 0.7];
    const lv = [0, 1, 2].map((i) => V[i] * d[0] + V[4 + i] * d[1] + V[8 + i] * d[2]);       // the light's direction in view space
    const len = Math.hypot(...lv) || 1;
    gl.uniform3f(u.uL, lv[0] / len, lv[1] / len, lv[2] / len);
    gl.uniform3f(u.uKey, key[0], key[1], key[2]); gl.uniform3f(u.uFill, fill[0], fill[1], fill[2]);
    const bg = ctx.background || [0.085, 0.085, 0.1];
    gl.uniform3f(u.uBg, bg[0], bg[1], bg[2]);
  }
  static typicalRadius(data) { return typicalRadius(data); }       // the largest true radius among the first live points: sizes the blur's range
  #quad(vao) { const gl = this.gl; gl.bindVertexArray(this.emptyVao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); gl.bindVertexArray(vao); }

  // ---- material: every grain a real instanced icosphere through the scene's own Phong program (see extendForInstancing)
  // ctx.scene = { program, instancing (the program was extended), materialId, shadow (the key light's view-projection) }
  // what materialPlan needs: the grain count and a grain's radius on the screen, from the largest live grain's true radius at the board's centre
  #sceneOf(ctx) {
    const c = ctx.camera, view = c && c.view;
    let radiusPx = 4;
    if (view) {
      const eye = -(view[2] * ctx.board[0] * 0.5 + view[6] * ctx.y + view[10] * ctx.board[1] * 0.5 + view[14]);
      radiusPx = typicalRadius(ctx.data) * ctx.map[2] * ctx.focal * ctx.H / 2 / sizeDepth(c.proj || PERSPECTIVE_W, -eye);       // [CURVE] clip.w: the eye depth, 1 under an orthographic projection
    }
    // The draw submits ctx.n instances, including parked law IDs, so detail
    // selection must use that submitted count rather than only live grains.
    const out = { ready: !!(ctx.scene && ctx.scene.instancing && ctx.scene.program), count: ctx.n, radiusPx,
      designedMesh: ctx.scene && ctx.scene.designedMesh };
    const tris = this.#emojiTris(ctx.scene);
    if (tris) out.tris = tris;
    return out;
  }
  // the emoji meshes' mean triangles per instance [full, front cap], from the meshes the page uploaded (null until atlas and meshes are here)
  #emojiTris(scene) {
    const E = this.emoji, meshes = scene && scene.meshes;
    if (!E || !E.meta || !meshes) return null;
    if (E.trisFor === meshes) return E.tris;
    const mean = (suffix) => {
      let sum = 0, k = 0;
      for (const kind of E.meta.kinds) { const m = kind.mesh && meshes[kind.mesh + suffix]; if (m) { sum += m.count / 3; k++; } }
      return k ? sum / k : 0;
    };
    const full = mean(""), lod = mean("_lod");
    E.trisFor = meshes; E.tris = full && lod ? [full, lod] : null;
    return E.tris;
  }
  #sphere(ctx, lod) {
    const gl = this.gl;
    if (this.sphereBuffer !== ctx.buffer) {                         // (a new point buffer: the instance attributes point at the new one)
      for (const m of this.spheres || []) if (m) { gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.vbo); gl.deleteBuffer(m.ibo); }
      this.spheres = []; this.sphereBuffer = ctx.buffer;
    }
    if (this.spheres[lod]) return this.spheres[lod];
    const mesh = icosphere(lod), L = INSTANCE_LOCATIONS;
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
    for (const at of [0, 1]) { gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at, 3, gl.FLOAT, false, 0, 0); }   // the unit sphere's position is its normal
    const ibo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, ctx.buffer);                      // the packed points, once per instance, in the layout PACKING describes
    gl.enableVertexAttribArray(L.position); gl.vertexAttribPointer(L.position, PACKING.position.size, gl.FLOAT, false, PACKING.stride, PACKING.position.offset); gl.vertexAttribDivisor(L.position, 1);
    gl.enableVertexAttribArray(L.color); gl.vertexAttribPointer(L.color, 4, gl.UNSIGNED_BYTE, true, PACKING.stride, PACKING.color.offset); gl.vertexAttribDivisor(L.color, 1);
    gl.enableVertexAttribArray(L.meta); gl.vertexAttribPointer(L.meta, 1, gl.FLOAT, false, PACKING.stride, PACKING.meta.offset); gl.vertexAttribDivisor(L.meta, 1);
    gl.bindVertexArray(null);
    return (this.spheres[lod] = { vao, vbo, ibo, count: mesh.indices.length });
  }
  // The OIT targets (accumulation, optical depth) with their own depth buffer (the scene's depth is blitted in per frame).
  #oitTarget(w, h, fmt) {         // (the depth format `fmt`: see OIT_DEPTH_FORMATS; a depth blit needs the scene framebuffer's)
    const gl = this.gl;
    let t = this.oit;
    if (t && t.w === w && t.h === h && t.fmt === fmt) return t;
    if (t) { for (const x of t.tex) gl.deleteTexture(x); gl.deleteFramebuffer(t.fbo); gl.deleteRenderbuffer(t.rb); this.oit = null; }
    const unit = gl.getParameter(gl.ACTIVE_TEXTURE), held = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const tex = [0, 1].map((i) => {
      const x = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, x); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, x, 0);
      return x;
    });
    const spec = OIT_DEPTH_FORMATS[fmt];
    const rb = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl[spec.internal], w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl[spec.attach], gl.RENDERBUFFER, rb);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindTexture(gl.TEXTURE_2D, held); gl.activeTexture(unit);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      for (const x of tex) gl.deleteTexture(x); gl.deleteFramebuffer(fbo); gl.deleteRenderbuffer(rb);
      throw new Error(`the OIT targets are incomplete here (0x${status.toString(16)})`);
    }
    return (this.oit = { w, h, fmt, fbo, tex, rb });
  }
  // Weighted blended OIT for the glass grains: accumulate into the OIT targets (depth-tested against the scene's own depth, which
  // is blitted in), then one composite over the scene.  Throws an Error (-> the alpha fallback, named in the title) if the device refuses.
  #oitDraw(ctx, mesh, at, guard) {
    const gl = this.gl, W = ctx.vp[2], H = ctx.vp[3];
    // the scene's framebuffer has a depth format of its own (the page's default one is the browser's choice): try the formats until one blits
    let t = null;
    const order = this.oitDepth ? [this.oitDepth, ...OIT_DEPTH_ORDER.filter((f) => f !== this.oitDepth)] : OIT_DEPTH_ORDER, tried = [];
    for (const fmt of order) {
      t = this.#oitTarget(W, H, fmt);
      while (gl.getError() !== gl.NO_ERROR) { /* (an earlier error is not this pass's) */ }
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, ctx.bound); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, t.fbo);
      gl.blitFramebuffer(ctx.vp[0], ctx.vp[1], ctx.vp[0] + W, ctx.vp[1] + H, 0, 0, W, H, gl.DEPTH_BUFFER_BIT, gl.NEAREST);
      const e = gl.getError();
      if (e === gl.NO_ERROR) { this.oitDepth = fmt; break; }
      tried.push(`${fmt}: 0x${e.toString(16)}`); t = null;
    }
    if (!t) throw new Error(`the scene's depth could not be copied in any depth format (${tried.join(", ")}; ${W}x${H}, ${gl.getParameter(gl.DEPTH_BITS)} depth bits)`);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo); gl.viewport(0, 0, W, H);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4)); gl.clearBufferfv(gl.COLOR, 1, new Float32Array(4));
    gl.uniform1i(at("uOit"), 1);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.enable(gl.CULL_FACE);                  // both faces of every bead accumulate: the back faces (the shader's inner colour), then the front
    gl.cullFace(gl.FRONT); gl.vertexAttribI4i(5, 1, 0, 0, 0);
    if (mesh.array) gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.count, ctx.n);
    else gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0, ctx.n);
    gl.cullFace(gl.BACK); gl.vertexAttribI4i(5, 0, 0, 0, 0);
    if (mesh.array) gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.count, ctx.n);
    else gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0, ctx.n);
    gl.disable(gl.CULL_FACE);
    gl.uniform1i(at("uOit"), 0);
    // the composite: the averaged colour with coverage 1 - exp(-tau) over the scene
    const cp = this.#program("oit.comp", QUAD_VS, OIT_COMPOSITE_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], W, H);
    gl.useProgram(cp.program);
    guard.bind(0, t.tex[0]); guard.bind(1, t.tex[1]);
    gl.uniform1i(cp.u.uAccum, 0); gl.uniform1i(cp.u.uTau, 1); gl.uniform2i(cp.u.uOrigin, ctx.vp[0], ctx.vp[1]);
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.#quad(ctx.vao);
  }
  // style: "material" (opaque grains, or glass when the picked material is), "stained" (glass: OIT), "stained_fast" (glass: alpha)
  #material(ctx, choice, style) {
    const gl = this.gl, S = ctx.scene, P = S.program, at = (name) => gl.getUniformLocation(P, name);
    const glass = style !== "material" || !!S.glass, fast = style === "stained_fast";
    const guard = new UnitGuard(gl);
    let verdict = null;
    if (glass) {
      verdict = glassMode({ fast, caps: this.caps, broken: this.oitBroken || (S.oit ? "" : `the scene shader could not be extended for OIT (${S.oitNote || "unknown"})`) });
      this.glassInfo = verdict.reason;
    }
    const mesh = this.#sphere(ctx, choice.lod);
    gl.bindBuffer(gl.ARRAY_BUFFER, ctx.buffer);
    const held = this.#sceneBegin(ctx);
    gl.bindVertexArray(mesh.vao);
    let oitDone = false;
    if (glass && verdict.mode === "oit") {
      try { this.#oitDraw(ctx, mesh, at, guard); oitDone = true; }
      catch (error) {
        this.oitBroken = String(error.message || error); this.glassInfo = glassMode({ caps: this.caps, broken: this.oitBroken }).reason;
        gl.uniform1i(at("uOit"), 0); gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
        gl.useProgram(P); gl.bindVertexArray(mesh.vao);
      }
    }
    if (oitDone) { /* composited above */ }
    else if (glass) {
      // The scene's glass pass (litPass): alpha-blended over the opaque scene, depth-TESTED against it, depth NOT written, drawn
      // in submission order (not sorted, not order-independent).  Here each sphere also sorts itself: its back faces first
      // (shown with the tint's inner colour), then its front faces.  (The stained_fast style, and the fallback without OIT.)
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT); gl.vertexAttribI4i(5, 1, 0, 0, 0);
      gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0, ctx.n);
      gl.cullFace(gl.BACK); gl.vertexAttribI4i(5, 0, 0, 0, 0);
      gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0, ctx.n);
      gl.disable(gl.CULL_FACE);
    } else {
      gl.depthMask(true); gl.disable(gl.BLEND);
      gl.drawElementsInstanced(gl.TRIANGLES, mesh.count, gl.UNSIGNED_SHORT, 0, ctx.n);
    }
    guard.restore();
    this.#sceneEnd(ctx, held);
    gl.bindVertexArray(ctx.vao);
  }
  // The scene's Phong program set up for instanced grains (the material and emoji-mesh styles): the flag on, the matrices, the board's
  // window, the colour source, and the constant attributes the scene's own draws set (aMatId, aUv, aGroupId, aCullImmune = 0: back
  // faces culled in the shader, aVelocity).  -> the attributes' held values for #sceneEnd.
  #sceneBegin(ctx) {
    const gl = this.gl, S = ctx.scene, P = S.program, at = (name) => gl.getUniformLocation(P, name), cam = ctx.camera;
    const held = [2, 3, 4, 5, 6].map((i) => [i, gl.getVertexAttrib(i, gl.CURRENT_VERTEX_ATTRIB)]);
    gl.useProgram(P);
    gl.uniform1i(at("uInstanced"), 1);
    gl.uniformMatrix4fv(at("uMVP"), false, ctx.viewProj); gl.uniformMatrix4fv(at("uMV"), false, cam.view);
    gl.uniformMatrix4fv(at("uShadowMatrices[0]"), false, S.shadow);
    gl.uniform4f(at("uInstMap"), ctx.map[0], ctx.map[1], ctx.map[2], 0); gl.uniform3f(at("uInstBoard"), ctx.board[0], ctx.board[1], ctx.y);
    const own = ctx.colorMode === "material" || !!S.designedMesh;     // saved meshes retain every designed material colour
    grainUniforms(gl, (n) => at(n), own ? null : ctx.tint, own ? null : ctx.grain);
    gl.uniform1f(at("uInstOwn"), own ? 1 : 0);
    // aGroupId: 0 (no group) or the material's own (an emitter group, or the ThermalChunk index of a blackbody finish: uThermalEmission on for it)
    gl.vertexAttribI4i(2, S.materialId, 0, 0, 0); gl.vertexAttrib2f(3, 0, 0); gl.vertexAttribI4i(4, S.group || 0, 0, 0, 0); gl.vertexAttribI4i(5, 0, 0, 0, 0); gl.vertexAttrib3f(6, 0, 0, 0);
    if (S.thermal) gl.uniform1i(at("uThermalEmission"), 1);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    return held;
  }
  #sceneEnd(ctx, held) {
    const gl = this.gl, S = ctx.scene, P = S.program, at = (name) => gl.getUniformLocation(P, name);
    gl.useProgram(P);
    gl.uniform1i(at("uInstanced"), 0);                              // the scene's pieces never see the flag
    gl.uniform1f(at("uInstRot"), 0);
    gl.uniform1i(at("uInstSpin"), 0);
    if (S.thermal) gl.uniform1i(at("uThermalEmission"), 0);
    for (const [i, v] of held) { if (v instanceof Int32Array) gl.vertexAttribI4i(i, v[0], v[1], v[2], v[3]); else if (v instanceof Uint32Array) gl.vertexAttribI4ui(i, v[0], v[1], v[2], v[3]); else gl.vertexAttrib4f(i, v[0], v[1], v[2], v[3]); }
  }

  // ---- emoji: the atlas and meta are page assets (web_export.py: emoji_atlas.png, emoji_atlas.json), fetched once on first use
  #emojiLoad() {
    if (this.emoji) return;
    const E = this.emoji = { meta: null, rects: null, words: null, bitmap: null, tex: null, tris: null, trisFor: null };
    const url = (name) => new URL(name, import.meta.url).href;
    const fail = (ids, why) => { for (const id of ids) this.broken[id] = why; };
    fetch(url("emoji_atlas.json")).then((r) => { if (!r.ok) throw new Error(`emoji_atlas.json: HTTP ${r.status}`); return r.json(); }).then((meta) => {
      const bad = checkAtlasMeta(meta);
      if (bad) return fail(["emoji", "emoji_mesh"], bad);
      E.meta = meta; E.rects = atlasRects(meta); E.words = meta.kinds.map((k) => packRgba8(k.color || [0.8, 0.8, 0.8]));
      return fetch(url("emoji_atlas.png")).then((r) => { if (!r.ok) throw new Error(`emoji_atlas.png: HTTP ${r.status}`); return r.blob(); })
        .then((blob) => createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" }))
        .then((bitmap) => { E.bitmap = bitmap; }, (error) => fail(["emoji"], `the emoji atlas image did not load (${error.message || error})`));
    }).catch((error) => fail(["emoji", "emoji_mesh"], `the emoji atlas did not load (${error.message || error})`));
  }
  // "" when the style's assets are here, else why not yet (the image variant needs the picture, the mesh variant only the meta)
  #emojiWait(style) {
    const E = this.emoji;
    if (!E || !E.meta) return "loading the emoji atlas";
    if (style === "emoji" && !E.bitmap && !E.tex) return "loading the emoji atlas image";
    return "";
  }
  #emojiTexture(guard) {
    const E = this.emoji, gl = this.gl;
    if (E.tex) return E.tex;
    const tex = gl.createTexture();
    guard.bind(0, tex);
    const flip = gl.getParameter(gl.UNPACK_FLIP_Y_WEBGL), pre = gl.getParameter(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    while (gl.getError() !== gl.NO_ERROR) { /* (an earlier error is not this upload's) */ }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, E.bitmap);
    const error = gl.getError();
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flip); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, pre);
    if (error !== gl.NO_ERROR) { gl.deleteTexture(tex); throw new Unavailable(`the emoji atlas could not be uploaded (0x${error.toString(16)})`); }
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    if (E.bitmap.close) E.bitmap.close();
    E.bitmap = null; E.tex = tex;
    return tex;
  }
  // the assignment's uniforms (EMOJI_KIND_GLSL): which kind a grain is, from its slot and radius
  #emojiKindUniforms(p, ctx) {
    const gl = this.gl, grain = ctx.grain || resolveColorSource({}).grain;
    gl.uniform4f(p.u.uEmoji, Math.max(0, EMOJI_ASSIGN_IDS.indexOf(EMOJI_OPTIONS.assign)), this.emoji.meta.kinds.length, grain.range[0], grain.range[1]);
    gl.uniform1fv(p.u.uEmojiBands, grain.bands);
  }
  #emoji(ctx, guard) {
    const gl = this.gl, E = this.emoji;
    let p;
    try { p = this.#program("emoji", EMOJI_VS, EMOJI_FS); } catch (error) { throw new Unavailable(`the emoji shader did not build (${String(error.message || error).slice(0, 160)})`); }
    gl.useProgram(p.program); this.#sprite(p, ctx, 1, { grow: 1, minPx: 3, pad: 0 });
    this.#emojiKindUniforms(p, ctx);
    gl.uniform1f(p.u.uInk, E.meta.ink); gl.uniform4fv(p.u.uEmojiRect, E.rects);
    gl.uniform1f(p.u.uNative, emojiNative(ctx.tint, ctx.grain) ? 1 : 0);
    guard.bind(0, this.#emojiTexture(guard)); gl.uniform1i(p.u.uAtlas, 0);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true); gl.disable(gl.BLEND);
    gl.drawArrays(gl.POINTS, 0, ctx.n);
  }
  // The mesh variant: the grains partitioned by kind (one counting pass, one placing pass over the packed points, on the CPU), each
  // kind drawn as ONE instanced batch of its mesh through the scene's Phong program.  A batch is the packed points of its grains
  // plus the displayed slot (BATCH_PACKING), so a grain keeps its fixed rotation as the others around it come and go.
  #emojiMesh(ctx, choice) {
    const gl = this.gl, S = ctx.scene, E = this.emoji, L = INSTANCE_LOCATIONS, kinds = E.meta.kinds, K = kinds.length;
    const suffix = choice.lod ? "_lod" : "";
    const meshes = kinds.map((k) => (k.mesh && S.meshes && S.meshes[k.mesh + suffix]) || null);
    const missing = kinds.filter((k, i) => !meshes[i]).map((k) => `${k.name}${suffix}`);
    if (missing.length) throw new Unavailable(`the scene has no emoji mesh for ${missing.join(", ")}`);
    // ---- partition by kind
    const t0 = performance.now();
    const n = ctx.n, data = ctx.data, bits = new Uint32Array(data.buffer, data.byteOffset, data.length);
    const grain = ctx.grain || resolveColorSource({}).grain, assign = EMOJI_OPTIONS.assign;
    const native = emojiNative(ctx.tint, ctx.grain) && ctx.colorMode !== "material";
    const sc = (this.emojiScratch ||= { kind: new Uint8Array(0), out: new Float32Array(0) });
    if (sc.kind.length < n) { sc.kind = new Uint8Array(n); sc.out = new Float32Array(n * 6); }
    const poseLimit = Math.min(n, ctx.spinPose?.length / 4 || 0);
    const counts = new Uint32Array(K), spinCounts = new Uint32Array(K), starts = new Uint32Array(K + 1), spinStarts = new Uint32Array(K + 1);
    for (let i = 0; i < n; i++) {
      const meta = data[5 * i + 4];
      if (!(meta >= 50)) { sc.kind[i] = 255; continue; }
      const flag = Math.floor(meta / 100 + 0.001), k = emojiKind(assign, i, (meta - flag * 100) * 0.001, K, grain.bands, grain.range);
      sc.kind[i] = k; counts[k]++;
      if (i < poseLimit) spinCounts[k]++;
    }
    for (let k = 0; k < K; k++) { starts[k + 1] = starts[k] + counts[k]; spinStarts[k + 1] = spinStarts[k] + spinCounts[k]; }
    const total = starts[K], cursor = starts.slice(0, K), out = sc.out, outBits = new Uint32Array(out.buffer);
    const poses = ctx.spinPose;
    const spinTotal = spinStarts[K], spinOut = spinTotal ? (sc.spin ||= new Float32Array(0)) : null;
    if (spinOut && spinOut.length < spinTotal * 4) sc.spin = new Float32Array(spinTotal * 4);
    const spinCursor = spinStarts.slice(0, K);
    for (let i = 0; i < n; i++) {
      const k = sc.kind[i];
      if (k === 255) continue;
      const j = cursor[k]++, o = j * 6, s = 5 * i;
      out[o] = data[s]; out[o + 1] = data[s + 1]; out[o + 2] = data[s + 2];
      outBits[o + 3] = native ? E.words[k] : bits[s + 3];
      out[o + 4] = data[s + 4]; out[o + 5] = i;
      if (spinOut && i < poseLimit) {
        const q = i * 4, d = spinCursor[k]++ * 4;
        sc.spin[d] = poses[q]; sc.spin[d + 1] = poses[q + 1]; sc.spin[d + 2] = poses[q + 2]; sc.spin[d + 3] = poses[q + 3];
      }
    }
    if (!total) return;
    const buffer = (E.batchBuffer ||= gl.createBuffer());
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW, 0, total * 6);
    if (spinOut) {
      const spinBuffer = (E.spinBuffer ||= gl.createBuffer());
      gl.bindBuffer(gl.ARRAY_BUFFER, spinBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sc.spin, gl.DYNAMIC_DRAW, 0, spinTotal * 4);
    }
    const partitionMs = performance.now() - t0;
    // ---- draw one batch per kind
    const held = this.#sceneBegin(ctx), at = (name) => gl.getUniformLocation(S.program, name);
    gl.uniform1f(at("uInstRot"), 1);
    gl.depthMask(true); gl.disable(gl.BLEND);
    let tris = 0;
    for (let k = 0; k < K; k++) {
      if (!counts[k]) continue;
      const mesh = meshes[k], vao = this.#emojiVao(E, kinds[k].mesh + suffix, mesh);
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);                                   // (the batch's byte offset moves with the counts)
      const setBatch = (first) => {
        const base = first * BATCH_PACKING.stride;
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.vertexAttribPointer(L.position, PACKING.position.size, gl.FLOAT, false, BATCH_PACKING.stride, base + PACKING.position.offset);
        gl.vertexAttribPointer(L.color, 4, gl.UNSIGNED_BYTE, true, BATCH_PACKING.stride, base + PACKING.color.offset);
        gl.vertexAttribPointer(L.meta, 1, gl.FLOAT, false, BATCH_PACKING.stride, base + PACKING.meta.offset);
        gl.vertexAttribPointer(L.id, 1, gl.FLOAT, false, BATCH_PACKING.stride, base + BATCH_PACKING.id.offset);
      };
      gl.uniform4f(at("uInstMesh"), mesh.center[0], mesh.center[1], mesh.center[2], 1 / Math.max(mesh.half[0], mesh.half[1], 1e-4));
      if (spinCounts[k]) {
        setBatch(starts[k]);
        gl.bindBuffer(gl.ARRAY_BUFFER, E.spinBuffer);
        gl.enableVertexAttribArray(L.quaternion);
        gl.vertexAttribPointer(L.quaternion, 4, gl.FLOAT, false, 16, spinStarts[k] * 16);
        gl.uniform1i(at("uInstSpin"), 1);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.count, spinCounts[k]);
      }
      if (counts[k] > spinCounts[k]) {
        setBatch(starts[k] + spinCounts[k]);
        gl.disableVertexAttribArray(L.quaternion);
        gl.vertexAttrib4f(L.quaternion, 0, 0, 0, 1);
        gl.uniform1i(at("uInstSpin"), 0);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.count, counts[k] - spinCounts[k]);
      }
      tris += counts[k] * mesh.count / 3;
    }
    this.emojiInfo = `${total.toLocaleString("en-US")} grains in ${counts.filter((c) => c).length} batches, ${Math.round(tris).toLocaleString("en-US")} triangles, partition + upload ${partitionMs.toFixed(2)} ms${spinOut ? `; physical tumble for the first ${Math.min(n, poses.length / 4).toLocaleString("en-US")} displayed IDs` : ""}`;
    this.#sceneEnd(ctx, held);
    gl.bindVertexArray(ctx.vao);
  }
  // a kind's VAO: the scene mesh's own vertex buffer (positions, normals: its interleaved layout, see app.js uploadMesh) plus the
  // instance attributes (whose pointers are set per draw, over the batch's current offset)
  #emojiVao(E, name, mesh) {
    const gl = this.gl, vaos = (E.vaos ||= new Map());
    const held = vaos.get(name);
    if (held && held.mesh === mesh) return held.vao;
    if (held) gl.deleteVertexArray(held.vao);
    gl.bindVertexArray(mesh.vao);
    const vbo = gl.getVertexAttrib(0, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    for (const at of [INSTANCE_LOCATIONS.position, INSTANCE_LOCATIONS.color, INSTANCE_LOCATIONS.meta, INSTANCE_LOCATIONS.id]) { gl.enableVertexAttribArray(at); gl.vertexAttribDivisor(at, 1); }
    gl.vertexAttribDivisor(INSTANCE_LOCATIONS.quaternion, 1);
    vaos.set(name, { mesh, vao });
    return vao;
  }

  #designedMesh(ctx, choice) {
    const gl = this.gl, original = ctx.scene.designedMesh;
    const mesh = !choice.lod || choice.lod === "full" ? original : original?.lods?.[choice.lod];
    if (!mesh || !mesh.count) throw new Unavailable("no saved object mesh is loaded");
    const held = this.#sceneBegin(ctx), at = (name) => gl.getUniformLocation(ctx.scene.program, name);
    gl.uniform1f(at("uInstRot"), 2);
    gl.uniform4f(at("uInstMesh"), mesh.center[0], mesh.center[1], mesh.center[2], mesh.scale);
    // The packed display slot is the orientation key.  The current sphere law
    // does not export spin/orientation, so this pose is deliberately fixed.
    gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
    if (mesh.opaque) {
      gl.bindVertexArray(this.#designedVao(mesh.opaque, ctx.buffer));
      gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.opaque.count, ctx.n);
    }
    if (mesh.glass) {
      gl.bindVertexArray(this.#designedVao(mesh.glass, ctx.buffer));
      const guard = new UnitGuard(gl);
      let composited = false;
      if (ctx.scene.oit && this.caps.floatRender) {
        try { this.#oitDraw(ctx, mesh.glass, at, guard); composited = true; this.glassInfo = "saved object glass: weighted blended OIT"; }
        catch (error) {
          this.oitBroken = String(error.message || error);
          gl.useProgram(ctx.scene.program); gl.bindVertexArray(this.#designedVao(mesh.glass, ctx.buffer));
          gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
          gl.uniform1i(at("uOit"), 0);
        }
      }
      if (!composited) {
        this.glassInfo = `saved object glass: alpha blend${this.oitBroken ? ` (${this.oitBroken})` : ""}`;
        gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
        gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT); gl.vertexAttribI4i(5, 1, 0, 0, 0);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.glass.count, ctx.n);
        gl.cullFace(gl.BACK); gl.vertexAttribI4i(5, 0, 0, 0, 0);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, mesh.glass.count, ctx.n);
        gl.disable(gl.CULL_FACE); gl.disable(gl.BLEND);
      }
      guard.restore();
    }
    this.#sceneEnd(ctx, held);
    gl.bindVertexArray(ctx.vao);
  }
  #designedVao(mesh, pointBuffer) {
    const gl = this.gl, vaos = (this.designedVaos ||= new WeakMap()), held = vaos.get(mesh);
    if (held && held.mesh === mesh && held.pointBuffer === pointBuffer) return held.vao;
    if (held) gl.deleteVertexArray(held.vao);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    for (const [at, size, integer] of [[0, 3, false], [1, 3, false], [2, 1, true]]) {
      gl.bindBuffer(gl.ARRAY_BUFFER, mesh.buffers[at]);
      gl.enableVertexAttribArray(at);
      if (integer) gl.vertexAttribIPointer(at, size, gl.INT, 0, 0);
      else gl.vertexAttribPointer(at, size, gl.FLOAT, false, 0, 0);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
    for (const [at, size, type, normalized, offset] of [
      [L.position, 3, gl.FLOAT, false, PACKING.position.offset],
      [L.color, 4, gl.UNSIGNED_BYTE, true, PACKING.color.offset],
      [L.meta, 1, gl.FLOAT, false, PACKING.meta.offset],
    ]) {
      gl.enableVertexAttribArray(at); gl.vertexAttribPointer(at, size, type, normalized, PACKING.stride, offset); gl.vertexAttribDivisor(at, 1);
    }
    // Without a separate batch the vertex shader receives the instance index
    // through gl_InstanceID; location 10 remains a constant zero for this path.
    gl.bindVertexArray(null);
    vaos.set(mesh, { vao, mesh, pointBuffer });
    return vao;
  }

  // ---- liquid
  #liquid(ctx, guard) {
    const gl = this.gl, { W, H, n } = ctx;
    const scale = n > 100000 ? 0.5 : 1, tw = Math.max(2, Math.ceil(W * scale)), th = Math.max(2, Math.ceil(H * scale));
    const fmt = this.#depthFormat();
    const D = this.#target("liq.depth", tw, th, { fmt, depth: true }), A = this.#target("liq.a", tw, th, { fmt }), B = this.#target("liq.b", tw, th, { fmt });
    const T = this.#target("liq.thick", tw, th, { fmt: "rgba16f", linear: true, mips: true });
    const dp = this.#program("liq.depth", SPRITE_VS, DEPTH_FS), tp = this.#program("liq.thick", SPRITE_VS, THICK_FS);
    const bp = this.#program("liq.blur", QUAD_VS, BLUR_FS), cp = this.#program("liq.comp", QUAD_VS, LIQUID_FS);
    const grow = 1.7, minPx = 3.0 * scale + 1.0;
    // 1. depth: the nearest sphere surface per texel (hardware depth test; the colour carries the eye distance)
    gl.bindFramebuffer(gl.FRAMEBUFFER, D.fbo); gl.viewport(0, 0, tw, th);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4)); gl.clearBufferfv(gl.DEPTH, 0, new Float32Array([1]));
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true); gl.disable(gl.BLEND);
    gl.useProgram(dp.program); this.#sprite(dp, ctx, scale, { grow, minPx, pad: 2 });
    gl.drawArrays(gl.POINTS, 0, n);
    // 2. thickness (and thickness-weighted colour), additively
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(tp.program); this.#sprite(tp, ctx, scale, { grow, minPx, pad: 2 }); gl.uniform1f(tp.u.uThickGain, 1.0);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.disable(gl.BLEND);
    // 3. a separable narrow-range bilateral blur of the depth
    const rWorld = ParticleStyles.typicalRadius(ctx.data) * ctx.map[2] * grow;
    const eyeC = -(ctx.camera.view[2] * ctx.board[0] * 0.5 + ctx.camera.view[6] * ctx.y + ctx.camera.view[10] * ctx.board[1] * 0.5 + ctx.camera.view[14]);
    const rPx = rWorld * ctx.focal * H / 2 * scale / sizeDepth(ctx.camera.proj || PERSPECTIVE_W, -eyeC);       // [CURVE] clip.w (1 under an orthographic projection)
    gl.useProgram(bp.program); gl.uniform1i(bp.u.uD, 0); gl.uniform1f(bp.u.uStep, Math.max(1, Math.round(rPx * 0.45))); gl.uniform1f(bp.u.uRange, Math.max(1e-4, rWorld * 1.5));
    gl.bindFramebuffer(gl.FRAMEBUFFER, A.fbo); guard.bind(0, D.tex); gl.uniform2i(bp.u.uDir, 1, 0); this.#quad(ctx.vao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, B.fbo); guard.bind(0, A.tex); gl.uniform2i(bp.u.uDir, 0, 1); this.#quad(ctx.vao);
    // 4. shade the surface over the scene (depth-tested against it at the surface's own depth)
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    gl.useProgram(cp.program);
    guard.bind(1, T.tex); gl.generateMipmap(gl.TEXTURE_2D);
    guard.bind(0, B.tex); gl.uniform1i(cp.u.uD, 0); gl.uniform1i(cp.u.uT, 1);
    const P = ctx.camera.proj;
    gl.uniform2f(cp.u.uProj0, P[0], P[8]); gl.uniform2f(cp.u.uProj5, P[5], P[9]); gl.uniform4f(cp.u.uProjZ, P[10], P[14], P[11], P[15]);
    gl.uniform1f(cp.u.uTexScale, tw / W); gl.uniform2f(cp.u.uTexSize, tw, th);
    this.#lighting(cp, ctx);
    gl.uniform1f(cp.u.uLod, Math.min(6, Math.max(0, Math.log2(Math.max(1, rPx * 1.2)))));
    gl.uniform1f(cp.u.uAbsorb, 1 / Math.max(1e-4, rWorld * 3.0 / grow));
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.#quad(ctx.vao);
  }

  // ---- the density map (the local area fraction the grains cover, and the moving share of it), half resolution
  #density(ctx, guard, scale, mips = false) {
    const gl = this.gl, { W, H, n } = ctx;
    const tw = Math.max(2, Math.ceil(W * scale)), th = Math.max(2, Math.ceil(H * scale));
    const T = this.#target(mips ? "density.heat" : "density", tw, th, { fmt: "rgba16f", linear: true, mips });
    const p = this.#program("density", SPRITE_VS, DENSITY_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, tw, th); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(p.program); this.#sprite(p, ctx, scale, { grow: 0, minPx: 16 * scale });
    gl.drawArrays(gl.POINTS, 0, n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    gl.disable(gl.BLEND);
    if (mips) { guard.bind(0, T.tex); gl.generateMipmap(gl.TEXTURE_2D); }
    return T;
  }

  // ---- sand
  #sand(ctx, guard) {
    const gl = this.gl, { n } = ctx;
    const ao = !!this.caps.floatRender;
    const occ = ao ? this.#density(ctx, guard, n > 100000 ? 0.5 : 0.75) : null;
    const p = this.#program("sand", SPRITE_VS, SAND_FS);
    gl.useProgram(p.program); this.#sprite(p, ctx, 1, { grow: 1, minPx: 2, pad: 2 });
    this.#lighting(p, ctx);
    gl.uniform1f(p.u.uAO, ao ? 0.45 : 0); gl.uniform2f(p.u.uViewport, ctx.W, ctx.H);
    if (occ) { guard.bind(0, occ.tex); gl.uniform1i(p.u.uOcc, 0); }
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, 0, n);
  }

  // ---- glow: with float targets the sprites sum in one and are tone-mapped on the way out (a dense pile saturates smoothly); without, they add straight onto the scene
  #glow(ctx, guard) {
    const gl = this.gl, { n } = ctx;
    const p = this.#program("glow", SPRITE_VS, GLOW_FS);
    if (!this.caps.floatRender) {
      gl.useProgram(p.program); this.#sprite(p, ctx, 1, { grow: 3.2, minPx: 7, pad: 2, full: 1 });
      gl.uniform1f(p.u.uGain, 0.35);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawArrays(gl.POINTS, 0, n);
      return;
    }
    const scale = n > 100000 ? 0.5 : 1, tw = Math.max(2, Math.ceil(ctx.W * scale)), th = Math.max(2, Math.ceil(ctx.H * scale));
    const T = this.#target("glow", tw, th, { fmt: "rgba16f", linear: true });
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, tw, th); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(p.program); this.#sprite(p, ctx, scale, { grow: 3.2, minPx: 7 * scale, pad: 2, full: 1 });
    gl.uniform1f(p.u.uGain, 1.0 * Math.min(1, 1.5 / Math.sqrt(Math.max(1, ctx.stride || 1))));
    gl.drawArrays(gl.POINTS, 0, n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    const r = this.#program("glow.resolve", PLANE_VS, GLOW_RESOLVE_FS);
    gl.bindVertexArray(this.emptyVao); gl.useProgram(r.program);
    gl.uniformMatrix4fv(r.u.uViewProj, false, ctx.viewProj); gl.uniform3f(r.u.uBoard, ctx.board[0], ctx.board[1], ctx.y); gl.uniform1f(r.u.uMargin, 0.25);
    guard.bind(0, T.tex); gl.uniform1i(r.u.uDens, 0); gl.uniform2f(r.u.uViewport, ctx.W, ctx.H);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(ctx.vao);
  }

  // ---- heat: the density / motion field through an adaptive log tone map (see "heat tone mapping" above)
  // ctx.heat = { fixed: bool, gamma: number } (the page's manual override).  this.heatInfo = what the title shows.
  static HIST_W = 96; static HIST_H = 64; static HIST_EVERY = 6;      // frames between the small histogram readbacks
  #heatRange(ctx, guard, T) {
    const gl = this.gl, h = (this.heatState ||= { range: null, frame: 0, at: ctx.now, mode: "" });
    const fixed = !!(ctx.heat && ctx.heat.fixed);
    if (fixed) { h.range = [...HEAT_FIXED]; h.mode = "fixed"; return h.range; }
    if (h.mode === "fixed") h.range = null;
    h.mode = "adaptive";
    const dt = Math.min(0.5, Math.max(0, (ctx.now - h.at) / 1000)); h.at = ctx.now;
    if (h.frame++ % ParticleStyles.HIST_EVERY === 0 || !h.range) {
      const R = this.#target("heat.hist", ParticleStyles.HIST_W, ParticleStyles.HIST_H, { fmt: "rgba8" });
      const hp = this.#program("heat.hist", QUAD_VS, HIST_FS);
      gl.bindFramebuffer(gl.FRAMEBUFFER, R.fbo); gl.viewport(0, 0, R.w, R.h);
      gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      gl.useProgram(hp.program); guard.bind(0, T.tex); gl.uniform1i(hp.u.uDens, 0); gl.uniform2f(hp.u.uSize, R.w, R.h);
      gl.uniform1f(hp.u.uLod, Math.max(0, Math.log2(Math.max(T.w / R.w, T.h / R.h))));        // the mip level that averages a histogram texel's footprint
      gl.bindVertexArray(this.emptyVao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); gl.bindVertexArray(ctx.vao);
      const px = (this.histPixels ||= new Uint8Array(R.w * R.h * 4));
      gl.readPixels(0, 0, R.w, R.h, gl.RGBA, gl.UNSIGNED_BYTE, px);                          // 24 KB, not the field
      const counts = new Float64Array(HEAT_HIST_BINS);
      for (let k = 0; k < px.length; k += 4) if (px[k + 3]) counts[px[k]]++;
      const target = heatTarget(counts);
      if (!h.range && target) h.range = [...target];                                         // (afterwards the range only follows h.target, smoothed below)
      h.target = target || h.target;
    }
    if (h.target && h.range) h.range = adaptRange(h.range, h.target, dt, HEAT_TAU);
    return h.range || [...HEAT_FIXED];
  }
  #heat(ctx, guard) {
    const gl = this.gl;
    const T = this.#density(ctx, guard, ctx.n > 100000 ? 0.5 : 0.75, true);
    const range = this.#heatRange(ctx, guard, T);
    const gamma = ctx.heat && Number.isFinite(ctx.heat.gamma) ? Math.min(3, Math.max(0.3, ctx.heat.gamma)) : 1;
    this.heatInfo = `heat range ${xToQuantity(range[0]).toPrecision(2)}..${xToQuantity(range[1]).toPrecision(2)} (${this.heatState.mode}${gamma !== 1 ? `, gamma ${gamma}` : ""})`;
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    const p = this.#program("heat", PLANE_VS, HEAT_FS);
    gl.bindVertexArray(this.emptyVao);
    gl.useProgram(p.program); gl.uniform4fv(p.u.uTint, ctx.tint || NO_TINT);
    gl.uniform2f(p.u.uRange, range[0], range[1]); gl.uniform1f(p.u.uGamma, gamma);
    gl.uniformMatrix4fv(p.u.uViewProj, false, ctx.viewProj); gl.uniform3f(p.u.uBoard, ctx.board[0], ctx.board[1], ctx.y); gl.uniform1f(p.u.uMargin, 0.0);
    guard.bind(0, T.tex); gl.uniform1i(p.u.uDens, 0); gl.uniform2f(p.u.uViewport, ctx.W, ctx.H);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(ctx.vao);
  }

  // ---- stamp: the memory under the points
  static FADE_PER_SECOND = 0.6;           // the stamped intensity lost per second (a full stamp lasts ~1.7 s)
  #stamp(ctx, guard) {
    const gl = this.gl;
    const tw = Math.min(1024, gl.getParameter(gl.MAX_TEXTURE_SIZE)), th = Math.max(64, Math.min(2048, Math.round(tw * ctx.board[1] / ctx.board[0])));
    const M = this.#target("stamp", tw, th, { fmt: "rgba8", linear: true });
    if (this.stamp !== M) {                      // a new memory (first use, or the board's shape changed): start empty
      this.stamp = M;
      gl.bindFramebuffer(gl.FRAMEBUFFER, M.fbo); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      this.stampAt = ctx.now; this.fadeCarry = 0;
    }
    const sp = this.#program("stamp", STAMP_VS, STAMP_FS), fp = this.#program("stamp.fade", QUAD_VS, FADE_FS), mp = this.#program("stamp.show", PLANE_VS, MEMORY_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, M.fbo); gl.viewport(0, 0, tw, th);
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND);
    // fade: the alpha (the intensity) loses whole 1/255 steps at the rate asked for; the colour is left alone
    const dt = Math.min(0.1, Math.max(0, (ctx.now - this.stampAt) / 1000)); this.stampAt = ctx.now;
    this.fadeCarry += dt * ParticleStyles.FADE_PER_SECOND * 255;
    const steps = Math.floor(this.fadeCarry); this.fadeCarry -= steps;
    if (steps > 0) {
      gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_REVERSE_SUBTRACT); gl.blendFuncSeparate(gl.ZERO, gl.ONE, gl.ONE, gl.ONE);
      gl.useProgram(fp.program); gl.uniform1f(fp.u.uFade, Math.min(255, steps) / 255);
      this.#quad(ctx.vao);
    }
    // stamp: the newest colour over the old where it lands, the intensity the larger of the two
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.MAX); gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE);
    gl.useProgram(sp.program); grainUniforms(gl, (n) => sp.u[n], ctx.tint, ctx.grain);
    gl.uniform4f(sp.u.uMap, ctx.map[0], ctx.map[1], ctx.map[2], 0); gl.uniform3f(sp.u.uBoard, ctx.board[0], ctx.board[1], ctx.y);
    gl.uniform1f(sp.u.uTexel, tw / ctx.board[0]); gl.uniform1f(sp.u.uMax, this.caps.maxPointSize); gl.uniform1f(sp.u.uGrow, 1.4);
    gl.drawArrays(gl.POINTS, 0, ctx.n);
    // show the memory on the board's plane, under the points PointCloud draws next
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    gl.blendEquation(gl.FUNC_ADD);
    gl.bindVertexArray(this.emptyVao);
    gl.useProgram(mp.program);
    gl.uniformMatrix4fv(mp.u.uViewProj, false, ctx.viewProj); gl.uniform3f(mp.u.uBoard, ctx.board[0], ctx.board[1], ctx.y); gl.uniform1f(mp.u.uMargin, 0.0);
    guard.bind(0, M.tex); gl.uniform1i(mp.u.uMem, 0); gl.uniform1f(mp.u.uOpacity, 0.9);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(ctx.vao);
  }
}

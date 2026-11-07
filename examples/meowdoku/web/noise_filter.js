// A RENDER-ONLY spectral noise filter for the compute games' point cloud (compute.js: PointCloud), layered over any
// particle style (particle_styles.js).  It never touches the law, the recording or the data.
//
// Idea: noise in the colour field is speckle -- variance that sits at the finest scales.  Where it is found, the colour is
// replaced by the one that describes the region: its coarse mean, or a fixed colour per noise CLASS (white, pink, brown,
// grey, solid), so noise inside an aggregate shifts into an informative solid colour.  Strata, streams and edges (variance
// at coarse scales) are left as drawn.
//
// Method (no FFT):
//   1. the points' colour field is splat into a board-space texture (premultiplied, opaque discs) and mip-mapped;
//   2. per texel, the gap between mip level b+k and b+k+1 (k = 0..4; b = the level of the median grain, 0 when the grain is a texel) is the variance in the band of scale 2^k texels:
//          m_k = dY^2 + 0.5 |dChroma|^2          (luminance-weighted metric; Y = Rec.709 luma)
//      stored for the 5 bands plus the luminance part, then mip-mapped again: the mean of m_k over one 2^5-texel window is
//      the local variance of that band (one pixel's gap is a single sample; the window is what makes it an estimate);
//   3. the slope of log2 variance against k is d.  A band-energy ladder of a field whose power spectrum goes as f^-beta in 2D
//      is v_k ~ 2^(k (beta - 2)), so   beta = 2 + d :  white 0, pink 1, brown 2; a step edge reads 3.
//   4. noise share = the fraction of the variance in the 3 finest bands; high = speckle (white .98, pink .9, brown .6),
//      low = coarse structure (a step edge .26).  A smooth ramp has no band gap at all (a box mean of a line is its centre): solid.  The replacement fades in over the threshold.
//   5. grey noise is white noise in the luminance-weighted metric: a white-ish class whose variance is mostly luminance.
//
// The pure part (the ladder fit, the classes, the palette, availability) is at the top and is node-tested; the GLSL below
// is built from the same constants.

export const NOISE_MODES = [["mean", "mean colour"], ["spectral", "spectral class"]];
export const NOISE_DEFAULTS = { on: false, mode: "mean", strength: 0.85, threshold: 0.5 };
export const GAPS = 5;               // bands k = 0..4 (6 mip levels read)
export const FINE = 3;               // the bands below this are "fine"
export const EPS_V = 1e-6;           // variance floor inside the log
export const FLOOR_W = 0.05;         // every band keeps this weight in the fit, so a lone band still reads as a steep slope
export const SOLID_LO = 1e-4;        // total variance (sum over bands) below this is solid; above SOLID_HI it is noise or structure
export const SOLID_HI = 4e-4;
export const WHITE_PINK = 0.5, PINK_BROWN = 1.5;        // beta class boundaries
export const GREY_LUM = 0.7;                            // luminance share of the variance above which white-ish noise is grey
export const PALETTE = { white: [0.92, 0.96, 1.0], pink: [1.0, 0.42, 0.72], brown: [0.6, 0.36, 0.16], grey: [0.5, 0.5, 0.5], solid: [0.2, 0.55, 0.9] };
export const NOISE_CLASSES = ["white", "pink", "brown", "grey", "solid"];
// The variance window: mip level WINDOW (32 texels), the SAME for every band.  A window that grows with the band (8 * 2^k) would make a
// step edge read exactly like brown noise (a flat ladder); one window makes the edge's ladder rise (beta 3, share .26) and brown's stay flat.
export const WINDOW = 5;
export const MEAN_LEVEL = 5;         // the "coarse-level mean": the top of the ladder

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const mixv = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);

// v: the 5 band variances (fine -> coarse).  beta = 2 + the weighted least-squares slope of log2(v + eps) against k.
// (Measured in the browser on FFT-shaped fields through the mip pyramid: white 0.10, pink 0.99, brown 2.03.)
export function fitBeta(v) {
  let sw = 0, sx = 0, sy = 0;
  const w = v.map((x) => FLOOR_W + x / (x + EPS_V)), y = v.map((x) => Math.log2(x + EPS_V));
  v.forEach((_, k) => { sw += w[k]; sx += w[k] * k; sy += w[k] * y[k]; });
  const xb = sx / sw, yb = sy / sw;
  let sxx = 0, sxy = 0;
  v.forEach((_, k) => { sxx += w[k] * (k - xb) * (k - xb); sxy += w[k] * (k - xb) * (y[k] - yb); });
  return clamp(2 + sxy / sxx, -1, 4);
}
export const totalVariance = (v) => v.reduce((a, b) => a + b, 0);
export function noiseShare(v) {
  const t = totalVariance(v);
  return t > 0 ? v.slice(0, FINE).reduce((a, b) => a + b, 0) / t : 0;
}
export const solidWeight = (total) => smooth(SOLID_LO, SOLID_HI, total);            // 0 = solid, 1 = there is variance to classify
export const noiseWeight = (share, threshold) => smooth(threshold - 0.1, threshold + 0.1, share);

// The class a ladder reads as, with the numbers behind it.  lum: the luminance share of the variance (0..1).
export function classify(v, lum = 0, threshold = NOISE_DEFAULTS.threshold) {
  const total = totalVariance(v), share = noiseShare(v), beta = fitBeta(v);
  let cls;
  if (solidWeight(total) < 0.5) cls = "solid";
  else if (beta < WHITE_PINK) cls = lum > GREY_LUM ? "grey" : "white";
  else if (beta < PINK_BROWN) cls = "pink";
  else cls = "brown";
  return { cls, beta, share, total, lum, noise: cls !== "solid" && noiseWeight(share, threshold) >= 0.5 };
}
// The fixed palette mixed by beta (white/grey -> pink at 1 -> brown at 2); grey replaces white as the luminance share rises.
export function paletteFor(beta, lum) {
  const w = mixv(PALETTE.white, PALETTE.grey, smooth(0.55, 0.85, lum));
  return beta < 1 ? mixv(w, PALETTE.pink, clamp(beta, 0, 1)) : mixv(PALETTE.pink, PALETTE.brown, clamp(beta - 1, 0, 1));
}
// The blend a texel gets: alpha of the replacement, and the colour source ("mean" -> the region's coarse mean).
export function replacement(v, lum, { mode, strength, threshold }) {
  const total = totalVariance(v), solid = 1 - solidWeight(total), nw = noiseWeight(noiseShare(v), threshold);
  const beta = fitBeta(v);
  if (mode === "spectral") {
    const classW = solid + (1 - solid) * nw;
    return { alpha: strength * classW, rgb: mixv(paletteFor(beta, lum), PALETTE.solid, solid) };
  }
  return { alpha: strength * (1 - solid) * nw, rgb: null };
}

// ---- availability (like particle_styles.js: the filter is off, with the reason, where it cannot run)
export function noiseAvailability(caps) {
  if (!caps) return { ok: false, note: "no WebGL context" };
  if (!caps.floatRender) return { ok: false, note: "needs float or half-float render targets (EXT_color_buffer_float), which this device lacks" };
  return { ok: true, note: caps.floatRender === "half" ? "half-float render targets" : "" };
}
export function chooseNoise(requested, caps, broken = "") {
  if (!requested) return { on: false, reason: "" };
  if (broken) return { on: false, reason: `noise filter unavailable (${broken})` };
  const av = noiseAvailability(caps);
  return av.ok ? { on: true, reason: av.note } : { on: false, reason: `noise filter unavailable (${av.note})` };
}

// ---------------------------------------------------------------- GL (not node-tested)
import { capabilitiesOf, UnitGuard, QUAD_VS, PLANE_VS } from "./particle_styles.js";

class Unavailable extends Error {}
const f = (x) => (Number.isInteger(x) ? `${x}.0` : String(x));
const vec3 = (c) => `vec3(${c.map(f).join(", ")})`;
const SCALE = 1024;       // variances are stored x1024 (half float has a short low end)

const SPLAT_VS = `#version 300 es
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform vec4 uMap; uniform vec3 uBoard; uniform float uTexel; uniform float uMax; uniform vec4 uTint;
out vec3 vColor;
void main() {
  vColor = uTint.w > 0.5 ? uTint.rgb : aColor.rgb; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
  if (aMeta < 50.0) return;
  float flag = floor(aMeta / 100.0 + 0.001);
  float radius = (aMeta - flag * 100.0) * 0.001;
  vec2 b = vec2(uBoard.x * 0.5 + (aPos.x - uMap.x) * uMap.z, uBoard.y * 0.5 - (aPos.y - uMap.y) * uMap.z);
  gl_Position = vec4(b / uBoard.xy * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = clamp(2.0 * radius * uMap.z * uTexel, 1.0, uMax);
}`;
// opaque discs, colour premultiplied (alpha 1): the mip means are then (colour * coverage, coverage)
const SPLAT_FS = `#version 300 es
precision highp float;
in vec3 vColor; out vec4 FragColor;
void main() { if (length(gl_PointCoord - 0.5) > 0.5) discard; FragColor = vec4(vColor, 1.0); }`;

// pass A: per texel, the band gaps between mip level k and k+1 of the colour field
const BANDS_FS = `#version 300 es
precision highp float;
uniform sampler2D uMap; uniform float uBase;
layout(location=0) out vec4 O0; layout(location=1) out vec4 O1;
// the mip level's mean (bilinear: translation-invariant, a step edge reads the same wherever it falls against the texel grid)
vec3 meanAt(float lod, vec2 uv) { vec4 t = textureLod(uMap, uv, lod); return t.rgb / max(t.a, 1e-3); }
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uMap, 0));
  vec4 t0 = texelFetch(uMap, ivec2(gl_FragCoord.xy), 0);
  if (t0.a < 0.5) { O0 = vec4(0.0); O1 = vec4(0.0); return; }
  vec3 prev = meanAt(uBase, uv);                                  // the ladder starts at the grain's own scale
  float m[${GAPS}]; float sumY = 0.0, sumM = 0.0;
  for (int k = 0; k < ${GAPS}; k++) {
    vec3 next = meanAt(uBase + float(k + 1), uv), d = prev - next;
    float dy = dot(d, vec3(0.2126, 0.7152, 0.0722)); vec3 dc = d - vec3(dy);
    m[k] = dy * dy + 0.5 * dot(dc, dc);
    sumY += dy * dy; sumM += m[k]; prev = next;
  }
  O0 = vec4(m[0], m[1], m[2], m[3]) * ${f(SCALE)};
  O1 = vec4(m[4], sumY, sumM, 1.0) * vec4(${f(SCALE)}, ${f(SCALE)}, ${f(SCALE)}, 1.0);
}`;

// pass B: windowed ladder -> slope, share, class -> the replacement (or, for the probe, the numbers)
const FIT_FS = `#version 300 es
precision highp float;
uniform sampler2D uMap; uniform sampler2D uV0; uniform sampler2D uV1;
uniform float uMaxLod, uBase, uStrength, uThreshold; uniform int uMode; uniform int uDebug;
out vec4 FragColor;
float smoothR(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(uV0, 0));
  float v[${GAPS}];
  for (int k = 0; k < ${GAPS - 1}; k++) {
    float lod = min(uBase + float(${WINDOW}), uMaxLod - 2.0);
    vec4 a = textureLod(uV0, uv, lod), b = textureLod(uV1, uv, lod);
    v[k] = a[k] / max(b.a, 1e-4) / ${f(SCALE)};
  }
  { vec4 b = textureLod(uV1, uv, min(uBase + float(${WINDOW}), uMaxLod - 2.0)); v[${GAPS - 1}] = b.r / max(b.a, 1e-4) / ${f(SCALE)}; }
  vec4 bl = textureLod(uV1, uv, min(uBase + float(${WINDOW}), uMaxLod - 2.0));
  float lum = clamp(bl.g / max(bl.b, 1e-9), 0.0, 1.0);
  // the fit (fitBeta)
  float sw = 0.0, sx = 0.0, sy = 0.0, w[${GAPS}], y[${GAPS}], total = 0.0, fine = 0.0;
  for (int k = 0; k < ${GAPS}; k++) {
    w[k] = ${f(FLOOR_W)} + v[k] / (v[k] + ${EPS_V.toExponential()});
    y[k] = log2(v[k] + ${EPS_V.toExponential()});
    sw += w[k]; sx += w[k] * float(k); sy += w[k] * y[k];
    total += v[k]; if (k < ${FINE}) fine += v[k];
  }
  float xb = sx / sw, yb = sy / sw, sxx = 0.0, sxy = 0.0;
  for (int k = 0; k < ${GAPS}; k++) { float dx = float(k) - xb; sxx += w[k] * dx * dx; sxy += w[k] * dx * (y[k] - yb); }
  float beta = clamp(2.0 + sxy / sxx, -1.0, 4.0);
  float share = total > 0.0 ? fine / total : 0.0;
  if (uDebug == 1) { FragColor = vec4(beta, share, total, lum); return; }
  float solid = 1.0 - smoothR(${SOLID_LO.toExponential()}, ${SOLID_HI.toExponential()}, total);
  float nw = smoothR(uThreshold - 0.1, uThreshold + 0.1, share);
  float cover = smoothR(0.25, 0.6, textureLod(uMap, uv, uBase + 2.0).a);
  vec3 white = mix(${vec3(PALETTE.white)}, ${vec3(PALETTE.grey)}, smoothR(0.55, 0.85, lum));
  vec3 pal = beta < 1.0 ? mix(white, ${vec3(PALETTE.pink)}, clamp(beta, 0.0, 1.0)) : mix(${vec3(PALETTE.pink)}, ${vec3(PALETTE.brown)}, clamp(beta - 1.0, 0.0, 1.0));
  vec3 rgb; float alpha;
  if (uMode == 1) { rgb = mix(pal, ${vec3(PALETTE.solid)}, solid); alpha = uStrength * (solid + (1.0 - solid) * nw); }
  else { vec4 t = textureLod(uMap, uv, min(uBase + float(${MEAN_LEVEL}), uMaxLod)); rgb = t.rgb / max(t.a, 1e-3); alpha = uStrength * (1.0 - solid) * nw; }
  if (uDebug == 2) { FragColor = vec4(rgb, alpha); return; }
  FragColor = vec4(rgb, alpha * cover);
}`;
const OVERLAY_FS = `#version 300 es
precision highp float; uniform sampler2D uOv; in vec2 vUV; out vec4 FragColor;
void main() {
  if (vUV.x < 0.0 || vUV.y < 0.0 || vUV.x > 1.0 || vUV.y > 1.0) discard;
  vec4 o = texture(uOv, vUV);
  if (o.a <= 0.003) discard;
  FragColor = o;
}`;

export class NoiseFilter {
  constructor(gl, link, compile) {
    this.gl = gl; this.link = link; this.compile = compile;
    this.caps = capabilitiesOf(gl);
    this.programs = new Map(); this.targets = new Map();
    this.emptyVao = gl.createVertexArray();
    this.broken = "";                    // why it failed here (never retried)
    this.fallback = "";                  // why the last request was not drawn ("" when it was)
    this.texW = 0;
  }

  // Draw the filter over the frame the point cloud just drew.  `settings`: { mode, strength, threshold }.  true when drawn.
  // ctx: { vao (PointCloud's, with the data uploaded), viewProj, view, board, y }.
  apply(data, ctx, settings) {
    const gl = this.gl;
    const choice = chooseNoise(true, this.caps, this.broken);
    this.fallback = choice.on ? "" : choice.reason;
    if (!choice.on || !data || data.length < 5) return false;
    const keep = this.#save(), guard = new UnitGuard(gl);
    let ok = true;
    try {
      const [x0, x1, y0, y1] = ctx.view, k = Math.min(ctx.board[0] / (x1 - x0), ctx.board[1] / (y1 - y0));
      const map = [(x0 + x1) / 2, (y0 + y1) / 2, k];
      const { tw, th, base } = this.#size(data, ctx.board, k);
      const field = this.#target("field", tw, th, { mips: true, attachments: 1 });
      this.#splat(data, ctx, map, field, tw);
      this.#mips(field, guard);
      const out = this.#analyse(field, tw, th, guard, { mode: settings.mode === "spectral" ? 1 : 0, strength: settings.strength, threshold: settings.threshold, debug: settings.debug || 0, base });
      this.last = { out, tw, th, base };                 // for the probe: the analysis target of the last frame (settings.debug 1 = the numbers)
      this.#overlay(out, ctx, guard, keep);
    } catch (error) {
      if (!(error instanceof Unavailable)) { this.#restore(keep, guard); throw error; }
      this.broken = error.message; this.fallback = `noise filter unavailable (${error.message})`; ok = false;
    }
    this.#restore(keep, guard);
    return ok;
  }

  // Test hook: run the classifier on an RGBA float image (w x h, alpha 1).  mode 1: per texel [beta, share, total variance, luminance share];
  // 2: the replacement [r, g, b, alpha] (before the coverage gate).  Returns a Float32Array of w*h*4, rows bottom-up like GL.
  measure(rgba, w, h, { mode = "mean", strength = 1, threshold = NOISE_DEFAULTS.threshold, debug = 1, base = 1 } = {}) {
    const gl = this.gl, keep = this.#save(), guard = new UnitGuard(gl);
    try {
      const field = this.#target("probe", w, h, { mips: true, attachments: 1 });
      guard.bind(0, field.texes[0]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, rgba);
      this.#mips(field, guard);
      const out = this.#analyse(field, w, h, guard, { mode: mode === "spectral" ? 1 : 0, strength, threshold, debug, base });
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
      const px = new Float32Array(w * h * 4); gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, px);
      return px;
    } finally { this.#restore(keep, guard); }
  }

  // ---- state
  #save() {
    const gl = this.gl;
    return { vp: gl.getParameter(gl.VIEWPORT), fbo: gl.getParameter(gl.FRAMEBUFFER_BINDING), program: gl.getParameter(gl.CURRENT_PROGRAM),
             vao: gl.getParameter(gl.VERTEX_ARRAY_BINDING), depth: gl.isEnabled(gl.DEPTH_TEST), depthFunc: gl.getParameter(gl.DEPTH_FUNC),
             depthMask: gl.getParameter(gl.DEPTH_WRITEMASK), blend: gl.isEnabled(gl.BLEND), cull: gl.isEnabled(gl.CULL_FACE),
             scissor: gl.isEnabled(gl.SCISSOR_TEST), offset: gl.isEnabled(gl.POLYGON_OFFSET_FILL), colorMask: gl.getParameter(gl.COLOR_WRITEMASK),
             offsetFactor: gl.getParameter(gl.POLYGON_OFFSET_FACTOR), offsetUnits: gl.getParameter(gl.POLYGON_OFFSET_UNITS), blendEq: [gl.getParameter(gl.BLEND_EQUATION_RGB), gl.getParameter(gl.BLEND_EQUATION_ALPHA)],
             blendFunc: [gl.getParameter(gl.BLEND_SRC_RGB), gl.getParameter(gl.BLEND_DST_RGB), gl.getParameter(gl.BLEND_SRC_ALPHA), gl.getParameter(gl.BLEND_DST_ALPHA)] };
  }
  #restore(s, guard) {
    const gl = this.gl;
    guard.restore();
    gl.bindFramebuffer(gl.FRAMEBUFFER, s.fbo); gl.viewport(s.vp[0], s.vp[1], s.vp[2], s.vp[3]);
    gl.useProgram(s.program); gl.bindVertexArray(s.vao);
    gl.colorMask(...s.colorMask); gl.blendEquationSeparate(...s.blendEq); gl.blendFuncSeparate(...s.blendFunc); gl.polygonOffset(s.offsetFactor, s.offsetUnits); gl.depthFunc(s.depthFunc); gl.depthMask(s.depthMask);
    const set = (cap, on) => (on ? gl.enable(cap) : gl.disable(cap));
    set(gl.DEPTH_TEST, s.depth); set(gl.BLEND, s.blend); set(gl.CULL_FACE, s.cull); set(gl.SCISSOR_TEST, s.scissor); set(gl.POLYGON_OFFSET_FILL, s.offset);
  }

  // ---- plumbing
  #program(name, vs, fs) {
    let p = this.programs.get(name);
    if (p) return p;
    const gl = this.gl;
    const program = this.link(this.compile(gl.VERTEX_SHADER, vs), this.compile(gl.FRAGMENT_SHADER, fs));
    const u = {};
    for (const m of (vs + fs).matchAll(/uniform\s+(?:highp\s+|mediump\s+)?(?:\w+)\s+([^;]+);/g)) for (const n of m[1].split(",")) u[n.trim()] = gl.getUniformLocation(program, n.trim());
    p = { program, u }; this.programs.set(name, p);
    return p;
  }
  // RGBA16F targets, with or without a mip chain, with 1 or 2 colour attachments (level 0 is what is drawn into)
  #target(key, w, h, { mips = false, attachments = 1, linear = true } = {}) {
    const gl = this.gl;
    let t = this.targets.get(key);
    if (t && t.w === w && t.h === h) return t;
    if (t) { t.texes.forEach((x) => gl.deleteTexture(x)); gl.deleteFramebuffer(t.fbo); this.targets.delete(key); }
    const unit = gl.getParameter(gl.ACTIVE_TEXTURE), held = gl.getParameter(gl.TEXTURE_BINDING_2D);
    const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const texes = [];
    for (let i = 0; i < attachments; i++) {
      const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texStorage2D(gl.TEXTURE_2D, mips ? Math.floor(Math.log2(Math.max(w, h))) + 1 : 1, gl.RGBA16F, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mips ? gl.LINEAR_MIPMAP_NEAREST : linear ? gl.LINEAR : gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0);
      texes.push(tex);
    }
    gl.bindTexture(gl.TEXTURE_2D, held); gl.activeTexture(unit);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      texes.forEach((x) => gl.deleteTexture(x)); gl.deleteFramebuffer(fbo);
      throw new Unavailable(`a half-float render target is incomplete here (0x${status.toString(16)})`);
    }
    t = { w, h, fbo, texes, mips, attachments };
    this.targets.set(key, t);
    return t;
  }
  #mips(target, guard) {
    const gl = this.gl;
    target.texes.forEach((tex) => { guard.bind(0, tex); gl.generateMipmap(gl.TEXTURE_2D); });
  }
  // the map is as wide as the grains need (~2 texels across the median grain), a power of two in [256, 2048], sticky.  `base`: the mip level of
  // the median grain (0 at the 2048 cap, up to 3 on a small cloud), where the band ladder starts: the grain is the finest scale there is.
  #size(data, board, k) {
    const n = data.length / 5, radii = [];            // points are [x, y, z, rgba8, meta]
    for (let i = 0, step = Math.max(1, Math.floor(n / 256)); i < n; i += step) {
      const meta = data[i * 5 + 4];
      if (meta >= 50) radii.push((meta - Math.floor(meta / 100 + 0.001) * 100) * 0.001);
    }
    radii.sort((a, b) => a - b);
    const diameter = 2 * (radii[radii.length >> 1] || 0.01) * k;                     // board units
    const want = Math.log2(clamp(board[0] / (diameter / 2), 256, 2048));
    this.texW = this.texW && Math.abs(want - Math.log2(this.texW)) < 0.75 ? this.texW : 2 ** clamp(Math.round(want), 8, 11);
    const tw = Math.min(this.texW, this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE));
    return { tw, th: Math.max(64, Math.round(tw * board[1] / board[0])), base: clamp(Math.round(Math.log2(Math.max(1, diameter * tw / board[0]))), 1, 3) };
  }

  // ---- passes
  #splat(data, ctx, map, field, tw) {
    const gl = this.gl, p = this.#program("splat", SPLAT_VS, SPLAT_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, field.fbo); gl.viewport(0, 0, field.w, field.h);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.SCISSOR_TEST); gl.colorMask(true, true, true, true);
    gl.bindVertexArray(ctx.vao);
    gl.useProgram(p.program);
    gl.uniform4f(p.u.uMap, map[0], map[1], map[2], 0); gl.uniform3f(p.u.uBoard, ctx.board[0], ctx.board[1], ctx.y);
    gl.uniform1f(p.u.uTexel, tw / ctx.board[0]); gl.uniform1f(p.u.uMax, this.caps.maxPointSize); gl.uniform4fv(p.u.uTint, ctx.tint || [0, 0, 0, 0]);
    gl.drawArrays(gl.POINTS, 0, data.length / 5);
    gl.bindVertexArray(null);
  }
  // bands -> windowed variance -> fit; returns the target holding the result
  #analyse(field, tw, th, guard, { mode, strength, threshold, debug, base = 1 }) {
    const gl = this.gl;
    const bands = this.#target("bands", tw, th, { mips: true, attachments: 2 }), out = this.#target("out", tw, th, { mips: false, attachments: 1 });
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.SCISSOR_TEST); gl.colorMask(true, true, true, true);
    gl.bindVertexArray(this.emptyVao);
    const a = this.#program("bands", QUAD_VS, BANDS_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, bands.fbo); gl.viewport(0, 0, tw, th);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.useProgram(a.program); guard.bind(0, field.texes[0]); gl.uniform1i(a.u.uMap, 0); gl.uniform1f(a.u.uBase, base);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.NONE]);
    this.#mips(bands, guard);
    const b = this.#program("fit", QUAD_VS, FIT_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo); gl.viewport(0, 0, tw, th);
    gl.useProgram(b.program);
    guard.bind(0, field.texes[0]); guard.bind(1, bands.texes[0]); guard.bind(2, bands.texes[1]);
    gl.uniform1i(b.u.uMap, 0); gl.uniform1i(b.u.uV0, 1); gl.uniform1i(b.u.uV1, 2);
    gl.uniform1f(b.u.uMaxLod, Math.floor(Math.log2(Math.max(tw, th)))); gl.uniform1f(b.u.uBase, base); gl.uniform1f(b.u.uStrength, strength); gl.uniform1f(b.u.uThreshold, threshold);
    gl.uniform1i(b.u.uMode, mode); gl.uniform1i(b.u.uDebug, debug);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return out;
  }
  // the replacement on the board's plane, over what the point cloud drew (nearer than the points: a polygon offset)
  #overlay(out, ctx, guard, keep) {
    const gl = this.gl, p = this.#program("overlay", PLANE_VS, OVERLAY_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, keep.fbo); gl.viewport(keep.vp[0], keep.vp[1], keep.vp[2], keep.vp[3]);
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.u.uViewProj, false, ctx.viewProj); gl.uniform3f(p.u.uBoard, ctx.board[0], ctx.board[1], ctx.y); gl.uniform1f(p.u.uMargin, 0.0);
    guard.bind(0, out.texes[0]); gl.uniform1i(p.u.uOv, 0);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);
    gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(-2, -2);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

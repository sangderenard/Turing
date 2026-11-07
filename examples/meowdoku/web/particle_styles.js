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
//   stamp   a persistent texture over the board that every frame's points stamp sprites into and that decays slowly
//           (trails; the start of "sprites stamped into memory").  8-bit, so it needs no float targets.  The current
//           points are drawn over it by PointCloud's own passes.
//
// The pure part (the registry, what a device can run, which style is used instead) is at the top and is node-tested.

export const STYLES = [
  { id: "points", label: "points", needs: null },
  { id: "liquid", label: "liquid", needs: "float" },
  { id: "sand", label: "sand", needs: null, wants: "float" },     // without float targets: no ambient occlusion, still sand
  { id: "glow", label: "glow", needs: null },
  { id: "heat", label: "heat", needs: "float" },
  { id: "stamp", label: "stamp (memory)", needs: null },
];
export const STYLE_IDS = STYLES.map((s) => s.id);
export const DEFAULT_STYLE = "points";
const MIN_POINT_SIZE = 8;     // a sprite style needs point sprites at least this wide

// caps: { floatRender: "float" | "half" | null, maxPointSize: number } (see capabilitiesOf); null = no GL at all
export function styleAvailability(id, caps) {
  const style = STYLES.find((s) => s.id === id);
  if (!style) return { ok: false, note: `unknown style '${id}'` };
  if (id === DEFAULT_STYLE) return { ok: true, note: "" };
  if (!caps) return { ok: false, note: "no WebGL context" };
  if (!(caps.maxPointSize >= MIN_POINT_SIZE)) return { ok: false, note: `this device draws points only up to ${caps.maxPointSize || 0} px` };
  if (style.needs === "float" && !caps.floatRender) return { ok: false, note: "needs float or half-float render targets (EXT_color_buffer_float), which this device lacks" };
  if (style.wants === "float" && !caps.floatRender) return { ok: true, note: "no float render targets here: without the ambient-occlusion darkening" };
  if (style.needs === "float" && caps.floatRender === "half") return { ok: true, note: "half-float render targets (less depth precision)" };
  return { ok: true, note: "" };
}

// The style to draw: the requested one when the device can run it (and it has not failed at run time -- `broken` maps id -> reason),
// else "points", with the reason (shown in the control's title).
export function chooseStyle(requested, caps, broken = {}) {
  const id = STYLE_IDS.includes(requested) ? requested : DEFAULT_STYLE;
  if (id === DEFAULT_STYLE) return { style: id, requested: id, reason: "" };
  if (broken[id]) return { style: DEFAULT_STYLE, requested: id, reason: `${id} unavailable (${broken[id]}): showing points` };
  const av = styleAvailability(id, caps);
  if (!av.ok) return { style: DEFAULT_STYLE, requested: id, reason: `${id} unavailable (${av.note}): showing points` };
  return { style: id, requested: id, reason: av.note };
}

export function capabilitiesOf(gl) {
  const full = !!gl.getExtension("EXT_color_buffer_float"), half = !!gl.getExtension("EXT_color_buffer_half_float");
  return { floatRender: full ? "float" : half ? "half" : null, maxPointSize: gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] };
}

// ---------------------------------------------------------------- shaders
// One sprite vertex stage for every style: the same board -> world -> clip placement as compute.js's POINT_VS, over the same
// vertex layout (location 0 position, 1 colour, 2 meta), with the sprite sized in TARGET pixels (uFocal is already scaled).
const SPRITE_VS = `#version 300 es
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
  vColor = aColor; vRho = 1.0; vRw = 0.0; vEye = 0.0; vMoving = 0.0; vArea = 0.0; vREff = 1.0; vRpx = 0.0;
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
const QUAD_VS = `#version 300 es
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
// a quad over the board's plane (the board's rectangle, plus a margin), like compute.js's RESOLVE_VS
const PLANE_VS = `#version 300 es
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
const HEAT_FS = `#version 300 es
precision highp float; precision highp int;
uniform sampler2D uDens; uniform vec2 uViewport; out vec4 FragColor;
vec3 ramp(float t) {
  if (t < 0.25) return mix(vec3(0.03, 0.0, 0.18), vec3(0.38, 0.0, 0.48), t / 0.25);
  if (t < 0.5) return mix(vec3(0.38, 0.0, 0.48), vec3(0.86, 0.14, 0.14), (t - 0.25) / 0.25);
  if (t < 0.75) return mix(vec3(0.86, 0.14, 0.14), vec3(1.0, 0.72, 0.08), (t - 0.5) / 0.25);
  return mix(vec3(1.0, 0.72, 0.08), vec3(1.0, 1.0, 0.86), (t - 0.75) / 0.25);
}
void main() {
  vec4 a = texture(uDens, gl_FragCoord.xy / uViewport);
  float d = a.r, m = a.g / max(d, 1e-3);                              // density; the share of it that is moving
  if (d < 0.01) discard;
  float t = clamp(0.1 + 0.55 * m + 0.35 * smoothstep(0.0, 1.0, d), 0.0, 1.0);
  FragColor = vec4(ramp(t), smoothstep(0.01, 0.3, d) * 0.94);
}`;
// the memory: stamps are drawn in the board's own coordinates (the texture covers the board), so the camera may move
const STAMP_VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=2) in float aMeta;
uniform vec4 uMap; uniform vec3 uBoard; uniform float uTexel; uniform float uMax; uniform float uGrow;
out vec4 vColor;
void main() {
  vColor = aColor; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0;
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

class Unavailable extends Error {}

// Save a texture unit's 2D binding before using it, put it back after: the scene's passes share the units (a float texture
// left on unit 0 made the pbr material table read garbage and the sun and the bulbs rendered black).
class UnitGuard {
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
    const choice = chooseStyle(style, this.caps, this.broken);
    this.fallback = choice.style === style ? "" : choice.reason;
    this.note = choice.reason;
    if (choice.style === DEFAULT_STYLE || !ctx.camera || !ctx.camera.proj || !ctx.camera.view) return false;
    const guard = new UnitGuard(gl);
    const keep = { cull: gl.isEnabled(gl.CULL_FACE), scissor: gl.isEnabled(gl.SCISSOR_TEST) };
    gl.disable(gl.CULL_FACE); gl.disable(gl.SCISSOR_TEST);
    let done = true;
    try {
      gl.bindVertexArray(ctx.vao);
      if (choice.style === "liquid") this.#liquid(ctx, guard);
      else if (choice.style === "sand") this.#sand(ctx, guard);
      else if (choice.style === "glow") this.#glow(ctx, guard);
      else if (choice.style === "heat") this.#heat(ctx, guard);
      else if (choice.style === "stamp") { this.#stamp(ctx, guard); done = false; }
    } catch (error) {
      if (!(error instanceof Unavailable)) throw error;
      this.broken[choice.style] = error.message; this.fallback = `${choice.style} unavailable (${error.message}): showing points`;
      done = false;
    } finally {
      guard.restore();
      gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
      gl.blendEquation(gl.FUNC_ADD); gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true);
      if (keep.cull) gl.enable(gl.CULL_FACE);
      if (keep.scissor) gl.enable(gl.SCISSOR_TEST);
      gl.bindVertexArray(done ? null : ctx.vao);
    }
    return done;
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
  // the largest true radius among the first live points (world units): sizes the blur's range
  static typicalRadius(data) {
    let r = 0;
    for (let i = 0, k = Math.min(data.length / 5, 4096); i < k; i++) {         // points are [x, y, z, rgba8, meta]
      const m = data[5 * i + 4];
      if (m >= 50) r = Math.max(r, (m - Math.floor(m / 100 + 0.001) * 100) * 0.001);
    }
    return r || 0.02;
  }
  #quad(vao) { const gl = this.gl; gl.bindVertexArray(this.emptyVao); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); gl.bindVertexArray(vao); }

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
    const rPx = rWorld * ctx.focal * H / 2 * scale / Math.max(1e-3, Math.abs(eyeC));
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
  #density(ctx, guard, scale) {
    const gl = this.gl, { W, H, n } = ctx;
    const tw = Math.max(2, Math.ceil(W * scale)), th = Math.max(2, Math.ceil(H * scale));
    const T = this.#target("density", tw, th, { fmt: "rgba16f", linear: true });
    const p = this.#program("density", SPRITE_VS, DENSITY_FS);
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, tw, th); gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(p.program); this.#sprite(p, ctx, scale, { grow: 0, minPx: 16 * scale });
    gl.drawArrays(gl.POINTS, 0, n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, ctx.bound); gl.viewport(ctx.vp[0], ctx.vp[1], ctx.vp[2], ctx.vp[3]);
    gl.disable(gl.BLEND);
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

  // ---- heat
  #heat(ctx, guard) {
    const gl = this.gl;
    const T = this.#density(ctx, guard, ctx.n > 100000 ? 0.5 : 0.75);
    const p = this.#program("heat", PLANE_VS, HEAT_FS);
    gl.bindVertexArray(this.emptyVao);
    gl.useProgram(p.program);
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
    gl.useProgram(sp.program);
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

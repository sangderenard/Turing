// The games page: boards of beveled buttons and 3D pieces (glyph extrusions)
// lit by Pluck's base-material Phong shader as compiled for WebGL 2, with a
// depth pre-pass from an orbiting key light for shadows.  The page draws
// whatever state the Python server hands it (games.py's protocol) and
// animates the difference between states; it decides nothing.

import { TuringWasmProgram, TuringGraphProgram } from "./turing-wasm-shell.js";
import { ComputeRuntime, PointCloud } from "./compute.js";
import { decodeSnapshot, SNAPSHOT_PROTOCOL, FLAG_CLEAR, BodyFlight, BODY_CORES } from "./bodies.js";
import { shipStepper, PilotTimeline, pilotControls as intervalControls } from "./pilot_timeline.js";

const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
if (!gl) throw new Error("WebGL 2 is required");
const $ = (id) => document.getElementById(id);
const status = $("status");

// ---------------------------------------------------------------- matrices
const M = {
  identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k*4 + r] * b[c*4 + k]; o[c*4 + r] = s;
    }
    return o;
  },
  translate: (x, y, z) => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]),
  scale: (x, y, z) => new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]),
  rotateX(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); },
  rotateY(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); },
  rotateZ(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1]); },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  },
  ortho(half, near, far) {
    const m = M.identity(); m[0] = 1 / half; m[5] = 1 / half; m[10] = -2 / (far - near); m[14] = -(far + near) / (far - near); return m;
  },
  lookAt(eye, target, up) {
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye), -dot(y,eye), -dot(z,eye), 1]);
  },
  invert(m) {
    const inv = new Float32Array(16), a = m;
    inv[0] = a[5]*a[10]*a[15]-a[5]*a[11]*a[14]-a[9]*a[6]*a[15]+a[9]*a[7]*a[14]+a[13]*a[6]*a[11]-a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15]+a[4]*a[11]*a[14]+a[8]*a[6]*a[15]-a[8]*a[7]*a[14]-a[12]*a[6]*a[11]+a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15]-a[4]*a[11]*a[13]-a[8]*a[5]*a[15]+a[8]*a[7]*a[13]+a[12]*a[5]*a[11]-a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]+a[4]*a[10]*a[13]+a[8]*a[5]*a[14]-a[8]*a[6]*a[13]-a[12]*a[5]*a[10]+a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15]+a[1]*a[11]*a[14]+a[9]*a[2]*a[15]-a[9]*a[3]*a[14]-a[13]*a[2]*a[11]+a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15]-a[0]*a[11]*a[14]-a[8]*a[2]*a[15]+a[8]*a[3]*a[14]+a[12]*a[2]*a[11]-a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15]+a[0]*a[11]*a[13]+a[8]*a[1]*a[15]-a[8]*a[3]*a[13]-a[12]*a[1]*a[11]+a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14]-a[0]*a[10]*a[13]-a[8]*a[1]*a[14]+a[8]*a[2]*a[13]+a[12]*a[1]*a[10]-a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15]-a[1]*a[7]*a[14]-a[5]*a[2]*a[15]+a[5]*a[3]*a[14]+a[13]*a[2]*a[7]-a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15]+a[0]*a[7]*a[14]+a[4]*a[2]*a[15]-a[4]*a[3]*a[14]-a[12]*a[2]*a[7]+a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15]-a[0]*a[7]*a[13]-a[4]*a[1]*a[15]+a[4]*a[3]*a[13]+a[12]*a[1]*a[7]-a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]+a[0]*a[6]*a[13]+a[4]*a[1]*a[14]-a[4]*a[2]*a[13]-a[12]*a[1]*a[6]+a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11]+a[1]*a[7]*a[10]+a[5]*a[2]*a[11]-a[5]*a[3]*a[10]-a[9]*a[2]*a[7]+a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11]-a[0]*a[7]*a[10]-a[4]*a[2]*a[11]+a[4]*a[3]*a[10]+a[8]*a[2]*a[7]-a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]+a[0]*a[7]*a[9]+a[4]*a[1]*a[11]-a[4]*a[3]*a[9]-a[8]*a[1]*a[7]+a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10]-a[0]*a[6]*a[9]-a[4]*a[1]*a[10]+a[4]*a[2]*a[9]+a[8]*a[1]*a[6]-a[8]*a[2]*a[5];
    const det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    for (let i = 0; i < 16; i++) inv[i] /= det;
    return inv;
  },
  apply(m, v) { const w = m[3]*v[0]+m[7]*v[1]+m[11]*v[2]+m[15]; return [(m[0]*v[0]+m[4]*v[1]+m[8]*v[2]+m[12])/w, (m[1]*v[0]+m[5]*v[1]+m[9]*v[2]+m[13])/w, (m[2]*v[0]+m[6]*v[1]+m[10]*v[2]+m[14])/w]; },
};
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
const mix = (a, b, t) => a.map((v, i) => v * (1 - t) + b[i] * t);
function rgbToHsv([r, g, b]) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
  let h = 0; if (d > 1e-6) { h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4; h /= 6; if (h < 0) h += 1; }
  return [h, max > 1e-6 ? d / max : 0, max];
}
function hsvToRgb([h, s, v]) {
  const i = Math.floor(((h % 1) + 1) % 1 * 6), f = ((h % 1) + 1) % 1 * 6 - i;
  const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
}

// ---------------------------------------------------------------- shaders
function compile(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
// THE SKY (a spherical game): single-scattering Rayleigh + Mie along each
// view ray, the sun's disc, and stars where the sky is dark -- sunrise and
// sunset from the same atmosphere that colours the Phong scene's sunlight.
// Drawn first, behind everything, without depth.
const SKY_VS = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vNdc;
void main() { vNdc = P[gl_VertexID]; gl_Position = vec4(P[gl_VertexID], 0.999, 1.0); }`;
const SKY_FS = `#version 300 es
precision highp float;
in vec2 vNdc; out vec4 FragColor;
uniform mat4 uInvViewProj; uniform vec3 uSun; uniform vec3 uTauR; uniform float uTauM; uniform float uExposure;
float airMass(float e) { float d = max(-2.5, degrees(e)); return 1.0 / (sin(radians(max(d, 0.0))) + 0.50572 * pow(d + 6.07995, -1.6364)); }
float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
void main() {
  vec4 a = uInvViewProj * vec4(vNdc, -1.0, 1.0), b = uInvViewProj * vec4(vNdc, 1.0, 1.0);
  vec3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
  float view_e = asin(clamp(dir.y, -1.0, 1.0)), sun_e = asin(clamp(uSun.y, -1.0, 1.0));
  float mu = dot(dir, uSun);
  vec3 tau = uTauR + vec3(uTauM);
  vec3 sunlight = exp(-tau * airMass(sun_e)) * clamp((degrees(sun_e) + 3.0) / 6.0, 0.0, 1.0);
  float phaseR = 0.0596831 * (1.0 + mu * mu);
  float g = 0.76, phaseM = 0.0795775 * (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5);
  float mv = airMass(max(view_e, 0.0));
  vec3 inscatter = (uTauR * phaseR + vec3(uTauM) * phaseM) / tau * (1.0 - exp(-tau * mv));
  vec3 sky = 18.0 * sunlight * inscatter;
  // twilight: the sky glows on after the sun is down
  sky += vec3(0.02, 0.025, 0.05) * clamp((degrees(sun_e) + 12.0) / 12.0, 0.0, 1.0) * (1.0 - clamp(view_e * 3.0, 0.0, 1.0));
  // the sun's disc and its glare
  sky += sunlight * (smoothstep(0.99985, 0.99995, mu) * 60.0 + pow(max(mu, 0.0), 400.0) * 3.0);
  // below the horizon: the haze over the far land, darkening downward
  if (dir.y < 0.0) sky = mix(sky, vec3(0.03, 0.035, 0.05) * (0.3 + sunlight), clamp(-dir.y * 4.0, 0.0, 1.0));
  // stars where the sky is dark
  float dark = 1.0 - clamp(dot(sky, vec3(0.3, 0.5, 0.2)) * 6.0, 0.0, 1.0);
  vec3 cell = floor(dir * 220.0);
  float star = step(0.9965, hash(cell)) * dark * step(0.0, dir.y);
  sky += vec3(star * (0.6 + 0.4 * hash(cell + 7.0)));
  vec3 mapped = vec3(1.0) - exp(-sky * uExposure * 2.4);
  FragColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}`;
function drawSky(cam, now) {
  if (state.flat || !state.game || !state.game.globe || !SKY.dir) return;
  if (!SKY.prog) { SKY.prog = link(compile(gl.VERTEX_SHADER, SKY_VS), compile(gl.FRAGMENT_SHADER, SKY_FS)); SKY.vao = gl.createVertexArray(); }
  gl.useProgram(SKY.prog); gl.bindVertexArray(SKY.vao);
  gl.uniformMatrix4fv(gl.getUniformLocation(SKY.prog, "uInvViewProj"), false, M.invert(cam.viewProj));
  gl.uniform3fv(gl.getUniformLocation(SKY.prog, "uSun"), new Float32Array(SKY.dir));
  gl.uniform3fv(gl.getUniformLocation(SKY.prog, "uTauR"), new Float32Array(SKY.tauR));
  gl.uniform1f(gl.getUniformLocation(SKY.prog, "uTauM"), SKY.tauM);
  gl.uniform1f(gl.getUniformLocation(SKY.prog, "uExposure"), state.debug.exposure);
  gl.depthMask(false); gl.disable(gl.DEPTH_TEST);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
  gl.bindVertexArray(null);
}
// CITY LIGHTS: each pin a glowing point, added up where lights crowd --
// not meshes, not lamps that light the land: points that read through cloud
// (drawn after the glass, depth-tested only against the land), brighter the
// darker the local sky.
// Like the night side seen from orbit: tiny hard points, each with its own
// colour (sodium amber to cool white, by a hash of where it stands), no blur.
const GLOW_VS = `#version 300 es
layout(location=0) in vec4 aPoint;      // x y z, light
uniform mat4 uViewProj;
out float vLight; out vec3 vTint;
void main() { vLight = aPoint.w; gl_Position = uViewProj * vec4(aPoint.xyz, 1.0);
  float h = fract(sin(dot(aPoint.xz, vec2(12.9898, 78.233))) * 43758.5453);
  vTint = mix(vec3(1.0, 0.62, 0.22), vec3(0.92, 0.95, 1.0), h * h);
  gl_PointSize = vLight > 0.6 ? 2.0 : 1.0; }`;
const GLOW_FS = `#version 300 es
precision highp float;
in float vLight; in vec3 vTint; out vec4 FragColor; uniform float uNight;
void main() { FragColor = vec4(vTint * (0.55 + 0.75 * vLight) * uNight, 1.0); }`;
const GLOW = { prog: null, vao: null, buf: null };
function drawCityLights(cam) {
  const pts = state.glowPoints || [];
  if (!pts.length || state.flat) return;
  if (!GLOW.prog) {
    GLOW.prog = link(compile(gl.VERTEX_SHADER, GLOW_VS), compile(gl.FRAGMENT_SHADER, GLOW_FS));
    GLOW.vao = gl.createVertexArray(); gl.bindVertexArray(GLOW.vao);
    GLOW.buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, GLOW.buf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }
  // each pin is a district: up to CLUSTER tiny lights scattered round it (a
  // fixed hash pattern, so they hold still), more the brighter it is
  const CLUSTER = 9, data = new Float32Array(pts.length * CLUSTER * 4);
  let n = 0;
  for (const p of pts) {
    const light = Math.min(1, Math.max(0, (p[3] - 0.06) / 0.06));
    const count = 2 + Math.round(light * (CLUSTER - 2));
    for (let j = 0; j < count; j++) {
      const s = Math.sin((p[0] * 12.99 + p[2] * 78.23 + j * 37.7)) * 43758.5453, r1 = s - Math.floor(s);
      const s2 = Math.sin((p[0] * 39.35 + p[2] * 11.13 + j * 91.3)) * 24634.6345, r2 = s2 - Math.floor(s2);
      const ang = r1 * 6.2832, rad = 0.05 + 0.22 * Math.sqrt(r2);
      data.set([p[0] + Math.cos(ang) * rad, p[1] + 0.03, p[2] + Math.sin(ang) * rad, light * (0.5 + 0.5 * r2)], n * 4);
      n++;
    }
  }
  const sunUp = SKY.dir ? SKY.dir[1] : -1;
  const night = Math.max(0.12, Math.min(1, (0.12 - sunUp) / 0.3));
  gl.useProgram(GLOW.prog); gl.bindVertexArray(GLOW.vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, GLOW.buf); gl.bufferData(gl.ARRAY_BUFFER, data.subarray(0, n * 4), gl.DYNAMIC_DRAW);
  gl.uniformMatrix4fv(gl.getUniformLocation(GLOW.prog, "uViewProj"), false, cam.viewProj);
  gl.uniform1f(gl.getUniformLocation(GLOW.prog, "uNight"), night);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); gl.depthMask(false);
  gl.drawArrays(gl.POINTS, 0, n);
  gl.depthMask(true); gl.disable(gl.BLEND); gl.bindVertexArray(null);
}
function link(vs, fs) {
  const p = gl.createProgram(); gl.attachShader(p, vs); gl.attachShader(p, fs); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const FLAT_VS = `#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aColor;
uniform mat4 uMVP; out vec3 vColor;
void main(){ vColor = aColor; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const FLAT_FS = `#version 300 es
precision highp float; in vec3 vColor; out vec4 FragColor; uniform float uAlpha; uniform vec3 uTint;
void main(){ FragColor = vec4(pow(vColor * uTint, vec3(1.0/2.2)), uAlpha); }`;
// Flat view: pure cel shading.  The key light sits at the camera and shines
// straight down, so N.z in view space IS the lighting; it is cut into a few
// hard bands over the palette colour of each vertex's material.  The palette
// (uPal) is indexed by the same material id the Phong program reads.
const CEL_PALETTE = 192;
const CEL_VS = `#version 300 es
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNorm; layout(location=2) in int aMat;
uniform mat4 uMVP; uniform mat4 uMV; flat out int vMat; out vec3 vN;
// Soft-body free-form deformation: a 2x2x2 lattice of control-point offsets (in
// units of the mesh's half-extent, corner i = x + 2y + 4z) displaces each vertex
// by their trilinear blend, which is what makes twist (and not only shear) show.
uniform int uFfd; uniform vec3 uCage[8]; uniform vec3 uCtr; uniform vec3 uHalf;
void main(){
  vec3 pos = aPos, n = aNorm;
  if (uFfd == 1) {
    vec3 u = clamp((aPos - uCtr) / uHalf * 0.5 + 0.5, 0.0, 1.0);
    vec3 o = vec3(0.0), dx = vec3(0.0), dy = vec3(0.0), dz = vec3(0.0);
    for (int i = 0; i < 8; i++) {
      float ix = float(i & 1), iy = float((i >> 1) & 1), iz = float((i >> 2) & 1);
      float wx = mix(1.0 - u.x, u.x, ix), wy = mix(1.0 - u.y, u.y, iy), wz = mix(1.0 - u.z, u.z, iz);
      o += wx * wy * wz * uCage[i];
      dx += (2.0 * ix - 1.0) * wy * wz * uCage[i];
      dy += wx * (2.0 * iy - 1.0) * wz * uCage[i];
      dz += wx * wy * (2.0 * iz - 1.0) * uCage[i];
    }
    pos = aPos + o * uHalf;
    // Jacobian columns in mesh space; the normal follows the cofactor (inverse transpose).
    vec3 a = (vec3(1.0, 0.0, 0.0) + 0.5 * dx) * uHalf / uHalf.x;
    vec3 b = (vec3(0.0, 1.0, 0.0) + 0.5 * dy) * uHalf / uHalf.y;
    vec3 c = (vec3(0.0, 0.0, 1.0) + 0.5 * dz) * uHalf / uHalf.z;
    vec3 nn = cross(b, c) * aNorm.x + cross(c, a) * aNorm.y + cross(a, b) * aNorm.z;
    if (dot(nn, nn) > 1e-12) n = normalize(nn);
  }
  vMat = aMat; vN = mat3(uMV) * n; gl_Position = uMVP * vec4(pos, 1.0);
}`;
const CEL_FS = `#version 300 es
precision highp float; in vec3 vN; flat in int vMat; out vec4 FragColor;
uniform vec3 uPal[${CEL_PALETTE}]; uniform int uOutline;
void main(){
  if (uOutline == 1) { FragColor = vec4(0.03, 0.03, 0.04, 1.0); return; }
  // the extruded walls are wound inward (see uploadMaterials), so flip on back faces
  vec3 N = normalize(gl_FrontFacing ? vN : -vN);
  float t = N.z;
  float band = t > 0.93 ? 1.0 : t > 0.70 ? 0.80 : t > 0.40 ? 0.62 : 0.46;
  vec3 c = uPal[clamp(vMat, 0, ${CEL_PALETTE - 1})];
  FragColor = vec4(pow(c * band, vec3(1.0 / 1.6)), 1.0);
}`;
// THE LENS: a planet game is rendered to an offscreen frame, then drawn to the
// screen through a gentle hyperbolic mapping -- each screen point samples the
// frame at radius atanh(r tanh k) / k, magnifying the centre and compressing
// the rim (k from the Curve slider; 0 is off).  Our own pass; the Phong
// shader and the scene are untouched.
const LENS_VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const LENS_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uFrame;
uniform float uK;
uniform vec2 uAspect;
out vec4 fragColor;
float atanh_(float x) { return 0.5 * log((1.0 + x) / (1.0 - x)); }
void main() {
  vec2 d = (vUv - 0.5) * uAspect;
  float r = length(d);
  float scale = 1.0;
  if (uK > 1e-4 && r > 1e-5) {
    float t = tanh(uK);
    scale = atanh_(clamp(r * t / 0.7071, 0.0, 0.9999)) / uK * 0.7071 / r;
  }
  vec2 uv = 0.5 + (d * scale) / uAspect;
  fragColor = (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) ? vec4(0.085, 0.085, 0.1, 1.0) : texture(uFrame, uv);
}`;
const lens = { fbo: null, tex: null, depth: null, w: 0, h: 0, program: null, vao: null };
// THE PROJECTION GLASS: a glass square hovering above the field shows the
// game as a wide-angle camera straight overhead sees it, through the
// hyperbolic lens -- from straight above you watch the broadcast; tilt and
// you see under the glass to the game playing out for real.  The broadcast
// is rendered every frame into colour layer BROADCAST_LAYER of the Phong
// shader's uColorUv stack, and the glass's material takes its albedo there.
const BROADCAST = { size: STACK_SIZE_LATE(), fov: 1.4, fbo: null, tex: null, depth: null, warp: null, quad: null, key: "" };
function STACK_SIZE_LATE() { return 512; }
function projectionGlassOn() { return onPlanet() && viewerWants("projection") && !!state.colourStack; }
function broadcastSpan() { return Math.max(state.extent.width, state.extent.depth) * 1.12; }
function renderBroadcast(now, casters) {
  const B = BROADCAST, n = B.size;
  if (!B.fbo) {
    B.tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE14); gl.bindTexture(gl.TEXTURE_2D, B.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    B.depth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, B.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, n, n);
    B.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, B.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, B.tex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, B.depth);
    B.warp = gl.createFramebuffer();
    lensTarget(1, 1);                                   // compiles the lens program
  }
  // the broadcast camera: straight down over the field's centre, wide angle
  const S = broadcastSpan(), height = (S / 2) / Math.tan(B.fov / 2);
  const target = [state.extent.width / 2, 0, state.extent.depth / 2], eye = [target[0], height, target[2]];
  const view = M.lookAt(eye, target, [0, 0, -1]), proj = M.perspective(B.fov, 1, 0.1, 300);
  gl.bindFramebuffer(gl.FRAMEBUFFER, B.fbo); gl.viewport(0, 0, n, n);
  gl.clearColor(0.085, 0.085, 0.1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  litPass({ view, proj, viewProj: M.multiply(proj, view), eye }, now, casters, { plate: false });
  // warp it into the glass's colour layer
  gl.bindFramebuffer(gl.FRAMEBUFFER, B.warp);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, state.colourStack, 0, BROADCAST_LAYER);
  gl.viewport(0, 0, n, n); gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  gl.useProgram(lens.program);
  gl.activeTexture(gl.TEXTURE14); gl.bindTexture(gl.TEXTURE_2D, B.tex);
  gl.uniform1i(gl.getUniformLocation(lens.program, "uFrame"), 14);
  gl.uniform1f(gl.getUniformLocation(lens.program, "uK"), Math.max(0.35, Number($("curve").value) / 100 * 1.2));
  gl.uniform2f(gl.getUniformLocation(lens.program, "uAspect"), 1, 1);
  gl.bindVertexArray(lens.vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
  gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, state.colourStack); gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  gl.enable(gl.DEPTH_TEST);
}
function drawProjectionGlass(cam, drawLit) {
  // Sized so that, seen from the main camera straight overhead, it covers
  // exactly the broadcast's footprint on the board; hovering above everything.
  const S = broadcastSpan(), y = TILE.height + 4.0 * DEPTH_LIFT + 1.2;
  const eyeHeight = Math.max(y + 0.5, cam.eye ? cam.eye[1] : y + 10);
  const side = S * (eyeHeight - y) / eyeHeight, cx = state.extent.width / 2, cz = state.extent.depth / 2;
  if (!BROADCAST.quad) {
    const pos = new Float32Array([-0.5, 0, -0.5, 0.5, 0, 0.5, 0.5, 0, -0.5, -0.5, 0, -0.5, -0.5, 0, 0.5, 0.5, 0, 0.5]);
    const nrm = new Float32Array(18); for (let i = 0; i < 6; i++) nrm[i * 3 + 1] = 1;
    // image top (v = 1) is the far side (-z), as the broadcast camera looks with -z up
    const uv = new Float32Array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0]);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    for (const [data, loc, size] of [[pos, 0, 3], [nrm, 1, 3], [uv, 3, 2]]) {
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    }
    gl.bindVertexArray(null);
    BROADCAST.quad = { vao, count: 6 };
  }
  drawLit(BROADCAST.quad.vao, BROADCAST.quad.count, M.multiply(M.translate(cx, y, cz), M.scale(side, 1, side)),
          state.materialIndex.broadcast);
}
function lensStrength() { const s = $("curve"); return onPlanet() && s ? Number(s.value) / 100 * 1.2 : 0; }
function lensTarget(w, h) {
  if (!lens.program) { lens.program = link(compile(gl.VERTEX_SHADER, LENS_VS), compile(gl.FRAGMENT_SHADER, LENS_FS)); lens.vao = gl.createVertexArray(); }
  if (lens.w !== w || lens.h !== h) {
    if (lens.fbo) { gl.deleteFramebuffer(lens.fbo); gl.deleteTexture(lens.tex); gl.deleteRenderbuffer(lens.depth); }
    lens.tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, lens.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    lens.depth = gl.createRenderbuffer(); gl.bindRenderbuffer(gl.RENDERBUFFER, lens.depth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
    lens.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, lens.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, lens.tex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, lens.depth);
    lens.w = w; lens.h = h;
  }
  return lens;
}
function lensPresent(w, h, k) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h);
  gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  gl.useProgram(lens.program);
  gl.activeTexture(gl.TEXTURE15); gl.bindTexture(gl.TEXTURE_2D, lens.tex);
  gl.uniform1i(gl.getUniformLocation(lens.program, "uFrame"), 15);
  gl.uniform1f(gl.getUniformLocation(lens.program, "uK"), k);
  gl.uniform2f(gl.getUniformLocation(lens.program, "uAspect"), w >= h ? w / h : 1, w >= h ? 1 : h / w);
  gl.bindVertexArray(lens.vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null);
  gl.enable(gl.DEPTH_TEST);
}
const DEPTH_VS = `#version 300 es
layout(location=0) in vec3 aPos; uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos, 1.0); }`;
const DEPTH_FS = `#version 300 es
precision highp float; void main(){}`;
const SHADOW_SIZE = 2048;

// ---------------------------------------------------------------- GL resources
function storageTexture(table, unit) {
  const data = new Float32Array(table.data);
  const width = 256, height = Math.max(1, Math.ceil(data.length / width));
  const padded = new Float32Array(width * height); padded.set(data);
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded);
  return tex;
}
function arrayTexture(unit, rgba) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA8, 1, 1, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([...rgba, ...rgba]));
  return tex;
}
// ---------------------------------------------------------------- scrolling terrain
// A procedural height map (tileable fractal value noise), shown through the
// Phong shader's colour texture stack: albedo by height (water, sand, grass,
// rock, snow), hillshaded from the map's own slopes so the relief reads.  The
// ground quad's UVs advance every frame (TERRAIN.speed board units a second),
// rolling the land beneath a game that declares a "terrain" layer.
const TERRAIN = { size: 256, layer: 0, tile: 6.0, speed: 2.4, scroll: 0, mesh: null, at: null };
function terrainHeights(n, seed = 7) {
  const lattice = (period, salt) => {
    const g = new Float32Array(period * period);
    for (let i = 0; i < g.length; i++) { const s = Math.sin((i + 1) * 12.9898 + salt * 78.233 + seed) * 43758.5453; g[i] = s - Math.floor(s); }
    return g;
  };
  const height = new Float32Array(n * n);
  let amp = 1.0, total = 0;
  for (let octave = 0, period = 4; octave < 6; octave++, period *= 2, amp *= 0.5) {
    const g = lattice(period, octave), cell = n / period;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const fx = x / cell, fy = y / cell, x0 = Math.floor(fx), y0 = Math.floor(fy);
      const tx = fx - x0, ty = fy - y0, sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
      const at = (a, b) => g[((b % period) * period) + (a % period)];
      const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx, bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx;
      height[y * n + x] += amp * (top + (bottom - top) * sy);
    }
    total += amp;
  }
  for (let i = 0; i < height.length; i++) height[i] /= total;
  return height;
}
function terrainTexture(unit, exported) {
  // The exported terrain (web_export: Pluck's DepthMesh relief + its colours)
  // when present; else the page's own flat procedural map.
  if (exported && exported.texture) {
    const n = exported.texture.size, raw = atob(exported.texture.rgba), rgba = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) rgba[i] = raw.charCodeAt(i);
    return uploadTerrainTexture(unit, n, rgba);
  }
  const n = TERRAIN.size, h = terrainHeights(n), rgba = new Uint8Array(n * n * 4);
  const ramp = [[0.00, [0.10, 0.22, 0.42]], [0.40, [0.20, 0.42, 0.62]], [0.44, [0.78, 0.72, 0.48]],
                [0.50, [0.32, 0.58, 0.26]], [0.62, [0.22, 0.44, 0.20]], [0.72, [0.46, 0.40, 0.34]], [0.84, [0.92, 0.93, 0.96]]];
  const colourOf = (v) => {
    for (let i = 1; i < ramp.length; i++) if (v <= ramp[i][0]) {
      const [a, ca] = ramp[i - 1], [b, cb] = ramp[i], t = (v - a) / (b - a);
      return ca.map((c, k) => c + (cb[k] - c) * t);
    }
    return ramp[ramp.length - 1][1];
  };
  const light = norm([-0.55, 0.75, -0.35]);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const v = h[y * n + x], water = v < 0.42;
    const dx = (h[y * n + (x + 1) % n] - h[y * n + (x + n - 1) % n]) * 18.0;
    const dy = (h[((y + 1) % n) * n + x] - h[((y + n - 1) % n) * n + x]) * 18.0;
    const shade = water ? 0.95 : Math.max(0.35, dot(norm([-dx, 1.0, -dy]), light) * 1.15);
    const c = colourOf(water ? Math.min(v, 0.41) : v);
    const o = (y * n + x) * 4;
    rgba[o] = Math.min(255, c[0] * shade * 255); rgba[o + 1] = Math.min(255, c[1] * shade * 255);
    rgba[o + 2] = Math.min(255, c[2] * shade * 255); rgba[o + 3] = 255;
  }
  return uploadTerrainTexture(unit, n, rgba);
}
function depthTexture(unit, exported) {
  const n = exported.size, raw = atob(exported.rgba), rgba = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) rgba[i] = raw.charCodeAt(i);
  return uploadTerrainTexture(unit, n, rgba);
}
// The colour stack holds two layers of STACK_SIZE: 0 the terrain (upsampled
// once here), 1 the projection glass's broadcast (rendered every frame).
const STACK_SIZE = 1024, BROADCAST_LAYER = 1;
function upsample(rgba, n, size) {
  if (n === size) return rgba;
  const out = new Uint8Array(size * size * 4), s = n / size;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const fx = x * s, fy = y * s, x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0;
    const x1 = (x0 + 1) % n, y1 = (y0 + 1) % n;
    for (let c = 0; c < 4; c++) {
      const a = rgba[(y0 * n + x0) * 4 + c], b = rgba[(y0 * n + x1) * 4 + c], d = rgba[(y1 * n + x0) * 4 + c], e = rgba[(y1 * n + x1) * 4 + c];
      out[(y * size + x) * 4 + c] = (a + (b - a) * tx) * (1 - ty) + (d + (e - d) * tx) * ty;
    }
  }
  return out;
}
function uploadTerrainTexture(unit, n, rgba) {
  if (unit === 5) {
    const layers = new Uint8Array(STACK_SIZE * STACK_SIZE * 4 * 2);
    layers.set(upsample(rgba, n, STACK_SIZE), 0);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA8, STACK_SIZE, STACK_SIZE, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, layers);
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
    state.colourStack = tex;
    return tex;
  }
  const tex = gl.createTexture();
  if (unit === 6) state.depthStack = { tex, n };
  gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA8, n, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
  gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
  return tex;
}
// The ground quad: positions, an up normal, and a UV buffer rewritten as it scrolls.
function terrainMesh(width, depth) {
  const m = 2.0, x0 = -m, x1 = width + m, z0 = -m, z1 = depth + m, y = -0.02;
  const pos = new Float32Array([x0, y, z0, x1, y, z1, x1, y, z0, x0, y, z0, x0, y, z1, x1, y, z1]);
  const nrm = new Float32Array(18); for (let i = 0; i < 6; i++) nrm[i * 3 + 1] = 1;
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buffer = (data, loc, size) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); return b; };
  buffer(pos, 0, 3); buffer(nrm, 1, 3);
  const uvBuffer = buffer(new Float32Array(12), 3, 2);
  gl.bindVertexArray(null);
  return { vao, count: 6, uvBuffer, corners: [[x0, z0], [x1, z1], [x1, z0], [x0, z0], [x0, z1], [x1, z1]], key: `${width}x${depth}` };
}
// The relief mesh exported from Pluck's DepthMesh: positions, normals and UVs
// fixed; the page only slides it along the board by the scroll, wrapping at
// one tile (the height map is tileable), so mountains pass beneath in true
// perspective, lit and shaded by the Phong lights through their own normals.
// SMALL PLANET.  The land lies on a sphere of radius PLANET.radius centred
// under the middle of the field.  Across the width the curve is baked into
// the mesh at load (it never moves sideways); along the length the mesh is
// cut into PLANET.strip-long strips, and each frame every strip is dropped by
// d^2 / 2R for its distance d from the field's centre and tilted to the
// sphere's slope there (atan(d / R)) -- so the land rises over the near
// horizon, passes beneath, and falls away beyond, shader untouched, no
// per-vertex work per frame.
// Strips are cut ONLY on grid-row boundaries (a whole number of the mesh's
// rows): a strip boundary through the middle of a row split its triangles
// between two strips moved differently -- the stair-step tears and gaps.
// Two 0.2 rows per visible strip, one 0.6 row per shadow strip; the rigid
// tilt then leaves (L/2)^2 / 2R ~ 0.0006 units at a seam.
const PLANET = { radius: 36.0, strip: 0.4, shadowStrip: 0.6 };
// The field lies on the same sphere as the land: everything drawn over a
// planet-terrain game drops by the curve at its own board position, so the
// play layer stays parallel to the surface it flies over (a flat field over
// curved land looked like the land breaching the clouds in the middle).
function onPlanet() { const g = state.game; return !state.flat && !!(g && g.scene && g.scene.terrain === true); }
function planetDrop(x, z) {
  if (!onPlanet()) return 0;
  const dx = x - state.extent.width / 2, dz = z - state.extent.depth / 2;
  return (dx * dx + dz * dz) / (2 * PLANET.radius);
}
function stripsOf(positions, extras, span, length = PLANET.strip) {
  // Partition a triangle soup by each triangle's centroid z into strips,
  // positions re-centred on the strip (z) and curved across the width (y).
  const tri = positions.length / 9, strips = new Map(), cx = span[0] / 2;
  for (let k = 0; k < tri; k++) {
    const zc = (positions[k * 9 + 2] + positions[k * 9 + 5] + positions[k * 9 + 8]) / 3;
    const index = Math.floor(zc / length);
    if (!strips.has(index)) strips.set(index, { index, pos: [], extra: extras.map(() => []) });
    const s = strips.get(index), centre = (index + 0.5) * length;
    for (let v = 0; v < 3; v++) {
      const o = k * 9 + v * 3, x = positions[o], dx = x - cx;
      s.pos.push(x, positions[o + 1] - (dx * dx) / (2 * PLANET.radius), positions[o + 2] - centre);
    }
    extras.forEach(([data, size], e) => { for (let v = 0; v < 3; v++) for (let c = 0; c < size; c++) s.extra[e].push(data[(k * 3 + v) * size + c]); });
  }
  return [...strips.values()].map((s) => ({ ...s, centre: (s.index + 0.5) * length }));
}
function reliefMesh(exported) {
  if (TERRAIN.relief !== undefined) return TERRAIN.relief;
  if (!exported || !exported.positions) return (TERRAIN.relief = null);
  // Buffers stay writable: a spherical game re-reads the relief under a
  // turned, travelling coordinate every frame (turnRelief).
  const buffer = (data, loc, size) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); return b; };
  // Each vertex's node on the mesh's grid (the DepthMesh grid it was read on).
  const nodesOf = (s, step) => {
    const n = s.pos.length / 3, node = new Int32Array(n), cols = Math.round(exported.span[0] / step) + 1;
    for (let v = 0; v < n; v++) node[v] = Math.round((s.pos[v * 3 + 2] + s.centre) / step) * cols + Math.round(s.pos[v * 3] / step);
    return node;
  };
  const strips = stripsOf(exported.positions, [[exported.normals, 3], [exported.uvs, 2]], exported.span).map((s) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const pos = new Float32Array(s.pos), nrm = new Float32Array(s.extra[0]), uv = new Float32Array(s.extra[1]);
    const buffers = [buffer(pos, 0, 3), buffer(nrm, 1, 3), buffer(uv, 3, 2)];
    gl.bindVertexArray(null);
    return { vao, count: s.pos.length / 3, centre: s.centre, pos, nrm, uv, buffers,
             node: exported.step ? nodesOf(s, exported.step) : null };
  });
  const shadows = exported.shadow_positions ? stripsOf(exported.shadow_positions, [], exported.span, PLANET.shadowStrip).map((s) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const pos = new Float32Array(s.pos), buffers = [buffer(pos, 0, 3)];
    gl.bindVertexArray(null);
    return { vao, count: s.pos.length / 3, centre: s.centre, pos, buffers,
             node: exported.shadow_step ? nodesOf(s, exported.shadow_step) : null };
  }) : [];
  // The relief map itself (the depth texture's R channel), through the
  // export's height law, for re-reading.
  let map = null;
  if (exported.depth && exported.depth.rgba) {
    const raw = atob(exported.depth.rgba), n = exported.depth.size;
    map = { n, h: new Float32Array(n * n) };
    for (let i = 0; i < n * n; i++) map.h[i] = Math.pow(raw.charCodeAt(i * 4) / 255, exported.relief_power || 1.4) * (exported.relief || 0.75);
  }
  // THE SURFACE: the whole relief as ONE mesh (no strips, so no seams):
  // every vertex is placed in board coordinates each frame -- its height read
  // at the turned map point, less the planet's drop -- and drawn untransformed.
  const surfaceOf = (positions, step, withShading) => {
    if (!positions || !step) return null;
    const n = positions.length / 3, cols = Math.round(exported.span[0] / step) + 1;
    const pos = new Float32Array(positions), node = new Int32Array(n);
    for (let v = 0; v < n; v++) node[v] = Math.round(pos[v * 3 + 2] / step) * cols + Math.round(pos[v * 3] / step);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const buffers = [buffer(pos, 0, 3)];
    let nrm = null, uv = null;
    if (withShading) {
      nrm = new Float32Array(exported.normals); uv = new Float32Array(exported.uvs);
      buffers.push(buffer(nrm, 1, 3), buffer(uv, 3, 2));
    }
    gl.bindVertexArray(null);
    return { vao, count: n, base: new Float32Array(positions), pos, nrm, uv, node, buffers };
  };
  const surface = surfaceOf(exported.positions, exported.step, true);
  const shadowSurface = surfaceOf(exported.shadow_positions, exported.shadow_step, false);
  return (TERRAIN.relief = { strips, shadows, tile: exported.tile, span: exported.span, step: exported.step,
                             shadowStep: exported.shadow_step, map, surface, shadowSurface });
}
// TURNING THE LAND (a spherical game, predict.patch).  The ship never moves
// on screen; the land is read from the same tileable relief map at a turned,
// travelling coordinate: board offset (X right, Y ahead of the ship) maps to
// map (u, v) = travel + (X cos yaw + Y sin yaw, X sin yaw - Y cos yaw).  The
// heights on the mesh's grid nodes, their normals from neighbouring nodes,
// and the colour/depth UVs are rewritten every frame -- the DepthMesh's own
// grid and strips, re-read, so relief, colour and depth turn together.
const TURN_LAND = { travel: [0, 0], c: 1, s: 0 };
// THE GLOBE'S LAND (a game that publishes predict.sphere): the engine's
// terrain_height law (compiled) sampled once over the whole sphere into a
// LAND.W x LAND.H latitude/longitude table; each frame the land's grid nodes
// are mapped from the patch to their points on the sphere about the ship's
// predicted place and heading, and read from the table -- height for the
// mesh, height and raw noise for its colour -- so the land is the sphere's,
// wherever you fly, with no seam and no flat map.
const LAND = { W: 720, H: 360, height: null, raw: null, frame: null };
function bakeLand() {
  if (LAND.height || !pred.terrain) return !!LAND.height;
  const { W, H } = LAND, height = new Float32Array(W * H), raw = new Float32Array(W * H), started = performance.now();
  for (let j = 0; j < H; j++) {
    const lat = -Math.PI / 2 + (j / (H - 1)) * Math.PI, cl = Math.cos(lat), sl = Math.sin(lat);
    for (let i = 0; i < W; i++) {
      const lon = -Math.PI + (i / W) * 2 * Math.PI;
      const r = pred.terrain.invoke({ x: cl * Math.cos(lon), y: cl * Math.sin(lon), z: sl });
      height[j * W + i] = r.height; raw[j * W + i] = r.raw;
    }
  }
  LAND.height = height; LAND.raw = raw;
  residentLand();
  console.info(`[meow] globe's land baked from terrain_height: ${W}x${H} in ${Math.round(performance.now() - started)} ms`);
  return true;
}
// The whole sphere's colour (colour stack layer 0) and relief (depth stack),
// as longitude x latitude images, uploaded ONCE: the land's vertices carry
// their latitude/longitude as UVs and the shader slices the resident sphere.
function residentLand() {
  const sphere = state.game && state.game.predict && state.game.predict.sphere;
  if (!sphere) return;
  const paint = (size, withColour) => {
    const img = new Uint8Array(size * size * 4);
    for (let j = 0; j < size; j++) {
      const lat = -Math.PI / 2 + ((j + 0.5) / size) * Math.PI, cl = Math.cos(lat), sl = Math.sin(lat);
      for (let i = 0; i < size; i++) {
        const lon = -Math.PI + ((i + 0.5) / size) * 2 * Math.PI;
        const [h, raw] = sampleLand(cl * Math.cos(lon), cl * Math.sin(lon), sl), o = (j * size + i) * 4;
        if (withColour) landColour(h, raw, sphere.sea, sphere.relief, img, o);
        else { const d = Math.round(255 * h / sphere.relief); img[o] = d; img[o + 1] = d; img[o + 2] = 255; img[o + 3] = 0; }
      }
    }
    return img;
  };
  gl.activeTexture(gl.TEXTURE0 + 5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, state.colourStack);
  gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, 0, STACK_SIZE, STACK_SIZE, 1, gl.RGBA, gl.UNSIGNED_BYTE, paint(STACK_SIZE, true));
  gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  if (state.depthStack) {
    const n = state.depthStack.n;
    gl.activeTexture(gl.TEXTURE0 + 6); gl.bindTexture(gl.TEXTURE_2D_ARRAY, state.depthStack.tex);
    gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, 0, n, n, 1, gl.RGBA, gl.UNSIGNED_BYTE, paint(n, false));
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
  }
}
function sampleLand(x, y, z) {
  const { W, H } = LAND;
  const lon = Math.atan2(y, x), lat = Math.asin(Math.max(-1, Math.min(1, z)));
  const fx = ((lon + Math.PI) / (2 * Math.PI)) * W, fy = ((lat + Math.PI / 2) / Math.PI) * (H - 1);
  const i0 = Math.floor(fx), j0 = Math.min(H - 2, Math.max(0, Math.floor(fy))), tx = fx - i0, ty = Math.min(1, Math.max(0, fy - j0));
  const a = ((i0 % W) + W) % W, b = (a + 1) % W, r0 = j0 * W, r1 = (j0 + 1) * W;
  const lerp = (T) => (T[r0 + a] + (T[r0 + b] - T[r0 + a]) * tx) * (1 - ty) + (T[r1 + a] + (T[r1 + b] - T[r1 + a]) * tx) * ty;
  return [lerp(LAND.height), lerp(LAND.raw)];
}
// the patch point (X right, Y ahead of the ship) on the sphere about (u0, h0)
function patchToSphere(X, Y, f) {
  const d = Math.hypot(X, Y);
  if (d < 1e-9) return f.u0;
  const a = d / f.radius, c = Math.cos(a), s = Math.sin(a);
  return [0, 1, 2].map((k) => f.u0[k] * c + ((f.h0[k] * Y + f.r0[k] * X) / d) * s);
}
function landColour(height, raw, sea, relief, out, o) {
  // Bands blended smoothly across their edges (smoothstep over a narrow
  // width), so coastlines and snowlines interpolate instead of stepping.
  const ss = (a, b, x) => { const k = Math.max(0, Math.min(1, (x - a) / (b - a))); return k * k * (3 - 2 * k); };
  const mixc = (p, q, k) => [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, p[2] + (q[2] - p[2]) * k];
  const deep = Math.min(1, Math.max(0, (sea - raw) / 0.12));
  const water = [0.12 - 0.08 * deep, 0.32 - 0.16 * deep, 0.52 - 0.18 * deep];
  const f = height / relief;
  const grass = [0.22 + 0.12 * Math.min(1, f / 0.32), 0.48 - 0.08 * Math.min(1, f / 0.32), 0.2];
  let land = mixc([0.78, 0.72, 0.5], grass, ss(0.01, 0.05, f));                 // sand into grass
  land = mixc(land, [0.47, 0.42, 0.36], ss(0.28, 0.4, f));                       // into rock
  land = mixc(land, [0.9, 0.91, 0.94], ss(0.62, 0.74, f));                       // into snow
  const c = mixc(water, land, ss(sea - 0.004, sea + 0.004, raw));               // the shore
  out[o] = c[0] * 255; out[o + 1] = c[1] * 255; out[o + 2] = c[2] * 255; out[o + 3] = 255;
}
// THE ROADS (game.roads: the server's plan, polylines on the sphere): draped
// on the land as thin ribbons -- each polyline subdivided along its great
// circles, projected into the patch about the ship's predicted place, set on
// the land's height from the same baked table.
const ROADS_VIEW = { vao: null, buffers: null, width: 0.07, sub: 6 };
// The click path (World Walk) is drawn the same way: a glowing ribbon from the walker to the goal.
const ROUTE_VIEW = { vao: null, buffers: null, width: 0.05, sub: 4 };
function drawRoads(drawLit) {
  const game = state.game;
  if (!game) return;
  drawRibbons(ROADS_VIEW, game.roads, state.materialIndex.road, drawLit, 0.025);
  const route = isWalker(game) ? walkRouteLine() : null;
  if (route) drawRibbons(ROUTE_VIEW, [route], state.materialIndex.city ?? state.materialIndex.road, drawLit, 0.06);
}
function drawRibbons(view, lines, material, drawLit, lift) {
  const game = state.game, f = LAND.frame, patch = game && game.predict && game.predict.patch;
  if (!lines || !f || !LAND.height || !patch || state.flat) return;
  const relief = TERRAIN.relief, y0 = relief ? terrainOrigin(relief)[1] : 0, R = f.radius;
  const toPatch = (u) => {
    const e = u[0] * f.r0[0] + u[1] * f.r0[1] + u[2] * f.r0[2], n = u[0] * f.h0[0] + u[1] * f.h0[1] + u[2] * f.h0[2];
    const c = u[0] * f.u0[0] + u[1] * f.u0[1] + u[2] * f.u0[2], rho = Math.hypot(e, n);
    const k = R * Math.atan2(rho, c) / Math.max(rho, 1e-9);
    return [patch.cx + e * k, patch.cy - n * k, c];
  };
  const pos = [], nrm = [];
  const near = Math.cos(22 / R);
  for (const line of lines) {
    let prev = null;
    for (let s = 0; s + 1 < line.length; s++) {
      const a = line[s], b = line[s + 1];
      for (let q = 0; q <= view.sub; q++) {
        if (q === 0 && s > 0) continue;
        const w = q / view.sub, m = [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
        const len = Math.hypot(m[0], m[1], m[2]), u = [m[0] / len, m[1] / len, m[2] / len];
        const [bx, bz, c] = toPatch(u);
        const inView = c > near && Math.abs(bx - patch.cx) < 9 && bz > -6 && bz < patch.cy + 6;
        const y = y0 + sampleLand(u[0], u[1], u[2])[0] + lift - planetDrop(bx, bz);
        const here = inView ? [bx, y, bz] : null;
        if (prev && here) {
          const dx = here[0] - prev[0], dz = here[2] - prev[2], l = Math.hypot(dx, dz) || 1;
          const sx = (-dz / l) * view.width, sz = (dx / l) * view.width;
          const p1 = [prev[0] - sx, prev[1], prev[2] - sz], p2 = [prev[0] + sx, prev[1], prev[2] + sz];
          const p3 = [here[0] - sx, here[1], here[2] - sz], p4 = [here[0] + sx, here[1], here[2] + sz];
          for (const v of [p1, p2, p4, p1, p4, p3]) { pos.push(...v); nrm.push(0, 1, 0); }
        }
        prev = here;
      }
    }
  }
  if (!pos.length) return;
  if (!view.vao) {
    view.vao = gl.createVertexArray(); gl.bindVertexArray(view.vao);
    view.buffers = [0, 1].map((loc) => { const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0); return buf; });
    gl.bindVertexArray(null);
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, view.buffers[0]); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, view.buffers[1]); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nrm), gl.DYNAMIC_DRAW);
  drawLit(view.vao, pos.length / 3, M.identity(), material);
}
function sphereLand(relief, now) {
  const game = state.game, patch = game.predict && game.predict.patch, sphere = game.predict && game.predict.sphere;
  if (!patch || !sphere || !relief.surface || !bakeLand()) return false;
  const ship = (pred.ship && pred.ship.ux != null ? pred.ship : game.you) || {};
  if (ship.ux == null) return false;
  const u0 = [ship.ux, ship.uy, ship.uz], h0 = [ship.hx, ship.hy, ship.hz];
  const r0 = [h0[1] * u0[2] - h0[2] * u0[1], h0[2] * u0[0] - h0[0] * u0[2], h0[0] * u0[1] - h0[1] * u0[0]];
  const frame = LAND.frame = { u0, h0, r0, radius: sphere.radius, relief: sphere.relief, sea: sphere.sea };
  const origin = terrainOrigin(relief), x0 = origin[0], z0 = origin[2], y0 = origin[1];
  const ext = state.extent, R = PLANET.radius, step = relief.step;
  const cols = Math.round(relief.span[0] / step) + 1, rows = Math.round(relief.span[1] / step) + 1;
  const hgt = new Float32Array(cols * rows), lonU = new Float32Array(cols * rows), latV = new Float32Array(cols * rows);
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const k = j * cols + i, X = x0 + i * step - patch.cx, Y = patch.cy - (z0 + j * step);
    const p = patchToSphere(X, Y, frame);
    hgt[k] = sampleLand(p[0], p[1], p[2])[0];
    // the node's place on the resident sphere image (longitude, latitude)
    lonU[k] = (Math.atan2(p[1], p[0]) + Math.PI) / (2 * Math.PI);
    latV[k] = (Math.asin(Math.max(-1, Math.min(1, p[2]))) + Math.PI / 2) / Math.PI;
  }
  const S = relief.surface;
  for (let v = 0; v < S.count; v++) {
    const k = S.node[v], i = k % cols, j = (k - i) / cols;
    const bx = x0 + S.base[v * 3], bz = z0 + S.base[v * 3 + 2], dx = bx - ext.width / 2, dz = bz - ext.depth / 2;
    S.pos[v * 3] = bx; S.pos[v * 3 + 1] = y0 + hgt[k] - (dx * dx + dz * dz) / (2 * R); S.pos[v * 3 + 2] = bz;
    const hl = hgt[j * cols + Math.max(0, i - 1)], hr = hgt[j * cols + Math.min(cols - 1, i + 1)];
    const hu = hgt[Math.max(0, j - 1) * cols + i], hd = hgt[Math.min(rows - 1, j + 1) * cols + i];
    const nx = -(hr - hl) / (2 * step) + dx / R, nz = -(hd - hu) / (2 * step) + dz / R, len = Math.hypot(nx, 1, nz);
    S.nrm[v * 3] = nx / len; S.nrm[v * 3 + 1] = 1 / len; S.nrm[v * 3 + 2] = nz / len;
    S.uv[v * 2] = lonU[k]; S.uv[v * 2 + 1] = latV[k];
  }
  // a triangle straddling the date line: lift its low longitudes by a turn
  // (the image repeats), so no triangle smears across the whole sphere
  for (let v = 0; v < S.count; v += 3) {
    const a = S.uv[v * 2], b = S.uv[v * 2 + 2], c = S.uv[v * 2 + 4];
    if (Math.max(a, b, c) - Math.min(a, b, c) > 0.5)
      for (let w = 0; w < 3; w++) if (S.uv[(v + w) * 2] < 0.5) S.uv[(v + w) * 2] += 1.0;
  }
  [S.pos, S.nrm, S.uv].forEach((data, b) => { gl.bindBuffer(gl.ARRAY_BUFFER, S.buffers[b]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data); });
  if (relief.shadowSurface) {
    const C = relief.shadowSurface;
    for (let v = 0; v < C.count; v++) {
      const bx = x0 + C.base[v * 3], bz = z0 + C.base[v * 3 + 2], dx = bx - ext.width / 2, dz = bz - ext.depth / 2;
      const p = patchToSphere(bx - patch.cx, patch.cy - bz, frame), [h] = sampleLand(p[0], p[1], p[2]);
      C.pos[v * 3] = bx; C.pos[v * 3 + 1] = y0 + h - SHADOW_SINK - (dx * dx + dz * dz) / (2 * R); C.pos[v * 3 + 2] = bz;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, C.buffers[0]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, C.pos);
  }
  return true;
}
const SHADOW_SINK = 0.07;
// The land's height under board point (x, z): the relief map read at the
// same turned, travelling coordinate the land is drawn with, on the land's
// own base (terrainOrigin) -- what a ground piece stands on.
function landHeight(x, z) {
  const relief = TERRAIN.relief, game = state.game, patch = game && game.predict && game.predict.patch;
  if (relief && patch && LAND.frame && LAND.height) {
    const p = patchToSphere(x - patch.cx, patch.cy - z, LAND.frame);
    return terrainOrigin(relief)[1] + sampleLand(p[0], p[1], p[2])[0];
  }
  if (!relief || !relief.map || !patch) return null;
  const c = TURN_LAND.c, s = TURN_LAND.s;
  const X = x - patch.cx, Y = patch.cy - z;
  const u = TURN_LAND.travel[0] + X * c + Y * s, v = TURN_LAND.travel[1] + X * s - Y * c;
  const n = relief.map.n, h = relief.map.h, toMap = n / relief.tile, wrap = (i) => ((i % n) + n) % n;
  const fu = u * toMap, fv = v * toMap, iu = Math.floor(fu), iv = Math.floor(fv), tu = fu - iu, tv = fv - iv;
  const a = wrap(iv), b = wrap(iv + 1), l = wrap(iu), r = wrap(iu + 1);
  const top = h[a * n + l] + (h[a * n + r] - h[a * n + l]) * tu, bottom = h[b * n + l] + (h[b * n + r] - h[b * n + l]) * tu;
  return terrainOrigin(relief)[1] + top + (bottom - top) * tv;
}
function turnRelief(relief, now) {
  const game = state.game, patch = game.predict && game.predict.patch;
  if (!relief.map || !relief.step || !patch) return false;
  // The land's state is the engine's (ground_travel: the server's, carried
  // on by the page's prediction of the same law) -- read, never integrated here.
  const ship = (pred.ship && pred.ship.map_gy != null ? pred.ship : game.you) || {};
  if (ship.map_gy == null) return false;
  TURN_LAND.travel = [ship.map_u, ship.map_v];
  TURN_LAND.c = -ship.map_gy; TURN_LAND.s = ship.map_gx;          // the frame's (cos, sin) of its heading
  const origin = terrainOrigin(relief), x0 = origin[0], z0 = origin[2], c = TURN_LAND.c, s = TURN_LAND.s;
  const n = relief.map.n, h = relief.map.h, toMap = n / relief.tile;
  const wrap = (i) => ((i % n) + n) % n;
  const sample = (u, v) => {
    const fu = u * toMap, fv = v * toMap, iu = Math.floor(fu), iv = Math.floor(fv), tu = fu - iu, tv = fv - iv;
    const a = wrap(iv), b = wrap(iv + 1), l = wrap(iu), r = wrap(iu + 1);
    const top = h[a * n + l] + (h[a * n + r] - h[a * n + l]) * tu, bottom = h[b * n + l] + (h[b * n + r] - h[b * n + l]) * tu;
    return top + (bottom - top) * tv;
  };
  const grid = (step) => {
    const cols = Math.round(relief.span[0] / step) + 1, rows = Math.round(relief.span[1] / step) + 1;
    const hgt = new Float32Array(cols * rows), mu = new Float32Array(cols * rows), mv = new Float32Array(cols * rows);
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const X = x0 + i * step - patch.cx, Y = patch.cy - (z0 + j * step);
      const u = TURN_LAND.travel[0] + X * c + Y * s, v = TURN_LAND.travel[1] + X * s - Y * c, k = j * cols + i;
      hgt[k] = sample(u, v); mu[k] = u; mv[k] = v;
    }
    return { cols, rows, hgt, mu, mv };
  };
  const fine = grid(relief.step), cx = relief.span[0] / 2;
  const ext = state.extent, R = PLANET.radius;
  if (relief.surface) {
    // one surface: board position = origin + mesh (x, z); height = the map's,
    // less the planet's drop there (the same drop every body is drawn with)
    const S = relief.surface, y0 = origin[1];
    for (let v = 0; v < S.count; v++) {
      const k = S.node[v], i = k % fine.cols, j = (k - i) / fine.cols;
      const bx = x0 + S.base[v * 3], bz = z0 + S.base[v * 3 + 2];
      const dx = bx - ext.width / 2, dz = bz - ext.depth / 2;
      S.pos[v * 3] = bx; S.pos[v * 3 + 1] = y0 + fine.hgt[k] - (dx * dx + dz * dz) / (2 * R); S.pos[v * 3 + 2] = bz;
      const hl = fine.hgt[j * fine.cols + Math.max(0, i - 1)], hr = fine.hgt[j * fine.cols + Math.min(fine.cols - 1, i + 1)];
      const hu = fine.hgt[Math.max(0, j - 1) * fine.cols + i], hd = fine.hgt[Math.min(fine.rows - 1, j + 1) * fine.cols + i];
      // slope of the land plus the slope of the planet's curve
      const nx = -(hr - hl) / (2 * relief.step) + dx / R, nz = -(hd - hu) / (2 * relief.step) + dz / R, len = Math.hypot(nx, 1, nz);
      S.nrm[v * 3] = nx / len; S.nrm[v * 3 + 1] = 1 / len; S.nrm[v * 3 + 2] = nz / len;
      S.uv[v * 2] = fine.mu[k] / relief.tile; S.uv[v * 2 + 1] = fine.mv[k] / relief.tile;
    }
    [S.pos, S.nrm, S.uv].forEach((data, b) => { gl.bindBuffer(gl.ARRAY_BUFFER, S.buffers[b]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data); });
    if (relief.shadowSurface) {
      const coarse = grid(relief.shadowStep), C = relief.shadowSurface;
      for (let v = 0; v < C.count; v++) {
        const bx = x0 + C.base[v * 3], bz = z0 + C.base[v * 3 + 2], dx = bx - ext.width / 2, dz = bz - ext.depth / 2;
        C.pos[v * 3] = bx; C.pos[v * 3 + 1] = y0 + coarse.hgt[C.node[v]] - SHADOW_SINK - (dx * dx + dz * dz) / (2 * R); C.pos[v * 3 + 2] = bz;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, C.buffers[0]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, C.pos);
    }
    return true;
  }
  for (const strip of relief.strips) {
    if (!strip.node) continue;
    for (let v = 0; v < strip.count; v++) {
      const k = strip.node[v], i = k % fine.cols, j = (k - i) / fine.cols;
      const dx = strip.pos[v * 3] - cx;
      strip.pos[v * 3 + 1] = fine.hgt[k] - (dx * dx) / (2 * PLANET.radius);
      const hl = fine.hgt[j * fine.cols + Math.max(0, i - 1)], hr = fine.hgt[j * fine.cols + Math.min(fine.cols - 1, i + 1)];
      const hu = fine.hgt[Math.max(0, j - 1) * fine.cols + i], hd = fine.hgt[Math.min(fine.rows - 1, j + 1) * fine.cols + i];
      const nx = -(hr - hl) / (2 * relief.step), nz = -(hd - hu) / (2 * relief.step), len = Math.hypot(nx, 1, nz);
      strip.nrm[v * 3] = nx / len; strip.nrm[v * 3 + 1] = 1 / len; strip.nrm[v * 3 + 2] = nz / len;
      strip.uv[v * 2] = fine.mu[k] / relief.tile; strip.uv[v * 2 + 1] = fine.mv[k] / relief.tile;
    }
    [strip.pos, strip.nrm, strip.uv].forEach((data, b) => { gl.bindBuffer(gl.ARRAY_BUFFER, strip.buffers[b]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data); });
  }
  if (relief.shadowStep) {
    const coarse = grid(relief.shadowStep);
    for (const strip of relief.shadows) {
      if (!strip.node) continue;
      for (let v = 0; v < strip.count; v++) {
        const dx = strip.pos[v * 3] - cx;
        // a hair under the fine land, so the coarse caster never cuts through
        // the surface it shades (that crossing was the rippling shadow)
        strip.pos[v * 3 + 1] = coarse.hgt[strip.node[v]] - SHADOW_SINK - (dx * dx) / (2 * PLANET.radius);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, strip.buffers[0]); gl.bufferSubData(gl.ARRAY_BUFFER, 0, strip.pos);
    }
  }
  return true;
}
function terrainOrigin(relief) {
  // Centred across the field; starting one tile above it and sliding down.
  return [state.extent.width / 2 - relief.span[0] / 2, -0.05, -relief.tile - 2.0 + TERRAIN.scroll];
}
function stripModel(relief, strip) {
  const [x0, y0, z0] = terrainOrigin(relief), z = z0 + strip.centre, d = z - state.extent.depth / 2;
  const drop = (d * d) / (2 * PLANET.radius), slope = Math.atan(d / PLANET.radius);
  return M.multiply(M.translate(x0, y0 - drop, z), M.rotateX(slope));
}
// The decimated relief into the shadow map (curved the same way): peaks
// shadow the land and what flies low beneath them, at a fraction of the
// full mesh's triangles.
function terrainShadowCasters() {
  const game = state.game;
  if (state.flat || !game || !(game.scene && game.scene.terrain === true)) return [];
  const relief = reliefMesh(state.assets && state.assets.terrain);
  if (relief && relief.shadowSurface && game.predict && game.predict.patch) return [[relief.shadowSurface.vao, relief.shadowSurface.count, M.identity()]];
  return relief ? relief.shadows.map((s) => [s.vao, s.count, stripModel(relief, s)]) : [];
}
function drawTerrain(now, drawLit) {
  const game = state.game;
  if (state.flat || !game || !(game.scene && game.scene.terrain === true)) return;
  const relief = reliefMesh(state.assets && state.assets.terrain);
  if (relief) {
    const frame = TERRAIN.at == null ? 0 : Math.min(0.1, now - TERRAIN.at);
    TERRAIN.at = now;
    if (game.predict && game.predict.patch && relief.surface) {
      // the land turns and travels under the ship: one continuous surface
      TERRAIN.scroll = 0;
      if (sphereLand(relief, now) || turnRelief(relief, now)) {
        drawLit(relief.surface.vao, relief.surface.count, M.identity(), state.materialIndex.terrain);
        drawRoads(drawLit);
        return;
      }
    }
    if (!game.over) TERRAIN.scroll = (TERRAIN.scroll + TERRAIN.speed * frame) % relief.tile;
    for (const strip of relief.strips) drawLit(strip.vao, strip.count, stripModel(relief, strip), state.materialIndex.terrain);
    return;
  }
  const key = `${state.extent.width}x${state.extent.depth}`;
  if (!TERRAIN.mesh || TERRAIN.mesh.key !== key) TERRAIN.mesh = terrainMesh(state.extent.width, state.extent.depth);
  const frame = TERRAIN.at == null ? 0 : Math.min(0.1, now - TERRAIN.at);
  TERRAIN.at = now;
  if (!game.over) TERRAIN.scroll = (TERRAIN.scroll + TERRAIN.speed * frame) % (TERRAIN.tile * 64);
  const uv = new Float32Array(12);
  TERRAIN.mesh.corners.forEach(([x, z], i) => { uv[i * 2] = x / TERRAIN.tile; uv[i * 2 + 1] = (z - TERRAIN.scroll) / TERRAIN.tile; });
  gl.bindBuffer(gl.ARRAY_BUFFER, TERRAIN.mesh.uvBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, uv);
  drawLit(TERRAIN.mesh.vao, TERRAIN.mesh.count, M.identity(), state.materialIndex.terrain);
}
function volumeTexture(unit) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]));
  return tex;
}
function buildShadowMap() {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + 9); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 1);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_MODE, gl.NONE);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, tex, 0, 0);
  gl.drawBuffers([gl.NONE]); gl.readBuffer(gl.NONE);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (!ok) throw new Error("shadow framebuffer incomplete");
  return { tex, fbo };
}
function flatVao(pos, col) {
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, count: pos.length / 3 };
}
function buildSphere(radius = 1, rings = 10, segments = 16) {
  const pos = [], col = [];
  const vert = (i, j) => { const th = i / rings * Math.PI, ph = j / segments * Math.PI * 2;
    return [radius * Math.sin(th) * Math.cos(ph), radius * Math.cos(th), radius * Math.sin(th) * Math.sin(ph)]; };
  for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
    const a = vert(i, j), b = vert(i + 1, j), c = vert(i + 1, j + 1), d = vert(i, j + 1);
    for (const q of [a, b, c, a, c, d]) { pos.push(...q); col.push(1.0, 0.93, 0.78); }
  }
  return flatVao(pos, col);
}
function sphereMesh(rings = 10, segments = 16) {
  const positions = [], normals = [];
  const vert = (i, j) => { const th = i / rings * Math.PI, ph = j / segments * Math.PI * 2;
    return [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)]; };
  for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
    const a = vert(i, j), b = vert(i + 1, j), c = vert(i + 1, j + 1), d = vert(i, j + 1);
    for (const q of [a, c, b, a, d, c]) { positions.push(...q); normals.push(...q); }
  }
  return { positions, normals, extent: 1 };
}
function uploadMesh(mesh) {
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const count = mesh.positions.length / 3, inter = new Float32Array(count * 8);
  for (let i = 0; i < count; i++) { inter.set(mesh.positions.slice(i*3, i*3+3), i*8); inter.set(mesh.normals.slice(i*3, i*3+3), i*8+3); }
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, inter, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 32, 24);
  gl.bindVertexArray(null);
  // bounding box: the soft-body cage's rest lattice sits on its corners
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) for (let k = 0; k < 3; k++) { const v = mesh.positions[i*3+k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
  const center = count ? lo.map((v, k) => (v + hi[k]) / 2) : [0, 0, 0], half = count ? lo.map((v, k) => Math.max((hi[k] - v) / 2, 1e-4)) : [1, 1, 1];
  return { vao, count, extent: mesh.extent, center, half };
}

// Client-side corner rounding of the extruded glyph meshes: weld, crease-angle smooth
// normals, Taubin smoothing of crease vertices only.  Server meshes stay flat.
const roundParams = { iters: 4, creaseDeg: 90 };      // defaults: fully rounded and creased
function roundMesh(mesh, iters, creaseDeg) {
  const P0 = mesh.positions, T = P0.length / 9, N0 = mesh.normals;
  if (!T || (P0.length % 9)) return mesh;
  const lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30];
  for (let i = 0; i < P0.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P0[i+k]); hi[k] = Math.max(hi[k], P0[i+k]); }
  const ctr = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2), diag = Math.hypot(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]) || 1, q = 1e-5 * diag;
  // 1. weld
  const key = new Map(), wid = new Int32Array(T * 3), wp = [];
  for (let i = 0; i < T * 3; i++) {
    const k = Math.round(P0[i*3]/q) + "," + Math.round(P0[i*3+1]/q) + "," + Math.round(P0[i*3+2]/q);
    let id = key.get(k); if (id === undefined) { id = wp.length / 3; key.set(k, id); wp.push(P0[i*3], P0[i*3+1], P0[i*3+2]); }
    wid[i] = id;
  }
  const V = wp.length / 3, P = Float64Array.from(wp);
  // consistent outward orientation of each face by edge propagation (walls are wound inward vs caps)
  const edges = new Map();
  for (let f = 0; f < T; f++) for (let e = 0; e < 3; e++) {
    const a = wid[f*3+e], b = wid[f*3+(e+1)%3], k = Math.min(a, b) * V + Math.max(a, b);
    let l = edges.get(k); if (!l) edges.set(k, l = []); l.push(f * 3 + e);
  }
  const ori = new Int8Array(T), seen = new Uint8Array(T);
  const geom = (f, out) => { const a = wid[f*3]*3, b = wid[f*3+1]*3, c = wid[f*3+2]*3;
    const ux = P[b]-P[a], uy = P[b+1]-P[a+1], uz = P[b+2]-P[a+2], vx = P[c]-P[a], vy = P[c+1]-P[a+1], vz = P[c+2]-P[a+2];
    out[0] = uy*vz-uz*vy; out[1] = uz*vx-ux*vz; out[2] = ux*vy-uy*vx; };
  const tmp = [0, 0, 0];
  for (let s = 0; s < T; s++) {
    if (seen[s]) continue;
    const comp = [s], stack = [s]; seen[s] = 1; ori[s] = 1;
    while (stack.length) {
      const f = stack.pop();
      for (let e = 0; e < 3; e++) {
        const a = wid[f*3+e], b = wid[f*3+(e+1)%3];
        for (const h of edges.get(Math.min(a, b) * V + Math.max(a, b))) {
          const g = (h / 3) | 0; if (seen[g]) continue;
          const same = wid[g*3+h%3] === a;     // same directed edge => inconsistent winding
          ori[g] = same ? -ori[f] : ori[f]; seen[g] = 1; stack.push(g); comp.push(g);
        }
      }
    }
    let vol = 0;
    for (const f of comp) { geom(f, tmp); const a = wid[f*3]*3;
      vol += ori[f] * (tmp[0]*(P[a]-ctr[0]) + tmp[1]*(P[a+1]-ctr[1]) + tmp[2]*(P[a+2]-ctr[2])); }
    if (vol < 0) for (const f of comp) ori[f] = -ori[f];
  }
  // sign relating the stored (render) normal to the outward normal, per face
  const rs = new Int8Array(T);
  for (let f = 0; f < T; f++) { geom(f, tmp);
    rs[f] = (N0[f*9]*tmp[0] + N0[f*9+1]*tmp[1] + N0[f*9+2]*tmp[2]) * ori[f] < 0 ? -1 : 1; }
  // adjacency
  const vf = Array.from({ length: V }, () => []), vn = Array.from({ length: V }, () => new Set());
  for (let f = 0; f < T; f++) for (let e = 0; e < 3; e++) {
    const a = wid[f*3+e]; vf[a].push(f); vn[a].add(wid[f*3+(e+1)%3]); vn[a].add(wid[f*3+(e+2)%3]);
  }
  const cosC = Math.cos(creaseDeg * Math.PI / 180);
  const faceN = new Float64Array(T * 3);
  const computeFaceN = () => { for (let f = 0; f < T; f++) { geom(f, tmp); const l = Math.hypot(tmp[0], tmp[1], tmp[2]) || 1;
    faceN[f*3] = tmp[0]/l*ori[f]; faceN[f*3+1] = tmp[1]/l*ori[f]; faceN[f*3+2] = tmp[2]/l*ori[f]; } };
  computeFaceN();
  // 3. Taubin on crease vertices only
  if (iters > 0) {
    const crease = new Uint8Array(V);
    for (let v = 0; v < V; v++) { const fs = vf[v];
      outer: for (let i = 0; i < fs.length; i++) for (let j = i + 1; j < fs.length; j++) {
        const a = fs[i]*3, b = fs[j]*3;
        if (faceN[a]*faceN[b] + faceN[a+1]*faceN[b+1] + faceN[a+2]*faceN[b+2] < cosC) { crease[v] = 1; break outer; } } }
    const nb = vn.map((s) => Array.from(s)), Q = new Float64Array(P.length);
    const pass = (lam) => { Q.set(P);
      for (let v = 0; v < V; v++) { if (!crease[v] || !nb[v].length) continue;
        let x = 0, y = 0, z = 0; for (const u of nb[v]) { x += P[u*3]; y += P[u*3+1]; z += P[u*3+2]; }
        const n = nb[v].length; Q[v*3] += lam * (x/n - P[v*3]); Q[v*3+1] += lam * (y/n - P[v*3+1]); Q[v*3+2] += lam * (z/n - P[v*3+2]); }
      P.set(Q); };
    for (let it = 0; it < iters; it++) { pass(0.5); pass(-0.53); }
    computeFaceN();
  }
  // 2. crease-angle smooth normals (angle weighted), emitted relative to the stored winding convention
  const pos = new Float32Array(T * 9), nrm = new Float32Array(T * 9);
  for (let f = 0; f < T; f++) for (let c = 0; c < 3; c++) {
    const v = wid[f*3+c], o = f*9 + c*3;
    pos[o] = P[v*3]; pos[o+1] = P[v*3+1]; pos[o+2] = P[v*3+2];
    let x = 0, y = 0, z = 0;
    for (const g of vf[v]) {
      if (faceN[f*3]*faceN[g*3] + faceN[f*3+1]*faceN[g*3+1] + faceN[f*3+2]*faceN[g*3+2] < cosC) continue;
      let k = 0; while (wid[g*3+k] !== v) k++;
      const a = wid[g*3+(k+1)%3]*3, b = wid[g*3+(k+2)%3]*3, vv = v*3;
      const ux = P[a]-P[vv], uy = P[a+1]-P[vv+1], uz = P[a+2]-P[vv+2], wx = P[b]-P[vv], wy = P[b+1]-P[vv+1], wz = P[b+2]-P[vv+2];
      const lu = Math.hypot(ux, uy, uz), lw = Math.hypot(wx, wy, wz); if (!lu || !lw) continue;
      const w = Math.acos(Math.max(-1, Math.min(1, (ux*wx + uy*wy + uz*wz) / (lu * lw))));
      x += faceN[g*3]*w; y += faceN[g*3+1]*w; z += faceN[g*3+2]*w;
    }
    const l = Math.hypot(x, y, z) || 1, s = rs[f] / l;
    nrm[o] = x*s; nrm[o+1] = y*s; nrm[o+2] = z*s;
  }
  return Object.assign({}, mesh, { positions: pos, normals: nrm });
}
function rebuildRoundedMeshes() {
  if (!state.assets) return;
  const t0 = performance.now();
  for (const [name, mesh] of Object.entries(state.assets.meshes)) {
    const old = state.meshes[name];
    if (old) { gl.bindVertexArray(old.vao); const b = gl.getVertexAttrib(0, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
      gl.bindVertexArray(null); gl.deleteVertexArray(old.vao); if (b) gl.deleteBuffer(b); }
    state.meshes[name] = uploadMesh(roundMesh(mesh, roundParams.iters, roundParams.creaseDeg));
  }
  console.log("roundMesh all meshes ms", (performance.now() - t0).toFixed(0));
}

// Lit geometry with a per-vertex material id: buttons, glass, anything
// built on the page.  ``faces`` is a list of [a, b, c, materialIndex].
function litMesh(faces) {
  const pos = [], nrm = [], mat = [];
  for (const [a, b, c, m] of faces) {
    const N = norm(cross(sub(b, a), sub(c, a)));
    for (const q of [a, b, c]) { pos.push(...q); nrm.push(...N); mat.push(m); }
  }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const nb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, nb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(nrm), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  const mb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, mb); gl.bufferData(gl.ARRAY_BUFFER, new Int32Array(mat), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribIPointer(2, 1, gl.INT, 0, 0);
  gl.bindVertexArray(null);
  return { vao, count: pos.length / 3, buffers: [pb, nb, mb] };
}
function box(x0, y0, z0, x1, y1, z1, m) {   // outward faces
  const p = (x, y, z) => [x, y, z];
  const f = [];
  const quad = (a, b, c, d) => { f.push([a, b, c, m], [a, c, d, m]); };
  quad(p(x0,y1,z0), p(x0,y1,z1), p(x1,y1,z1), p(x1,y1,z0));           // top
  quad(p(x0,y0,z0), p(x1,y0,z0), p(x1,y0,z1), p(x0,y0,z1));           // bottom
  quad(p(x0,y0,z1), p(x1,y0,z1), p(x1,y1,z1), p(x0,y1,z1));           // front (+z)
  quad(p(x1,y0,z0), p(x0,y0,z0), p(x0,y1,z0), p(x1,y1,z0));           // back
  quad(p(x1,y0,z1), p(x1,y0,z0), p(x1,y1,z0), p(x1,y1,z1));           // right
  quad(p(x0,y0,z0), p(x0,y0,z1), p(x0,y1,z1), p(x0,y1,z0));           // left
  return f;
}

// ---------------------------------------------------------------- scene state
window.meow = {};
const state = window.meow;
Object.assign(state, {
  assets: null, catalog: null, programs: {}, meshes: {}, units: {},
  game: null,                       // the last state from the server
  pieces: new Map(),                // id -> live piece
  buttons: null, rackButtons: null, glass: null,
  materialIndex: {},                // palette name (and name@hint / name@sel) -> material id
  saturation: 2.0, tiltDeg: 90, time: 0,              // default: fully saturated
  debug: { shadows: true, exposure: 0.42 },
});
const TILE = { height: 0.16, inset: 0.11, gap: 0.03 };
const GLASS = { thickness: 0.22, margin: 0.6 };
// Meowdoku's colour and surface finish live at the material-table seam.
// A future posed cat mesh can use the same "cat" material as today's extrusion.
const MEOW_LOOK = {
  background: [0.89, 0.94, 0.94],
  ink: [0.10, 0.16, 0.23],
  cat: [0.16, 0.29, 0.43],
  x: [0.19, 0.20, 0.32],
  mouse: [0.30, 0.24, 0.39],
};
function meowMaterial(name, rgb) {
  if (name.endsWith("@hint")) return [mix(rgb, [0.18, 0.63, 0.49], 0.28), { roughness: 0.62 }];
  if (name.endsWith("@sel")) return [mix(rgb, [1.0, 0.72, 0.27], 0.32), { roughness: 0.53 }];
  if (name.startsWith("region")) return [mix(rgb, [1, 1, 1], 0.24), { roughness: 0.82 }];
  if (name === "cat") return [MEOW_LOOK.cat, { roughness: 0.48 }];
  if (name === "x") return [MEOW_LOOK.x, { roughness: 0.88 }];
  if (name === "mouse") return [MEOW_LOOK.mouse, { roughness: 0.72 }];
  if (name === "count") return [MEOW_LOOK.ink, { roughness: 0.84 }];
  if (name === "spent") return [[0.43, 0.47, 0.50], { roughness: 0.9 }];
  return [rgb, { roughness: 0.58 }];
}

// ---------------------------------------------------------------- materials from the palette
function uploadMaterials(game) {
  // Base tables hold the three piece materials; every palette colour of the
  // state becomes a record (X mark's template, albedo replaced), plus a
  // lit "@hint" and "@sel" variant per colour and a glass record.
  const base = state.assets.materials, x = state.assets.meshes.x_mark.material;
  const names = Object.keys(game.palette);
  const sat = state.saturation;
  const colour = (rgb) => { const [h, s, v] = rgbToHsv(rgb); return hsvToRgb([h, Math.min(1, s * sat), v]); };
  const records = [];                                   // [name, rgb, extra]
  for (const name of names) {
    const rgb = colour(game.palette[name]);
    const glow = game.emissive || {};
    for (const [role, tint, emission] of [[name, rgb, glow[name]],
      [`${name}@hint`, mix(rgb, [0.45, 0.95, 0.45], 0.55), glow["@hint"]],
      [`${name}@sel`, mix(rgb, [1.0, 0.85, 0.3], 0.6), null]]) {
      const [albedo, finish] = game.game === "meowdoku" ? meowMaterial(role, tint) : [tint, {}];
      records.push([role, albedo, { ...finish, ...(emission ? { emission } : {}) }]);
    }
  }
  records.push(["glass", game.game === "meowdoku" ? [0.26, 0.46, 0.49] : [0.80, 0.90, 1.0],
    game.game === "meowdoku" ? { roughness: 0.84, transmission: 0.0, opacity: 1.0, ior: 1.3 }
      : { roughness: 0.04, transmission: 0.85, opacity: 0.28, ior: 1.5 }]);
  // Emitters are surfaces the shader lights from within (pbr emission): the
  // sun's orb and the arcade bulbs, each bulb colour at ARCADE_LEVELS steps.
  records.push(["sun_orb", [0.2, 0.2, 0.2], { emission: [2.2, 2.0, 1.7] }]);
  // The vital bars: a glass tube each (clear) and its fill (glowing glass).
  // Clouds: white frosted glass (rough, partly transmissive, softly lit from within).
  // Clouds: thin white frosted glass, low over the land.
  records.push(["cloud@glass", [0.97, 0.98, 1.0], { roughness: 0.65, transmission: 0.7, opacity: 0.34, ior: 1.25, emission: [0.06, 0.07, 0.08] }]);
  // Rain clouds (smoke met a cloud): middle-grey glass, middle opacity.
  records.push(["rain@glass", [0.5, 0.52, 0.56], { roughness: 0.7, transmission: 0.45, opacity: 0.5, ior: 1.25 }]);
  // Thunderstorms (a quiet sky): big dark-grey towers of frosted glass.
  records.push(["storm@glass", [0.28, 0.29, 0.33], { roughness: 0.75, transmission: 0.35, opacity: 0.66, ior: 1.2 }]);
  // Smoke where craft were destroyed: black frosted glass as dense as the old
  // clouds, thinning as it ages (smoke_0 .. smoke_3).
  for (let k = 0; k < 4; k++)
    records.push([`smoke_${k}@glass`, [0.03 + 0.03 * k, 0.03 + 0.03 * k, 0.035 + 0.03 * k],
      { roughness: 0.8, transmission: 0.15, opacity: 0.86 - 0.16 * k, ior: 1.15 }]);
  // Lightning: a glowing ribbon of glass, lit from within.
  records.push(["lightning@glow", [0.9, 0.94, 1.0], { roughness: 0.1, transmission: 0.15, opacity: 0.96, ior: 1.3, emission: [6.0, 6.6, 8.2] }]);
  records.push(["zap@glow", [0.5, 1.0, 0.4], { roughness: 0.1, transmission: 0.15, opacity: 0.96, ior: 1.3, emission: [1.6, 5.0, 1.2] }]);
  records.push(["road", [0.16, 0.15, 0.14], { roughness: 0.95 }]);
  records.push(["bar_tube", [0.85, 0.9, 1.0], { roughness: 0.04, transmission: 0.8, opacity: 0.22, ior: 1.5 }]);
  records.push(["bar_shield", [0.3, 0.6, 1.0], { roughness: 0.06, transmission: 0.45, opacity: 0.75, ior: 1.4, emission: [0.12, 0.38, 1.1] }]);
  records.push(["bar_health", [1.0, 0.25, 0.3], { roughness: 0.06, transmission: 0.45, opacity: 0.75, ior: 1.4, emission: [1.1, 0.18, 0.22] }]);
  records.push(["bar_throttle", [1.0, 0.85, 0.2], { roughness: 0.06, transmission: 0.45, opacity: 0.8, ior: 1.4, emission: [1.2, 0.95, 0.15] }]);
  records.push(["bar_boss", [0.3, 1.0, 0.4], { roughness: 0.06, transmission: 0.45, opacity: 0.8, ior: 1.4, emission: [0.2, 1.1, 0.3] }]);
  // Scrolling ground: albedo from colour layer TERRAIN.layer of the shader's
  // uColorUv texture stack (texstack field 1 = colour layer, 9 = blend).
  records.push(["broadcast", [1.0, 1.0, 1.0], { roughness: 0.12, transmission: 0.12, opacity: 0.9, ior: 1.5,
    colorLayer: BROADCAST_LAYER, colorBlend: 1.0 }]);
  records.push(["terrain", [1.0, 1.0, 1.0], { roughness: 0.9, colorLayer: TERRAIN.layer, colorBlend: 1.0,
    depthLayer: TERRAIN.layer, depthScaleMm: (state.assets && state.assets.terrain && state.assets.terrain.depth
      && state.assets.terrain.depth.relief_mm) || 0.0 }]);
  // The craft's energy shield: a glassy bubble that glows blue, in SHIELD_STEPS
  // charge steps (opacity and glow rise with the charge).
  for (let k = 0; k < SHIELD_STEPS; k++) {
    const charge = k / (SHIELD_STEPS - 1);
    records.push([`shield_${k}`, [0.35, 0.65, 1.0], { roughness: 0.05, transmission: 0.6, ior: 1.2,
      opacity: 0.06 + 0.22 * charge, emission: [0.1 * charge, 0.35 * charge, 0.9 * charge] }]);
  }
  ARCADE.colours.forEach((c, ci) => ARCADE_LEVELS.forEach((level, li) =>
    records.push([`bulb${ci}_${li}`, c.map((v) => 0.25 * v), { emission: c.map((v) => v * level), roughness: 0.2 }])));
  // The rocket's aeronautic marker lights: red port, green starboard, white
  // tail, and the white anti-collision strobe (dark between flashes).
  records.push(["nav_red", [0.3, 0.02, 0.02], { emission: [2.6, 0.12, 0.1], roughness: 0.2 }]);
  records.push(["beam", [0.2, 0.9, 0.8], { roughness: 0.1, transmission: 0.3, opacity: 0.85, ior: 1.2, emission: [0.5, 2.8, 2.4] }]);
  records.push(["nav_green", [0.02, 0.3, 0.06], { emission: [0.1, 2.4, 0.5], roughness: 0.2 }]);
  records.push(["nav_white", [0.3, 0.3, 0.3], { emission: [1.6, 1.6, 1.5], roughness: 0.2 }]);
  records.push(["nav_strobe", [0.3, 0.3, 0.3], { emission: [2.4, 2.4, 2.5], roughness: 0.2 }]);
  records.push(["nav_off", [0.12, 0.12, 0.13], { roughness: 0.3 }]);
  const tables = {};
  state.materialIndex = {};
  state.celPalette = new Float32Array(CEL_PALETTE * 3);
  {
    const f = base.pbr.floats_per_record;
    for (let r = 0; r < Math.min(base.pbr.records, CEL_PALETTE); r++) for (let c = 0; c < 3; c++) state.celPalette[r * 3 + c] = base.pbr.data[r * f + c];
  }
  for (const [tname, table] of Object.entries(base)) {
    const f = table.floats_per_record, data = Array.from(table.data), template = data.slice(x * f, (x + 1) * f);
    records.forEach(([name, rgb, extra], k) => {
      const record = template.slice();
      if (tname === "pbr") {
        record[0] = rgb[0]; record[1] = rgb[1]; record[2] = rgb[2];
        record[3] = extra.roughness ?? 0.72; record[4] = 0.0; record[5] = extra.transmission ?? 0.0;
        record[6] = extra.ior ?? 1.5; record[7] = extra.opacity ?? 1.0;
        const emission = extra.emission || [0, 0, 0];
        record[8] = emission[0]; record[9] = emission[1]; record[10] = emission[2];
      }
      if (tname === "texstack" && extra.colorLayer != null) {
        record[1] = extra.colorLayer; record[9] = extra.colorBlend ?? 1.0;
      }
      if (tname === "texstack" && extra.depthLayer != null) {
        // texstack 2 = depth layer, 4 = depth scale (mm), 6 = depth bias (mm);
        // 5 / 7 = thickness scale / bias (the G channel, same map).
        record[2] = extra.depthLayer; record[4] = extra.depthScaleMm ?? 0.0; record[6] = 0.0;
        record[5] = extra.depthScaleMm ?? 0.0; record[7] = 0.0;
      }
      if (tname === "phong") {
        record[1] = extra.transmission ? 0.5 : Math.min(record[1], 0.10);
        // The extruded meshes' walls are wound inward (measured: wall signed volume is
        // negative, stored normals follow the winding), so seen from outside they are
        // BACK faces, and the shader paints back faces with the phong record's "inner"
        // colour (floats 4..6) instead of albedo.  The X-mark template's inner is
        // (0.055, 0.055, 0.066): every wall of every piece rendered near-black.  The
        // material is the palette colour on both sides, so inner = albedo.
        record[4] = rgb[0]; record[5] = rgb[1]; record[6] = rgb[2];
      }
      data.push(...record);
      state.materialIndex[name] = base.pbr.records + k;
      if (tname === "pbr" && base.pbr.records + k < CEL_PALETTE) state.celPalette.set(rgb, (base.pbr.records + k) * 3);
    });
    tables[tname] = { floats_per_record: f, records: table.records + records.length, data };
  }
  state.materialsVersion = (state.materialsVersion || 0) + 1;
  storageTexture(tables.pbr, 0); storageTexture(tables.phong, 1); storageTexture(tables.enamel, 2); storageTexture(tables.texstack, 3);
}

// ---------------------------------------------------------------- boards
function buttonFaces(rows, cols, cellName, origin, cellSize, hintOf) {
  const faces = [], g = TILE.gap * cellSize, i = TILE.inset * cellSize, h = TILE.height * cellSize;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const name = cellName(r, c); if (!name) continue;
    const m = state.materialIndex[hintOf ? hintOf(r, c, name) : name] ?? state.materialIndex[name];
    const x0 = origin[0] + c * cellSize + g, x1 = origin[0] + (c + 1) * cellSize - g;
    const z0 = origin[1] + r * cellSize + g, z1 = origin[1] + (r + 1) * cellSize - g;
    const b = [[x0,0,z0],[x1,0,z0],[x1,0,z1],[x0,0,z1]];
    const tp = [[x0+i,h,z0+i],[x1-i,h,z0+i],[x1-i,h,z1-i],[x0+i,h,z1-i]];
    faces.push([tp[0], tp[2], tp[1], m], [tp[0], tp[3], tp[2], m]);
    for (let k = 0; k < 4; k++) { const k2 = (k + 1) % 4; faces.push([b[k], tp[k], tp[k2], m], [b[k], tp[k2], b[k2], m]); }
  }
  return faces;
}
function rackLayout(game) {
  if (!game.rack) return null;
  const size = game.cols / game.rack.cols;                        // the rack spans the board's width
  return { size, origin: [0, game.rows + 1.0], rows: game.rack.rows, cols: game.rack.cols };
}
function disposeMesh(mesh) {
  if (!mesh) return;
  for (const buffer of mesh.buffers || []) gl.deleteBuffer(buffer);
  gl.deleteVertexArray(mesh.vao);
}
function rebuildBoards(game) {
  // Only when the board itself changed: a real-time game replies ~16-30 times
  // a second with the same board, and every rebuild used to allocate (and
  // never free) a new set of GPU buffers.
  const key = JSON.stringify([state.materialsVersion, game.board, game.width, game.height, game.rows, game.cols,
                              game.cells, game.hints, game.selected, game.hearts != null, game.rack && game.rack.cells]);
  if (key === state.boardKey) return;
  state.boardKey = key;
  disposeMesh(state.buttons); disposeMesh(state.rackButtons); disposeMesh(state.trayButtons); state.trayButtons = null; disposeMesh(state.glass);
  const hintOf = (r, c, name) => {
    const i = r * game.cols + c;
    if (game.selected === i) return `${name}@sel`;
    if (game.hints[i]) return `${name}@hint`;
    return name;
  };
  // A free board has no cells: one floor slab the entities move over.
  state.buttons = game.board === "free"
    ? litMesh(box(0, 0, 0, game.width, TILE.height, game.height, state.materialIndex.floor))
    : litMesh(buttonFaces(game.rows, game.cols, (r, c) => game.cells[r * game.cols + c], [0, 0], 1.0, hintOf));
  const rack = rackLayout(game);
  state.rackButtons = rack ? litMesh(buttonFaces(rack.rows, rack.cols, (r, c) => game.rack.cells[r * rack.cols + c] || null, rack.origin, rack.size)) : null;
  const trayFaces = game.powerups ? trayButtonFaces(game) : null;
  state.trayButtons = trayFaces && trayFaces.length ? litMesh(trayFaces) : null;
  const depth = rack ? rack.origin[1] + rack.rows * rack.size
    : game.rows + (game.powerups ? TRAY.row + TRAY.countDrop + 0.45 : game.hearts != null ? 1.5 : 0);
  const width = game.board === "free" ? game.width : game.cols;
  state.extent = { width, depth: game.board === "free" ? game.height : depth };
  const m = GLASS.margin, t = GLASS.thickness, gm = state.materialIndex.glass;
  if (game.scene && game.scene.terrain === true && !state.flat) {
    // The plate lies on the planet's curve: a fine grid of glass (it never
    // scrolls, so the curve is baked once).
    const faces = [], step = 0.5, x0 = -m, x1 = width + m, z0 = -m, z1 = state.extent.depth + m;
    const at = (x, z) => [x, -planetDrop(x, z), z];
    for (let x = x0; x < x1 - 1e-6; x += step) for (let z = z0; z < z1 - 1e-6; z += step) {
      const xa = Math.min(x + step, x1), za = Math.min(z + step, z1);
      faces.push([at(x, z), at(x, za), at(xa, za), gm], [at(x, z), at(xa, za), at(xa, z), gm]);
    }
    state.glass = litMesh(faces);
  } else {
    state.glass = litMesh(box(-m, -t, -m, width + m, 0.0, state.extent.depth + m, gm));
  }
}

// ---------------------------------------------------------------- camera and light
// SCENE-CAMERA GESTURES: a view offset (yaw, pitch, log-zoom, pan on the ground)
// that composes on top of whatever camera the game is in (board, flat, follow);
// the gesture layer further down moves the goal, stepCamView eases `cur` toward it.
const camView = {
  key: null, lastR: 1, active: 0,
  cur: { yaw: 0, pitch: 0, lz: 0, px: 0, pz: 0 }, goal: { yaw: 0, pitch: 0, lz: 0, px: 0, pz: 0 },
  vel: { yaw: 0, pitch: 0, lz: 0, px: 0, pz: 0 },
};
function camera() {
  const ext = state.extent || { width: 6, depth: 6 };
  const aspect = canvas.width / canvas.height;
  const el = state.tiltDeg * Math.PI / 180;                        // 90 = straight down
  // Frame the whole extent: the half-size that must fit, over the lens'
  // half-angle, with room for the bar and the hearts; a lower camera sees
  // the board foreshortened and backs off a little more.
  // A free board is sized to the screen already, so it is framed tightly.
  const pad = state.game && state.game.board === "free" ? 0.5 : 1.2, slack = state.game && state.game.board === "free" ? 1.02 : 1.12;
  const halfFit = Math.max((ext.depth + pad) / 2, (ext.width + pad) / (2 * aspect)) * slack;
  const declared = cameraGestures(), V = camView.cur, az = V.yaw;
  const centre = declared && declared.target ? declared.target : [ext.width / 2, 0, ext.depth / 2];
  const target = [centre[0] + V.px, centre[1], centre[2] + V.pz];
  if (state.flat) {
    // Flat view: orthographic, looking straight down; the same framing as the lens below.
    const eye = [target[0], 60, target[2]], view = M.lookAt(eye, target, [-Math.sin(az), 0, -Math.cos(az)]);
    const hh = halfFit * Math.exp(V.lz), hw = hh * aspect, near = 1, far = 120;
    camView.lastR = hh;
    const proj = new Float32Array([1 / hw,0,0,0, 0,1 / hh,0,0, 0,0,-2 / (far - near),0, 0,0,-(far + near) / (far - near),1]);
    return { view, proj, viewProj: M.multiply(proj, view), eye };
  }
  const RAISE = 1.12;                                              // the eye a little higher off the board
  // FIRST PERSON (the Camera control, N): the eye just ahead of the ship's
  // nose, looking along its heading -- up the board -- and a little down at
  // the land; a spherical game turns the world under it, so this is the
  // view out of the cockpit.
  const own = noseCam() && state.game && state.game.predict && state.game.input === "realtime"
    ? [...state.pieces.values()].find((p) => p.entity && p.lane === state.game.predict.lane && p.removing == null) : null;
  if (own) {
    const at = worldOf(own), nose = (own.size || 0.5) * 1.1;
    const eye = [at[0], at[1] + 0.22, at[2] - nose];
    const look = [at[0], at[1] - 0.9 + NOSE_PITCH.value * 4.2, at[2] - nose - 7.0];
    const view = M.lookAt(eye, look, [0, 1, 0]);
    const proj = M.perspective(COCKPIT.fov, aspect, 0.05, 300);
    return { view, proj, viewProj: M.multiply(proj, view), eye, nose: true, aspect };
  }
  const dist = RAISE * halfFit / Math.tan(0.31) * (1 + 0.35 * Math.cos(el)) * Math.exp(V.lz);
  camView.lastR = dist;
  const el2 = Math.max(0.05, Math.min(Math.PI / 2, el + V.pitch));    // the gesture tilt, on top of the game's own
  const eye = [target[0] + dist * Math.cos(el2) * Math.sin(az), target[1] + dist * Math.sin(el2), target[2] + dist * Math.cos(el2) * Math.cos(az)];
  const up = el2 > 1.45 ? [-Math.sin(az), 0, -Math.cos(az)] : [0, 1, 0];
  const view = M.lookAt(eye, target, up);
  const proj = M.perspective(0.62, aspect, 0.1, 300);
  return { view, proj, viewProj: M.multiply(proj, view), eye, target };
}
const light = { angle: 0.9, radius: 0.72, height: 0.8, follow: false, pointer: null, position: [0, 1, 0] };
state.light = light;
// ---------------------------------------------------------------- result splash and auto restart
// When a game ends (won, lost, or over), show the result over the board, then
// -- unless Auto restart is off -- start again: a fresh puzzle, word or match
// for the turn-based games, a reset for the real-time ones.
const SPLASH_S = 2.6;
const splash = { key: null, ended: false, timer: null };
function resultSplash(game) {
  const key = `${game.game}|${game.id}`, ended = !!(game.won || game.lost || game.over);
  if (key !== splash.key) { splash.key = key; splash.ended = ended; hideSplash(); return; }
  if (ended && !splash.ended) {
    const box = $("splash"), win = !!game.won;
    box.className = `show ${win ? "win" : "loss"}`;
    $("splash-title").textContent = win ? (game.game === "meowdoku" ? "Solved!" : "You win!") : "Game over";
    $("splash-detail").textContent = String(game.status || "").split(" · ").slice(0, 3).join(" · ");
    clearTimeout(splash.timer);
    splash.timer = setTimeout(() => {
      hideSplash();
      const current = state.game ? `${state.game.game}|${state.game.id}` : null;
      // Off, or the player already moved on to another game, or a game this
      // page only watches (the arena, Lichess): leave it.
      if (!$("autorestart").checked || current !== key || game.input === "watch") return;
      if (game.input === "realtime") act({ type: "reset" }); else $("new").click();
    }, SPLASH_S * 1000);
  }
  if (!ended && splash.ended) hideSplash();
  splash.ended = ended;
}
// Restarting a forgotten session is a recovery, never a loop: at most one
// automatic new game per RESTART_GAP_S, doubling while the server keeps
// answering "no game" (up to 30 s), so a confused session cannot hammer the
// server with new games.
const RESTART_GAP_S = 2;
const restartGate = { at: -Infinity, gap: RESTART_GAP_S };
function restartAllowed() {
  const now = performance.now() / 1000;
  if (ticks.restarting || now - restartGate.at < restartGate.gap) return false;
  restartGate.gap = now - restartGate.at < 2 * restartGate.gap ? Math.min(30, restartGate.gap * 2) : RESTART_GAP_S;
  restartGate.at = now;
  return true;
}
function hideSplash() { clearTimeout(splash.timer); $("splash").className = ""; }

// ---------------------------------------------------------------- lightning
// A strike's two ends arrive as ribbon pieces spawned together (the cloud and
// the craft it hit; a lone end is a strike to the land).  Between them a
// jagged ribbon: fractal midpoint displacement re-drawn every frame so it
// flickers, a camera-facing strip of glowing glass, fading as the ends age.
const bolt = { vao: null, buffers: null };
function worldOf(piece) {
  const depth = state.game && state.game.predict && state.game.predict.depth;
  const lift = depth ? (piece.h || 0) * DEPTH_LIFT : 0;
  return [piece.x, TILE.height + 0.2 + lift - planetDrop(piece.x, piece.z), piece.z];
}
function jagged(a, b, depth, roughness) {
  let points = [a, b];
  for (let level = 0, amp = roughness; level < depth; level++, amp *= 0.55) {
    const next = [points[0]];
    for (let i = 1; i < points.length; i++) {
      const p = points[i - 1], q = points[i];
      next.push([(p[0] + q[0]) / 2 + (Math.random() - 0.5) * amp, (p[1] + q[1]) / 2 + (Math.random() - 0.5) * amp * 0.5,
                 (p[2] + q[2]) / 2 + (Math.random() - 0.5) * amp], q);
    }
    points = next;
  }
  return points;
}
function strikeGroups() {
  const groups = new Map();
  for (const piece of state.ribbonEnds || []) { const key = Math.round(piece.t0 * 20); (groups.get(key) || groups.set(key, []).get(key)).push(piece); }
  return [...groups.values()];
}
// The rocket's headlights: two warm-white SPOT lights at the nose, aimed
// along the direction of travel (up the screen) and down onto the land.
// The craft flies ~0.2 above the board, so a shallow dip throws the beam
// ahead (a long ellipse on the land) instead of pooling under the nose.
const HEADLIGHT = { inner: 9 * Math.PI / 180, outer: 18 * Math.PI / 180, intensity: 70, dip: 0.16, spread: 0.16 };
function headlights(cam) {
  const game = state.game;
  if (isWalker(game)) return [];               // a walker carries no headlights
  if (state.flat || !game || game.input !== "realtime" || !game.you || !game.you.alive || !onPlanet()) return [];
  const ship = [...state.pieces.values()].find((p) => p.entity && p.lane === (game.predict ? game.predict.lane : 0) && p.removing == null);
  if (!ship) return [];
  const base = worldOf(ship), dir = norm([0, -HEADLIGHT.dip, -1]);
  const viewDir = (d) => { const a = M.apply(cam.view, base), b = M.apply(cam.view, [base[0] + d[0], base[1] + d[1], base[2] + d[2]]);
    return norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]); };
  const vdir = viewDir(dir), cone = [Math.cos(HEADLIGHT.inner), Math.cos(HEADLIGHT.outer)];
  return [-1, 1].map((side) => {
    const at = [base[0] + side * HEADLIGHT.spread, base[1] + 0.05, base[2] - ship.size * 0.9];
    return { view: M.apply(cam.view, at), colour: [1.0, 0.94, 0.82], level: 0, intensity: HEADLIGHT.intensity,
             spotDir: vdir, spotCone: cone };
  });
}
// A bomb's blast lights the scene in its own glow colour (the game's
// emissive palette entry for it), flaring and fading over its life.
// MARKER LIGHTS on the player's rocket (aeronautic: red port wingtip, green
// starboard, white tail, a white strobe double-flashing every 1.2 s), riding
// the drawn ship (nose up the screen: port is -x), scaled with its height.
// Not drawn from inside the cockpit.
function markerLights(now) {
  const game = state.game;
  if (isWalker(game)) return [];               // a walker has no wingtip lights or strobe: that is the rocket's
  if (state.flat || state.camMode === "nose" || !game || game.input !== "realtime" || !game.predict || !game.you || !game.you.alive) return [];
  const ship = [...state.pieces.values()].find((p) => p.entity && p.lane === game.predict.lane && p.removing == null);
  if (!ship) return [];
  // the drawn size grows with height (DEPTH_SCALE), as placeEntity draws it
  const depth = game.predict.depth, hRel = depth ? Math.max(0, Math.min(1, (ship.h || 0) / depth)) : 0;
  const grow = depth ? DEPTH_SCALE[0] + (DEPTH_SCALE[1] - DEPTH_SCALE[0]) * hRel : 1;
  const at = worldOf(ship), s = (ship.size || 0.5) * grow;
  const phase = now % 1.2, strobe = phase < 0.06 || (phase > 0.16 && phase < 0.22);
  // tiny balls set into the hull's surface (wingtips, tail, the strobe on the spine)
  const put = (dx, dz, up = 0.32) => [at[0] + dx * s, at[1] + up * s, at[2] + dz * s];
  const bulb = 0.045 * s;
  return [
    { at: put(-0.8, 0.3), material: "nav_red", size: bulb, light: [1.0, 0.08, 0.06], intensity: 0.8 },
    { at: put(0.8, 0.3), material: "nav_green", size: bulb, light: [0.08, 1.0, 0.3], intensity: 0.8 },
    { at: put(0, 1.2, 0.28), material: "nav_white", size: bulb * 0.9, light: [1, 1, 0.95], intensity: 0.5 },
    { at: put(0, 0.1, 0.42), material: strobe ? "nav_strobe" : "nav_off", size: bulb, light: [1, 1, 1], intensity: strobe ? 1.2 : 0.0 },
  ];
}
function blastLights(cam, now) {
  const glow = (state.game && state.game.emissive) || {}, lights = [];
  if (state.flat) return lights;
  for (const piece of state.pieces.values()) {
    if (!piece.entity || piece.removing != null || !glow[piece.color] || piece.ribbon) continue;
    if (!String(piece.color).startsWith("blast")) continue;
    const age = now - piece.t0, flare = Math.max(0, 1 - age / 0.5) * (0.75 + 0.25 * Math.random());
    const g = glow[piece.color], peak = Math.max(...g);
    const at = worldOf(piece);
    lights.push({ view: M.apply(cam.view, [at[0], at[1] + 0.3, at[2]]), colour: g.map((v) => v / peak), level: 0, intensity: 55 * flare });
  }
  return lights;
}
function lightningLights(cam, now) {
  const lights = [];
  if (state.flat) return lights;
  for (const group of strikeGroups()) {
    group.sort((a, b) => (b.h || 0) - (a.h || 0));
    const top = worldOf(group[0]);
    const bottom = group[1] ? worldOf(group[1]) : [top[0], 0.02 - planetDrop(top[0], top[2]), top[2]];
    const fade = Math.max(0, 1 - (now - group[0].t0) / 0.3), flicker = 0.6 + 0.4 * Math.random();
    const middle = top.map((v, k) => (v + bottom[k]) / 2);
    for (const [point, weight] of [[top, 0.7], [middle, 1.0], [bottom, 0.9]])
      lights.push({ view: M.apply(cam.view, point), colour: [0.78, 0.86, 1.0], level: 0, intensity: 38 * weight * fade * flicker });
  }
  return lights;
}
function drawLightning(cam, now, drawLit) {
  const ends = state.ribbonEnds || [];
  if (!ends.length || state.flat) return;
  const groups = new Map();
  for (const piece of ends) { const key = `${piece.color}:${Math.round(piece.t0 * 20)}`; (groups.get(key) || groups.set(key, []).get(key)).push(piece); }
  const pos = [], nrm = [], zpos = [], znrm = [];
  const eye = cam.eye || [0, 30, 0];
  for (const group of groups.values()) {
    group.sort((a, b) => (b.h || 0) - (a.h || 0));
    const zap = group[0].color === "zap";
    const top = worldOf(group[0]);
    const bottom = group[1] ? worldOf(group[1]) : [top[0], 0.02 - planetDrop(top[0], top[2]), top[2]];
    const path = zap ? [top, bottom] : jagged(top, bottom, 5, 0.9);
    const into = zap ? [zpos, znrm] : [pos, nrm];
    const fade = Math.max(0.2, 1 - (now - group[0].t0) / 0.25);
    const width = 0.07 * fade;
    for (let i = 1; i < path.length; i++) {
      const p = path[i - 1], q = path[i], dir = sub(q, p), toEye = sub(eye, p);
      const side = norm(cross(dir, toEye)).map((v) => v * width);
      const a = sub(p, side), b = [p[0] + side[0], p[1] + side[1], p[2] + side[2]];
      const c = sub(q, side), d = [q[0] + side[0], q[1] + side[1], q[2] + side[2]];
      const n = norm(toEye);
      for (const v of [a, b, d, a, d, c]) { into[0].push(...v); into[1].push(...n); }
    }
  }
  if (!pos.length && !zpos.length) return;
  if (!bolt.vao) {
    bolt.vao = gl.createVertexArray(); gl.bindVertexArray(bolt.vao);
    bolt.buffers = [0, 1].map((loc) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0); return b; });
    gl.bindVertexArray(null);
  }
  for (const [P, N, material] of [[pos, nrm, "lightning@glow"], [zpos, znrm, "zap@glow"]]) {
    if (!P.length) continue;
    gl.bindBuffer(gl.ARRAY_BUFFER, bolt.buffers[0]); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(P), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, bolt.buffers[1]); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(N), gl.DYNAMIC_DRAW);
    drawLit(bolt.vao, P.length / 3, M.identity(), state.materialIndex[material]);
  }
}

// ---------------------------------------------------------------- vital bars
// Shield (blue glass, left) and health (red glass, right): tubes standing just
// outside the arcade light strips along the field, each filled from the
// bottom of the screen in proportion to its value (eased).  Shown for a game
// whose ship carries health points (``you.health``).
const vitals = { shield: 0, health: 0, at: null, meshes: {} };
// litMesh bakes the material into each vertex, so each bar part has its own
// unit box, rebuilt when the material table is (re)uploaded.
// A glass cylinder along z (radius 0.5, length 1, capped): the bars' tube and core.
function cylinderFaces(id, segments = 24) {
  const faces = [], ring = (z) => Array.from({ length: segments }, (_, i) => {
    const a = (i / segments) * Math.PI * 2; return [0.5 * Math.cos(a), 0.5 + 0.5 * Math.sin(a), z]; });
  const lo = ring(-0.5), hi = ring(0.5), cLo = [0, 0.5, -0.5], cHi = [0, 0.5, 0.5];
  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    faces.push([lo[i], hi[i], hi[j], id], [lo[i], hi[j], lo[j], id], [cLo, lo[j], lo[i], id], [cHi, hi[i], hi[j], id]);
  }
  return faces;
}
function vitalBox(material) {
  const id = state.materialIndex[material], key = `${material}:${id}:${state.materialsVersion}`;
  if (!vitals.meshes[material] || vitals.meshes[material].key !== key)
    vitals.meshes[material] = { key, mesh: litMesh(cylinderFaces(id)) };
  return vitals.meshes[material].mesh;
}
function drawVitals(now, drawLit) {
  const game = state.game, you = game && game.you;
  if (state.flat || !you || !(you.health > 0 || vitals.health > 0.001) || game.input !== "realtime") return;
  const frame = vitals.at == null ? 1 : Math.min(0.1, now - vitals.at); vitals.at = now;
  const ease = 1 - Math.exp(-frame / 0.12);
  vitals.shield += ((you.shield || 0) - vitals.shield) * ease;
  vitals.health += ((you.health || 0) - vitals.health) * ease;
  const ext = state.extent, m = GLASS.margin * 0.6 + 0.55, length = ext.depth * 0.86, z1 = ext.depth * 0.93;
  // Each bar is drawn in short segments laid on the planet's curve.
  const SEG = 12;
  const segment = (mesh, x, y, za, zb, w) => {
    const zc = (za + zb) / 2, len = zb - za;
    const slope = onPlanet() ? Math.atan((zc - ext.depth / 2) / PLANET.radius) : 0;
    drawLit(mesh.vao, mesh.count, M.multiply(M.multiply(M.translate(x, y - planetDrop(x, zc), zc), M.rotateX(slope)),
      M.scale(w, w, len * 1.02)), null);
  };
  const bar = (x, value, material) => {
    const w = 0.34;
    const tube = vitalBox("bar_tube"), core = vitalBox(material), top = z1 - length;
    for (let i = 0; i < SEG; i++) segment(tube, x, 0.05, top + (i * length) / SEG, top + ((i + 1) * length) / SEG, w);
    const fill = Math.max(0, Math.min(1, value)) * length;
    if (fill > 0.01) {
      const n = Math.max(1, Math.ceil((fill / length) * SEG)), start = z1 - fill;
      for (let i = 0; i < n; i++) segment(core, x, 0.08, start + (i * fill) / n, start + ((i + 1) * fill) / n, w * 0.7);
    }
  };
  // THE THROTTLE (yellow, from the centre out: the throttle's fraction, the
  // page's own predicted speed so it answers at once) inside a glass tube as
  // long as the FUEL left; and the BOSS's health (green) while it flies.
  const flight = game.flight, boss = game.boss;
  if (flight) {
    const spd = pred.ship && pred.ship.spd != null ? pred.ship.spd : flight.min + flight.throttle * (flight.max - flight.min);
    const target = Math.max(0, Math.min(1, (spd - flight.min) / (flight.max - flight.min)));
    vitals.throttle = (vitals.throttle ?? target) + (target - (vitals.throttle ?? target)) * ease;
    vitals.fuel = (vitals.fuel ?? flight.fuel) + (flight.fuel - (vitals.fuel ?? flight.fuel)) * ease;
  }
  vitals.boss = boss ? (vitals.boss ?? 1) + (boss.hp / boss.max - (vitals.boss ?? 1)) * ease : 0;
  // A horizontal bar from the centre out: in the cockpit on the plate (plate
  // units), on the map across the field (board units).
  const across = (place, centre, half, value, material, fuelLeft) => {
    const tube = vitalBox("bar_tube"), core = vitalBox(material);
    place(tube, centre, half * 2 * Math.max(0.04, fuelLeft ?? 1), 1.0);
    if (value > 0.005) place(core, centre, half * 2 * value, 0.7);
  };
  if (noseCam()) {
    // In the cockpit the bars leave the map: they stand on the hull's inner
    // wall either side of the window (plate units; the window spans +-0.8),
    // their glowing glass filling from the bottom.
    if (!COCKPIT.model) return;
    const wall = (x, value, material) => {
      const L = 0.95, y0 = COCKPIT.window.cy - 0.05, r = 0.075;
      const upright = (mesh, centre, length, radius) => drawLit(mesh.vao, mesh.count,
        M.multiply(COCKPIT.model, M.multiply(M.translate(x, centre, 0.02), M.multiply(M.rotateX(-Math.PI / 2), M.scale(radius, radius, length)))), null);
      upright(vitalBox("bar_tube"), y0, L, r);
      const fill = Math.max(0, Math.min(1, value)) * L;
      if (fill > 0.005) upright(vitalBox(material), y0 - L / 2 + fill / 2, fill, r * 0.7);
    };
    wall(-0.93, vitals.shield, "bar_shield");
    wall(0.93, vitals.health, "bar_health");
    // throttle under the window, boss health over it (lying along x)
    const plate = (y) => (mesh, cx, length, k) => drawLit(mesh.vao, mesh.count,
      M.multiply(COCKPIT.model, M.multiply(M.translate(cx, y, 0.02), M.multiply(M.rotateY(Math.PI / 2), M.scale(0.06 * k, 0.06 * k, length)))), null);
    if (flight) across(plate(COCKPIT.window.cy - COCKPIT.window.b - 0.12), 0, 0.6, vitals.throttle, "bar_throttle", vitals.fuel);
    if (boss) across(plate(COCKPIT.window.cy + COCKPIT.window.b + 0.1), 0, 0.6, vitals.boss, "bar_boss");
    return;
  }
  bar(-m, vitals.shield, "bar_shield");
  bar(ext.width + m, vitals.health, "bar_health");
  // on the map: throttle along the bottom edge, boss health along the top
  const lying = (z) => (mesh, cx, length, k) => drawLit(mesh.vao, mesh.count,
    M.multiply(M.translate(cx, 0.06 + (k < 1 ? 0.03 : 0) - planetDrop(cx, z), z), M.multiply(M.rotateY(Math.PI / 2), M.scale(0.3 * k, 0.3 * k, length))), null);
  if (flight) across(lying(ext.depth - 0.55), ext.width / 2, ext.width * 0.36, vitals.throttle, "bar_throttle", vitals.fuel);
  if (boss) across(lying(0.45), ext.width / 2, ext.width * 0.36, vitals.boss, "bar_boss");
}

// ---------------------------------------------------------------- the craft's shield
// A bubble around the player's ship, as charged as the server says (``you``'s
// shield); a hit it absorbs (the charge falling) flashes it.  Drawn with the
// glass: blended, no depth write.
const SHIELD_STEPS = 6;
const shieldView = { charge: 0, last: null, flashAt: -10 };
function noteShield(game, now) {
  const charge = game && game.you ? Number(game.you.shield || 0) : 0;
  if (shieldView.last != null && charge < shieldView.last - 0.02) shieldView.flashAt = now;
  shieldView.last = charge; shieldView.charge = charge;
}
// A geodesic sphere: an icosahedron subdivided ``levels`` times, projected to
// the unit sphere, FLAT facets (each triangle its own normal) so it reads as
// cut glass.
function geodesicMesh(levels = 2) {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
               [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map(norm);
  let faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
               [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let l = 0; l < levels; l++) {
    const cache = new Map(), next = [];
    const mid = (a, b) => { const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (!cache.has(key)) { verts.push(norm(verts[a].map((v, k) => (v + verts[b][k]) / 2))); cache.set(key, verts.length - 1); }
      return cache.get(key); };
    for (const [a, b, c] of faces) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    faces = next;
  }
  return faces.map(([a, b, c]) => [verts[a], verts[b], verts[c]]);
}
const shieldFacets = {};
function shieldGlobe(material) {
  const id = state.materialIndex[material], key = `${id}:${state.materialsVersion}`;
  if (!shieldFacets[material] || shieldFacets[material].key !== key)
    shieldFacets[material] = { key, mesh: litMesh(geodesicMesh(2).map(([a, b, c]) => [a, b, c, id])) };
  return shieldFacets[material].mesh;
}
// THE COCKPIT (first person): the rocket's hull around the eye with a
// window cut through it -- the hull plate minus a rounded window (a
// superellipse), triangulated as the ring between the window's edge and the
// plate's, plus the window's sill (its cut edge, receding), in the rocket's
// own material and lit with the scene.  It rides the camera.
const COCKPIT = { fov: 1.05, distance: 0.1, window: { a: 0.8, b: 0.58, p: 4.5, cy: 0.14 }, sill: 0.09, mesh: null };
function cockpitMesh() {
  const material = state.materialIndex.rocket;
  if (COCKPIT.mesh && COCKPIT.mesh.material === material) return COCKPIT.mesh;
  const { a, b, p, cy } = COCKPIT.window, N = 128, faces = [];
  const rim = (k) => {
    const th = (k / N) * 2 * Math.PI, c = Math.cos(th), s = Math.sin(th);
    const r = Math.pow(Math.pow(Math.abs(c) / a, p) + Math.pow(Math.abs(s) / b, p), -1 / p);   // the window's edge
    const out = 2.2 / Math.max(Math.abs(c), Math.abs(s));                                        // well past the screen (the lens bends edges)
    return { inner: [r * c, cy + r * s, 0], outer: [out * c, cy + out * s, 0], sill: [r * c, cy + r * s, -COCKPIT.sill] };
  };
  for (let k = 0; k < N; k++) {
    const q = rim(k), r = rim(k + 1);
    faces.push([q.inner, q.outer, r.outer, material], [q.inner, r.outer, r.inner, material]);   // the plate, facing the eye
    faces.push([q.sill, q.inner, r.inner, material], [q.sill, r.inner, r.sill, material]);      // the cut edge, receding
  }
  COCKPIT.mesh = { ...litMesh(faces), material };
  return COCKPIT.mesh;
}
function drawCockpit(cam, drawLit) {
  if (!cam.nose) return;
  const mesh = cockpitMesh(), d = COCKPIT.distance, h = d * Math.tan(COCKPIT.fov / 2), w = h * (cam.aspect || 1);
  const model = M.multiply(M.invert(cam.view), M.multiply(M.translate(0, 0, -d), M.scale(w, h, h)));
  drawLit(mesh.vao, mesh.count, model, mesh.material);
  COCKPIT.model = model;                            // the vitals mount on this plate (drawVitals)
}
// THE LASER'S BEAM: the engine's beam (``you.beam``: its length, the first
// thing it burns) drawn as a straight emissive ray from the nose up the
// screen -- shown at once while the page's own trigger holds a laser.
function drawBeam(now, drawLit) {
  const game = state.game, you = game && game.you, a = game && game.arsenal;
  if (isWalker(game)) return;
  if (state.flat || !you || !you.alive || !a) return;
  const holding = held.has("fire") || (pads.right && pads.right.active);
  const length = you.beam > 0 ? you.beam : holding && (state.primarySlot || 0) === 1 && a.primary[1].owned ? a.beam_range : 0;
  if (length <= 0) return;
  const ship = [...state.pieces.values()].find((p) => p.entity && p.lane === game.predict.lane && p.removing == null);
  if (!ship) return;
  const at = worldOf(ship), nose = (ship.size || 0.5) * 1.1, flicker = 0.85 + 0.15 * Math.sin(now * 90);
  const tube = vitalBox("beam");
  drawLit(tube.vao, tube.count, M.multiply(M.translate(at[0], at[1] - 0.04, at[2] - nose - length / 2),
    M.scale(0.07 * flicker, 0.07 * flicker, length)), null);
}
function drawShield(now, drawLit) {
  const game = state.game;
  if (isWalker(game)) return;                  // (no rocket's shield bubble on a walker)
  if (state.camMode === "nose") return;             // first person: the eye is inside the shield
  if (state.flat || !game || game.input !== "realtime" || !game.you || !game.you.alive) return;
  const ship = [...state.pieces.values()].find((p) => p.entity && p.lane === (game.predict ? game.predict.lane : 0) && p.removing == null);
  if (!ship) return;
  const flash = Math.max(0, 1 - (now - shieldView.flashAt) / 0.4);
  // At rest only a faint rim (the craft must show through a full shield);
  // each absorbed hit flares it.
  const charge = Math.min(1, 0.18 * shieldView.charge + flash);
  if (charge <= 0.02) return;
  // A bounding sphere of geodesic glass enclosing the whole craft (its drawn
  // size grows with height, so the globe follows it), flaring on a hit.
  const depth = game.predict && game.predict.depth, hRel = depth ? Math.max(0, Math.min(1, (ship.h || 0) / depth)) : null;
  const grow = hRel == null ? 1 : DEPTH_SCALE[0] + (DEPTH_SCALE[1] - DEPTH_SCALE[0]) * hRel;
  const lift = hRel == null ? 0 : (ship.h || 0) * DEPTH_LIFT;
  const step = Math.round(charge * (SHIELD_STEPS - 1)), r = ship.size * grow * 1.55 * (1 + 0.12 * flash);
  const globe = shieldGlobe(`shield_${step}`);
  const model = M.multiply(M.translate(ship.x, TILE.height + 0.04 + lift + ship.size * grow * 0.25 - planetDrop(ship.x, ship.z), ship.z), M.scale(r, r, r));
  drawLit(globe.vao, globe.count, model, null);
}

// ---------------------------------------------------------------- arcade lights
// The ring can be switched off entirely (the Arcade lights checkbox, kept per
// browser); off, draw() neither lights nor draws a single bulb.
function arcadeEnabled() { const box = $("arcade"); return !box || box.checked; }
// The scene contract: a layer is drawn only if the game declares it (its
// state's ``scene``; a game that says nothing has every layer) and the
// viewer's control for it is on.
function layerOn(name) {
  const declared = !(state.game && state.game.scene && state.game.scene[name] === false);
  return declared && viewerWants(name === "arcade" ? "arcade" : "plate");
}
// The viewer's switch for a layer, PER GAME: a choice made in a game is kept
// for that game; with none, the game's own preference (``scene_prefer``,
// e.g. bullet hell's glowing plate and arcade lights) and otherwise off.
function viewerWants(control) {
  const game = state.game ? state.game.game : "";
  try { const stored = localStorage.getItem(`meowdoku_${control}_${game}`); if (stored) return stored === "on"; } catch {}
  const prefer = state.game && state.game.scene_prefer;
  if (prefer) return control === "arcade" ? !!prefer.arcade : control === "projection" ? !!prefer.projection : !!(prefer.glass || prefer.floor);
  return false;
}
function syncLayerControls() {
  for (const id of ["arcade", "plate", "projection"]) { const box = $(id); if (box) box.checked = viewerWants(id); }
}
// The switches show and set the current game's choice (see viewerWants).
for (const id of ["arcade", "plate", "projection"]) {
  const box = $(id); if (!box) continue;
  box.addEventListener("change", () => {
    try { localStorage.setItem(`meowdoku_${id}_${state.game ? state.game.game : ""}`, box.checked ? "on" : "off"); } catch {}
    box.blur();
  });
}
// Pinball-style bulbs ring the board just outside its glass edge.  Idle: a slow
// chase.  When the game's declared score rises: a comet walks the ring with a
// fading tail.  On a win: the whole ring flashes.  Their colour and activity
// also tint the fill light, so the pieces catch the show.
//: Emission steps of a bulb (linear), the step drawn follows its brightness.
const ARCADE_LEVELS = [0.15, 0.6, 1.3, 2.4];
// Colours 0-3 are the ring's own idle colours; 4+ are event colours: a side's
// colour (white gold, black violet, checkers red) and the amber of a check.
const ARCADE = { lamp: 2.6, spacing: 0.55,
                 colours: [[1.0, 0.18, 0.12], [1.0, 0.62, 0.08], [1.0, 0.95, 0.85], [0.2, 0.55, 1.0],
                           [1.0, 0.82, 0.35], [0.62, 0.3, 1.0], [1.0, 0.12, 0.1], [1.0, 0.55, 0.0]],
                 sideColour: { white: 4, black: 5, red: 6, player: 4 }, checkColour: 7,
                 events: [], peak: 0, key: null, last: null };

// Score events, read from what each state DECLARES: a board game's material
// (a capture is the other side's material falling), its check and outcome;
// any game's running score.  Each event's size is normalised by the largest
// gain seen so far in this game (the "peak"), floored at a pawn's worth, so
// the scale follows the game: the first queen taken is huge, a later pawn
// small.  The size picks the pattern.
function arcadeEvents(game, now) {
  const key = `${game.game}|${game.id}`, last = ARCADE.last;
  ARCADE.last = { material: game.material ? { ...game.material } : null, score: game.score,
                  check: !!game.check, outcome: game.outcome ?? null, won: !!game.won, turn: game.status };
  if (ARCADE.key !== key || !last) { ARCADE.key = key; ARCADE.peak = 0; ARCADE.events = []; return; }
  const add = (kind, side, value) => {
    ARCADE.peak = Math.max(ARCADE.peak, value);
    const size = value / Math.max(ARCADE.peak, 1);
    const pattern = kind === "win" ? "flash" : kind === "check" ? "check" : size >= 0.67 ? "twin" : size >= 0.34 ? "comet" : "sparkle";
    ARCADE.events.push({ pattern, side, size, at: now });
    if (ARCADE.events.length > 12) ARCADE.events.shift();
  };
  if (game.material && last.material) {
    const sides = game.sides || Object.keys(game.material);
    for (const side of sides) {
      const drop = (last.material[side] ?? 0) - (game.material[side] ?? 0);
      const scorer = sides.find((s) => s !== side);
      if (drop > 0 && scorer) add("capture", scorer, drop);
    }
    if (game.check && !last.check) {
      const checked = (game.status || "").split(" ")[0], giver = sides.find((s) => s !== checked);
      if (giver) add("check", checked, 1);
    }
    if (game.outcome && game.outcome !== "draw" && last.outcome !== game.outcome) add("win", game.outcome, 1);
  } else if (game.score != null && last.score != null && game.score > last.score) {
    add("score", "player", game.score - last.score);
  }
  if (game.won && !last.won && !game.material) add("win", "player", 1);
}
function arcadeBulbs() {
  const ext = state.extent || { width: 6, depth: 6 }, m = GLASS.margin * 0.6;
  const x0 = -m, z0 = -m, x1 = ext.width + m, z1 = ext.depth + m;
  const perimeter = 2 * ((x1 - x0) + (z1 - z0)), count = Math.max(16, Math.round(perimeter / ARCADE.spacing));
  const bulbs = [];
  for (let i = 0; i < count; i++) {
    let s = i / count * perimeter, x, z;
    if (s < x1 - x0) { x = x0 + s; z = z0; } else if ((s -= x1 - x0) < z1 - z0) { x = x1; z = z0 + s; }
    else if ((s -= z1 - z0) < x1 - x0) { x = x1 - s; z = z1; } else { s -= x1 - x0; x = x0; z = z1 - s; }
    bulbs.push([x, 0.45 - planetDrop(x, z), z]);   // on posts, so their light falls onto the tiles
  }
  return bulbs;
}
// Where a side's events start on the ring: white (and a solo player) at the
// near edge, black at the far edge; the bulb nearest that edge's middle.
function sideOrigin(side, bulbs) {
  const ext = state.extent || { width: 6, depth: 6 };
  const near = side === "black" ? [ext.width / 2, -1] : [ext.width / 2, ext.depth + 1];
  let best = 0, bestDistance = Infinity;
  bulbs.forEach((p, i) => { const d = Math.hypot(p[0] - near[0], p[2] - near[1]); if (d < bestDistance) { bestDistance = d; best = i; } });
  return best;
}
// One bulb's light now: {level 0..1, colour index}.  Idle chase underneath;
// every live event on top, the brightest wins.
function arcadeBulb(i, bulbs, now) {
  const count = bulbs.length;
  let level = ((i + Math.floor(now * 6)) % 6 === 0) ? 0.55 : 0.1, ci = i % 4;
  for (const event of ARCADE.events) {
    const age = now - event.at, home = sideOrigin(event.side, bulbs), colour = ARCADE.sideColour[event.side] ?? 4;
    const ring = (a) => Math.min((a - i + count) % count, (i - a + count) % count);
    let l = 0, c = colour;
    if (event.pattern === "flash" && age < 2.6) l = Math.floor(age * 8) % 2 === 0 ? 1.0 : 0.15;
    else if (event.pattern === "check" && age < 1.4) {
      l = ring(home) < count / 4 ? 0.9 * (0.5 + 0.5 * Math.sin(age * 18)) : 0; c = ARCADE.checkColour;
    } else if (event.pattern === "sparkle" && age < 0.8) {
      const near = ring(home) < count / 4, twinkle = Math.sin(i * 12.9898 + Math.floor(age * 14) * 78.233) > 0.6;
      l = near && twinkle ? 0.5 + 0.5 * event.size : 0;
    } else if ((event.pattern === "comet" || event.pattern === "twin") && age < 1.6) {
      const speed = 30 + 30 * event.size, tail = 6 + 10 * event.size, travelled = age * speed;
      const heads = event.pattern === "twin" ? [home + travelled, home - travelled] : [home + travelled];
      for (const h of heads) {
        const head = ((Math.round(h) % count) + count) % count;
        const behind = event.pattern === "twin" && h < home ? (i - head + count) % count : (head - i + count) % count;
        if (behind < tail) l = Math.max(l, 1.0 - behind / tail);
      }
      if (event.pattern === "twin") l = Math.max(l, 0.35 * Math.max(0, 1 - age / 1.6));   // the whole ring pulses
    }
    if (l > level) { level = l; ci = c; }
  }
  ARCADE.events = ARCADE.events.filter((event) => now - event.at < 3);
  return { level, ci };
}

// The key light is a lamp in the scene, not a page angle: a small glowing body
// with inertia, pulled by a critically damped spring toward a target that
// lives in the board's 3D space -- a slow wander low over the pieces (so it
// rakes their sides and their shadows swing), or, in pointer mode, the point
// above the board under the pointer.  Its light falls off with distance, so
// what it passes close to glows and the far side dims.
const LAMP = { omega: 1.6 };
// The sun: a soft light at an angle over the board, placed by the user's local
// time of day -- its bearing goes once round the board per day, it stands
// highest at noon, low and warm at dawn and dusk, and at night it is a dim
// blue moon on the far side.
function sunState() {
  const d = new Date(), hours = d.getHours() + d.getMinutes() / 60 + d.getSeconds() / 3600;
  const day = Math.sin((hours - 6) / 24 * 2 * Math.PI);          // +1 noon, 0 dawn/dusk, -1 midnight
  const azimuth = hours / 24 * 2 * Math.PI;
  const elevation = (day >= 0 ? 15 + 55 * day : 35) * Math.PI / 180;
  const warm = Math.max(0, 1 - Math.abs(day) * 3) * (day > -0.2 ? 1 : 0);
  const colour = day >= 0 ? mix([1.0, 0.97, 0.9], [1.0, 0.72, 0.45], warm) : mix([0.55, 0.65, 1.0], [1.0, 0.72, 0.45], warm);
  const strength = day >= 0 ? 0.75 + 0.25 * day : 0.42;
  return { azimuth, elevation, colour, strength };
}
// THE SUN ON THE GLOBE (a spherical game publishes ``globe``): the sun's
// direction is fixed in the planet's frame, so its place in the local sky is
// the player's place and heading -- elevation from the local vertical, bearing
// from the heading (turned on by the page's predicted yaw, as the world is).
// Board frame: x right, y up, -z ahead.
function globeSun(now) {
  const game = state.game, g = game && game.globe;
  if (!g || state.flat) return null;
  const u = g.u, s = norm(g.sun);
  let h = g.h;
  const turn = pred.ship && pred.ship.yaw != null ? pred.ship.yaw - g.yaw : 0;
  if (turn) { const r = cross(h, u); h = norm(h.map((v, k) => v * Math.cos(turn) + r[k] * Math.sin(turn))); }
  const r = cross(h, u), dir = [dot3(s, r), dot3(s, u), -dot3(s, h)];
  const target = norm(dir);
  // Eased: the published place arrives a few times a second.
  if (!SKY.dir) SKY.dir = target;
  const k = 1 - Math.exp(-Math.min(0.1, Math.max(0, now - (SKY.at ?? now))) / 0.35);
  SKY.at = now;
  SKY.dir = norm(SKY.dir.map((v, i) => v + (target[i] - v) * k));
  return SKY.dir;
}
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// The atmosphere: optical depths at the zenith (Rayleigh per colour, Mie),
// the air mass along a ray at elevation e (Kasten & Young), and the sun's
// colour after crossing it -- what lights the Phong scene and the sky.
const SKY = { tauR: [0.07, 0.16, 0.36], tauM: 0.03, dir: null, at: null, prog: null, vao: null };
function airMass(elevation) {
  const deg = Math.max(-2.5, elevation * 180 / Math.PI);
  return 1 / (Math.sin(Math.max(deg, 0) * Math.PI / 180) + 0.50572 * Math.pow(deg + 6.07995, -1.6364));
}
function sunThroughAir(dir) {
  const e = Math.asin(Math.max(-1, Math.min(1, dir[1]))), m = airMass(e);
  const above = Math.max(0, Math.min(1, (e * 180 / Math.PI + 3) / 6));        // the disc sinking through the horizon
  return SKY.tauR.map((t) => Math.exp(-(t + SKY.tauM) * m) * above);
}
function lampTarget(now, ext) {
  const span = Math.max(ext.width, ext.depth);
  if (light.follow && light.pointer) return [light.pointer[0], 0.55 * span, light.pointer[1]];
  const globe = globeSun(now);
  if (globe) {
    const reach = 0.95 * span;
    return [ext.width / 2 + reach * globe[0], reach * Math.max(0.05, globe[1]) + 0.3, ext.depth / 2 + reach * globe[2]];
  }
  const sun = sunState(), reach = 0.95 * span;
  return [ext.width / 2 + reach * Math.cos(sun.elevation) * Math.cos(sun.azimuth),
          reach * Math.sin(sun.elevation) + 0.3,
          ext.depth / 2 + reach * Math.cos(sun.elevation) * Math.sin(sun.azimuth)];
}
function updateLight(now) {
  const ext = state.extent || { width: 6, depth: 6 }, span = Math.max(ext.width, ext.depth);
  if (state.flat) light.position = [ext.width / 2, light.height * span * 0.9, ext.depth / 2];   // co-located with the camera, straight down
  else {
    const target = lampTarget(now, ext);
    if (!light.body) light.body = { p: target.slice(), v: [0, 0, 0], at: now };
    const body = light.body, dt = Math.min(0.05, Math.max(0, now - body.at)), w = LAMP.omega;
    body.at = now;
    // Critically damped spring, integrated implicitly: stable at any frame time.
    for (let k = 0; k < 3; k++) {
      const accel = w * w * (target[k] - body.p[k]) - 2 * w * body.v[k];
      body.v[k] = (body.v[k] + dt * accel) / (1 + dt * dt * w * w + 2 * dt * w);
      body.p[k] += body.v[k] * dt;
    }
    body.p[1] = Math.max(0.45, body.p[1]);
    light.position = body.p.slice();
  }
  const target = [ext.width / 2, 0, ext.depth / 2];
  const depth = Math.hypot(...sub(light.position, target));
  light.view = M.lookAt(light.position, target, [0, 1, 0]);
  light.viewProj = M.multiply(M.ortho(span * 0.95, depth * 0.1, depth * 2.5), light.view);
}

// ---------------------------------------------------------------- pieces and animation
// Flat view: pieces lie on the tiles, squashed along the extrusion axis.
const FLAT_THICK = 0.04, FLAT_LIFT = 0.02, FLAT_OUTLINE = 1.07;
const ease = { out: (t) => 1 - (1 - t) ** 3, in: (t) => t ** 3, bounce(t) { return t < 0.85 ? t / 0.85 : 1 + 0.15 * Math.sin((t - 0.85) / 0.15 * Math.PI) * (1 - t); } };
const LEAN = { cat: 0.35, x_mark: 0.35, heart: 0.35, disc: 0.0, letter: 0.3, chess: 0.55 };
// The goldfish lives move like the hearts they replaced; a mouse drops in like an X.
const MESH_FAMILY = { fish: "heart", mouse: "x_mark" };
const family = (mesh) => (mesh.length === 1 ? "letter" : ["king", "queen", "rook", "bishop", "knight", "pawn"].includes(mesh) ? "chess" : MESH_FAMILY[mesh] || mesh);
// The power-up tray: a row under the lives.  Each item sits on its own
// button tile (it reads as a button from above); its count is set small
// UNDER the tile, outside the button, so it stays understated.
const TRAY = { row: 2.0, spacing: 1.5, tile: 1.0, countDrop: 0.8 };
function trayCentre(game, k) { return [game.cols / 2 + (Number(k) - 1) * TRAY.spacing, game.rows + TRAY.row]; }
function trayPosition(game, cell) {
  const [, k, part] = cell.split(":");
  const [x, z] = trayCentre(game, k);
  return part === "count" ? [x, z + TRAY.countDrop, "traycount"] : [x, z, "tray"];
}
function trayButtonFaces(game) {
  // One raised tile per power-up, in the board's own tile finish.
  const kinds = Object.keys(game.powerups || {}).length, material = (game.cells || [])[0];
  if (!kinds || !material) return null;
  const faces = [];
  for (let k = 0; k < kinds; k++) {
    const [x, z] = trayCentre(game, k);
    faces.push(...buttonFaces(1, 1, () => material, [x - TRAY.tile / 2, z - TRAY.tile / 2], TRAY.tile));
  }
  return faces;
}
function cellPosition(game, cell, rackPiece) {
  if (typeof cell === "string" && cell.startsWith("tray:")) return trayPosition(game, cell);
  if (typeof cell === "string" && cell.startsWith("hud:")) {
    const k = Number(cell.slice(4)); return [game.cols / 2 + (k - 1) * 0.9, game.rows + 0.9, "hud"];
  }
  if (rackPiece) { const rack = rackLayout(game), r = Math.floor(cell / rack.cols), c = cell % rack.cols;
    return [rack.origin[0] + (c + 0.5) * rack.size, rack.origin[1] + (r + 0.5) * rack.size, "rack", rack.size]; }
  return [(cell % game.cols) + 0.5, Math.floor(cell / game.cols) + 0.5, "board"];
}
// ---------------------------------------------------------------- the walker (World Walk)
// A game whose player is a walker (``game.avatar === "walker"``) has no ship: no throttle to hold, no marker
// lights, strobe, headlights or shield.  The page predicts its walk with the server's own laws, as
// WebAssembly cores of the very text the law calls: ``walk_speed`` (base speed x the node's surface material
// factor over 1 + 6 slope: the table below is the server's, node by node), ``walk_route`` (the turn, and how
// squarely it faces the next waypoint), ``walk_drive`` (the pace it eases to: nothing asked, it stops), then
// ``sphere_turn`` and ``sphere_advance`` as for any body on the globe.  The click path is the server's
// (``game.walk.route``: node ids); free walking cancels it, as in the law.
const walk = { version: null, nodes: null, mat: null, slope: null, routeId: 0, path: null };
const isWalker = (game = state.game) => !!(game && game.avatar === "walker");
// (a walker has no rocket to sit in: the first-person cockpit is the ship's)
const noseCam = () => state.camMode === "nose" && !isWalker();
function fibonacciNodes(n) {          // roads.fibonacci_sphere: the nodes' places (the world's data, not physics)
  const out = new Float64Array(n * 3), golden = Math.PI * (3 - Math.sqrt(5));
  for (let k = 0; k < n; k++) {
    const kk = k + 0.5, z = 1 - 2 * kk / n, r = Math.sqrt(1 - z * z), phi = kk * golden;
    out[k * 3] = r * Math.cos(phi); out[k * 3 + 1] = r * Math.sin(phi); out[k * 3 + 2] = z;
  }
  return out;
}
function syncWalk(game) {
  const w = game && game.walk;
  if (!isWalker(game) || !w) return;
  if (!walk.nodes || walk.nodes.length !== w.nodes * 3) walk.nodes = fibonacciNodes(w.nodes);
  if (walk.version !== w.version || !walk.mat) {
    walk.version = w.version;
    walk.mat = Float64Array.from(w.mat, (ch) => Number(ch));
    walk.slope = Float64Array.from(atob(w.slope), (ch) => ch.charCodeAt(0) / 100);
  }
  const route = w.route;
  if (route && route.id !== walk.routeId) {         // a new click path: the page starts down it at once
    walk.routeId = route.id; walk.path = route.path.slice(1);
    if (pred.ship) { pred.ship.wp = 0; pred.ship.route = 1; }
  } else if (!route && walk.path && game.walking && !game.walking.active) walk.path = null;
}
function nearestNode(u, v, w) {
  const n = walk.nodes; let best = 0, bestDot = -2;
  for (let k = 0; k < n.length; k += 3) {
    const d = n[k] * u + n[k + 1] * v + n[k + 2] * w;
    if (d > bestDot) { bestDot = d; best = k / 3; }
  }
  return best;
}
function stepWalker(ship, input, seconds, motion) {
  const W = motion.walk, R = W.radius;
  if (!pred.walkSpeed || !pred.walkDrive || !pred.walkRoute || !pred.turn || !pred.advance || !pred.travel || !walk.nodes || !walk.mat || ship.ux == null) return;
  for (let left = seconds; left > 1e-9;) {
    const dt = Math.min(PRED_STEP, left); left -= dt;
    const node = nearestNode(ship.ux, ship.uy, ship.uz);
    const top = pred.walkSpeed.invoke({ base: W.base, mat: walk.mat[node], slope: walk.slope[node] }).speed;
    const turnIn = input.move_axis || 0, walkIn = input.move_axis_y || 0;
    const held = Math.abs(turnIn) > W.dead || Math.abs(walkIn) > W.dead ? 1 : 0;
    const path = walk.path, len = path ? path.length : 0;
    const route = (ship.route || 0) * (1 - held) * (len ? 1 : 0);
    const k = len ? path[Math.min(Math.max(ship.wp || 0, 0), len - 1)] : 0;
    const r = pred.walkRoute.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: ship.hx, hy: ship.hy, hz: ship.hz,
      wx: walk.nodes[k * 3], wy: walk.nodes[k * 3 + 1], wz: walk.nodes[k * 3 + 2], route, turn_in: turnIn,
      rate: W.turn, route_rate: W.route_turn, reach: W.reach, dt, radius: R });
    const spd = pred.walkDrive.invoke({ spd: ship.spd || 0, top, fwd: -walkIn, route, align: r.align, dt, tau: W.tau }).spd_next;
    const t = pred.turn.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: ship.hx, hy: ship.hy, hz: ship.hz, angle: r.angle });
    const a = pred.advance.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: t.hx_next, hy: t.hy_next, hz: t.hz_next,
      distance: spd * dt, radius: R });
    const g = pred.travel.invoke({ u: ship.map_u, v: ship.map_v, gx: ship.map_gx, gy: ship.map_gy, turn: r.angle, speed: spd, dt });
    ship.map_u = g.u_next; ship.map_v = g.v_next; ship.map_gx = g.gx_next; ship.map_gy = g.gy_next;
    ship.yaw = (ship.yaw ?? 0) + r.angle; ship.vyaw = r.angle / dt; ship.vspd = 0;
    const walled = walk.mat[nearestNode(a.ux_next, a.uy_next, a.uz_next)] > W.wall - 0.5;
    if (walled) {                                     // a wall bars its node: it turns in place
      ship.hx = t.hx_next; ship.hy = t.hy_next; ship.hz = t.hz_next; ship.spd = 0;
    } else {
      ship.ux = a.ux_next; ship.uy = a.uy_next; ship.uz = a.uz_next; ship.hx = a.hx_next; ship.hy = a.hy_next; ship.hz = a.hz_next; ship.spd = spd;
    }
    if (route) { ship.wp = (ship.wp || 0) + r.arrived; ship.route = ship.wp < len ? 1 : 0; }
    else ship.route = 0;
  }
}
// The click path as the page draws it: from the walker to the goal, along the nodes it has yet to reach.
function walkRouteLine() {
  const ship = pred.ship, path = walk.path;
  if (!ship || !path || !ship.route || ship.ux == null || !walk.nodes) return null;
  const line = [[ship.ux, ship.uy, ship.uz]];
  for (let i = Math.max(0, ship.wp || 0); i < path.length; i++) line.push([walk.nodes[path[i] * 3], walk.nodes[path[i] * 3 + 1], walk.nodes[path[i] * 3 + 2]]);
  return line.length > 1 ? line : null;
}
// A tap on a walker's globe: the board point under the finger -> the patch about the walker -> the globe.
// "walk there" sends the point as a click path; a build tool sends it to be placed on the road beside it.
function walkTap(e) {
  const game = state.game, f = LAND.frame, patch = game && game.predict && game.predict.patch, p = boardPlane(e.clientX, e.clientY);
  if (!p || !f || !patch) return;
  const u = patchToSphere(p.x - patch.cx, patch.cy - p.z, f);
  const tool = game.tool || "walk";
  act(tool === "walk" ? { type: "worldwalk_goto", u } : { type: "worldwalk_place", tool, u });
}
// ---------------------------------------------------------------- ship prediction
// Client-side prediction with server reconciliation.  The player's ship moves
// by the server law's own motion function (entity_motion.damped_axis),
// compiled to WebAssembly.  Input is recorded as INTERVALS -- the controls
// held and for how long, a new one whenever the controls change -- and each
// tick sends the closed intervals, numbered.  The server lands each as its
// own round with those controls; its reply names the last one landed (ack)
// and the ship after it (you).  The page resets to that and replays every
// later interval plus the one still open, so the ship answers at once and
// lands where the server puts it.  What little differs (the two sides'
// integration step sizes) eases out over ~60 ms.
const pred = { program: null, seq: 0, sent: [], closed: [], open: null, ship: null, offset: [0, 0], at: null, lane: null,
               shots: [], shotSeq: 0, lastShot: -10 };
window.meowdokuDebug = { pred, get sock() { return sock; } };   // read-only inspection handle (play tests)
const sameControls = (a, b) => Object.keys(a).every((k) => a[k] === b[k]);
// ERROR SYNC never jumps the picture.  A correction (the server's word against
// what is drawn) is eased away over a time that grows with its size -- a
// hair in ~60 ms, a unit or more over up to ~180 ms -- so a large disagreement
// glides rather than pops.  Only a jump beyond SYNC_SNAP (a respawn, a tile
// recycled to the far edge, a rewind) is taken at once.
const SYNC_SNAP = 6.0;
const syncEase = (error, frame) => 1 - Math.exp(-frame / (0.06 + 0.12 * Math.min(1, error / 2)));
const shipStep = shipStepper(pred, () => view);
// A walker (World Walk) steps by the walking laws; everything else by the
// shared ship stepper (pilot_timeline.js).
function stepShip(ship, input, seconds, motion) {
  if (motion.walk) return stepWalker(ship, input, seconds, motion);
  return shipStep(ship, input, seconds, motion);
}
// The page predicts its ship always: from its own controls, or -- while the
// demo pilot flies -- from the pilot's INTERVAL TIMELINE (pilot_timeline.js):
// the pilot decides on its own slow clock, each decision takes effect
// motion.pilot.predelay of law time after it is taken and is streamed ahead of
// its start (you.intervals), and the page applies it over exactly its span of
// law time, on the same compiled laws.  Only its own flying is sent.
const predicting = (game) => !!(pred.program && game && game.input === "realtime" && game.predict);
const piloting = (game) => !!(game && game.autopilot && game.you && game.you.pilot);
const pilotLine = new PilotTimeline();           // the pilot's intervals, keyed to law time
const pilotRest = intervalControls([0, 0, 0, 0, 0]);
// The predicted pilot ship is always the server's state at the last snapshot (pilotBase, at law time pilotBaseT)
// carried to the law time now (pilotAt) through the timeline, in steps that end where the server's do: its tick
// windows (the spacing of the snapshots' law clocks, pilotWindow).  The laws are discretised, so the same steps
// give the same ship; a finer step of the page's own would be a (small, growing) disagreement.
let pilotBase = null, pilotBaseT = 0, pilotAt = null, pilotSeen = -Infinity, pilotWindow = 1 / 40;
window.meowdokuDebug.pilotLine = pilotLine;
function reconcile(game) {
  if (!predicting(game) || !game.you || !game.you.alive) { pred.ship = null; return; }
  pred.lane = game.predict.lane;
  const before = pred.ship ? [pred.ship.x, pred.ship.y] : null;
  if (game.ack != null) pred.sent = pred.sent.filter((interval) => interval.seq > game.ack);
  const ship = { ...game.you };
  if (isWalker(game)) {          // the click path's progress: the server's word (the page's own is kept below while it agrees)
    const w = game.walking;
    ship.route = w && w.id === walk.routeId && w.active ? 1 : 0; ship.wp = w ? w.wp : 0;
    if (pred.ship && w && w.id === walk.routeId && !w.active) pred.ship.route = 0;      // the server's path is done or cancelled
  }
  if (piloting(game)) {
    // the pilot's ship: the server's state at the snapshot's LAW time, carried
    // on to now by the intervals in effect over each span of law time
    pred.sent = []; pred.closed = []; pred.open = null;
    if (game.you.clock < pilotSeen - 1) { pilotLine.clear(); pilotSeen = -Infinity; }   // the law clock went back: a new game
    if (game.you.clock > pilotSeen && pilotSeen > -Infinity) pilotWindow = Math.min(STEP_CAP_S, Math.max(1e-3, game.you.clock - pilotSeen));
    pilotSeen = game.you.clock;
    pilotLine.accept(game.you.intervals, game.you.clock);
    // unwinding the latency: the snapshot's age since it arrived plus the
    // half round trip it took to come (the server stepped it that long ago)
    const age = Math.min(0.3, Math.max(0, performance.now() / 1000 - (game.you.at ?? 0) + 0.5 * (sock.rtt || 0)));
    pilotBase = { ...game.you }; pilotBaseT = game.you.clock;
    pilotAt = pilotBaseT + age;
    pilotLine.integrate(stepShip, ship, pilotBaseT, pilotAt, game.predict, pilotRest, pilotWindow);
  }
  for (const interval of [...pred.sent, ...pred.closed, ...(pred.open ? [pred.open] : [])]) stepShip(ship, interval.controls, interval.dur, game.predict);
  // AUTHORITY WITHIN TOLERANCE.  Page and server are two recordings of the
  // same signal, ordered by input seq, not one state stepped in lockstep.
  // While the page's own ship stays within the game's adoption tolerance
  // (``predict.adopt``, the same bound the server adopts the page's reports
  // under) the page keeps it: it holds control, the server follows.  Only a
  // disagreement past the tolerance hands control back -- the server's
  // replay becomes the ship and the difference is eased away, never jumped.
  // Axes the motion law does not declare are the server's alone.
  // Only axes the game will adopt are the page's to keep; a scrolled axis is
  // still stepped here by the motion law, but from the server's state.
  const adoptable = game.predict.adopt_axes || ["x", "y"];
  const declared = [["x", "vx"], ["y", "vy"], ["z", "vz"], ["yaw", "vyaw"], ["spd", "vspd"], ["bank", "vbank"]].filter(([pos]) => game.predict[pos] && adoptable.includes(pos));
  const land = game.predict.travel && pred.ship && pred.ship.map_u != null
    ? Math.hypot(pred.ship.map_u - ship.map_u, pred.ship.map_v - ship.map_v)
      + (game.predict.sphere && pred.ship.ux != null
         ? game.predict.sphere.radius * Math.hypot(pred.ship.ux - ship.ux, pred.ship.uy - ship.uy, pred.ship.uz - ship.uz) : 0)
    : 0;
  const error = pred.ship ? Math.hypot(...declared.map(([pos]) => pred.ship[pos] - ship[pos]), land) : Infinity;
  const tolerance = Number(game.predict.adopt ?? 0);
  // (the demo pilot's ship is the server's: rebased on it every snapshot,
  // carried forward by the snapshot's age -- the page never keeps its own)
  if (pred.ship && error <= tolerance && !piloting(game)) {
    for (const [pos, vel] of declared) { ship[pos] = pred.ship[pos]; ship[vel] = pred.ship[vel]; }
    for (const name of game.predict.travel || []) if (pred.ship[name] != null) ship[name] = pred.ship[name];
    // The server's own axes still move only on its word -- eased, not jumped.
    if (before) ["x", "y"].forEach((pos, i) => { if (!declared.some(([p]) => p === pos)) pred.offset[i] += before[i] - ship[pos]; });
    syncFaithfulness(0);
  } else if (before) {
    pred.offset = [pred.offset[0] + before[0] - ship.x, pred.offset[1] + before[1] - ship.y];
    if (Math.hypot(...pred.offset) > SYNC_SNAP) pred.offset = [0, 0];
    syncFaithfulness(Math.hypot(before[0] - ship.x, before[1] - ship.y));
  }
  pred.ship = ship;
}
// The predictor runs on its own clock, PREDICT_HZ, unbound from the display's
// refresh: controls are sampled, input intervals closed and the ship stepped
// at that rate whether the page draws at 30, 60 or 144 frames a second (or
// not at all); drawing just reads the latest predicted ship.
const PREDICT_HZ = 240;
setInterval(() => predictFrame(performance.now() / 1000), 1000 / PREDICT_HZ);
function predictFrame(now) {
  const game = state.game;
  const frame = pred.at == null ? 0 : Math.min(0.1, Math.max(0, now - pred.at));
  pred.at = now;
  if (!predicting(game)) return;
  if (piloting(game)) {                          // the demo pilot flies: predict with its controls, send nothing
    if (pred.ship && pilotBase) {
      pilotAt = Math.min(pilotAt + frame, pilotBaseT + 0.5);
      const ahead = { ...pilotBase };
      pilotLine.integrate(stepShip, ahead, pilotBaseT, pilotAt, game.predict, pilotRest, pilotWindow);
      Object.assign(pred.ship, ahead);
    }
    const decayP = 1 - syncEase(Math.hypot(...pred.offset), frame);
    pred.offset = [pred.offset[0] * decayP, pred.offset[1] * decayP];
    return;
  }
  // The open interval runs while the controls hold; a change closes it.
  const c = controls();
  if (!pred.open || !sameControls(pred.open.controls, c)) {
    if (pred.open && pred.open.dur > 0) pred.closed.push(pred.open);
    pred.open = { controls: c, dur: 0 };
  }
  pred.open.dur += frame;
  if (pred.ship) stepShip(pred.ship, c, frame, game.predict);
  const decay = 1 - syncEase(Math.hypot(...pred.offset), frame);
  pred.offset = [pred.offset[0] * decay, pred.offset[1] * decay];
  flyShots(frame, game);
  predictShots(now, game, c);
}
// A sphere game's provisional shot flies the server's law: sphere_advance by
// its speed (the shot's lane in scroller_step moves only by that and by
// climb), and is drawn where sphere_project puts it in the patch about the
// ship AS PREDICTED -- the frame the page draws everything else in.
function flyShots(seconds, game) {
  const R = game.predict.sphere && game.predict.sphere.radius;
  if (!R || !pred.advance || !pred.climb) return;
  for (const shot of pred.shots) {
    const s = shot.flown;
    if (!s) continue;
    const a = pred.advance.invoke({ ux: s.ux, uy: s.uy, uz: s.uz, hx: s.hx, hy: s.hy, hz: s.hz, distance: s.spd * seconds, radius: R });
    s.ux = a.ux_next; s.uy = a.uy_next; s.uz = a.uz_next; s.hx = a.hx_next; s.hy = a.hy_next; s.hz = a.hz_next;
    s.z = pred.climb.invoke({ z: s.z, vz: s.vz, dt: seconds }).z_next;
  }
}
// The primary weapon the server fires: the blaster's volley is bolts (the
// laser is a beam and the missiles their own bodies, not shots).
function blasterVolley(game) {
  const a = game.arsenal && game.arsenal.primary, slot = state.primarySlot || 0;
  const owned = a && a[slot] && a[slot].owned;
  return a && (owned ? slot : 0) === 0 ? a[0] : null;
}
// PREDICTED SHOTS.  The page's own shots leave the muzzle of the ship it
// draws the moment fire is held, under the weapon the game declares (muzzle
// offsets, velocity, reload, shots in flight), instead of appearing when the
// server's spawn arrives -- a round trip late and already downrange.  When the
// server reports a shot of ours, it takes over the oldest provisional one,
// starting from where that one is drawn; one the server never confirms fades
// within PROVISIONAL_S.  Provisional shots hit nothing: hits are the server's.
const PROVISIONAL_S = 0.45;
function ownShotsInFlight(weapon) {
  let count = pred.shots.length;
  for (const piece of state.pieces.values())
    if (piece.entity && !piece.provisional && piece.owner === pred.lane && piece.mesh === weapon.mesh && piece.removing == null) count++;
  return count;
}
function predictShots(now, game, c) {
  const drawNow = performance.now() / 1000;
  for (const shot of pred.shots.filter((s) => drawNow - s.at > PROVISIONAL_S)) {
    const piece = state.pieces.get(shot.id);
    if (piece && piece.removing == null) piece.removing = state.time;
  }
  pred.shots = pred.shots.filter((s) => drawNow - s.at <= PROVISIONAL_S);
  const weapon = game.predict && game.predict.weapon;
  if (weapon && weapon.slope != null) return predictVolleys(now, game, c, weapon, drawNow);
  if (!weapon || !pred.ship || !c.fire) return;
  if (now - pred.lastShot < Math.max(weapon.reload, 0.02)) return;
  if (ownShotsInFlight(weapon) + weapon.muzzles.length > weapon.max_in_flight) return;
  const x0 = pred.ship.x + pred.offset[0], z0 = pred.ship.y + pred.offset[1];
  for (const [mx, mz] of weapon.muzzles) {
    const id = `prov:${++pred.shotSeq}`;
    state.pieces.set(id, { id, entity: true, provisional: true, lane: -1, owner: pred.lane,
      sx: x0 + mx, sz: z0 + mz, x: x0 + mx, z: z0 + mz, vx: weapon.velocity[0], vz: weapon.velocity[1], tRecv: drawNow,
      mesh: weapon.mesh, color: weapon.color, size: weapon.size, roll: 0, spin: 0, bob: 0, lean: 0.55,
      t0: state.time, removing: null, seed: 0, birth: null });
    pred.shots.push({ id, at: drawNow });
  }
  pred.lastShot = now;
}
// A SPHERICAL GAME'S FIRE.  WHEN a volley may leave is the engine's fire_gate
// -- the very law scroller_step calls: the fire mode, held fire and the last
// step's, the reload clock, a burst's count, the volley against the free shot
// lanes -- run on the server's own step grid (``clock.dt``, the dt system's
// step the server reports), the law's state (the reload clock and the burst
// count) the page's own between the server's words.  WHAT leaves is the
// engine's shot_spawn, flown by sphere_advance and climb.  A laser is a beam
// and a missile its own body: the server's, so the gate only keeps their clocks.
function predictVolleys(now, game, c, weapon, drawNow) {
  const ship = pred.ship, R = game.predict.sphere && game.predict.sphere.radius;
  if (!pred.gate || !pred.shot || !R || !ship) return;
  const fire = pred.fire || (pred.fire = { at: null, owed: 0, timer: 0, burst: 0, prev: 0 });
  fire.owed += fire.at == null ? 0 : Math.min(0.1, Math.max(0, now - fire.at));
  fire.at = now;
  const step = Math.min(STEP_CAP_S, Math.max(1 / 240, (game.clock && game.clock.dt) || 1 / 60));
  const primary = game.arsenal && game.arsenal.primary, slot = state.primarySlot || 0;
  const gun = primary && primary[slot] && primary[slot].owned ? slot : 0;      // the primary the server fires: the chosen slot if owned, else the blaster
  const volley = primary && primary[gun], mult = volley ? volley.mult : 0, fan = volley ? volley.fan : 0;
  const held = c.fire ? 1 : 0;
  while (fire.owed >= step - 1e-12) {
    fire.owed -= step;
    const gate = pred.gate.invoke({ fire_mode: state.fireMode ?? 1, held, prev: fire.prev, timer: fire.timer, burst_left: fire.burst,
      ready: game.you && game.you.alive ? 1 : 0, laser: gun === 1 ? 1 : 0, missile: gun === 2 ? 1 : 0, shots_n: 1 + mult,
      free: weapon.max_in_flight - ownShotsInFlight(weapon), dt: step, reload: weapon.reload, missile_reload: weapon.missile_reload,
      burst_count: weapon.burst_count, cont_factor: weapon.cont_factor, burst_factor: weapon.burst_factor });
    fire.timer = gate.timer_next; fire.burst = gate.burst_next; fire.prev = held;
    if (!(gate.volley > 0.5) || gun !== 0 || ship.ux == null) continue;
    const pitch = Math.max(-1, Math.min(1, c.move_axis_z || 0));
    for (let rank = 0; rank < gate.spawn_n; rank++) {
      const r = pred.shot.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: ship.hx, hy: ship.hy, hz: ship.hz,
        c: rank - 0.5 * mult, fan, laser: 0, ship_spd: ship.spd || 0, pitch,
        ahead: weapon.muzzles.ahead, spacing: weapon.muzzles.spacing, fan_step: weapon.fan_step, speed: weapon.speed,
        laser_mult: weapon.laser_mult, slope: weapon.slope, radius: R });
      const id = `prov:${++pred.shotSeq}`;
      const flown = { ux: r.ux_next, uy: r.uy_next, uz: r.uz_next, hx: r.hx_next, hy: r.hy_next, hz: r.hz_next,
                      spd: r.speed_next, vz: r.vz_next, z: ship.z || 0 };
      state.pieces.set(id, { id, entity: true, provisional: true, lane: -1, owner: pred.lane, flown,
        sx: 0, sz: 0, x: 0, z: 0, vx: 0, vz: 0, tRecv: drawNow,
        mesh: weapon.mesh, color: weapon.color, size: weapon.size, roll: 0, spin: 0, bob: 0, lean: 0.55,
        t0: state.time, removing: null, seed: 0, birth: null });
      pred.shots.push({ id, at: drawNow, flown });
    }
  }
}
const STEP_CAP_S = 1 / 30;            // the longest law step the server takes (scroller.STEP_CAP)
// Ticks leave on a steady clock, up to three in flight, outside the action
// queue -- so the update rate is the clock's, not one round trip's.  Each
// resends every interval not yet acknowledged; the server skips any it has
// landed, so a reordered or lost request costs nothing.  A reply older than
// the newest one applied is dropped.
// The world time a state stands for, on the page's clock: with input-timed
// rounds, the moment the last landed interval closed.
function worldStamp(game, now) {
  let stamp = now;
  if (predicting(game) && game.ack != null) {
    const landed = pred.sent.find((interval) => interval.seq === game.ack);
    stamp = landed ? landed.closedAt : (state.lastStamp ?? now);
  }
  return (state.lastStamp = Math.min(stamp, now));
}

// ---------------------------------------------------------------- real-time socket
// One WebSocket per page (realtime_socket.py): each tick's input goes up as a
// small JSON frame; each answer is a binary snapshot delta (spawns with their
// look, entities whose motion changed, lanes gone, the ship and the ack),
// with the status as a JSON "meta" frame a few times a second.
const sock = { ws: null, open: false, outstanding: 0, names: [], gen: 0, lanes: new Map(), bytes: 0, frames: 0 };
function ensureSocket() {
  if (sock.ws || typeof WebSocket === "undefined") return;
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  sock.ws = ws;
  ws.onopen = () => { sock.open = true; sock.outstanding = 0; sock.names = []; sock.lanes.clear(); sock.meta = null; };
  ws.onclose = () => { sock.ws = null; sock.open = false; sock.outstanding = 0; };
  ws.onerror = () => {};
  ws.onmessage = (event) => {
    // A message for a game the page has left (ticks still in flight when the
    // player switched games) is dropped: it must never put that game back on
    // screen, nor restart anything.
    const current = state.game, live = !!(current && current.input === "realtime");
    if (typeof event.data === "string") {
      const message = JSON.parse(event.data);
      if (message.meta) {
        if (!live || (message.meta.game && message.meta.game !== current.game)) return;
        sock.meta = { ...(sock.meta || {}), ...message.meta }; applyState({ ...sock.meta });
      }
      if (message.error) {
        sock.outstanding = Math.max(0, sock.outstanding - 1);
        if (sock.sentAt) sock.sentAt.shift();                 // an answered tick, all the same
        if (live && /no game/.test(message.error) && restartAllowed()) {
          ticks.restarting = true;
          newGame({ game: state.game.game, aspect: canvas.clientWidth / Math.max(1, canvas.clientHeight) }).finally(() => { ticks.restarting = false; });
        }
      }
      return;
    }
    sock.outstanding = Math.max(0, sock.outstanding - 1);
    // the round trip of the tick this snapshot answers (smoothed)
    const sentAt = sock.sentAt && sock.sentAt.shift();
    if (sentAt != null) { const rtt = performance.now() / 1000 - sentAt; sock.rtt = sock.rtt == null ? rtt : sock.rtt * 0.85 + rtt * 0.15; }
    if (!live) return;
    sock.bytes += event.data.byteLength; sock.frames += 1;
    applySnapshot(new DataView(event.data));
  };
}
// A projectile is born at its owner's muzzle.  By the time the page hears of
// it the server's projectile has flown on, from where the SERVER had the
// owner -- not where the page draws it (the player's ship is predicted ahead).
// So it is drawn leaving the owner as drawn, at the muzzle offset the server
// used, and eased onto its true path over BIRTH_S.
const BIRTH_S = 0.12;
function birthOffset(ownerLane, age, x, y, vx, vy, you, stamp) {
  if (ownerLane === 65535 || age > 0.35) return null;
  let server, drawn;
  if (ownerLane === pred.lane && pred.ship) {
    server = [you.x, you.y];
    drawn = [pred.ship.x + pred.offset[0], pred.ship.y + pred.offset[1]];
  } else {
    const owner = state.pieces.get(sock.lanes.get(ownerLane));
    if (!owner) return null;
    server = [owner.sx + owner.vx * (stamp - owner.tRecv), owner.sz + owner.vz * (stamp - owner.tRecv)];
    drawn = [owner.x, owner.z];
  }
  const muzzle = [x - vx * age - server[0], y - vy * age - server[1]];
  return { dx: drawn[0] + muzzle[0] - x, dz: drawn[1] + muzzle[1] - y };
}
function applySnapshot(view) {
  const game = state.game; if (!game || game.input !== "realtime") return;
  const now = state.time;
  // The frame layout is bodies.js's (realtime_socket.PROTOCOL): a page and a
  // server of different layouts must not misparse each other -- say so instead.
  const frame = decodeSnapshot(view, sock.names);
  if (frame.version !== SNAPSHOT_PROTOCOL) {
    if (!sock.versionWarned) { sock.versionWarned = true; status.textContent = `the server speaks snapshot v${frame.version}, this page v${SNAPSHOT_PROTOCOL}: reload the page (the server was updated)`; }
    return;
  }
  const { flags, ack, you } = frame;
  sync.hint = frame.hint;
  you.at = performance.now() / 1000;
  const frameGame = { ...game, ack: ack < 0 ? null : ack, you };
  state.game.you = you;
  noteShield(frameGame, now);
  const stamp = worldStamp(frameGame, now);
  const drop = (lane) => { const key = sock.lanes.get(lane); const piece = key && state.pieces.get(key); if (piece && piece.removing == null) piece.removing = now; sock.lanes.delete(lane); };
  if (flags & FLAG_CLEAR) { for (const lane of [...sock.lanes.keys()]) drop(lane);
                   for (const piece of state.pieces.values()) if (piece.entity && piece.removing == null) piece.removing = now; }
  for (const lane of frame.gone) drop(lane);
  for (const r of frame.spawned) {
    drop(r.lane);
    const id = `${r.lane}:s${++sock.gen}`;
    sock.lanes.set(r.lane, id);
    // (a body of a spherical game is born where the laws carry it, below; a flat
    // game's projectile is eased onto its path from its owner as drawn)
    const birth = r.body ? null : birthOffset(r.owner, r.age, r.x, r.y, r.vx, r.vy, you, stamp);
    const piece = { id, entity: true, lane: r.lane, owner: r.owner === 65535 ? null : r.owner, sx: r.x, sz: r.y, x: r.x, z: r.y, vx: r.vx, vz: r.vy,
                    sh: r.z, h: r.z, vh: r.vz,
                    tRecv: stamp, mesh: r.mesh, color: r.color, size: r.size, roll: r.roll, rollBase: r.roll, orient: r.orient, spin: r.spin, bob: r.bob, lean: r.lean,
                    face: r.face, translucent: r.translucent, ribbon: r.ribbon, ground: r.ground, glow: r.glow,
                    t0: now, removing: null, seed: 0, birth };
    if (r.body) piece.body = BodyFlight.held(r, stamp);
    if (r.owner === pred.lane && pred.shots.length) {
      const adopted = pred.shots.shift(), shown = state.pieces.get(adopted.id);
      if (shown) {
        // our own shot, confirmed: it leaves from where the provisional one is drawn and glides onto the server's path
        const at = piece.body ? carriedPatch(piece, now) : null;
        if (at) piece.birth = { dx: shown.x - at.x, dz: shown.z - at.z };
        else { const lead = Math.max(0, now - stamp); piece.birth = { dx: shown.x - (r.x + r.vx * lead), dz: shown.z - (r.y + r.vy * lead) }; }
        state.pieces.delete(adopted.id);
      }
    }
    state.pieces.set(id, piece);
  }
  for (const r of frame.updated) {
    const piece = state.pieces.get(sock.lanes.get(r.lane));
    if (!piece) continue;
    Object.assign(piece, { sx: r.x, sz: r.y, vx: r.vx, vz: r.vy, sh: r.z, vh: r.vz, tRecv: stamp, mesh: r.mesh, color: r.color });
    if (r.body) { piece.body = BodyFlight.held(r, stamp); piece.rollBase = piece.roll = r.roll; }
  }
  reconcile(frameGame);
}

const ticks = { inFlight: 0, sent: 0, applied: -1 };
function sendTick(game) {
  let payload = { type: "tick", ...controls() };
  if (predicting(game) && !piloting(game)) {
    if (pred.open && pred.open.dur > 0) { pred.closed.push(pred.open); pred.open = { controls: pred.open.controls, dur: 0 }; }
    const at = performance.now() / 1000;
    pred.sent.push(...pred.closed.map((interval) => ({ ...interval, seq: ++pred.seq, closedAt: at })));
    pred.closed = [];
    if (pred.sent.length > 240) pred.sent.splice(0, pred.sent.length - 240);
    payload.inputs = pred.sent.map((interval) => ({ seq: interval.seq, dur: interval.dur, ...interval.controls }));
    // CLIENT UPDATES SERVER: the open interval was just cut at zero length, so
    // the predicted ship stands exactly at the end of interval pred.seq --
    // where the engine's motion law (motion.wasm) put it.  The server adopts
    // it for our lane once it has landed that interval, within its bound.
    if (pred.ship && pred.sent.length)
      payload.ship = { seq: pred.seq, x: pred.ship.x, y: pred.ship.y, vx: pred.ship.vx, vy: pred.ship.vy,
                       z: pred.ship.z || 0, vz: pred.ship.vz || 0,
                       yaw: pred.ship.yaw || 0, vyaw: pred.ship.vyaw || 0, spd: pred.ship.spd || 0, vspd: pred.ship.vspd || 0,
                       map_u: pred.ship.map_u || 0, map_v: pred.ship.map_v || 0, map_gx: pred.ship.map_gx || 0,
                       map_gy: pred.ship.map_gy ?? -1, bank: pred.ship.bank ?? 0, vbank: pred.ship.vbank ?? 0,
                       ux: pred.ship.ux ?? 0, uy: pred.ship.uy ?? 0, uz: pred.ship.uz ?? 1,
                       hx: pred.ship.hx ?? 0, hy: pred.ship.hy ?? 1, hz: pred.ship.hz ?? 0,
                       wp: pred.ship.wp ?? 0, route: pred.ship.route ?? 0 };
  }
  if (sock.open) {
    sock.outstanding++;
    (sock.sentAt ||= []).push(performance.now() / 1000);      // for the round trip (latency unwinding)
    sock.ws.send(JSON.stringify(payload));
    return Promise.resolve();
  }
  ensureSocket();
  const id = ++ticks.sent;
  ticks.inFlight++;
  return fetch("/api/action", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
    .then((response) => response.json())
    .then((data) => {
      // A restarted server has forgotten this page's session: start the same
      // game again rather than ticking into refusals.
      if (data.error && /no game/.test(data.error) && restartAllowed()) {
        ticks.restarting = true;
        newGame({ game: state.game.game, aspect: canvas.clientWidth / Math.max(1, canvas.clientHeight) })
          .finally(() => { ticks.restarting = false; });
        return;
      }
      if (data.error || id < ticks.applied || !state.game || data.game !== state.game.game) return;
      ticks.applied = id;
      applyState(data);
    })
    .catch((error) => console.warn(error))
    .finally(() => { ticks.inFlight--; });
}
// SYNC RELAX for faithful clients.  A page whose predictions keep landing
// where the server puts the ship (each reply's correction under
// SYNC.faithful) earns a longer delay between check-ins, 20% more per
// faithful reply; one visible correction (over SYNC.drift) snaps it back to
// the fastest.  The relaxed delay never exceeds the server's sync hint -- the
// longest wait its danger allows (time to contact with the nearest hostile)
// -- and any change of input is a check-in at once, so a press is never
// held back by a relaxed clock.  Unpredicted pages keep the fastest clock.
const SYNC = { fastest: 0.03, longest: 0.25, faithful: 0.01, drift: 0.04, grow: 1.2 };
const sync = { interval: SYNC.fastest, hint: SYNC.longest, next: 0, streak: 0 };
window.meowdokuDebug.sync = sync;
function syncFaithfulness(correction) {
  if (correction > SYNC.drift) { sync.interval = SYNC.fastest; sync.streak = 0; return; }
  if (correction < SYNC.faithful) { sync.streak += 1; sync.interval = Math.min(sync.interval * SYNC.grow, SYNC.longest); }
}
setInterval(() => {
  const game = state.game;
  if (!game || game.input !== "realtime" || game.over || document.hidden || ticks.inFlight + sock.outstanding >= 3) return;
  const now = performance.now() / 1000;
  // Holding fire is activity: the server must hear of it at once to spawn the
  // real shots the page is already showing.
  const firing = predicting(game) && controls().fire > 0;
  const allowed = predicting(game) && !firing ? Math.min(sync.interval, Math.max(SYNC.fastest, sync.hint)) : SYNC.fastest;
  const inputChanged = pred.closed.length > 0;               // a press, release or steer since the last check-in
  if (now < sync.next && !inputChanged) return;
  sync.next = now + allowed;
  sendTick(game);
}, 5);

function entityModel(piece, now) {
  // Free-board entities: drawn where the server last put them, carried on by
  // the velocity it reported (at most a frame's worth, so a late reply never
  // flings anything), popping in when born and shrinking when gone.
  // The target is the server's position carried on by its velocity for at
  // most about one tick; the drawn position eases toward it (a ~45 ms time
  // constant) instead of jumping, so reply-to-reply latency differences and
  // a released key never show as a snap.
  // Carried forward from the moment the server's state stands for, for as
  // long as it takes the next reply to come (constant-velocity motion is
  // exact; anything else is corrected as it arrives) -- never frozen.
  // (With socket deltas an unchanged entity is carried from its last record,
  // at most one keyframe -- 0.5 s of world time -- old.)
  // (a spherical game's body is never extrapolated linearly: with its cores not
  // loaded it stands where the server last put it)
  const mesh = state.meshes[piece.mesh], lead = piece.body ? 0 : Math.min(Math.max(0, now - piece.tRecv), 1.0);
  const own = state.game && state.game.predict && piece.lane === state.game.predict.lane && piece.mesh !== "boom" && piece.removing == null;
  if (own && drawnShip.at !== now) {             // once per frame (the shadow and lit passes both draw it)
    drawnShip.at = now;
    drawnShip[pred.ship && piece.lane === pred.lane ? "page" : "server"] += 1;
  }
  if (pred.ship && piece.lane === pred.lane && piece.mesh !== "boom" && piece.removing == null) {
    // The player's own ship: the prediction, not the server's last word.
    piece.x = pred.ship.x + pred.offset[0]; piece.z = pred.ship.y + pred.offset[1]; piece.drawnAt = now;
    if (state.game.roomview) piece.z = roomShip(state.game).y;
    piece.h = pred.ship.z || 0;
    return placeEntity(piece, now, piece.x, piece.z, mesh);
  }
  if (piece.flown && pred.ship && pred.project && pred.ship.ux != null) {
    // The player's own provisional shot (engine law, flyShots): its patch
    // position about the ship as predicted -- sphere_project, no linear guess.
    const patch = state.game.predict.patch, f = piece.flown, ship = pred.ship;
    const p = pred.project.invoke({ ux: f.ux, uy: f.uy, uz: f.uz, cx: ship.ux, cy: ship.uy, cz: ship.uz,
                                    hx: ship.hx, hy: ship.hy, hz: ship.hz, radius: state.game.predict.sphere.radius });
    piece.x = patch.cx + p.X + pred.offset[0]; piece.z = patch.cy - p.Y + pred.offset[1]; piece.h = f.z; piece.drawnAt = now;
    return placeEntity(piece, now, piece.x, piece.z, mesh);
  }
  if (piece.body && !own) {
    // A body of a spherical game: carried on from the state the server sent
    // by the engine's laws and projected into the patch about the ship (see
    // carriedPatch) -- never extrapolated with a velocity of the page's own.
    const at = carriedPatch(piece, now);
    if (at) {
      piece.x = at.x + pred.offset[0]; piece.z = at.y + pred.offset[1]; piece.h = at.z; piece.drawnAt = now;
      if (piece.orient) piece.roll = piece.rollBase - at.heading;
      return placeEntity(piece, now, piece.x, piece.z, mesh);
    }
  }
  const tx = piece.sx + piece.vx * lead, tz = piece.sz + piece.vz * lead;
  // Each frame the drawn position first moves by the velocity (so a shot
  // never trails), then only the remaining error is eased away.
  const first = piece.drawnAt == null;
  const frame = first ? 0 : Math.max(0, Math.min(0.1, now - piece.drawnAt));
  piece.drawnAt = now;
  const px = piece.x + piece.vx * frame, pz = piece.z + piece.vz * frame;
  const error = Math.hypot(tx - px, tz - pz);
  const step = first ? 1 : syncEase(error, frame);
  const far = error > SYNC_SNAP;                                     // a respawn or rewind: go there
  const x = far || step >= 1 ? tx : px + (tx - px) * step;
  const z = far || step >= 1 ? tz : pz + (tz - pz) * step;
  piece.x = x; piece.z = z;
  // Height above the board (a game with a third axis), extrapolated and eased the same way.
  const th = (piece.sh || 0) + (piece.vh || 0) * lead, ph = (piece.h ?? th) + (piece.vh || 0) * frame;
  piece.h = far || step >= 1 ? th : ph + (th - ph) * step;
  // A body declared to face its flight (a missile): yaw to its heading on the
  // board, pitch to its climb or fall -- the drawn velocity, every frame.
  if (piece.face) {
    const ground = Math.hypot(piece.vx || 0, piece.vz || 0);
    if (ground > 1e-3 || Math.abs(piece.vh || 0) > 1e-3) {
      piece.roll = 0.785 - Math.atan2(piece.vx || 0, -(piece.vz || 0));
      piece.lean = Math.atan2(piece.vh || 0, Math.max(ground, 1e-3));
    }
  }
  return placeEntity(piece, now, x, z, mesh);
}
// AUTHORITY STATS: which ship the page drew each frame (its own prediction
// or the server's), beside the server's count of who authored the ship on
// each landed step (game.authority).  The rest of the world is always the
// server's, extrapolated between snapshots.
const drawnShip = { page: 0, server: 0, at: null };
// A game's selectable MODES (game.modes: labels and which are on): one
// toggle each, sent as {"type": "mode", name, on}; the law reads it.
// WORDLE'S DICTIONARY: which word list is active (the answer is any of its
// words) and, for the complete list, how many letters; a change starts a
// new word.  Remembered by this page between games.
const wordleChoice = (() => { try { return JSON.parse(localStorage.getItem("wordleChoice")) || {}; } catch (_) { return {}; } })();
function renderWordleOptions(game) {
  const box = $("wordle-opts"), d = game && game.dictionary;
  if (!box) return;
  if (!d) { if (box.childElementCount) box.replaceChildren(); return; }
  const key = `${d.name}:${d.length}`;
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  const restart = () => {
    try { localStorage.setItem("wordleChoice", JSON.stringify(wordleChoice)); } catch (_) { /* private window */ }
    saveSettings();
    newGame({ game: "wordle", seed: Math.floor(Math.random() * 100000), ...wordleChoice });
  };
  const pick = document.createElement("select");
  for (const [name, label] of Object.entries(d.choices)) pick.append(new Option(label, name, false, name === d.name));
  pick.onchange = () => { wordleChoice.dictionary = pick.value; restart(); };
  const len = document.createElement("select");
  for (const n of d.lengths[d.name] || [5]) len.append(new Option(`${n} letters`, String(n), false, n === d.length));
  len.disabled = (d.lengths[d.name] || [5]).length < 2;
  len.onchange = () => { wordleChoice.length = Number(len.value); restart(); };
  const size = document.createElement("span");
  size.textContent = ` ${d.size.toLocaleString()} words`;
  box.replaceChildren("Dictionary ", pick, " ", len, size);
}
// THE PLAYER'S INVENTORY (the server's profile, by this browser's player
// cookie): the name it knows us by, points, the success streak, banked and
// ongoing power-ups, the shop; page settings saved to the profile; completed
// boards ticked in the picker.
const inventoryView = { applied: false, awardKey: null, awardAt: 0, completedKey: "" };
function renderInventory(game) {
  const inv = game && game.inventory;
  if (game && game.player && document.activeElement !== $("player") && $("player").value !== game.player) {
    $("player").value = game.player;
    try { localStorage.setItem("meowdoku_player", game.player); } catch { /* private window */ }
  }
  if (!inv) return;
  $("shop-toggle").textContent = `Shop · ${inv.points} pts`;
  if (!inventoryView.applied) { inventoryView.applied = true; applySettings(inv.settings || {}); }
  const done = (inv.completed && inv.completed.meowdoku) || [], key = done.join(",");
  if (key !== inventoryView.completedKey) {
    inventoryView.completedKey = key;
    for (const option of $("board").options) {
      const plain = option.textContent.replace(/^✓ /, "");
      option.textContent = done.includes(option.value) ? `✓ ${plain}` : plain;
    }
  }
  if ($("shop").classList.contains("open")) showShop(inv);
}
function awardText(game) {
  const a = game && game.award;
  if (!a) return "";
  const key = JSON.stringify(a);
  if (key !== inventoryView.awardKey) { inventoryView.awardKey = key; inventoryView.awardAt = performance.now(); }
  if (performance.now() - inventoryView.awardAt > 8000) return "";
  return ` · +${a.points} points${a.multiplier > 1 ? ` (x${a.multiplier} streak)` : ""}`;
}
function showShop(inv) {
  inv = inv || (state.game && state.game.inventory);
  const panel = $("shop");
  if (!inv) { panel.innerHTML = `<div class="muted">no player yet</div>`; return; }
  const streaks = Object.entries(inv.streak || {}).filter(([, n]) => n > 0).map(([g, n]) => `${escapeHtml(g)} ${n} in a row (x${Math.min(n, inv.multiplier_max)})`).join(", ");
  const bank = Object.entries(inv.bank || {}).map(([k, n]) => `${escapeHtml(k)} ${n}`).join(" · ");
  const ongoing = inv.ongoing ? `${escapeHtml(inv.ongoing.item)}: ${inv.ongoing.puzzles} puzzles left` : "none";
  panel.innerHTML = `<h3>${escapeHtml((state.game && state.game.player) || "You")} · ${inv.points} points</h3>`
    + `<div class="muted">a solved puzzle pays lives x your streak; Wordle unused guesses x streak; arcade games score / 500</div>`
    + `<div class="row"><span>streak</span><span>${streaks || "none yet"}</span></div>`
    + `<div class="row"><span>banked power-ups</span><span>${bank}</span></div>`
    + `<div class="row"><span>ongoing</span><span>${ongoing}</span></div>`
    + `<h3>Buy</h3>` + inv.shop.map((it) => `<div class="row"><span>${escapeHtml(it.label)}</span>`
      + `<button data-buy="${escapeHtml(it.item)}" ${inv.points < it.cost ? "disabled" : ""}>${it.cost} pts</button></div>`).join("");
  for (const button of panel.querySelectorAll("button[data-buy]"))
    button.onclick = () => act({ type: "buy", item: button.dataset.buy }).catch((e) => console.warn(e));
}
// Settings ride on the profile: saved when changed, applied once on arrival.
function currentSettings() {
  return { camMode: state.camMode || "board", pads: $("pads") ? $("pads").checked : false,
           bombStyle: state.bombStyle || 0, wordle: { ...wordleChoice } };
}
function saveSettings() { if (state.game) act({ type: "settings", values: currentSettings() }).catch(() => {}); }
function applySettings(values) {
  if (values.camMode && values.camMode !== (state.camMode || "board")) toggleCamera();
  if (typeof values.pads === "boolean" && $("pads")) { $("pads").checked = values.pads; showPads(state.game); }
  if (Number.isInteger(values.bombStyle)) { state.bombStyle = (values.bombStyle + BOMB_STYLES.length - 1) % BOMB_STYLES.length; cycleBomb(); }
  if (values.wordle && typeof values.wordle === "object") Object.assign(wordleChoice, values.wordle);
}
function renderModes(game) {
  const box = $("modes"), modes = game && game.modes;
  if (!box) return;
  if (!modes) { if (box.childElementCount) box.replaceChildren(); return; }
  const key = `${game.game}:${Object.keys(modes.labels).join(",")}`;
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.replaceChildren(...Object.entries(modes.labels).map(([name, label]) => {
      const tag = document.createElement("label"), input = document.createElement("input");
      input.type = "checkbox"; input.dataset.mode = name;
      input.onchange = () => act({ type: "mode", name, on: input.checked });
      tag.append(label + " ", input);
      return tag;
    }));
  }
  for (const input of box.querySelectorAll("input[data-mode]")) {
    if (document.activeElement !== input) input.checked = !!modes.on[input.dataset.mode];
  }
}
function authorityText(game) {
  const a = game && game.authority;
  if (!a) return "";
  const pct = (n, d) => d ? `${Math.round(100 * n / d)}%` : "-";
  const landed = (a.client || 0) + (a.kept || 0) + (a.unreported || 0) + (a.server || 0);
  const drawn = drawnShip.page + drawnShip.server;
  const axes = (a.axes || []).join("") || "none";
  const scroll = (a.scroll_axes || []).length ? ` (${a.scroll_axes.join("")} always server: the world scrolls it)` : "";
  return `ship steps: client ${pct(a.client || 0, landed)} · server kept ${pct(a.kept || 0, landed)} · `
       + `server alone ${pct((a.unreported || 0) + (a.server || 0), landed)} of ${landed}, client axes ${axes}${scroll}`
       + ` · frames drawn from page ${pct(drawnShip.page, drawn)} · world 100% server`;
}
// THE VIEW (a world-frame game: predict.camera).  Lanes arrive at their world
// x; the screen's left edge is the view.  The page runs the law's own view
// rule on the ship it draws -- view = max(view, x - follow * width) -- and
// never falls behind the server's view; a view far behind the page's (a new
// run) is taken at once.
const view = { x: 0, at: null };
// A ROOM-GRAPH GAME (``game.roomview``: the room's size in tiles).  The screen is a window ``game.width`` x
// ``game.height`` tiles onto the room, centred on the player and clamped to the room (a room smaller than the
// window is centred in it): the formula of delve_rules.camera_axis, on the ship the page draws.
const roomCam = { at: null, x: 0, y: 0 };
function roomShip(game) {
  // the ship as drawn: the prediction sideways, the server's word vertically carried on by its velocity (a frame's worth)
  const ship = pred.ship || game.you;
  if (!ship) return null;
  const age = pred.ship && ship.at != null ? Math.min(0.1, Math.max(0, performance.now() / 1000 - ship.at)) : 0;
  return { x: ship.x + (pred.ship ? pred.offset[0] : 0), y: ship.y + (pred.ship ? pred.offset[1] : 0) + (ship.vy || 0) * age };
}
function roomView(now) {
  const game = state.game, rv = game && game.roomview;
  if (!rv) return null;
  if (roomCam.at !== now) {
    roomCam.at = now;
    const at = roomShip(game) || { x: rv.w / 2, y: rv.h / 2 };
    const axis = (pos, win, room) => room <= win ? (room - win) / 2 : Math.min(Math.max(pos - win / 2, 0), room - win);
    roomCam.x = axis(at.x, game.width, rv.w); roomCam.y = axis(at.y, game.height, rv.h);
  }
  return roomCam;
}
const viewY = (now) => { const rv = roomView(now); return rv ? rv.y : 0; };
function viewX(now) {
  const rcam = roomView(now);
  if (rcam) return rcam.x;
  // The engine's follow_view (the law's own), from the server's view with the
  // ship the page draws -- the same rule, the same compiled code.
  const game = state.game, cam = game && game.predict && game.predict.camera;
  if (!cam) return 0;
  if (view.at !== now) {
    view.at = now;
    const shipX = pred.ship ? pred.ship.x + pred.offset[0] : (game.you ? game.you.x : 0);
    view.x = pred.follow ? pred.follow.invoke({ view: cam.x, x: shipX, follow: cam.follow, width: cam.width }).view_next : cam.x;
  }
  return view.x;
}
const DEPTH_LIFT = 0.5, DEPTH_SCALE = [0.4, 1.8];       // exaggerated: small near the floor, large near the ceiling
// A SERVER BODY IN THE PATCH (a spherical game: predict.patch).  The body is
// carried on from the state the server sent -- by the engine's flight laws, as
// the server's step flies it (bodies.js: sphere_turn, sphere_advance, climb or
// the trajectory step) -- up to ``now`` (at most a second past the record), and
// projected into the patch about the ship the page draws, where sphere_project
// puts it: the turn the player's yaw makes under the world is in the
// projection, not a rotation of the page's own.  Once per frame.
function carriedPatch(piece, now) {
  const game = state.game, cfg = game && game.predict;
  const ship = pred.ship && pred.ship.ux != null ? pred.ship : game && game.you;
  if (!pred.bodies || !cfg || !cfg.bodies || !cfg.patch || !cfg.sphere || !ship || ship.ux == null) return null;
  if (piece.carried && piece.carried.at === now) return piece.carried;
  const b = piece.body, until = Math.min(now, piece.tRecv + 1.0);
  pred.bodies.carry(b, until - b.at, cfg.bodies, cfg.sphere);
  b.at = Math.max(b.at, until);
  const at = pred.bodies.patch(b, ship, cfg.patch, cfg.sphere.radius);
  at.heading = piece.orient ? pred.bodies.heading(b, ship) : 0;
  at.at = now;
  return (piece.carried = at);
}
function placeEntity(piece, now, x, z, mesh) {
  x -= viewX(now);                                // world frame -> the screen (0 unless the game scrolls a view)
  z -= viewY(now);                                // (a room-graph game scrolls vertically too)
  if (piece.birth) {
    const t = (now - piece.t0) / BIRTH_S;
    if (t >= 1) piece.birth = null;
    else { const k = 1 - ease.out(t); x += piece.birth.dx * k; z += piece.birth.dz * k; }
  }
  let s = piece.birth ? 1 : Math.min(1, (now - piece.t0) / 0.12), y = 0;
  if (piece.removing != null) s = 1 - ease.in(Math.min(1, (now - piece.removing) / 0.3));
  if (piece.mesh === "boom") s *= 1 + 0.35 * Math.sin(Math.min(1, (now - piece.t0) / 0.35) * Math.PI);
  if (piece.bob) y = 0.06 * Math.sin(now * 5 + piece.sx * 0.7);
  if (state.flat) y = 0;
  // THE HEIGHT LAYER: a game with a third axis (``predict.depth``) draws
  // height compressed (DEPTH_LIFT per unit) and exaggerates scale with it,
  // DEPTH_SCALE[0] at the floor to DEPTH_SCALE[1] at the ceiling, so the
  // perspective reads strongly while the layer stays close to the board.
  const depth = state.game && state.game.predict && state.game.predict.depth;
  const hRel = depth ? Math.max(0, Math.min(1, (piece.h || 0) / depth)) : null;
  const grow = hRel == null || state.flat ? 1 : DEPTH_SCALE[0] + (DEPTH_SCALE[1] - DEPTH_SCALE[0]) * hRel;
  const lift = hRel == null || state.flat ? 0 : (piece.h || 0) * DEPTH_LIFT;
  const lean = state.flat ? 0 : piece.lean ?? 0.55, scale0 = piece.size * grow, scale = scale0 / mesh.extent * Math.max(s, 0.001);
  const land = piece.ground && !state.flat ? landHeight(x, z) : null;
  let m = M.translate(x, state.flat ? TILE.height + FLAT_LIFT
                      : land != null ? land + scale0 * 0.35 - planetDrop(x, z)
                      : TILE.height + scale0 * Math.sin(lean) + 0.04 + y + lift - planetDrop(x, z), z);
  m = M.multiply(m, M.rotateY(piece.spin * now));
  m = M.multiply(m, M.rotateX(-Math.PI / 2 + lean));
  m = M.multiply(m, M.rotateZ(piece.roll));
  return M.multiply(m, M.scale(scale, scale, state.flat ? scale * FLAT_THICK : scale));
}
function pieceModel(piece, now) {
  if (piece.entity) return entityModel(piece, now);
  const age = now - piece.t0, fam = family(piece.mesh), mesh = state.meshes[piece.mesh];
  let y = 0, spin = 0, s = 1, tilt = 0, x = piece.x, z = piece.z;
  if (piece.removing != null) { const t = Math.min(1, (now - piece.removing) / 0.3); s = 1 - ease.in(t); y = 0.2 * t; }
  else if (piece.from) {                                          // a move: slide with a small arc
    const slide = state.game && state.game.input === "watch" ? Math.min(0.45, Math.max(0.08, (state.moveInterval || 0.3) * 0.9)) : 0.45;
    const t = Math.min(1, age / slide), k = ease.out(t);
    x = piece.from[0] + (piece.x - piece.from[0]) * k; z = piece.from[1] + (piece.z - piece.from[1]) * k; y = 0.5 * Math.sin(Math.PI * t);
    if (t >= 1) piece.from = null;
  }
  else if (fam === "cat") { const T = 1.1, t = Math.min(1, age / T);
    y = t < 0.9 ? 1.6 * Math.sin(Math.PI * t / 0.9) : 0; spin = t < 0.25 ? 0 : 2 * Math.PI * ease.out(Math.min(1, (t - 0.25) / 0.65));
    s = t > 0.9 ? 1 - 0.18 * Math.sin((t - 0.9) / 0.1 * Math.PI) : 1; if (t >= 1) s = 1 + 0.03 * Math.sin(now * 2.1 + piece.seed); }
  else if (fam === "x_mark") { const t = Math.min(1, age / 0.55); y = 3.0 * (1 - ease.bounce(t)); tilt = 0.6 * (1 - t); }
  else if (fam === "heart") { spin = now * 0.8 + piece.seed * 1.3; y = 0.05 * Math.sin(now * 1.7 + piece.seed); }
  else { const t = Math.min(1, age / 0.4); y = 1.2 * (1 - ease.out(t)); s = 0.6 + 0.4 * ease.out(t); }   // letters, chess, discs drop in
  const scale0 = piece.size * (piece.cellSize || 1);
  const scale = scale0 / mesh.extent * s;
  const flat = state.flat;
  if (flat) { y = 0; tilt = 0; }                                   // no hop, no lean
  const lean = flat ? 0 : LEAN[fam] ?? 0.35;
  const base = piece.place === "hud" || piece.place === "traycount" ? 0.0 : TILE.height * (piece.cellSize || 1);
  const rest = flat ? base + FLAT_LIFT : base + scale0 * Math.sin(lean + tilt) + 0.04 * s;
  let m = M.translate(x, rest + y, z);
  m = M.multiply(m, M.rotateY(spin));
  m = M.multiply(m, M.rotateX(-Math.PI / 2 + lean + tilt));
  m = M.multiply(m, M.scale(scale, scale, flat ? scale * FLAT_THICK : scale));
  return m;
}
// ---------------------------------------------------------------- soft bodies
// Every drawn piece carries a 2x2x2 spring cage on its mesh's bounding box
// (corner i = x + 2y + 4z, offsets in units of the half-extent).  Each cage
// node is a damped oscillator back to rest.  The piece's own drawn motion drives
// it as an inertial (non-inertial-frame) impulse: a change of the piece's
// velocity dv and of its angular velocity dw kicks node n by
//     du = -weight(n) * (dv + dw x r_n) / halfExtent_world
// where weight is larger on the side away from the "base" (high above the tile
// in the lit view, the glyph's head in the flat view), so lag becomes SHEAR, a
// landing/hop becomes squash/stretch and a turn becomes twist.  Between kicks
// the oscillator is advanced with its closed-form solution (a fixed 2x2
// transition matrix per frame), so it is stable at any frame time.  The same
// kernel exists twice, as CPU JS and as a WebGPU compute shader; both read the
// per-slot parameter block P (16 floats) and keep D (offsets) and U (rates).
const WOB_CAP = 512, WOB_TAU = 0.02, WOB_DMAX = 0.45, WOB_GAIN = 0.3, WOB_ZETA = 0.22;
const wob = {
  on: false, useGpu: false, freq: 8, frameNo: 0, lastNow: null, hi: -1,   // wobble off by default
  D: new Float32Array(WOB_CAP * 24), U: new Float32Array(WOB_CAP * 24), P: new Float32Array(WOB_CAP * 16),
  owner: new Array(WOB_CAP).fill(null), touched: new Int32Array(WOB_CAP), resetFrame: new Int32Array(WOB_CAP),
  free: Array.from({ length: WOB_CAP }, (_, i) => WOB_CAP - 1 - i), gpu: null, gpuState: "off", resetAll: false,
  msStep: 0, msSample: 0, msTotal: 0,
};
state.wobble = wob;

function wobTransition(dt, freq) {                       // exact step of d'' + 2 z w d' + w^2 d = 0
  // Damping follows stiffness: the ratio scales with sqrt(freq / 8), so a softer
  // spring is also a looser one (decay rate z*w goes as freq^1.5).
  const z = Math.min(0.6, Math.max(0.03, WOB_ZETA * Math.sqrt(freq / 8)));
  const w = 2 * Math.PI * freq, wd = w * Math.sqrt(1 - z * z), e = Math.exp(-z * w * dt);
  const C = Math.cos(wd * dt), S = Math.sin(wd * dt), zw = z * w;
  return [e * (C + zw / wd * S), e * S / wd, -e * w * w / wd * S, e * (C - zw / wd * S)];
}

// CPU kernel: P/D/U are typed arrays; slots 0..hi; T = [a, b, c, e].
function wobCpuKernel(P, D, U, hi, T) {
  const [ta, tb, tc, te] = T, k3 = [0, 0, 0];
  for (let s = 0; s <= hi; s++) {
    const p = s * 16, f = P[p + 3]; if (f < 0.5) continue;
    const base = s * 24;
    if (f > 1.5) for (let k = 0; k < 24; k++) { D[base + k] = 0; U[base + k] = 0; }
    const dvx = P[p], dvy = P[p+1], dvz = P[p+2], wx = P[p+4], wy = P[p+5], wz = P[p+6];
    const hx = P[p+8], hy = P[p+9], hz = P[p+10], gx = P[p+12], gy = P[p+13], gz = P[p+14];
    for (let n = 0; n < 8; n++) {
      const rx = n & 1 ? 1 : -1, ry = n & 2 ? 1 : -1, rz = n & 4 ? 1 : -1;
      const wt = 0.15 + 0.85 * (0.5 + 0.5 * (gx * rx + gy * ry + gz * rz));
      const px = hx * rx, py = hy * ry, pz = hz * rz;
      k3[0] = (dvx + (wy * pz - wz * py)) / hx; k3[1] = (dvy + (wz * px - wx * pz)) / hy; k3[2] = (dvz + (wx * py - wy * px)) / hz;
      const b = base + n * 3;
      for (let j = 0; j < 3; j++) {
        const kick = Math.max(-40, Math.min(40, -wt * WOB_GAIN * k3[j]));
        const u0 = U[b + j] + kick, d0 = D[b + j];
        const d1 = ta * d0 + tb * u0, u1 = tc * d0 + te * u0;
        D[b + j] = Math.max(-WOB_DMAX, Math.min(WOB_DMAX, d1)); U[b + j] = u1;
      }
    }
  }
}

const WOB_WGSL = `
struct G { a: f32, b: f32, c: f32, e: f32, dmax: f32, n: u32, gain: f32, pad: f32 };
@group(0) @binding(0) var<uniform> g : G;
@group(0) @binding(1) var<storage, read> P : array<f32>;
@group(0) @binding(2) var<storage, read_write> D : array<f32>;
@group(0) @binding(3) var<storage, read_write> U : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let t = id.x;
  if (t >= g.n * 8u) { return; }
  let s = t / 8u; let n = t % 8u; let p = s * 16u;
  let f = P[p + 3u];
  if (f < 0.5) { return; }
  let b = s * 24u + n * 3u;
  var d = vec3<f32>(D[b], D[b + 1u], D[b + 2u]);
  var u = vec3<f32>(U[b], U[b + 1u], U[b + 2u]);
  if (f > 1.5) { d = vec3<f32>(0.0); u = vec3<f32>(0.0); }
  let dv = vec3<f32>(P[p], P[p + 1u], P[p + 2u]);
  let w = vec3<f32>(P[p + 4u], P[p + 5u], P[p + 6u]);
  let h = vec3<f32>(P[p + 8u], P[p + 9u], P[p + 10u]);
  let gd = vec3<f32>(P[p + 12u], P[p + 13u], P[p + 14u]);
  let r = vec3<f32>(select(-1.0, 1.0, (n & 1u) != 0u), select(-1.0, 1.0, (n & 2u) != 0u), select(-1.0, 1.0, (n & 4u) != 0u));
  let wt = 0.15 + 0.85 * (0.5 + 0.5 * dot(gd, r));
  let k = (dv + cross(w, h * r)) / h;
  let kick = clamp(-wt * g.gain * k, vec3<f32>(-40.0), vec3<f32>(40.0));
  u = u + kick;
  let d1 = g.a * d + g.b * u;
  let u1 = g.c * d + g.e * u;
  let dc = clamp(d1, vec3<f32>(-g.dmax), vec3<f32>(g.dmax));
  D[b] = dc.x; D[b + 1u] = dc.y; D[b + 2u] = dc.z;
  U[b] = u1.x; U[b + 1u] = u1.y; U[b + 2u] = u1.z;
}`;

// One WebGPU cage: persistent D/U on the GPU, per-frame P and transition
// uploaded, D read back (asynchronously, through a small ring of staging buffers).
function makeGpuCage(device, cap) {
  const S = GPUBufferUsage, mk = (size, usage) => device.createBuffer({ size, usage });
  const pbuf = mk(cap * 64, S.STORAGE | S.COPY_DST), dbuf = mk(cap * 96, S.STORAGE | S.COPY_SRC | S.COPY_DST),
        ubuf = mk(cap * 96, S.STORAGE | S.COPY_DST), gbuf = mk(32, S.UNIFORM | S.COPY_DST);
  const module = device.createShaderModule({ code: WOB_WGSL });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [gbuf, pbuf, dbuf, ubuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const staging = Array.from({ length: 3 }, () => mk(cap * 96, S.MAP_READ | S.COPY_DST));
  const free = staging.slice(), gv = new ArrayBuffer(32), gf = new Float32Array(gv), gu = new Uint32Array(gv);
  return {
    device,
    // One step; resolves with the offsets read back (null if every staging buffer is still in flight).
    step(P, hi, T) {
      const n = hi + 1; if (n <= 0) return Promise.resolve(null);
      device.queue.writeBuffer(pbuf, 0, P, 0, n * 16);
      gf[0] = T[0]; gf[1] = T[1]; gf[2] = T[2]; gf[3] = T[3]; gf[4] = WOB_DMAX; gu[5] = n; gf[6] = WOB_GAIN;
      device.queue.writeBuffer(gbuf, 0, gv);
      const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(n * 8 / 64)); pass.end();
      const st = free.pop(), bytes = n * 96;
      if (st) enc.copyBufferToBuffer(dbuf, 0, st, 0, bytes);
      device.queue.submit([enc.finish()]);
      if (!st) return Promise.resolve(null);
      return st.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
        const out = new Float32Array(st.getMappedRange(0, bytes).slice(0)); st.unmap(); free.push(st); return out;
      });
    },
    destroy() { for (const b of [pbuf, dbuf, ubuf, gbuf, ...staging]) b.destroy(); },
  };
}
async function wobInitGpu() {
  if (wob.gpu || wob.gpuState === "pending") return;
  wob.gpuState = "pending";
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); if (!adapter) throw new Error("no adapter");
    const device = await adapter.requestDevice();
    device.lost.then(() => { wob.gpu = null; wob.gpuState = "lost"; wob.useGpu = false; syncWobbleUi(); });
    wob.gpu = makeGpuCage(device, WOB_CAP); wob.gpuState = "ready"; wob.resetAll = true;
  } catch (e) { console.warn("WebGPU wobble unavailable:", e); wob.gpuState = "failed"; wob.useGpu = false; syncWobbleUi(); }
}
// Run the same random inputs through the CPU kernel and a fresh GPU cage; returns the largest difference.
async function wobbleParity(steps = 120) {
  if (!navigator.gpu) return null;
  const device = await (await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })).requestDevice(), cap = 16, cage = makeGpuCage(device, cap);
  const P = new Float32Array(cap * 16), D = new Float32Array(cap * 24), U = new Float32Array(cap * 24);
  let seed = 12345; const rnd = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5;
  let worst = 0;
  for (let t = 0; t < steps; t++) {
    for (let s = 0; s < cap; s++) {
      const p = s * 16; P[p] = rnd() * 6; P[p+1] = rnd() * 6; P[p+2] = rnd() * 6; P[p+3] = t === 0 ? 2 : (s % 5 === 4 ? 0 : 1);
      P[p+4] = rnd() * 20; P[p+5] = rnd() * 20; P[p+6] = rnd() * 20;
      P[p+8] = 0.1 + Math.abs(rnd()); P[p+9] = 0.1 + Math.abs(rnd()); P[p+10] = 0.05 + Math.abs(rnd());
      P[p+12] = rnd() * 2; P[p+13] = rnd() * 2; P[p+14] = rnd() * 2;
    }
    const T = wobTransition(0.005 + Math.abs(rnd()) * 0.2, 8);
    wobCpuKernel(P, D, U, cap - 1, T);
    const out = await cage.step(P, cap - 1, T);
    for (let i = 0; i < cap * 24; i++) worst = Math.max(worst, Math.abs(out[i] - D[i]));
  }
  cage.destroy(); device.destroy(); return worst;
}
wob.parity = wobbleParity;

// ---- sampling a piece's drawn motion -> the cage's parameter block
function wobSlot(piece) {
  let w = piece.wob; if (!w) w = piece.wob = { slot: -1, t: null, p: [0, 0, 0], R: null, v: [0, 0, 0], om: [0, 0, 0] };
  if (w.slot < 0) {
    if (!wob.free.length) return null;
    w.slot = wob.free.pop(); wob.owner[w.slot] = piece; wob.resetFrame[w.slot] = wob.frameNo; w.fresh = true; w.t = null;
    w.v = [0, 0, 0]; w.om = [0, 0, 0];
    wob.D.fill(0, w.slot * 24, w.slot * 24 + 24); wob.U.fill(0, w.slot * 24, w.slot * 24 + 24);
  }
  wob.touched[w.slot] = wob.frameNo; if (w.slot > wob.hi) wob.hi = w.slot;
  return w;
}
function wobSample(piece, m, now) {
  const w = wobSlot(piece); if (!w) return;
  const p = w.slot * 16, P = wob.P;
  const s0 = Math.hypot(m[0], m[1], m[2]), s1 = Math.hypot(m[4], m[5], m[6]), s2 = Math.hypot(m[8], m[9], m[10]);
  const mesh = state.meshes[piece.mesh];
  if (s0 < 1e-5 || s1 < 1e-5 || s2 < 1e-5 || !mesh) { P[p+3] = 0; w.t = null; return; }
  const R = [m[0]/s0, m[1]/s0, m[2]/s0, m[4]/s1, m[5]/s1, m[6]/s1, m[8]/s2, m[9]/s2, m[10]/s2];   // columns = local axes in the world
  const pos = [m[12], m[13], m[14]], sc = [s0, s1, s2];
  const dv = [0, 0, 0], dw = [0, 0, 0];
  const dt = w.t == null ? 0 : now - w.t;
  if (w.t != null && dt > 1e-4 && dt <= 0.1 && w.R) {
    const vr = [(pos[0]-w.p[0])/dt, (pos[1]-w.p[1])/dt, (pos[2]-w.p[2])/dt];
    // relative rotation Q = Rn Ro^T; its antisymmetric part is sin(theta) * axis
    const A = R, B = w.R, Q = (i, j) => A[i]*B[j] + A[3+i]*B[3+j] + A[6+i]*B[6+j];   // Q[row i][col j] (columns stored as A[3c+row])
    const vee = [(Q(2, 1) - Q(1, 2)) / 2, (Q(0, 2) - Q(2, 0)) / 2, (Q(1, 0) - Q(0, 1)) / 2];
    const wr = [vee[0]/dt, vee[1]/dt, vee[2]/dt];
    if (Math.hypot(...vr) < 30 && Math.hypot(...wr) < 60) {            // a respawn or flip is not motion
      const a = 1 - Math.exp(-dt / WOB_TAU);
      for (let k = 0; k < 3; k++) {
        const vn = w.v[k] + (vr[k] - w.v[k]) * a, on = w.om[k] + (wr[k] - w.om[k]) * a;
        dv[k] = vn - w.v[k]; dw[k] = on - w.om[k]; w.v[k] = vn; w.om[k] = on;
      }
    } else { w.v = [0, 0, 0]; w.om = [0, 0, 0]; }
  }
  w.t = now; w.p = pos; w.R = R;
  // world -> mesh-local (R^T); half-extents measured in the world
  const loc = (v) => [R[0]*v[0] + R[1]*v[1] + R[2]*v[2], R[3]*v[0] + R[4]*v[1] + R[5]*v[2], R[6]*v[0] + R[7]*v[1] + R[8]*v[2]];
  const dvl = loc(dv), dwl = loc(dw);
  const hw = [Math.max(sc[0] * mesh.half[0], 0.03), Math.max(sc[1] * mesh.half[1], 0.03), Math.max(sc[2] * mesh.half[2], 0.03)];
  let g;
  if (state.flat) g = [0, 1, 0];                                       // plan view: the glyph's head lags its foot
  else {                                                                // lit view: high above the tile lags
    g = [R[1] * hw[0], R[4] * hw[1], R[7] * hw[2]]; const ym = Math.abs(g[0]) + Math.abs(g[1]) + Math.abs(g[2]);
    g = ym > 1e-6 ? [g[0]/ym, g[1]/ym, g[2]/ym] : [0, 0, 0];
  }
  P[p] = dvl[0]; P[p+1] = dvl[1]; P[p+2] = dvl[2]; P[p+3] = (w.fresh || wob.resetAll) ? 2 : 1;
  P[p+4] = dwl[0]; P[p+5] = dwl[1]; P[p+6] = dwl[2];
  P[p+8] = hw[0]; P[p+9] = hw[1]; P[p+10] = hw[2]; P[p+12] = g[0]; P[p+13] = g[1]; P[p+14] = g[2];
  if (wob.resetAll) { wob.resetFrame[w.slot] = wob.frameNo; wob.D.fill(0, w.slot * 24, w.slot * 24 + 24); }
  w.fresh = false;
}
function wobStep(now) {
  const dt = Math.min(0.1, Math.max(0.001, now - (wob.lastNow ?? now))); wob.lastNow = now;
  const T = wobTransition(dt, wob.freq), hi = wob.hi;
  if (hi >= 0 && wob.useGpu && wob.gpu) {
    const issued = wob.frameNo;
    wob.gpu.step(wob.P, hi, T).then((out) => {
      if (!out) return;
      for (let s = 0; s <= hi; s++) if (wob.owner[s] && wob.resetFrame[s] <= issued) wob.D.set(out.subarray(s * 24, s * 24 + 24), s * 24);
    }).catch((e) => console.warn("wobble readback", e));
  } else if (hi >= 0) wobCpuKernel(wob.P, wob.D, wob.U, hi, T);
  wob.resetAll = false;
}
// Per frame: sample every piece, step, free the slots no piece touched, then deform.
function wobFrame(casters, now) {
  const t0 = performance.now();
  wob.frameNo++; wob.hi = -1; wob.P.fill(0);
  for (const c of casters) if (c[3]) wobSample(c[3], c[2], now);
  const t1 = performance.now();
  wobStep(now);
  for (let s = 0; s < WOB_CAP; s++) {
    const o = wob.owner[s]; if (!o || wob.touched[s] === wob.frameNo) continue;
    o.wob.slot = -1; wob.owner[s] = null; wob.free.push(s);
  }
  wob.msSample = t1 - t0; wob.msStep = performance.now() - t1;
  // best-fit affine for the lit model; the flat shader takes the raw cage
  for (const c of casters) {
    const piece = c[3]; if (!piece) continue;
    if (!piece.wob || piece.wob.slot < 0) { piece.cage = null; continue; }
    const mesh = state.meshes[piece.mesh], b = piece.wob.slot * 24, D = wob.D;
    piece.cage = { slot: piece.wob.slot, mesh };
    if (state.flat) continue;
    // least squares over the +-1 lattice: translation = mean offset, linear part = sum(off_n r_n^T) / 8
    const t = [0, 0, 0], L = new Float64Array(9);
    for (let n = 0; n < 8; n++) {
      const r = [n & 1 ? 1 : -1, n & 2 ? 1 : -1, n & 4 ? 1 : -1];
      for (let j = 0; j < 3; j++) { const d = D[b + n*3 + j]; t[j] += d / 8; for (let k = 0; k < 3; k++) L[j*3 + k] += d * r[k] / 8; }
    }
    const H = mesh.half, C = mesh.center, A = new Float32Array(16); A[15] = 1;
    for (let j = 0; j < 3; j++) {
      let off = C[j] + H[j] * t[j];
      for (let k = 0; k < 3; k++) { const v = H[j] * ((j === k ? 1 : 0) + L[j*3 + k]) / H[k]; A[k*4 + j] = v; off -= v * C[k]; }
      A[12 + j] = off;
    }
    c[2] = M.multiply(c[2], A);
  }
  wob.msTotal = performance.now() - t0;
}
function syncWobbleUi() {
  const g = $("wobble-gpu"); if (!g) return;
  g.checked = wob.useGpu; $("wobble-gpu-label").style.display = navigator.gpu ? "" : "none";
  $("wobble-gpu-label").title = wob.gpuState === "failed" ? "WebGPU failed; using the CPU" : "Run the cage update as a WebGPU compute shader (results are read back one or two frames late)";
}

function applyState(game) {
  // While the real-time socket is open it owns the entities; a state that
  // arrives by HTTP (reset, layout, the socket's own meta) leaves them alone.
  if (game.input === "realtime" && sock.open) delete game.entities;
  const now = state.time, previous = state.game;
  state.game = game;
  document.body.dataset.game = game.game;
  if (!previous || previous.game !== game.game || previous.id !== game.id || previous.rows !== game.rows) {
    state.pieces.clear(); uploadMaterials(game);
    pred.sent = []; pred.closed = []; pred.open = null; pred.ship = null; ticks.applied = -1; pred.shots = [];
    drawnShip.page = drawnShip.server = 0;
  } else if (previous.palette_name !== game.palette_name) uploadMaterials(game);
  rebuildBoards(game);
  delveOverlay(game);
  arcadeEvents(game, now);
  if (game.you) noteShield(game, now);
  if (game.sync_hint != null) sync.hint = game.sync_hint;
  resultSplash(game);
  const seen = new Set();
  const place = (p, rackPiece) => {
    const [x, z, where, cellSize] = cellPosition(game, p.cell, rackPiece);
    const key = rackPiece ? `rack:${p.id}` : p.id;
    seen.add(key);
    const live = state.pieces.get(key);
    const size = where === "tray" ? 0.3 : where === "traycount" ? 0.19 : where === "hud" ? 0.42 : family(p.mesh) === "cat" ? 0.38 : family(p.mesh) === "chess" ? 0.42 : family(p.mesh) === "disc" ? 0.34 : 0.36;
    if (live && live.removing == null) {
      if (Math.abs(live.x - x) > 1e-6 || Math.abs(live.z - z) > 1e-6) { live.from = [live.x, live.z]; live.t0 = now; live.x = x; live.z = z; }
      live.mesh = p.mesh; live.color = p.color;
    } else {
      state.pieces.set(key, { id: key, mesh: p.mesh, color: p.color, x, z, place: where, cellSize: cellSize || 1, size, t0: now, removing: null, seed: Math.random() * 6.28, from: null });
    }
  };
  for (const p of game.pieces) place(p, false);
  // The world time this state stands for, on the page's clock: with
  // input-timed rounds, the moment the last landed interval closed.
  const stamp = worldStamp(game, now);
  for (const e of game.entities || []) {
    seen.add(e.id);
    const live = state.pieces.get(e.id);
    if (live && live.removing == null) Object.assign(live, { sx: e.x, sz: e.y, vx: e.vx, vz: e.vy, sh: e.z || 0, vh: e.vz || 0, tRecv: stamp, mesh: e.mesh, color: e.color, size: e.size, roll: e.roll, spin: e.spin, bob: e.bob, ground: !!e.ground });
    else state.pieces.set(e.id, { id: e.id, entity: true, lane: Number(String(e.id).split(":")[0]), sx: e.x, sz: e.y, x: e.x, z: e.y, vx: e.vx, vz: e.vy, sh: e.z || 0, h: e.z || 0, vh: e.vz || 0, tRecv: stamp, mesh: e.mesh, color: e.color,
                                  size: e.size, roll: e.roll, spin: e.spin, bob: e.bob, face: !!e.face, translucent: !!e.translucent, ribbon: !!e.ribbon, ground: !!e.ground, glow: !!e.glow, lean: e.lean, t0: now, removing: null, seed: 0 });
  }
  if (game.rack) for (const p of game.rack.pieces) place(p, true);
  syncWalk(game);
  if (game.input === "realtime") reconcile(game); else pred.ship = null;
  for (const [key, live] of state.pieces) if (!seen.has(key) && live.removing == null && (game.entities || !live.entity)) live.removing = now;
  const best = (game.record && game.record.best ? ` · ${game.player}'s best ${game.record.best.score}` : "") + awardText(game);
  renderModes(game);
  renderGameControls(game);
  chessClockSync(game);
  compute.sync(game.compute || null);
  renderHotbars(game);
  showPads(game);
  renderInventory(game);
  renderWordleOptions(game);
  const authority = authorityText(game);
  status.textContent = `${game.status} · ${game.title}${best}${authority ? " · " + authority : ""}${game.compute && compute.ready ? " · " + compute.statusText() : ""}`;
  $("hud").textContent = `${game.status}${best}`;
  if (game.record && game.record.plays !== state.recordPlays) { state.recordPlays = game.record.plays; if ($("scores").classList.contains("open")) showScores(); }
  const lichess = game.lichess_control;
  $("lichess-stop").disabled = !lichess || lichess.pause_requested;
  $("lichess-resume").disabled = !lichess || !lichess.pause_requested;
  syncMeowdokuBar(game);
  if (game.text_board) $("position").textContent =
    `${game.turn} to move${game.check ? " (check)" : ""}${game.outcome ? ` · game over: ${game.outcome}` : ""}\n${game.text_board}`
    + (game.legal_uci && game.legal_uci.length ? `\nlegal: ${game.legal_uci.join(" ")}` : "");
  drawLearnPanel(game);
  syncLayerControls();
  if (game.learning) {
    const L = game.learning;
    $("demo-learn").textContent = L.demo_learning ? "Stop demo learning" : "Start demo learning";
    $("learn-status").textContent = `${L.human_games} of your games kept`
      + (L.last ? ` · trained on ${L.last}${L.at ? " at " + L.at : ""}` : "") + (L.demo ? ` · ${L.demo}` : "");
  }
  if (game.game === "chess_arena") $("arena-stop").textContent = game.arena_running ? "Stop training" : "Resume training";
  // The arena plays itself: while it is the game on screen, ask the server
  // for one move every few hundred milliseconds (through the same queue).
  // Watch mode: the server's thread moves; poll again as soon as this reply
  // is in (a short breath so the UI thread keeps up) and time the moves so
  // the slide animation lasts about as long as a move takes to arrive.
  clearTimeout(state.arenaTimer);
  if (game.input === "watch") {
    if (previous && previous.game === game.game && game.moves !== previous.moves) {
      const gap = now - (state.lastMoveAt || now);
      state.moveInterval = state.moveInterval ? state.moveInterval * 0.7 + gap * 0.3 : gap;
      state.lastMoveAt = now;
    }
    if (!document.hidden) state.arenaTimer = setTimeout(() => { if (state.game === game) act({ type: "tick" }); }, 40);
  }
  // Real time: the next tick leaves as soon as this one is in (a short breath
  // for the UI thread), carrying the held controls; the server lands the
  // wall time between ticks.
  if (game.input === "text") { const entry = $("entry"); if (entry.value.toUpperCase() !== game.text) entry.value = game.text; }
}

// ---------------------------------------------------------------- room-graph overlays
// A room-graph game (``game.roomview``, ``game.world``): a fade with the room's name whenever the room changes
// (``roomview.serial``), and the MAP (M or the Map button): the rooms visited on the coarse grid, their joins
// (a one-way drop is dashed), the room you are in and where you stand.  Self-contained: DOM made here.
const delveUi = { serial: null, id: null, fade: null, name: null, map: null };
function delveOverlay(game) {
  const rv = game && game.roomview;
  if (!rv) {
    if (delveUi.fade) delveUi.fade.style.opacity = 0;
    if (delveUi.map) delveUi.map.hidden = true;
    delveUi.serial = null;
    return;
  }
  if (!delveUi.fade) {
    const style = document.createElement("style");
    style.textContent = "#delve-fade{position:fixed;inset:0;background:#05060c;opacity:0;pointer-events:none;z-index:40;display:flex;"
      + "align-items:flex-end;justify-content:center;padding-bottom:12vh;color:#e8e6f0;font:600 1.1rem system-ui,sans-serif}"
      + "#delve-map{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:41;background:rgba(8,9,16,.92);"
      + "border:1px solid #4a4a66;border-radius:10px;padding:10px;max-width:92vw;max-height:80vh;box-shadow:0 8px 40px #000a}"
      + "#delve-map svg{display:block;max-width:86vw;max-height:70vh}#delve-map small{display:block;color:#9a98b0;margin-top:6px;font:12px system-ui,sans-serif}";
    document.head.append(style);
    delveUi.fade = Object.assign(document.createElement("div"), { id: "delve-fade" });
    delveUi.name = document.createElement("span");
    delveUi.fade.append(delveUi.name);
    delveUi.map = Object.assign(document.createElement("div"), { id: "delve-map", hidden: true });
    document.body.append(delveUi.fade, delveUi.map);
  }
  if (delveUi.serial != null && rv.serial !== delveUi.serial) {
    // a quick fade from black into the next room, its name on the dark
    delveUi.name.textContent = game.room ? game.room.name : "";
    delveUi.fade.style.transition = "none"; delveUi.fade.style.opacity = 0.92; void delveUi.fade.offsetWidth;
    delveUi.fade.style.transition = "opacity .55s ease-out .12s"; delveUi.fade.style.opacity = 0;
  }
  delveUi.serial = rv.serial;
  const world = game.world;
  delveUi.map.hidden = !(world && world.show);
  if (!delveUi.map.hidden) {
    const key = JSON.stringify([world.rooms.map((r) => r.seen), world.current, world.joins.length]);
    const you = game.you && game.room ? [game.room.gx * world.unit[0] + game.you.x, game.room.gy * world.unit[1] + game.you.y] : null;
    if (delveUi.key !== key || you) {
      delveUi.key = key;
      const [gw, gh] = world.grid, [uw, uh] = world.unit, cw = 30, ch = cw * uh / uw;
      const sx = (tx) => (tx / uw) * cw, sy = (ty) => (ty / uh) * ch;
      let svg = `<svg viewBox="-4 -4 ${gw * cw + 8} ${gh * ch + 8}" xmlns="http://www.w3.org/2000/svg">`;
      for (const r of world.rooms) {
        if (!r.seen) continue;
        const cur = r.id === world.current;
        svg += `<rect x="${r.gx * cw + 1}" y="${r.gy * ch + 1}" width="${r.cw * cw - 2}" height="${r.ch * ch - 2}" rx="3" `
          + `fill="${cur ? "#3b3f7a" : "#232640"}" stroke="${cur ? "#ffd75e" : "#7a7aa8"}" stroke-width="${cur ? 2 : 1}"/>`;
      }
      for (const j of world.joins) {
        const x = sx(j.x), y = sy(j.y), vertical = j.side === "L" || j.side === "R";
        svg += `<rect x="${x - (vertical ? 2 : 6)}" y="${y - (vertical ? 6 : 2)}" width="${vertical ? 4 : 12}" height="${vertical ? 12 : 4}" `
          + `fill="${j.oneway ? "#ff9f6e" : "#9ee6a8"}"${j.oneway ? ' stroke="#05060c" stroke-dasharray="2 2"' : ""}/>`;
      }
      if (you) svg += `<circle cx="${sx(you[0])}" cy="${sy(you[1])}" r="3.2" fill="#7dff8a" stroke="#05060c"/>`;
      delveUi.map.innerHTML = svg + `</svg><small>${world.rooms.filter((r) => r.seen).length}/${world.rooms.length} rooms · `
        + `green: door · orange: one-way drop · M closes</small>`;
    }
  }
}
// ---------------------------------------------------------------- compute products
// A game that declares ``game.compute`` (see compute.js) has its law run by the
// shared compute runtime -- WebGPU compute, or its wasm product -- and its
// particles drawn here, in the shared scene, as GL points.  A game's own
// buttons, pickers, ranges and selects come as ``game.controls_ui``.
let cloud = null;
const compute = new ComputeRuntime({
  onStatus: () => { const g = state.game; if (g && g.compute) status.textContent = `${g.status} · ${compute.statusText()}`; },
  onSamples: (post, fields) => act({ type: post.action, ...fields }),
  onSupport: () => { if (state.game) renderGameControls(state.game); },
});
window.meowdokuDebug.compute = compute;
function renderGameControls(game) {
  const box = $("gamectl"), specs = game && game.controls_ui;
  if (!box) return;
  if (!specs) { if (box.childElementCount) box.replaceChildren(); delete box.dataset.key; return; }
  const key = `${game.game}:${specs.map((s) => s.id).join(",")}`;
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.replaceChildren(...specs.map((s) => {
      let el;
      if (s.kind === "button") { el = document.createElement("button"); el.textContent = s.label; el.onclick = () => act({ type: s.action }); }
      else if (s.kind === "file") {
        el = document.createElement("label"); el.className = "gctl-file"; el.append(s.label);
        const input = document.createElement("input"); input.type = "file"; input.accept = "image/*"; input.hidden = true;
        if (s.capture) input.setAttribute("capture", s.capture);
        input.onchange = async () => { const f = input.files[0]; input.value = ""; await compute.pickImage(f, s.sample); };
        el.append(input);
      } else {
        el = document.createElement("label"); const tag = document.createElement("span"); tag.textContent = s.label + " "; el.append(tag);
        const input = document.createElement(s.kind === "select" ? "select" : "input");
        if (s.kind === "range") Object.assign(input, { type: "range", min: s.min, max: s.max, step: s.step });
        else input.innerHTML = s.options.map(([v, text]) => `<option value="${v}">${text}</option>`).join("");
        input.onchange = () => act({ type: s.action, [s.field]: s.kind === "range" || typeof s.options[0][0] === "number" ? Number(input.value) : input.value });
        el.append(input);
      }
      el.dataset.ctl = s.id;
      return el;
    }));
  }
  for (const s of specs) {
    const el = box.querySelector(`[data-ctl="${s.id}"]`);
    if (!el) continue;
    if (s.kind === "button") el.disabled = s.enabled === false;
    const input = el.querySelector("input:not([type=file]), select");
    if (input && s.id === "size" && compute.support) {
      // a size whose bindings this device cannot structurally run is offered but disabled, with the reason (memory is never a gate: it is allocated and a failure reported)
      for (const option of input.options) {
        const verdict = compute.support[option.value];
        option.disabled = !!verdict && !verdict.ok;
        const base = option.dataset.label || (option.dataset.label = option.textContent);
        option.textContent = verdict && !verdict.ok ? `${base} (unsupported here)` : base;
        option.title = verdict && !verdict.ok ? verdict.reason : verdict && verdict.note ? verdict.note : "";
        if (verdict && !verdict.ok) input.title = verdict.reason;
      }
    }
    if (input && document.activeElement !== input) input.value = String(s.value);
  }
}
function pointsFrame(cam, now) {
  const g = state.game;
  if (!g || !g.compute) return;
  compute.frame(now);
  const data = compute.ready && compute.product ? compute.product.points() : null;
  const rec = compute.rec;
  if (!data || !rec) return;
  if (!cloud) cloud = new PointCloud(gl, link, compile);
  const ext = state.extent, eye = cam.eye, target = cam.target || [ext.width / 2, 0, ext.depth / 2];
  const dist = state.flat ? 1 : Math.hypot(eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]);
  const pxPerUnit = cam.proj[5] * canvas.height / 2 / dist;
  cloud.draw(data, { viewProj: cam.viewProj, view: rec.geometry.view, board: [g.width, g.height],
                     y: TILE.height + 0.03, pxPerUnit, stride: rec.display_stride });
}

// ---------------------------------------------------------------- draw
// The lit scene from ``cam`` into whatever framebuffer is bound: the main
// view, or the projection glass's broadcast camera (opts.plate: draw the
// glass itself -- never into its own broadcast).
function litPass(cam, now, casters, opts = {}) {
  // pass 2: the lit scene
  const P = state.programs.phong; gl.useProgram(P);
  const span = Math.max(state.extent.width, state.extent.depth);
  const keyView = M.apply(cam.view, light.position);
  const sun = sunState(), boardCentre = [state.extent.width / 2, 0, state.extent.depth / 2];
  const sunReach = Math.hypot(...sub(light.position, boardCentre));
  let keyColour = state.flat ? [1.0, 0.95, 0.86] : sun.colour, fillColour = [0.75, 0.85, 1.0];
  const globe = !state.flat && SKY.dir && state.game && state.game.globe ? SKY.dir : null;
  if (globe) {
    // The sun's light as the atmosphere leaves it (reddening at sunrise and
    // sunset, gone at night), and the sky's blue (deepening to night) as the
    // fill; a little starlight keeps the night readable.
    const through = sunThroughAir(globe), peak = Math.max(...through, 1e-4);
    keyColour = through.map((v) => v / peak);
    sun.strength = Math.min(1, peak * 1.25);
    const day = Math.max(0, Math.min(1, (globe[1] + 0.12) / 0.4));
    fillColour = [0.35 + 0.4 * day, 0.42 + 0.43 * day, 0.62 + 0.38 * day].map((v) => v * (0.35 + 0.65 * day));
  }
  // The sun's intensity is set for its distance to the board's centre (soft,
  // even light); the fill light carries the arcade ring's colour and activity.
  // The house light is turned down so the arcade bulbs carry the scene: the
  // sun at ~60%, a softer fill; then each bulb is a point light in its own
  // colour at its own place, as bright as the chase makes it.
  // With the Arcade lights control off the ring is gone: no bulb lights, no
  // bulb meshes, and the house light returns to full (the turn-down was for
  // the bulbs' sake).
  const arcadeOn = !state.flat && layerOn("arcade");
  const house = arcadeOn ? 1.0 : 1.6;
  const bulbs = arcadeOn ? arcadeBulbs() : [];
  const bulbStates = bulbs.map((_, i) => arcadeBulb(i, bulbs, now));
  const bulbLights = bulbs.map((p, i) => ({ view: M.apply(cam.view, p), colour: ARCADE.colours[bulbStates[i].ci],
                                           level: bulbStates[i].level }));
  // Lightning lights the scene: each strike is a blue-white point light at
  // the cloud, the impact and the bolt's middle, fading with the strike.
  bulbLights.push(...lightningLights(cam, now));
  bulbLights.push(...blastLights(cam, now));
  for (const nav of markerLights(now)) bulbLights.push({ view: M.apply(cam.view, nav.at), colour: nav.light, level: 0, intensity: nav.intensity });
  bulbLights.push(...headlights(cam));
  const count = Math.min(3 + bulbLights.length, 100);
  gl.uniform1i(gl.getUniformLocation(P, "uNumLights"), count);
  gl.uniform3fv(gl.getUniformLocation(P, "uLightPos[0]"), new Float32Array([...keyView, 0.9 * span, 0.4 * span, 0.4 * span - 1.0,
    -0.6 * span, -0.1 * span, 0.3 * span - 1.0, ...bulbLights.slice(0, count - 3).flatMap((b) => b.view)]));
  gl.uniform3fv(gl.getUniformLocation(P, "uLightColor[0]"), new Float32Array([...keyColour, ...fillColour, 1.0, 0.8, 0.7,
    ...bulbLights.slice(0, count - 3).flatMap((b) => b.colour)]));
  gl.uniform1fv(gl.getUniformLocation(P, "uLightIntensity[0]"), new Float32Array([
    (state.flat ? 2.6 : 1.45 * house * sun.strength) * sunReach * sunReach, 0.19 * house * span * span, 0.14 * house * span * span,
    ...bulbLights.slice(0, count - 3).map((b) => (b.intensity ?? ARCADE.lamp * b.level))]));
  gl.uniform1iv(gl.getUniformLocation(P, "uLightGroupId[0]"), new Int32Array(count).fill(-1));
  // Spot cones (Pluck's light loop): a direction and (cos inner, cos outer)
  // per light; zero direction = a point light (the key, fills, bulbs, lightning).
  const spotDir = new Float32Array(count * 3), spotCone = new Float32Array(count * 2);
  bulbLights.slice(0, count - 3).forEach((b, i) => {
    if (!b.spotDir) return;
    spotDir.set(b.spotDir, (i + 3) * 3); spotCone.set(b.spotCone, (i + 3) * 2);
  });
  gl.uniform3fv(gl.getUniformLocation(P, "uLightSpotDir[0]"), spotDir);
  gl.uniform2fv(gl.getUniformLocation(P, "uLightSpotCone[0]"), spotCone);
  gl.uniform1f(gl.getUniformLocation(P, "uLightCalibration"), 1.0);
  gl.uniformMatrix3fv(gl.getUniformLocation(P, "uCatCcmMatrix"), false, new Float32Array([1,0,0, 0,1,0, 0,0,1]));
  gl.uniform1i(gl.getUniformLocation(P, "uEnableSpecular"), 1); gl.uniform1i(gl.getUniformLocation(P, "uEnableEmissionDirect"), 0);
  gl.uniform1i(gl.getUniformLocation(P, "uRenderPass"), 0); gl.uniform1f(gl.getUniformLocation(P, "uFieldGain"), 0.0);
  gl.uniform1f(gl.getUniformLocation(P, "uExposure"), state.debug.exposure);
  gl.uniform1i(gl.getUniformLocation(P, "uThermalEmission"), 0); gl.uniform1f(gl.getUniformLocation(P, "uThermalGain"), 0.08);
  gl.uniform1f(gl.getUniformLocation(P, "uThermalRefK"), 1000.0); gl.uniform1f(gl.getUniformLocation(P, "uThermalFloorK"), 720.0);
  for (const [name, unit] of Object.entries(state.units)) gl.uniform1i(gl.getUniformLocation(P, name), unit);
  gl.uniform1i(gl.getUniformLocation(P, "uShadowEnabled"), state.debug.shadows ? 1 : 0);
  gl.uniform1i(gl.getUniformLocation(P, "uShadowLightCount"), 1);
  gl.uniform2f(gl.getUniformLocation(P, "uShadowTexelSize"), 1 / SHADOW_SIZE, 1 / SHADOW_SIZE);
  gl.uniform1f(gl.getUniformLocation(P, "uShadowBias"), 0.0015);
  const shadowLoc = gl.getUniformLocation(P, "uShadowMatrices[0]");
  const drawLit = (vao, count, model, materialId) => {
    const mv = M.multiply(cam.view, model);
    gl.uniformMatrix4fv(gl.getUniformLocation(P, "uMVP"), false, M.multiply(cam.proj, mv));
    gl.uniformMatrix4fv(gl.getUniformLocation(P, "uMV"), false, mv);
    gl.uniformMatrix4fv(shadowLoc, false, M.multiply(light.viewProj, model));
    gl.bindVertexArray(vao);
    if (materialId != null) gl.vertexAttribI4i(2, materialId, 0, 0, 0);
    // aCullImmune = 1: the extruded meshes' walls are wound inward, so draw both sides
    // (the shader flips back-face normals).
    gl.vertexAttribI4i(4, 0, 0, 0, 0); gl.vertexAttribI4i(5, 1, 0, 0, 0); gl.vertexAttrib3f(6, 0, 0, 0); gl.vertexAttrib2f(3, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, count);
  };
  drawTerrain(now, drawLit);
  for (const [vao, count, model, piece] of casters) drawLit(vao, count, model, piece ? state.materialIndex[piece.color] : null);
  // Emitters: the sun's orb and the arcade bulbs, Phong-shaded with emissive
  // materials (a bulb's level picks its brightness step).
  if (!state.flat) {
    const orb = state.meshes.__sphere;
    drawLit(orb.vao, orb.count, M.multiply(M.translate(...light.position), M.scale(0.16, 0.16, 0.16)), state.materialIndex.sun_orb);
    bulbs.forEach((p, i) => {
      const { level, ci } = bulbStates[i];
      const step = ARCADE_LEVELS.reduce((best, v, k) => (Math.abs(v - 2.4 * level) < Math.abs(ARCADE_LEVELS[best] - 2.4 * level) ? k : best), 0);
      drawLit(orb.vao, orb.count, M.multiply(M.translate(...p), M.scale(0.075, 0.075, 0.075)), state.materialIndex[`bulb${ci}_${step}`]);
    });
    for (const nav of markerLights(now)) drawLit(orb.vao, orb.count, M.multiply(M.translate(...nav.at), M.scale(nav.size, nav.size, nav.size)), state.materialIndex[nav.material]);
  }
  drawCockpit(cam, drawLit);
  // the glass plate under everything: blended, no depth write
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
  if (layerOn("glass")) drawLit(state.glass.vao, state.glass.count, M.identity(), null);
  drawLightning(cam, now, drawLit);
  for (const piece of state.glassPieces || []) {
    const mesh = state.meshes[piece.mesh] || state.meshes.disc;
    drawLit(mesh.vao, mesh.count, pieceModel(piece, now), state.materialIndex[`${piece.color}@glass`] ?? state.materialIndex[piece.color]);
  }
  drawShield(now, drawLit);
  drawBeam(now, drawLit);
  drawVitals(now, drawLit);
  if (opts.plate) drawProjectionGlass(cam, drawLit);
  gl.depthMask(true); gl.disable(gl.BLEND);
}

function draw(ms) {
  const now = ms / 1000; stepCamView(now - (state.time || now)); state.time = now;
  const w = canvas.clientWidth * devicePixelRatio | 0, h = canvas.clientHeight * devicePixelRatio | 0;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const clear = state.game?.game === "meowdoku" ? MEOW_LOOK.background : [0.085, 0.085, 0.1];
  gl.viewport(0, 0, w, h); gl.clearColor(...clear, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  if (!state.game) { requestAnimationFrame(draw); return; }
  const cam = camera(); updateLight(now);
  const casters = state.game && state.game.board === "free" && !layerOn("floor") ? []
    : [[state.buttons.vao, state.buttons.count, M.identity(), null]];
  if (state.rackButtons) casters.push([state.rackButtons.vao, state.rackButtons.count, M.identity(), null]);
  if (state.trayButtons) casters.push([state.trayButtons.vao, state.trayButtons.count, M.identity(), null]);
  const removed = [];
  for (const [key, piece] of state.pieces) {
    if (piece.removing != null && now - piece.removing > 0.3) { removed.push(key); continue; }
    if (piece.ribbon) { if (piece.removing == null) (state.ribbonEnds ||= []).push(piece); continue; }
    if (piece.glow) { if (piece.removing == null) { const m = pieceModel(piece, now); (state.glowPoints ||= []).push([m[12], m[13], m[14], piece.size]); } continue; }
    if (piece.translucent && !state.flat) { (state.glassPieces ||= []).push(piece); continue; }
    // First person: the eye is in the ship's nose -- its body is not drawn
    // (its model is still advanced: the camera rides it).
    if (noseCam() && piece.entity && state.game.predict && piece.lane === state.game.predict.lane) { pieceModel(piece, now); continue; }
    casters.push([state.meshes[piece.mesh].vao, state.meshes[piece.mesh].count, pieceModel(piece, now), piece]);
  }
  for (const key of removed) state.pieces.delete(key);
  if (wob.on) wobFrame(casters, now); else for (const c of casters) if (c[3]) c[3].cage = null;
  if (state.flat) { drawFlat(cam, casters); pointsFrame(cam, now); requestAnimationFrame(draw); return; }
  // pass 1: depth from the key light
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.shadowMap.fbo); gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE); gl.clear(gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.programs.depth); gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(2.0, 4.0);
  for (const [vao, count, model] of [...casters, ...terrainShadowCasters()]) {
    gl.uniformMatrix4fv(gl.getUniformLocation(state.programs.depth, "uMVP"), false, M.multiply(light.viewProj, model));
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, count);
  }
  gl.disable(gl.POLYGON_OFFSET_FILL);
  // The projection glass's broadcast: the scene through its wide-angle camera,
  // warped into colour layer 1 (the glass shows it); the screen lens steps
  // aside while the glass carries the warp.
  const plateOn = projectionGlassOn();
  if (plateOn) renderBroadcast(now, casters);
  const lensK = plateOn ? 0 : lensStrength();
  gl.bindFramebuffer(gl.FRAMEBUFFER, lensK > 0 ? lensTarget(w, h).fbo : null); gl.viewport(0, 0, w, h);
  if (lensK > 0) { gl.clearColor(...clear, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); }
  drawSky(cam, now);                                // into whatever target the scene draws in (the lens's, or the screen)
  litPass(cam, now, casters, { plate: plateOn });
  drawCityLights(cam);
  pointsFrame(cam, now);
  state.glowPoints = [];
  state.glassPieces = [];
  state.ribbonEnds = [];
  // (the sun and the bulbs are drawn as emissive geometry in the lit pass)
  gl.bindVertexArray(null);
  if (lensK > 0) lensPresent(w, h, lensK);
  requestAnimationFrame(draw);
}

// Flat view: no shadow pass, no glass, no bulb.  Each piece is drawn twice: first a
// slightly enlarged, slightly lower, near-black copy (the outline), then the
// cel-shaded piece over it.
function drawFlat(cam, casters) {
  const P = state.programs.cel; gl.useProgram(P);
  gl.uniform3fv(gl.getUniformLocation(P, "uPal[0]"), state.celPalette);
  const mvpLoc = gl.getUniformLocation(P, "uMVP"), mvLoc = gl.getUniformLocation(P, "uMV"), outLoc = gl.getUniformLocation(P, "uOutline");
  const ffdLoc = gl.getUniformLocation(P, "uFfd"), cageLoc = gl.getUniformLocation(P, "uCage[0]"), ctrLoc = gl.getUniformLocation(P, "uCtr"), halfLoc = gl.getUniformLocation(P, "uHalf");
  const drawOne = (vao, count, model, materialId, outline, piece) => {
    const cg = piece && piece.cage;
    gl.uniform1i(ffdLoc, cg ? 1 : 0);
    if (cg) { gl.uniform3fv(cageLoc, wob.D.subarray(cg.slot * 24, cg.slot * 24 + 24)); gl.uniform3fv(ctrLoc, cg.mesh.center); gl.uniform3fv(halfLoc, cg.mesh.half); }
    const mv = M.multiply(cam.view, model);
    gl.uniformMatrix4fv(mvpLoc, false, M.multiply(cam.proj, mv)); gl.uniformMatrix4fv(mvLoc, false, mv);
    gl.uniform1i(outLoc, outline ? 1 : 0);
    gl.bindVertexArray(vao);
    if (materialId != null) gl.vertexAttribI4i(2, materialId, 0, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, count);
  };
  for (const [vao, count, model, piece] of casters) if (!piece) drawOne(vao, count, model, null, false);
  const grow = M.scale(FLAT_OUTLINE, FLAT_OUTLINE, 1), drop = M.translate(0, -0.006, 0);
  for (const [vao, count, model, piece] of casters) if (piece) drawOne(vao, count, M.multiply(drop, M.multiply(model, grow)), null, true, piece);
  for (const [vao, count, model, piece] of casters) if (piece) drawOne(vao, count, model, state.materialIndex[piece.color], false, piece);
  gl.bindVertexArray(null);
}

// ---------------------------------------------------------------- server
async function api(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await response.json();
  if (data.error) { status.textContent = data.error; throw new Error(data.error); }
  return data;
}
// Actions run one after another: a swipe or a select-then-move sends
// several, and a late reply must never overwrite a newer state.
let chain = Promise.resolve();
const act = (payload) => {
  const step = chain.then(async () => { if (payload.type !== "tick") console.log("[meow] action", JSON.stringify(payload)); applyState(await api("/api/action", payload)); }).catch((e) => console.warn(e));
  chain = step;
  return step;
};
// ---------------------------------------------------------------- chess against the network
// The user's move animates at once (the page lit the legal squares, so it
// knows the move is legal); the server's reply confirms it, or the piece
// slides back.  The network searches on the server while ``game.thinking``;
// the page polls for its move.  The clock is the server's: each side's time
// as of the last state, ticked down locally for the running side.
const chessClock = { box: null, snap: null, poll: null, asked: false };
function optimisticMove(game, h) {
  if (!game || game.game !== "chess_vs_net" || game.thinking || game.outcome || game.selected == null || !h || h.kind !== "board") return;
  const to = h.r * game.cols + h.c;
  if (!game.hints || !game.hints[to]) return;                      // not one of the lit squares: the server decides
  const mover = game.pieces.find((p) => p.cell === game.selected), live = mover && state.pieces.get(mover.id);
  if (!live || live.removing != null) return;
  const victim = game.pieces.find((p) => p.cell === to && p.id !== mover.id), taken = victim && state.pieces.get(victim.id);
  if (taken && taken.removing == null) taken.removing = state.time;
  const [x, z] = cellPosition(game, to, false);
  live.from = [live.x, live.z]; live.t0 = state.time; live.x = x; live.z = z;
}
const clockText = (s) => { s = Math.max(0, s); return s < 10 ? s.toFixed(1) : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
function chessClockBox() {
  if (chessClock.box) return chessClock.box;
  const box = document.createElement("div");
  box.id = "chess-clock";
  box.style.cssText = "position:fixed;left:12px;top:calc(var(--bar-h, 40px) + 10px);z-index:3;display:none;font:14px/1.3 system-ui,sans-serif;color:#e8e8ea;background:rgba(22,22,26,0.82);border:1px solid #3a3a42;border-radius:10px;padding:8px 10px;min-width:150px";
  box.innerHTML = `<style>#chess-clock .side{display:flex;justify-content:space-between;gap:14px;padding:3px 6px;border-radius:6px;font-variant-numeric:tabular-nums}
    #chess-clock .side b{font:600 22px/1.1 ui-monospace,Consolas,monospace}#chess-clock .side.run{background:#3b5a8a}#chess-clock .side.low b{color:#ff8a7a}
    #chess-clock .think{min-height:18px;opacity:.85;margin-top:4px}#chess-clock .dots::after{content:"";animation:cc-dots 1s steps(4,end) infinite}
    @keyframes cc-dots{0%{content:""}25%{content:"."}50%{content:".."}75%{content:"..."}}
    #chess-clock select{background:#26262c;color:#e8e8ea;border:1px solid #3a3a42;border-radius:6px;padding:3px 6px;font:inherit;margin-top:6px;width:100%}</style>
    <div class="side" data-side="black"><span>Network</span><b>--</b></div>
    <div class="side" data-side="white"><span>You</span><b>--</b></div>
    <div class="think"></div><select title="Time control" aria-label="Time control"></select>`;
  box.querySelector("select").onchange = (e) => act({ type: "clock", name: e.target.value });
  document.body.append(box);
  return (chessClock.box = box);
}
function chessClockSync(game) {
  clearTimeout(chessClock.poll);
  if (!game || game.game !== "chess_vs_net" || !game.clock) { if (chessClock.box) chessClock.box.style.display = "none"; chessClock.snap = null; return; }
  const box = chessClockBox(), c = game.clock;
  box.style.display = "block";
  chessClock.snap = { clock: c, at: performance.now() }; chessClock.asked = false;
  const select = box.querySelector("select");
  if (select.dataset.key !== c.controls.join()) { select.dataset.key = c.controls.join(); select.innerHTML = c.controls.map((n) => `<option>${n}</option>`).join(""); }
  if (document.activeElement !== select) select.value = c.control;
  const think = box.querySelector(".think");
  think.classList.toggle("dots", !!game.thinking);
  think.textContent = c.flagged ? `${c.flagged} lost on time` : game.thinking ? "network thinking" : "";
  chessClockTick();
  // The network's move arrives on the server's thread: ask again soon.
  if (game.thinking) chessClock.poll = setTimeout(() => { if (state.game === game) act({ type: "tick" }); }, 150);
}
function chessClockTick() {
  const snap = chessClock.snap, box = chessClock.box;
  if (!snap || !box) return;
  const c = snap.clock, spent = (performance.now() - snap.at) / 1000;
  for (const side of ["white", "black"]) {
    const row = box.querySelector(`[data-side="${side}"]`);
    const left = c.timed ? c[side] - (c.running === side ? spent : 0) : null;
    row.querySelector("b").textContent = c.timed ? clockText(left) : "--";
    row.classList.toggle("run", c.running === side);
    row.classList.toggle("low", c.timed && left < 20);
    if (c.running === side && left <= 0 && !chessClock.asked) { chessClock.asked = true; act({ type: "tick" }); }   // the server confirms the flag
  }
}
setInterval(chessClockTick, 100);
const newGame = (payload) => {
  if (payload.game === "meowdoku" && $("palette").value) payload = { ...payload, palette: $("palette").value };
  const step = chain.then(async () => { console.log("[meow] new", JSON.stringify(payload)); applyState(await api("/api/new", payload)); }).catch((e) => console.warn(e));
  chain = step;
  return step;
};

// ---------------------------------------------------------------- demo player
// The trained autopilot (autopilot.py, run by the server) is the demo player:
// with a game that has one open and no control from the user for DEMO_START_S
// (DEMO_IDLE_S after the user last played), the page hands the ship to it;
// the user's first control takes it back and starts a fresh game.
const DEMO_GAMES = new Set(["space_invaders", "bullet_hell"]);
const DEMO_START_S = 3, DEMO_IDLE_S = 15;
const demo = { key: null, since: 0, lastInput: -Infinity, pending: false };
const userControlling = () => { const c = controls(); return !!(c.move_axis || c.move_axis_y || c.move_axis_z || c.fire || c.alt_fire || c.steer_x >= 0); };
setInterval(() => {
  const game = state.game, now = performance.now() / 1000;
  if (!game || !DEMO_GAMES.has(game.game) || game.input !== "realtime" || demo.pending) return;
  // Opening another game starts its demo clock fresh; a reset of the same
  // game (the user taking over) keeps the user's last input.
  if (game.game !== demo.key) { demo.key = game.game; demo.since = now; demo.lastInput = -Infinity; }
  if (userControlling()) {
    demo.lastInput = now;
    if (game.autopilot) {                                  // the user takes over: a fresh game, theirs
      demo.pending = true;
      act({ type: "autopilot", on: false }).then(() => act({ type: "reset" })).finally(() => { demo.pending = false; });
    }
    return;
  }
  const idle = now - Math.max(demo.since + DEMO_START_S - DEMO_IDLE_S, demo.lastInput);
  if (!game.autopilot && !game.over && idle >= DEMO_IDLE_S) {
    demo.pending = true;
    act({ type: "autopilot", on: true }).finally(() => { demo.pending = false; });
  }
}, 100);

// ---------------------------------------------------------------- picking and input
// Real-time controls: held keys, or a held pointer that steers the player
// toward the finger and fires while down.
const held = new Set();
let steer = null;
function boardPlane(clientX, clientY) {
  const cam = camera(), rect = canvas.getBoundingClientRect();
  const ndc = [((clientX - rect.left) / rect.width) * 2 - 1, 1 - ((clientY - rect.top) / rect.height) * 2];
  const inv = M.invert(cam.viewProj), near = M.apply(inv, [ndc[0], ndc[1], -1]), far = M.apply(inv, [ndc[0], ndc[1], 1]);
  const t = (TILE.height - near[1]) / (far[1] - near[1]);
  return t >= 0 ? { x: near[0] + t * (far[0] - near[0]) + viewX(performance.now() / 1000), z: near[2] + t * (far[2] - near[2]) + viewY(performance.now() / 1000) } : null;
}
const twoAxis = () => !!(state.game && state.game.predict && (state.game.predict.y || state.game.predict.yaw));
const climbAxis = () => !!(state.game && state.game.controls && state.game.controls.includes("move_axis_y") && !twoAxis());
function controls() {
  // Keys give an axis; a held pointer gives the board point it asks the ship
  // to reach (on a two-axis board, a little above the finger so the finger
  // never covers it).  The law integrates the ship toward either with
  // damping; the page never closes a loop over a stale position.
  const axis = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
  const axisY = (held.has("down") ? 1 : 0) - (held.has("up") ? 1 : 0);
  const out = { move_axis: axis, steer_x: steer ? Math.max(0, steer.x) : -1, fire: steer || held.has("fire") ? 1 : 0 };
  if (twoAxis()) { out.move_axis_y = axisY; out.steer_y = steer ? Math.max(0, steer.z - 1.4) : -1; }
  // A game with no ship to predict that still reads the pointer on both axes (a tile board: the cell
  // under the finger) declares ``steer_y``; it gets the board point itself, not lifted above the finger.
  else if (state.game && state.game.controls && state.game.controls.includes("steer_y")) out.steer_y = steer ? Math.max(0, steer.z) : -1;
  // A climbing game (a ladder) takes the vertical stick without predicting that axis: the server's alone.
  if (climbAxis()) out.move_axis_y = axisY;
  if (heightAxis()) out.move_axis_z = (held.has("rise") ? 1 : 0) - (held.has("dive") ? 1 : 0);
  if (bombs()) { out.alt_fire = held.has("alt") ? 1 : 0; out.bomb_style = state.bombStyle || 0; }
  if (banking()) {
    out.move_axis_r = (held.has("rud_right") ? 1 : 0) - (held.has("rud_left") ? 1 : 0);
    out.fire_mode = state.fireMode ?? 1; out.primary_slot = state.primarySlot || 0;
  }
  // First person: the mouse's height on the screen pitches the nose; pitch
  // climbs or dives (the height axis), stopping at the floor and ceiling.
  if (noseCam() && heightAxis() && Math.abs(NOSE_PITCH.value) > NOSE_PITCH.dead) {
    out.move_axis_z = Math.max(-1, Math.min(1, out.move_axis_z + NOSE_PITCH.value));
  }
  // THE THUMBPADS (analog, mode-2 style): left = turn (sideways) and
  // throttle (forward = faster); right = climb (forward = up), its sideways
  // adding to the turn, and holding it fires; a double tap on the left
  // drops a bomb.  They add to the keys and replace a steering finger.
  if (pads.left.active || pads.right.active || pads.altUntil > performance.now()) {
    const clamp = (v) => Math.max(-1, Math.min(1, v));
    out.move_axis = clamp(out.move_axis + (banking() ? 0 : pads.left.x) + pads.right.x);
    if (banking()) out.move_axis_r = clamp((out.move_axis_r || 0) + pads.left.x);
    out.steer_x = -1;
    if ("move_axis_y" in out) { out.move_axis_y = clamp(out.move_axis_y + pads.left.y); out.steer_y = -1; }
    if ("move_axis_z" in out) out.move_axis_z = clamp(out.move_axis_z - pads.right.y);
    if (pads.right.active) out.fire = 1;
    if (bombs() && pads.altUntil > performance.now()) out.alt_fire = 1;
  }
  return out;
}
// A game with a third axis (height above the board): E / PageUp rise,
// Q / PageDown dive, or the Rise / Dive buttons (held).
function heightAxis() { return !!(state.game && state.game.predict && state.game.predict.z); }
// THE THUMBPADS: two on-screen sticks for a real-time game (shown on touch
// screens, or with the Thumbpads control).  Each tracks one pointer: its
// offset from the pad's centre over the pad's radius is the stick (-1..1).
const PAD_DEAD = { turn: 0.22, forward: 0.1 };
const pads = { left: { x: 0, y: 0, active: false, id: null, taps: [] }, right: { x: 0, y: 0, active: false, id: null, taps: [] },
               altUntil: 0, built: false };
function buildPads() {
  if (pads.built) return;
  pads.built = true;
  const make = (side) => {
    const pad = document.createElement("div"), knob = document.createElement("div");
    pad.className = `thumbpad thumbpad-${side}`; knob.className = "thumbpad-knob";
    pad.append(knob); document.body.append(pad);
    const s = pads[side];
    const move = (e) => {
      const r = pad.getBoundingClientRect(), radius = r.width / 2;
      let dx = (e.clientX - (r.left + radius)) / radius, dy = (e.clientY - (r.top + radius)) / radius;
      const len = Math.hypot(dx, dy); if (len > 1) { dx /= len; dy /= len; }
      // A dead zone per axis, rescaled so the stick still reaches 1 at the rim:
      // a thumb resting near the centre turns (sideways) or throttles nothing.
      const deadband = (v, d) => Math.abs(v) <= d ? 0 : Math.sign(v) * (Math.abs(v) - d) / (1 - d);
      s.x = deadband(dx, PAD_DEAD.turn); s.y = deadband(dy, PAD_DEAD.forward);
      knob.style.transform = `translate(${dx * radius * 0.6}px, ${dy * radius * 0.6}px)`;
    };
    pad.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      try { pad.setPointerCapture(e.pointerId); } catch (_) { /* a pointer the browser is not tracking */ }
      s.id = e.pointerId; s.active = true; move(e);
      const now = performance.now();
      s.taps = [...s.taps.filter((t0) => now - t0 < 320), now];
      if (side === "left" && s.taps.length >= 2) { pads.altUntil = now + 260; s.taps = []; }   // double tap: a bomb
    });
    pad.addEventListener("pointermove", (e) => { if (s.active && e.pointerId === s.id) { e.preventDefault(); move(e); } });
    const end = (e) => { if (e.pointerId !== s.id) return; s.active = false; s.id = null; s.x = s.y = 0; knob.style.transform = ""; };
    pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
  };
  make("left"); make("right");
  const style = document.createElement("style");
  style.textContent = `.thumbpad{position:fixed;bottom:22px;width:132px;height:132px;border-radius:50%;
    background:radial-gradient(circle,rgba(255,255,255,.10),rgba(255,255,255,.04));border:1px solid rgba(255,255,255,.28);
    touch-action:none;z-index:20;display:none}
    .thumbpad-left{left:22px}.thumbpad-right{right:22px}
    .thumbpad-knob{position:absolute;left:36px;top:36px;width:60px;height:60px;border-radius:50%;
    background:rgba(255,255,255,.32);border:1px solid rgba(255,255,255,.5);pointer-events:none}
    body.pads-on .thumbpad{display:block}`;
  document.head.append(style);
}
function showPads(game) {
  buildPads();
  const wanted = $("pads") ? $("pads").checked : matchMedia("(pointer: coarse)").matches;
  document.body.classList.toggle("pads-on", !!(wanted && game && game.input === "realtime" && game.predict));
}
// ALTERNATE FIRE (a game whose controls take it): B or X drops a bomb (a
// second press bursts a hanging parachute bomb), C cycles the style.
const BOMB_STYLES = ["dumb bomb", "parachute bomb", "ground seeker"];
const FIRE_MODES = ["continuous", "auto", "burst", "single"];
function cycleFireMode() { state.fireMode = ((state.fireMode ?? 1) + 1) % FIRE_MODES.length; renderHotbars(state.game); }
// THE HOTBARS: primary weapons (1 2 3) and secondary bombs (4 5 6), each with
// its own upgrades; the active ones lit; the fire mode beside them (T).
function renderHotbars(game) {
  const box = $("hotbars"), a = game && game.arsenal;
  if (!box) return;
  if (!a) { if (box.childElementCount) box.replaceChildren(); return; }
  const key = JSON.stringify([a, state.primarySlot, state.bombStyle, state.fireMode]);
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  const slot = (label, title, active, enabled, pick) => {
    const b = document.createElement("button");
    b.textContent = label; b.title = title; b.disabled = !enabled;
    if (active) b.style.outline = "2px solid #ffd76a";
    b.onclick = pick;
    return b;
  };
  const prim = a.primary.map((w, i) => slot(`${i + 1} ${w.name}${w.owned ? ` x${w.mult}${w.fan ? ` f${w.fan}` : ""}` : ""}`,
    w.owned ? `${w.name}: x${w.mult} fan ${w.fan}` : `${w.name}: pick one up`, (state.primarySlot || 0) === i, w.owned,
    () => { state.primarySlot = i; renderHotbars(state.game); }));
  const sec = a.secondary.map((w, i) => slot(`${i + 4} ${w.name} x${w.count}`, `${w.name}: ${w.count} a drop`, (state.bombStyle || 0) === i, true,
    () => { state.bombStyle = i; $("bomb-style").textContent = `Style: ${BOMB_STYLES[i]}`; renderHotbars(state.game); }));
  const mode = slot(`Fire: ${FIRE_MODES[state.fireMode ?? 1]} (T)`, "continuous / auto / burst / single", false, true, cycleFireMode);
  const last = document.createElement("span");
  last.textContent = ` upgrades go to: ${a.last}`;
  box.replaceChildren(...prim, " ", ...sec, " ", mode, last);
}
function bombs() { const g = state.game; return !!(g && g.controls && g.controls.includes && g.controls.includes("alt_fire")); }
function cycleBomb() { state.bombStyle = ((state.bombStyle || 0) + 1) % BOMB_STYLES.length; $("bomb-style").textContent = `Style: ${BOMB_STYLES[state.bombStyle]}`; if (inventoryView.applied) saveSettings(); }
const banking = () => !!(state.game && state.game.predict && state.game.predict.bank);
function keyControl(key) {
  const k = key.length === 1 ? key.toLowerCase() : key;
  // Two-stick flying: arrows roll and pitch, A/D rudder, W/S throttle.
  if (banking()) {
    const fly = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "rise", ArrowDown: "dive", a: "rud_left", d: "rud_right",
                  w: "up", s: "down", e: "rise", q: "dive", PageUp: "rise", PageDown: "dive", " ": "fire" }[k];
    if (fly) return fly;
  }
  const common = { ArrowLeft: "left", a: "left", ArrowRight: "right", d: "right", " ": "fire" };
  if (common[k]) return common[k];
  if (heightAxis()) { const lift = { e: "rise", PageUp: "rise", q: "dive", PageDown: "dive" }[k]; if (lift) return lift; }
  if (bombs() && (k === "b" || k === "x")) return "alt";
  if (twoAxis() || climbAxis()) return { ArrowUp: "up", w: "up", ArrowDown: "down", s: "down" }[k];
  return { ArrowUp: "fire", w: "fire" }[k];
}
window.addEventListener("keyup", (e) => { const k = keyControl(e.key); if (k) held.delete(k); });
window.addEventListener("blur", () => held.clear());
function hit(clientX, clientY) {
  // the pointer's hit on the plane of the button tops: a board cell, a rack cell, or nothing
  const game = state.game; if (!game) return null;
  const cam = camera(), rect = canvas.getBoundingClientRect();
  const ndc = [((clientX - rect.left) / rect.width) * 2 - 1, 1 - ((clientY - rect.top) / rect.height) * 2];
  const inv = M.invert(cam.viewProj);
  const near = M.apply(inv, [ndc[0], ndc[1], -1]), far = M.apply(inv, [ndc[0], ndc[1], 1]);
  const t = (TILE.height - near[1]) / (far[1] - near[1]); if (!(t >= 0)) return null;
  const x = near[0] + t * (far[0] - near[0]), z = near[2] + t * (far[2] - near[2]);
  for (const piece of game.pieces || []) {
    if (!piece.power) continue;
    const [px, pz] = cellPosition(game, piece.cell);
    if (Math.hypot(x - px, z - pz) < 0.6) return { kind: "power", power: piece.power, key: `power:${piece.power}` };
  }
  const c = Math.floor(x), r = Math.floor(z);
  if (c >= 0 && c < game.cols && r >= 0 && r < game.rows) return { kind: "board", r, c, key: `${r},${c}` };
  const rack = rackLayout(game);
  if (rack) { const rc = Math.floor((x - rack.origin[0]) / rack.size), rr = Math.floor((z - rack.origin[1]) / rack.size);
    if (rc >= 0 && rc < rack.cols && rr >= 0 && rr < rack.rows) return { kind: "rack", r: rr, c: rc, key: `rack:${rr},${rc}` }; }
  return null;
}
let press = null, clickTimer = null, lastClick = null, tap = null;
canvas.addEventListener("pointerdown", (e) => {
  if (state.game && state.game.input === "realtime") {
    if (isWalker()) { tap = { x: e.clientX, y: e.clientY, t: performance.now() }; return; }       // a walker is not steered by a held finger
    steer = boardPlane(e.clientX, e.clientY); canvas.setPointerCapture(e.pointerId); return;
  }
  press = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button, painted: new Set(), moved: false };
  const h = hit(e.clientX, e.clientY); if (h) press.start = h;
});
canvas.addEventListener("pointermove", async (e) => {
  const game = state.game; if (!game) return;
  if (light.follow) {
    const cam = camera(), rect = canvas.getBoundingClientRect();
    const ndc = [((e.clientX - rect.left) / rect.width) * 2 - 1, 1 - ((e.clientY - rect.top) / rect.height) * 2];
    const inv = M.invert(cam.viewProj), near = M.apply(inv, [ndc[0], ndc[1], -1]), far = M.apply(inv, [ndc[0], ndc[1], 1]);
    const yPlane = light.height * Math.max(state.extent.width, state.extent.depth) * 0.9, t = (yPlane - near[1]) / (far[1] - near[1]);
    if (t >= 0) light.pointer = [near[0] + t * (far[0] - near[0]), near[2] + t * (far[2] - near[2])];   // the board point the lamp flies above
    return;
  }
  if (game.input === "realtime") { if (steer) steer = boardPlane(e.clientX, e.clientY) || steer; return; }
  if (!press) return;
  if (Math.hypot(e.clientX - press.x, e.clientY - press.y) > 8) press.moved = true;
  // meowdoku: a swipe paints X marks on every cell it crosses
  if (game.input === "meowdoku" && press.moved && press.button === 0) {
    // Walk the pointer's path in small steps so a fast swipe still paints
    // every cell it crossed, not only the sampled ones.
    const from = press.last || [press.x, press.y], to = [e.clientX, e.clientY];
    const steps = Math.max(1, Math.ceil(Math.hypot(to[0] - from[0], to[1] - from[1]) / 6));
    for (let k = 0; k <= steps; k++) {
      const h = hit(from[0] + (to[0] - from[0]) * k / steps, from[1] + (to[1] - from[1]) * k / steps);
      if (h && h.kind === "board" && !press.painted.has(h.key)) { press.painted.add(h.key); act({ type: "cell", r: h.r, c: h.c, button: "paint" }); }
    }
    press.last = to;
  }
});
canvas.addEventListener("pointercancel", () => { steer = null; tap = null; });
canvas.addEventListener("pointerup", async (e) => {
  steer = null;
  if (tap) {                                  // a walker: a short tap in place walks there (or builds)
    const t = tap; tap = null;
    if (isWalker() && Math.hypot(e.clientX - t.x, e.clientY - t.y) < 14 && performance.now() - t.t < 700) walkTap(e);
    return;
  }
  const game = state.game; const p = press; press = null;
  if (!p || !game || light.follow) return;
  const h = hit(e.clientX, e.clientY);
  if (game.input === "meowdoku") {
    if (p.moved) return;                                                    // the swipe already painted
    if (h && h.kind === "power") { await act({ type: "power", kind: h.power }); return; }   // a tray item: use it
    if (!h || h.kind !== "board") return;
    const now = performance.now();
    const long = p.button === 2 || now - p.t > 450;
    if (state.tool === "cat" && !long) {                                    // Cats tool: a tap is a cat, at once
      if (lastClick && lastClick.key === h.key && now - lastClick.t < 320) return;   // the second tap of a double tap
      lastClick = { key: h.key, t: now };
      await act({ type: "cell", r: h.r, c: h.c, button: "cat" }); return;
    }
    if (long) { await act({ type: "cell", r: h.r, c: h.c, button: "mouse" }); return; }   // right click / long press
    if (lastClick && lastClick.key === h.key && now - lastClick.t < 320) {   // Mice tool: double tap is a cat
      clearTimeout(clickTimer); clickTimer = null; lastClick = null;
      await act({ type: "cell", r: h.r, c: h.c, button: "double" }); return;
    }
    lastClick = { key: h.key, t: now };
    clearTimeout(clickTimer);
    clickTimer = setTimeout(() => { clickTimer = null; act({ type: "cell", r: h.r, c: h.c, button: "mouse" }); }, 330);
    return;
  }
  if (game.input === "watch") return;
  if (game.input === "select") {
    // tap: select or move; swipe from a piece to a square: select then move
    if (p.moved && p.start && p.start.kind === "board" && h && h.kind === "board" && h.key !== p.start.key) {
      await act({ type: "cell", r: p.start.r, c: p.start.c, button: "primary" });
      optimisticMove(state.game, h);
      await act({ type: "cell", r: h.r, c: h.c, button: "primary" }); return;
    }
    if (h && h.kind === "board") { optimisticMove(game, h); await act({ type: "cell", r: h.r, c: h.c, button: "primary" }); }
    return;
  }
  if (game.input === "text" && h && h.kind === "rack") {
    const letter = (game.rack.letters[h.r] || "")[h.c];
    if (letter) await act({ type: "text", text: game.text + letter });
  }
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// ---------------------------------------------------------------- scene-camera gestures
// One generic layer for every game.  A game declares in its state which input
// belongs to the camera:
//   camera_gestures: { orbit: "one" | "two" | null, zoom: true, pan: true, tilt: true,
//                      target: [x, y, z] | null, limits: { zoom: [lo, hi], elevation: [lo, hi], pan: radius } }
// absent/null = no camera gestures (the game keeps all its input).  Gestures
// move a view offset (camView) composed on top of the game's camera mode.
//   one finger drag (orbit "one", past 8 px)  yaw/pitch around the target
//   two fingers: pinch = zoom, centre drag = pan (orbit "two": orbit), twist = tilt
//   double tap (orbit "one") or two-finger tap = reset the view
//   mouse: wheel = zoom, right/middle drag = orbit, shift+drag = pan, left drag = orbit when orbit "one"
// A game that declares nothing may be given a default here (below).
const CAMERA_GESTURE_DEFAULTS = { bullet_hell: { orbit: null, zoom: true, pan: true, tilt: true } };
function cameraGestures() {
  const g = state.game; if (!g || (state.camMode === "nose" && !state.flat)) return null;
  const d = g.camera_gestures === undefined ? CAMERA_GESTURE_DEFAULTS[g.game] : g.camera_gestures;
  if (!d) return null;
  const ext = state.extent || { width: 6, depth: 6 }, lim = d.limits || {};
  return { orbit: d.orbit || null, zoom: d.zoom !== false, pan: d.pan !== false, tilt: d.tilt !== false,
           target: Array.isArray(d.target) ? d.target : null,
           zoomLim: (lim.zoom || [0.3, 3]).map(Math.log), elLim: lim.elevation || [0.1, Math.PI / 2],
           panLim: lim.pan != null ? lim.pan : Math.max(ext.width, ext.depth) };
}
const CV_KEYS = ["yaw", "pitch", "lz", "px", "pz"];
function resetCameraView() { for (const k of CV_KEYS) { camView.goal[k] = 0; camView.vel[k] = 0; } }
function clampCamGoal(gest) {
  const G = camView.goal, base = state.tiltDeg * Math.PI / 180;
  G.lz = Math.max(gest.zoomLim[0], Math.min(gest.zoomLim[1], G.lz));
  G.pitch = Math.max(gest.elLim[0] - base, Math.min(gest.elLim[1] - base, G.pitch));
  const r = Math.hypot(G.px, G.pz); if (r > gest.panLim) { G.px *= gest.panLim / r; G.pz *= gest.panLim / r; }
}
function stepCamView(dt) {
  const g = state.game, key = g ? g.game : null;
  if (key !== camView.key) { camView.key = key; for (const k of CV_KEYS) camView.cur[k] = camView.goal[k] = camView.vel[k] = 0; }
  dt = Math.max(0, Math.min(0.1, dt || 0));
  const gest = cameraGestures(), G = camView.goal, C = camView.cur;
  if (gest && !camView.active) {                                        // inertia: a released flick coasts and damps
    const decay = Math.exp(-dt * 4);
    for (const k of CV_KEYS) { G[k] += camView.vel[k] * dt; camView.vel[k] *= decay; if (Math.abs(camView.vel[k]) < 1e-4) camView.vel[k] = 0; }
    clampCamGoal(gest);
  }
  const ease = 1 - Math.exp(-dt * 16);
  for (const k of CV_KEYS) C[k] += (G[k] - C[k]) * ease;
}
const gpointers = new Map();           // pointerId -> { x, y, sx, sy, t, type }
let gsession = null;                   // { mode: "pending" | "orbit" | "pan" | "two" | "dead", ... }
let gLastTap = null;
function camPush(k, delta, dt) {       // move a goal and keep a smoothed velocity for the flick
  camView.goal[k] += delta;
  const v = delta / Math.max(dt, 0.008); camView.vel[k] = camView.vel[k] * 0.5 + v * 0.5;
}
function camOrbit(dx, dy, gest, dt) {
  camPush("yaw", -dx * 0.006, dt);
  if (gest.tilt) camPush("pitch", dy * 0.006, dt);
}
function camPan(dx, dy, dt) {
  const rect = canvas.getBoundingClientRect(), s = 2 * camView.lastR * Math.tan(0.31) / rect.height, az = camView.cur.yaw;
  camPush("px", (-Math.cos(az) * dx - Math.sin(az) * dy) * s, dt);
  camPush("pz", (Math.sin(az) * dx - Math.cos(az) * dy) * s, dt);
}
function claimCamera() { steer = null; tap = null; press = null; }   // a camera gesture cancels the game's pending tap/steer
function twoPoints() { const [a, b] = [...gpointers.values()]; return { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) || 1, ang: Math.atan2(b.y - a.y, b.x - a.x) }; }
canvas.addEventListener("pointerdown", (e) => {
  const gest = cameraGestures(); if (!gest) return;
  const now = performance.now();
  gpointers.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, t: now, type: e.pointerType });
  camView.active = gpointers.size;
  for (const k of CV_KEYS) camView.vel[k] = 0;
  if (e.pointerType === "mouse") {
    const mode = e.button === 1 || e.button === 2 ? "orbit" : (e.shiftKey && e.button === 0 ? "pan" : null);
    if (mode) { e.preventDefault(); e.stopImmediatePropagation(); claimCamera(); try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* untracked */ } }
    gsession = { mode: mode || "pending", t: now, last: now };
    return;
  }
  if (gpointers.size === 1) { gsession = { mode: "pending", t: now, last: now }; return; }      // the game sees the first finger
  e.stopImmediatePropagation(); claimCamera();
  if (gpointers.size === 2) { const p = twoPoints(); gsession = { mode: "two", t: now, last: now, moved: false, cx: p.cx, cy: p.cy, d: p.d, ang: p.ang }; }
  else if (gsession) gsession.mode = "dead";
}, true);
canvas.addEventListener("pointermove", (e) => {
  const P = gpointers.get(e.pointerId); if (!P || !gsession) return;
  const gest = cameraGestures(); if (!gest) return;
  const now = performance.now(), dt = (now - gsession.last) / 1000, dx = e.clientX - P.x, dy = e.clientY - P.y;
  P.x = e.clientX; P.y = e.clientY;
  const S = gsession;
  if (S.mode === "pending") {
    if (!(gest.orbit === "one" && Math.hypot(P.x - P.sx, P.y - P.sy) > 8)) return;   // under the threshold: still a tap for the game
    S.mode = "orbit"; claimCamera(); try { canvas.setPointerCapture(e.pointerId); } catch (_) { /* untracked */ }
    camOrbit(P.x - P.sx, P.y - P.sy, gest, 0.016);
  } else if (S.mode === "orbit") camOrbit(dx, dy, gest, dt);
  else if (S.mode === "pan") { if (gest.pan) camPan(dx, dy, dt); }
  else if (S.mode === "two" && gpointers.size === 2) {
    const p = twoPoints();
    if (Math.hypot(p.cx - S.cx, p.cy - S.cy) > 10 || Math.abs(Math.log(p.d / S.d)) > 0.08) S.moved = true;
    if (gest.zoom) camPush("lz", Math.log(S.d / p.d), dt);
    const cdx = p.cx - S.cx, cdy = p.cy - S.cy;
    if (gest.orbit === "two") camOrbit(cdx, cdy, gest, dt); else if (gest.pan) camPan(cdx, cdy, dt);
    let dang = p.ang - S.ang; dang = Math.atan2(Math.sin(dang), Math.cos(dang));
    if (gest.tilt) camPush("pitch", -dang * 0.9, dt);
    S.cx = p.cx; S.cy = p.cy; S.d = p.d; S.ang = p.ang;
  } else { e.stopImmediatePropagation(); return; }
  clampCamGoal(gest); S.last = now; e.preventDefault(); e.stopImmediatePropagation();
}, true);
function endGesturePointer(e) {
  const P = gpointers.get(e.pointerId); if (!P) return;
  const gest = cameraGestures(), now = performance.now(), S = gsession;
  gpointers.delete(e.pointerId);
  camView.active = gpointers.size;
  if (S && now - S.last > 90) for (const k of CV_KEYS) camView.vel[k] = 0;      // held still before lifting: no flick
  if (S && gest && e.type === "pointerup") {
    if (S.mode === "pending" && gest.orbit === "one" && now - P.t < 300 && Math.hypot(P.x - P.sx, P.y - P.sy) < 10) {
      if (gLastTap && now - gLastTap.t < 320 && Math.hypot(P.x - gLastTap.x, P.y - gLastTap.y) < 30) { resetCameraView(); gLastTap = null; }
      else gLastTap = { t: now, x: P.x, y: P.y };
    } else if (S.mode === "two" && !S.moved && now - S.t < 300 && gest.orbit !== "one") resetCameraView();   // two-finger tap
  }
  if (S && S.mode !== "pending") e.stopImmediatePropagation();
  if (S && S.mode === "two" && gpointers.size === 1) S.mode = "dead";           // the lone finger left over does nothing
  if (!gpointers.size) gsession = null;
}
canvas.addEventListener("pointerup", endGesturePointer, true);
canvas.addEventListener("pointercancel", endGesturePointer, true);
canvas.addEventListener("dblclick", (e) => { const gest = cameraGestures(); if (gest && gest.orbit === "one" && e.pointerType !== "touch") resetCameraView(); });
canvas.addEventListener("wheel", (e) => {
  const gest = cameraGestures(); if (!gest || !gest.zoom) return;
  e.preventDefault();
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
  camView.goal.lz += e.deltaY * unit * 0.0012; clampCamGoal(gest);
}, { passive: false });
window.meowdokuDebug.camView = camView;
window.meowdokuDebug.resetCameraView = resetCameraView;
document.addEventListener("visibilitychange", () => { if (!document.hidden && state.game && (state.game.input === "watch" || state.game.input === "realtime")) act({ type: "tick" }); });
// In fullscreen the bar is hidden, and a hidden control takes no key: its
// shortcuts (undo, reset, new, light, flat, tools, hint) are swallowed here,
// in the capture phase, before any other handler sees them.  Fullscreen's own
// keys (F, Escape) and every game's own keys pass; R still restarts a game
// that has ended, as its status line offers.
const BAR_KEYS = new Set(["u", "r", "n", "l", "v"]);
window.addEventListener("keydown", (e) => {
  if (!document.body.classList.contains("immersive") || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof HTMLInputElement && e.target.type === "text") return;
  const game = state.game, key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (!BAR_KEYS.has(key)) return;
  if (game && game.input === "text") return;                       // wordle: letters are the game
  if (game && game.input === "realtime" && keyControl(e.key)) return; // the game's own keys
  if (key === "r" && game && game.over) return;
  e.stopImmediatePropagation();
}, true);
window.addEventListener("keydown", async (e) => {
  const game = state.game; if (!game) return;
  if (e.target === $("entry")) return;
  if (game.input === "realtime" && bombs() && e.key.toLowerCase() === "c") { e.preventDefault(); cycleBomb(); return; }
  if (game.input === "realtime" && banking()) {
    const n = Number(e.key);
    if (n >= 1 && n <= 3) { e.preventDefault(); state.primarySlot = n - 1; renderHotbars(game); return; }
    if (n >= 4 && n <= 6) { e.preventDefault(); state.bombStyle = n - 4; $("bomb-style").textContent = `Style: ${BOMB_STYLES[n - 4]}`; renderHotbars(game); return; }
    if (e.key.toLowerCase() === "t") { e.preventDefault(); cycleFireMode(); return; }
  }
  if (game.input === "realtime" && game.roomview && e.key.toLowerCase() === "m") { e.preventDefault(); act({ type: "delve_map" }); return; }
  if (game.input === "realtime" && game.predict && e.key.toLowerCase() === "n") { e.preventDefault(); toggleCamera(); return; }
  if (game.input === "realtime" && keyControl(e.key)) { e.preventDefault(); held.add(keyControl(e.key)); return; }
  if (game.input === "text") {
    if (/^[a-zA-Z]$/.test(e.key)) return act({ type: "text", text: game.text + e.key.toUpperCase() });
    if (e.key === "Backspace") return act({ type: "backspace" });
    if (e.key === "Enter") return act({ type: "submit" });
  }
  if (e.key === "u" || e.key === "U") act({ type: "undo" });
  if (e.key === "r" || e.key === "R") act({ type: "reset" });
  if (e.key === "n" || e.key === "N") $("new").click();
  if (e.key === "l" || e.key === "L") toggleLight();
});
$("undo").onclick = () => act({ type: "undo" });
$("reset").onclick = () => act({ type: "reset" });
// Typed chess moves: UCI in the box, Submit (or Enter).  The result line says
// what happened, including why a move was refused.
async function submitMove() {
  const box = $("move-input"), text = box.value.trim();
  if (!text) return;
  $("move-result").textContent = "…";
  try {
    const response = await fetch("/api/action", { method: "POST", headers: { "Content-Type": "application/json" },
                                                  body: JSON.stringify({ type: "move", uci: text }) });
    const data = await response.json();
    if (data.error) { $("move-result").textContent = `refused: ${data.error}`; return; }
    box.value = "";
    $("move-result").textContent = `played ${text}`;
    applyState(data);
  } catch (error) { $("move-result").textContent = `failed: ${error}`; }
}
$("move-submit").onclick = submitMove;
$("move-input").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); submitMove(); } e.stopPropagation(); });
$("bomb-style").onclick = cycleBomb;
$("pads").checked = matchMedia("(pointer: coarse)").matches;
$("pads").onchange = () => { showPads(state.game); saveSettings(); };
const NOSE_PITCH = { value: 0, dead: 0.08 };
canvas.addEventListener("mousemove", (e) => {
  if (state.camMode !== "nose") return;
  const r = canvas.getBoundingClientRect();
  NOSE_PITCH.value = Math.max(-1, Math.min(1, -((e.clientY - r.top) / r.height - 0.5) * 2.2));
});
canvas.addEventListener("mouseleave", () => { NOSE_PITCH.value = 0; });
function toggleCamera() { state.camMode = state.camMode === "nose" ? "board" : "nose"; $("camera-mode").textContent = state.camMode === "nose" ? "Camera: nose (N)" : "Camera: board (N)"; if (inventoryView.applied) saveSettings(); }
$("camera-mode").onclick = toggleCamera;
for (const [id, name] of [["rise", "rise"], ["dive", "dive"], ["bomb", "alt"]]) {
  const button = $(id);
  const on = (e) => { e.preventDefault(); held.add(name); }, off = () => held.delete(name);
  button.addEventListener("pointerdown", on); button.addEventListener("pointerup", off);
  button.addEventListener("pointerleave", off); button.addEventListener("pointercancel", off);
}
// THE LEARNING PANEL: while demo learning is on, a pulsing badge (amber while
// a training pass runs), bars for recent demo runs (green kept for training,
// grey skipped under the running average), a line of training loss, and a
// flash each time a pass completes.
const learnView = { lastTrained: null, flashTimer: null };
function drawLearnPanel(game) {
  const L = game && game.learning, panel = $("learn-panel");
  const on = !!(L && L.demo_learning);
  panel.classList.toggle("show", on);
  if (!on) return;
  panel.classList.toggle("training", !!L.training);
  // Bars: finished demo runs (green kept, grey under the average) and
  // mid-game SPAN flushes (blue: lead-ups to scoring sent to training).
  const history = L.history || [], passes = history.filter((h) => h.kind === "trained");
  const bars = history.filter((h) => h.kind === "run" || h.kind === "spans"), runs = bars.filter((h) => h.kind === "run");
  const spans = bars.filter((h) => h.kind === "spans");
  const W = 220, H = 64, svg = [];
  const top = Math.max(1, ...bars.map((r) => r.score));
  const bw = W / Math.max(12, bars.length);
  bars.forEach((r, i) => {
    const h = Math.max(2, (r.score / top) * (H - 14));
    const fill = r.kind === "spans" ? "#6ab8ff" : r.kept ? "#7cf7a6" : "#5a5a66";
    svg.push(`<rect x="${(i * bw + 1).toFixed(1)}" y="${(H - h).toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}"/>`);
  });
  if (passes.length > 1) {
    const losses = passes.map((p) => Math.log10(Math.max(1e-6, p.loss))), lo = Math.min(...losses), hi = Math.max(...losses);
    const pts = losses.map((v, i) => `${((i / (losses.length - 1)) * (W - 4) + 2).toFixed(1)},${(4 + (hi === lo ? 0.5 : (hi - v) / (hi - lo)) * (H - 22)).toFixed(1)}`);
    svg.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="#ffd76a" stroke-width="1.5"/>`);
  }
  $("learn-chart").innerHTML = svg.join("");
  const last = passes[passes.length - 1];
  const net = L.network;
  const netText = net ? (net.arch === "token" ? `token net · ${net.tokens} bodies · ${net.params.toLocaleString()} weights · ${net.outputs} controls`
                                              : `MLP · ${net.params.toLocaleString()} weights · ${net.outputs} controls`) : "no network yet";
  $("learn-text").textContent = `${netText} · ${spans.length} span flushes · ${runs.filter((r) => r.kept).length} of ${runs.length} runs kept`
    + (last ? ` · last pass loss ${last.loss} on ${last.rows} rows (${last.at})` : " · waiting for a span of action");
  if (last && last.at !== learnView.lastTrained) {
    if (learnView.lastTrained !== null) {
      panel.classList.add("flash"); clearTimeout(learnView.flashTimer);
      learnView.flashTimer = setTimeout(() => panel.classList.remove("flash"), 900);
    }
    learnView.lastTrained = last.at;
  }
}
$("demo-learn").onclick = () => act({ type: "demo_learning", on: !(state.game && state.game.learning && state.game.learning.demo_learning) });
$("arena-save").onclick = () => act({ type: "arena_save" });
$("arena-stop").onclick = () => act({ type: state.game && state.game.arena_running ? "arena_stop" : "arena_resume" });
$("lichess-stop").onclick = () => act({ type: "lichess_pause" });
$("lichess-resume").onclick = () => act({ type: "lichess_resume" });
$("entry").addEventListener("input", (e) => act({ type: "text", text: e.target.value }));
$("entry").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); act({ type: "submit" }); } });
$("enter").onclick = () => act({ type: "submit" });
$("back").onclick = () => act({ type: "backspace" });
$("round").oninput = (e) => { roundParams.iters = Number(e.target.value); rebuildRoundedMeshes(); };
$("crease").oninput = (e) => { roundParams.creaseDeg = Number(e.target.value); rebuildRoundedMeshes(); };
$("tilt").oninput = (e) => { state.tiltDeg = Number(e.target.value); };
$("saturation").oninput = (e) => { state.saturation = Number(e.target.value) / 100; if (state.game) { uploadMaterials(state.game); rebuildBoards(state.game); } };
function toggleLight() { light.follow = !light.follow; $("light").textContent = light.follow ? "Light: pointer (L)" : "Light: sun (L)"; }
$("light").onclick = toggleLight;
function toggleFlat() { state.flat = !state.flat; $("flat").textContent = state.flat ? "Flat: on (V)" : "Flat (V)"; }
$("flat").onclick = toggleFlat;
$("wobble").onchange = (e) => { wob.on = e.target.checked; wob.lastNow = null; };
$("wobble-k").oninput = (e) => { wob.freq = Number(e.target.value); };
$("wobble-gpu").onchange = async (e) => {
  wob.useGpu = e.target.checked; wob.resetAll = true;
  if (wob.useGpu) { await wobInitGpu(); if (!wob.gpu) wob.useGpu = false; }
  syncWobbleUi();
};
syncWobbleUi();
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type === "text") return;
  if ((e.key === "v" || e.key === "V") && !e.ctrlKey && !e.metaKey && !(state.game && state.game.input === "text")) toggleFlat();
});

// ---------------------------------------------------------------- player profile and scores
// The name is a label (no password); the browser remembers it and hands it
// back to the server after a reload, since the server's session is a cookie.
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function rememberedName() { try { return localStorage.getItem("meowdoku_player") || ""; } catch { return ""; } }
async function setPlayer(name) {
  name = name.trim(); if (!name) return;
  try {
    await api("/api/profile", { name });
    try { localStorage.setItem("meowdoku_player", name); } catch {}
    // A ticking game refreshes itself; others read their state once so the
    // status line shows the new player's best.
    if (state.game && state.game.input !== "realtime" && state.game.input !== "watch")
      chain = chain.then(async () => applyState(await (await fetch("/api/state")).json())).catch((e) => console.warn(e));
    if ($("scores").classList.contains("open")) showScores();
  } catch (e) { console.warn(e); }
}
async function showScores() {
  const data = await (await fetch("/api/profile")).json(), panel = $("scores");
  const game = data.game || "this game", mine = data.profile && data.game ? (data.profile.games[data.game] || null) : null;
  const rows = data.leaderboard.map((r, k) => `<tr class="${r.name === data.player ? "me" : ""}"><td>${k + 1}</td><td>${escapeHtml(r.name)}</td><td class="n">${r.score}</td><td class="muted">${escapeHtml(r.detail)}</td></tr>`).join("");
  const recent = mine ? mine.recent.slice(0, 8).map((r) => `<tr><td class="n">${r.score}</td><td class="muted">${escapeHtml(r.detail)} · ${new Date(r.at * 1000).toLocaleString()}</td></tr>`).join("") : "";
  panel.innerHTML = `<h3>Top scores · ${escapeHtml(game)}</h3>` + (rows ? `<table>${rows}</table>` : `<div class="muted">no finished games yet</div>`)
    + `<h3>${data.player ? escapeHtml(data.player) : "You"}${mine ? ` · ${mine.plays} played` : ""}</h3>`
    + (data.player ? (recent ? `<table>${recent}</table>` : `<div class="muted">finish a game to record a score</div>`) : `<div class="muted">type a name in Player to keep scores</div>`);
}
$("player").value = rememberedName();
$("player").addEventListener("change", (e) => setPlayer(e.target.value));
$("player").addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") e.target.blur(); });
// ---------------------------------------------------------------- layout
// A free board's width follows the screen: after a resize or a fullscreen
// switch settles, the server widens or narrows the field mid-game.
let layoutTimer = null;
new ResizeObserver(() => {
  clearTimeout(layoutTimer);
  layoutTimer = setTimeout(() => {
    if (state.game && state.game.board === "free") act({ type: "layout", aspect: Math.min(3.0, Math.max(0.4, canvas.clientWidth / Math.max(1, canvas.clientHeight))) });   // a game declares its world; a strip or a column of screen never reshapes it
  }, 250);
}).observe(canvas);

// ---------------------------------------------------------------- fullscreen
// The canvas takes the whole screen: the bar hides (so the canvas's top
// offset falls to 0) and, where the browser allows it, the page goes
// fullscreen too.  Leaving the browser's fullscreen (Esc) leaves this mode.
function setImmersive(on) {
  document.body.classList.toggle("immersive", on);
  document.documentElement.style.setProperty("--bar-h", on ? "0px" : $("bar").offsetHeight + "px");
  const root = document.documentElement;
  if (on && !document.fullscreenElement && root.requestFullscreen) root.requestFullscreen().catch(() => {});
  if (!on && document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
}
const toggleImmersive = () => setImmersive(!document.body.classList.contains("immersive"));
$("fullscreen").onclick = toggleImmersive;
$("fs-exit").onclick = () => setImmersive(false);
document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement && document.body.classList.contains("immersive")) setImmersive(false); });
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement && e.target.type === "text") return;
  if (e.key === "f" || e.key === "F") toggleImmersive();
  else if (e.key === "Escape" && document.body.classList.contains("immersive")) setImmersive(false);
});
$("scores-toggle").onclick = () => { const panel = $("scores"); panel.classList.toggle("open"); if (panel.classList.contains("open")) showScores(); };
$("shop-toggle").onclick = () => { const panel = $("shop"); panel.classList.toggle("open"); if (panel.classList.contains("open")) showShop(); };

// ---------------------------------------------------------------- meowdoku tools, hint, palette
state.tool = "mouse";             // a tap marks a mouse, a double tap is a cat (as before the tools)
function setTool(tool) {
  state.tool = tool;
  $("tool-cat").classList.toggle("active", tool === "cat");
  $("tool-mouse").classList.toggle("active", tool === "mouse");
}
$("tool-cat").onclick = () => setTool("cat");
$("tool-mouse").onclick = () => setTool("mouse");
// Power-ups are objects in the tray beside the board (tap one to use it):
// Cat reveals a cat, Mouse drops three mice on cells that hold no cat, the
// Hint bulb lights the next deduction; each shows how many are left.
$("palette").onchange = (e) => act({ type: "palette", name: e.target.value });
function syncMeowdokuBar(game) {
  if (game.game !== "meowdoku") return;
  const select = $("palette");
  if (select.options.length !== game.palettes.length) select.innerHTML = game.palettes.map((n) => `<option value="${n}">${n}</option>`).join("");
  select.value = game.palette_name;
}

// ---------------------------------------------------------------- game picker
function fillPickers() {
  const games = state.catalog.games, gameSelect = $("game"), levelSelect = $("level"), boardSelect = $("board");
  gameSelect.innerHTML = games.map((g) => `<option value="${g}">${g}</option>`).join("");
  const levels = Object.keys(state.catalog.meowdoku.levels);
  levelSelect.innerHTML = levels.map((k) => `<option value="${k}">${k} (${state.catalog.meowdoku.levels[k]})</option>`).join("");
  const refill = () => {
    const chosen = state.catalog.meowdoku.boards.filter((b) => b.level === levelSelect.value);
    boardSelect.innerHTML = chosen.map((b) => `<option value="${b.id}">${b.id} · ${b.type}</option>`).join("");
  };
  levelSelect.onchange = refill;
  levelSelect.value = levels.find((k) => k.startsWith("small/forced")) || levels[0]; refill();
  gameSelect.onchange = () => { document.body.dataset.game = gameSelect.value; $("new").click(); };
  boardSelect.onchange = () => newGame({ game: "meowdoku", id: boardSelect.value });
  $("new").onclick = async () => {
    const g = gameSelect.value;
    if (g === "meowdoku") {
      const chosen = state.catalog.meowdoku.boards.filter((b) => b.level === levelSelect.value);
      const board = chosen[Math.floor(Math.random() * chosen.length)]; boardSelect.value = board.id;
      newGame({ game: "meowdoku", id: board.id });
    } else if (g === "wordle") newGame({ game: "wordle", seed: Math.floor(Math.random() * 100000), ...wordleChoice });
    else newGame({ game: g, aspect: canvas.clientWidth / Math.max(1, canvas.clientHeight) });
  };
}

// ---------------------------------------------------------------- boot
(async () => {
  const [vert, frag, assets, catalog] = await Promise.all([
    fetch("shaders/base_material.webgl.vert.glsl").then((r) => r.text()),
    fetch("shaders/base_material.webgl.shadow.frag.glsl").then((r) => r.text()),
    fetch("assets.json").then((r) => r.json()),
    fetch("/api/catalog").then((r) => r.json()),
  ]);
  state.assets = assets; state.catalog = catalog;
  state.programs.phong = link(compile(gl.VERTEX_SHADER, vert), compile(gl.FRAGMENT_SHADER, frag));
  state.programs.flat = link(compile(gl.VERTEX_SHADER, FLAT_VS), compile(gl.FRAGMENT_SHADER, FLAT_FS));
  state.programs.cel = link(compile(gl.VERTEX_SHADER, CEL_VS), compile(gl.FRAGMENT_SHADER, CEL_FS));
  state.programs.depth = link(compile(gl.VERTEX_SHADER, DEPTH_VS), compile(gl.FRAGMENT_SHADER, DEPTH_FS));
  state.shadowMap = buildShadowMap(); state.bulb = buildSphere(); state.meshes.__sphere = uploadMesh(sphereMesh());
  arrayTexture(4, [0, 255, 128, 255]); terrainTexture(5, assets.terrain);
  // The Phong shader's depth stack (uDepthUv, unit 6): layer 0 is the terrain's
  // height map (exported in the shader's channel layout); no export, the 1x1 default.
  if (assets.terrain && assets.terrain.depth) depthTexture(6, assets.terrain.depth); else arrayTexture(6, [0, 0, 0, 255]);
  arrayTexture(7, [0, 0, 0, 0]);
  volumeTexture(8); storageTexture({ data: [0.0] }, 10);
  state.units = { turing_ssbo_pbr: 0, turing_ssbo_phong: 1, turing_ssbo_enamel: 2, turing_ssbo_texstack: 3,
                  uEmitUv: 4, uColorUv: 5, uDepthUv: 6, uRemitUv: 7, uFieldVolume: 8, uShadowMaps: 9, turing_ssbo_thermal_k: 10 };
  for (const [name, mesh] of Object.entries(assets.meshes)) state.meshes[name] = uploadMesh(roundMesh(mesh, roundParams.iters, roundParams.creaseDeg));
  // (the exported "mouse" mesh is not drawn: X marks are notes, not mice)
  try {
    pred.program = await TuringWasmProgram.load("motion.wasm", "motion.abi.json");
    // the engine's other laws the page runs (entity_motion.PREDICTORS)
    pred.travel = await TuringWasmProgram.load("travel.wasm", "travel.abi.json");
    pred.follow = await TuringWasmProgram.load("follow.wasm", "follow.abi.json");
    pred.bankTurn = await TuringWasmProgram.load("bank.wasm", "bank.abi.json");
    // the sphere: flight (turn, air, advance) and the land's law
    pred.turn = await TuringWasmProgram.load("turn.wasm", "turn.abi.json");
    pred.advance = await TuringWasmProgram.load("advance.wasm", "advance.abi.json");
    pred.air = await TuringWasmProgram.load("air.wasm", "air.abi.json");
    // the player's shots: birth, climb and the patch projection (entity_motion.shot_spawn / climb / sphere_project)
    pred.shot = await TuringWasmProgram.load("shot.wasm", "shot.abi.json");
    pred.climb = await TuringWasmProgram.load("climb.wasm", "climb.abi.json");
    pred.project = await TuringWasmProgram.load("project.wasm", "project.abi.json");
    // the trigger's gate (when a volley may leave), a body's heading in the patch, and the
    // trajectory step missiles and bombs fly on: the server's laws, the same text
    pred.gate = await TuringWasmProgram.load("gate.wasm", "gate.abi.json");
    pred.heading = await TuringWasmProgram.load("heading.wasm", "heading.abi.json");
    pred.ballistic = await TuringWasmProgram.load("ballistic.wasm", "ballistic.abi.json");
    // the server's bodies, carried on between snapshots by those laws (bodies.js)
    if (BODY_CORES.every((stem) => pred[stem])) pred.bodies = new BodyFlight(Object.fromEntries(BODY_CORES.map((stem) => [stem, pred[stem]])));
    pred.terrain = await TuringWasmProgram.load("terrain.wasm", "terrain.abi.json");
    // the walker's laws (World Walk): the surface speed, the pace, the turn toward a waypoint
    pred.walkSpeed = await TuringWasmProgram.load("walkspeed.wasm", "walkspeed.abi.json");
    pred.walkDrive = await TuringWasmProgram.load("walkdrive.wasm", "walkdrive.abi.json");
    pred.walkRoute = await TuringWasmProgram.load("walkroute.wasm", "walkroute.abi.json");
  } catch (error) { console.warn("[meow] ship predictor unavailable; playing unpredicted", error); }
  // The autopilot policy network (policy_wasm.py): the MLP forward as a wasm
  // core with the live weights fed as its inputs.  Inspection only -- nothing
  // pilots with it.  window.meowdokuDebug.policy.evaluate(features) -> action[].
  try {
    const program = await TuringWasmProgram.load("policy.wasm", "policy.abi.json");
    const weights = await fetch("policy.weights.json").then((response) => response.json());
    const feeds = Object.fromEntries(Object.entries(weights).map(([name, w]) => [name, w.data]));
    window.meowdokuDebug.policy = {
      program, weights, features: program.contract.features, outputs: program.contract.outputs_width,
      evaluate(features) {
        if (features.length !== this.features) throw new Error(`policy takes ${this.features} features, got ${features.length}`);
        return Array.from(program.invoke({ ...feeds, x: features }).action);
      },
    };
  } catch (error) { console.warn("[meow] autopilot policy unavailable", error); }
  // The traced graph products (graph_wasm.py): the bullet-hell law and the token
  // pilot as buffer-table wasm (TuringGraphProgram).  Inspection only -- the page
  // neither predicts nor pilots with them yet.
  //   meowdokuDebug.lawWasm.run({alive, x, ..., dt}) -> {x_next, ..., dt_limit}
  //   meowdokuDebug.pilotWasm.evaluate(tokenFeatures) -> action[]
  // Loaded in the BACKGROUND: the law module is 875 KB with 61 MB of buffers,
  // and a phone can take long (or fail) to compile and grow it -- awaiting it
  // here held the game pickers empty until it finished.
  (async () => {
  try {
    const law = await TuringGraphProgram.load("law.wasm", "law.abi.json");
    window.meowdokuDebug.lawWasm = {
      program: law,
      run(columns) {
        for (const name of law.inputNames()) { if (name in columns) law.setInput(name, columns[name]); }
        law.run();
        return Object.fromEntries(law.outputNames().map((name) => [name, law.output(name)]));
      },
    };
  } catch (error) { console.warn("[meow] law wasm unavailable", error); }
  try {
    const pilot = await TuringGraphProgram.load("pilot.wasm", "pilot.abi.json");
    const weights = await fetch("pilot.weights.json").then((response) => response.json());
    for (const [name, w] of Object.entries(weights)) pilot.setInput(name, w.data);
    window.meowdokuDebug.pilotWasm = {
      program: pilot, weights, features: pilot.abi.features, outputs: pilot.abi.outputs_width,
      run(inputs) { for (const name in inputs) pilot.setInput(name, inputs[name]); pilot.run(); return pilot.output("action"); },
      evaluate(features) {
        if (features.length !== this.features) throw new Error(`pilot takes ${this.features} features, got ${features.length}`);
        pilot.setInput("x", features); pilot.run();
        return Array.from(pilot.output("action"));
      },
    };
  } catch (error) { console.warn("[meow] pilot wasm unavailable", error); }
  })();
  fillPickers();
  if (rememberedName()) await setPlayer(rememberedName());
  requestAnimationFrame(draw);
  $("new").click();
})().catch((error) => { status.textContent = String(error); console.error(error); });

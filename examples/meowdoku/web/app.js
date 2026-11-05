// The games page: boards of beveled buttons and 3D pieces (glyph extrusions)
// lit by Pluck's base-material Phong shader as compiled for WebGL 2, with a
// depth pre-pass from an orbiting key light for shadows.  The page draws
// whatever state the Python server hands it (games.py's protocol) and
// animates the difference between states; it decides nothing.

import { TuringWasmProgram } from "./turing-wasm-shell.js";

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
const roundParams = { iters: 2, creaseDeg: 40 };
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
  saturation: 1.0, tiltDeg: 90, time: 0,
  debug: { shadows: true, exposure: 0.42 },
});
const TILE = { height: 0.16, inset: 0.11, gap: 0.03 };
const GLASS = { thickness: 0.22, margin: 0.6 };

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
    records.push([name, rgb, glow[name] ? { emission: glow[name] } : {}]);
    records.push([`${name}@hint`, mix(rgb, [0.45, 0.95, 0.45], 0.55), glow["@hint"] ? { emission: glow["@hint"] } : {}]);
    records.push([`${name}@sel`, mix(rgb, [1.0, 0.85, 0.3], 0.6), {}]);
  }
  records.push(["glass", [0.80, 0.90, 1.0], { roughness: 0.04, transmission: 0.85, opacity: 0.28, ior: 1.5 }]);
  // Emitters are surfaces the shader lights from within (pbr emission): the
  // sun's orb and the arcade bulbs, each bulb colour at ARCADE_LEVELS steps.
  records.push(["sun_orb", [0.2, 0.2, 0.2], { emission: [2.2, 2.0, 1.7] }]);
  // The craft's energy shield: a glassy bubble that glows blue, in SHIELD_STEPS
  // charge steps (opacity and glow rise with the charge).
  for (let k = 0; k < SHIELD_STEPS; k++) {
    const charge = k / (SHIELD_STEPS - 1);
    records.push([`shield_${k}`, [0.35, 0.65, 1.0], { roughness: 0.05, transmission: 0.6, ior: 1.2,
      opacity: 0.06 + 0.22 * charge, emission: [0.1 * charge, 0.35 * charge, 0.9 * charge] }]);
  }
  ARCADE.colours.forEach((c, ci) => ARCADE_LEVELS.forEach((level, li) =>
    records.push([`bulb${ci}_${li}`, c.map((v) => 0.25 * v), { emission: c.map((v) => v * level), roughness: 0.2 }])));
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
  disposeMesh(state.buttons); disposeMesh(state.rackButtons); disposeMesh(state.glass);
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
  const depth = rack ? rack.origin[1] + rack.rows * rack.size
    : game.rows + (game.powerups ? TRAY.row + 0.7 : game.hearts != null ? 1.5 : 0);
  const width = game.board === "free" ? game.width : game.cols;
  state.extent = { width, depth: game.board === "free" ? game.height : depth };
  const m = GLASS.margin, t = GLASS.thickness, gm = state.materialIndex.glass;
  state.glass = litMesh(box(-m, -t, -m, width + m, 0.0, state.extent.depth + m, gm));
}

// ---------------------------------------------------------------- camera and light
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
  const target = [ext.width / 2, 0, ext.depth / 2];
  if (state.flat) {
    // Flat view: orthographic, looking straight down; the same framing as the lens below.
    const eye = [target[0], 60, target[2]], view = M.lookAt(eye, target, [0, 0, -1]);
    const hh = halfFit, hw = halfFit * aspect, near = 1, far = 120;
    const proj = new Float32Array([1 / hw,0,0,0, 0,1 / hh,0,0, 0,0,-2 / (far - near),0, 0,0,-(far + near) / (far - near),1]);
    return { view, proj, viewProj: M.multiply(proj, view), eye };
  }
  const dist = halfFit / Math.tan(0.31) * (1 + 0.35 * Math.cos(el));
  const eye = [target[0], target[1] + dist * Math.sin(el), target[2] + dist * Math.cos(el)];
  const up = el > 1.45 ? [0, 0, -1] : [0, 1, 0];
  const view = M.lookAt(eye, target, up);
  const proj = M.perspective(0.62, aspect, 0.1, 300);
  return { view, proj, viewProj: M.multiply(proj, view), eye };
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
function hideSplash() { clearTimeout(splash.timer); $("splash").className = ""; }

// ---------------------------------------------------------------- the craft's shield
// A bubble around the player's ship, as charged as the server says (``you``'s
// shield); a hit it absorbs (the charge falling) flashes it.  Drawn with the
// glass: blended, no depth write.
const SHIELD_STEPS = 6;
const shieldView = { charge: 0, last: null, flashAt: -10 };
function noteShield(game, now) {
  const charge = game && game.you ? Number(game.you.shield || 0) : 0;
  if (shieldView.last != null && charge < shieldView.last - 0.2) shieldView.flashAt = now;
  shieldView.last = charge; shieldView.charge = charge;
}
function drawShield(now, drawLit) {
  const game = state.game;
  if (state.flat || !game || game.input !== "realtime" || !game.you || !game.you.alive) return;
  const ship = [...state.pieces.values()].find((p) => p.entity && p.lane === (game.predict ? game.predict.lane : 0) && p.removing == null);
  if (!ship) return;
  const flash = Math.max(0, 1 - (now - shieldView.flashAt) / 0.4);
  const charge = Math.min(1, shieldView.charge + flash);
  if (charge <= 0.02) return;
  const step = Math.round(charge * (SHIELD_STEPS - 1)), r = ship.size * 1.25 * (1 + 0.15 * flash);
  const model = M.multiply(M.translate(ship.x, TILE.height + ship.size * 0.55, ship.z), M.scale(r, r * 0.7, r));
  drawLit(state.meshes.__sphere.vao, state.meshes.__sphere.count, model, state.materialIndex[`shield_${step}`]);
}

// ---------------------------------------------------------------- arcade lights
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
    bulbs.push([x, 0.45, z]);                      // on posts, so their light falls onto the tiles
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
function lampTarget(now, ext) {
  const span = Math.max(ext.width, ext.depth);
  if (light.follow && light.pointer) return [light.pointer[0], 0.55 * span, light.pointer[1]];
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
// The power-up tray: a row under the lives; each item with its count beside it.
const TRAY = { row: 2.05, spacing: 2.0, count: 0.62 };
function trayPosition(game, cell) {
  const [, k, part] = cell.split(":");
  const x = game.cols / 2 + (Number(k) - 1) * TRAY.spacing - 0.3 + (part === "count" ? TRAY.count : 0);
  return [x, game.rows + TRAY.row, "hud"];
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
const pred = { program: null, seq: 0, sent: [], closed: [], open: null, ship: null, offset: [0, 0], at: null, lane: null };
const PRED_STEP = 1 / 240;            // the predictor's own rate (PREDICT_HZ)
window.meowdokuDebug = { pred, get sock() { return sock; } };   // read-only inspection handle (play tests)
const sameControls = (a, b) => Object.keys(a).every((k) => a[k] === b[k]);
function stepShip(ship, input, seconds, motion) {
  for (let left = seconds; left > 1e-9;) {
    const dt = Math.min(PRED_STEP, left); left -= dt;
    for (const [axis, pos, vel, move, target] of [["x", "x", "vx", "move_axis", "steer_x"], ["y", "y", "vy", "move_axis_y", "steer_y"]]) {
      if (!motion[axis]) continue;
      const r = pred.program.invoke({ x: ship[pos], v: ship[vel], axis: input[move] || 0, target: input[target] ?? -1,
        lo: motion[axis][0], hi: motion[axis][1], speed: motion.speed, tau: motion.tau, reach: motion.reach, dt });
      ship[pos] = r.x_next; ship[vel] = r.v_next;
    }
  }
}
const predicting = (game) => !!(pred.program && game && game.input === "realtime" && game.predict);
function reconcile(game) {
  if (!predicting(game) || !game.you || !game.you.alive) { pred.ship = null; return; }
  pred.lane = game.predict.lane;
  const before = pred.ship ? [pred.ship.x, pred.ship.y] : null;
  if (game.ack != null) pred.sent = pred.sent.filter((interval) => interval.seq > game.ack);
  const ship = { ...game.you };
  for (const interval of [...pred.sent, ...pred.closed, ...(pred.open ? [pred.open] : [])]) stepShip(ship, interval.controls, interval.dur, game.predict);
  if (before) { pred.offset = [pred.offset[0] + before[0] - ship.x, pred.offset[1] + before[1] - ship.y];
                if (Math.hypot(...pred.offset) > 1.5) pred.offset = [0, 0]; }
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
  // The open interval runs while the controls hold; a change closes it.
  const c = controls();
  if (!pred.open || !sameControls(pred.open.controls, c)) {
    if (pred.open && pred.open.dur > 0) pred.closed.push(pred.open);
    pred.open = { controls: c, dur: 0 };
  }
  pred.open.dur += frame;
  if (pred.ship) stepShip(pred.ship, c, frame, game.predict);
  const decay = Math.exp(-frame / 0.06);
  pred.offset = [pred.offset[0] * decay, pred.offset[1] * decay];
}
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
        if (live && /no game/.test(message.error) && !ticks.restarting) {
          ticks.restarting = true;
          newGame({ game: state.game.game, aspect: canvas.clientWidth / Math.max(1, canvas.clientHeight) }).finally(() => { ticks.restarting = false; });
        }
      }
      return;
    }
    sock.outstanding = Math.max(0, sock.outstanding - 1);
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
  let o = 0;
  const u8 = () => view.getUint8(o++), u16 = () => { const v = view.getUint16(o, true); o += 2; return v; };
  const i16 = () => { const v = view.getInt16(o, true); o += 2; return v; }, f32 = () => { const v = view.getFloat32(o, true); o += 4; return v; };
  const i32 = () => { const v = view.getInt32(o, true); o += 4; return v; };
  if (u8() !== 1) return;
  const flags = u8(), ack = i32();
  const you = { x: f32(), y: f32(), vx: f32(), vy: f32(), alive: u8() === 1, shield: f32() };
  for (let n = u8(); n > 0; n--) { const len = u8(); sock.names.push(new TextDecoder().decode(new Uint8Array(view.buffer, view.byteOffset + o, len))); o += len; }
  const frameGame = { ...game, ack: ack < 0 ? null : ack, you };
  state.game.you = you;
  noteShield(frameGame, now);
  const stamp = worldStamp(frameGame, now);
  const drop = (lane) => { const key = sock.lanes.get(lane); const piece = key && state.pieces.get(key); if (piece && piece.removing == null) piece.removing = now; sock.lanes.delete(lane); };
  if (flags & 2) { for (const lane of [...sock.lanes.keys()]) drop(lane);
                   for (const piece of state.pieces.values()) if (piece.entity && piece.removing == null) piece.removing = now; }
  for (let n = u16(); n > 0; n--) drop(u16());
  for (let n = u16(); n > 0; n--) {
    const lane = u16(), mesh = sock.names[u8()], color = sock.names[u8()], size = u16() / 1000, roll = i16() / 1000, spin = i16() / 1000, bob = u8();
    const lean = i16() / 1000, ownerLane = u16(), age = f32();
    const x = f32(), y = f32(), vx = f32(), vy = f32();
    drop(lane);
    const id = `${lane}:s${++sock.gen}`;
    sock.lanes.set(lane, id);
    state.pieces.set(id, { id, entity: true, lane, sx: x, sz: y, x, z: y, vx, vz: vy, tRecv: stamp, mesh, color, size, roll, spin, bob, lean,
                           t0: now, removing: null, seed: 0, birth: birthOffset(ownerLane, age, x, y, vx, vy, you, stamp) });
  }
  for (let n = u16(); n > 0; n--) {
    const lane = u16(), mesh = sock.names[u8()], color = sock.names[u8()], x = f32(), y = f32(), vx = f32(), vy = f32();
    const piece = state.pieces.get(sock.lanes.get(lane));
    if (piece) Object.assign(piece, { sx: x, sz: y, vx, vz: vy, tRecv: stamp, mesh, color });
  }
  reconcile(frameGame);
}

const ticks = { inFlight: 0, sent: 0, applied: -1 };
function sendTick(game) {
  let payload = { type: "tick", ...controls() };
  if (predicting(game)) {
    if (pred.open && pred.open.dur > 0) { pred.closed.push(pred.open); pred.open = { controls: pred.open.controls, dur: 0 }; }
    const at = performance.now() / 1000;
    pred.sent.push(...pred.closed.map((interval) => ({ ...interval, seq: ++pred.seq, closedAt: at })));
    pred.closed = [];
    if (pred.sent.length > 240) pred.sent.splice(0, pred.sent.length - 240);
    payload.inputs = pred.sent.map((interval) => ({ seq: interval.seq, dur: interval.dur, ...interval.controls }));
  }
  if (sock.open) {
    sock.outstanding++;
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
      if (data.error && /no game/.test(data.error) && !ticks.restarting) {
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
setInterval(() => {
  const game = state.game;
  if (!game || game.input !== "realtime" || game.over || document.hidden || ticks.inFlight + sock.outstanding >= 3) return;
  sendTick(game);
}, 30);

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
  const mesh = state.meshes[piece.mesh], lead = Math.min(Math.max(0, now - piece.tRecv), 1.0);
  if (pred.ship && piece.lane === pred.lane && piece.mesh !== "boom" && piece.removing == null) {
    // The player's own ship: the prediction, not the server's last word.
    piece.x = pred.ship.x + pred.offset[0]; piece.z = pred.ship.y + pred.offset[1]; piece.drawnAt = now;
    return placeEntity(piece, now, piece.x, piece.z, mesh);
  }
  const tx = piece.sx + piece.vx * lead, tz = piece.sz + piece.vz * lead;
  // Each frame the drawn position first moves by the velocity (so a shot
  // never trails), then only the remaining error is eased away.
  const frame = piece.drawnAt == null ? 0 : Math.max(0, Math.min(0.1, now - piece.drawnAt));
  const step = piece.drawnAt == null ? 1 : 1 - Math.exp(-frame / 0.045);
  piece.drawnAt = now;
  const px = piece.x + piece.vx * frame, pz = piece.z + piece.vz * frame;
  const far = Math.hypot(tx - px, tz - pz) > 1.5;                    // a respawn or rewind: go there
  const x = far || step >= 1 ? tx : px + (tx - px) * step;
  const z = far || step >= 1 ? tz : pz + (tz - pz) * step;
  piece.x = x; piece.z = z;
  return placeEntity(piece, now, x, z, mesh);
}
function placeEntity(piece, now, x, z, mesh) {
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
  const lean = state.flat ? 0 : piece.lean ?? 0.55, scale0 = piece.size, scale = scale0 / mesh.extent * Math.max(s, 0.001);
  let m = M.translate(x, state.flat ? TILE.height + FLAT_LIFT : TILE.height + scale0 * Math.sin(lean) + 0.04 + y, z);
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
  const base = piece.place === "hud" ? 0.0 : TILE.height * (piece.cellSize || 1);
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
  on: true, useGpu: false, freq: 8, frameNo: 0, lastNow: null, hi: -1,
  D: new Float32Array(WOB_CAP * 24), U: new Float32Array(WOB_CAP * 24), P: new Float32Array(WOB_CAP * 16),
  owner: new Array(WOB_CAP).fill(null), touched: new Int32Array(WOB_CAP), resetFrame: new Int32Array(WOB_CAP),
  free: Array.from({ length: WOB_CAP }, (_, i) => WOB_CAP - 1 - i), gpu: null, gpuState: "off", resetAll: false,
  msStep: 0, msSample: 0, msTotal: 0,
};
state.wobble = wob;

function wobTransition(dt, freq) {                       // exact step of d'' + 2 z w d' + w^2 d = 0
  const w = 2 * Math.PI * freq, z = WOB_ZETA, wd = w * Math.sqrt(1 - z * z), e = Math.exp(-z * w * dt);
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
    const adapter = await navigator.gpu.requestAdapter(); if (!adapter) throw new Error("no adapter");
    const device = await adapter.requestDevice();
    device.lost.then(() => { wob.gpu = null; wob.gpuState = "lost"; wob.useGpu = false; syncWobbleUi(); });
    wob.gpu = makeGpuCage(device, WOB_CAP); wob.gpuState = "ready"; wob.resetAll = true;
  } catch (e) { console.warn("WebGPU wobble unavailable:", e); wob.gpuState = "failed"; wob.useGpu = false; syncWobbleUi(); }
}
// Run the same random inputs through the CPU kernel and a fresh GPU cage; returns the largest difference.
async function wobbleParity(steps = 120) {
  if (!navigator.gpu) return null;
  const device = await (await navigator.gpu.requestAdapter()).requestDevice(), cap = 16, cage = makeGpuCage(device, cap);
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
    pred.sent = []; pred.closed = []; pred.open = null; pred.ship = null; ticks.applied = -1;
  } else if (previous.palette_name !== game.palette_name) uploadMaterials(game);
  rebuildBoards(game);
  arcadeEvents(game, now);
  if (game.you) noteShield(game, now);
  resultSplash(game);
  const seen = new Set();
  const place = (p, rackPiece) => {
    const [x, z, where, cellSize] = cellPosition(game, p.cell, rackPiece);
    const key = rackPiece ? `rack:${p.id}` : p.id;
    seen.add(key);
    const live = state.pieces.get(key);
    const size = where === "hud" ? 0.42 : family(p.mesh) === "cat" ? 0.38 : family(p.mesh) === "chess" ? 0.42 : family(p.mesh) === "disc" ? 0.34 : 0.36;
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
    if (live && live.removing == null) Object.assign(live, { sx: e.x, sz: e.y, vx: e.vx, vz: e.vy, tRecv: stamp, mesh: e.mesh, color: e.color, size: e.size, roll: e.roll, spin: e.spin, bob: e.bob });
    else state.pieces.set(e.id, { id: e.id, entity: true, lane: Number(String(e.id).split(":")[0]), sx: e.x, sz: e.y, x: e.x, z: e.y, vx: e.vx, vz: e.vy, tRecv: stamp, mesh: e.mesh, color: e.color,
                                  size: e.size, roll: e.roll, spin: e.spin, bob: e.bob, t0: now, removing: null, seed: 0 });
  }
  if (game.rack) for (const p of game.rack.pieces) place(p, true);
  if (game.input === "realtime") reconcile(game); else pred.ship = null;
  for (const [key, live] of state.pieces) if (!seen.has(key) && live.removing == null && (game.entities || !live.entity)) live.removing = now;
  const best = game.record && game.record.best ? ` · ${game.player}'s best ${game.record.best.score}` : "";
  status.textContent = `${game.status} · ${game.title}${best}`;
  $("hud").textContent = `${game.status}${best}`;
  if (game.record && game.record.plays !== state.recordPlays) { state.recordPlays = game.record.plays; if ($("scores").classList.contains("open")) showScores(); }
  const lichess = game.lichess_control;
  $("lichess-stop").disabled = !lichess || lichess.pause_requested;
  $("lichess-resume").disabled = !lichess || !lichess.pause_requested;
  syncMeowdokuBar(game);
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

// ---------------------------------------------------------------- draw
function draw(ms) {
  const now = ms / 1000; state.time = now;
  const w = canvas.clientWidth * devicePixelRatio | 0, h = canvas.clientHeight * devicePixelRatio | 0;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h); gl.clearColor(0.085, 0.085, 0.1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  if (!state.game) { requestAnimationFrame(draw); return; }
  const cam = camera(); updateLight(now);
  const casters = [[state.buttons.vao, state.buttons.count, M.identity(), null]];
  if (state.rackButtons) casters.push([state.rackButtons.vao, state.rackButtons.count, M.identity(), null]);
  const removed = [];
  for (const [key, piece] of state.pieces) {
    if (piece.removing != null && now - piece.removing > 0.3) { removed.push(key); continue; }
    casters.push([state.meshes[piece.mesh].vao, state.meshes[piece.mesh].count, pieceModel(piece, now), piece]);
  }
  for (const key of removed) state.pieces.delete(key);
  if (wob.on) wobFrame(casters, now); else for (const c of casters) if (c[3]) c[3].cage = null;
  if (state.flat) { drawFlat(cam, casters); requestAnimationFrame(draw); return; }
  // pass 1: depth from the key light
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.shadowMap.fbo); gl.viewport(0, 0, SHADOW_SIZE, SHADOW_SIZE); gl.clear(gl.DEPTH_BUFFER_BIT);
  gl.useProgram(state.programs.depth); gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(2.0, 4.0);
  for (const [vao, count, model] of casters) {
    gl.uniformMatrix4fv(gl.getUniformLocation(state.programs.depth, "uMVP"), false, M.multiply(light.viewProj, model));
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, count);
  }
  gl.disable(gl.POLYGON_OFFSET_FILL); gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h);
  // pass 2: the lit scene
  const P = state.programs.phong; gl.useProgram(P);
  const span = Math.max(state.extent.width, state.extent.depth);
  const keyView = M.apply(cam.view, light.position);
  const sun = sunState(), boardCentre = [state.extent.width / 2, 0, state.extent.depth / 2];
  const sunReach = Math.hypot(...sub(light.position, boardCentre));
  const keyColour = state.flat ? [1.0, 0.95, 0.86] : sun.colour, fillColour = [0.75, 0.85, 1.0];
  // The sun's intensity is set for its distance to the board's centre (soft,
  // even light); the fill light carries the arcade ring's colour and activity.
  // The house light is turned down so the arcade bulbs carry the scene: the
  // sun at ~60%, a softer fill; then each bulb is a point light in its own
  // colour at its own place, as bright as the chase makes it.
  const bulbs = state.flat ? [] : arcadeBulbs();
  const bulbStates = bulbs.map((_, i) => arcadeBulb(i, bulbs, now));
  const bulbLights = bulbs.map((p, i) => ({ view: M.apply(cam.view, p), colour: ARCADE.colours[bulbStates[i].ci],
                                           level: bulbStates[i].level }));
  const count = Math.min(3 + bulbLights.length, 100);
  gl.uniform1i(gl.getUniformLocation(P, "uNumLights"), count);
  gl.uniform3fv(gl.getUniformLocation(P, "uLightPos[0]"), new Float32Array([...keyView, 0.9 * span, 0.4 * span, 0.4 * span - 1.0,
    -0.6 * span, -0.1 * span, 0.3 * span - 1.0, ...bulbLights.slice(0, count - 3).flatMap((b) => b.view)]));
  gl.uniform3fv(gl.getUniformLocation(P, "uLightColor[0]"), new Float32Array([...keyColour, ...fillColour, 1.0, 0.8, 0.7,
    ...bulbLights.slice(0, count - 3).flatMap((b) => b.colour)]));
  gl.uniform1fv(gl.getUniformLocation(P, "uLightIntensity[0]"), new Float32Array([
    (state.flat ? 2.6 : 1.45 * sun.strength) * sunReach * sunReach, 0.19 * span * span, 0.14 * span * span,
    ...bulbLights.slice(0, count - 3).map((b) => ARCADE.lamp * b.level)]));
  gl.uniform1iv(gl.getUniformLocation(P, "uLightGroupId[0]"), new Int32Array(count).fill(-1));
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
  }
  // the glass plate under everything: blended, no depth write
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
  drawLit(state.glass.vao, state.glass.count, M.identity(), null);
  drawShield(now, drawLit);
  gl.depthMask(true); gl.disable(gl.BLEND);
  // (the sun and the bulbs are drawn as emissive geometry in the lit pass)
  gl.bindVertexArray(null);
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
const newGame = (payload) => {
  if (payload.game === "meowdoku" && $("palette").value) payload = { ...payload, palette: $("palette").value };
  const step = chain.then(async () => { console.log("[meow] new", JSON.stringify(payload)); applyState(await api("/api/new", payload)); }).catch((e) => console.warn(e));
  chain = step;
  return step;
};

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
  return t >= 0 ? { x: near[0] + t * (far[0] - near[0]), z: near[2] + t * (far[2] - near[2]) } : null;
}
const twoAxis = () => !!(state.game && state.game.predict && state.game.predict.y);
function controls() {
  // Keys give an axis; a held pointer gives the board point it asks the ship
  // to reach (on a two-axis board, a little above the finger so the finger
  // never covers it).  The law integrates the ship toward either with
  // damping; the page never closes a loop over a stale position.
  const axis = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
  const axisY = (held.has("down") ? 1 : 0) - (held.has("up") ? 1 : 0);
  const out = { move_axis: axis, steer_x: steer ? Math.max(0, steer.x) : -1, fire: steer || held.has("fire") ? 1 : 0 };
  if (twoAxis()) { out.move_axis_y = axisY; out.steer_y = steer ? Math.max(0, steer.z - 1.4) : -1; }
  return out;
}
function keyControl(key) {
  const k = key.length === 1 ? key.toLowerCase() : key;
  const common = { ArrowLeft: "left", a: "left", ArrowRight: "right", d: "right", " ": "fire" };
  if (common[k]) return common[k];
  if (twoAxis()) return { ArrowUp: "up", w: "up", ArrowDown: "down", s: "down" }[k];
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
let press = null, clickTimer = null, lastClick = null;
canvas.addEventListener("pointerdown", (e) => {
  if (state.game && state.game.input === "realtime") { steer = boardPlane(e.clientX, e.clientY); canvas.setPointerCapture(e.pointerId); return; }
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
canvas.addEventListener("pointercancel", () => { steer = null; });
canvas.addEventListener("pointerup", async (e) => {
  steer = null;
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
      await act({ type: "cell", r: h.r, c: h.c, button: "primary" }); return;
    }
    if (h && h.kind === "board") await act({ type: "cell", r: h.r, c: h.c, button: "primary" });
    return;
  }
  if (game.input === "text" && h && h.kind === "rack") {
    const letter = (game.rack.letters[h.r] || "")[h.c];
    if (letter) await act({ type: "text", text: game.text + letter });
  }
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
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
    if (state.game && state.game.board === "free") act({ type: "layout", aspect: canvas.clientWidth / Math.max(1, canvas.clientHeight) });
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
    } else if (g === "wordle") newGame({ game: "wordle", seed: Math.floor(Math.random() * 100000) });
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
  arrayTexture(4, [0, 255, 128, 255]); arrayTexture(5, [255, 255, 255, 0]); arrayTexture(6, [0, 0, 0, 255]); arrayTexture(7, [0, 0, 0, 0]);
  volumeTexture(8); storageTexture({ data: [0.0] }, 10);
  state.units = { turing_ssbo_pbr: 0, turing_ssbo_phong: 1, turing_ssbo_enamel: 2, turing_ssbo_texstack: 3,
                  uEmitUv: 4, uColorUv: 5, uDepthUv: 6, uRemitUv: 7, uFieldVolume: 8, uShadowMaps: 9, turing_ssbo_thermal_k: 10 };
  for (const [name, mesh] of Object.entries(assets.meshes)) state.meshes[name] = uploadMesh(roundMesh(mesh, roundParams.iters, roundParams.creaseDeg));
  // (the exported "mouse" mesh is not drawn: X marks are notes, not mice)
  try {
    pred.program = await TuringWasmProgram.load("motion.wasm", "motion.abi.json");
  } catch (error) { console.warn("[meow] ship predictor unavailable; playing unpredicted", error); }
  fillPickers();
  if (rememberedName()) await setPlayer(rememberedName());
  requestAnimationFrame(draw);
  $("new").click();
})().catch((error) => { status.textContent = String(error); console.error(error); });

// Meowdoku WebGL page: the tiles are 3D models (the glyph extrusions) lit by
// Pluck's base-material Phong shader as compiled for WebGL 2; the board is
// a prebaked region map with its relaxed ring hues; the game state machine
// is the Python session behind /api (the dt-system law), and this page only
// animates the difference between successive states.

const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
if (!gl) throw new Error("WebGL 2 is required");
const status = document.getElementById("status");

// ---------------------------------------------------------------- matrices
const M = {
  identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
  multiply(a, b) {            // column-major a*b
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
  lookAt(eye, target, up) {
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
                             -dot(x,eye), -dot(y,eye), -dot(z,eye), 1]);
  },
  invert(m) {   // general 4x4 inverse (for picking)
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

// ---------------------------------------------------------------- shaders
function compile(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + "\n" + src.split("\n").slice(0, 3).join("\n"));
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
precision highp float; in vec3 vColor; out vec4 FragColor;
void main(){ FragColor = vec4(pow(vColor, vec3(1.0/2.2)), 1.0); }`;

// ---------------------------------------------------------------- storage textures (the compiled SSBOs)
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
function arrayTexture(unit, rgba) {   // 1x1x2 identity texture arrays (base_gl_renderer's defaults)
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

// ---------------------------------------------------------------- scene
const state = {
  assets: null, boards: null, phong: null, flat: null, meshes: {}, tiles: null,
  board: null, game: null, pieces: new Map(), hearts: [3, 3], lastMove: null, level: null, time: 0,
};
const hsv = (h, s, v) => {
  const i = Math.floor(((h % 1) + 1) % 1 * 6), f = ((h % 1) + 1) % 1 * 6 - i;
  const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
};

function uploadMesh(mesh) {
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const count = mesh.positions.length / 3;
  const inter = new Float32Array(count * 8);
  for (let i = 0; i < count; i++) {
    inter.set(mesh.positions.slice(i*3, i*3+3), i*8); inter.set(mesh.normals.slice(i*3, i*3+3), i*8+3);
  }
  const vbo = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.bufferData(gl.ARRAY_BUFFER, inter, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
  gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 2, gl.FLOAT, false, 32, 24);
  gl.bindVertexArray(null);
  return { vao, count, material: mesh.material, extent: mesh.extent };
}

function buildTiles(board) {
  const n = board.n, pos = [], col = [], gap = 0.04;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const rgb = hsv(board.hues[board.region[r*n + c]], 0.34, 0.92);
    const x0 = c + gap, x1 = c + 1 - gap, z0 = r + gap, z1 = r + 1 - gap;
    const quad = [[x0,0,z0],[x1,0,z0],[x1,0,z1],[x0,0,z0],[x1,0,z1],[x0,0,z1]];
    for (const p of quad) { pos.push(...p); col.push(...rgb); }
  }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(col), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, count: pos.length / 3 };
}

// ---------------------------------------------------------------- camera
function camera() {
  const n = state.board ? state.board.n : 6;
  const aspect = canvas.width / canvas.height;
  const dist = (n + 2.2) * (aspect < 1 ? 1.75 / aspect : 1.45) + 1.0;
  const eye = [n / 2, dist * 0.78, n / 2 + 0.6 + dist * 0.72];
  const view = M.lookAt(eye, [n / 2, 0, n / 2 + 0.9], [0, 1, 0]);
  const proj = M.perspective(0.62, aspect, 0.1, 200);
  return { view, proj, viewProj: M.multiply(proj, view), eye };
}

// ---------------------------------------------------------------- animation curves
const ease = { out: (t) => 1 - (1 - t) ** 3, in: (t) => t ** 3, bounce(t) { return t < 0.85 ? t / 0.85 : 1 + 0.15 * Math.sin((t - 0.85) / 0.15 * Math.PI) * (1 - t); } };
function pieceModel(piece, now) {
  const age = now - piece.t0, k = piece.kind;
  let y = 0, spin = 0, s = 1, tilt = 0;
  if (piece.removing != null) {                      // shrink out
    const t = Math.min(1, (now - piece.removing) / 0.3); s = 1 - ease.in(t); y = 0.2 * t;
  } else if (k === "cat") {                          // jump up, spin on the way down, settle with a squash
    const T = 1.1, t = Math.min(1, age / T);
    y = 1.6 * Math.sin(Math.PI * Math.min(1, t / 0.9)) * (t < 0.9 ? 1 : 0);
    spin = t < 0.25 ? 0 : 2 * Math.PI * ease.out(Math.min(1, (t - 0.25) / 0.65));
    s = t > 0.9 ? 1 - 0.18 * Math.sin((t - 0.9) / 0.1 * Math.PI) : 1;
    if (t >= 1) { s = 1 + 0.03 * Math.sin(now * 2.1 + piece.r); }   // breathing
  } else if (k === "x") {                            // fall in with a small bounce
    const T = 0.55, t = Math.min(1, age / T);
    y = 3.0 * (1 - ease.bounce(t)); tilt = 0.6 * (1 - t);
  } else if (k === "heart") {                        // slow turn
    spin = now * 0.8 + piece.r * 1.3; y = 0.05 * Math.sin(now * 1.7 + piece.r);
  }
  const mesh = state.meshes[k === "x" ? "x_mark" : k];
  const scale = (k === "heart" ? 0.42 : 0.4) / mesh.extent * s;
  let m = M.translate(piece.x, y + 0.02, piece.z);
  m = M.multiply(m, M.rotateY(spin + (k === "x" ? 0 : 0)));
  m = M.multiply(m, M.rotateX(-Math.PI / 2 + 0.35 + tilt));     // glyph plane (x, y) lies on the board, leaning back toward the camera
  m = M.multiply(m, M.scale(scale, scale, scale));
  return m;
}

// ---------------------------------------------------------------- draw
function draw(now) {
  state.time = now;
  const w = canvas.clientWidth * devicePixelRatio | 0, h = canvas.clientHeight * devicePixelRatio | 0;
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.085, 0.085, 0.1, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
  if (!state.board) { requestAnimationFrame(draw); return; }
  const cam = camera();
  // tiles
  gl.useProgram(state.flat); gl.uniformMatrix4fv(gl.getUniformLocation(state.flat, "uMVP"), false, cam.viewProj);
  gl.bindVertexArray(state.tiles.vao); gl.drawArrays(gl.TRIANGLES, 0, state.tiles.count);
  // pieces through the Phong program
  const P = state.phong; gl.useProgram(P);
  const lights = state.assets.lights, n = state.board.n;
  const lpos = [], lcol = [], lint = [], lgid = [];
  for (const L of lights) {                                       // lights ride with the camera, in view space, scaled to the board
    lpos.push(L.position[0] * n, L.position[1] * n, L.position[2] * n * 0.5 - 1.0);
    lcol.push(...L.color); lint.push(L.intensity * n * n * 1.7); lgid.push(-1);
  }
  gl.uniform1i(gl.getUniformLocation(P, "uNumLights"), lights.length);
  gl.uniform3fv(gl.getUniformLocation(P, "uLightPos[0]"), new Float32Array(lpos));
  gl.uniform3fv(gl.getUniformLocation(P, "uLightColor[0]"), new Float32Array(lcol));
  gl.uniform1fv(gl.getUniformLocation(P, "uLightIntensity[0]"), new Float32Array(lint));
  gl.uniform1iv(gl.getUniformLocation(P, "uLightGroupId[0]"), new Int32Array(lgid));
  gl.uniform1f(gl.getUniformLocation(P, "uLightCalibration"), 1.0);
  gl.uniformMatrix3fv(gl.getUniformLocation(P, "uCatCcmMatrix"), false, new Float32Array([1,0,0, 0,1,0, 0,0,1]));
  gl.uniform1i(gl.getUniformLocation(P, "uEnableSpecular"), 1);
  gl.uniform1i(gl.getUniformLocation(P, "uEnableEmissionDirect"), 0);
  gl.uniform1i(gl.getUniformLocation(P, "uRenderPass"), 0);
  gl.uniform1f(gl.getUniformLocation(P, "uFieldGain"), 0.0);
  for (const [name, unit] of Object.entries(state.units)) gl.uniform1i(gl.getUniformLocation(P, name), unit);
  const removed = [];
  for (const [key, piece] of state.pieces) {
    if (piece.removing != null && now - piece.removing > 0.3) { removed.push(key); continue; }
    const mesh = state.meshes[piece.kind === "x" ? "x_mark" : piece.kind];
    const model = pieceModel(piece, now);
    const mv = M.multiply(cam.view, model), mvp = M.multiply(cam.proj, mv);
    gl.uniformMatrix4fv(gl.getUniformLocation(P, "uMVP"), false, mvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(P, "uMV"), false, mv);
    gl.bindVertexArray(mesh.vao);
    gl.vertexAttribI4i(2, mesh.material, 0, 0, 0); gl.vertexAttribI4i(4, 0, 0, 0, 0); gl.vertexAttribI4i(5, 0, 0, 0, 0); gl.vertexAttrib3f(6, 0, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
  }
  for (const key of removed) state.pieces.delete(key);
  gl.bindVertexArray(null);
  requestAnimationFrame(draw);
}

// ---------------------------------------------------------------- game state -> pieces
function applyState(game) {
  const previous = state.game; state.game = game;
  const n = game.n, now = state.time;
  for (let i = 0; i < n * n; i++) {
    const r = Math.floor(i / n), c = i % n, key = `cell-${i}`, piece = state.pieces.get(key);
    const want = game.cat[i] ? "cat" : game.x_mark[i] ? "x" : null;
    const have = piece && piece.removing == null ? piece.kind : null;
    if (want === have) continue;
    if (piece && piece.removing == null) piece.removing = now;
    if (want) state.pieces.set(want === have ? key : `${key}-${now}`, { kind: want, x: c + 0.5, z: r + 0.5, r, t0: now, removing: null });
  }
  for (let h = 0; h < 3; h++) {
    const key = `heart-${h}`, piece = state.pieces.get(key), alive = h < game.hearts;
    if (alive && !(piece && piece.removing == null)) state.pieces.set(key, { kind: "heart", x: n / 2 + (h - 1) * 0.9, z: n + 0.9, r: h, t0: now - 5, removing: null });
    if (!alive && piece && piece.removing == null) piece.removing = now;
  }
  const lost = previous && previous.hearts > game.hearts;
  status.textContent = game.won ? "solved! all cats home" : game.lost ? "out of hearts" :
    `${game.hearts} ♥ · ${game.moves} moves${lost ? " · wrong cell" : ""} · ${state.board.level} · ${state.board.type}`;
}

async function api(path, body) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await response.json();
  if (data.error) { status.textContent = data.error; throw new Error(data.error); }
  return data;
}
async function newBoard(board) {
  state.board = board; state.tiles = buildTiles(board); state.pieces.clear();
  applyState(await api("/api/new", { id: board.id }));
}

// ---------------------------------------------------------------- picking
function cellAt(clientX, clientY) {
  const cam = camera(), rect = canvas.getBoundingClientRect();
  const ndc = [((clientX - rect.left) / rect.width) * 2 - 1, 1 - ((clientY - rect.top) / rect.height) * 2];
  const inv = M.invert(cam.viewProj);
  const near = M.apply(inv, [ndc[0], ndc[1], -1]), far = M.apply(inv, [ndc[0], ndc[1], 1]);
  const t = -near[1] / (far[1] - near[1]);                          // hit the y = 0 board plane
  if (!(t >= 0)) return null;
  const x = near[0] + t * (far[0] - near[0]), z = near[2] + t * (far[2] - near[2]);
  const n = state.board.n, c = Math.floor(x), r = Math.floor(z);
  return (c >= 0 && c < n && r >= 0 && r < n) ? { r, c } : null;
}
let press = null;
canvas.addEventListener("pointerdown", (e) => { press = { x: e.clientX, y: e.clientY, t: performance.now(), button: e.button }; });
canvas.addEventListener("pointerup", async (e) => {
  if (!press || !state.board) return;
  const long = performance.now() - press.t > 450, button = (press.button === 2 || long) ? "secondary" : "primary";
  press = null;
  const cell = cellAt(e.clientX, e.clientY);
  console.log("[meowdoku] click", e.clientX, e.clientY, "->", cell, button);
  if (cell) applyState(await api("/api/move", { r: cell.r, c: cell.c, button }));
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("keydown", async (e) => {
  if (e.key === "u" || e.key === "U") applyState(await api("/api/undo"));
  if (e.key === "r" || e.key === "R") applyState(await api("/api/reset"));
  if (e.key === "n" || e.key === "N") document.getElementById("new").click();
});
document.getElementById("undo").onclick = async () => applyState(await api("/api/undo"));
document.getElementById("reset").onclick = async () => applyState(await api("/api/reset"));

// ---------------------------------------------------------------- level picker
function fillPickers() {
  const levelSelect = document.getElementById("level"), boardSelect = document.getElementById("board");
  const levels = Object.keys(state.boards.levels);
  levelSelect.innerHTML = levels.map((k) => `<option value="${k}">${k} (${state.boards.levels[k]})</option>`).join("");
  const refill = () => {
    const chosen = state.boards.boards.filter((b) => b.level === levelSelect.value);
    boardSelect.innerHTML = chosen.map((b) => `<option value="${b.id}">${b.id} · ${b.type}</option>`).join("");
  };
  levelSelect.onchange = refill; refill();
  document.getElementById("new").onclick = () => {
    const chosen = state.boards.boards.filter((b) => b.level === levelSelect.value);
    const board = chosen[Math.floor(Math.random() * chosen.length)];
    boardSelect.value = board.id; newBoard(board);
  };
  boardSelect.onchange = () => newBoard(state.boards.boards.find((b) => b.id === boardSelect.value));
}

// ---------------------------------------------------------------- boot
(async () => {
  const [vert, frag, assets, boards] = await Promise.all([
    fetch("shaders/base_material.webgl.vert.glsl").then((r) => r.text()),
    fetch("shaders/base_material.webgl.frag.glsl").then((r) => r.text()),
    fetch("assets.json").then((r) => r.json()),
    fetch("boards.json").then((r) => r.json()),
  ]);
  state.assets = assets; state.boards = boards;
  state.phong = link(compile(gl.VERTEX_SHADER, vert), compile(gl.FRAGMENT_SHADER, frag));
  state.flat = link(compile(gl.VERTEX_SHADER, FLAT_VS), compile(gl.FRAGMENT_SHADER, FLAT_FS));
  // The compiled SSBOs as R32F textures, the identity UV stacks and the empty field volume.
  storageTexture(assets.materials.pbr, 0); storageTexture(assets.materials.phong, 1);
  storageTexture(assets.materials.enamel, 2); storageTexture(assets.materials.texstack, 3);
  arrayTexture(4, [0, 255, 128, 255]); arrayTexture(5, [255, 255, 255, 0]); arrayTexture(6, [0, 0, 0, 255]); arrayTexture(7, [0, 0, 0, 0]);
  volumeTexture(8);
  state.units = { turing_ssbo_pbr: 0, turing_ssbo_phong: 1, turing_ssbo_enamel: 2, turing_ssbo_texstack: 3,
                  uEmitUv: 4, uColorUv: 5, uDepthUv: 6, uRemitUv: 7, uFieldVolume: 8 };
  for (const role of ["cat", "x_mark", "heart"]) state.meshes[role] = uploadMesh(assets.meshes[role]);
  fillPickers();
  requestAnimationFrame(draw);
  document.getElementById("level").value = Object.keys(boards.levels).find((k) => k.startsWith("small/forced")) || Object.keys(boards.levels)[0];
  document.getElementById("level").onchange();
  document.getElementById("new").click();
})().catch((error) => { status.textContent = String(error); console.error(error); });

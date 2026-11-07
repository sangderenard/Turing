// MOBROAD, the page's side (examples/meowdoku/mobroad.py; the mob is the fountain law's mobroad variant, fountain.py ``make_law_mobroad``).
//
// WHO OWNS WHAT.  The mob (every grain) is the law the compute runtime runs here; the server never sees a grain.  The server owns the shooters, the bullets
// and the panels.  Two one-way links join them:
//   server -> law   each batch the bullets' places become the law's kill segments (``mobroadInputs`` = fountain.py ``mobroad_inputs``; a test holds them equal):
//                   a bullet is the SEGMENT from where it was last written to where it is now, so a fast bullet cannot tunnel a grain;
//   law -> server   the law's own cumulative counts (n_dead, n_goal, n_crush) ride the tick (``report``); the server scores and loses from them.
// The panels are glass boxes in the scene's translucent material path; their values are labels projected from the scene's camera.

export const MOB_KZ = 32;
export const KZ_FIELDS = ["on", "ax", "ay", "bx", "by", "r"];

// fountain.py ``mobroad_inputs``.  ``bullets``: [{id, x, y, kr}] in board units (x right, y DOWN); ``prev``: Map id -> [x, y] last written; ``map``: the
// game's board map (half, top, k).  The MOB_KZ most advanced bullets (smallest y, then id) are used, the rest slots off.
export function mobroadInputs(bullets, prev, map) {
  const { k, half, top } = map, out = {};
  const used = [...(bullets || [])].sort((a, b) => (a.y - b.y) || (String(a.id) < String(b.id) ? -1 : String(a.id) > String(b.id) ? 1 : 0)).slice(0, MOB_KZ);
  for (let slot = 0; slot < MOB_KZ; slot++) {
    const b = used[slot], p = b && prev && prev.get(b.id) ? prev.get(b.id) : b && [b.x, b.y];
    const v = b ? [1, p[0] / k - half, top - p[1] / k, b.x / k - half, top - b.y / k, b.kr / k] : [0, 0, 0, 0, 0, 0];
    KZ_FIELDS.forEach((f, i) => { out[`kz${slot}_${f}`] = Math.fround(v[i]); });
  }
  return out;
}

// fountain.py ``mobroad_target``: the player at board (bx, by) as the pather's target (the law's ``tgt_x, tgt_y``, fountain units)
export function mobroadTarget(bx, by, map) {
  const { k, half, top } = map;
  return { tgt_x: Math.fround(bx / k - half), tgt_y: Math.fround(top - by / k) };
}

// fountain.py ``mobroad_pather_inputs``: the pather's weights as the law's uniforms (``pw<i>``)
export function mobroadPather(w) {
  const out = {};
  w.forEach((v, i) => { out[`pw${i}`] = Math.fround(v); });
  return out;
}

const KIND_MATERIAL = { split: "sapphire@glass", damage: "jade@glass", rate: "ruby@glass", shooters: "opal@glass" };
const KIND_LABEL = { split: "SPLIT", damage: "DAMAGE", rate: "FIRE RATE", shooters: "SHOOTERS" };
const KIND_COLOR = { split: "#7fb2ff", damage: "#7fe3a0", rate: "#ff7f8e", shooters: "#d8e6ff" };
const PANEL_H = 1.1;

// the panel's text: a pass-through panel shows the multiplier it gives (``x2.50``), a completion panel its progress to the target
export function panelText(p) {
  if (!p.alive) return ["SPENT", KIND_LABEL[p.kind]];
  if (p.kind === "rate" || p.kind === "shooters") return [`${p.value.toFixed(1)} / ${p.target.toFixed(1)}`, `${KIND_LABEL[p.kind]} x${p.amount.toFixed(0)}`];
  return [`x${p.value.toFixed(2)}`, KIND_LABEL[p.kind]];
}

function project(vp, x, y, z, w, h) {
  const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12], cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13], cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
  return cw > 1e-6 ? [(cx / cw * 0.5 + 0.5) * w, (1 - (cy / cw * 0.5 + 0.5)) * h] : null;
}

export class MobroadPage {
  constructor() {
    this.meshes = []; this.key = ""; this.root = null; this.labels = new Map(); this.hud = null; this.toggle = null;
    let kind = null;
    try { kind = new URLSearchParams(location.search).get("pather") || localStorage.getItem("mobroad.pather"); } catch (e) { /* no storage: the default */ }
    this.patherKind = kind === "hand" ? "hand" : "trained";             // which weights the law walks the mob with
  }

  // THE PLAYER (the pather's target), board units: the server's player piece (lane 0), else where it starts
  target(state, game) {
    for (const piece of state.pieces.values()) if (piece.entity && piece.mesh === "rocket" && piece.removing == null && String(piece.id).startsWith("0:")) return [piece.sx, piece.sz];
    const m = game.mobroad.map;
    return [m.board[0] / 2, m.board[1] - 0.9];
  }

  // THE PATHER'S WEIGHTS the law is given (the spec carries both sets), per the page's toggle
  pather(game) {
    const p = game.compute && game.compute.pather;
    return p ? p[this.patherKind] || p[p.default] : null;
  }

  // THE BULLETS at this instant (the server's, carried on by their velocity since the snapshot), as the kill segments' endpoints: [{id, x, y, kr}]
  collect(state, gameNow, game) {
    const out = [], per = game.mobroad.reach_per_size;
    for (const piece of state.pieces.values()) {
      if (!piece.entity || piece.mesh !== "bolt" || piece.removing != null) continue;
      const lead = Math.min(Math.max(0, gameNow - (piece.tRecv ?? gameNow)), 0.1);
      out.push({ id: piece.id, x: piece.sx + piece.vx * lead, y: piece.sz + piece.vz * lead, kr: (piece.size || 0.26) * per });
    }
    return out;
  }

  // the law's counts, for the tick (null until the law has reported for THIS run)
  report(game, compute) {
    const m = game && game.mobroad, d = compute && compute.diag;
    if (!m || !d || !compute.spec || compute.spec.epoch !== m.epoch || d.n_dead === undefined) return null;
    return { epoch: m.epoch, dead: d.n_dead | 0, goal: d.n_goal | 0, crush: d.n_crush | 0, live: d.n_live | 0, step: d.step | 0 };
  }

  #ensureDom() {
    if (this.root) return;
    const root = document.createElement("div");
    root.id = "mobroad-hud";
    root.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:6;font:600 13px system-ui,sans-serif;color:#fff;text-shadow:0 1px 3px #000";
    document.body.appendChild(root);
    const hud = document.createElement("div");
    hud.style.cssText = "position:absolute;left:50%;top:150px;transform:translate(-50%,-100%);padding:3px 12px;border-radius:12px;background:#0009;white-space:nowrap";
    root.appendChild(hud);
    const toggle = document.createElement("button");
    toggle.style.cssText = "position:absolute;right:10px;bottom:10px;pointer-events:auto;padding:4px 10px;border-radius:12px;border:1px solid #fff6;background:#0009;color:#fff;font:600 12px system-ui,sans-serif;cursor:pointer";
    toggle.title = "the mob's pather: the hand-set weights or the trained ones (runtime uniforms of the law: no recompile)";
    toggle.onclick = () => {
      this.patherKind = this.patherKind === "hand" ? "trained" : "hand";
      try { localStorage.setItem("mobroad.pather", this.patherKind); } catch (e) { /* per-viewer convenience only */ }
      this.#label();
    };
    root.appendChild(toggle);
    this.root = root; this.hud = hud; this.toggle = toggle; this.#label();
  }

  #label() { if (this.toggle) this.toggle.textContent = `pather: ${this.patherKind}`; }

  // called with every state; hides the overlay for any other game
  update(game) {
    if (!game || !game.mobroad) { if (this.root) this.root.style.display = "none"; return; }
    this.#ensureDom();
    this.root.style.display = "";
    const m = game.mobroad;
    this.hud.textContent = `down ${m.dead}/${m.active} · through ${m.goal}/${m.lose_at} · crush ${m.crush} · rate x${m.rate.toFixed(2).replace(/\.?0+$/, "")} · shooters ${m.shooters}`;
  }

  // the panels, as glass: meshes rebuilt when a panel's kind / liveness changes, labels re-projected every frame
  draw(api, cam, game, drawLit) {
    if (!game || !game.mobroad) return;
    this.#ensureDom();
    const panels = game.mobroad.panels, idx = api.materialIndex(), floor = api.floorY();
    const key = JSON.stringify([panels.map((p) => [p.id, p.kind, p.alive, p.x, p.y, p.hw]), idx["sapphire@glass"] ?? -1, game.width, game.height]);
    if (key !== this.key) {
      for (const m of this.meshes) api.disposeMesh(m);
      this.meshes = panels.filter((p) => p.alive).map((p) => api.litMesh(api.box(p.x - p.hw, floor, p.y - p.thick, p.x + p.hw, floor + PANEL_H, p.y + p.thick, idx[KIND_MATERIAL[p.kind]] ?? idx["housing@glass"])));
      this.key = key;
    }
    for (const m of this.meshes) drawLit(m.vao, m.count, api.model(), null);
    const rect = api.canvas.getBoundingClientRect();
    const top = project(cam.viewProj, game.width / 2, floor, 0, rect.width, rect.height);           // the status overlay rides the road's top edge
    if (top) { this.hud.style.left = `${rect.left + top[0]}px`; this.hud.style.top = `${Math.max(rect.top + 6, rect.top + top[1] - 6)}px`; }
    for (const p of panels) {
      let el = this.labels.get(p.id);
      if (!el) {
        el = document.createElement("div");
        el.style.cssText = "position:absolute;transform:translate(-50%,-50%);text-align:center;white-space:nowrap;line-height:1.05;transition:font-size 80ms";
        this.root.appendChild(el); this.labels.set(p.id, el);
      }
      const at = project(cam.viewProj, p.x, floor + 0.5 * PANEL_H, p.y, rect.width, rect.height);
      if (!at) { el.style.display = "none"; continue; }
      const [big, small] = panelText(p);
      el.style.display = "";
      el.style.left = `${rect.left + at[0]}px`; el.style.top = `${rect.top + at[1]}px`;
      el.style.color = p.alive ? KIND_COLOR[p.kind] : "#889";
      el.style.fontSize = `${Math.round(18 + 14 * p.flash)}px`;
      el.innerHTML = `<div>${big}</div><div style="font-size:10px;letter-spacing:.12em;opacity:.85">${small}</div>`;
    }
  }
}

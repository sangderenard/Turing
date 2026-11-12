// /worldmap.html: a compact repository world map.  Regions (folders) nest as a squarified treemap by bytes, rooms (files) are the
// cells inside them, portals (cross-file references) are curves drawn only for the selected room.  Plain ES module, canvas 2D, no libraries;
// everything is set with textContent / createElement, never innerHTML.  The layout functions at the top are pure and exported so Node can test them.

// ---------------------------------------------------------------- pure layout
const weight = (bytes) => Math.max(Number(bytes) || 0, 1);

/** Per-map structure built once: kids[region] = child region ids, roomsOf[region] = room ids, total[region] = subtree bytes. */
export function buildIndex(map) {
  const regions = map.regions || [], rooms = map.rooms || [];
  const kids = regions.map(() => []), roomsOf = regions.map(() => []);
  const total = new Float64Array(regions.length);
  for (const g of regions) if (g.id !== 0 && g.parent !== null && g.parent !== undefined && g.parent >= 0) kids[g.parent].push(g.id);
  for (const m of rooms) { roomsOf[m.region].push(m.id); total[m.region] += weight(m.bytes); }
  for (let i = regions.length - 1; i > 0; i--) { const p = regions[i].parent; if (p >= 0 && p !== i) total[p] += total[i]; }   // parents precede children
  return { kids, roomsOf, total, regions, rooms };
}

/** Squarified treemap: items = [{key, w}] (w > 0), rect = {x,y,w,h}; returns [{key, x, y, w, h}] with areas proportional to w. */
export function squarify(items, rect) {
  const out = [];
  const sorted = items.slice().sort((a, b) => b.w - a.w);
  let sum = 0;
  for (const it of sorted) sum += it.w;
  if (!sorted.length || sum <= 0 || rect.w <= 0 || rect.h <= 0) return out;
  const scale = (rect.w * rect.h) / sum;
  let x = rect.x, y = rect.y, w = rect.w, h = rect.h, i = 0;
  const worst = (mx, mn, rowArea, side) => {
    const s2 = side * side, r2 = rowArea * rowArea;
    return Math.max((s2 * mx) / r2, r2 / (s2 * mn));
  };
  while (i < sorted.length) {
    const side = Math.min(w, h);
    let rowArea = sorted[i].w * scale, mx = rowArea, mn = rowArea, j = i + 1;
    while (j < sorted.length) {                          // sorted descending: the new item is the row's smallest
      const a = sorted[j].w * scale;
      if (worst(mx, a, rowArea + a, side) <= worst(mx, mn, rowArea, side)) { rowArea += a; mn = a; j++; } else break;
    }
    const thick = rowArea / side;                        // the strip's extent across the short side
    let pos = 0;
    for (let k = i; k < j; k++) {
      const len = (sorted[k].w * scale) / thick;
      if (w >= h) out.push({ key: sorted[k].key, x, y: y + pos, w: thick, h: len });      // strip is a column at the left
      else out.push({ key: sorted[k].key, x: x + pos, y, w: len, h: thick });             // strip is a row at the top
      pos += len;
    }
    if (w >= h) { x += thick; w -= thick; } else { y += thick; h -= thick; }
    i = j;
  }
  return out;
}

/**
 * Nested layout of region `root` filling [0,width]x[0,height].  Returns {rr, rm, hdr, root}: rr[4*r..] = region rect (w=0 when not placed),
 * rm[4*m..] = room rect, hdr[r] = label-strip height reserved at the top of region r.  A region's content (child regions + its own rooms)
 * is squarified inside its rect minus `pad` on every side and minus the header strip.
 */
export function layoutMap(index, width, height, root = 0, opts = {}) {
  const pad = opts.pad ?? 2, headerH = opts.header ?? 14, headerMin = opts.headerMinWidth ?? 56;
  const { kids, roomsOf, total, rooms, regions } = index;
  const rr = new Float64Array(regions.length * 4), rm = new Float64Array(rooms.length * 4), hdr = new Float64Array(regions.length);
  const place = (r, rect) => {
    rr[4 * r] = rect.x; rr[4 * r + 1] = rect.y; rr[4 * r + 2] = rect.w; rr[4 * r + 3] = rect.h;
    const top = r !== root && rect.w >= headerMin && rect.h >= headerH * 3 ? headerH : 0;
    hdr[r] = top;
    const inner = { x: rect.x + pad, y: rect.y + pad + top, w: rect.w - 2 * pad, h: rect.h - 2 * pad - top };
    if (inner.w <= 0 || inner.h <= 0) return;
    const items = [];
    for (const k of kids[r]) if (total[k] > 0) items.push({ key: "r" + k, w: total[k] });
    for (const m of roomsOf[r]) items.push({ key: "m" + m, w: weight(rooms[m].bytes) });
    for (const c of squarify(items, inner)) {
      const id = +c.key.slice(1);
      if (c.key[0] === "m") { rm[4 * id] = c.x; rm[4 * id + 1] = c.y; rm[4 * id + 2] = c.w; rm[4 * id + 3] = c.h; }
      else place(id, c);
    }
  };
  place(root, { x: 0, y: 0, w: width, h: height });
  return { rr, rm, hdr, root };
}

const inside = (arr, i, x, y) => arr[4 * i + 2] > 0 && x >= arr[4 * i] && x < arr[4 * i] + arr[4 * i + 2] && y >= arr[4 * i + 1] && y < arr[4 * i + 1] + arr[4 * i + 3];

/** Deepest region under (x,y) and the room cell there (-1 when over the region's own padding / header / a gap). */
export function hitTest(index, layout, x, y) {
  let r = layout.root;
  if (!inside(layout.rr, r, x, y)) return { region: -1, room: -1 };
  for (;;) {
    let next = -1;
    for (const k of index.kids[r]) if (inside(layout.rr, k, x, y)) { next = k; break; }
    if (next < 0) break;
    r = next;
  }
  for (const m of index.roomsOf[r]) if (inside(layout.rm, m, x, y)) return { region: r, room: m };
  return { region: r, room: -1 };
}

// ---------------------------------------------------------------- per-room portal adjacency (typed arrays, built once)
export function buildAdjacency(map) {
  const n = (map.rooms || []).length, p = map.portals || [];
  const outStart = new Uint32Array(n + 1), inStart = new Uint32Array(n + 1);
  for (const e of p) { outStart[e[0] + 1]++; inStart[e[1] + 1]++; }
  for (let i = 0; i < n; i++) { outStart[i + 1] += outStart[i]; inStart[i + 1] += inStart[i]; }
  const outPortal = new Uint32Array(p.length), inPortal = new Uint32Array(p.length);   // portal indices grouped by source / target room
  const oc = outStart.slice(0, n), ic = inStart.slice(0, n);
  for (let k = 0; k < p.length; k++) { outPortal[oc[p[k][0]]++] = k; inPortal[ic[p[k][1]]++] = k; }
  return { outStart, outPortal, inStart, inPortal };
}

// ---------------------------------------------------------------- page
const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#86bc86", "#d37295", "#8cd17d"];
const OTHER = "#7b7f87";
const PORTAL_CAP = 400, LINK_CAP = 200;

function startPage() {
  const $ = (id) => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined && text !== null) n.textContent = String(text); if (cls) n.className = cls; return n; };
  const canvas = $("map"), ctx = canvas.getContext("2d");
  const S = { map: null, idx: null, adj: null, layout: null, view: 0, sel: -1, hover: -1, cam: { k: 1, x: 0, y: 0 }, links: false,
              query: "", matches: null, paths: null, langColor: new Map(), W: 0, H: 0, dpr: 1 };
  let drag = null;

  async function getJson(url) {
    const r = await fetch(url, { credentials: "same-origin", cache: "no-store" });
    let body = null;
    try { body = await r.json(); } catch (_) { /* not json */ }
    return { status: r.status, body };
  }

  const roomPath = (m) => { const room = S.map.rooms[m], g = S.map.regions[room.region]; return g.path ? g.path + "/" + room.name : room.name; };
  const colorOf = (lang) => S.langColor.get(lang || "") || OTHER;
  const setStatus = (text, bad) => { const s = $("status"); s.textContent = text; s.className = bad ? "bad" : "mut"; };

  async function loadIndex() {
    const { status, body } = await getJson("/api/worldmap/index");
    if (status !== 200 || !body) return setStatus(status === 403 ? "Developer access is required." : "Could not load the index (" + status + ").", true);
    const repos = body.repos || [];
    const sel = $("repo");
    sel.replaceChildren();
    for (const r of repos) { const o = el("option", r.repo + " @ " + String(r.commit || "").slice(0, 8)); o.value = r.dir + "|" + r.commit; sel.append(o); }
    if (!repos.length) return setStatus("No world maps have been generated yet.");
    sel.addEventListener("change", () => loadMap(sel.value));
    loadMap(sel.value);
  }

  async function loadMap(value) {
    const [dir, commit] = value.split("|");
    setStatus("Loading map...");
    const { status, body } = await getJson("/api/worldmap/map?repo=" + encodeURIComponent(dir) + "&commit=" + encodeURIComponent(commit));
    if (status !== 200 || !body || !body.regions) return setStatus(status === 404 ? "That map is missing." : "Could not load the map (" + status + ").", true);
    S.map = body; S.idx = buildIndex(body); S.adj = buildAdjacency(body);
    S.paths = null; S.matches = null; S.sel = -1; S.hover = -1; S.view = 0;
    const counts = new Map();
    for (const m of body.rooms) counts.set(m.language || "", (counts.get(m.language || "") || 0) + weight(m.bytes));
    const langs = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    S.langColor = new Map(langs.slice(0, PALETTE.length).map(([l], i) => [l, PALETTE[i]]));
    const legend = $("legend");
    legend.replaceChildren();
    const chip = (color, label) => { const s = el("span", null, "chip"), sw = el("i"); sw.style.background = color; s.append(sw, document.createTextNode(label)); legend.append(s); };
    langs.slice(0, PALETTE.length).forEach(([l], i) => chip(PALETTE[i], l || "other"));
    if (langs.length > PALETTE.length) chip(OTHER, "other");
    const when = body.generated ? new Date(body.generated * 1000).toLocaleString() : "?";
    $("stats").textContent = body.regions.length + " regions, " + body.rooms.length + " rooms, " + (body.portals || []).length + " portals, " +
      (body.unresolved_count || 0) + " unresolved, commit " + String(body.commit || "").slice(0, 8) + ", generated " + when;
    setStatus("");
    relayout(); renderPanel(); draw();
  }

  function relayout() {
    S.layout = layoutMap(S.idx, S.W, S.H, S.view);
    S.cam = { k: 1, x: 0, y: 0 };
    breadcrumb();
  }

  function breadcrumb() {
    const bar = $("crumbs");
    bar.replaceChildren();
    const chain = [];
    for (let r = S.view; ; r = S.map.regions[r].parent) { chain.unshift(r); if (r === 0 || !(S.map.regions[r].parent >= 0)) break; }
    chain.forEach((r, i) => {
      if (i) bar.append(document.createTextNode(" / "));
      const b = el("button", r === 0 ? (S.map.repo || "root") : S.map.regions[r].path.split("/").pop(), "crumb");
      b.type = "button";
      b.addEventListener("click", () => { S.view = r; relayout(); draw(); });
      bar.append(b);
    });
  }

  function resize() {
    const box = canvas.parentElement.getBoundingClientRect();
    S.dpr = window.devicePixelRatio || 1;
    S.W = Math.max(200, Math.floor(box.width));
    S.H = Math.max(240, Math.min(Math.floor(window.innerHeight * 0.72), 900));
    canvas.width = Math.floor(S.W * S.dpr); canvas.height = Math.floor(S.H * S.dpr);
    canvas.style.width = S.W + "px"; canvas.style.height = S.H + "px";
    if (S.map) { relayout(); draw(); }
  }

  const sx = (x) => x * S.cam.k + S.cam.x, sy = (y) => y * S.cam.k + S.cam.y;
  const roomCenter = (m) => { const a = S.layout.rm; return a[4 * m + 2] > 0 ? [sx(a[4 * m] + a[4 * m + 2] / 2), sy(a[4 * m + 1] + a[4 * m + 3] / 2)] : null; };

  function draw() {
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.fillStyle = "#17171a";
    ctx.fillRect(0, 0, S.W, S.H);
    if (!S.map) return;
    const { rr, rm, hdr } = S.layout, k = S.cam.k, rooms = S.map.rooms;
    ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "top"; ctx.lineWidth = 1;
    // room cells first: skip the ones off screen or under a pixel
    for (let m = 0; m < rooms.length; m++) {
      const w = rm[4 * m + 2] * k, h = rm[4 * m + 3] * k;
      if (w <= 0 || (w < .6 && h < .6)) continue;
      const x = sx(rm[4 * m]), y = sy(rm[4 * m + 1]);
      if (x > S.W || y > S.H || x + w < 0 || y + h < 0) continue;
      ctx.fillStyle = colorOf(rooms[m].language);
      ctx.globalAlpha = S.matches && !S.matches.has(m) ? .15 : .9;
      ctx.fillRect(x, y, Math.max(w - 1, .5), Math.max(h - 1, .5));
    }
    ctx.globalAlpha = 1;
    // region frames and labels, only for visible regions big enough to read
    const stack = [S.view];
    while (stack.length) {
      const r = stack.pop();
      const x = sx(rr[4 * r]), y = sy(rr[4 * r + 1]), w = rr[4 * r + 2] * k, h = rr[4 * r + 3] * k;
      if (w < 3 || h < 3 || x > S.W || y > S.H || x + w < 0 || y + h < 0) continue;
      if (r !== S.view) { ctx.strokeStyle = "rgba(190,200,220,.5)"; ctx.strokeRect(x + .5, y + .5, w - 1, h - 1); }
      if (hdr[r] && w > 56) {
        ctx.fillStyle = "#e4e7ec"; ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, hdr[r] * k); ctx.clip();
        ctx.fillText(S.map.regions[r].path.split("/").pop(), x + 3, y + 2); ctx.restore();
      }
      for (const c of S.idx.kids[r]) stack.push(c);
    }
    if (S.matches) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      for (const m of S.matches) if (rm[4 * m + 2] > 0) ctx.strokeRect(sx(rm[4 * m]) - .5, sy(rm[4 * m + 1]) - .5, Math.max(rm[4 * m + 2] * k, 3), Math.max(rm[4 * m + 3] * k, 3));
      ctx.lineWidth = 1;
    }
    if (S.links) drawRegionLinks();
    if (S.sel >= 0) drawPortals(S.sel);
    for (const m of [S.hover, S.sel]) {
      if (m < 0 || rm[4 * m + 2] <= 0) continue;
      ctx.strokeStyle = m === S.sel ? "#ffd166" : "#fff"; ctx.lineWidth = 2;
      ctx.strokeRect(sx(rm[4 * m]), sy(rm[4 * m + 1]), rm[4 * m + 2] * k, rm[4 * m + 3] * k);
      ctx.lineWidth = 1;
    }
  }

  function curve(a, b, color) {
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1];
    ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.quadraticCurveTo(mx - dy * .25, my + dx * .25, b[0], b[1]); ctx.stroke();
  }

  function portalLists(m) {
    const p = S.map.portals || [], a = S.adj, out = [], inn = [];
    for (let i = a.outStart[m]; i < a.outStart[m + 1]; i++) out.push(p[a.outPortal[i]]);
    for (let i = a.inStart[m]; i < a.inStart[m + 1]; i++) inn.push(p[a.inPortal[i]]);
    return { out, inn };
  }

  function drawPortals(m) {
    const { out, inn } = portalLists(m), from = roomCenter(m);
    if (!from) return;
    ctx.lineWidth = 1; ctx.globalAlpha = .75;
    for (const e of out.slice(0, PORTAL_CAP)) { const t = roomCenter(e[1]); if (t) curve(from, t, "#ff6b6b"); }     // out: red
    for (const e of inn.slice(0, PORTAL_CAP)) { const t = roomCenter(e[0]); if (t) curve(t, from, "#4dd0e1"); }     // in: cyan
    ctx.globalAlpha = 1;
  }

  function drawRegionLinks() {
    const links = (S.map.region_links || []).slice().sort((a, b) => b[2] - a[2]).slice(0, LINK_CAP);
    const rc = (r) => { const a = S.layout.rr; return a[4 * r + 2] > 0 ? [sx(a[4 * r] + a[4 * r + 2] / 2), sy(a[4 * r + 1] + a[4 * r + 3] / 2)] : null; };
    ctx.globalAlpha = .55;
    for (const [a, b, n] of links) {
      const pa = rc(a), pb = rc(b);
      if (!pa || !pb) continue;
      ctx.lineWidth = 0.6 + Math.log(1 + n);
      curve(pa, pb, "#c3a6ff");
    }
    ctx.globalAlpha = 1; ctx.lineWidth = 1;
  }

  function renderPanel() {
    const box = $("panel");
    box.replaceChildren();
    if (S.sel < 0) { box.append(el("p", "Click a room to see what it calls and what calls it.", "mut")); return; }
    const m = S.sel, room = S.map.rooms[m], { out, inn } = portalLists(m);
    box.append(el("h2", roomPath(m)), el("p", [room.language || "other", room.lines + " lines", room.bytes + " bytes"].join(", "), "mut"));
    const section = (title, list, which) => {
      box.append(el("h3", title + " (" + list.length + ")"));
      if (list.length > PORTAL_CAP) box.append(el("p", "Showing the first " + PORTAL_CAP + " of " + list.length + "; the map draws the same " + PORTAL_CAP + ".", "mut"));
      const ul = el("ul", null, "plist");
      for (const e of list.slice(0, PORTAL_CAP)) {
        const other = which === "out" ? e[1] : e[0];
        const li = el("li"), b = el("button", roomPath(other), "link");
        b.type = "button";
        b.addEventListener("click", () => jumpTo(other));
        li.append(b, el("span", " " + ((S.map.kinds || [])[e[3]] ?? "") + " line " + e[2], "mut"));
        ul.append(li);
      }
      box.append(ul);
    };
    section("Calls out", out, "out");
    section("Called in", inn, "in");
  }

  function jumpTo(m) {
    if (S.layout.rm[4 * m + 2] <= 0) { S.view = S.map.rooms[m].region; relayout(); }
    S.sel = m; renderPanel(); draw();
  }

  const pointer = (ev) => { const b = canvas.getBoundingClientRect(); return [(ev.clientX - b.left - S.cam.x) / S.cam.k, (ev.clientY - b.top - S.cam.y) / S.cam.k]; };

  canvas.addEventListener("mousemove", (ev) => {
    if (drag) { S.cam.x += ev.movementX; S.cam.y += ev.movementY; drag.moved = true; return draw(); }
    if (!S.map) return;
    const [x, y] = pointer(ev), h = hitTest(S.idx, S.layout, x, y), tip = $("tip");
    if (h.room !== S.hover) { S.hover = h.room; draw(); }
    if (h.room >= 0) {
      const r = S.map.rooms[h.room];
      tip.textContent = roomPath(h.room) + " | " + r.lines + " lines | out " + r.out + " | in " + r.in;
    } else tip.textContent = h.region >= 0 ? S.map.regions[h.region].path || "(root)" : "";
  });
  canvas.addEventListener("mousedown", () => { drag = { moved: false }; });
  window.addEventListener("mouseup", () => { setTimeout(() => { drag = null; }, 0); });
  canvas.addEventListener("click", (ev) => {
    if (!S.map || (drag && drag.moved)) return;
    const [x, y] = pointer(ev), h = hitTest(S.idx, S.layout, x, y);
    if (h.room >= 0) { S.sel = h.room; renderPanel(); draw(); }
    else if (h.region >= 0 && h.region !== S.view) { S.view = h.region; relayout(); draw(); }   // zoom into the clicked folder
  });
  canvas.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const b = canvas.getBoundingClientRect(), px = ev.clientX - b.left, py = ev.clientY - b.top;
    const f = ev.deltaY < 0 ? 1.2 : 1 / 1.2, k = Math.min(60, Math.max(1, S.cam.k * f)), real = k / S.cam.k;
    S.cam.x = px - (px - S.cam.x) * real; S.cam.y = py - (py - S.cam.y) * real; S.cam.k = k;
    if (k === 1) { S.cam.x = 0; S.cam.y = 0; }
    draw();
  }, { passive: false });

  $("links").addEventListener("change", (ev) => { S.links = ev.target.checked; draw(); });
  $("search").addEventListener("input", (ev) => {
    S.query = ev.target.value.trim().toLowerCase();
    if (!S.map) return;
    if (!S.query) S.matches = null;
    else {
      if (!S.paths) S.paths = S.map.rooms.map((_, m) => roomPath(m).toLowerCase());
      S.matches = new Set();
      for (let m = 0; m < S.paths.length; m++) if (S.paths[m].includes(S.query)) S.matches.add(m);
    }
    $("hits").textContent = S.matches ? S.matches.size + (S.matches.size === 1 ? " match" : " matches") : "";
    draw();
  });
  window.addEventListener("resize", resize);
  resize();
  loadIndex();
}

if (typeof document !== "undefined" && document.getElementById("map")) startPage();

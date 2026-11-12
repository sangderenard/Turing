// node tests/worldmap_layout_check.mjs : the pure treemap layout of examples/meowdoku/web/worldmap.js
import assert from "node:assert/strict";
import { buildIndex, layoutMap, hitTest, buildAdjacency } from "../examples/meowdoku/web/worldmap.js";

function synth(nRooms, seed = 7) {
  let s = seed;
  const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  const regions = [{ id: 0, path: "", parent: -1 }];
  for (let i = 1; i < 300; i++) regions.push({ id: i, path: "d" + i, parent: Math.floor(rnd() * i) });
  const rooms = [];
  for (let i = 0; i < nRooms; i++) rooms.push({ id: i, region: Math.floor(rnd() * regions.length), name: "f" + i, language: "py", bytes: 1 + Math.floor(rnd() * rnd() * 50000), lines: 10, out: 0, in: 0 });
  const portals = [];
  for (let i = 0; i < nRooms * 3; i++) portals.push([Math.floor(rnd() * nRooms), Math.floor(rnd() * nRooms), 1, 0]);
  return { regions, rooms, portals };
}

const eps = 1e-6;
const within = (a, i, b, j) => a[4 * i] >= b[4 * j] - eps && a[4 * i + 1] >= b[4 * j + 1] - eps &&
  a[4 * i] + a[4 * i + 2] <= b[4 * j] + b[4 * j + 2] + eps && a[4 * i + 1] + a[4 * i + 3] <= b[4 * j + 1] + b[4 * j + 3] + eps;
const overlap = (a, i, b, j) => Math.min(a[4 * i] + a[4 * i + 2], b[4 * j] + b[4 * j + 2]) - Math.max(a[4 * i], b[4 * j]) > 1e-6 &&
  Math.min(a[4 * i + 1] + a[4 * i + 3], b[4 * j + 1] + b[4 * j + 3]) - Math.max(a[4 * i + 1], b[4 * j + 1]) > 1e-6;

const map = synth(1500);
const idx = buildIndex(map);
const L = layoutMap(idx, 1000, 700);
const { rr, rm } = L;

// every placed room lies inside its region; every placed child region lies inside its parent
let placed = 0;
for (const m of map.rooms) if (rm[4 * m.id + 2] > 0) { placed++; assert.ok(within(rm, m.id, rr, m.region), "room " + m.id + " outside region " + m.region); }
for (const g of map.regions.slice(1)) if (rr[4 * g.id + 2] > 0) assert.ok(within(rr, g.id, rr, g.parent), "region " + g.id + " outside parent");
assert.ok(placed > 1000, "most rooms are placed: " + placed);

// siblings (rooms and child regions of one region) never overlap; room areas are proportional to bytes within a region
for (let r = 0; r < map.regions.length; r++) {
  const cells = [];
  for (const m of idx.roomsOf[r]) if (rm[4 * m + 2] > 0) cells.push([rm, m]);
  for (const k of idx.kids[r]) if (rr[4 * k + 2] > 0) cells.push([rr, k]);
  for (let i = 0; i < cells.length; i++) for (let j = i + 1; j < cells.length; j++) assert.ok(!overlap(cells[i][0], cells[i][1], cells[j][0], cells[j][1]), "overlap in region " + r);
  const ratios = idx.roomsOf[r].filter((m) => rm[4 * m + 2] > 0).map((m) => (rm[4 * m + 2] * rm[4 * m + 3]) / map.rooms[m].bytes);
  if (ratios.length > 1) { const mx = Math.max(...ratios), mn = Math.min(...ratios); assert.ok(mx / mn < 1.02, "areas not proportional in region " + r + ": " + mx / mn); }
}
// a room's area is proportional to its bytes versus a child region's subtree bytes too
for (let r = 0; r < map.regions.length; r++) {
  const ratios = [];
  for (const m of idx.roomsOf[r]) if (rm[4 * m + 2] > 0) ratios.push((rm[4 * m + 2] * rm[4 * m + 3]) / map.rooms[m].bytes);
  for (const k of idx.kids[r]) if (rr[4 * k + 2] > 0 && idx.total[k] > 0) ratios.push((rr[4 * k + 2] * rr[4 * k + 3]) / idx.total[k]);
  if (ratios.length > 1) { const mx = Math.max(...ratios), mn = Math.min(...ratios); assert.ok(mx / mn < 1.02, "mixed areas not proportional in region " + r); }
}

// hit test returns the right room (centre of a sample of rooms) and the right region over a header
let tested = 0;
for (const m of map.rooms) {
  if (rm[4 * m.id + 2] < 2 || rm[4 * m.id + 3] < 2) continue;
  const h = hitTest(idx, L, rm[4 * m.id] + rm[4 * m.id + 2] / 2, rm[4 * m.id + 1] + rm[4 * m.id + 3] / 2);
  assert.equal(h.room, m.id); assert.equal(h.region, m.region);
  if (++tested >= 400) break;
}
assert.ok(tested > 50);
assert.deepEqual(hitTest(idx, L, -5, -5), { region: -1, room: -1 });

// a sub-view lays out only that subtree, filling the viewport
const sub = idx.kids[0].find((k) => idx.total[k] > 0);
const LS = layoutMap(idx, 800, 600, sub);
assert.equal(LS.rr[4 * sub + 2], 800); assert.equal(LS.rr[4 * sub + 3], 600);

// adjacency
const adj = buildAdjacency(map);
let outTotal = 0;
for (let m = 0; m < map.rooms.length; m++) { for (let i = adj.outStart[m]; i < adj.outStart[m + 1]; i++) assert.equal(map.portals[adj.outPortal[i]][0], m); outTotal += adj.outStart[m + 1] - adj.outStart[m]; }
assert.equal(outTotal, map.portals.length);

// scale: 6,000 rooms in under 500 ms (index + layout), 20,000 portals' adjacency
const big = synth(6000, 11); big.portals = synth(6000, 12).portals.slice(0, 20000);
const t0 = performance.now();
const bi = buildIndex(big), bl = layoutMap(bi, 1200, 800), ba = buildAdjacency(big);
const ms = performance.now() - t0;
assert.ok(ms < 500, "6000 rooms took " + ms.toFixed(0) + " ms");
assert.ok(bl.rm.length === 6000 * 4 && ba.outStart.length === 6001);
console.log("worldmap layout ok: " + placed + " rooms placed, 6000-room layout " + ms.toFixed(0) + " ms");

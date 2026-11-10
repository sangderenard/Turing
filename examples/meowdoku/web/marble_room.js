// The physical room shared by preview and photography. Only the room grows
// or its floor lowers for the photography table; subject coordinates stay put.
export function studioRoomBounds(scale = 6, floorDrop = 0) {
  if (!Number.isFinite(scale) || scale < 1 || scale > 8)
    throw new RangeError("Studio room scale must be between 1 and 8");
  if (!Number.isFinite(floorDrop) || floorDrop < 0 || floorDrop > 10)
    throw new RangeError("Studio floor drop must be between 0 and 10");
  return {leftX: -4 * scale, rightX: 4 * scale,
    backZ: -3.1 * scale, frontZ: 4 * scale,
    floorY: -1.02 - floorDrop, ceilingY: -1.02 + 4.02 * scale};
}

const WINDOW_X = [-1.2, 1.2], WINDOW_Y = [0.3, 2.3];

// Eight material cells per axis. Window boundaries subdivide geometry, not
// checker cells. Sharing the exact axis coordinates on all six surfaces also
// keeps every edge paired, including the corners adjacent to the rear window.
function intervals(low, high, cuts = []) {
  const points = Array.from({length: 9}, (_, i) =>
    i === 8 ? high : low + (high - low) * i / 8);
  const result = [];
  for (let i = 0; i < 8; i++) {
    const split = [points[i], ...cuts.filter(x => x > points[i] && x < points[i + 1]), points[i + 1]];
    for (let j = 0; j < split.length - 1; j++)
      result.push({low: split[j], high: split[j + 1], cell: i - 4});
  }
  return result;
}

// Faces retain the Studio mesh contract: [a,b,c,material,[na,nb,nc]]. The
// window replaces rear-wall material with glass; no face is ever omitted.
export function studioRoomFaces({scale = 6, floorDrop = 0, windowOn = false, glassMaterial, material, onSurface} = {}) {
  const {leftX, rightX, backZ, frontZ, floorY, ceilingY} = studioRoomBounds(scale, floorDrop);
  if (typeof material !== "function") throw new TypeError("Studio room material must be a function");
  if (typeof windowOn !== "boolean") throw new TypeError("Studio room window must be a boolean");
  if (windowOn && (!Number.isInteger(glassMaterial) || glassMaterial < 0))
    throw new RangeError("Studio room window needs a glass material ID");
  const xs = intervals(leftX, rightX, windowOn ? WINDOW_X : []);
  const ys = intervals(floorY, ceilingY, windowOn ? WINDOW_Y : []);
  const zs = intervals(backZ, frontZ);
  const faces = [];
  function surface(id, prefix, us, vs, point, normal, reverse = false, rear = false) {
    const first=faces.length;
    for (const u of us) for (const v of vs) {
      const cx = (u.low + u.high) / 2, cy = (v.low + v.high) / 2;
      const glass = rear && windowOn && cx >= WINDOW_X[0] && cx < WINDOW_X[1] &&
        cy >= WINDOW_Y[0] && cy < WINDOW_Y[1];
      const id = glass ? glassMaterial : material(prefix, u.cell, v.cell);
      const a = point(u.low, v.low), b = point(u.high, v.low);
      const c = point(u.high, v.high), d = point(u.low, v.high);
      const ns = [normal, normal, normal];
      if (reverse) faces.push([a, c, b, id, ns], [a, d, c, id, ns]);
      else faces.push([a, b, c, id, ns], [a, c, d, id, ns]);
    }
    onSurface?.({id,bodyId:"room",construction:"inward-room-plane",closed:false,convex:true,faces:faces.slice(first)});
  }
  surface("back","wall", xs, ys, (x, y) => [x, y, backZ], [0, 0, 1], false, true);
  surface("front","wall", xs, ys, (x, y) => [x, y, frontZ], [0, 0, -1], true);
  surface("left","wall", zs, ys, (z, y) => [leftX, y, z], [1, 0, 0], true);
  surface("right","wall", zs, ys, (z, y) => [rightX, y, z], [-1, 0, 0]);
  surface("floor","floor", xs, zs, (x, z) => [x, floorY, z], [0, 1, 0], true);
  surface("ceiling","ceiling", xs, zs, (x, z) => [x, ceilingY, z], [0, -1, 0]);
  return faces;
}

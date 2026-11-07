// THE WORLD ZOOM: how much of a scrolling game's world the scene camera shows.
//
// Render-only.  Zoom 1 shows the field (the screen the law was written for); below 1 packs more world into the same
// screen, above 1 narrows it.  The scene objects are not touched: the existing camera is moved (app.js ``camera()``
// reads ``viewWindow`` for where the window centres), eased by a critically damped spring in log-zoom.
//
// What the zoom can show is bounded by the world the game supplies beyond its field's edges (``reach``, which the game
// declares and the server publishes in ``state.zoom``): the server widens the bodies it sends with the zoom
// (world_view.py ``view_window``), up to that reach, and the page clamps to the floor it implies.  This is the same
// text as world_view.py's, so the page and the server agree on the window.
export const ZOOM_MAX = 2.5;
const NO_REACH = { left: 0, right: 0, up: 0, down: 0 };
const reachOf = (reach) => ({ ...NO_REACH, ...(reach || {}) });
const pos = (v) => Math.max(0, Number(v) || 0);

// the least zoom at which the window still lies inside the world the game supplies
export function zoomFloor(width, height, reach) {
  const r = reachOf(reach);
  return Math.min(1, Math.max(width / (width + pos(r.left) + pos(r.right)), height / (height + pos(r.up) + pos(r.down))));
}
export function clampZoom(zoom, width, height, reach, max = ZOOM_MAX) {
  const z = Number(zoom);
  if (!Number.isFinite(z)) return 1;
  return Math.min(max, Math.max(zoomFloor(width, height, reach), z));
}
// The world rectangle (screen frame: x right from the field's left edge, y down from its top) that ``zoom`` shows.
// Zooming OUT keeps the field's centre and slides the window to stay inside the supplied world (identical to
// world_view.py); zooming IN follows ``focus`` (the ship, if given) inside the field instead of the centre.
export function viewWindow(width, height, zoom, reach, focus = null, limit = true) {
  const r = reachOf(reach), z = limit ? clampZoom(zoom, width, height, reach) : zoom;
  const w = width / z, h = height / z;
  let x0 = (width - w) / 2, y0 = (height - h) / 2;
  if (z < 1) {
    x0 = Math.min(Math.max(x0, -pos(r.left)), width + pos(r.right) - w);
    y0 = Math.min(Math.max(y0, -pos(r.up)), height + pos(r.down) - h);
  } else if (z > 1 && focus) {
    x0 = Math.min(Math.max(focus[0] - w / 2, 0), width - w);
    y0 = Math.min(Math.max(focus[1] - h / 2, 0), height - h);
  }
  return { x0, x1: x0 + w, y0, y1: y0 + h, zoom: z };
}
// the window's centre relative to the field's centre: where the camera's target moves
export function windowShift(width, height, zoom, reach, focus = null) {
  const v = viewWindow(width, height, zoom, reach, focus);
  return [(v.x0 + v.x1) / 2 - width / 2, (v.y0 + v.y1) / 2 - height / 2];
}

// A CRITICALLY DAMPED SPRING (no overshoot, no ringing; the fastest approach that does not overshoot), in closed form so
// the step is exact at any frame length: x'' = -w^2 (x - target) - 2 w x'.
export function springStep(x, v, target, dt, omega) {
  if (!(dt > 0)) return { x, v };
  const d = x - target, e = Math.exp(-omega * dt), b = v + omega * d;
  return { x: target + (d + b * dt) * e, v: (v - omega * b * dt) * e };
}
export const ZOOM_OMEGA = 9;           // 1/s: settles within about half a second

// the wheel's log-zoom per pixel of delta (a notch of 100 px is a tenth of an octave... of the log scale)
export const wheelLog = (deltaY, unit = 1) => deltaY * unit * 0.0012;

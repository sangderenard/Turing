// [STUDIO] The recording parameters of the loading animation (pure: no DOM, no GPU, so node holds it in a test).
// loading_record.js reads them; the studio page (studio.js) edits them and hands them to the recorder in the page's query string
// (`?record_loading=1&rp=<json>`).  A bare `?record_loading=1` is DEFAULTS, i.e. what the recorder always did.
export const BITS = [10, 11, 12, 16];
export const COLOUR_SOURCES = [[2, "time strata"], [0, "trig (atan2)"], [1, "rotation"]];      // fountain.py `mode` (fountain_mode)
// fountain.py SIZES.  The studio further restricts this list to entries present in the deployed manifest.
export const SIZES = ["small", "4k", "16k", "64k", "256k", "1M", "4M", "10M"];
export const SIZE_IDS = { small: 256, "4k": 4096, "16k": 16384, "64k": 65536,
  "256k": 262144, "1M": 1048576, "4M": 4194304, "10M": 10000000 };
export const CHAMBERS = [                                                                     // the 2-D chambers (fountain.py CHAMBER_CHOICES)
  ["sdf_hourglass", "hourglass"], ["sdf_circle", "circle"], ["sdf_square", "square"], ["sdf_triangle", "triangle"], ["sdf_annulus", "annulus"],
  ["box", "box (open basin)"], ["funnel", "funnel"], ["hexagon", "hexagon"], ["ramp", "ramp"], ["ngon5", "pentagon"], ["ngon7", "heptagon"], ["sdf_toy", "oil & glycerine toy"],
];
export const RAW_CAP = 1 << 20;                  // the server's cap on an uploaded raw stream (loading_anim.py MAX_BYTES)
export const BUDGET = 900 * 1024;                // the budget the legacy recorder fits to
export const LADDER = [[12, 400, 12], [10, 300, 11], [8, 256, 10], [6, 200, 10]];          // [fps, grains, bits] tried in turn when fitting

export const DEFAULTS = Object.freeze({
  chamber: "sdf_hourglass", size: "small",
  mix: null,                       // fountain `mix` (poly mix, 0..1); null: the game's own
  grainScale: 1,                   // multiplies every species radius (r0..r3)
  grains: 400,                     // the reduced grains kept in the animation
  fps: 12, bits: 12,               // the kept frames per simulated second; position bits kept (the stream stays u16, low bits zero)
  settle: 40,                      // the settle threshold: kinetic energy per id, in multiples of the law's `settle.ke_per_id`
  turns: 4,                        // case turns (even: the animation ends in the pose it began in)
  flipSeconds: 0.4,                // how swift the turn is (simulated seconds)
  flipDirection: 1,                // +1 or -1 about the selected case axis
  flipAxis: "y",                  // world Y: case normal (2-D); world X/Z use the 3-D hourglass law
  speed: 0,                        // the fountain speed (0: as fast as the device allows)
  cap: 240,                        // the duration cap, simulated seconds
  colour: 2,                       // fountain colour mode (preview and metadata: the stream carries positions only)
  fit: true,                       // tighten the ladder until the raw stream fits the budget (the studio turns this off to keep what you chose)
});

const num = (v, lo, hi, d) => { v = Number(v); return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d; };
const pick = (v, list, d) => (list.includes(v) ? v : d);

// Anything -> a complete, clamped parameter object.  Unknown keys are dropped.
export function normalize(raw = {}) {
  const r = raw && typeof raw === "object" ? raw : {}, D = DEFAULTS;
  const turns = Math.round(num(r.turns, 2, 12, D.turns));
  return {
    chamber: pick(r.chamber, CHAMBERS.map((c) => c[0]), D.chamber),
    size: pick(r.size, SIZES, D.size),
    mix: r.mix === null || r.mix === undefined || r.mix === "" ? null : num(r.mix, 0, 1, 0),
    grainScale: num(r.grainScale, 0.5, 2, D.grainScale),
    grains: Math.round(num(r.grains, 16, 1024, D.grains)),
    fps: Math.round(num(r.fps, 2, 30, D.fps)),
    bits: pick(Number(r.bits), BITS, D.bits),
    settle: num(r.settle, 1, 400, D.settle),
    turns: turns - (turns % 2),                      // even
    flipSeconds: num(r.flipSeconds, 0.1, 2, D.flipSeconds),
    flipDirection: pick(Number(r.flipDirection), [-1, 1], D.flipDirection),
    flipAxis: pick(r.flipAxis, ["x", "y", "z"], D.flipAxis),
    speed: pick(Number(r.speed), [0, 1, 2, 4], D.speed),
    cap: num(r.cap, 10, 900, D.cap),
    colour: pick(Number(r.colour), COLOUR_SOURCES.map((c) => c[0]), D.colour),
    fit: r.fit === undefined ? D.fit : !!r.fit,
  };
}

// `?record_loading=1[&studio=1][&rp=<json>]` -> { on, studio, params }.  A malformed rp is the defaults.
export function fromQuery(search) {
  const q = new URLSearchParams(search || "");
  let raw = {};
  try { raw = JSON.parse(q.get("rp") || "{}"); } catch { raw = {}; }
  return { on: q.get("record_loading") === "1", studio: q.get("studio") === "1", params: normalize(raw) };
}

export function toQuery(params, { studio = true } = {}) {
  const q = new URLSearchParams({ record_loading: "1" });
  if (studio) q.set("studio", "1");
  q.set("rp", JSON.stringify(normalize(params)));
  return "?" + q.toString();
}

// The attempts the recorder makes for `params`: with `fit`, the chosen setting first and then the tighter rungs of LADDER (never a
// looser one); without it, only what was chosen.  Each is [fps, grains, bits].
export function attempts(params) {
  const p = normalize(params), first = [p.fps, p.grains, p.bits];
  if (!p.fit) return [first];
  const rest = LADDER.filter(([f, g, b]) => f <= p.fps && g <= p.grains && b <= p.bits && !(f === p.fps && g === p.grains && b === p.bits));
  return [first, ...rest];
}

// The metadata that goes with an upload (flat: the server keeps number / bool / short string values).
export function metadata(params, extra = {}) {
  const p = normalize(params);
  return { ...p, mix: p.mix === null ? "default" : p.mix, ...extra };
}

// A planning estimate of the raw stream: header + per-grain radius + per-frame angle + the first frame + the delta frames, 2 bytes per value.
export function rawEstimate(params, frames) {
  const p = normalize(params);
  return 30 + 2 * p.grains + 2 * frames + 4 * p.grains * frames;
}

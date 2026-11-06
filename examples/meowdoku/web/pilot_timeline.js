// The page's flight of its own ship by the engine's laws (the compiled cores in ``pred``), and the pilot's
// interval timeline.  No DOM here: app.js imports it, and so does the headless check that holds the page's
// autopilot prediction against the server's.
export const PRED_STEP = 1 / 240;            // the predictor's own rate (PREDICT_HZ)

// ``pred``: { program, bankTurn, turn, advance, air, travel } wasm cores; ``view``: () => the camera { x }.
export function shipStepper(pred, view) {
  return function stepShip(ship, input, seconds, motion, stepSize = PRED_STEP) {
    for (let left = seconds; left > 1e-9;) {
      const dt = Math.min(stepSize, left); left -= dt;
      const yawBefore = ship.yaw, zBefore = ship.z || 0;
      for (const [axis, pos, vel, move, target] of [["x", "x", "vx", "move_axis", "steer_x"], ["y", "y", "vy", "move_axis_y", "steer_y"],
                                                    ["z", "z", "vz", "move_axis_z", "steer_z"],
                                                    ["yaw", "yaw", "vyaw", "move_axis", null], ["spd", "spd", "vspd", "move_axis_y", null],
                                                    ["bank", "bank", "vbank", "move_axis", null]]) {
        // Only the engine's own motion law moves the ship here (the same
        // damped_axis, compiled to wasm).  An axis that law does not declare is
        // the server's alone: the page never guesses it with physics of its own.
        if (!motion[axis]) continue;
        const bind = motion.bind && motion.bind[axis];
        if (bind) {
          // A bound axis (a spherical game's turn, throttle): its own input,
          // sign and rates; a turn's held finger steers for its bearing from
          // the ship, as the law computes it.
          const patch = motion.patch || {};
          const steering = bind.finger && input.steer_x >= 0;
          const idle = Math.abs(input[bind.move] || 0) < 0.05;
          // the bearing's dead zone is the law's (bind.finger_dead): straight
          // ahead steers nothing, past the edge it counts from the edge
          const bearing = steering ? Math.atan2(input.steer_x - patch.cx, patch.cy - input.steer_y) : 0;
          const dead = bind.finger_dead || 0;
          const finger = Math.abs(bearing) < dead ? 0 : bearing - Math.sign(bearing) * dead;
          const aim = steering ? ship[pos] + finger
            : bind.level != null && idle ? bind.level : -1;          // a released stick returns the axis to its level
          const r = pred.program.invoke({ x: ship[pos], v: ship[vel], axis: (bind.sign ?? 1) * (input[bind.move] || 0), target: aim,
            lo: motion[axis][0], hi: motion[axis][1], speed: bind.speed ?? motion.speed, tau: bind.tau ?? motion.tau,
            reach: bind.reach ?? motion.reach, dt });
          ship[pos] = r.x_next; ship[vel] = r.v_next;
          continue;
        }
        // The walls ride the view in a world-frame game (the law's own bounds).
        const cam = axis === "x" && motion.camera, base = cam ? Math.max(view().x, cam.x) : 0;
        const lo = cam ? base + cam.margin : motion[axis][0], hi = cam ? base + cam.width - cam.margin : motion[axis][1];
        const r = pred.program.invoke({ x: ship[pos], v: ship[vel], axis: input[move] || 0, target: input[target] ?? -1,
          lo, hi, speed: motion.speed, tau: motion.tau, reach: motion.reach, dt });
        ship[pos] = r.x_next; ship[vel] = r.v_next;
      }
      // A banked craft turns: the engine's bank_turn, after the axes, as the law does.
      if (motion.bank_turn && pred.bankTurn && ship.bank != null) {
        ship.yaw = pred.bankTurn.invoke({ yaw: ship.yaw, bank: ship.bank, level: motion.bank_turn.level,
                                          gain: motion.bank_turn.gain, dt }).yaw_next;
      }
      // The ship's flight on the sphere: the engine's sphere_turn by this
      // step's yaw, then sphere_advance by its speed in the air at its height.
      if (motion.sphere && pred.advance && pred.turn && pred.air && ship.ux != null) {
        const t = pred.turn.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: ship.hx, hy: ship.hy, hz: ship.hz,
                                     angle: (ship.yaw ?? 0) - (yawBefore ?? 0) });
        const air = pred.air.invoke({ z: zBefore, fall: motion.sphere.air[0], gain: motion.sphere.air[1] }).factor;
        const a = pred.advance.invoke({ ux: ship.ux, uy: ship.uy, uz: ship.uz, hx: t.hx_next, hy: t.hy_next, hz: t.hz_next,
                                        distance: (ship.spd || 0) * dt * air, radius: motion.sphere.radius });
        ship.ux = a.ux_next; ship.uy = a.uy_next; ship.uz = a.uz_next; ship.hx = a.hx_next; ship.hy = a.hy_next; ship.hz = a.hz_next;
      }
      // The land under the ship: the engine's ground_travel, as the law steps it.
      if (motion.travel && pred.travel && ship.map_gy != null) {
        const g = pred.travel.invoke({ u: ship.map_u, v: ship.map_v, gx: ship.map_gx, gy: ship.map_gy,
                                       turn: (ship.yaw ?? 0) - (yawBefore ?? 0), speed: ship.spd || 0, dt });
        ship.map_u = g.u_next; ship.map_v = g.v_next; ship.map_gx = g.gx_next; ship.map_gy = g.gy_next;
      }
    }
  }
}

// THE PILOT'S INTERVAL TIMELINE.  The law's pilot decides on its own slow clock (scroller.PILOT_INTERVAL) and a
// decision takes effect PILOT_PREDELAY later, so every interval {seq, start, length, controls} reaches the page
// before its start, keyed to LAW time.  The controls in effect at law time t are those of the latest interval
// whose start is <= t (held until the next one starts).  The page applies each over exactly its span of law time
// with the same laws (``integrate``), so it needs no guess of the server's controls and no latency estimate.
export const pilotControls = (c) => ({ move_axis: c[0], move_axis_y: c[1], move_axis_z: c[2], move_axis_r: c[3], fire: c[4], steer_x: -1, steer_y: -1 });
export class PilotTimeline {
  constructor() { this.items = []; this.late = 0; }
  clear() { this.items = []; }
  // merge a snapshot's records (by seq); ``at`` the snapshot's law time: one that starts before it arrived late
  accept(records, at) {
    for (const r of records || []) {
      if (this.items.some((i) => i.seq === r.seq)) continue;
      if (r.start < at - 1e-9) this.late++;
      this.items.push(r);
    }
    this.items.sort((a, b) => a.start - b.start);
    // forget what ended well before ``at`` (keep the one in effect then)
    let keep = 0;
    while (keep + 1 < this.items.length && this.items[keep + 1].start <= at - 2) keep++;
    if (keep) this.items.splice(0, keep);
  }
  inEffect(t) {
    let found = null;
    for (const i of this.items) { if (i.start <= t + 1e-12) found = i; else break; }
    return found;
  }
  // The ship from law time t0 to t1, each interval over exactly its span, in steps that end where the SERVER's
  // do: the server lands its ticks as windows (the spacing of the snapshots' law clocks, ``window``) and cuts a
  // step where a decision takes effect (scroller's dt_limit); the laws are discretised, so the page steps the
  // same way from the snapshot it starts from (t0).  ``rest`` before any interval is known.
  integrate(stepShip, ship, t0, t1, motion, rest, window = PRED_STEP) {
    let t = t0;
    while (t1 - t > 1e-12) {
      const now = this.inEffect(t);
      const next = this.items.find((i) => i.start > t + 1e-12);
      const boundary = t0 + (Math.floor((t - t0) / window + 1e-9) + 1) * window;
      const end = Math.min(t1, boundary, next ? next.start : Infinity);
      stepShip(ship, now ? pilotControls(now.controls) : rest, end - t, motion, 1);
      t = end;
    }
  }
}

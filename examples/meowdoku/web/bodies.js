// The snapshot frame (realtime_socket.py: PROTOCOL) read, and the bodies of a
// spherical game carried on by the engine's laws.  No DOM here: the page
// imports it, and so does the headless check that holds it against the server.
//
// The page never moves a body with arithmetic of its own.  A body the server
// sends arrives with the state its step flew it from -- place and heading on
// the sphere, ground speed, the rate its heading was turned at, height and
// climb rate, and the flight that names the laws -- and is carried on between
// snapshots by those laws, the compiled cores of entity_motion.PREDICTORS:
// sphere_turn, sphere_advance (a craft's by air_gain), climb, or the
// trajectory step of a missile or a bomb; then projected into the patch about
// the page's own ship by sphere_project.

export const SNAPSHOT_PROTOCOL = 13;          // realtime_socket.PROTOCOL
export const FLAG_KEYFRAME = 1, FLAG_CLEAR = 2, FLAG_BODIES = 4, FLAG_PILOTED = 8;

// One frame: { version } when the layout is another version, else the header,
// the lanes gone and the records spawned and updated.  ``names`` is the
// connection's appearance-name table, which the frame may extend.
export function decodeSnapshot(view, names) {
  let o = 0;
  const u8 = () => view.getUint8(o++), u16 = () => { const v = view.getUint16(o, true); o += 2; return v; };
  const i16 = () => { const v = view.getInt16(o, true); o += 2; return v; }, f32 = () => { const v = view.getFloat32(o, true); o += 4; return v; };
  const i32 = () => { const v = view.getInt32(o, true); o += 4; return v; };
  const u32 = () => { const v = view.getUint32(o, true); o += 4; return v; }, f64 = () => { const v = view.getFloat64(o, true); o += 8; return v; };
  const version = u8();
  if (version !== SNAPSHOT_PROTOCOL) return { version };
  const flags = u8(), ack = i32();
  const you = { x: f32(), y: f32(), vx: f32(), vy: f32(), alive: u8() === 1, shield: f32() };
  const hint = f32();
  you.z = f32(); you.vz = f32(); you.health = f32();
  you.yaw = f32(); you.vyaw = f32(); you.spd = f32(); you.vspd = f32();
  you.map_u = f32(); you.map_v = f32(); you.map_gx = f32(); you.map_gy = f32();
  you.bank = f32(); you.vbank = f32(); you.beam = f32();
  you.ux = f32(); you.uy = f32(); you.uz = f32(); you.hx = f32(); you.hy = f32(); you.hz = f32();
  you.pilot = [f32(), f32(), f32(), f32(), f32()];
  // the tracking cannon: the locked lane (0 none) and the gun mount's offsets from the nose
  you.lock = f32(); you.mount_yaw = f32(); you.mount_vyaw = f32(); you.mount_pitch = f32(); you.mount_vpitch = f32();
  you.coy = f32(); you.buf = f32(); you.pf = f32();       // a platformer's vertical_axis state
  // the law time of this state, and the pilot's interval records (seq, law start, length, the five controls):
  // each arrives before its start (scroller.PILOT_PREDELAY)
  you.clock = f64();
  // the timing exchange (pilot_clock.js): the server wall instant this state stands for, the page's send time of the tick
  // this answers (NaN: none), the server's wall at its receipt and as the reply was cut
  you.lawWall = f64(); you.echo = f64(); you.recv = f64(); you.sent = f64();
  you.piloted = (flags & FLAG_PILOTED) !== 0;
  you.intervals = [];
  for (let n = u8(); n > 0; n--) { const seq = u32(), start = f64(), length = f32(); you.intervals.push({ seq, start, length, controls: [f32(), f32(), f32(), f32(), f32()] }); }
  for (let n = u8(); n > 0; n--) { const len = u8(); names.push(new TextDecoder().decode(new Uint8Array(view.buffer, view.byteOffset + o, len))); o += len; }
  // (a body's flight state: u8 flight, f32 u (3), h (3), spd, turn)
  const body = () => (flags & FLAG_BODIES) ? { flight: u8(), ux: f32(), uy: f32(), uz: f32(), hx: f32(), hy: f32(), hz: f32(), spd: f32(), turn: f32() } : null;
  const gone = []; for (let n = u16(); n > 0; n--) gone.push(u16());
  const spawned = [];
  for (let n = u16(); n > 0; n--) {
    const lane = u16(), mesh = names[u8()], color = names[u8()], size = u16() / 1000, roll = i16() / 1000, spin = i16() / 1000;
    const look = u8(), lean = i16() / 1000, owner = u16(), age = f32();
    const x = f32(), y = f32(), vx = f32(), vy = f32(), z = f32(), vz = f32();
    spawned.push({ lane, mesh, color, size, roll, spin, bob: look & 1, face: (look & 2) !== 0, translucent: (look & 4) !== 0,
                   ribbon: (look & 8) !== 0, ground: (look & 16) !== 0, glow: (look & 32) !== 0, orient: (look & 64) !== 0, scenery: (look & 128) !== 0,
                   lean, owner, age, x, y, vx, vy, z, vz, body: body() });
  }
  const updated = [];
  for (let n = u16(); n > 0; n--) {
    const lane = u16(), mesh = names[u8()], color = names[u8()], roll = i16() / 1000;
    const x = f32(), y = f32(), vx = f32(), vy = f32(), z = f32(), vz = f32();
    updated.push({ lane, mesh, color, roll, x, y, vx, vy, z, vz, body: body() });
  }
  return { version, flags, ack, you, hint, gone, spawned, updated };
}

// The cores a page needs to carry bodies on, loaded by ``load`` (stem -> program).
export const BODY_CORES = ["turn", "advance", "air", "climb", "ballistic", "project", "heading"];

export class BodyFlight {
  constructor(cores) { this.k = cores; }

  static ready(cores) { return BODY_CORES.every((stem) => cores && cores[stem]); }

  // A body as the page holds it after a record: the flight state, the height
  // and climb rate (the record's), the world time it stands for.
  static held(record, at) {
    const b = record.body;
    return { flight: b.flight, ux: b.ux, uy: b.uy, uz: b.uz, hx: b.hx, hy: b.hy, hz: b.hz, spd: b.spd, turn: b.turn,
             z: record.z, vz: record.vz, at };
  }

  // Carry ``b`` on ``seconds`` of world time, the way the server's step flies a
  // body: the heading turned by the rate it was last sent at, the body flown
  // along its great circle (a craft's distance by the air at its height) and
  // climbed; a missile or a bomb on the trajectory step instead (drag and
  // gravity), whose horizontal advance is flown along the heading and whose
  // speed, height and climb come from the law.  In sub-steps no longer than the
  // server's step cap (``bodies.step``).
  carry(b, seconds, bodies, sphere) {
    if (!(seconds > 0)) return b;
    const n = Math.max(1, Math.min(60, Math.ceil(seconds / bodies.step - 1e-9)));
    const dt = seconds / n, F = bodies.flight, ball = bodies.ballistic;
    for (let i = 0; i < n; i++) {
      const t = this.k.turn.invoke({ ux: b.ux, uy: b.uy, uz: b.uz, hx: b.hx, hy: b.hy, hz: b.hz, angle: b.turn * dt });
      let distance, z, spd = b.spd, vz = b.vz;
      if (b.flight === F.missile || b.flight === F.bomb || b.flight === F.canopy) {
        const [diameter, drag] = b.flight === F.missile ? ball.missile : b.flight === F.bomb ? ball.bomb : ball.canopy;
        const r = this.k.ballistic.invoke({ position_x_m: 0, position_y_m: b.z, position_z_m: 0,
          velocity_x_m_s: b.spd, velocity_y_m_s: b.vz, velocity_z_m_s: 0, dt, mass_kg: ball.mass, diameter_m: diameter,
          drag_coefficient: drag, air_density_kg_m3: ball.air_density, gravity_m_s2: ball.gravity,
          wind_x_m_s: 0, wind_y_m_s: 0, wind_z_m_s: 0 });
        distance = r.position_next_x_m; spd = r.velocity_next_x_m_s; z = r.position_next_y_m; vz = r.velocity_next_y_m_s;
      } else {
        const air = b.flight === F.craft ? this.k.air.invoke({ z: b.z, fall: sphere.air[0], gain: sphere.air[1] }).factor : 1;
        distance = b.spd * dt * air;
        z = this.k.climb.invoke({ z: b.z, vz: b.vz, dt }).z_next;
      }
      const a = this.k.advance.invoke({ ux: b.ux, uy: b.uy, uz: b.uz, hx: t.hx_next, hy: t.hy_next, hz: t.hz_next, distance, radius: sphere.radius });
      b.ux = a.ux_next; b.uy = a.uy_next; b.uz = a.uz_next; b.hx = a.hx_next; b.hy = a.hy_next; b.hz = a.hz_next;
      b.spd = spd; b.z = z; b.vz = vz;
    }
    return b;
  }

  // The body in the patch about ``ship`` (its place and heading on the sphere):
  // where sphere_project puts it, and how sphere_heading turns it.
  patch(b, ship, patch, radius) {
    const p = this.k.project.invoke({ ux: b.ux, uy: b.uy, uz: b.uz, cx: ship.ux, cy: ship.uy, cz: ship.uz,
                                    hx: ship.hx, hy: ship.hy, hz: ship.hz, radius });
    return { x: patch.cx + p.X, y: patch.cy - p.Y, z: b.z };
  }

  heading(b, ship) {
    return this.k.heading.invoke({ hx: b.hx, hy: b.hy, hz: b.hz, cx: ship.ux, cy: ship.uy, cz: ship.uz,
                                  gx: ship.hx, gy: ship.hy, gz: ship.hz }).angle;
  }
}

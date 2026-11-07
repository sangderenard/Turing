// The phone's gravity, as the law's runtime input.
//
// `downInDevice` turns the browser's orientation angles into the direction of gravity in the DEVICE frame (x to the
// right of the screen, y to its top, z out of it); `gravityInScreen` rotates that by the screen's own orientation and
// keeps the part IN the screen's plane -- the fountain is a 2-D world drawn on the screen, so a phone lying flat has no
// pull in it and an upright one has the full pull.  Both are pure (a test holds them in node).
//
// `GravitySensor` is the live source.  It is ON by default and has an off switch.  iOS only delivers the events after
// `requestPermission()` was called from a user gesture, so `grant()` is to be called from the first tap.  The values go
// to `onGravity(gx, gy)` as fractions of g in the screen's plane: gravity = (gx, gy) * G in the law's own units.
const RAD = Math.PI / 180;

export function downInDevice(beta, gamma) {
  const b = beta * RAD, g = gamma * RAD;
  return [Math.sin(g) * Math.cos(b), -Math.sin(b), -Math.cos(g) * Math.cos(b)];
}

// rotate a device-frame vector into the screen frame; `angle` is screen.orientation.angle (degrees)
export function toScreen([x, y, z], angle = 0) {
  const a = angle * RAD, c = Math.cos(a), s = Math.sin(a);
  return [x * c - y * s, x * s + y * c, z];
}

export function gravityInScreen(beta, gamma, angle = 0) {
  const v = toScreen(downInDevice(beta, gamma), angle);
  return [v[0], v[1]];
}

// DeviceMotion fallback: `accelerationIncludingGravity` is the reaction to gravity on Android (+y when upright) and its
// opposite on iOS; either way gravity points the other way from the reaction.
export function gravityFromAcceleration(ax, ay, az, angle = 0, reaction = true) {
  const m = Math.hypot(ax, ay, az) || 1, sign = reaction ? -1 : 1;
  const v = toScreen([sign * ax / m, sign * ay / m, sign * az / m], angle);
  return [v[0], v[1]];
}

// screen.orientation.angle where there is one (iOS before 16.4 has only window.orientation)
export function screenAngle(w) {
  if (!w) return 0;
  const a = (w.screen && w.screen.orientation && w.screen.orientation.angle) ?? w.orientation ?? 0;
  return ((Number(a) % 360) + 360) % 360;
}

// How the page holds its layout while the phone turns: the CSS transform that turns content laid out at ``lockedAngle``
// (``w`` x ``h`` CSS px) so it stays fixed to the device when the screen is at ``angle``.  The browser rotated the
// content by the turn; this turns it back (CSS positive = clockwise; a screen angle grows counterclockwise), then
// shifts it back on screen.  ``null``: no turn, nothing to do.
export function layoutTransform(lockedAngle, angle, w, h) {
  const d = (((angle - lockedAngle) % 360) + 360) % 360;
  if (d === 0) return null;
  if (d === 90) return `translate(0px, ${w}px) rotate(-90deg)`;
  if (d === 180) return `translate(${w}px, ${h}px) rotate(180deg)`;
  return `translate(${h}px, 0px) rotate(90deg)`;
}

export class GravitySensor {
  constructor({ onGravity, win = typeof window === "undefined" ? null : window, onState = () => {} }) {
    this.onGravity = onGravity; this.win = win; this.onState = onState;
    this.wanted = true; this.state = "idle"; this.listening = false; this.last = null; this.lockedAngle = null;
    this.ios = !!(win && win.DeviceOrientationEvent && typeof win.DeviceOrientationEvent.requestPermission === "function");
    this.haveOrientation = false;
    this.orient = (e) => {
      if (e.beta == null || e.gamma == null) return;
      this.haveOrientation = true;
      this.#emit(gravityInScreen(e.beta, e.gamma, this.#angle()));
    };
    this.motion = (e) => {
      const a = e.accelerationIncludingGravity;
      if (this.haveOrientation || !a || a.x == null) return;
      this.#emit(gravityFromAcceleration(a.x, a.y, a.z, this.#angle(), !this.ios));
    };
  }

  // The angle of the frame the world is drawn in: the screen's, unless the page holds its layout fixed against a turn of
  // the phone (``lockedAngle``, see app.js layoutLock) -- then the world turns with the device and gravity is read in it.
  #angle() {
    if (this.lockedAngle != null) return this.lockedAngle;
    return screenAngle(this.win);
  }

  #emit(g) {
    this.last = g;
    if (this.state !== "on") { this.state = "on"; this.onState(this.state); }
    this.onGravity(g[0], g[1]);
  }

  // the first tap (iOS needs it); a no-op elsewhere.  Returns the resulting state.
  // Both iOS requests are made SYNCHRONOUSLY inside the gesture: a request issued after an await has lost the tap's user
  // activation and throws, which used to fail the whole grant (the switch then had to be cycled to try again).  The
  // orientation answer decides; the motion one is only the fallback source.
  async grant() {
    if (!this.wanted || this.listening || !this.win) return this.state;
    if (this.ios) {
      const W = this.win, ask = (E) => {
        try { return Promise.resolve(E && typeof E.requestPermission === "function" ? E.requestPermission() : "absent"); }
        catch (error) { return Promise.resolve("error"); }
      };
      const orientation = ask(W.DeviceOrientationEvent), motion = ask(W.DeviceMotionEvent);
      const [o, m] = await Promise.all([orientation.catch(() => "error"), motion.catch(() => "error")]);
      if (o === "denied" || (o !== "granted" && m !== "granted")) { this.state = "denied"; this.onState(this.state); return this.state; }
    }
    this.#listen();
    return this.state;
  }

  #listen() {
    if (this.listening || !this.win) return;
    this.win.addEventListener("deviceorientation", this.orient);
    this.win.addEventListener("devicemotion", this.motion);
    this.listening = true; this.state = this.state === "on" ? "on" : "waiting"; this.onState(this.state);
  }

  // On everywhere but iOS needs no gesture: start listening at once when allowed.
  start() { if (this.wanted && !this.ios) this.#listen(); }

  set enabled(on) {
    this.wanted = !!on;
    if (!on) { this.state = "off"; this.last = null; this.onState(this.state); this.onGravity(null, null); }
    else { this.state = "idle"; this.onState(this.state); this.listening ? (this.state = "waiting") : this.start(); }
  }
  get enabled() { return this.wanted; }
}

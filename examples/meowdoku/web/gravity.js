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

export class GravitySensor {
  constructor({ onGravity, win = typeof window === "undefined" ? null : window, onState = () => {} }) {
    this.onGravity = onGravity; this.win = win; this.onState = onState;
    this.wanted = true; this.state = "idle"; this.listening = false; this.last = null;
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

  #angle() {
    const w = this.win;
    return (w.screen && w.screen.orientation && w.screen.orientation.angle) ?? w.orientation ?? 0;
  }

  #emit(g) {
    this.last = g;
    if (this.state !== "on") { this.state = "on"; this.onState(this.state); }
    this.onGravity(g[0], g[1]);
  }

  // the first tap (iOS needs it); a no-op elsewhere.  Returns the resulting state.
  async grant() {
    if (!this.wanted || this.listening || !this.win) return this.state;
    try {
      if (this.ios) {
        const r = await this.win.DeviceOrientationEvent.requestPermission();
        if (r !== "granted") { this.state = "denied"; this.onState(this.state); return this.state; }
        if (this.win.DeviceMotionEvent && typeof this.win.DeviceMotionEvent.requestPermission === "function") await this.win.DeviceMotionEvent.requestPermission();
      }
    } catch (error) { this.state = "denied"; this.onState(this.state); return this.state; }
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

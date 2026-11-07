// THE GAME CLOCK SPEED: law seconds landed per wall second, one value shared by the page and the server.
//
// The speed is NOT the law: dt and the law's text are the same at any speed.  It scales how many fixed steps run per wall
// second -- the server lands ``speed * wall`` seconds a tick, and the page lands ``speed * frame`` in its own prediction --
// so the page keeps ONE clock for everything that measures the game's world in seconds: ``GameClock.at(realSeconds)``,
// the wall integrated by the speed in force (the *virtual* wall).  The timing exchange the ticks carry (pilot_clock.js)
// is in that clock on both sides (entities.py ``virtual``), so the law clock and the wall it is locked to still advance
// together and a handoff at 0.1x is as smooth as at 1x.  Wall time proper stays for what is not the world: the network,
// the camera's ease, the interface.
export const SPEED = { min: 0.05, max: 2, default: 0.4 };       // world_view.py

export const clampSpeed = (v) => {
  const s = Number(v);
  return Number.isFinite(s) ? Math.min(SPEED.max, Math.max(SPEED.min, s)) : SPEED.default;
};
// the slider is logarithmic: u in [0, 1] over [min, max]
export const speedToSlider = (s) => Math.log(clampSpeed(s) / SPEED.min) / Math.log(SPEED.max / SPEED.min);
export const sliderToSpeed = (u) => clampSpeed(SPEED.min * Math.pow(SPEED.max / SPEED.min, Math.min(1, Math.max(0, Number(u)))));
export const speedLabel = (s) => `${s < 0.1 ? s.toFixed(2) : s.toFixed(s < 1 ? 2 : 1).replace(/0$/, "")}x`;

export class GameClock {
  constructor(speed = SPEED.default, real = 0) { this.speed = clampSpeed(speed); this.v = real; this.real = real; }
  // the virtual wall at the real instant ``real``
  at(real) { return this.v + this.speed * (real - this.real); }
  // change the speed at ``real``: the virtual wall is continuous across it
  set(speed, real) {
    const s = clampSpeed(speed);
    if (s === this.speed) return false;
    this.v = this.at(real); this.real = real; this.speed = s;
    return true;
  }
  // a speed of exactly 1 and no history, for games that do not take the control
  lock(real) { return this.set(1, real); }
}

// THE LAW CLOCK: the page's law time, locked to the server's.
//
// A pilot's ship is a function of LAW time (the intervals it flies are keyed to it), so what the picture shows at a
// page instant t is the ship at law time tau(t) -- and the picture is smooth exactly when tau(t) is: it runs at the
// wall's rate and never jumps.  The server's law clock advances with its own wall clock (entities.py: a tick lands the
// wall time since the previous tick began; a snapshot carries ``law_wall``, the server's wall instant of that tick),
// so the law is a fixed offset from the server's wall, and the server's wall is a fixed offset from the page's.
// This module reads the second offset the way NTP does and the first one exactly:
//
//   each tick the page stamps its send time ct; the reply carries it back (echo) with the server's receipt (recv) and
//   send (sent) instants; the page stamps its arrival ca.  Then
//       delay = (ca - ct) - (sent - recv)            the network's share (the server's compute is taken out)
//       theta = ((recv - ct) + (sent - ca)) / 2      server wall minus page wall
//   the sample with the LEAST delay in the recent window is the truest (queueing only ever adds), and
//       epoch = law_wall - clock                     the server wall instant of law time zero (exact, every reply)
//   give   tau(t) = t + theta - epoch.
//
// The two offsets move at their own bounded rates, so no single reply moves tau by more than a rate allows:
//   theta   is measured, noisily: it follows the best sample at THETA_SLEW seconds per second;
//   epoch   is exact, and changes only when the server's law and wall part company (a hand-over between a person's
//           input-timed intervals, whose clock is the page's measured durations, and the pilot's wall-timed ticks; a tick
//           the server cut short): it follows at EPOCH_SLEW, a time warp of that fraction until the clock has caught up.
// A disagreement beyond SNAP (a new game, a long stall) is taken at once, because slewing it would be a long lie.
export class LawClock {
  constructor({ window = 16, slew = 0.004, epochSlew = 0.25, snap = 0.5 } = {}) {
    this.window = window; this.slew = slew; this.epochSlew = epochSlew; this.snap = snap;
    this.samples = [];          // { theta, delay, at }: recent exchanges
    this.theta = null; this.epoch = null;
    this.offset = null;         // tau(t) = t + offset
    this.snaps = 0;             // how many times the offset was taken in one jump (a first reading counts)
    this.lastAt = null;
    this.last = null;           // the last sample, for inspection
    // THE GAME CLOCK SPEED (game_clock.js): this clock runs in the game's virtual wall.  The speed is constant between
    // changes, and a change is announced by the server's counter ``gen``: the readings before it were taken against a
    // clock that ran at another rate, so the window is dropped and the first reading after it is taken whole (``rebased``
    // says so, for the caller to absorb the jump into its time warp rather than show it).
    this.gen = null; this.rebased = false;
  }
  clear() { this.samples = []; this.offset = null; this.theta = null; this.epoch = null; this.lastAt = null; this.last = null; }
  // ``m``: { ct, ca, recv, sent, lawWall, clock }; a reply to a tick that carried no stamp (echo NaN) has no exchange to read
  sample(m) {
    this.rebased = false;
    if (m.gen !== undefined && m.gen !== this.gen) {
      if (this.gen !== null) { this.samples = []; this.offset = null; this.rebased = true; }
      this.gen = m.gen;
    }
    if (![m.ct, m.ca, m.recv, m.sent, m.lawWall, m.clock].every(Number.isFinite)) return false;
    const delay = (m.ca - m.ct) - (m.sent - m.recv);
    const s = { theta: ((m.recv - m.ct) + (m.sent - m.ca)) / 2, delay: Math.max(0, delay), at: m.ca };
    this.last = s;
    this.samples.push(s);
    if (this.samples.length > this.window) this.samples.shift();
    let best = this.samples[0];
    for (const c of this.samples) if (c.delay < best.delay) best = c;
    const epoch = m.lawWall - m.clock;
    const target = best.theta - epoch;
    this.want = { theta: best.theta, epoch, offset: target };
    if (this.offset == null || Math.abs(target - this.offset) > this.snap) {
      this.theta = best.theta; this.epoch = epoch; this.offset = target; this.snaps++;
    } else {
      const dt = this.lastAt == null ? 0 : Math.max(0, m.ca - this.lastAt);
      const follow = (now, want, rate) => now + Math.max(-rate * dt, Math.min(rate * dt, want - now));
      this.theta = follow(this.theta, best.theta, this.slew);
      this.epoch = follow(this.epoch, epoch, this.epochSlew);
      this.offset = this.theta - this.epoch;
    }
    this.lastAt = m.ca;
    return true;
  }
  // Take the newest reading whole.  For a moment when the picture is being re-based on a different bookkeeping anyway
  // (the hand-over from a person to the pilot), where following at a bounded rate would only be a long disagreement.
  takeNewest() {
    if (!this.want) return;
    this.theta = this.want.theta; this.epoch = this.want.epoch; this.offset = this.want.offset; this.snaps++;
  }
  ready() { return this.offset != null; }
  // the law time at page time t, and the page time of law time `clock`
  lawAt(t) { return t + this.offset; }
  pageAt(clock) { return clock - this.offset; }
}

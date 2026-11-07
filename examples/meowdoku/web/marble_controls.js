// Control directions are fractions of the 3-D law's gravitational acceleration.
// The course lies in law X/Y; law -Z points into its backplate. At rest the
// marble presses into that plate, and steering tips that vector within X/Y.
export function marbleGravityVector(x, down, phone = null) {
  if (phone && phone.length === 3 && phone.every(Number.isFinite)) {
    const length = Math.hypot(...phone);
    if (length > 0.01) return phone.map(v => v / length);
  }
  const lateral = Math.hypot(x, down);
  const limit = 0.95;
  const scale = lateral > limit ? limit / lateral : 1;
  const gx = x * scale, gy = down === 0 ? 0 : -down * scale;
  return [gx, gy, -Math.sqrt(1 - gx * gx - gy * gy)];
}

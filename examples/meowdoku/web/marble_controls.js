// Control directions are fractions of the 3-D law's gravitational acceleration.
// x is across the course, y is vertical, z is through its depth.
export function marbleGravityVector(x, z, phone = null) {
  if (phone && phone.length === 3 && phone.every(Number.isFinite)) {
    const length = Math.hypot(...phone);
    if (length > 0.01) return phone.map(v => v / length);
  }
  const lateral = Math.hypot(x, z);
  const limit = 0.7;
  const scale = lateral > limit ? limit / lateral : 1;
  const gx = x * scale, gz = z * scale;
  return [gx, -Math.sqrt(1 - gx * gx - gz * gz), gz];
}

// Orientation of a modeled sphere from the 3-D law's contact-solved angular velocity.
// Quaternion order is [x, y, z, w] in the law's X/Y/Z frame.
export function integrateAngular(q, omega, seconds) {
  const length = Math.hypot(...omega), angle = length * seconds;
  if (!(angle > 0) || !Number.isFinite(angle)) return [...q];
  const scale = Math.sin(angle / 2) / length;
  const [x, y, z] = omega.map((v) => v * scale), w = Math.cos(angle / 2);
  const [a, b, c, d] = q;
  const next = [w*a + x*d + y*c - z*b, w*b - x*c + y*d + z*a,
    w*c + x*b - y*a + z*d, w*d - x*a - y*b - z*c];
  const n = Math.hypot(...next);
  return next.map((v) => v / n);
}

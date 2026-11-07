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

// Allocation-free form for a displayed field of glyphs. `source` and `target`
// contain xyzw quaternions; `omega` holds xyz triples in displayed-slot order.
export function integrateAngularField(source, target, omega, seconds, count) {
  for (let i = 0; i < count; i++) {
    const q = i * 4, k = i * 3;
    const wx = omega[k], wy = omega[k + 1], wz = omega[k + 2];
    const a = source ? source[q] : 0, b = source ? source[q + 1] : 0;
    const c = source ? source[q + 2] : 0, d = source ? source[q + 3] : 1;
    const length = Math.hypot(wx, wy, wz), angle = length * seconds;
    if (!(angle > 0) || !Number.isFinite(angle)) {
      target[q] = a; target[q + 1] = b; target[q + 2] = c; target[q + 3] = d;
      continue;
    }
    const scale = Math.sin(angle / 2) / length;
    const x = wx * scale, y = wy * scale, z = wz * scale, w = Math.cos(angle / 2);
    const na = w*a + x*d + y*c - z*b, nb = w*b - x*c + y*d + z*a;
    const nc = w*c + x*b - y*a + z*d, nd = w*d - x*a - y*b - z*c;
    const inv = 1 / Math.hypot(na, nb, nc, nd);
    target[q] = na * inv; target[q + 1] = nb * inv;
    target[q + 2] = nc * inv; target[q + 3] = nd * inv;
  }
  return target;
}

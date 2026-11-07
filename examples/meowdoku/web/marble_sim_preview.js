// The Studio's editable object drives one particle in the existing fountain3d
// product. These are native law fields, independent of the object's SI mass.
export function marblePreviewSpec(base, physical, epoch) {
  const fields = {
    r0: ["sim-radius", 0.005, 0.020], rho0: ["sim-rho", 0.2, 5],
    mu0: ["sim-mu", 0, 1], e0: ["sim-e", 0.02, 0.95],
    roll0: ["sim-roll", 0, 0.3], irr0: ["sim-irr", 0, 0.3],
    omega: ["sim-omega", 40, 200],
  };
  const uniforms = { ...base.uniforms };
  for (const [name, [source, lo, hi]] of Object.entries(fields)) {
    const value = Number(physical[source]);
    if (!Number.isFinite(value) || value < lo || value > hi)
      throw new RangeError(`${source} outside the native law's ${lo}–${hi} range`);
    uniforms[name] = value;
  }
  Object.assign(uniforms, { wt0: 1, wt1: 0, wt2: 0, wt3: 0,
    emit_rate: base.studio_count > 1 ? base.uniforms.emit_rate : 1e9 });
  return { ...base, epoch, uniforms, run: "continuous", speed: 1 };
}

export function livePreviewParticle(points, id = 0) {
  const offset = id * 5;
  if (!points || points.length < offset + 5 || !(points[offset + 4] >= 50)) return null;
  const flag = Math.floor(points[offset + 4] / 100 + 0.001);
  return { x: points[offset], y: points[offset + 1], z: points[offset + 2],
    radius: (points[offset + 4] - flag * 100) * 0.001 };
}

export function previewLawToScene(view, depth, radius) {
  const [x0, x1, y0, y1] = view;
  const scale = 3.5 / Math.max(x1 - x0, y1 - y0, depth);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return { scale, particleScale: radius * scale,
    point: ([x, y, z]) => [(x - mx) * scale, z * scale, -(y - my) * scale] };
}

// Screen right/up projected onto the law's X/Y course plane. Camera yaw is
// the only input: pitch changes the view's height, not a D-pad's meaning.
export function cameraNudgeGravity(dx, up, yaw, strength = 0.72) {
  const length = Math.hypot(dx, up), gain = length > 1 ? strength / length : strength;
  const x = gain * (Math.cos(yaw) * dx + Math.sin(yaw) * up);
  const y = gain * (-Math.sin(yaw) * dx + Math.cos(yaw) * up);
  return [x, y, -Math.sqrt(Math.max(0, 1 - x*x - y*y))];
}

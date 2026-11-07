// The Studio's editable object drives one particle in the existing fountain3d
// product. These are native law fields, independent of the object's SI mass.
export function marblePreviewSpec(base, physical, epoch) {
  const fields = {
    r0: ["sim-radius", 0.005, 0.020], rho0: ["sim-rho", 0.2, 5],
    mu0: ["sim-mu", 0, 1], e0: ["sim-e", 0.02, 0.95],
    roll0: ["sim-roll", 0, 0.3], irr0: ["sim-irr", 0, 0.3],
  };
  const uniforms = { ...base.uniforms };
  for (const [name, [source, lo, hi]] of Object.entries(fields)) {
    const value = Number(physical[source]);
    if (!Number.isFinite(value) || value < lo || value > hi)
      throw new RangeError(`${source} outside the native law's ${lo}–${hi} range`);
    uniforms[name] = value;
  }
  Object.assign(uniforms, { wt0: 1, wt1: 0, wt2: 0, wt3: 0, emit_rate: 1e9 });
  return { ...base, epoch, uniforms, run: "continuous", speed: 1 };
}

export function livePreviewParticle(points) {
  if (!points || points.length < 5 || !(points[4] >= 50)) return null;
  const flag = Math.floor(points[4] / 100 + 0.001);
  return { x: points[0], y: points[1], z: points[2],
    radius: (points[4] - flag * 100) * 0.001 };
}

export function previewLawToScene(view, depth, radius) {
  const [x0, x1, y0, y1] = view;
  const scale = 3.5 / Math.max(x1 - x0, y1 - y0, depth);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return { scale, particleScale: radius * scale,
    point: ([x, y, z]) => [(x - mx) * scale, z * scale, -(y - my) * scale] };
}

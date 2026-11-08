// Prescribed translation of a containing case in its law frame.  The law sees
// the second derivative as a separate inertial acceleration; the renderer
// uses the very same displacement.  Time is simulated time, never wall time.
export function caseVibration(kind, hz, amplitude, axis, phase, time) {
  if (!['sander', 'vibrator'].includes(kind) || !(hz > 0) || !(amplitude > 0))
    return { displacement: [0, 0, 0], acceleration: [0, 0, 0] };
  const w = 2 * Math.PI * hz, u = w * time + phase * Math.PI / 180;
  let displacement;
  if (kind === 'sander') displacement = [amplitude * Math.cos(u), amplitude * Math.sin(u), 0];
  else {
    const index = { x: 0, y: 1, z: 2 }[axis] ?? 2;
    displacement = [0, 0, 0]; displacement[index] = amplitude * Math.sin(u);
  }
  return { displacement, acceleration: displacement.map((v) => -w * w * v) };
}

// A single directional tap of the case. Both displacement and acceleration
// vanish at the endpoints, so the wall returns to its starting pose without
// a velocity jump. The initial inertial acceleration of a grain points in
// `direction`; the return stroke closes the case trajectory.
export function caseNudge(direction, strength, startStep, step, dt, duration = 0.22) {
  const u = (step - startStep) * dt / duration;
  if (!(u > 0 && u < 1) || !(strength > 0))
    return { displacement: [0, 0, 0], acceleration: [0, 0, 0] };
  const a = -0.012 * Math.min(1, strength), v = u * (1 - u);
  const distance = a * 64 * v * v * v;
  const acceleration = a * 384 * v * (1 - 5 * u + 5 * u * u) / (duration * duration);
  return { displacement: [direction[0] * distance, direction[1] * distance, 0],
    acceleration: [direction[0] * acceleration, direction[1] * acceleration, 0] };
}

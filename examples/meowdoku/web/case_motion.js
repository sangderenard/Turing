// Prescribed finite case motion in simulation steps.  The law and renderer evaluate this same trajectory.
// Quintic ease has zero velocity and acceleration at both ends; its derivative maxima are 15/8 and 10/sqrt(3).
export function caseFlipAt(flip, step) {
  const u = Math.max(0, Math.min(1, (step - flip.startStep) / flip.steps));
  const s = u * u * u * (10 + u * (-15 + 6 * u));
  const ds = 30 * u * u * (1 - u) * (1 - u);
  const dds = 60 * u - 180 * u * u + 120 * u * u * u;
  const seconds = flip.steps * flip.dt;
  return { angle: flip.from + flip.delta * s, omega: -flip.delta * ds / seconds, domega: -flip.delta * dds / (seconds * seconds) };
}
export function caseFlipSteps(delta, requestedSeconds, dt, maxOmega, maxDomega) {
  const seconds = Math.max(requestedSeconds, (15 / 8) * Math.abs(delta) / maxOmega,
    Math.sqrt((10 / Math.sqrt(3)) * Math.abs(delta) / maxDomega));
  return Math.max(1, Math.ceil(seconds / dt));
}

// The law's axes map to the scene as x -> world X, y -> world -Z, z -> world Y.
export function lawAxis(worldAxis) {
  return worldAxis === "x" ? [1, 0, 0] : worldAxis === "z" ? [0, -1, 0] : [0, 0, 1];
}
export function rotateLawVector(vector, worldAxis, angle) {
  const a = lawAxis(worldAxis), c = Math.cos(angle), s = Math.sin(angle);
  const dot = a[0] * vector[0] + a[1] * vector[1] + a[2] * vector[2];
  return [vector[0] * c + (a[1] * vector[2] - a[2] * vector[1]) * s + a[0] * dot * (1 - c),
    vector[1] * c + (a[2] * vector[0] - a[0] * vector[2]) * s + a[1] * dot * (1 - c),
    vector[2] * c + (a[0] * vector[1] - a[1] * vector[0]) * s + a[2] * dot * (1 - c)];
}

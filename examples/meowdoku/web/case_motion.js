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

// Visual preview only. Rotate and Rock use the case-angle trajectory in compute.js;
// Flip uses the same finite quintic turn as the physical case motion.
import { caseFlipAt } from "./case_motion.js";

export function previewMotionAngle(kind, hz, phaseDegrees, arcDegrees, seconds) {
  const rate = Math.max(0, Number(hz) || 0);
  const phase = (Number(phaseDegrees) || 0) * Math.PI / 180;
  if (kind === "rotate") return 2 * Math.PI * rate * seconds + phase;
  if (kind === "rock") return (Math.PI / 180) * (Number(arcDegrees) || 0) / 2 *
    Math.sin(2 * Math.PI * rate * seconds + phase);
  if (kind === "flip" && rate > 0) {
    const position = rate * seconds + phase / (2 * Math.PI);
    const turn = Math.floor(position), fraction = position - turn;
    // Each cycle gives a half turn, then rests. The moving interval is the
    // same quintic trajectory used by the case's physical Flip.
    return caseFlipAt({ from: turn * Math.PI, delta: Math.PI,
      startStep: 0, steps: 0.7, dt: 1 }, fraction).angle;
  }
  return 0;
}

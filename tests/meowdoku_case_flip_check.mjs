import assert from "node:assert/strict";
import { caseFlipAt, caseFlipSteps } from "../examples/meowdoku/web/case_motion.js";
import { caseModel } from "../examples/meowdoku/web/gravity.js";
import { VisualTape, filterTape } from "../examples/meowdoku/web/loading_anim.js";

const dt = 1 / 360, omegaCap = 2, accelCap = 6;
const steps = caseFlipSteps(Math.PI, 0.4, dt, omegaCap, accelCap);
const flip = { from: 0, delta: Math.PI, startStep: 120, steps, dt };
assert.ok(steps * dt > 0.4);
const start = caseFlipAt(flip, 120);
assert.equal(start.angle, 0);
assert.ok(Math.abs(start.omega) < 1e-12 && Math.abs(start.domega) < 1e-12);
const end = caseFlipAt(flip, 120 + steps);
assert.ok(Math.abs(end.angle - Math.PI) < 1e-12 && Math.abs(end.omega) < 1e-12 && Math.abs(end.domega) < 1e-12);
let last = 0;
for (let k = 120; k <= 120 + steps; k++) {
  const m = caseFlipAt(flip, k);
  assert.ok(m.angle >= last - 1e-12 && Math.abs(m.omega) <= omegaCap + 1e-12 && Math.abs(m.domega) <= accelCap + 1e-12);
  last = m.angle;
}
const half = caseFlipAt(flip, 120 + steps / 2);
assert.ok(Math.abs(half.angle - Math.PI / 2) < 1e-12 && half.omega < 0);
// The exact displayed angle drives the model of the case, including the flat board and its outlined wall.
const model = caseModel(half.angle, 0, 0);
assert.ok(Math.abs(model[0]) < 1e-6 && Math.abs(model[8] + 1) < 1e-6);
const tape = new VisualTape({ sourceSnap: 6, stride: 1, sourceCount: 8, maxGrains: 8 });
for (let k = 1; k <= 12; k++) {
  const data = new Float32Array(40);
  for (let i = 0; i < 8; i++) { data[5 * i] = i / 8; data[5 * i + 1] = k / 20; data[5 * i + 4] = 120; }
  tape.push(k, data);
}
const saved = filterTape(tape, { from: 1, to: 12, maxGrains: 8, fps: 60, bits: 12, dt,
  angleAtStep: (step) => Math.PI * (step - 6) / 66 });
assert.equal(saved.angle[0], 0);
assert.ok(Math.abs(saved.angle.at(-1) - 32768) <= 1); // a half turn of the case is in the saved view angle
console.log(JSON.stringify({ steps, seconds: steps * dt, midpoint: half.angle, omega: half.omega }));

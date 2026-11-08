import assert from 'node:assert/strict';
import { caseVibration, caseNudge } from '../examples/meowdoku/web/case_vibration.js';

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
const off = caseVibration(null, 5, 0.02, 'z', 0, 1);
assert.deepEqual(off.displacement, [0, 0, 0]);
assert.deepEqual(off.acceleration, [0, 0, 0]);

const a = 0.02, hz = 5, w = 2 * Math.PI * hz;
const sand = caseVibration('sander', hz, a, 'z', 0, 0);
near(sand.displacement[0], a); near(sand.displacement[1], 0);
near(sand.acceleration[0], -a * w * w);
near(sand.acceleration[1], 0);
const quarter = caseVibration('sander', hz, a, 'z', 0, 1 / (4 * hz));
near(quarter.displacement[0], 0); near(quarter.displacement[1], a);
const vib = caseVibration('vibrator', hz, a, 'z', 90, 0);
assert.deepEqual(vib.displacement.slice(0, 2), [0, 0]);
near(vib.displacement[2], a);
near(vib.acceleration[2], -a * w * w);
const nudgeAt = (u, strength = 1) => caseNudge([1, 0], strength, 10, 10 + u * 220, 0.001, 0.22);
for (const u of [-0.1, 0, 1, 1.1]) {
  assert.deepEqual(nudgeAt(u).displacement, [0, 0, 0]);
  assert.deepEqual(nudgeAt(u).acceleration, [0, 0, 0]);
}
near(nudgeAt(0.5).displacement[0], -0.012);
assert.ok(nudgeAt(0.5).acceleration[0] > 0);
near(nudgeAt(0.5, 0.5).acceleration[0], nudgeAt(0.5).acceleration[0] * 0.5);
for (const u of [0.15, 0.4, 0.7, 0.9]) {
  const h = 1e-3, dt = 0.22 * h;
  const d2 = (nudgeAt(u + h).displacement[0] - 2 * nudgeAt(u).displacement[0] + nudgeAt(u - h).displacement[0]) / (dt * dt);
  assert.ok(Math.abs(d2 - nudgeAt(u).acceleration[0]) < 0.002);
}
console.log('case vibration displacement/acceleration: ok');

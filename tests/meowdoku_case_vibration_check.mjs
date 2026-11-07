import assert from 'node:assert/strict';
import { caseVibration } from '../examples/meowdoku/web/case_vibration.js';

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
console.log('case vibration displacement/acceleration: ok');

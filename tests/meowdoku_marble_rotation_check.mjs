import assert from 'node:assert/strict';
import { integrateAngular } from '../examples/meowdoku/web/marble_rotation.js';

const near = (actual, expected, tolerance = 1e-10) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const still = [0, 0, 0, 1];
assert.deepEqual(integrateAngular(still, [0, 0, 0], 1), still);
const half = integrateAngular(still, [0, 0, Math.PI], 0.5);
near(half[2], Math.SQRT1_2);
near(half[3], Math.SQRT1_2);
const full = integrateAngular(half, [0, 0, Math.PI], 0.5);
near(full[2], 1);
near(full[3], 0);
near(Math.hypot(...full), 1);
const mixed = integrateAngular(integrateAngular(still, [2, 0, 0], 0.1), [0, 3, 0], 0.2);
near(Math.hypot(...mixed), 1);
assert.notDeepEqual(mixed, still);
console.log('marble physical angular integration: ok');

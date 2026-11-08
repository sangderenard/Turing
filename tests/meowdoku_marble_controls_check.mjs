import assert from 'node:assert/strict';
import { marbleGravityVector, marbleThumbVector } from '../examples/meowdoku/web/marble_controls.js';
import { fountainGravitySource } from '../examples/meowdoku/web/gravity.js';

const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-12, `${a} != ${b}`);
const rest = marbleGravityVector(0,0);
assert.deepEqual(rest,[0,0,-1]);
for (const [x,down] of [[.5,0],[0,.5],[-.7,.7],[1,1]]) {
  const g=marbleGravityVector(x,down);
  near(Math.hypot(...g),1);
  assert.ok(g[2]<0);
  assert.equal(g[0]===0,x===0);
  assert.equal(g[1]===0,down===0);
}
near(marbleGravityVector(.5,0)[0],.5);
near(marbleGravityVector(0,.5)[1],-.5);
assert.ok(marbleGravityVector(.9,0)[0] > .8);
assert.deepEqual(marbleGravityVector(0,0,[0,-9.8,0]),[0,-1,0]);
near(marbleGravityVector(0,0,[.3,-.4,-.5])[2],-.5/Math.sqrt(.5));
assert.deepEqual(marbleThumbVector(50,50,0,0,100,100),[0,0]);
assert.deepEqual(marbleThumbVector(83,50,0,0,100,100),[1,0]);
assert.deepEqual(marbleThumbVector(50,83,0,0,100,100),[0,1]);
near(Math.hypot(...marbleThumbVector(200,200,0,0,100,100)),1);
// Fresh Fountain3D and Marble→Fountain3D with no active sensor both use the
// authored downward gravity, never Marble's backplate-normal vector.
const fountainDefault = [0, -1, 0];
assert.deepEqual(fountainGravitySource(false, false, null) || fountainDefault, fountainDefault);
assert.deepEqual(fountainGravitySource(false, true, null) || fountainDefault, fountainDefault);
assert.deepEqual(fountainGravitySource(true, true, [0, 0, -1]) || fountainDefault, fountainDefault);
assert.deepEqual(fountainGravitySource(false, true, [0.3, -0.9, -0.1]), [0.3, -0.9, -0.1]);
console.log('marble gravity axes and magnitude: ok');

import assert from 'node:assert/strict';
import { marbleGravityVector } from '../examples/meowdoku/web/marble_controls.js';

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
console.log('marble gravity axes and magnitude: ok');

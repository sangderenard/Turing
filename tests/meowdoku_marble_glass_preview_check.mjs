import assert from "node:assert/strict";
import {isOpaquePreviewMaterial,triangleCentres,transparentTriangleOrder} from "../examples/meowdoku/web/marble_glass_preview.js";
assert(isOpaquePreviewMaterial(0,1),"opaque materials retain opaque pass");
assert(!isOpaquePreviewMaterial(1,0),"dielectric glass uses transparent pass");
assert(!isOpaquePreviewMaterial(0,.5),"ordinary alpha material retains transparency");
const centres=triangleCentres([
  [[-1,0,-1],[1,0,-1],[0,0,-1]],
  [[-1,0,-3],[1,0,-3],[0,0,-3]],
  [[-1,0,-2],[1,0,-2],[0,0,-2]],
]);
const identity=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
assert.deepEqual([...transparentTriangleOrder(centres,identity)],[3,4,5,6,7,8,0,1,2]);
const reverse=[...identity];reverse[10]=-1;
assert.deepEqual([...transparentTriangleOrder(centres,reverse)],[0,1,2,6,7,8,3,4,5]);
assert.equal(transparentTriangleOrder([],identity).length,0);
console.log("PASS preview transparency: opaque/transparent partition and camera-dependent back-to-front triangles");

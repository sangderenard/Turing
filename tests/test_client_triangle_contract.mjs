import assert from 'node:assert/strict';
import {validateTriangleScene} from '../examples/meowdoku/web/client_triangle_kernel.js';
const tri={v0:[0,0,0],e1:[1,0,0],e2:[0,1,0],owner:7};
const s={revision:'r',triangles:[tri]};assert.equal(validateTriangleScene(s),s);
for(const bad of [{...s,revision:''},{...s,triangles:[]},{...s,triangles:[tri,tri]},{...s,unsupportedGeometry:['sphere']},{...s,triangles:[{...tri,v0:[NaN,0,0]}]},{...s,triangles:[{...tri,owner:16777216}]}])assert.throws(()=>validateTriangleScene(bad));
console.log('triangle contract checks passed');

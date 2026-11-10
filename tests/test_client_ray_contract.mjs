import assert from 'node:assert/strict';
import {validateSphereScene,validateRayBatch} from '../examples/meowdoku/web/client_ray_kernel.js';
const scene={revision:'r',spheres:[{center:[0,0,0],radius:1,owner:0}]};
assert.equal(validateSphereScene(scene),scene);
for(const bad of [{...scene,revision:''},{...scene,spheres:[]},{...scene,spheres:[...scene.spheres,...scene.spheres]},{revision:'r',spheres:[{center:[0,0,0],radius:0,owner:1}]}]) assert.throws(()=>validateSphereScene(bad));
const ray=new Float32Array([0,0,-3,0,0,0,1,10]);
assert.equal(validateRayBatch(ray),1);
for(const bad of [Array.from(ray),new Float32Array(0),new Float32Array(9),new Float32Array(4097*8),new Float32Array([0,0,0,0,0,0,0,10]),new Float32Array([0,0,0,2,0,0,1,1]),new Float32Array([NaN,0,0,0,0,0,1,10])]) assert.throws(()=>validateRayBatch(bad));
console.log('client ray contract checks passed');

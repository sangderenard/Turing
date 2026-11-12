import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {fieldCategoryRuns,fieldCategoryEnabled,withSceneCategory} from '../examples/meowdoku/web/marble_field_categories.js';
import {transparentTriangleOrder} from '../examples/meowdoku/web/marble_glass_preview.js';
import {STUDIO_SCENE_PART_GROUP} from '../examples/meowdoku/web/studio_scene_builder.js';
assert.deepEqual(STUDIO_SCENE_PART_GROUP,{fixture:'stage',room:'room',backdrop:'room',backdropHolder:'room',table:'stage',gobo:'room',lightStand:'stage',shopLight:'room'});
const face=[[0,0,0],[1,0,0],[0,1,0],7,[[0,0,1],[0,0,1],[0,0,1]]];
const faces=[...withSceneCategory([face],'subject'),...withSceneCategory([face],'stage'),...withSceneCategory([face],'room')];
assert.deepEqual(faces.map(f=>f[3]),[7,7,7]); // Exact same material in all owners.
assert.deepEqual(faces.map(f=>f.sceneCategory),['subject','stage','room']);
assert.equal(face.sceneCategory,undefined); // Do not mutate canonical builder geometry.
const page=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const source=page.slice(page.indexOf('function drawTransparent('),page.indexOf('function digitFaces('));
const calls=[],gl={TRIANGLES:4,UNSIGNED_INT:5125,ELEMENT_ARRAY_BUFFER:1,DYNAMIC_DRAW:2,
  bindVertexArray(){},bindBuffer(){},bufferData(target,order){this.order=Array.from(order);},
  drawElements(mode,count,type,offset){calls.push({category:gl.category,indices:gl.order.slice(offset/4,offset/4+count)});}};
const context={gl,transparentTriangleOrder,fieldCategoryRuns};vm.createContext(context);vm.runInContext(source,context);
const centres=[[0,0,-1],[0,0,-5],[0,0,-3],[0,0,-2],[0,0,-4],[0,0,-6]],
  categories=['subject','stage','room','subject','stage',null],
  result={vao:1,indexBuffer:2,count:18,centres,categories,sortAxis:null},
  view=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const expected=Array.from(transparentTriangleOrder(centres,view));
for(let mask=0;mask<8;mask++) {
  calls.length=0;
  context.drawTransparent(result,view,category=>{gl.category=category;gl.enabled=fieldCategoryEnabled(mask,category);});
  assert.deepEqual(calls.flatMap(c=>c.indices),expected,'Category runs must preserve every index in globally sorted order');
  assert.deepEqual(calls.map(c=>c.category),[null,'stage','room','subject']);
  for(const category of ['subject','stage','room',null,'particle','cage']) {
    const bit={subject:1,stage:2,room:4}[category]??0;
    assert.equal(fieldCategoryEnabled(mask,category),!!(mask&bit));
  }
}
// Repeated draw uses cached runs, and camera reversal rebuilds them without splitting categories.
const reverse=[...view];reverse[10]=-1;calls.length=0;
context.drawTransparent(result,reverse,category=>gl.category=category);
assert.deepEqual(calls.flatMap(c=>c.indices),Array.from(transparentTriangleOrder(centres,reverse)));
assert.throws(()=>fieldCategoryRuns(['subject'],new Uint32Array([1,2,3])),/lineage/);
assert.throws(()=>fieldCategoryRuns(['subject'],new Uint32Array([0,2,1])),/lineage/);
assert.deepEqual(fieldCategoryRuns([]),[]);
console.log('PASS exact category lineage, shared materials, all 8 masks, unknown fallback, globally sorted transparent runs');

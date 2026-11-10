import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {DEFAULT_TRIANGLE_BUDGET, EXACT_SPHERES_READY, geometryPolicy, photoGeometryPolicy,
  createGeometryPolicyControls, showGeometryDiagnostic} from "../examples/meowdoku/web/marble_geometry_policy.js";
import {packMarblePhoto} from "../examples/meowdoku/web/marble_photo.js";

assert.deepEqual(geometryPolicy(), {triangleBudget:250000,sphereCoverage:"mesh"});
assert.equal(DEFAULT_TRIANGLE_BUDGET,250000);
for(const triangleBudget of [1000,250000,1000000])for(const sphereCoverage of ["mesh","exact"])
  assert.deepEqual(geometryPolicy({triangleBudget,sphereCoverage}),{triangleBudget,sphereCoverage});
for(const triangleBudget of [999,1000001,1000.5,NaN,Infinity,true,false,null,"250000",[],{}])
  assert.throws(()=>geometryPolicy({triangleBudget}),/Triangle budget/);
for(const sphereCoverage of [null,true,0,"analytic","Mesh",[],{}])
  assert.throws(()=>geometryPolicy({sphereCoverage}),/Choose Saved triangle mesh/);
assert.deepEqual(photoGeometryPolicy({getElementById:()=>null}),geometryPolicy(),"legacy DOM defaults are explicit");

class Element extends EventTarget {
  constructor(value=""){super();this.value=value;this.textContent="";}
  setCustomValidity(value){this.validationMessage=value;}
}
const elements=new Map([
  ["photo-triangle-budget",new Element("123456")],["photo-sphere-coverage",new Element("mesh")],
  ["photo-sphere-exact",new Element("exact")],["photo-geometry-diagnostic",new Element()],
]);
const doc={getElementById:id=>elements.get(id)},budget=elements.get("photo-triangle-budget"),coverage=elements.get("photo-sphere-coverage");
createGeometryPolicyControls(doc);
assert.equal(elements.get("photo-sphere-exact").disabled,!EXACT_SPHERES_READY);
assert.deepEqual(photoGeometryPolicy(doc),{triangleBudget:123456,sphereCoverage:"mesh"});
for(const bad of [""," ","999","1000001","1000.5","NaN","Infinity"]){
  budget.value=bad;budget.dispatchEvent(new Event("input"));
  assert.match(budget.validationMessage,/Triangle budget/);
  assert.throws(()=>photoGeometryPolicy(doc),/Triangle budget/);
  assert.equal(budget.value,bad,"invalid edits are not silently clamped or replaced");
}
budget.value="123456";budget.dispatchEvent(new Event("change"));
assert.equal(budget.validationMessage,"");
coverage.value="exact";coverage.dispatchEvent(new Event("change"));
if(!EXACT_SPHERES_READY)assert.throws(()=>photoGeometryPolicy(doc),/unavailable until renderer validation/);
else assert.deepEqual(photoGeometryPolicy(doc),{triangleBudget:123456,sphereCoverage:"exact"});
coverage.value="mesh";coverage.dispatchEvent(new Event("change"));
assert.equal(budget.value,"123456","coverage does not overwrite the saved numeric budget");

const face=[[-1,0,0],[1,0,0],[0,1,0],7,[[0,0,1],[0,0,1],[0,0,1]]];
const geometry={interior:[face],glass:[face]},digest="a".repeat(64);
const unpack=buffer=>{const size=new DataView(buffer).getUint32(4,true);return {
  config:JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,size))),
  mesh:new Uint8Array(buffer,16+size),count:new DataView(buffer).getUint32(8,true),
};};
const original=unpack(packMarblePhoto(geometry,digest));
assert.equal(original.config.triangle_budget,250000);assert.equal(original.config.sphere_coverage,"mesh");
for(const sphereCoverage of ["mesh","exact"])for(const live of [undefined,{session_id:"b".repeat(32),history_samples:16,checkpoint:true}]){
  const result=unpack(packMarblePhoto(geometry,digest,{triangleBudget:1000,sphereCoverage,live,maxBounces:37,bounceMode:"auto"}));
  assert.equal(result.config.triangle_budget,1000);assert.equal(result.config.sphere_coverage,sphereCoverage);
  assert.equal(result.config.max_bounces,37);assert.equal(result.config.bounce_mode,"auto");
  assert.deepEqual(result.config.live,live);
  assert.deepEqual(result.mesh,original.mesh,"all saved positions, normals and material IDs stay byte-identical");
  assert.equal(result.count,2);
}
assert.throws(()=>packMarblePhoto(geometry,digest,{triangleBudget:999}),/Triangle budget/);
assert.throws(()=>packMarblePhoto(geometry,digest,{sphereCoverage:"polygon"}),/Choose Saved triangle mesh/);

showGeometryDiagnostic(doc,{geometry_stats:{sphere_coverage:"exact",triangle_budget:250000,input_triangles:97472,effective_triangles:3256,analytic_spheres:5}});
const output=elements.get("photo-geometry-diagnostic");
assert.match(output.textContent,/3,256 scene triangles \/ 250,000 budget/);
assert.match(output.textContent,/97,472 input triangles; 5 analytic spheres/);
showGeometryDiagnostic(doc,{geometry_stats:{sphere_coverage:"exact",triangle_budget:250000,input_triangles:1,effective_triangles:"1",analytic_spheres:1}});
assert.match(output.textContent,/diagnostic is invalid/);
showGeometryDiagnostic(doc,{});assert.match(output.textContent,/not recorded/);
showGeometryDiagnostic(doc,null);assert.match(output.textContent,/next photo or live scene/);

const html=readFileSync(new URL("../examples/meowdoku/web/marble_studio.html",import.meta.url),"utf8");
const tag=html.match(/<input id="photo-triangle-budget"[^>]*>/)?.[0];
for(const field of ['type="number"','min="1000"','max="1000000"','step="1"','value="250000"','required'])assert.ok(tag.includes(field));
assert.match(html,/<option value="mesh"[^>]*>Saved triangle mesh<\/option>/);
if(!EXACT_SPHERES_READY)assert.match(html,/<option id="photo-sphere-exact" value="exact" disabled>/);
else {
 assert.match(html,/<option id="photo-sphere-exact" value="exact" selected>/,"fresh pages explicitly select validated Exact");
 assert.match(html,/Save older designs again/);assert.match(html,/There is no automatic fallback/);
 assert.match(html,/Coatings, painted bands and non-spherical shapes are unsupported/);
}
assert.ok(html.indexOf('id="photo-triangle-budget"')<html.indexOf('id="photo-camera"'),"geometry controls are prominent, before camera controls");
for(const path of ["web_export.py","server.py"]){
 const text=readFileSync(new URL(`../examples/meowdoku/${path}`,import.meta.url),"utf8");
 assert.ok(text.includes("marble_geometry_policy.js"),`${path} includes the helper asset`);
}
console.log("Exposure geometry: strict ranges, legacy defaults, guarded exact UI, unchanged saved mesh, mode parity and worker diagnostics PASS");

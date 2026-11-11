// Actual Node-only scene generation and existing binary ray packing.
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
for(const name of ['document','window','WebGL2RenderingContext','OffscreenCanvas'])
  assert.equal(typeof globalThis[name],'undefined','Node unexpectedly supplies '+name);
globalThis.fetch=()=>{throw Error('No network allowed for detached scene generation');};
const {buildStudioAgentProduct}=await import('../examples/meowdoku/web/marble_studio_agent.js');
const root=new URL('../',import.meta.url);
const {assets,authoring}=JSON.parse(await readFile(new URL('tests/fixtures/studio_scene_builder_authoring.json',root),'utf8'));
const settings={object:authoring.object,lighting:authoring.lighting,
  camera:{eye_m:[0,.08,.07],target_m:[0,.08,.18],fov_degrees:18,aperture_radius_m:.002,focus_distance_m:.11}};
const request={schema:'turing.studio-agent-request.v1',operation:'scene',settings};
const original=JSON.stringify(request);
const built=await buildStudioAgentProduct(request,{assets});
assert.equal(built.contentType,'application/json');
const scene=JSON.parse(new TextDecoder().decode(built.bytes));
assert(scene.fixtures.vertices.length>0&&scene.geometry.triangleCount>0);
assert(scene.objectSnapshot.geometry_sha256);
assert.deepEqual(scene.camera,settings.camera);
assert.equal(JSON.stringify(request),original,'Builder mutated caller settings');
const baked=await buildStudioAgentProduct({schema:request.schema,operation:'scene'},{assets,defaults:settings});
assert.deepEqual(baked.bytes,built.bytes,'Configured defaults differ from explicit settings');
const ray=await buildStudioAgentProduct({...request,operation:'ray-payload',photo:{resolution:128,seconds:.1,maxBounces:1}},{assets});
assert.equal(ray.contentType,'application/octet-stream');
assert.equal(new TextDecoder().decode(ray.bytes.slice(0,4)),'MRP1');
const view=new DataView(ray.bytes.buffer,ray.bytes.byteOffset,ray.bytes.byteLength);
const config=JSON.parse(new TextDecoder().decode(ray.bytes.slice(16,16+view.getUint32(4,true))));
assert.deepEqual(config.fixtures,scene.fixtures);
assert.equal(config.object_snapshot.geometry_sha256,scene.objectSnapshot.geometry_sha256);
assert.equal(config.exposure_seconds,.1);assert.equal(config.camera.resolution,128);
await assert.rejects(buildStudioAgentProduct({...request,shader:'invented'},{assets}),/Unknown/);
await assert.rejects(buildStudioAgentProduct({...request,settings:{...settings,camera:{}}},{assets}),/focus/);
await assert.rejects(buildStudioAgentProduct({...request,operation:'render'},{assets}),/scene or ray-payload/);
for(const controls of [{'light-strength-left':-10},{'floor-material-a':'nonexistent-material'}])
  await assert.rejects(buildStudioAgentProduct({...request,settings:{...settings,lighting:{...settings.lighting,
    controls:{...settings.lighting.controls,...controls}}}},{assets}),/Invalid|Unavailable/);
const out=new URL('build/dot-studio-agent-20261111/',root);await mkdir(out,{recursive:true});
await writeFile(new URL('ray.mrp1',out),ray.bytes);
await writeFile(new URL('request.json',out),original);
await writeFile(new URL('receipt.json',out),JSON.stringify({ok:true,runtime:'Node',browser:false,dom:false,shader:false,
  subject_triangles:scene.geometry.triangleCount,fixture_triangles:scene.fixtures.material_ids.length,
  scene_bytes:built.bytes.length,ray_bytes:ray.bytes.length,
  ray_sha256:createHash('sha256').update(ray.bytes).digest('hex')},null,2));
console.log('Actual Studio settings -> scene/MRP1 passed in Node with no DOM/WebGL and a forbidden network boundary');

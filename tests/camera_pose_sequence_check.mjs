// Sequence controls pass through the existing payload and Studio agent boundary.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {buildStudioAgentProduct} from '../examples/meowdoku/web/marble_studio_agent.js';
const {assets,authoring}=JSON.parse(await readFile(new URL('fixtures/studio_scene_builder_authoring.json',import.meta.url),'utf8'));
const settings={object:authoring.object,lighting:authoring.lighting,
 camera:{eye_m:[0,.08,.07],target_m:[0,.08,.18],fov_degrees:18,aperture_radius_m:.002,focus_distance_m:.11}};
const poses=[{eye_m:[.01,.08,.07]},{target_m:[.02,.08,.18],fov_degrees:45}];
const request={schema:'turing.studio-agent-request.v1',operation:'ray-payload',settings,
 photo:{resolution:128,cameraPoses:poses}};
const original=JSON.stringify(request);
const product=await buildStudioAgentProduct(request,{assets});
const view=new DataView(product.bytes.buffer,product.bytes.byteOffset,product.bytes.byteLength);
const config=JSON.parse(new TextDecoder().decode(product.bytes.slice(16,16+view.getUint32(4,true))));
assert.deepEqual(config.camera_poses,poses);
assert.equal(config.orbit_frames,1);
assert.equal(JSON.stringify(request),original);
for(const options of [{cameraPoses:[]},{cameraPoses:Array(33).fill({})},{orbitFrames:8},{oversampling:2},{tiled:{tile_size:128}}])
 await assert.rejects(buildStudioAgentProduct({...request,photo:{...request.photo,...options}},{assets}),/Camera poses/);
console.log('Camera pose controls preserve order and reject mixed modes in Studio agent MRP1 payload.');

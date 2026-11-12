import assert from 'node:assert/strict';
import fs from 'node:fs';
import {composeLivingDataStudio} from '../examples/meowdoku/web/living_data_studio_adapter.js';
import {validateStudioSubjectPacket} from '../examples/meowdoku/web/marble_studio_subject_renderer.js';
const path=new URL('../build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json',import.meta.url);
if(!fs.existsSync(path)){console.log('SKIP: exact Studio capture fixture not materialized');process.exit(0);}
const asset=JSON.parse(fs.readFileSync(path,'utf8'));
const materials=Object.fromEntries(Object.entries(asset.render.materials).map(([name,table])=>[name,table.data.slice(0,table.layout.floats_per_record)]));
const mesh={schema:'living-data-realized-mesh.v1',coordinate_space:'world',up_axis:'Y',world:{objects:[{identity:'test:triangle'}]},stride:9,world_identity:'test:synthetic-triangle',source_sha256:'1'.repeat(64),renderer_identity:'test:native/pluck-phong',
 vertices:[0,0,0,0,0,1,1,0,0, 1,0,0,0,0,1,1,0,0, 0,1,0,0,0,1,1,0,0],material_colors:[[1,0,0]],materials,
 identity_spans:[{identity:'test:triangle',firstVertex:0,vertexCount:3}],semantic_part_spans:[],overlays:[]};
const before=JSON.stringify(mesh),composition=await composeLivingDataStudio({mesh,studioSceneAsset:asset});
assert.equal(JSON.stringify(mesh),before);assert.equal(composition.packet.schema,'turing.living-data-studio-composition.v1');
assert.equal(composition.provenance.rig_identity,asset.identity);assert.equal(composition.packet.scene.subject.id,mesh.world_identity);
assert.deepEqual(composition.packet.scene.subject.faces[0].slice(0,3),[[0,0,0],[1,0,0],[0,1,0]]);
validateStudioSubjectPacket(composition.packet,composition.samplers);
await assert.rejects(()=>composeLivingDataStudio({mesh:{...mesh,identity_spans:[]},studioSceneAsset:asset}),/account for all/);
console.log('Explicit Studio composition validation passed; no GPU or faithful appearance claim');

for(const change of [{schema:'bad'},{coordinate_space:'bad'},{renderer_identity:'bad'},{identity_spans:[{firstVertex:0,vertexCount:3}]},{semantic_part_spans:[{identity:'part',objectIdentity:'missing',firstVertex:0,vertexCount:3}]}])await assert.rejects(()=>composeLivingDataStudio({mesh:{...mesh,...change},studioSceneAsset:asset}));
const lightTransform=[1,0,0,0,0,1,0,0,0,0,1,0,25,4,-2,1];
const shifted=await composeLivingDataStudio({mesh,studioSceneAsset:asset,lightTransform});
assert.deepEqual(shifted.provenance.light_transform,lightTransform);
assert.equal(shifted.provenance.mesh_sha256,composition.provenance.mesh_sha256);
assert.deepEqual(shifted.packet.scene.subject.faces,composition.packet.scene.subject.faces);
lightTransform[12]=999;assert.equal(shifted.lightTransform[12],25);
await assert.rejects(()=>composeLivingDataStudio({mesh,studioSceneAsset:asset,lightTransform:[1,0,0,0,0,2,0,0,0,0,1,0,0,0,0,1]}),/uniform positive scale/);

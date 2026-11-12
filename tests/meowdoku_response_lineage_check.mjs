// CPU identity copies only. No page, GL context, render, or ray evaluation.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {RESPONSE_INFORMATION,RESPONSE_LINEAGE,recipientId,copyResponseIdentity,tagSubjectResponseFaces,
  tagPartResponseFaces,copiedResponseSources,fixtureResponseLineage,responseRecipientIds} from '../examples/meowdoku/web/marble_response_lineage.js';
import {withSceneCategory} from '../examples/meowdoku/web/marble_field_categories.js';
import {triangleCentres,transparentTriangleOrder} from '../examples/meowdoku/web/marble_glass_preview.js';
import {phongInterior} from '../examples/meowdoku/web/marble_bubbles.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

const face=z=>[[0,0,z],[1,0,z],[0,1,z],5,[[0,0,1],[0,0,1],[0,0,1]]];
const geometry={interior:[face(-2),face(-1),face(0)],glass:[face(1)],bubbles:[{start:1,count:1}]};
const original=JSON.stringify(geometry);
assert.equal(copiedResponseSources(geometry.interior),null);
tagSubjectResponseFaces(geometry,true);
const visible=[...phongInterior(geometry),...geometry.glass];
assert.deepEqual(visible.map(f=>f.responseSource.sourceFace),[0,2,3],'bubble omission does not renumber original source');
const cloned=withSceneCategory(visible,'subject').map(f=>copyResponseIdentity(f,[...f]));
assert.deepEqual(copiedResponseSources(cloned),copiedResponseSources(visible));
const simple={version:RESPONSE_LINEAGE,spheres:[],fixture_parts:{version:RESPONSE_LINEAGE,parts:[]}};
assert.deepEqual(Array.from(responseRecipientIds(copiedResponseSources(cloned),simple)),[0x10000001,0x10000003,0x10000004]);
assert.equal(new Set(responseRecipientIds(copiedResponseSources(cloned),simple)).size,3,'same material does not alias transparent layers');
const order=transparentTriangleOrder(triangleCentres(cloned),[1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1]);
assert.deepEqual(Array.from(order),[6,7,8,3,4,5,0,1,2],'transparent sort retains original vertex/face ordinals');
assert.equal(cloned[Math.floor(order[0]/3)].responseSource.sourceFace,3);
const sphere={...simple,spheres:[[0,17,2,2]]};
assert.deepEqual(Array.from(responseRecipientIds(copiedResponseSources(cloned),sphere)),[0x10000001,0x30000001,0x30000001]);
tagSubjectResponseFaces(geometry,false);assert.equal(JSON.stringify(geometry),original,'ordinary geometry unchanged');

const parts=[{responseSourceId:'room/window/0',faces:[face(0),face(1)]},
  {responseSourceId:'table/pane/0',faces:[face(2),face(3)]}];
const fixture_parts=fixtureResponseLineage(parts);
assert.deepEqual(fixture_parts.parts,[{source_id:'room/window/0',start:0,count:2},{source_id:'table/pane/0',start:2,count:2}]);
parts.forEach(p=>tagPartResponseFaces(p,p.responseSourceId,true));
assert.deepEqual(Array.from(responseRecipientIds(copiedResponseSources(parts.flatMap(p=>p.faces)),{...simple,fixture_parts})),
  [0x20000001,0x20000002,0x20000003,0x20000004]);
assert.deepEqual(Array.from(responseRecipientIds([null,{namespace:2,sourceId:'missing',sourceFace:0}],simple)),[0,0]);
assert.throws(()=>fixtureResponseLineage([parts[0],parts[0]]),/Unique/);
assert.equal(recipientId(0,-1),0);
for(const v of [[1,-1],[1,.5],[1,0x0fffffff],[4,0]])assert.throws(()=>recipientId(...v));

const unpack=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
const fixtures={vertices:[],material_ids:[],pbr:[],response_lineage:fixture_parts},digest='a'.repeat(64);
const plain=unpack(packMarblePhoto(geometry,digest,{fixtures}));
assert(!Object.hasOwn(plain,'light_field_information'));assert(!Object.hasOwn(plain.fixtures,'response_lineage'));
const legacy=unpack(packMarblePhoto(geometry,digest,{fixtures,receiveLightField:true}));
assert(!Object.hasOwn(legacy,'light_field_information'));assert(!Object.hasOwn(legacy.fixtures,'response_lineage'));
const opted=unpack(packMarblePhoto(geometry,digest,{fixtures,receiveLightField:true,lightFieldInformation:RESPONSE_INFORMATION}));
assert.equal(opted.light_field_information,RESPONSE_INFORMATION);assert.deepEqual(opted.fixtures.response_lineage,fixture_parts);
assert.equal(fixtures.response_lineage,fixture_parts,'packing never mutates input');
for(const options of [{lightFieldInformation:RESPONSE_INFORMATION},
  {receiveLightField:true,lightFieldInformation:'spectral-arrivals-v1'},
  {receiveLightField:true,lightFieldInformation:false}])assert.throws(()=>packMarblePhoto(geometry,digest,options));
for(const filename of ['marble_photo.js','marble_live.js']) {
  const source=readFileSync(new URL(`../examples/meowdoku/web/${filename}`,import.meta.url),'utf8');
  assert.match(source,/fieldTicket\?\.information\?\{lightFieldInformation:fieldTicket\.information\}/);
}
console.log('PASS exact original subject/fixture/sphere identities, omitted bubbles, EBO sorting, opt-in request packing');

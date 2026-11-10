import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {snootSphereGuide,kickerSubjectRays,tableGuideScene,KickerTableQuery,summarizeKickerSamples,kickerTableClearanceBound} from '../examples/meowdoku/web/marble_kicker_guide.js';
import {defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
const role={pos:[0,0,0],dir:[0,0,1],radius:.2,gobo:true,shroudLength:.44,shroudRadius:.22,shroudExitRadius:.22};
assert.equal(snootSphereGuide(role,{center:[0,0,3],radius:.5}).state,'clear');
assert.equal(snootSphereGuide(role,{center:[0,0,1],radius:.5}).state,'partial');
assert.equal(snootSphereGuide(role,{center:[2,0,1],radius:.1}).state,'clipped');
assert.equal(snootSphereGuide(role,{center:[0,0,-1],radius:.1}).state,'clipped');
assert.equal(snootSphereGuide(role,{center:[0,0,.3],radius:.1}).state,'clipped','Subject wholly behind exit');
assert.equal(snootSphereGuide(role,{center:[0,0,.44],radius:.1}).state,'partial','Subject crosses exit plane');
assert.equal(snootSphereGuide({...role,gobo:false},{center:[2,0,1],radius:.1}).state,'open');
assert.equal(snootSphereGuide({...role,pos:[2,3,4],dir:[0,1,0]},{center:[2,6,4],radius:.5}).state,'clear');
const cone=snootSphereGuide(role,{center:[0,0,3],radius:.5});
assert.ok(cone.innerHalfAngleDegrees<cone.outerHalfAngleDegrees,'Actual polygon aperture is not replaced by a circular cone');
assert.ok(snootSphereGuide({...role,shroudExitRadius:.03},{center:[0,0,3],radius:.5}).state==='partial');
assert.ok(snootSphereGuide({...role,shroudOffset:2,shroudExitRadius:2},{center:[0,0,3],radius:.5}).state==='partial','Extended collar can constrain an expanded exit');
assert.throws(()=>snootSphereGuide({...role,dir:[0,0,0]},{center:[0,0,3],radius:.5}),/nonzero/);

const table=defaultStudioTable(),floorY=-1.02-tableFloorDrop(table),transform=studioScaleTransform({version:1,mode:'fixed',unitMm:304.8},10,true);
const subject={center:transform.subjectCenter,radius:transform.subjectRadius},scene=tableGuideScene(table,floorY,transform.unitM);
assert.equal(scene.triangles.length,108);assert.ok(scene.triangles.some(t=>t.part.startsWith('skirt-')));
const aimed=(y)=>{const pos=[-6,y,-6],d=subject.center.map((v,i)=>v-pos[i]),n=Math.hypot(...d);return {...role,pos,dir:d.map(v=>v/n)};};
const safeBound=kickerTableClearanceBound(aimed(-.75),subject,table,floorY);
assert.equal(safeBound.minimumHeight,4.2);assert.equal(safeBound.state,'clear');
assert.equal(kickerTableClearanceBound(aimed(-1),subject,table,floorY).state,'unproven','Emitter-center clearance is not a complete emitter guarantee');
const rays=kickerSubjectRays(aimed(-.5),subject,transform.unitM);assert.equal(rays.length,49);
for(const ray of rays){const endpoint=ray.origin.map((v,i)=>v+ray.direction[i]*ray.maximum),radius=Math.hypot(...endpoint.map((v,i)=>v-subject.center[i]*transform.unitM));assert.ok(Math.abs(radius-transform.radiusM)<1e-10);}
assert.throws(()=>kickerSubjectRays({...role,pos:subject.center},subject),/inside/);

// Actual shared slab/skirt mesh through the existing, digest-verified original
// compiled hit_tri. No JavaScript triangle-intersection substitute.
const oldFetch=globalThis.fetch;globalThis.fetch=async url=>{try{return new Response(await readFile(url));}catch{return new Response('',{status:404});}};
try{
 const query=await KickerTableQuery.create(new URL('../build/dot-client-ray-20261109/studio-compute/',import.meta.url));
 const heights=[-2,-1.1,-1.05,-1.02,-1,-.9,-.5],batch=heights.flatMap(y=>kickerSubjectRays(aimed(y),subject,transform.unitM));
 const result=await query.query(scene,batch),summaries=heights.map((y,i)=>({y,...summarizeKickerSamples(result.hits.slice(i*49,(i+1)*49))}));
 assert.equal(summaries[0].state,'all');assert.ok(summaries[0].parts.some(p=>p.startsWith('skirt-')));
 assert.equal(summaries.at(-1).state,'none');assert.ok(summaries.some(s=>s.state==='some'),'Height sweep resolves partial table clipping');
 for(const s of summaries)assert.match(s.message,/sampled geometric coverage/);
 const noTable=await query.query(tableGuideScene({...table,enabled:false},floorY,transform.unitM),rays);assert.equal(summarizeKickerSamples(noTable.hits).state,'none');
 const canceled=new AbortController();canceled.abort();await assert.rejects(()=>query.query(scene,rays,{signal:canceled.signal}),/canceled/);
 console.log(JSON.stringify({passed:true,scope:result.scope,backend:'original compiled hit_tri Wasm',heights:summaries.map(({y,crossed,total,parts})=>({y,crossed,total,parts}))}));
}finally{globalThis.fetch=oldFetch;}
console.log('Exact polygon-aperture cone/sphere guide, physical subject scaling, actual table/skirt sampled height transitions and cancellation passed');

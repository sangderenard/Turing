import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {defaultFixtureSizes,validateFixtureSizes,applyFixtureSizes,selectFixtureHead,createFixtureSizeControls,SIZED_FIXTURES} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {defaultLightTools,migrateLightTools,validateLightTools,compatibleLightTools,lightToolFootprint} from '../examples/meowdoku/web/marble_light_tools.js';
import {radialLightPose,radialLightPosition} from '../examples/meowdoku/web/marble_light_placement.js';
import {computeStudioFixtureRoles,buildStudioFixtureFaces,studioSceneMaterials} from '../examples/meowdoku/web/studio_scene_builder.js';
import {validateStudioLighting} from '../examples/meowdoku/web/marble_studio_lighting.js';
import {studioLightingFields} from '../examples/meowdoku/web/studio_lighting_fields.js';
import {buildStudioAgentProduct} from '../examples/meowdoku/web/marble_studio_agent.js';
import {snootSphereGuide,kickerTableClearanceBound} from '../examples/meowdoku/web/marble_kicker_guide.js';
import {defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
const close=(a,b)=>assert(Math.abs(a-b)<1e-9,`${a} != ${b}`),copy=structuredClone;
const fixture=JSON.parse(await readFile(new URL('./fixtures/studio_scene_builder_authoring.json',import.meta.url),'utf8'));
const {assets,authoring}=fixture,names=Object.keys(assets.material_index);
const original=validateStudioLighting(authoring.lighting,studioLightingFields(names),names);
const legacy=defaultFixtureSizes(true),fresh=defaultFixtureSizes(),freshTools=defaultLightTools(fresh);
assert(legacy.lights.every(l=>l.head===undefined),'Legacy profiles do not silently acquire a different head');
assert.deepEqual(validateFixtureSizes(legacy),legacy);
for(const id of ['kicker-left','kicker-right']){
 assert.equal(fresh.lights.find(l=>l.id===id).head,'softbox-1x4');
 assert.deepEqual(lightToolFootprint(id,fresh),{width:1,height:4});
 assert.deepEqual(compatibleLightTools(id,fresh),['grid']);
 const tool=freshTools.attachments.find(t=>t.lightId===id);
 assert.equal(tool.kind,'grid');assert.deepEqual([tool.settings.cellsX,tool.settings.cellsY],[2,8]);
 assert.equal(tool.settings.interiorMaterial,'housing');assert.equal(tool.settings.outerMaterial,'housing');
}
// All ten lights use the same head selection, not role-specific substitutes.
for(const {id}of SIZED_FIXTURES)for(const head of ['round','softbox-1x4','native']){
 const selected=selectFixtureHead(legacy,migrateLightTools(legacy),id,head);
 assert.deepEqual(validateFixtureSizes(copy(selected.sizes),false),selected.sizes);
 assert.deepEqual(validateLightTools(copy(selected.tools),selected.sizes),selected.tools);
 if(head==='softbox-1x4')assert(selected.tools.attachments.some(t=>t.lightId===id&&t.kind==='grid'));
 if(head==='round')assert(!selected.tools.attachments.some(t=>t.lightId===id&&t.kind==='grid'));
}
assert.throws(()=>selectFixtureHead(legacy,migrateLightTools(legacy),'kicker-left','unknown'),/Unavailable/);
const halfScale=selectFixtureHead(legacy,migrateLightTools(legacy),'kicker-left','softbox-1x4',{unitMm:152.4});
const physical=lightToolFootprint('kicker-left',halfScale.sizes);close(physical.width*152.4,304.8);close(physical.height*152.4,1219.2);
const before=JSON.stringify({legacy,tools:migrateLightTools(legacy)});
assert.throws(()=>selectFixtureHead(legacy,migrateLightTools(legacy),'kicker-left','softbox-1x4',{unitMm:.1}),/dimension/);
assert.equal(JSON.stringify({legacy,tools:migrateLightTools(legacy)}),before,'Rejected selection cannot mutate a saved profile');

const lighting={...copy(original),fixtureSizes:fresh,lightTools:freshTools,table:defaultStudioTable(),scale:{version:1,mode:'fixed',unitMm:304.8},roomOverrides:{version:1,values:[]}};
const center=[0,-.9,0];
for(const id of ['kicker-left','kicker-right']){
 const p=lighting.poses.poses.find(p=>p.id===id),oldY=p.pos[1];
 for(const arc of [-180,-90,0,45,90,180]){
  const pos=radialLightPosition(p.pos,id,{arc,radius:6},center),read=radialLightPose(pos,id,center);
  close(read.radius,6);close(read.arc,arc);assert.equal(pos[1],oldY);
 }
 p.pos=radialLightPosition(p.pos,id,{arc:90,radius:6},center);p.aimAtSubject=true;
 close(p.pos[0],id.endsWith('left')?-6:6);close(p.pos[2],0);
}
const materials=studioSceneMaterials(assets,null,lighting),roles=computeStudioFixtureRoles({lighting,...materials,subjectPosition:center}).fixtureRoles;
for(const index of [3,4]){
 const role=roles[index];assert.equal(role.round,false);assert.equal(role.grid,true);assert.equal(role.gobo,false);
 assert.deepEqual([role.halfW*2,role.halfH*2],[1,4]);assert.equal(role.mat,materials.fixtureMaterialBase+index);
 assert.deepEqual(role.pos,lighting.poses.poses[index].pos);
 const parts=[],faces=buildStudioFixtureFaces([role],materials.fixtureMaterialBase+10,p=>parts.push(p),materials.ids);
 assert(faces.flatMap(f=>f.slice(0,3)).flat().every(Number.isFinite));
 const emitter=faces.filter(f=>f[3]===role.mat);assert.equal(emitter.length,2,'Only the real rectangular face emits, never grid bars');
 const area=emitter.reduce((sum,f)=>{const a=f[1].map((v,i)=>v-f[0][i]),b=f[2].map((v,i)=>v-f[0][i]);return sum+Math.hypot(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])/2;},0);close(area,4);
 assert(parts[0].components.some(c=>c.id.startsWith('modifier/grid')));
 assert.equal(parts[0].components.reduce((n,p)=>n+p.triangleCount,0),faces.length);
 assert.equal(snootSphereGuide(role,{center,radius:.03}).state,'unverified','Grid clearance is not mislabeled as a circular open cone');
 assert(Number.isFinite(kickerTableClearanceBound(role,{center,radius:.03},lighting.table,-5).minimumHeight));
}
// The fresh wall-fitting pass must accept rectangular kickers without a radius.
const noPose={...lighting,poses:undefined},defaults=computeStudioFixtureRoles({lighting:noPose,...materials,subjectPosition:center}).fixtureRoles;
const bounds=studioRoomBounds(Number(lighting.controls['room-scale']),tableFloorDrop(lighting.table));
for(const i of [3,4])for(const p of buildStudioFixtureFaces([defaults[i]],materials.fixtureMaterialBase+10,undefined,materials.ids).flatMap(f=>f.slice(0,3))){
 assert(p.every(Number.isFinite));assert(p[1]>=bounds.floorY-1e-8&&p[0]>=bounds.leftX-1e-8&&p[0]<=bounds.rightX+1e-8&&p[2]>=bounds.backZ-1e-8&&p[2]<=bounds.frontZ+1e-8);
}

// Actual emitted scene and MRP1 payload, with unchanged material identities.
const settings={object:authoring.object,lighting,camera:{eye_m:[0,.08,.07],target_m:[0,.08,.18],fov_degrees:18,aperture_radius_m:.002,focus_distance_m:.11}};
const request={schema:'turing.studio-agent-request.v1',operation:'scene',settings},built=await buildStudioAgentProduct(request,{assets}),scene=JSON.parse(new TextDecoder().decode(built.bytes));
const ray=await buildStudioAgentProduct({...request,operation:'ray-payload',photo:{resolution:128,seconds:.1,maxBounces:1}},{assets});
const view=new DataView(ray.bytes.buffer,ray.bytes.byteOffset,ray.bytes.byteLength),config=JSON.parse(new TextDecoder().decode(ray.bytes.slice(16,16+view.getUint32(4,true))));
assert.deepEqual(config.fixtures,scene.fixtures);
for(const i of [3,4])assert.equal(scene.fixtures.material_ids.filter(id=>id===i).length,2,'Physical ray payload has exactly two emissive face triangles per kicker');
assert.deepEqual(scene.fixtures.pbr.slice(10*16+8,10*16+11),[0,0,0],'Housing and grid row remains non-emissive');

// Real DOM event handlers, including replacement, repeat, failed edit and load.
class Element{
 constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.dataset={};}
 append(...children){this.children.push(...children);}replaceChildren(...children){this.children=children;}
 setAttribute(k,v){this[k]=v;}addEventListener(k,fn){(this.events[k]??=[]).push(fn);}fire(k){for(const fn of this.events[k]??[])fn();}
}
const doc={createElement:tag=>new Element(tag,doc)},root=doc.createElement('div'),all=n=>[n,...n.children.filter(n=>typeof n==='object').flatMap(all)],find=id=>all(root).find(n=>n.id===id);
let changes=0,unitMm=304.8,blocked=false;
const ui=createFixtureSizeControls(root,()=>changes++,()=>{if(blocked)throw Error('Measured fixture clearance failed');},{materialNames:names,getUnitMm:()=>unitMm});
assert(ui.ownsControl('fixture-head-kicker-left'));assert.equal(find('fixture-head-kicker-left').value,'softbox-1x4');
const change=(id,value)=>{find(id).value=value;find(id).fire('change');};
change('fixture-head-kicker-left','round');assert.equal(ui.getState().lights[3].head,'round');assert(!ui.tools.getState().attachments.some(t=>t.lightId==='kicker-left'));
change('light-tool-attach-kicker-left','snoot');assert(ui.tools.getState().attachments.some(t=>t.lightId==='kicker-left'&&t.kind==='snoot'));
unitMm=152.4;change('fixture-head-kicker-left','softbox-1x4');assert.equal(ui.getState().lights[3].width,2);assert.equal(find('light-tool-kicker-left-snoot-length'),undefined);assert(find('light-tool-kicker-left-grid-depth'));
assert.deepEqual(compatibleLightTools('kicker-left',ui.getState()),['grid']);
const saved=copy({sizes:ui.getState(),tools:ui.tools.getState()}),count=changes;
blocked=true;change('fixture-head-kicker-left','round');assert.deepEqual(ui.getState(),saved.sizes);assert.deepEqual(ui.tools.getState(),saved.tools);assert.equal(changes,count);assert.equal(find('fixture-head-kicker-left').value,'softbox-1x4');blocked=false;
unitMm=.1;change('fixture-head-kicker-left','softbox-1x4');assert.deepEqual(ui.getState(),saved.sizes);
unitMm=304.8;change('fixture-head-kicker-left','round');ui.setState(saved.sizes);ui.tools.setState(saved.tools);assert.equal(find('fixture-head-kicker-left').value,'softbox-1x4');assert.equal(find('fixture-size-kicker-left-height').value,'2');
change('fixture-head-key-left','softbox-1x4');assert(find('light-tool-key-left-grid-depth'));change('fixture-head-key-left','round');assert.equal(find('light-tool-key-left-grid-depth'),undefined);
console.log(JSON.stringify({passed:true,heads:SIZED_FIXTURES.length,kickerFeet:[1,4],gridCells:[2,8],fixtureTriangles:scene.fixtures.material_ids.length,rayBytes:ray.bytes.length,coverage:'real controls, shared roles/mesh/materials, radial side poses, floor fit, legacy preservation, serialized scene and MRP1; no GPU exposure'}));

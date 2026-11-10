import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {defaultFixtureSizes,applyFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {defaultLightTools,migrateLightTools,validateLightTools,createLightTool,applyLightTools,compatibleLightTools,lightToolFootprint} from '../examples/meowdoku/web/marble_light_tools.js';
import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const context=vm.createContext({STUDIO_LIGHTS,tableLightModifierFaces});
vm.runInContext(source.slice(source.indexOf('function quadFace('),source.indexOf('function digitFaces(')),context);
vm.runInContext(source.slice(source.indexOf('function buildFixtureFaces('),source.indexOf('export function sceneForPhoto(')),context);
const bases=[
 {halfW:.38,halfH:.38},{halfW:.38,halfH:.38},
 {round:true,radius:.3,gobo:true,shroudLength:1.2,shroudRadius:.375},
 {round:true,radius:.2,gobo:true},{round:true,radius:.2,gobo:true},
 {halfW:.3,halfH:.3},{halfW:.3,halfH:.3},...tableLightRoles(),
].map((r,i)=>({pos:[0,0,0],dir:[0,0,1],roleIndex:i,mat:10+i,...r}));
const sizes=defaultFixtureSizes(true),legacy=applyFixtureSizes(bases,sizes),migrated=migrateLightTools(sizes);
assert.equal(migrated.attachments.length,6);
assert.deepEqual(validateLightTools(JSON.parse(JSON.stringify(migrated)),sizes),migrated);
const migratedRoles=applyLightTools(legacy,migrated,sizes);
assert.deepEqual(structuredClone(context.buildFixtureFaces(migratedRoles,99)),structuredClone(context.buildFixtureFaces(legacy,99)),'Migration preserves exact legacy vertices, normals, winding, order, materials and groups');
assert.ok(!migrated.attachments.some(t=>t.lightId==='fill-upper'));
const freshSizes=defaultFixtureSizes(),fresh=defaultLightTools(freshSizes),freshRoles=applyLightTools(applyFixtureSizes(bases,freshSizes),fresh,freshSizes);
assert.deepEqual([freshRoles[6].halfW*2,freshRoles[6].halfH*2],[8,8]);
assert.deepEqual([freshRoles[5].halfW*2,freshRoles[5].halfH*2],[4,2]);
assert.equal(freshRoles[6].gridDepth,1);
assert.deepEqual(compatibleLightTools('background'),['snoot']);assert.deepEqual(compatibleLightTools('fill-upper'),['grid']);
assert.throws(()=>createLightTool('background','grid',sizes),/rectangular/);
assert.throws(()=>createLightTool('key-left','snoot',sizes),/round/);
const sparse={version:1,attachments:[]},removed=applyLightTools(legacy,sparse,sizes);
assert.ok(removed.every(r=>!r.grid&&!r.gobo));
assert.equal(context.buildFixtureFaces([removed[3]],99).length,48);
assert.equal(context.buildFixtureFaces([migratedRoles[3]],99).length,72,'One snoot only, no duplicate legacy tube');
const taper=createLightTool('kicker-left','snoot',sizes);taper.settings.length=.8;taper.settings.exitAperture=.1;taper.offset=.3;
const tapered=applyLightTools(legacy,{version:1,attachments:[taper]},sizes)[3],parts=[];
const faces=context.buildFixtureFaces([tapered],99,p=>parts.push(structuredClone(p)));
assert.equal(faces.length,96,'One mount collar and one tapered snoot');
const group=parts[0].componentGroups.find(g=>g.id==='shroud');
assert.equal(group.intendedShape,'open-ended-polygonal-frustum');
assert.equal(group.triangleCount,48);
const tips=faces.slice(group.firstTriangle,group.firstTriangle+group.triangleCount).flatMap(f=>f.slice(0,3)).filter(p=>Math.abs(p[2]-1.1)<1e-12);
assert.ok(tips.length>0&&tips.every(p=>Math.abs(Math.hypot(p[0],p[1])-.05)<1e-12));
assert.ok(faces.every(f=>Number.isInteger(f[3])&&(f[3]===99||f[3]===13)));
for(const f of faces.slice(-24).filter((_,i)=>i%2===0)){
 const edge=f[1].map((v,i)=>v-f[0][i]);
 assert.ok(Math.abs(edge.reduce((sum,v,i)=>sum+v*f[4][0][i],0))<1e-12,'Taper normals agree with the actual longitudinal slope');
}
const grid=createLightTool('fill-upper','grid',freshSizes);grid.settings={cellsX:4,cellsY:8,depth:1,thickness:.04};grid.offset=.4;
const gridRole=applyLightTools(applyFixtureSizes(bases,freshSizes),{version:1,attachments:[grid]},freshSizes)[6];
const gridFaces=tableLightModifierFaces(gridRole,99),coordinates=gridFaces.flatMap(f=>f.slice(0,3));
assert.equal(gridFaces.length,(4+8+2+4)*12);
assert.ok(Math.abs(Math.max(...coordinates.map(v=>v[0]))-5.002)<1e-12);
assert.ok(Math.abs(Math.min(...coordinates.map(v=>v[2]))+.002)<1e-12,'Apron still starts at the can lip');
assert.ok(Math.abs(Math.max(...coordinates.map(v=>v[2]))-1.402)<1e-12);
assert.deepEqual(lightToolFootprint('fill-upper',freshSizes),{width:8,height:8});
for(const mutate of [s=>s.attachments.push(structuredClone(s.attachments[0])),s=>s.attachments[0].kind='lens',s=>s.attachments[0].offset=NaN,s=>s.attachments[0].settings.exitAperture=0]){
 const bad=structuredClone(migrated);mutate(bad);assert.throws(()=>validateLightTools(bad,sizes));
}
const badGrid={version:1,attachments:[grid]};badGrid.attachments[0].settings.cellsX=33;assert.throws(()=>validateLightTools(badGrid,freshSizes));
console.log('Light tools: exact legacy mesh migration, new fill dimensions/deep grid, compatibility, removal, one tapered snoot, real collar, sealed grid extension and numeric materials passed');

const actualWrongShape=structuredClone(legacy);actualWrongShape[9].round=true;actualWrongShape[9].radius=.9;
assert.throws(()=>applyLightTools(actualWrongShape,migrated,sizes),/Grid requires a rectangular fixture/);
actualWrongShape[9]=structuredClone(legacy[9]);actualWrongShape[3].round=false;
assert.throws(()=>applyLightTools(actualWrongShape,migrated,sizes),/Snoot requires a round fixture/);
console.log('Attachments validate actual emitter shape before writing grid or snoot geometry fields');

// Material assignment follows actual face orientation, not whole-bar recoloring.
const splitGridRole={pos:[0,0,0],dir:[0,0,1],mat:10,grid:true,halfW:1,halfH:1,gridCellsX:2,gridCellsY:2,gridDepth:.3,gridThickness:.02,gridInteriorMaterialId:31,gridOuterMaterialId:32};
const gridParts=[];const split=tableLightModifierFaces(splitGridRole,99,p=>gridParts.push(p));
for(const part of gridParts){const faces=split.slice(part.firstTriangle,part.firstTriangle+part.triangleCount);
 if(part.id.startsWith('grid-apron'))assert.ok(faces.every(f=>f[3]===32));
 else {assert.equal(faces.filter(f=>f[3]===31).length,part.id.endsWith('-1')?4:2);assert.ok(faces.slice(8).every(f=>f[3]===32));}
}
assert.ok(defaultLightTools(sizes).attachments.filter(t=>t.kind==='grid').every(t=>t.settings.interiorMaterial==='polished_silver'));
const old=defaultLightTools(sizes);for(const t of old.attachments)if(t.kind==='grid'){delete t.settings.interiorMaterial;delete t.settings.outerMaterial;}
assert.ok(validateLightTools(old,sizes).attachments.filter(t=>t.kind==='grid').every(t=>t.settings.interiorMaterial==='housing'));
assert.throws(()=>validateLightTools(defaultLightTools(sizes),sizes,'bare',['no-such-material']),/Unavailable grid material/);
console.log('Grid tunnel faces, caps, exterior edges, sealed apron, explicit fresh polished defaults and legacy housing preservation passed');

// Accent regions partition existing faces, rather than adding filter sheets.
{
 const role={pos:[.2,.3,.4],dir:[0,0,1],halfW:1,halfH:1,grid:true,gridCellsX:4,gridCellsY:4,gridDepth:.3,gridThickness:.03,gridInteriorMaterialId:39,gridOuterMaterialId:3,gridAccentMaterialId:40,mat:17};
 const plain=tableLightModifierFaces(role,3);
 const area=faces=>faces.reduce((sum,f)=>{const u=f[1].map((v,i)=>v-f[0][i]),v=f[2].map((v,i)=>v-f[0][i]);return sum+.5*Math.hypot(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);},0);
 const accent={enabled:false,material:'polished_gold',pattern:'checker',size:1,scope:'interiors',width:.2};
 assert.deepEqual(tableLightModifierFaces({...role,gridAccent:accent},3),plain);
 for(const pattern of ['checker','rows','columns'])for(const scope of ['interiors','walls','corners']){
  const faces=tableLightModifierFaces({...role,gridAccent:{...accent,enabled:true,pattern,scope}},3);
  assert(faces.some(f=>f[3]===40));assert(faces.some(f=>f[3]===39));assert(Math.abs(area(faces)-area(plain))<1e-9);
  assert(faces.every(f=>f.slice(0,3).flat().every(Number.isFinite)));
  for(const f of faces.filter(f=>f[3]===40))assert(scope==='walls'?Math.abs(f[4][0][2])===1:f[4][0][2]===0,'Accents remain on the requested physical faces');
  assert.deepEqual(faces,tableLightModifierFaces({...role,gridAccent:{...accent,enabled:true,pattern,scope}},3),'Deterministic repeatable kernel');
 }
 const whole=tableLightModifierFaces({...role,gridAccent:{...accent,enabled:true}},3),corners=tableLightModifierFaces({...role,gridAccent:{...accent,enabled:true,scope:'corners'}},3);
 assert(area(corners.filter(f=>f[3]===40))<area(whole.filter(f=>f[3]===40)));
 const t=createLightTool('underlight','grid',sizes);t.settings.accent={...accent,enabled:true};
 const state={version:1,attachments:[t]};assert.deepEqual(validateLightTools(state,sizes,'bare',['polished_silver','polished_gold']),state);
 for(const change of [{size:0},{size:1.1},{width:0},{pattern:'random'},{scope:'fake'},{material:'missing'}]){
  const bad=structuredClone(state);Object.assign(bad.attachments[0].settings.accent,change);assert.throws(()=>validateLightTools(bad,sizes,'bare',['polished_silver','polished_gold']));
 }
 const applied=applyLightTools(legacy,state,sizes);const under=applied.find(r=>r.roleIndex===9);assert.equal(under.gridAccentMaterial,'polished_gold');
 console.log('Grid accents: all 9 pattern/scope combinations, deterministic material segmentation, surface-area preservation, exact disabled mesh, narrow corner strips, strict validation and shared material routing passed');
}

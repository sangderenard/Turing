import {applyLightTools,defaultLightTools,migrateLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from "../examples/meowdoku/web/marble_fixture_sizes.js";
// Metadata-only fixture refinements against the incumbent primitive sequence.
// No source vertex repair, rounded-coordinate weld tolerance, or hull fitting.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {tableLightRoles,tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {defaultStudioTable,FLOOR_FIXTURE_INDICES} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const modifiers=fs.readFileSync(new URL('../examples/meowdoku/web/marble_table_lights.js',import.meta.url),'utf8');
const html=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
const controls={};
for(const m of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g))controls[m[2]]={value:/value="([^"]*)"/.exec(m[1])?.[1]??'',checked:/\bchecked\b/.test(m[1])};
for(const m of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const opts=[...m[2].matchAll(/<option([^>]*)>/g)],opt=opts.find(o=>o[1].includes('selected'))??opts[0];
  controls[m[1]]={value:opt?/value="([^"]*)"/.exec(opt[1])?.[1]??'':''};
}
const context=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$:id=>controls[id],STUDIO_LIGHTS,FIXTURE_MAT_BASE:10,studioLighting:null,
  studioTableState:()=>defaultStudioTable(),studioRoomBounds,studioRoomScale:()=>3,studioFloorDrop:()=>3,
  FLOOR_FIXTURE_INDICES,tableLightRoles,tableLightModifierFaces,rgb:value=>[1,3,5].map(k=>parseInt(value.slice(k,k+2),16)/255)});
vm.runInContext(source.slice(source.indexOf('const ROOM_FIXTURES ='),source.indexOf('const currentCameraOptions =')),context);
vm.runInContext(source.slice(source.indexOf('function quadFace('),source.indexOf('function digitFaces(')),context);
vm.runInContext(source.slice(source.indexOf('function studioRenderState('),source.indexOf('function surfaceMaterial(')),context);
vm.runInContext(source.slice(source.indexOf('function computeFixtureRoles('),source.indexOf('export function sceneForPhoto(')),context);
// The pre-metadata rendering sequence, using the same unchanged primitives.
vm.runInContext(`function legacyFixtureFaces(roles,material){const faces=[];for(const role of roles){
 const size=role.round?role.radius:Math.max(role.halfW,role.halfH),depth=role.housingDepth??size*1.4;
 if(role.round){faces.push(...tubeHousing(role.pos,role.dir,role.radius*1.25,depth,material,role.mat));
  const emitter=circleFace(role.pos,role.dir,role.radius,role.mat,12,role.mat);
  faces.push(...emitter.map(([a,b,c,m,n,g])=>[a,c,b,m,[n[0],n[2],n[1]],g]));
  if(role.gobo)faces.push(...snootTube(role.pos,role.dir,role.shroudRadius??role.radius*1.1,role.shroudLength??size*2.2,material,role.mat));
 }else{faces.push(...boxHousing(role.pos,role.dir,role.halfW*1.25,role.halfH*1.25,depth,material,role.mat));
  const emitter=quadFace(role.pos,role.dir,role.halfW,role.halfH,role.mat,role.mat);
  faces.push(...emitter.map(([a,b,c,m,n,g])=>[a,c,b,m,[n[0],n[2],n[1]],g]));}
 faces.push(...tableLightModifierFaces(role,material));}return faces;}`,context);
const oldModifier=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),});
vm.runInContext(modifiers.slice(modifiers.indexOf('export const TABLE_STAND ='),modifiers.indexOf('// These three')).replace('export const','const')+
 modifiers.slice(modifiers.indexOf('const add ='),modifiers.indexOf('export function tableLightModifierFaces')).replaceAll('export function','function')+`
function oldModifiers(role,housingMaterial){if(role.tableStand){const axes=[[1,0,0],[0,1,0],[0,0,1]];
 return [...solidBox(add(role.pos,[0,1,0],-TABLE_STAND.baseOffset),axes,[TABLE_STAND.halfWidth,TABLE_STAND.baseHalfHeight,TABLE_STAND.halfLength],housingMaterial,role.mat),
 ...solidBox(add(role.pos,[0,1,0],-.14),axes,[.025,.12,.025],housingMaterial,role.mat)];}
 if(!role.grid)return [];const axes=basis(role.dir),depth=.20,thickness=.025,faces=[];
 for(const [axis,count,span,other]of [[0,6,role.halfW,role.halfH],[1,8,role.halfH,role.halfW]])for(let i=0;i<=count;i++){
  const center=add(add(role.pos,axes[axis],-span+2*span*i/count),role.dir,depth/2+.002);
  const half=axis===0?[thickness/2,other+thickness/2,depth/2]:[other+thickness/2,thickness/2,depth/2];
  faces.push(...solidBox(center,axes,half,housingMaterial,role.mat));}return faces;}`,oldModifier);
const roles=structuredClone(vm.runInContext('computeFixtureRoles().fixtureRoles',context)),parts=[];
const actual=context.buildFixtureFaces(roles,99,part=>parts.push(structuredClone(part)));
assert.deepEqual(structuredClone(actual),structuredClone(context.legacyFixtureFaces(roles,99)),'Exact pre-metadata geometry/order/materials/normals/groups retained');
for(const role of [...roles,...roles.filter(r=>r.grid).map(r=>({...r,dir:[.3,.4,Math.sqrt(.75)]}))]) {
  const before=structuredClone(oldModifier.oldModifiers(role,99)),after=tableLightModifierFaces(role,99);
  assert.deepEqual(after.slice(0,before.length),before,'Existing stand/grid-bar geometry retained');
  assert.equal(after.length-before.length,role.grid?48:0,'Enabled grid additionally seals the can lip');
}
for(const part of parts){
  assert.equal(part.id,STUDIO_LIGHTS[roles[parts.indexOf(part)].roleIndex].id);
  const ownership=new Array(part.faces.length).fill(0),ids=new Set();
  for(const component of part.components){
    assert(!ids.has(component.id));ids.add(component.id);assert.equal(component.bodyId,part.bodyId);
    assert.equal(component.contactAdmission,'unverified');assert(!('convex' in component));assert(!('closed' in component));
    for(let i=component.firstTriangle;i<component.firstTriangle+component.triangleCount;i++){assert(i<ownership.length);ownership[i]++;}
  }
  assert(ownership.every(n=>n===1),'Every source triangle belongs to exactly one nested component');
}
const stats=(faces,convert=x=>x)=>{
  const vertices=new Map(),triangles=[],edges=new Map();
  for(const face of faces){const t=face.slice(0,3).map(p=>{const k=p.map(convert).join(',');if(!vertices.has(k))vertices.set(k,vertices.size);return vertices.get(k);});
    const i=triangles.length;triangles.push(t);for(let e=0;e<3;e++){const a=t[e],b=t[(e+1)%3],key=a<b?`${a},${b}`:`${b},${a}`;
      if(!edges.has(key))edges.set(key,[]);edges.get(key).push({triangle:i,direction:a<b?1:-1});}}
  const adjacency=triangles.map(()=>[]);for(const use of edges.values())for(const a of use)for(const b of use)if(a!==b)adjacency[a.triangle].push(b.triangle);
  const visited=new Set();let connected=0;for(let i=0;i<triangles.length;i++)if(!visited.has(i)){connected++;const pending=[i];while(pending.length){const j=pending.pop();if(visited.has(j))continue;visited.add(j);pending.push(...adjacency[j]);}}
  return {triangles:faces.length,vertices:vertices.size,edgeConnectedComponents:connected,
    boundaryEdges:[...edges.values()].filter(v=>v.length===1).length,nonmanifoldEdges:[...edges.values()].filter(v=>v.length>2).length,
    sameDirectionPairedEdges:[...edges.values()].filter(v=>v.length===2&&v[0].direction===v[1].direction).length};
};
const topology=parts.map(part=>({id:part.id,source:stats(part.faces),nativeF32:stats(part.faces,Math.fround),
  groups:part.componentGroups.map(group=>({id:group.id,source:stats(part.faces.slice(group.firstTriangle,group.firstTriangle+group.triangleCount)),
    nativeF32:stats(part.faces.slice(group.firstTriangle,group.firstTriangle+group.triangleCount),Math.fround)}))}));
if(process.argv.includes('--json'))console.log(JSON.stringify({roles,parts,topology}));
else {console.log(`PASS fixture metadata: ${parts.length} roles, ${actual.length} owned faces, ${parts.reduce((s,p)=>s+p.components.length,0)} exact component ranges`);console.log(JSON.stringify(topology));}

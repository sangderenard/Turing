import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {defaultShopLights,setShopLightSource,validateShopLights,shopLightEmitterAllocations,writeShopLightEmitterRows,shopLightEmissionRecords,studioShopLightGeometry} from '../examples/meowdoku/web/marble_shop_lights.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
const profiles=JSON.parse(fs.readFileSync('build/dot-named-emitters-20261110/emitter_profiles.json','utf8'));
const ids={metal_silver:1,black_matte_paint:2},bounds=studioRoomBounds(6,3.98);
let state=defaultShopLights();for(const f of state.fixtures)state=setShopLightSource(state,f.id,'led-array');
for(const f of state.fixtures)f.exposureMode='continuous';
const rows=shopLightEmitterAllocations(state,3),data=new Float32Array(19*16);
assert.deepEqual(rows.map(r=>r.materialId),[3,11,4,12]);
writeShopLightEmitterRows(data,16,3,state,{profiles});
const geometry=studioShopLightGeometry(state,ids,rows,{bounds,unitMm:304.8});
assert.deepEqual(geometry.parts.map(p=>p.components.filter(c=>c.id.startsWith('led-')).length),[768,672]);
assert.equal(geometry.emitters.length,4);
assert.equal(new Set(geometry.emitters.map(e=>e.groupId)).size,4,'Every panel/color group owns a distinct triangle ID');
for(const emitter of geometry.emitters){
 assert(emitter.triangles.every(i=>geometry.faces[i][3]===emitter.materialId));
 assert(emitter.triangles.every(i=>geometry.faces[i][5]===emitter.groupId),'Actual emitting triangles carry their group ID');
 assert(emitter.surfaceAreaM2>0);assert(emitter.position.every(Number.isFinite));
}
assert.notDeepEqual(Array.from(data.slice(3*16+8,3*16+11)),Array.from(data.slice(11*16+8,11*16+11)),'Distinct original native LED profiles survive RGB projection');
const compact=new Map(rows.map((r,i)=>[r.materialId,i]));
const records=shopLightEmissionRecords(state,3,compact,profiles);
assert.deepEqual(records.records.map(r=>r.profile),['led_cool_white','led_warm_white','led_cool_white','led_warm_white']);
assert(records.records.every(r=>r.gain===1));
// Execute the existing application registration/uniform writer, not a test clone.
// This is an offline CPU boundary assertion; it does not claim a new GPU run.
for(const file of ['examples/meowdoku/web/marble_studio.js','build/meowdoku/web/marble_studio.js']){
 const text=fs.readFileSync(file,'utf8'),start=text.indexOf('    const coneAngles='),end=text.indexOf('    field(P,"uLightCalibration"',start);
 assert(start>=0&&end>start);
 const uniforms=new Map(),put=(key,value)=>uniforms.set(key,Array.isArray(value)||ArrayBuffer.isView(value)?Array.from(value):value);
 vm.runInNewContext(text.slice(start,end),{fixtureRoles:[],roleColor:[],roleStrength:[],displayedShop:geometry,
  tables:{pbr:{data,table:{floats_per_record:16}}},view:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
  transformPoint:(_,p)=>p,P:0,loc:(_,name)=>name,
  gl:{uniform1i:put,uniform1f:put,uniform2fv:put,uniform3fv:put}});
 assert.equal(uniforms.get('uNumLights'),4);
 for(const [i,e]of geometry.emitters.entries()){
  assert.equal(uniforms.get(`uLightGroupId[${i}]`),e.groupId);
  assert.deepEqual(uniforms.get(`uLightColor[${i}]`),Array.from(data.slice(e.materialId*16+8,e.materialId*16+11)));
  assert.equal(uniforms.get(`uLightIntensity[${i}]`),e.surfaceAreaStudio);
 }
 assert.notDeepEqual(uniforms.get('uLightColor[0]'),uniforms.get('uLightColor[1]'));
}
// Same original source strength per unit emitting area, not a brightness boost.
const dark=structuredClone(state);for(const f of dark.fixtures)f.dimmer=0;
writeShopLightEmitterRows(data,16,3,dark,{profiles});
for(const row of rows)assert.deepEqual(Array.from(data.slice(row.materialId*16+8,row.materialId*16+11)),[0,0,0]);
assert.deepEqual(studioShopLightGeometry(dark,ids,rows,{bounds,unitMm:304.8}),geometry);
const old=structuredClone(state);for(const f of old.fixtures)delete f.ledLayout;
const legacy=validateShopLights(old,Object.keys(ids));assert(legacy.fixtures.every(f=>f.ledLayout==='legacy'));
assert.equal(shopLightEmitterAllocations(legacy,3).length,2);
assert(studioShopLightGeometry(legacy,ids,shopLightEmitterAllocations(legacy,3),{bounds,unitMm:304.8}).parts.every(p=>p.components.filter(c=>c.id.startsWith('led-')).length===32));
// Every package stays closed, with outward winding and physically separated disks.
for(const part of geometry.parts)for(const component of part.components.filter(c=>c.id.startsWith('led-'))){
 const faces=part.faces.slice(component.firstTriangle,component.firstTriangle+component.triangleCount);
 assert.equal(faces.length,48);
 const pts=faces.flatMap(f=>f.slice(0,3));
 assert(Math.abs(Math.max(...pts.map(p=>p[0]))-Math.min(...pts.map(p=>p[0]))-1/30)<1e-10);
 for(const [a,b,c,,normals]of faces){
  const u=b.map((v,i)=>v-a[i]),v=c.map((v,i)=>v-a[i]);
  const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
  assert(normals.every(q=>q.reduce((s,x,i)=>s+x*n[i],0)>0));
 }
}
if(process.argv.includes('--export')){
 writeShopLightEmitterRows(data,16,3,state,{profiles,channel:'exposure'});
 fs.writeFileSync('build/dot-led-density-20261110/native-fixtures.json',JSON.stringify({
  vertices:geometry.faces.flatMap(f=>f.slice(0,3).flatMap(p=>p.map(v=>v*.3048))),
  material_ids:geometry.faces.map(f=>f[3]),pbr:Array.from(data),
  emitter_profiles:shopLightEmissionRecords(state,3,new Map(rows.map(r=>[r.materialId,r.materialId])),profiles),
 }));
}
console.log('PASS dense LEDs: 768/672 physical circular packages; distinct cool/warm shared material and native records, dimmer zero, unchanged legacy geometry, normals and dimensions.');

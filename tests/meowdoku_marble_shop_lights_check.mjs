import assert from 'node:assert/strict';
import {SHOP_LIGHT_MODELS,defaultShopLights,createShopLight,validateShopLights,setShopLightSource,
  writeShopLightEmitterRows as writeRows,studioShopLightGeometry,shopLightEmitterAllocations,shopLightEmissionRecords} from '../examples/meowdoku/web/marble_shop_lights.js';
import {writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
const shopEmissionProfiles={records:['fluorescent_cool_white','fluorescent_warm_white','led_cool_white','led_warm_white'].map(key=>({key,revision:'a'.repeat(64),preview_rgb:[1,1,1],filter_rgb_matrix:[[1,0,0],[0,1,0],[0,0,1]]}))};
const writeShopLightEmitterRows=(data,stride,base,state,options={})=>writeRows(data,stride,base,state,{profiles:shopEmissionProfiles,...options});
const ids={metal_silver:1,black_matte_paint:2},names=Object.keys(ids),bounds=studioRoomBounds(6,3.98),state=defaultShopLights();
const copy=v=>structuredClone(v),near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
assert.deepEqual(validateShopLights(undefined,[]),{version:1,backingMaterial:'metal_silver',topMaterial:'black_matte_paint',fixtures:[]});
assert.deepEqual(validateShopLights(JSON.parse(JSON.stringify(state)),names),state);
assert.throws(()=>validateShopLights(state,[]),/catalogue/);
for(const mutation of [s=>s.fixtures[0].model='none',s=>s.fixtures[0].dimmer=1.01,s=>s.fixtures[0].dimmer=NaN,
  s=>s.fixtures[0].source='point',s=>s.fixtures[0].filter=[1,2,1],s=>s.fixtures[0].ceilingDropFeet=-1,
  s=>s.fixtures[0].x=Infinity,s=>s.fixtures[1].id=s.fixtures[0].id,s=>s.fixtures[0].exposureMode='steady']) {
  const invalid=copy(state);mutation(invalid);assert.throws(()=>validateShopLights(invalid,names));
}
assert.throws(()=>writeRows(new Float32Array(5*16),16,3,state),/Missing canonical emission profile/);
const pbr=new Float32Array(5*16);pbr.set([.91,.93,.96,.1,1,0,1.5,1],16);pbr.set([.025,.025,.025,.8,0,0,1.5,1],32);
const matrixProfiles={records:shopEmissionProfiles.records.map(r=>({...r,filter_rgb_matrix:[[2,-1,.5],[-.5,2,0],[0,0,3]]}))};
const matrixState=copy(state);matrixState.fixtures[0].filter=[.2,.3,.4];matrixState.fixtures[0].dimmer=.5;
const matrixRows=new Float32Array(5*16);writeRows(matrixRows,16,3,matrixState,{profiles:matrixProfiles});
assert.deepEqual(Array.from(matrixRows.slice(56,59)),[.15,.25,.6].map(Math.fround),'Signed native spectral projection precedes nonnegative clamp and dimmer');
matrixState.fixtures[0].filter=[0,1,0];writeRows(matrixRows,16,3,matrixState,{profiles:matrixProfiles});
assert.equal(matrixRows[56],0,'Negative projected channel clips only after spectral filtering');
const originalRows=pbr.slice(0,3*16),rows=writeShopLightEmitterRows(pbr,16,3,state);
assert.deepEqual(pbr.slice(0,3*16),originalRows,'Authored catalogue rows are never replaced');
const reference=new Float32Array(pbr.length);writeStudioFixtureRows(reference,16,3,[[1,1,1],[1,1,1]],[1,1]);
assert.deepEqual(pbr.slice(3*16),reference.slice(3*16),'One authoritative Studio emitter writer');
assert.deepEqual(rows.map(r=>[r.id,r.materialId,r.groupId]),[['shop-long',3,3],['shop-wide',4,4]]);
const before=copy({state,ids,rows,bounds}),geometry=studioShopLightGeometry(state,ids,rows,{bounds,unitMm:304.8});
assert.deepEqual({state,ids,rows,bounds},before);
assert.equal(geometry.faces.length,360);assert.equal(geometry.parts.length,2);assert.equal(geometry.emitters.length,2);
assert.deepEqual(geometry.emitters.map(e=>e.triangles.length),[48,96]);
function closed(faces,convert=x=>x) {
  const edges=new Map();let volume=0;
  for(const [a,b,c,,normals]of faces) {
    const n=cross(sub(b,a),sub(c,a));assert(Math.hypot(...n)>0);assert(normals.every(v=>dot(n,v)>0),'Winding matches smooth/flat normals');
    volume+=dot(a,cross(b,c))/6;
    for(const [p,q]of [[a,b],[b,c],[c,a]]) {
      const ka=p.map(convert).join(','),kb=q.map(convert).join(','),key=[ka,kb].sort().join('|');
      const uses=edges.get(key)??[];uses.push(ka<kb?1:-1);edges.set(key,uses);
    }
  }
  assert(volume>0);assert([...edges.values()].every(v=>v.length===2&&v[0]===-v[1]));
}
for(const [partIndex,part]of geometry.parts.entries()) {
  const spec=SHOP_LIGHT_MODELS[state.fixtures[partIndex].model];
  assert.equal(part.components.filter(c=>c.id.startsWith('tube-')).length,spec.tubes);
  const ownership=new Array(part.faces.length).fill(0);
  for(const component of part.components) {
    const faces=part.faces.slice(component.firstTriangle,component.firstTriangle+component.triangleCount);
    closed(faces);closed(faces,Math.fround);
    for(let i=component.firstTriangle;i<component.firstTriangle+component.triangleCount;i++)ownership[i]++;
    assert.equal(component.contactAdmission,'unverified');
    if(component.id.startsWith('tube-')) {
      const points=faces.flatMap(f=>f.slice(0,3));near(Math.max(...points.map(p=>p[0]))-Math.min(...points.map(p=>p[0])),spec.lengthFeet);
      assert.equal(faces.filter(f=>f[3]===rows[partIndex].materialId).length,24);
    }
    if(component.id.startsWith('mount-'))near(Math.max(...faces.flatMap(f=>f.slice(0,3).map(p=>p[1]))),bounds.ceilingY);
    if(component.id==='housing')for(const f of faces)assert.equal(f[3],f[4][0][1]===-1?ids.metal_silver:ids.black_matte_paint);
  }
  assert(ownership.every(v=>v===1));
  const expectedArea=2*(1/24)*12*Math.sin(Math.PI/12)*spec.lengthFeet*spec.tubes;
  near(geometry.emitters[partIndex].surfaceAreaStudio,expectedArea);
  near(geometry.emitters[partIndex].surfaceAreaM2,expectedArea*.3048**2);
  assert(geometry.emitters[partIndex].triangles.every(i=>geometry.faces[i][3]===rows[partIndex].materialId));
}
const loweredBounds={...bounds,ceilingY:bounds.ceilingY-2};
const lowered=studioShopLightGeometry(state,ids,rows,{bounds:loweredBounds,unitMm:304.8});
geometry.faces.forEach((face,i)=>face.slice(0,3).forEach((p,j)=>near(lowered.faces[i][j][1],p[1]-2)));
const alternateUnits=152.4,converted=defaultShopLights(alternateUnits),larger=Object.fromEntries(Object.entries(bounds).map(([k,v])=>[k,v*2]));
const convertedGeometry=studioShopLightGeometry(converted,ids,rows,{bounds:larger,unitMm:alternateUnits});
geometry.faces.forEach((face,i)=>face.slice(0,3).forEach((p,j)=>p.forEach((v,k)=>near(convertedGeometry.faces[i][j][k]*alternateUnits,v*304.8))));
geometry.emitters.forEach((e,i)=>near(e.surfaceAreaM2,convertedGeometry.emitters[i].surfaceAreaM2));
assert.throws(()=>studioShopLightGeometry(defaultShopLights(10),ids,rows,{bounds,unitMm:10}),/shop-long:.*does not fit/);
assert.throws(()=>studioShopLightGeometry(state,ids,[{...rows[0],materialId:1},rows[1]],{bounds,unitMm:304.8}),/catalogue rows/);
assert.throws(()=>studioShopLightGeometry(state,ids,[rows[0],{...rows[1],groupId:rows[0].groupId}],{bounds,unitMm:304.8}),/unique/);
const dimmed=copy(state);dimmed.fixtures[0].dimmer=.25;dimmed.fixtures[0].filter=[.1,.2,.8];
writeShopLightEmitterRows(pbr,16,3,dimmed);near(pbr[3*16+8],Math.fround(.025));near(pbr[3*16+9],Math.fround(.05));near(pbr[3*16+10],Math.fround(.2));
assert.deepEqual(studioShopLightGeometry(dimmed,ids,rows,{bounds,unitMm:304.8}).faces,geometry.faces,'Dimming alters real shared emission, never geometry');
writeShopLightEmitterRows(pbr,16,3,state,{channel:'exposure'});
for(const row of rows)assert.deepEqual(Array.from(pbr.slice(row.materialId*16+8,row.materialId*16+11)),[0,0,0]);
assert.deepEqual(studioShopLightGeometry(state,ids,rows,{bounds,unitMm:304.8}).faces,geometry.faces,'Exposure off retains physical tubes/housing');
const burst=copy(state);burst.fixtures[0].exposureMode='burst';const beforeBurst=pbr.slice();
assert.throws(()=>writeShopLightEmitterRows(pbr,16,3,burst,{channel:'exposure'}),/authoritative timing-packet/);assert.deepEqual(pbr,beforeBurst);
const continuous=copy(dimmed);continuous.fixtures[0].exposureMode='continuous';
assert.deepEqual(validateShopLights(JSON.parse(JSON.stringify(continuous)),names),continuous);
writeShopLightEmitterRows(pbr,16,3,continuous,{channel:'exposure'});
assert.deepEqual(Array.from(pbr.slice(56,59)),[.025,.05,.2].map(Math.fround));
assert.deepEqual(Array.from(pbr.slice(72,75)),[0,0,0],'Default-off sibling stays off');
continuous.fixtures[0].enabled=false;
writeShopLightEmitterRows(pbr,16,3,continuous,{channel:'exposure'});
assert.deepEqual(Array.from(pbr.slice(56,59)),[0,0,0],'Override does not enable disabled fixtures');
continuous.fixtures[0].enabled=true;continuous.fixtures[0].dimmer=0;
writeShopLightEmitterRows(pbr,16,3,continuous,{channel:'exposure'});
assert.deepEqual(Array.from(pbr.slice(56,59)),[0,0,0],'Zero dimmer stays off');
assert.throws(()=>writeShopLightEmitterRows(new Float32Array(64),16,3,state),/Allocate/);
let leds=copy(state);for(const f of leds.fixtures)f.ledLayout='legacy';for(const f of state.fixtures)leds=setShopLightSource(leds,f.id,'led-array');
const ledGeometry=studioShopLightGeometry(leds,ids,rows,{bounds,unitMm:304.8});
assert.equal(ledGeometry.faces.length,3144);
assert(ledGeometry.parts.every(p=>p.components.filter(c=>c.id.startsWith('led-')).length===32));
assert(ledGeometry.emitters.every(e=>e.triangles.length===384&&e.triangles.every(i=>ledGeometry.faces[i][4][0][1]===-1)));
for(const part of ledGeometry.parts)for(const c of part.components)closed(part.faces.slice(c.firstTriangle,c.firstTriangle+c.triangleCount));
assert.deepEqual(state.fixtures.map(f=>f.source),['tubes','tubes']);
const disabled=copy(state);disabled.fixtures[0].enabled=false;
assert.equal(studioShopLightGeometry(disabled,ids,rows,{bounds,unitMm:304.8}).parts[0].id,'shop-wide');
assert.equal(writeShopLightEmitterRows(pbr,16,3,disabled)[1].materialId,4,'Disabled fixture never renumbers another source');
assert.deepEqual(studioShopLightGeometry(undefined,{},[],{}),{faces:[],parts:[],emitters:[]});
writeShopLightEmitterRows(pbr,16,3,state);
assert.deepEqual(shopLightEmitterAllocations(state,3).map(r=>r.materialId),[3,4]);
for(const emitter of geometry.emitters){
  const acc=[0,0,0];let area=0;
  for(const index of emitter.triangles){const f=geometry.faces[index],a=Math.hypot(...cross(sub(f[1],f[0]),sub(f[2],f[0])))/2;
    area+=a;for(let axis=0;axis<3;axis++)acc[axis]+=a*(f[0][axis]+f[1][axis]+f[2][axis])/3;}
  for(let axis=0;axis<3;axis++)near(emitter.position[axis],acc[axis]/area);
}
const packed=g=>({vertices:g.faces.flatMap(f=>f.slice(0,3).flatMap(p=>p.map(v=>v*.3048))),material_ids:g.faces.map(f=>f[3]),pbr:Array.from(pbr)});
if(process.argv.includes('--json'))console.log(JSON.stringify({tubes:packed(geometry),leds:packed(ledGeometry)}));
else console.log('PASS isolated shop lights: physical8ft/two-tube and4ft/four-tube meshes, ceiling mounts, silver backing/black top, swappable32-LED arrays, shared dimmer rows, exact emitter ownership/area, normals/closure, unit conversion, legacy empty and honest burst gate.');

// Source level is the existing scalar emitter strength; dimmer only attenuates it.
const levelState=copy(state);levelState.fixtures[0].sourceLevel=80;levelState.fixtures[0].dimmer=.25;levelState.fixtures[0].exposureMode='continuous';
const levelRows=new Float32Array(5*16);writeShopLightEmitterRows(levelRows,16,3,levelState);
assert.deepEqual(Array.from(levelRows.slice(56,59)),[20,20,20]);
assert.equal(shopLightEmissionRecords(levelState,3,new Map([[3,0],[4,1]]),shopEmissionProfiles).records[0].gain,20);
const oldLevel=copy(state);delete oldLevel.fixtures[0].sourceLevel;assert.equal(validateShopLights(oldLevel,names).fixtures[0].sourceLevel,1);
for(const value of [-1,401,Infinity,NaN]){const bad=copy(state);bad.fixtures[0].sourceLevel=value;assert.throws(()=>validateShopLights(bad,names));}
levelState.fixtures[0].dimmer=0;writeShopLightEmitterRows(levelRows,16,3,levelState);assert.equal(levelRows[56],0);
if(!process.argv.includes('--json'))console.log('PASS source level: shared material/native gain product, legacy1, bounds and zero dimmer');

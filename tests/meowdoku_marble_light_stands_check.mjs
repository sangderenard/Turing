import * as sceneBuilder from '../examples/meowdoku/web/studio_scene_builder.js';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {DEFAULT_LIGHT_STAND_MATERIAL,LIGHT_STAND_DIMENSIONS,lightStandLayout,lightStandFaces} from '../examples/meowdoku/web/marble_light_stands.js';
import {tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';

const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const primitives=vm.createContext({});
Object.assign(primitives,sceneBuilder,{buildFixtureFaces:(roles,housing,onPart)=>sceneBuilder.buildStudioFixtureFaces(roles,housing,onPart,primitives.ids??{})});
const unit=v=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const room=studioRoomBounds(3,3.98);
const rect={pos:[0,3,0],dir:[0,-1,0],halfW:4,halfH:4,round:false,grid:true,
  gridDepth:1,gridCellsX:8,gridCellsY:8,gridThickness:.025,mat:4,roleIndex:6};
const round={pos:[2,1,3],dir:unit([-2,-1,-3]),radius:.2,round:true,mat:7,roleIndex:3};
const actualVertices=role=>{
  const depth=role.housingDepth??(role.round?role.radius:Math.max(role.halfW,role.halfH))*1.4;
  const faces=role.round?primitives.tubeHousing(role.pos,role.dir,role.radius*1.25,depth,99,role.mat):
    primitives.boxHousing(role.pos,role.dir,role.halfW*1.25,role.halfH*1.25,depth,99,role.mat);
  faces.push(...(role.round?primitives.circleFace(role.pos,role.dir,role.radius,role.mat):
    primitives.quadFace(role.pos,role.dir,role.halfW,role.halfH,role.mat)),...tableLightModifierFaces(role,99));
  return structuredClone(faces.flatMap(face=>face.slice(0,3)));
};
function topology(faces,convert=x=>x) {
  const ids=new Map(),edges=new Map();
  for(const face of faces) {
    const vertices=face.slice(0,3).map(p=>{const key=p.map(convert).join(',');if(!ids.has(key))ids.set(key,ids.size);return ids.get(key);});
    assert.equal(new Set(vertices).size,3,'No collapsed triangle');
    for(let i=0;i<3;i++) {
      const a=vertices[i],b=vertices[(i+1)%3],key=[Math.min(a,b),Math.max(a,b)].join(',');
      const uses=edges.get(key)??[];uses.push(a<b?1:-1);edges.set(key,uses);
    }
  }
  assert([...edges.values()].every(uses=>uses.length===2&&uses[0]===-uses[1]),'Every capped pipe edge has two opposing uses');
}
function verify(role,bounds=room,vertices=actualVertices(role)) {
  const before=structuredClone({role,bounds,vertices}),options={bounds,fixtureVertices:vertices};
  const layout=lightStandLayout(role,bounds.floorY,options);
  assert(layout.available,layout.reason);
  const components=[],faces=lightStandFaces(role,bounds.floorY,71,c=>components.push(c),options);
  assert.equal(components.length,5);
  assert.deepEqual(components.map(c=>c.id),['arm','mast','foot-0','foot-1','foot-2']);
  assert.equal(faces.length,5*4*LIGHT_STAND_DIMENSIONS.segments);
  assert(faces.every(face=>face[3]===71&&face[5]===role.mat),'One shared material, original fixture group');
  for(const [i,component]of components.entries()) {
    assert.equal(component.firstTriangle,i*48);assert.equal(component.triangleCount,48);
    assert.equal(component.contactAdmission,'unverified');assert(!('closed' in component));assert(!('convex' in component));
    const part=faces.slice(component.firstTriangle,component.firstTriangle+component.triangleCount);
    topology(part);topology(part,Math.fround);
    for(const [a,b,c,,normals]of part) {
      const geometric=cross(sub(b,a),sub(c,a));
      assert(Math.hypot(...geometric)>0);
      assert(normals.every(n=>dot(geometric,n)>0),'Ray winding agrees with Phong normals');
      assert(normals.every(n=>Math.abs(Math.hypot(...n)-1)<1e-12));
    }
    const points=part.flatMap(face=>face.slice(0,3));
    assert(points.every(p=>p.every(Number.isFinite)));
    assert(points.every(p=>p[0]>=bounds.leftX-1e-12&&p[0]<=bounds.rightX+1e-12&&
      p[2]>=bounds.backZ-1e-12&&p[2]<=bounds.frontZ+1e-12&&p[1]>=bounds.floorY-1e-12&&p[1]<=bounds.ceilingY+1e-12));
    if(component.id.startsWith('foot-'))assert(Math.abs(Math.min(...points.map(p=>p[1]))-bounds.floorY)<1e-12,'Each pipe foot reaches floor');
    if(component.id==='arm')assert(points.every(p=>dot(sub(p,role.pos),unit(role.dir))<0),'Mount arm remains behind emitting plane');
    else {
      const support=Math.max(...vertices.map(p=>dot(p,layout.exclusionDirection)));
      assert(points.every(p=>dot(p,layout.exclusionDirection)>support),'Mast/feet stay outside actual fixture projection');
    }
  }
  assert.deepEqual({role,bounds,vertices},before,'Building a stand never edits saved pose/size/bounds/mesh');
  return {layout,faces};
}

assert.equal(DEFAULT_LIGHT_STAND_MATERIAL,'black_matte_paint');
verify(rect); // Requested 8-by-8 overhead, one-unit-deep real egg-crate.
verify(round);
verify({...round,dir:[0,1,0],pos:[0,-4.5,0]});
verify({...round,dir:[0,-1,0]});
verify({...round,dir:[1,0,0]});
verify({...round,radius:.01,dir:[0,1,0],pos:[0,0,0]});
verify({...rect,pos:[0,-1,0],dir:[0,1,0],halfW:2,halfH:2,grid:false});
for(const dir of [[.2,.9,-.3],[-.7,-.4,.1],[.8,-.001,-.6],[1e-8,1,0]])
  verify({...round,dir:unit(dir)});

// A room wall blocks the first side. The same fixture uses the other side,
// without moving its authored position, orientation, or accessories.
const nearWall={...round,pos:[-1.8,1,0],dir:[0,0,1]};
const smallRoom={leftX:-2,rightX:2,backZ:-2,frontZ:2,floorY:-2,ceilingY:3};
const selected=verify(nearWall,smallRoom).layout;
assert(selected.mastTop[0]>nearWall.pos[0]);
assert(selected.mastTop[0]+LIGHT_STAND_DIMENSIONS.spread<smallRoom.rightX);

// Actual geometry overrides the conservative built-in envelope. Forcing broad
// tool bounds forces the stand farther out, instead of crossing that footprint.
const largerVertices=[...actualVertices(round),[-4,-2,-4],[-4,5,4],[4,-2,-4],[4,5,4]];
assert.equal(verify(round,room,largerVertices).layout.footprintSource,'fixture-vertices');
assert.equal(lightStandLayout(round,room.floorY,{bounds:room}).footprintSource,'conservative-fixture-envelope');

// Room and low-floor failure are explicit and do not silently omit a support.
const tiny={leftX:-.2,rightX:.2,backZ:-.2,frontZ:.2,floorY:-2,ceilingY:3};
const cramped={...round,pos:[0,0,0],dir:[0,1,0]};
assert.equal(lightStandLayout(cramped,-2,{bounds:tiny}).available,false);
assert.match(lightStandLayout(cramped,-2,{bounds:tiny}).reason,/No tested side or rear/);
assert.throws(()=>lightStandFaces(cramped,-2,71,undefined,{bounds:tiny}),/No tested side or rear/);
const low={...round,pos:[0,-2.1,0],dir:[0,1,0]};
assert.match(lightStandLayout(low,-2).reason,/too low/);
assert.throws(()=>lightStandFaces(low,-2,71),/too low/);
assert.throws(()=>lightStandLayout(round,NaN),/floor/);
assert.throws(()=>lightStandLayout(round,0,{bounds:room}),/same room floor/);
assert.throws(()=>lightStandLayout({...round,dir:[0,0,0]},-2),/nonzero/);
assert.throws(()=>lightStandLayout({...round,radius:-1},-2),/dimensions/);
assert.throws(()=>lightStandLayout(round,-2,{fixtureVertices:[]}),/vertices/);
assert.throws(()=>lightStandLayout(round,-2,{fixtureVertices:[[1,NaN,2]]}),/vertices/);
assert.throws(()=>lightStandFaces(round,-2,-1),/material/);

console.log('PASS isolated light stands: shared triangle tuples, five capped pipes, real mesh footprint, aperture clearance, room-side selection, floor contact, outward winding, exact/native-f32 component closure, and immutable saved inputs.');

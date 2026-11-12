import * as sceneBuilder from '../examples/meowdoku/web/studio_scene_builder.js';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {lightCeilingSupportLayout,lightCeilingSupportFaces} from '../examples/meowdoku/web/marble_light_ceiling_support.js';
import {LIGHT_STAND_DIMENSIONS} from '../examples/meowdoku/web/marble_light_stands.js';
import {defaultFixtureSizes,applyFixtureSizes} from '../examples/meowdoku/web/marble_fixture_sizes.js';
import {defaultLightTools,applyLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {STUDIO_LIGHTS} from '../examples/meowdoku/web/marble_light_strengths.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
import {defaultStudioScale} from '../examples/meowdoku/web/marble_studio_scale.js';

// The current production builder supplies every housing, emitter and tool point.
// No test-only approximate softbox stands in for its actual shared geometry.
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const primitives=vm.createContext({STUDIO_LIGHTS,tableLightModifierFaces,ids:{polished_silver:39,polished_gold:40}});
Object.assign(primitives,sceneBuilder,{buildFixtureFaces:(roles,housing,onPart)=>sceneBuilder.buildStudioFixtureFaces(roles,housing,onPart,primitives.ids??{})});
const builderStart=source.indexOf('function buildFixtureFaces(');
Object.assign(primitives,sceneBuilder,{buildFixtureFaces:(roles,housing,onPart)=>sceneBuilder.buildStudioFixtureFaces(roles,housing,onPart,primitives.ids??{})});
const unit=v=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const room=studioRoomBounds(6,3.98),sizes=defaultFixtureSizes(),tools=defaultLightTools(sizes);
function upperAt(y) {
  const pos=[0,y,.3];
  return applyLightTools(applyFixtureSizes([{pos,dir:unit(pos.map(v=>-v)),halfW:.3,halfH:.3,
    round:false,roleIndex:6,mat:8}],sizes),tools,sizes)[0];
}
const actualVertices=role=>structuredClone(primitives.buildFixtureFaces([role],99).flatMap(face=>face.slice(0,3)));
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
  assert([...edges.values()].every(uses=>uses.length===2&&uses[0]===-uses[1]),'Every pipe edge has two opposing uses');
}
function verify(role,bounds=room,vertices=actualVertices(role)) {
  const before=structuredClone({role,bounds,vertices}),options={bounds,fixtureVertices:vertices};
  const layout=lightCeilingSupportLayout(role,bounds.ceilingY,options);
  assert(layout.available,layout.reason);
  const components=[],faces=lightCeilingSupportFaces(role,bounds.ceilingY,71,c=>components.push(c),options);
  assert.equal(components.length,12);
  assert.equal(faces.length,12*4*LIGHT_STAND_DIMENSIONS.segments);
  assert.equal(layout.footprintSource,'fixture-vertices');
  assert(faces.every(face=>face[3]===71&&face[5]===role.mat),'Shared material and existing fixture group');
  assert(components.every(c=>c.contactAdmission==='unverified'&&!('closed' in c)&&!('convex' in c)));
  assert.equal(layout.mounts.length,4);
  assert.equal(new Set(layout.mounts.map(m=>m.ceiling.join(','))).size,4,'Four distinct ceiling anchors');
  for(const [i,component]of components.entries()) {
    assert.equal(component.firstTriangle,i*48);assert.equal(component.triangleCount,48);
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
    if(component.id.startsWith('attachment-'))
      assert(points.every(p=>dot(sub(p,role.pos),unit(role.dir))<0),'Pins stay behind the emitting plane');
    else assert(points.every(p=>dot(p,layout.exclusionDirection)>layout.fixtureSupport),'Cable/plate meshes clear the complete fixture support plane');
    if(component.id.startsWith('ceiling-mount-'))
      assert.equal(Math.max(...points.map(p=>p[1])),bounds.ceilingY,'Each ceiling plate reaches the actual room ceiling');
  }
  assert.deepEqual({role,bounds,vertices},before,'Derived supports do not rewrite saved pose, size, bounds or mesh');
  return {layout,faces};
}

assert.equal(defaultStudioScale().unitMm,304.8,'Default one scene unit is one physical foot');
const measured=[];
for(const y of [3.2,5.2,7.2]) {
  const role=upperAt(y),vertices=actualVertices(role),result=verify(role,room,vertices);
  assert.equal(role.halfW*2,8);assert.equal(role.halfH*2,8);assert.equal(role.gridDepth,1);
  assert.equal(result.layout.housingSurface,'back');
  measured.push({centerY:y,fixtureBottom:Math.min(...vertices.map(p=>p[1])),fixtureTop:result.layout.fixtureTop,
    ceilingY:room.ceilingY,clearance:result.layout.fixtureCeilingClearance});
}
const rect={pos:[0,1,0],dir:[0,-1,0],halfW:1,halfH:.5,round:false,roleIndex:6,mat:8,grid:false};
verify(rect);
for(const dir of [[1,0,0],[0,0,1],[.4,-.9,.3],[.4,.9,.3],[-.8,.1,.6],[-.8,-.1,-.6]])verify({...rect,dir:unit(dir)});
verify({...rect,halfW:.01,halfH:.02,housingDepth:.003});

// Complete tool geometry, not the naked housing, controls cable separation.
const wideVertices=[...actualVertices(rect),[0,4,0]];
const wide=verify(rect,room,wideVertices).layout;
assert(wide.mounts.every(m=>m.cableStart[1]>4));

function unavailable(role,bounds,vertices,pattern) {
  const options={bounds,fixtureVertices:vertices??actualVertices(role)};
  const layout=lightCeilingSupportLayout(role,bounds.ceilingY,options);
  assert.equal(layout.available,false);assert.deepEqual(layout.segments,[]);assert.match(layout.reason,pattern);
  let callbacks=0;
  assert.throws(()=>lightCeilingSupportFaces(role,bounds.ceilingY,71,()=>callbacks++,options),pattern);
  assert.equal(callbacks,0,'Failure emits no partial component callbacks');
}
unavailable({...rect,dir:[0,1,0]},room,undefined,/Aim the softbox/);
unavailable(rect,{...room,ceilingY:2},undefined,/reach the ceiling/);
const top=Math.max(...actualVertices(rect).map(p=>p[1]));
unavailable(rect,{...room,ceilingY:top+.001},undefined,/cables and mounts do not fit/);
unavailable(rect,{...room,leftX:-.2,rightX:.2},undefined,/extend outside/);
unavailable(rect,room,[...actualVertices(rect),[room.rightX+1,1,0]],/extend outside/);
assert.throws(()=>lightCeilingSupportLayout(rect,NaN,{bounds:room,fixtureVertices:actualVertices(rect)}),/height/);
assert.throws(()=>lightCeilingSupportLayout(rect,1,{bounds:room,fixtureVertices:actualVertices(rect)}),/same room ceiling/);
assert.throws(()=>lightCeilingSupportLayout(rect,room.ceilingY,{bounds:room}),/actual fixture vertices/);
assert.throws(()=>lightCeilingSupportLayout(rect,room.ceilingY,{bounds:room,fixtureVertices:[]}),/actual fixture vertices/);
assert.throws(()=>lightCeilingSupportLayout({...rect,dir:[0,0,0]},room.ceilingY,{bounds:room,fixtureVertices:actualVertices(rect)}),/nonzero/);
assert.throws(()=>lightCeilingSupportLayout({...rect,round:true},room.ceilingY,{bounds:room,fixtureVertices:actualVertices(rect)}),/rectangular/);
assert.throws(()=>lightCeilingSupportFaces(rect,room.ceilingY,-1),/material/);
console.log('Default upper softbox clearance, full production mesh:',JSON.stringify(measured));
console.log('PASS ceiling support: actual fixture/tool vertices, four ceiling cables, rear/side housing mounts, shared pipe primitive/material tuples, explicit room/ceiling failures, immutable inputs, f32 component closure and outward normals.');

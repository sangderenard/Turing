import assert from 'node:assert/strict';
import { diceSolid } from '../examples/meowdoku/web/dice_mesh.js';
import { marbleMass } from '../examples/meowdoku/web/marble_mass.js';

const kinds=['d4','d6','d8','d10','d12','d20'];
for(const kind of kinds) {
  const count=+kind.slice(1), meshes={};
  for(const mode of ['flat','raised','recessed']) {
    const mesh=diceSolid(kind,{shellMaterial:1,markMaterial:2,markingMode:mode,
      compensateRaised:false});
    meshes[mode]=mesh;
    assert.equal(mesh.faceInfo.length,count);
    assert.equal(mesh.glass.length,mesh.triangleCount);
    assert(mesh.glass.some(face=>face[3]===2),'lettering must use its own Phong material');
    const props=marbleMass(mesh,{radiusM:0.01,densityByRole:{shell:2500}});
    assert(props.massKg>0&&props.inertiaKgM2.every((row,i)=>row[i]>0));
    mesh.mass=props.massKg;
    // Directed edge closure checks both watertight tessellation and winding.
    const edges=new Map(), key=p=>p.map(x=>(Math.abs(x)<1e-8?0:x).toFixed(8)).join(',');
    for(const [a,b,c] of mesh.glass) for(const [u,v] of [[a,b],[b,c],[c,a]]) {
      const U=key(u),V=key(v),edge=U<V?`${U}|${V}`:`${V}|${U}`;
      const state=edges.get(edge)||{count:0,orientation:0};
      state.count++;state.orientation+=U<V?1:-1;edges.set(edge,state);
    }
    for(const edge of edges.values()) assert.deepEqual(edge,{count:2,orientation:0},`${kind} ${mode}: open or inverted edge`);
  }
  assert(meshes.recessed.mass<meshes.flat.mass&&meshes.flat.mass<meshes.raised.mass,
    `${kind}: relief must change integrated mass`);
  if(kind==='d4') {
    const perVertex=new Map();
    for(const face of meshes.flat.faceInfo) {
      assert.equal(face.markings.length,3);
      for(const mark of face.markings) perVertex.set(mark.number,(perVertex.get(mark.number)||0)+1);
    }
    assert.deepEqual([...perVertex.entries()].sort((a,b)=>a[0]-b[0]),[[1,3],[2,3],[3,3],[4,3]]);
  } else {
    const infos=meshes.flat.faceInfo, names=infos.map(f=>f.number).sort((a,b)=>a-b);
    assert.deepEqual(names,Array.from({length:count},(_,i)=>kind==='d10'?i:i+1));
    for(const face of infos) {
      const opposite=infos.find(other=>other!==face&&face.normal.reduce((s,x,i)=>s+x*other.normal[i],0)<-0.999);
      assert(opposite,`${kind}: missing opposite`);
      assert.equal(face.number+opposite.number,kind==='d10'?9:count+1);
    }
    if(kind==='d10') {
      assert.deepEqual(infos.filter(x=>x.normal[1]>0).map(x=>x.number).sort((a,b)=>a-b),[0,2,4,6,8]);
      assert.deepEqual(infos.filter(x=>x.normal[1]<0).map(x=>x.number).sort((a,b)=>a-b),[1,3,5,7,9]);
      const around=infos.filter(x=>x.normal[1]>0).sort((a,b)=>
        Math.atan2(a.normal[2],a.normal[0])-Math.atan2(b.normal[2],b.normal[0]));
      assert.deepEqual(around.map(x=>x.number),[0,8,2,6,4]);
    }
  }
  console.log(`${kind}: closed mesh, face numbering, marking material and relief mass passed`);
}

const bulk=(mesh)=>marbleMass(mesh,{radiusM:0.01,densityByRole:{shell:2500}});
const rounded=diceSolid('d6',{shellMaterial:1,markMaterial:2,bevelRadius:0.05});
const sharp=diceSolid('d6',{shellMaterial:1,markMaterial:2,bevelRadius:0});
const h=1/Math.sqrt(3),r=0.05;
let edgeSamples=0;
for(const triangle of rounded.glass) for(const p of triangle.slice(0,3)) {
  if(p[0]>h-r+1e-5&&p[1]>h-r+1e-5&&Math.abs(p[2])<h-r-0.02) {
    assert(Math.abs(Math.hypot(p[0]-(h-r),p[1]-(h-r))-r)<1e-7,
      'beveled cube edge must be a circular radius-r cross-section');
    edgeSamples++;
  }
}
assert(edgeSamples>100,'edge fillet must have resolved samples');
let cap=rounded.glass[0][0];
for(const triangle of rounded.glass) for(const p of triangle.slice(0,3))
  if(p[0]+p[1]+p[2]>cap[0]+cap[1]+cap[2]) cap=p;
assert(Math.abs(Math.hypot(...cap.map(x=>x-(h-r)))-r)<1e-7,
  'beveled cube corner must be a radius-r spherical cap');
for(const face of sharp.faceInfo) assert(sharp.glass.slice(face.start,face.start+face.count)
  .filter(triangle=>triangle[3]===2).length>=100,'face glyph must have adequate marked triangles');
assert(bulk(rounded).massKg<bulk(sharp).massKg,'bevel must remove real volume');
const recessedFace=diceSolid('d6',{shellMaterial:1,markMaterial:2,faceRelief:0.015});
assert(bulk(recessedFace).massKg<bulk(sharp).massKg,'face relief must remove real volume');
const raised=diceSolid('d6',{shellMaterial:1,markMaterial:2,markingMode:'raised',
  reliefDepth:0.025,bevelRadius:0.05,faceRelief:0.01,compensateRaised:true});
const raisedUncompensated=diceSolid('d6',{shellMaterial:1,markMaterial:2,markingMode:'raised',
  reliefDepth:0.025,bevelRadius:0.05,faceRelief:0.01,compensateRaised:false});
assert(bulk(raised).massKg<bulk(raisedUncompensated).massKg,
  'raised glyph compensation must inset substrate and alter mass');
let raisedEdgeSamples=0;
for(const triangle of raised.glass) for(const p of triangle.slice(0,3))
  if(p[0]>h-r+1e-5&&p[1]>h-r+1e-5&&Math.abs(p[2])<h-r-0.02) {
    assert(Math.abs(Math.hypot(p[0]-(h-r),p[1]-(h-r))-r)<1e-7,
      'face relief must leave the circular edge fillet intact');
    raisedEdgeSamples++;
  }
assert(raisedEdgeSamples>100);
const support=diceSolid('d6',{shellMaterial:1,markMaterial:2}).faceInfo.map(face=>{
  const p=sharp.glass[face.start][0];return face.normal.reduce((s,x,i)=>s+x*p[i],0);
});
for(const face of raised.faceInfo) {
  const max=Math.max(...raised.glass.slice(face.start,face.start+face.count)
    .flatMap(tri=>tri.slice(0,3).map(p=>face.normal.reduce((s,x,i)=>s+x*p[i],0))));
  assert(max<=support[face.faceIndex]+1e-8,'compensated mark must not protrude past support plane');
}
assert(bulk(raised).inertiaKgM2.every((row,i)=>row[i]>0));
console.log('bevel, face relief and raised support compensation alter closed-mesh mechanics');

for(const kind of kinds) {
  const mesh=diceSolid(kind,{shellMaterial:1,markMaterial:2,markingMode:'raised',
    reliefDepth:0.025,bevelRadius:0.05,faceRelief:0.01,compensateRaised:true,subdivisions:24});
  const heights=mesh.faceInfo.map(face=>{
    let max=-Infinity;
    for(const triangle of mesh.glass) for(const p of triangle.slice(0,3))
      max=Math.max(max,face.normal.reduce((s,x,i)=>s+x*p[i],0));
    return max;
  });
  assert(Math.max(...heights)-Math.min(...heights)<1e-6,`${kind}: unequal face support heights`);
  assert(bulk(mesh).massKg>0&&bulk(mesh).inertiaKgM2.every((row,i)=>row[i]>0));
  console.log(`${kind}: equal compensated support height, spread ${Math.max(...heights)-Math.min(...heights)}`);
}

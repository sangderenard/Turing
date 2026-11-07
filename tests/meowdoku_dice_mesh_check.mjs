import assert from 'node:assert/strict';
import { diceSolid } from '../examples/meowdoku/web/dice_mesh.js';
import { marbleMass } from '../examples/meowdoku/web/marble_mass.js';

const kinds=['d4','d6','d8','d10','d12','d20'];
for(const kind of kinds) {
  const count=+kind.slice(1), meshes={};
  for(const mode of ['flat','raised','recessed']) {
    const mesh=diceSolid(kind,{shellMaterial:1,markMaterial:2,markingMode:mode});
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
  }
  console.log(`${kind}: closed mesh, face numbering, marking material and relief mass passed`);
}

import assert from 'node:assert/strict';
import {catEyeMarble} from '../examples/meowdoku/web/marble_mesh.js';
import {cutConvexShell} from '../examples/meowdoku/web/gem_cutter.js';

const original=catEyeMarble({glass:1,eye:2,accent:3}).glass;
const volume=faces=>Math.abs(faces.reduce((sum,[a,b,c])=>sum+
  (a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6,0));
const key=p=>p.map(x=>(Math.abs(x)<1e-8?0:x).toFixed(7)).join(',');
for(const style of ['brilliant','step'])for(const strength of [0.5,1]) {
  const faces=cutConvexShell(original,{style,strength,capMaterial:1});
  assert.deepEqual(faces,cutConvexShell(original,{style,strength,capMaterial:1}),`${style} is deterministic`);
  assert(volume(faces)>0&&volume(faces)<volume(original),`${style} must remove stock`);
  const edges=new Map();
  for(const [a,b,c] of faces)for(const [u,v] of [[a,b],[b,c],[c,a]]) {
    const U=key(u),V=key(v),edge=U<V?`${U}|${V}`:`${V}|${U}`;
    const state=edges.get(edge)||{count:0,orientation:0};
    state.count++;state.orientation+=U<V?1:-1;edges.set(edge,state);
  }
  for(const edge of edges.values())assert.deepEqual(edge,{count:2,orientation:0},`${style} ${strength}: open or inverted edge`);
  console.log(`${style} ${strength}: ${faces.length} triangles, volume ${volume(faces).toFixed(5)}`);
}

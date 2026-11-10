import assert from 'node:assert/strict';
import {wideLightSupportRequired,wideLightSupportLayout,wideLightSupportFaces} from '../examples/meowdoku/web/marble_wide_light_supports.js';
import {backdropSupportLayout,backdropSupportFaces} from '../examples/meowdoku/web/marble_backdrop_supports.js';
import {backdropFaces,defaultBackdropLayers} from '../examples/meowdoku/web/marble_backdrop.js';
import {studioRoomBounds} from '../examples/meowdoku/web/marble_room.js';
const bounds=studioRoomBounds(6,3.98),floorY=bounds.floorY;
const role={pos:[0,3,0],dir:[0,0,1],halfW:2,halfH:1,round:false,mat:50,roleIndex:5};
const vertices=[];for(const x of [-2.5,2.5])for(const y of [1.75,4.25])for(const z of [-2.8,0])vertices.push([x,y,z]);
const options={bounds,fixtureVertices:vertices,unitMm:304.8};
assert.equal(wideLightSupportRequired(role,304.8),true);
assert.equal(wideLightSupportRequired({...role,halfW:1.99},304.8),false);
assert.equal(wideLightSupportRequired({...role,halfW:4},152.4),true);
assert.equal(wideLightSupportRequired({...role,round:true},304.8),false);
assert.throws(()=>wideLightSupportRequired(role,undefined),/real Studio/);
const saved=structuredClone({role,vertices,options}),layout=wideLightSupportLayout(role,floorY,options);
assert.equal(layout.available,true);assert.equal(layout.segments.length,10);
assert.deepEqual(layout.layouts[0].exclusionDirection,layout.layouts[1].exclusionDirection.map(v=>-v));
assert(layout.layouts[0].mastTop[0]>2.5&&layout.layouts[1].mastTop[0]<-2.5);
const components=[],faces=wideLightSupportFaces(role,floorY,38,c=>components.push(c),options);
assert.equal(faces.length,480);assert.equal(components.length,10);assert(faces.every(f=>f[3]===38&&f[5]===50));
assert.deepEqual({role,vertices,options},saved);
const narrowBounds={...bounds,leftX:-2.7};
assert.equal(wideLightSupportLayout(role,floorY,{...options,bounds:narrowBounds}).available,false);
let calls=0;assert.throws(()=>wideLightSupportFaces(role,floorY,38,()=>calls++,{...options,bounds:narrowBounds}),/opposite.*does not fit/);assert.equal(calls,0);
const wrong={...options,supportDirections:[[0,0,1]]};
// A caller cannot replace the module's required opposed side pair.
assert.deepEqual(wideLightSupportLayout(role,floorY,wrong).segments,layout.segments);

const ids={white_plaster:36},backdrop=backdropFaces({count:3,material:'white_plaster',sweep:true,layers:defaultBackdropLayers(),wallZ:bounds.backZ,floorY},ids);
const actual=backdrop.flatMap(f=>f.slice(0,3)),before=structuredClone(actual);
const holder=backdropSupportLayout(actual,floorY,{bounds,unitMm:304.8});
assert.equal(holder.available,true);assert.equal(holder.segments.length,11);
assert.equal(holder.segments.filter(s=>s.id.endsWith('/mast')).length,2);
assert.equal(holder.segments.filter(s=>s.id.includes('/foot-')).length,6);
assert.equal(holder.segments.filter(s=>s.id==='crossbar').length,1);
const parts=[],holderFaces=backdropSupportFaces(actual,floorY,38,p=>parts.push(p),{bounds,unitMm:304.8});
assert.equal(holderFaces.length,528);assert.equal(parts.length,1);assert.equal(parts[0].components.length,11);
assert.deepEqual(actual,before);assert(holderFaces.every(f=>f[3]===38));
for(const f of holderFaces)for(const p of f.slice(0,3)){
 assert(p[0]>=bounds.leftX&&p[0]<=bounds.rightX);assert(p[1]>=bounds.floorY&&p[1]<=bounds.ceilingY);assert(p[2]>=bounds.backZ&&p[2]<=bounds.frontZ);
}
const crossbar=holder.segments.find(s=>s.id==='crossbar');
const contact=[0,holder.top,holder.topZ];
assert(holderFaces.slice(0,48).some(f=>{const a=f[1].map((v,i)=>v-f[0][i]),b=f[2].map((v,i)=>v-f[0][i]);const n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];return n[0]===0&&Math.abs(contact.reduce((sum,v,i)=>sum+(v-f[0][i])*n[i],0))<1e-12;}),'Backdrop top contacts an actual crossbar facet, not only its circular bound');
assert.equal(backdropSupportLayout(actual,floorY,{bounds:{...bounds,ceilingY:3},unitMm:304.8}).available,false);
assert.throws(()=>backdropSupportFaces(actual,floorY,38,null,{bounds:{...bounds,leftX:-4.01},unitMm:304.8}),/Backdrop holder:/);
assert.deepEqual(backdropSupportFaces([],floorY,38,null,{}),[]);
assert.throws(()=>backdropSupportLayout(actual,floorY,{bounds,unitMm:0}),/millimetres/);
function closedAndOutward(faces){
 const edges=new Map();
 for(const [a,b,c,,normals]of faces){
  const ab=b.map((v,i)=>v-a[i]),ac=c.map((v,i)=>v-a[i]);
  const n=[ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0]];
  assert(Math.hypot(...n)>0);assert(normals.every(m=>m.reduce((sum,v,i)=>sum+v*n[i],0)>0));
  for(const [p,q]of [[a,b],[b,c],[c,a]]){const x=p.map(Math.fround).join(','),y=q.map(Math.fround).join(','),key=[x,y].sort().join('|');const uses=edges.get(key)??[];uses.push(x<y?1:-1);edges.set(key,uses);}
 }
 assert([...edges.values()].every(v=>v.length===2&&v[0]===-v[1]));
}
for(const c of components)closedAndOutward(faces.slice(c.firstTriangle,c.firstTriangle+c.triangleCount));
for(const c of parts[0].components)closedAndOutward(holderFaces.slice(c.firstTriangle,c.firstTriangle+c.triangleCount));
console.log('PASS dual wide-box supports at four physical feet; opposite sides, real vertices, atomic bounds failure; actual-layer backdrop frame with two tripods/crossbar, shared pipe/materials, unchanged input, native-f32 closure and outward normals.');

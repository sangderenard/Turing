// Phong study: intersect the existing convex shell with actual cutting planes.
// Each pass clips the current boundary and closes the new facet with a planar cap.
const EPS = 1e-8;
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub = (a,b) => a.map((v,i)=>v-b[i]);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = a => { const d=Math.hypot(...a);return a.map(v=>v/d); };
const lerp = (a,b,t) => a.map((v,i)=>v+(b[i]-v)*t);
const area2 = (a,b,c) => Math.hypot(...cross(sub(b,a),sub(c,a)));

function cut(polygons, {normal,limit}, capMaterial) {
  const kept=[], boundary=new Map();
  const key=p=>p.map(v=>Math.round(v*1e8)).join(',');
  for(const face of polygons) {
    const next=[],hits=[];
    for(let i=0;i<face.vertices.length;i++) {
      const a=face.vertices[i],b=face.vertices[(i+1)%face.vertices.length];
      const da=dot(normal,a.p)-limit,db=dot(normal,b.p)-limit;
      if(da<=EPS)next.push(a);
      if((da < -EPS && db > EPS)||(da > EPS && db < -EPS)) {
        const t=da/(da-db),v={p:lerp(a.p,b.p,t),n:unit(lerp(a.n,b.n,t))};
        next.push(v);hits.push(v.p);
      } else if(Math.abs(da)<=EPS && Math.abs(db)>EPS)hits.push(a.p);
    }
    if(next.length>=3)kept.push({...face,vertices:next});
    if(hits.length>=2)for(const p of hits)boundary.set(key(p),p);
  }
  const points=[...boundary.values()];
  if(points.length<3)return kept;
  const center=[0,1,2].map(i=>points.reduce((s,p)=>s+p[i],0)/points.length);
  const tangent=unit(cross(normal,Math.abs(normal[1])<0.9?[0,1,0]:[1,0,0]));
  const bitangent=cross(normal,tangent);
  points.sort((a,b)=>Math.atan2(dot(sub(a,center),bitangent),dot(sub(a,center),tangent))-
    Math.atan2(dot(sub(b,center),bitangent),dot(sub(b,center),tangent)));
  // A single cap polygon keeps the clipped solid closed for all later cuts.
  kept.push({material:capMaterial,vertices:points.map(p=>({p,n:normal}))});
  return kept;
}

export function gemCutPlanes(style='brilliant',strength=1) {
  if(!['brilliant','step'].includes(style))throw new RangeError('unsupported gem cut');
  if(!Number.isFinite(strength)||strength<=0||strength>1)throw new RangeError('cut strength must be in (0, 1]');
  const planes=[];
  const add=(normal,limit)=>planes.push({normal,limit:1+(limit-1)*strength});
  add([0,1,0],0.55);add([0,-1,0],0.84);
  const sides=16;
  for(let i=0;i<sides;i++) {
    const theta=2*Math.PI*i/sides,radial=[Math.cos(theta),0,Math.sin(theta)];
    add(radial,0.94);
    if(style==='brilliant') {
      add([radial[0],0.88,radial[2]],0.94);
      add([radial[0],-1.10,radial[2]],0.94);
      const mid=theta+Math.PI/sides,r=[Math.cos(mid),0,Math.sin(mid)];
      add([r[0],0.68,r[2]],0.84);
      add([r[0],-0.82,r[2]],0.88);
    } else {
      add([radial[0],0.34,radial[2]],0.90);
      add([radial[0],-0.38,radial[2]],0.90);
      add([radial[0],0.78,radial[2]],0.83);
      add([radial[0],-0.88,radial[2]],0.83);
    }
  }
  return planes;
}

export function cutConvexShell(faces,{style='brilliant',strength=1,capMaterial}={}) {
  if(!Array.isArray(faces)||!faces.length)throw new RangeError('a closed shell is required');
  if(!Number.isInteger(capMaterial)||capMaterial<0)throw new RangeError('cap material is required');
  let polygons=faces.map(face=>({material:face[3],vertices:[0,1,2].map(i=>({p:face[i],n:face[4][i]}))}));
  for(const plane of gemCutPlanes(style,strength))polygons=cut(polygons,plane,capMaterial);
  const result=[];
  for(const face of polygons)for(let i=1;i+1<face.vertices.length;i++) {
    const a=face.vertices[0],b=face.vertices[i],c=face.vertices[i+1];
    if(area2(a.p,b.p,c.p)<1e-12)continue;
    result.push([a.p,b.p,c.p,face.material,[a.n,b.n,c.n]]);
  }
  if(result.length<4)throw new Error('the cut removed the entire shell');
  return result;
}

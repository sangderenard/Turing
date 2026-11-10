import assert from "node:assert/strict";
import {makeStudioCourse,studioCourseMarbleFit,STUDIO_COURSE_KINDS} from "../examples/meowdoku/web/marble_studio_courses.js";
const sub=(a,b)=>a.map((v,k)=>v-b[k]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a.reduce((v,x,k)=>v+x*b[k],0);
const key=ids=>[...ids].sort((a,b)=>a-b).join(":");
function checkClosed(triangles) {
  const edges=new Map();
  for(const ids of triangles)for(let k=0;k<3;k++) {
    const a=ids[k],b=ids[(k+1)%3],id=key([a,b]);
    const record=edges.get(id)||{count:0,winding:0};
    record.count++;record.winding+=a<b?1:-1;edges.set(id,record);
  }
  for(const record of edges.values())assert.deepEqual(record,{count:2,winding:0});
}
const signedVolume=(triangles,vertices)=>triangles.reduce((sum,ids)=>{
  const [a,b,c]=ids.map(i=>vertices[i]);return sum+dot(a,cross(b,c))/6;
},0);
let courses=0;
for(const kind of STUDIO_COURSE_KINDS)for(const [width,length,segments] of [[5.2,3,48],[16,10,48],[2,2,16],[2,20,32],[20,2,128]]) {
  const c=makeStudioCourse({kind,stage:{width,length,baseY:-1},materialId:7,railMaterialId:9,segments});
  courses++;
  assert.equal(c.collisionReady,false);
  assert.equal(c.collisionRequirement,"authored-compound-union-manifold");
  assert.equal(c.faces.length,c.faceRecords.length);
  assert.ok(c.maxGrade>0&&c.maxGrade<=.05);
  assert.equal(c.clearWidth,Math.min(width,length)*.10);
  assert.ok(c.bounds.low[0]>=-width/2&&c.bounds.high[0]<=width/2);
  assert.ok(c.bounds.low[2]>=-length/2&&c.bounds.high[2]<=length/2);
  assert.equal(c.bounds.low[1],-1);
  assert.ok(c.bounds.high[0]-c.bounds.low[0]>.70*width);
  assert.ok(c.route.every((p,i)=>i===0||p[1]<=c.route[i-1][1]+1e-14));
  const internal=new Map(),surfacePoints=new Map(),partMap=new Map(c.parts.map(p=>[p.id,p]));
  for(const part of c.parts) {
    assert.equal(part.bodyId,c.id);assert.equal(part.closed,true);assert.equal(part.convex,true);
    assert.equal(part.construction,"vertical-triangle-prism");
    assert.equal(part.vertexIndices.length,6);assert.equal(part.triangles.length,8);
    assert.equal(part.triangleMeta.length,8);checkClosed(part.triangles);
    assert.ok(signedVolume(part.triangles,c.vertices)>0);
    part.triangles.forEach((ids,index)=>{
      const [a,b,d]=ids.map(i=>c.vertices[i]),normal=cross(sub(b,a),sub(d,a));
      assert.ok(Math.hypot(...normal)>0);
      // Test the claimed convex boundary against ALL six actual vertices.
      for(const vertex of part.vertexIndices.map(i=>c.vertices[i]))
        assert.ok(dot(normal,sub(vertex,a))<=1e-9*Math.hypot(...normal));
      const meta=part.triangleMeta[index];assert.ok(c.surfaces[meta.surfaceId]);
      if(meta.boundary==="internal") {
        assert.ok(partMap.has(meta.adjacentPartId));
        const id=key(ids);if(!internal.has(id))internal.set(id,[]);
        internal.get(id).push({part:part.id,ids,meta});
      }
    });
  }
  for(const records of internal.values()) {
    assert.equal(records.length,2);
    assert.equal(records[0].meta.surfaceId,records[1].meta.surfaceId);
    assert.equal(records[0].meta.adjacentPartId,records[1].part);
    assert.equal(records[1].meta.adjacentPartId,records[0].part);
    const normal=ids=>{const [a,b,d]=ids.map(i=>c.vertices[i]);return cross(sub(b,a),sub(d,a));};
    assert.ok(dot(normal(records[0].ids),normal(records[1].ids))<0);
  }
  c.faces.forEach((face,index)=>{
    const record=c.faceRecords[index],part=partMap.get(record.partId);
    assert.deepEqual(record.indices,part.triangles[record.triangleIndex]);
    assert.equal(part.triangleMeta[record.triangleIndex].boundary,"exterior");
    assert.equal(face[3],part.materialId);
    record.indices.forEach((id,k)=>assert.equal(face[k],c.vertices[id]));
    const n=face[4][0];assert.ok(Math.abs(Math.hypot(...n)-1)<1e-12);
    assert.ok(dot(n,cross(sub(face[1],face[0]),sub(face[2],face[0])))>0);
    if(!surfacePoints.has(record.surfaceId))surfacePoints.set(record.surfaceId,[]);
    surfacePoints.get(record.surfaceId).push(...face.slice(0,3));
  });
  checkClosed(c.faceRecords.map(record=>record.indices));
  const wholeVolume=signedVolume(c.faceRecords.map(r=>r.indices),c.vertices);
  const piecesVolume=c.parts.reduce((sum,part)=>sum+signedVolume(part.triangles,c.vertices),0);
  assert.ok(wholeVolume>0);assert.ok(Math.abs(wholeVolume-piecesVolume)<1e-10*Math.max(1,wholeVolume));
  // Certificates are authored, but check that their actual exported geometry
  // still satisfies them. Test tolerance is not used to create ownership.
  for(const [id,plane] of Object.entries(c.planes))for(const p of surfacePoints.get(id)||[])
    assert.ok(Math.abs(dot(plane.normal,sub(p,plane.point)))<1e-10);
  if(kind==="gentle-s") {
    const run=c.surfaces[`${c.id}/run/incline`];assert.equal(run.kind,"planar");assert.equal(run.footprint,"nonconvex");
  } else {
    for(const zone of ["entry","exit"]) {
      const run=c.surfaces[`${c.id}/run/${zone}`];assert.equal(run.kind,"planar");assert.equal(run.footprint,"convex");
    }
  }
  for(const group of Object.values(c.surfaceGroups))assert.equal(group.collisionReady,false);
  assert.deepEqual(c.spawn.surfacePoint,c.route[c.spawn.routeIndex]);
  assert.deepEqual(c.finish.surfacePoint,c.route[c.finish.routeIndex]);
  const before=JSON.stringify(c);
  assert.equal(studioCourseMarbleFit(c,c.clearWidth*.1).fits,true);
  assert.equal(studioCourseMarbleFit(c,c.clearWidth).fits,false);
  assert.equal(JSON.stringify(c),before);
}
const settings={kind:"gentle-s",stage:{width:5.2,length:3,baseY:-1},materialId:7,segments:16};
assert.deepEqual(makeStudioCourse(settings),makeStudioCourse({...settings,radius:200}));
for(const bad of [{kind:"loop"},{stage:{width:1,length:3,baseY:0}},{stage:{width:3,length:3,baseY:NaN}},
  {segments:15},{segments:16.5},{segments:129},{materialId:-1},{materialId:1.1},{railMaterialId:NaN}])
  assert.throws(()=>makeStudioCourse({...settings,...bad}));
for(const radius of [0,-1,NaN,Infinity])assert.throws(()=>studioCourseMarbleFit(makeStudioCourse(settings),radius));
console.log(`PASS ${courses} stage courses: shared exact faces, closed convex partitions, exterior union, ownership, gentle grade and fixed scale`);

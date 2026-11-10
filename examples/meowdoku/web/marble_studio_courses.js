// Authored stage geometry, in the SAME Studio coordinates as room/table faces.
// This module does not run physics. Each closed convex construction part uses
// the exact shared vertices below. Their compound union still needs a native
// contact manifold: summing the parts would create false seam forces.
export const STUDIO_COURSE_KINDS = ["gentle-s", "broad-return"];
const WIDTH_FRACTION = .10, GRADE = .018, RAIL_HEIGHT = .55;
const PROFILE = [-.65, -.55, -.5, 0, .5, .55, .65];
const HEIGHTS = [RAIL_HEIGHT, RAIL_HEIGHT, 0, 0, 0, RAIL_HEIGHT, RAIL_HEIGHT];
const sub = (a,b) => a.map((v,k)=>v-b[k]);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = v => {const n=Math.hypot(...v);if(!(n>0))throw new Error("Degenerate course face");return v.map(x=>x/n);};
const edgeKey = (a,b) => a<b?`${a}:${b}`:`${b}:${a}`;

function coursePath(kind, width, length, segments) {
  const span=Math.min(width,length), result=[];
  const put=(x,z,tx,tz,zone)=>result.push({x,z,tangent:unit([tx,0,tz]),zone});
  if(kind==="gentle-s") {
    for(let i=0;i<=segments;i++) {
      const t=i/segments;
      put(width*.72*(t-.5),span*.10*Math.sin(2*Math.PI*t),
        width*.72,span*.20*Math.PI*Math.cos(2*Math.PI*t),"s");
    }
  } else {
    const radius=.28*span, start=-.36*width, turn=.36*width-radius;
    const straight=turn-start, arc=Math.PI*radius, total=straight*2+arc;
    const straightSteps=Math.max(2,Math.round(segments*straight/total));
    const arcSteps=Math.max(8,segments-2*straightSteps);
    for(let i=0;i<=straightSteps;i++)put(start+straight*i/straightSteps,-radius,1,0,"entry");
    for(let i=1;i<=arcSteps;i++) {
      const angle=-Math.PI/2+Math.PI*i/arcSteps;
      put(turn+radius*Math.cos(angle),radius*Math.sin(angle),-Math.sin(angle),Math.cos(angle),"turn");
    }
    for(let i=1;i<=straightSteps;i++)put(turn-straight*i/straightSteps,radius,-1,0,"exit");
  }
  let distance=0;
  result.forEach((p,i)=>{if(i)distance+=Math.hypot(p.x-result[i-1].x,p.z-result[i-1].z);p.distance=distance;});
  return result;
}

export function makeStudioCourse({kind="gentle-s",stage,materialId,railMaterialId=materialId,segments=48}={}) {
  if(!STUDIO_COURSE_KINDS.includes(kind))throw new RangeError("Unknown Studio course");
  if(!stage||![stage.width,stage.length,stage.baseY].every(Number.isFinite)||
      stage.width<2||stage.width>20||stage.length<2||stage.length>20)
    throw new RangeError("Course stage needs finite width/length in 2..20 and a base height");
  if(!Number.isInteger(segments)||segments<16||segments>128)throw new RangeError("Course segments must be 16..128");
  for(const id of [materialId,railMaterialId])if(!Number.isInteger(id)||id<0)throw new RangeError("Course material ID must be nonnegative");
  const {width,length,baseY}=stage, bodyId=`studio-course:${kind}`;
  const clearWidth=Math.min(width,length)*WIDTH_FRACTION, thickness=clearWidth*.12;
  const path=coursePath(kind,width,length,segments), last=path.length-1, total=path.at(-1).distance;
  const vertices=[], patches=[], planes={},surfaces={},surfaceGroups={};
  const top=(i,j)=>i*PROFILE.length+j, count=path.length*PROFILE.length;
  const bottom=index=>index+count;
  const baseHeight=(p)=>baseY+thickness+GRADE*(kind==="gentle-s"?path.at(-1).x-p.x:total-p.distance);
  for(let i=0;i<path.length;i++)for(let j=0;j<PROFILE.length;j++) {
    const p=path[i],side=[-p.tangent[2],0,p.tangent[0]],u=PROFILE[j]*clearWidth;
    const x=p.x+side[0]*u,z=p.z+side[2]*u;
    // The S floor is one authored inclined plane. Its rails share that plane's
    // slope, so their surface ownership never needs a tolerance-based weld.
    const y=kind==="gentle-s"?baseY+thickness+GRADE*(path.at(-1).x-x):baseHeight(p);
    const height=i===0||i===last?RAIL_HEIGHT:HEIGHTS[j];
    vertices.push([x,y+clearWidth*height,z]);
  }
  for(let index=0;index<count;index++)vertices.push([vertices[index][0],baseY,vertices[index][2]]);
  for(let i=0;i<last;i++)for(let j=0;j<PROFILE.length-1;j++) {
    const role=i===0||i===last-1?"end-stop":j===2||j===3?"run":j===0||j===5?"rail":"bank";
    const a=top(i,j),b=top(i,j+1),c=top(i+1,j+1),d=top(i+1,j);
    for(const [half,indices] of [[0,[a,b,c]],[1,[a,c,d]]]) {
      let surfaceId=`${bodyId}/${role}/${i}/${j}/${half}`;
      const planar=role==="run"||role==="rail";
      const zone=path[i].zone===path[i+1].zone?path[i].zone:null;
      const commonPlane=planar&&(kind==="gentle-s"||zone==="entry"||zone==="exit");
      if(commonPlane)
        surfaceId=`${bodyId}/${role}/${role==="rail"?(j===0?"left":"right")+"/":""}${kind==="gentle-s"?"incline":zone}`;
      const side=j<2?"left":j>3?"right":"",routeSurfaceId=`${bodyId}/${role}/${
        role==="end-stop"?(i===0?"start":"finish"):side||"continuous"}`;
      const certificate={bodyId,kind:"planar",footprint:commonPlane&&kind==="gentle-s"?"nonconvex":"convex",
        construction:commonPlane?(kind==="gentle-s"?"inclined-s-ribbon":"straight-route-strip"):"triangle",role};
      patches.push({indices,role,surfaceId,routeSurfaceId,certificate,materialId:role==="run"?materialId:railMaterialId});
    }
  }
  // Adjacency is constructed from shared vertex IDs, never coordinate welding.
  const adjacency=new Map();
  patches.forEach((patch,index)=>patch.indices.forEach((a,k)=>{
    const b=patch.indices[(k+1)%3],key=edgeKey(a,b);
    if(!adjacency.has(key))adjacency.set(key,[]);adjacency.get(key).push(index);
  }));
  const parts=[],faces=[],faceRecords=[];
  patches.forEach((patch,index)=>{
    const [a,b,c]=patch.indices,A=bottom(a),B=bottom(b),C=bottom(c),partId=`${bodyId}/prism/${index}`;
    const triangles=[[a,b,c],[A,C,B]],meta=[
      {surfaceId:patch.surfaceId,routeSurfaceId:patch.routeSurfaceId,boundary:"exterior",role:patch.role},
      {surfaceId:`${bodyId}/base`,boundary:"exterior",role:"base"},
    ];
    for(const [u,v] of [[a,b],[b,c],[c,a]]) {
      const key=edgeKey(u,v),owners=adjacency.get(key),internal=owners.length===2;
      if(owners.length<1||owners.length>2)throw new Error("Non-manifold authored course edge");
      const record={surfaceId:`${bodyId}/${internal?"construction":"rim"}/${key}`,
        boundary:internal?"internal":"exterior",role:internal?"construction":"side",
        ...(internal?{adjacentPartId:`${bodyId}/prism/${owners.find(owner=>owner!==index)}`}:{})};
      // Both owners split a shared quad on the SAME authored diagonal, with
      // opposite winding. Index ordering is identity, not a geometric weld.
      triangles.push(...(u<v?[[u,bottom(u),bottom(v)],[u,bottom(v),v]]:
        [[v,bottom(u),bottom(v)],[v,u,bottom(u)]]));meta.push({...record},{...record});
    }
    const part={id:partId,bodyId,closed:true,convex:true,construction:"vertical-triangle-prism",
      vertexIndices:[a,b,c,A,B,C],triangles,triangleMeta:meta,materialId:patch.materialId};
    parts.push(part);
    surfaces[patch.surfaceId]??={...patch.certificate};
    surfaces[`${bodyId}/base`]??={bodyId,kind:"planar",footprint:"nonconvex",construction:"route-footprint",role:"base"};
    surfaceGroups[patch.routeSurfaceId]??={bodyId,role:patch.role,
      kind:kind==="gentle-s"&&(patch.role==="run"||patch.role==="rail")?"planar":"folded",
      footprint:"nonconvex",construction:"continuous-route-surface",triangles:[],collisionReady:false};
    surfaceGroups[patch.routeSurfaceId].triangles.push({partId,triangleIndex:0});
    triangles.forEach((indices,triangleIndex)=>{
      const m=meta[triangleIndex];
      surfaces[m.surfaceId]??={bodyId,kind:"planar",footprint:"convex",construction:"vertical-edge-quad",role:m.role};
      if(m.boundary!=="exterior")return;
      const [p,q,r]=indices.map(id=>vertices[id]),normal=unit(cross(sub(q,p),sub(r,p)));
      const face=[p,q,r,patch.materialId,[normal,normal,normal]];
      faces.push(face);faceRecords.push({partId,triangleIndex,surfaceId:m.surfaceId,
        ...(m.routeSurfaceId?{routeSurfaceId:m.routeSurfaceId}:{}),role:m.role,indices:[...indices]});
      if(m.role==="run"||m.role==="rail"||m.role==="base") {
        if(!planes[m.surfaceId])planes[m.surfaceId]={point:[...p],normal:[...normal]};
      }
    });
  });
  const grades=faceRecords.flatMap((record,index)=>record.role==="run"?
    [Math.hypot(faces[index][4][0][0],faces[index][4][0][2])/faces[index][4][0][1]]:[]);
  const maxGrade=Math.max(...grades);
  if(!(maxGrade<=.05))throw new Error("Authored course exceeds its five-percent running-surface grade");
  const route=path.map(p=>[p.x,baseHeight(p),p.z]);
  const bounds={low:[0,1,2].map(axis=>Math.min(...vertices.map(p=>p[axis]))),
    high:[0,1,2].map(axis=>Math.max(...vertices.map(p=>p[axis])))};
  if(bounds.low[0]<-width/2||bounds.high[0]>width/2||bounds.low[2]<-length/2||bounds.high[2]>length/2)
    throw new Error("Course footprint exceeds its authored stage");
  return {id:bodyId,kind,stage:{width,length,baseY},vertices,parts,faces,faceRecords,planes,surfaces,surfaceGroups,route,
    clearWidth,maxGrade,bounds,
    spawn:{surfacePoint:[...route[2]],forward:[...path[2].tangent],routeIndex:2},
    finish:{surfacePoint:[...route[last-2]],forward:[...path[last-2].tangent],routeIndex:last-2},
    recommendedMaximumRadius:clearWidth*.18,
    collisionReady:false,collisionRequirement:"authored-compound-union-manifold",
    contactNote:"Convex parts are construction partitions of ONE body. Internal faces and inter-part seams must not produce contacts."};
}

export function studioCourseMarbleFit(course,radius) {
  if(!Number.isFinite(radius)||radius<=0)throw new RangeError("Marble radius must be positive in Studio units");
  // Advice only. Never resize the stage, saved marble, or simulation radius.
  return {fits:radius<=course.recommendedMaximumRadius,radius,
    recommendedMaximumRadius:course.recommendedMaximumRadius,
    clearWidth:course.clearWidth};
}

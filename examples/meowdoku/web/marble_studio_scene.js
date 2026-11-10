// Pack the existing renderer's authored faces, never alternate collision
// shapes. Construction metadata names exact source parts; it is not a native
// admission certificate for their union, concavity or numerical conditioning.
const vec=(v,n,name)=>{
  if(!(Array.isArray(v)||ArrayBuffer.isView(v))||v.length!==n||!Array.from(v).every(Number.isFinite))
    throw new TypeError(`${name} needs ${n} finite values`);
  return Array.from(v);
};

// The same PBR writes used by the actual Phong frame and a detached capture.
export function writeStudioFixtureRows(data,stride,base,colors,strengths) {
  for(let i=0;i<strengths.length;i++) {
    const b=(base+i)*stride,c=colors[i],s=strengths[i];
    data[b+8]=c[0]*s;data[b+9]=c[1]*s;data[b+10]=c[2]*s;
    // A filter/diffuser remains a material when its source is switched off.
    // Colour is an illustrative matte surface reflectance, independent of power.
    data[b]=c[0];data[b+1]=c[1];data[b+2]=c[2];data[b+3]=1;data[b+4]=0;
    data[b+5]=0;data[b+7]=1;
  }
  return data;
}

export function packStudioScene({channel,unitM,rayOriginM,parts,pbr,subject,camera,authoring,room}) {
  if(!["studio","exposure"].includes(channel))throw new Error("The scene channel must be explicit");
  if(!Number.isFinite(unitM)||unitM<=0)throw new RangeError("Invalid Studio metre conversion");
  const origin=vec(rayOriginM,3,"Ray origin"),seen=new Set();
  if(!Array.isArray(parts)||!Number.isInteger(pbr?.stride)||pbr.stride<16||
      !pbr.data||pbr.data.length%pbr.stride)throw new TypeError("Exact authored parts and PBR rows are required");
  const rows=Array.from({length:pbr.data.length/pbr.stride},(_,i)=>Array.from(pbr.data.slice(i*pbr.stride,i*pbr.stride+16))).flat();
  if(!rows.every(Number.isFinite))throw new TypeError("Nonfinite scene material");
  const vertices=[],rayNormals=[],materialIds=[],sources=[],captured=[];
  const toRay=point=>point.map((v,i)=>v*unitM+origin[i]);
  for(const part of parts) {
    if(!part||typeof part.id!=="string"||!part.id||seen.has(part.id)||!Array.isArray(part.faces)||!part.faces.length)
      throw new Error("Every authored part needs a unique stable ID and faces");
    seen.add(part.id);
    const first=materialIds.length;
    const faces=part.faces.map((face,index)=>{
      const a=vec(face[0],3,"Triangle"),b=vec(face[1],3,"Triangle"),c=vec(face[2],3,"Triangle"),id=face[3];
      if(!Number.isInteger(id)||id<0||id>=rows.length/16)throw new RangeError("Triangle material is absent");
      const normals=face[4]?.map(n=>vec(n,3,"Normal"));
      if(normals?.length!==3)throw new TypeError("Three authored normals are required");
      vertices.push(...toRay(a),...toRay(b),...toRay(c));materialIds.push(id);
      rayNormals.push(...normals.flat());
      sources.push({partId:part.id,bodyId:part.bodyId??part.id,triangle:index});
      return [a,b,c,id,normals,...face.slice(5)];
    });
    const {faces:_,...metadata}=part;
    captured.push({...structuredClone(metadata),faces,firstTriangle:first,triangleCount:faces.length});
  }
  let capturedSubject=null;
  if(subject) {
    const model=vec(subject.model,16,"Displayed subject transform");
    if(model[3]!==0||model[7]!==0||model[11]!==0||model[15]!==1)throw new Error("Subject model must be affine");
    if(!Array.isArray(subject.faces)||!subject.faces.length)throw new Error("Displayed subject faces are required");
    const transform=([x,y,z])=>[model[0]*x+model[4]*y+model[8]*z+model[12],
      model[1]*x+model[5]*y+model[9]*z+model[13],model[2]*x+model[6]*y+model[10]*z+model[14]];
    const subjectVertices=[],subjectIds=[];
    for(const face of subject.faces) {
      for(let i=0;i<3;i++)subjectVertices.push(...toRay(transform(vec(face[i],3,"Subject vertex"))));
      if(!Number.isInteger(face[3])||face[3]<0||face[3]>=rows.length/16)throw new RangeError("Subject material is absent");
      subjectIds.push(face[3]);
    }
    capturedSubject={...structuredClone(subject),model,ray:{vertices:subjectVertices,material_ids:subjectIds},
      position:[model[12],model[13],model[14]]};
  }
  return {schema:"turing.studio-visible-scene.v1",channel,frame:{unitM,rayOriginM:origin},parts:captured,
    ray:{vertices,normals:rayNormals,material_ids:materialIds,pbr:rows},triangleSources:sources,subject:capturedSubject,
    camera:structuredClone(camera),authoring:structuredClone(authoring),room:structuredClone(room),
    collisionReady:false,admissionRequired:"exact-source-topology-and-contact-ownership"};
}

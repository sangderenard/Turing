// Identity copies only. Never match positions/materials or evaluate ray response.
export const RESPONSE_INFORMATION='evaluated-responses-v1';
export const RESPONSE_LINEAGE='studio-response-lineage-v1';
export const RESPONSE_ATTRIBUTE=7;
const LIMIT=0x0fffffff;

export function recipientId(namespace,index) {
  if(namespace===0)return 0;
  if(![1,2,3].includes(namespace)||!Number.isSafeInteger(index)||index<0||index>=LIMIT)
    throw Error('Invalid copied response recipient identity.');
  return ((namespace<<28)|(index+1))>>>0;
}

export function copyResponseIdentity(source,target) {
  if(source.responseSource)target.responseSource={...source.responseSource};
  return target;
}

export function tagSubjectResponseFaces(geometry,enabled=false) {
  const faces=[...geometry.interior,...geometry.glass];
  faces.forEach((face,index)=>{
    if(enabled)face.responseSource={namespace:1,sourceId:'subject',sourceFace:index};
    else delete face.responseSource;
  });
}

export function tagPartResponseFaces(part,sourceId,enabled=false) {
  if(!enabled)return;
  if(typeof sourceId!=='string'||!sourceId||!Array.isArray(part.faces))throw Error('Original part identity is required.');
  part.faces.forEach((face,index)=>{face.responseSource={namespace:2,sourceId,sourceFace:index};});
}

export function copiedResponseSources(faces) {
  if(!faces.some(face=>face.responseSource))return null;
  return faces.map(face=>face.responseSource?{...face.responseSource}:null);
}

export function fixtureResponseLineage(parts) {
  const seen=new Set();let start=0;
  const records=parts.map(part=>{
    const source_id=part.responseSourceId;
    if(typeof source_id!=='string'||!source_id||seen.has(source_id)||!Array.isArray(part.faces))
      throw Error('Unique original fixture part identities are required.');
    seen.add(source_id);const record={source_id,start,count:part.faces.length};start+=record.count;return record;
  });
  return {version:RESPONSE_LINEAGE,parts:records};
}

export function responseRecipientIds(sources,lineage) {
  if(lineage?.version!==RESPONSE_LINEAGE)throw Error('Matching copied native response lineage is required.');
  const parts=lineage.fixture_parts?.parts??[],byPart=new Map(parts.map(p=>[p.source_id,p]));
  const spheres=lineage.spheres??[];
  return Uint32Array.from(sources,source=>{
    if(!source)return 0;
    if(source.namespace===1) {
      // Explicit original sphere membership ranges, not hit-position inference.
      for(const sphere of spheres) {
        const [index,,start,count]=sphere;
        if(source.sourceFace>=start&&source.sourceFace<start+count)return recipientId(3,index);
      }
      return recipientId(1,source.sourceFace);
    }
    if(source.namespace===2) {
      const part=byPart.get(source.sourceId);
      if(!part||source.sourceFace<0||source.sourceFace>=part.count)return 0;
      return recipientId(2,part.start+source.sourceFace);
    }
    return 0;
  });
}

export function bindResponseIdentity(gl,resource,lineage) {
  // Called only by the validated, explicitly opted-in response consumer.
  // Existing positions/normals/material/group and transparent EBO are untouched.
  if(!resource?.responseSources)return false;
  const ids=responseRecipientIds(resource.responseSources,lineage),data=new Uint32Array(ids.length*3);
  ids.forEach((id,i)=>data.fill(id,i*3,i*3+3));
  const vao=gl.getParameter(gl.VERTEX_ARRAY_BINDING),array=gl.getParameter(gl.ARRAY_BUFFER_BINDING);
  try {
    if(!resource.responseIdentityBuffer){resource.responseIdentityBuffer=gl.createBuffer();resource.buffers.push(resource.responseIdentityBuffer);}
    gl.bindVertexArray(resource.vao);gl.bindBuffer(gl.ARRAY_BUFFER,resource.responseIdentityBuffer);
    gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);gl.enableVertexAttribArray(RESPONSE_ATTRIBUTE);
    gl.vertexAttribIPointer(RESPONSE_ATTRIBUTE,1,gl.UNSIGNED_INT,0,0);
  }finally{gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,array);}
  return true;
}

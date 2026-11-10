// Exact, camera-sampled presentation subset publication. This module neither
// renders a visibility pass nor estimates which triangles are visible. The
// current Phong/game shaders do not emit a primary triangle-ID attachment.
// A real ID-pass producer must supply mesh/revision/camera-bound observations.
// Keep the complete source for new views, shadows, glass/refraction and physics.

const MAX_TRIANGLES=200000,MAX_VIEWS=4096;
const finite=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
const clone=value=>structuredClone(value);
const freeze=value=>{
  if(value&&typeof value==='object'&&!Object.isFrozen(value)) {
    for(const child of Object.values(value))freeze(child);
    Object.freeze(value);
  }
  return value;
};
const canonical=value=>{
  if(value===null||typeof value==='boolean'||typeof value==='string')return JSON.stringify(value);
  if(typeof value==='number') {
    if(!Number.isFinite(value))throw Error('Nonfinite publication data.');
    return JSON.stringify(value);
  }
  if(Array.isArray(value))return `[${value.map(canonical).join(',')}]`;
  if(value&&Object.getPrototypeOf(value)===Object.prototype)
    return `{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  throw Error('Publication data must be plain JSON.');
};
export async function phongPublicationDigest(value) {
  const bytes=new TextEncoder().encode(canonical(value));
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
}

function checkSource(source) {
  const {id,faces,model,frame,pbr}=source;
  if(typeof id!=='string'||!id||!Array.isArray(faces)||!faces.length||faces.length>MAX_TRIANGLES||
      !finite(model,16)||model[3]!==0||model[7]!==0||model[11]!==0||model[15]!==1||
      !(frame?.unitM>0)||!Number.isFinite(frame.unitM)||!finite(frame.rayOriginM,3)||
      !Array.isArray(pbr)||!pbr.length||pbr.length%16||!pbr.every(Number.isFinite))
    throw Error('Exact subject faces, affine model, frame and PBR rows are required.');
  for(const face of faces) {
    if(!Array.isArray(face)||face.length<5||!face.slice(0,3).every(v=>finite(v,3))||
        !Number.isSafeInteger(face[3])||face[3]<0||face[3]>=pbr.length/16||
        !Array.isArray(face[4])||face[4].length!==3||!face[4].every(v=>finite(v,3)))
      throw Error('An exact source triangle, normal or material ID is absent.');
  }
}

export async function capturePhongTriangleSource(capture) {
  const state=capture?.state,subject=state?.subject;
  if(state?.schema!=='turing.studio-visible-scene.v1'||state.channel!=='studio'||
      !Number.isSafeInteger(capture.revision)||capture.revision<0||!subject)
    throw Error('A revision-bound displayed Studio subject is required.');
  const source=clone({id:subject.id,faces:subject.faces,model:subject.model,
    frame:state.frame,pbr:state.ray?.pbr});
  checkSource(source);
  if(!Array.isArray(subject.ray?.material_ids)||subject.ray.material_ids.length!==source.faces.length||
      !subject.ray.material_ids.every((id,i)=>id===source.faces[i][3]))
    throw Error('Subject face IDs and captured ray material IDs disagree.');
  const mesh={id:source.id,faces:source.faces,model:source.model,frame:source.frame};
  return freeze({schema:'turing.phong-triangle-source.v1',revision:capture.revision,
    ...source,meshSha256:await phongPublicationDigest(mesh),
    materialSha256:await phongPublicationDigest(source.pbr),
    triangleSources:source.faces.map((_,triangle)=>({partId:source.id,bodyId:source.id,triangle})),
    materialIds:source.faces.map(face=>face[3]),
    materialAuthority:'Exact source IDs; complete Studio render tables must accompany the game asset.'});
}

export async function bindPhongVisibilityViews(views) {
  if(!Array.isArray(views)||!views.length||views.length>MAX_VIEWS)
    throw Error('Provide 1-4096 explicit sampled camera views.');
  const seen=new Set(),bound=[];
  for(const value of views) {
    const {cameraSha256:priorDigest,...view}=clone(value);
    if(!Number.isSafeInteger(view.index)||view.index<0||seen.has(view.index)||
        !finite(view.direction,3)||!(Math.hypot(...view.direction)>0)||
        !finite(view.camera?.eye_m,3)||!finite(view.camera?.target_m,3)||
        view.camera.eye_m.every((x,i)=>x===view.camera.target_m[i])||
        typeof view.condition_key!=='string'||!view.condition_key)
      throw Error('Sampled views need unique indices and exact camera/condition descriptors.');
    seen.add(view.index);
    // Bind the entire descriptor, not only the eye position or an inferred view.
    const cameraSha256=await phongPublicationDigest(view);
    if(priorDigest!==undefined&&priorDigest!==cameraSha256)throw Error('Sampled camera descriptor changed.');
    bound.push({...view,cameraSha256});
  }
  return freeze(bound);
}

export async function publishSampledPhongTriangles(source,views,samples) {
  if(source?.schema!=='turing.phong-triangle-source.v1'||!Number.isSafeInteger(source.revision))
    throw Error('Use the exact captured source mesh.');
  checkSource(source);
  const meshSha256=await phongPublicationDigest({id:source.id,faces:source.faces,model:source.model,frame:source.frame});
  if(meshSha256!==source.meshSha256||await phongPublicationDigest(source.pbr)!==source.materialSha256)
    throw Error('Source mesh or materials changed after capture.');
  if(source.triangleSources?.length!==source.faces.length||source.materialIds?.length!==source.faces.length||
      source.triangleSources.some((row,i)=>row.partId!==source.id||row.bodyId!==source.id||row.triangle!==i)||
      source.materialIds.some((id,i)=>id!==source.faces[i][3]))
    throw Error('Original triangle/source/material identity mapping changed.');
  const bound=await bindPhongVisibilityViews(views),byView=new Map(bound.map(view=>[view.index,view]));
  if(!Array.isArray(samples)||!samples.length||samples.length>MAX_VIEWS)
    throw Error('Missing actual primary triangle-ID pass observations.');
  const selected=new Set(),sampleViews=new Set(),receipts=[];
  for(const sample of samples) {
    const view=byView.get(sample.viewIndex),ids=sample.triangleIds;
    if(sample.schema!=='turing.phong-primary-visibility-sample.v1'||
        sample.revision!==source.revision||sample.meshSha256!==meshSha256||
        sample.materialSha256!==source.materialSha256||!view||sample.cameraSha256!==view.cameraSha256||
        sample.producer?.kind!=='primary-raster-triangle-id'||sample.producer.readbackComplete!==true||
        typeof sample.producer.id!=='string'||!sample.producer.id||
        !Number.isSafeInteger(sample.sampleCount)||sample.sampleCount<1||
        !Array.isArray(ids)||ids.length>sample.sampleCount||ids.length>MAX_TRIANGLES||
        ids.some(id=>!Number.isSafeInteger(id)||id<0||id>=source.faces.length))
      throw Error('Visibility observations must be actual, complete readbacks bound to this mesh and camera.');
    sampleViews.add(view.index);
    for(const id of ids)selected.add(id);
    receipts.push(clone(sample));
  }
  // Unobserved planned cameras are not coverage. Preserve their absence explicitly.
  const sourceIndices=[...selected].sort((a,b)=>a-b),sourceToDraw=Array(source.faces.length).fill(-1);
  sourceIndices.forEach((sourceIndex,drawIndex)=>{sourceToDraw[sourceIndex]=drawIndex;});
  const detached=freeze(clone(source));
  return freeze({schema:'turing.phong-sampled-triangle-publication.v1',revision:source.revision,
    meshSha256,materialSha256:source.materialSha256,
    sourceIndices,sourceToDraw,
    faces:sourceIndices.map(i=>detached.faces[i]),
    triangleSources:sourceIndices.map(i=>detached.triangleSources[i]),
    materialIds:sourceIndices.map(i=>detached.materialIds[i]),
    fullSource:detached,views:bound,samples:receipts,
    sampledViewIndices:[...sampleViews].sort((a,b)=>a-b),
    unsampledViewIndices:bound.filter(view=>!sampleViews.has(view.index)).map(view=>view.index),
    scope:'Union of observed primary triangles at the supplied sampled cameras only.',
    consumerPolicy:{presentationOnly:true,universalVisibility:false,
      physicsPruning:false,shadowPruning:false,refractionPruning:false,
      transparentLayerCompleteness:false,geodesicExpressionMinimization:false,
      fallback:'Keep fullSource for new poses, unsampled views, transparent layers, shadows, refraction and physics.'},
    producerStatus:'Caller-supplied observations; this helper does not implement or simulate the missing Phong ID pass.'});
}

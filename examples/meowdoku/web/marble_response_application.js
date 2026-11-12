// GPU layout of native application rows. Sorting copies the seven-key index;
// values, original sample weights, and floating-point bits are never changed.
export function responseApplicationTexture(rows,maxTextureSize) {
  if(!(rows instanceof Uint32Array)||rows.length%12||!Number.isSafeInteger(maxTextureSize)||maxTextureSize<3)
    throw Error('Invalid native response application rows or texture limit.');
  const count=rows.length/12,indices=Array.from({length:count},(_,i)=>i);
  const compare=(a,b)=>{for(let j=0;j<7;j++){const d=rows[a*12+j]-rows[b*12+j];if(d)return d;}return 0;};
  indices.sort(compare);
  const width=Math.min(Math.floor(maxTextureSize/3)*3,Math.max(3,count*3)),height=Math.max(1,Math.ceil(count*3/width));
  if(height>maxTextureSize)throw Error('Native response application exceeds texture limits.');
  const words=new Uint32Array(width*height*4);
  for(let i=0;i<count;i++) {
    if(i&&compare(indices[i-1],indices[i])===0)throw Error('Duplicate native response application key.');
    const offset=indices[i]*12;
    if(rows[offset+6]>2||rows[offset+7]>rows[offset+8]||rows[offset+8]===0)
      throw Error('Unsupported component or invalid original sample weights.');
    for(let j=9;j<12;j++)if(!Number.isFinite(new Float32Array(rows.buffer,rows.byteOffset+(offset+j)*4,1)[0]))
      throw Error('Nonfinite native response application value.');
    words.set(rows.subarray(offset,offset+12),i*12);
  }
  return {words,width,height,count};
}

// Only compare exact supplied context. This function does not generate camera
// optics, infer a current context from a sample, or widen angular support.
export function matchResponseQuery(application,query) {
  if(!query||!application||!Array.isArray(application.query_contexts))return -1;
  const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?
    Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])])):value;
  const fingerprint=value=>JSON.stringify(canonical(value));
  const matches=application.query_contexts.filter(candidate=>
    ['origin_m','lens','film','layer_specs','transport'].every(key=>fingerprint(candidate[key])===fingerprint(query[key])));
  return matches.length===1?matches[0].id:-1;
}

export function currentResponseQuery(application,originM,liveRevision) {
  if(!application||application.current_live_revision!==liveRevision||
      !/^[0-9a-f]{64}$/.test(application.current_camera_revision)||
      !/^[0-9a-f]{64}$/.test(application.current_transport_scene_sha256)||
      !Array.isArray(application.current_query_context_ids)||!Array.isArray(application.query_contexts)||
      !originM||originM.length!==3||!Array.from(originM).every(Number.isFinite))return -1;
  const current=new Set(application.current_query_context_ids);
  const matches=application.query_contexts.filter(query=>current.has(query.id)&&
    query.transport?.transport_scene_sha256===application.current_transport_scene_sha256&&
    Array.isArray(query.origin_m)&&query.origin_m.length===3&&
    query.origin_m.every((v,i)=>Object.is(Math.fround(v),Math.fround(originM[i]))));
  return matches.length===1?matches[0].id:-1;
}

// The native scene store is the authority. These are typed copies and index
// validation, not a radiance estimator, optical replay, or a pixel image.
export const SCENE_RESPONSE_SCHEMA='studio-scene-responses-v1';
export const SCENE_RESPONSE_SCHEMA_V2='studio-scene-responses-v2';
export const isSceneResponseSchema=value=>value===SCENE_RESPONSE_SCHEMA||value===SCENE_RESPONSE_SCHEMA_V2;
export const CAMERA_PAIR_CONTRACT=Object.freeze({version:'native-camera-response-pairs-v2',row_words:256,
  interval_schema:'native-camera-intervals-v1',interval_offset_words:0,interval_words:128,
  response_schema:'native-camera-response-copies-v1',response_offset_words:128,response_words:128,
  alignment:'same-dispatch-row-ordinal',attachment:'native-only-unlocated',application_ready:false});
export const SCENE_RESPONSE_BYTES=64*1024*1024;
export async function receiveSceneResponses(manifest,payload,{sceneRevision,information,maxPayloadBytes=SCENE_RESPONSE_BYTES,recordIndexBytes=0,cameraQuerySegments=false}={}) {
  if(information!=='evaluated-responses-v1')throw Error('Scene response decoding requires explicit information opt-in.');
  if(typeof cameraQuerySegments!=='boolean'||cameraQuerySegments&&(!recordIndexBytes||manifest?.schema!==SCENE_RESPONSE_SCHEMA_V2))
    throw Error('Camera query segments require a boolean opt-in, v2 envelope and positive index budget.');
  if(!Number.isSafeInteger(recordIndexBytes)||recordIndexBytes<0||recordIndexBytes>SCENE_RESPONSE_BYTES)
    throw Error('Invalid explicit scene-record index budget.');
  if(!Number.isSafeInteger(maxPayloadBytes)||maxPayloadBytes<0||maxPayloadBytes>SCENE_RESPONSE_BYTES||
      !(payload instanceof ArrayBuffer)||payload.byteLength>maxPayloadBytes)
    throw Error('Scene response payload exceeds the explicit memory budget.');
  if(typeof sceneRevision!=='string'||manifest?.scene_revision!==sceneRevision)throw Error('Stale scene response identity.');
  const metadata=validateSceneResponseManifest(manifest);
  if(payload.byteLength!==metadata.payload_bytes)throw Error('Incomplete native scene-response bytes.');
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',payload)),v=>v.toString(16).padStart(2,'0')).join('');
  if(digest!==metadata.payload_sha256)throw Error('Scene response payload hash mismatch.');
  const result=decodeSceneResponses(payload,metadata);
  const recordIndex=recordIndexBytes?indexSceneRecords(result,recordIndexBytes,cameraQuerySegments):null;
  return {...result,recordIndex,memory:{payloadBytes:payload.byteLength,decodePeakPayloadBytes:payload.byteLength*2,
    recordIndexBytes:recordIndex?.bytes??0,recordIndexScratchReserve:recordIndex?.scratchReserve??0,gpuBytes:0}};
}
const UINT=0xffffffff,HEX=/^[0-9a-f]{64}$/;
const integer=(v,max=UINT)=>Number.isSafeInteger(v)&&v>=0&&v<=max;
const vector=(v,n=3)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const FAMILY_CODECS=Object.freeze({1:['forward','evaluated-forward-responses-v1'],
  2:['reactive','evaluated-forward-responses-v1'],3:['camera_rhs_summands','native-camera-response-pairs-v1'],
  4:['camera_complete_sample','native-camera-scene-sample-v1']});

export function validateSceneResponseManifest(value) {
  if(!value||!isSceneResponseSchema(value.schema)||value.byte_order!=='little'||
      !HEX.test(value.scene_revision)||!integer(value.publication_sequence,Number.MAX_SAFE_INTEGER)||
      !integer(value.payload_bytes,SCENE_RESPONSE_BYTES)||!HEX.test(value.payload_sha256)||
      value.complete_retained_snapshot!==true||value.physical_completeness_claim!==false||
      value.application_ready!==false)
    throw Error('Invalid or unsupported native scene-response manifest.');
  const pairsV2=value.schema===SCENE_RESPONSE_SCHEMA_V2;
  if(pairsV2) {
    const contract=value.camera_pair_contract;
    if(!contract||typeof contract!=='object'||Array.isArray(contract)||
        Object.keys(contract).length!==Object.keys(CAMERA_PAIR_CONTRACT).length||
        Object.entries(CAMERA_PAIR_CONTRACT).some(([key,expected])=>!Object.hasOwn(contract,key)||contract[key]!==expected))
      throw Error('Invalid exact camera-pair contract.');
  } else if(Object.hasOwn(value,'camera_pair_contract'))throw Error('Camera-pair contract requires the v2 envelope.');
  for(const [id,[name,codec]]of Object.entries(FAMILY_CODECS))
    if(value.families?.[id]?.name!==name||value.families[id].codec!==(pairsV2&&id==='3'?'native-camera-response-pairs-v2':codec))
      throw Error('Unknown native scene-response family.');
  const layout=[['scene_index',80],['native_rows',512],['camera_rhs_rows',pairsV2?1024:1536],['scalar_cells',24]];
  if(value.application!==undefined)layout.push(['application_rows',48]);
  if(!Array.isArray(value.sections)||value.sections.length!==layout.length)
    throw Error('Incomplete native scene-response sections.');
  let end=0;
  for(let i=0;i<layout.length;i++) {
    const section=value.sections[i],[name,stride]=layout[i];
    if(section.name!==name||section.stride!==stride||section.offset!==end||
        !integer(section.count,Math.floor(SCENE_RESPONSE_BYTES/stride)))
      throw Error('Invalid or overlapping native scene-response section.');
    end+=section.count*stride;
  }
  if(end!==value.payload_bytes)throw Error('Incomplete native scene-response payload.');
  const r=value.retention;
  if(!r||r.mode!=='bounded-fifo'||!integer(r.capacity)||r.capacity<1||
      !integer(r.retained_entries)||r.retained_entries>r.capacity||r.retained_entries!==value.sections[0].count||
      ['evicted_entries','duplicate_entries','invalidated_entries','incomplete_dispatches'].some(k=>!integer(r[k],Number.MAX_SAFE_INTEGER))||
      r.unavailable_on_eviction!==true)
    throw Error('Invalid native scene-response retention receipt.');
  const index=value.indexing;
  if(!index||index.semantics!=='lookup buckets only; exact positions/directions retained'||!['icosahedron12-v1','axes6-v1'].includes(index.stencil)||
      !(index.bounds===null&&index.dimensions===null||Array.isArray(index.bounds)&&index.bounds.length===2&&
        index.bounds.every(v=>vector(v))&&vector(index.dimensions)&&index.dimensions.every(n=>integer(n,256)&&n>0)&&
        index.bounds[1].every((n,i)=>n>index.bounds[0][i])))
    throw Error('Invalid native scene-response indexing contract.');
  const ids=new Set();
  if(!Array.isArray(value.contexts)||value.contexts.length>r.retained_entries)
    throw Error('Invalid native scene-response contexts.');
  for(const c of value.contexts) {
    if(!integer(c.id)||ids.has(c.id)||c.scene_revision!==value.scene_revision||
        c.physical_scene_identity!==value.scene_revision||typeof c.stream_id!=='string'||!c.stream_id||c.stream_id.length>256||
        !integer(c.sequence,Number.MAX_SAFE_INTEGER)||!HEX.test(c.source_sha256)||
        typeof c.semantics!=='string'||!c.semantics||!c.dispatch||typeof c.dispatch!=='object')
      throw Error('Invalid native response context identity.');
    ids.add(c.id);
  }
  const scalar=value.scalar_snapshot,m=scalar?.metadata;
  if(scalar==null) {if(value.sections[3].count)throw Error('Missing scalar snapshot metadata.');}
  else if(!m||!Array.isArray(m.identity)||m.identity.length!==2||
      !integer(m.total_cells)||!integer(m.bands,8)||m.bands<1||!integer(m.scanned_cells)||m.scanned_cells>m.total_cells||
      !integer(m.omitted_unvisited,Number.MAX_SAFE_INTEGER)||!integer(m.evicted_entries,Number.MAX_SAFE_INTEGER)||
      typeof m.complete_snapshot!=='boolean'||m.complete_snapshot&&m.scanned_cells!==m.total_cells||
      typeof scalar.semantics!=='string'||!m.metadata||typeof m.metadata!=='object')
    throw Error('Invalid scalar snapshot receipt.');
  if(value.application!==undefined) {
    const a=value.application,queries=new Set();
    if(!a||a.version!=='native-evaluated-rgb-v1'||a.ready!==false||a.section!=='application_rows'||
        a.support!=='recipient-cell-stencil-v1'||a.rgb!=='linear-srgb'||a.rgb_premultiplied!==false||
        a.alpha!=='available-original-camera-samples'||a.component_stage!=='native-film-filtered-before-linear-profile'||
        a.physical_completeness_claim!==false||a.specular_override_supported!==false||!Array.isArray(a.query_contexts))
      throw Error('Invalid native application contract.');
    for(const q of a.query_contexts) {
      if(!integer(q.id)||queries.has(q.id)||!vector(q.origin_m)||!q.lens||typeof q.lens!=='object'||
          !q.film||typeof q.film!=='object'||!Array.isArray(q.layer_specs)||q.layer_specs.length!==3||
          q.transport?.scene_revision!==value.scene_revision||!HEX.test(q.transport?.source_sha256)||
          q.transport?.shared_band0_path!==true)throw Error('Invalid native application query identity.');
      queries.add(q.id);
    }
    if(!Array.isArray(a.current_query_context_ids)||new Set(a.current_query_context_ids).size!==a.current_query_context_ids.length||
        a.current_query_context_ids.some(id=>!queries.has(id)))throw Error('Invalid current application query selection.');
    if(a.current_query_context_ids.length&&(!HEX.test(a.current_transport_scene_sha256)||
        !a.current_camera_revision||!integer(a.current_live_revision,Number.MAX_SAFE_INTEGER)||
        a.query_contexts.some(q=>a.current_query_context_ids.includes(q.id)&&q.transport.transport_scene_sha256!==a.current_transport_scene_sha256)))
      throw Error('Current application queries do not match the recorded transport identity.');
  }
  return structuredClone(value);
}

export function decodeSceneResponses(buffer,manifest) {
  const metadata=validateSceneResponseManifest(manifest);
  if(!(buffer instanceof ArrayBuffer)||buffer.byteLength!==metadata.payload_bytes)
    throw Error('Incomplete native scene-response bytes.');
  const view=new DataView(buffer),copy=section=>{
    const words=new Uint32Array(section.count*section.stride/4);
    for(let i=0;i<words.length;i++)words[i]=view.getUint32(section.offset+i*4,true);
    return words;
  };
  // Keep integer payload bits, including non-displayable floating-point data.
  const [index,nativeRows,cameraRhsRows,scalarCells,applicationRows]=metadata.sections.map(copy),
    references=[new Set(),new Set()],contexts=new Set(metadata.contexts.map(c=>c.id)),usedContexts=new Set(),
    families={forward:0,reactive:0,camera_rhs_summands:0,camera_complete_sample:0};
  for(let i=0;i<index.length;i+=20) {
    const family=index[i],ref=index[i+3],which=family===3?1:0,section=metadata.sections[which?2:1],
      namespace=index[i+4],recipient=index[i+5],primitive=index[i+6],present=index[i+11];
    if(!FAMILY_CODECS[family]||!contexts.has(index[i+2])||ref>=section.count||references[which].has(ref)||
        namespace>3||namespace===0&&recipient!==UINT||namespace!==0&&recipient>=0x0fffffff||
        ![UINT,0,1,2].includes(primitive)||index[i+7]>1||present>1||![1,2].includes(index[i+15])||
        index[i+19]!==UINT&&index[i+19]>=(metadata.indexing.stencil==='axes6-v1'?6:12))
      throw Error('Invalid native scene-response index identity.');
    if(present)for(let j=8;j<11;j++)
      if(!Number.isFinite(new Float32Array(index.buffer,index.byteOffset+(i+j)*4,1)[0]))
        throw Error('Nonfinite available native response position.');
    for(let j=12;j<15;j++)if(!Number.isFinite(new Float32Array(index.buffer,index.byteOffset+(i+j)*4,1)[0]))
      throw Error('Nonfinite native response direction.');
    const cell=index.subarray(i+16,i+19),dims=metadata.indexing.dimensions;
    if(!cell.every(v=>v===UINT)&&(!dims||!present||cell.some((v,j)=>v>=dims[j])))
      throw Error('Native response cell exceeds its recorded domain.');
    if(family===4) {
      const row=ref*128;
      // Exact producer attachment. A payload cannot be relabelled as another
      // position/direction or completed sample by its outer index.
      if(index[i+1]!==1||index[i+15]!==2||present!==nativeRows[row+11]||primitive!==nativeRows[row+3]||
          [8,9,10,12,13,14].some(j=>index[i+j]!==nativeRows[row+j]))
        throw Error('Completed native sample attachment disagrees with its payload.');
    }
    if(family===3&&metadata.schema===SCENE_RESPONSE_SCHEMA_V2) {
      const row=ref*256,bands=cameraRhsRows[row+6],mask=cameraRhsRows[row+40];
      // A camera pair is one actual interval and its aligned response row.
      // No raw hit, first-visible hit, or entry origin is promoted to a guessed
      // surface recipient or spatial application point by this envelope.
      if(index[i+1]!==1||namespace!==0||recipient!==UINT||primitive!==cameraRhsRows[row+126]||
          index[i+7]!==1||present!==0||index[i+15]!==2||
          [8,9,10].some(j=>index[i+j]!==0)||!cell.every(v=>v===UINT)||
          [12,13,14].some(j=>index[i+j]!==cameraRhsRows[row+j]))
        throw Error('Camera pair attachment disagrees with its native-only payload.');
      if(bands<1||bands>8||mask===0||(mask&~((1<<bands)-1))!==0)
        throw Error('Invalid native camera-pair active bands.');
    }
    references[which].add(ref);usedContexts.add(index[i+2]);families[FAMILY_CODECS[family][0]]++;
  }
  if(references.some((set,i)=>set.size!==metadata.sections[i?2:1].count)||usedContexts.size!==contexts.size)
    throw Error('Unreferenced native scene-response data.');
  const scalarKeys=new Set(),sm=metadata.scalar_snapshot?.metadata;
  for(let i=0;i<scalarCells.length;i+=6) {
    const cell=scalarCells[i],band=scalarCells[i+1],key=cell*sm.bands+band;
    // Loss-only ABI: rounded1, clipped2, invalid4, sum-overflow8,
    // attempt-overflow16, add-count-overflow32. Not scalar-query visit flags.
    if(cell>=sm.total_cells||cell>=sm.scanned_cells||band>=sm.bands||scalarKeys.has(key)||scalarCells[i+5]&~63)
      throw Error('Invalid or repeated original scalar cell.');
    scalarKeys.add(key);
  }
  const appKeys=new Set(),queries=new Set(metadata.application?.query_contexts.map(q=>q.id)??[]),dims=metadata.indexing.dimensions;
  if(applicationRows)for(let i=0;i<applicationRows.length;i+=12) {
    const recipient=applicationRows[i],namespace=recipient>>>28,ordinal=recipient&0x0fffffff;
    if(namespace<1||namespace>3||ordinal===0||!dims||[1,2,3].some((j)=>applicationRows[i+j]>=dims[j-1])||
        applicationRows[i+4]>=(metadata.indexing.stencil==='axes6-v1'?6:12)||!queries.has(applicationRows[i+5])||
        applicationRows[i+6]>3||applicationRows[i+7]>applicationRows[i+8]||applicationRows[i+8]===0||
        !Array.from(new Float32Array(applicationRows.buffer,applicationRows.byteOffset+(i+9)*4,3)).every(Number.isFinite))
      throw Error('Invalid native application row identity or values.');
    const key=Array.from(applicationRows.subarray(i,i+7)).join(':');
    if(appKeys.has(key))throw Error('Duplicate native application support component.');appKeys.add(key);
  }
  return {metadata,index,nativeRows,cameraRhsRows,scalarCells,applicationRows:applicationRows??new Uint32Array(0),families,applicationReady:false};
}

// A derived lookup over the existing exact records, never a deposition grid.
// No RGB, weights, sample availability, or payload words are changed. Endpoint
// support is bit 1; a copied finite forward/reactive segment is bit 2. Segment
// Camera query-parametric support is bit 4, explicitly opted in and derived
// from copied origin + direction * hit parameter, never first-hit position.
// Segment references are not additional observations and must not be summed per cell.
function indexSceneRecords(snapshot,budget,cameraQuerySegments) {
  const {metadata,index,nativeRows,cameraRhsRows}=snapshot,rows=index.length/20,stride=Math.max(8,rows*8),
    {bounds,dimensions}=metadata.indexing,scratchReserve=65536;
  if(budget<scratchReserve)throw Error('Scene-record index budget cannot reserve bounded scratch storage.');
  const shape=dimensions??[0,0,0],cellCount=shape.reduce((a,b)=>a*b,1);
  if(bounds&&shape.some((n,j)=>{
    const width=bounds[1][j]-bounds[0][j];
    return !Number.isFinite(width)||width<=0||!Number.isFinite(n/width)||n/width===0;
  }))throw Error('Scene-record index needs finite, representable grid spans.');
  if(!Number.isSafeInteger(cellCount*stride))throw Error('Scene-record index key exceeds exact integer storage.');
  const sourceFloats=new Float32Array(nativeRows.buffer,nativeRows.byteOffset,nativeRows.length),
    cameraFloats=new Float32Array(cameraRhsRows.buffer,cameraRhsRows.byteOffset,cameraRhsRows.length),
    indexFloats=new Float32Array(index.buffer,index.byteOffset,index.length),
    times=new Float64Array(shape.reduce((a,b)=>a+b,2)),cells=new Uint32Array(times.length),
    start=new Float64Array(3),end=new Float64Array(3),delta=new Float64Array(3);
  const stats={rows,pointRows:0,segmentRows:0,cameraQueryRows:0,cameraQueryUnavailableRows:0,segmentUnavailableRows:0,unindexedRows:0,references:0};
  const linear=p=>(p[2]*shape[1]+p[1])*shape[0]+p[0];
  const pointCell=offset=>{
    if(!bounds||!index[offset+11])return -1;
    for(let j=0;j<3;j++) {
      const p=(indexFloats[offset+8+j]-bounds[0][j])/(bounds[1][j]-bounds[0][j])*shape[j];
      if(!Number.isFinite(p)||p<0||p>=shape[j])return -1;
      end[j]=Math.floor(p);
    }
    return linear(end);
  };
  function segmentCells(offset) {
    const family=index[offset],base=index[offset+3]*128;
    if(cameraQuerySegments&&family===3) {
      const pair=index[offset+3]*256,t=cameraFloats[pair+11];
      if(!bounds||cameraRhsRows[pair+4]>=0x80000000||!Number.isFinite(t)||t<=0)return -1;
      let nonzero=false;
      for(let j=0;j<3;j++) {
        const o=cameraFloats[pair+8+j],d=cameraFloats[pair+12+j],endpoint=o+d*t,
          width=bounds[1][j]-bounds[0][j];
        if(!Number.isFinite(o)||!Number.isFinite(d)||!Number.isFinite(endpoint))return -1;
        nonzero ||= d!==0;
        start[j]=(o-bounds[0][j])/width*shape[j];
        end[j]=(endpoint-bounds[0][j])/width*shape[j];delta[j]=end[j]-start[j];
        if(!Number.isFinite(start[j])||!Number.isFinite(end[j])||!Number.isFinite(delta[j]))return -1;
      }
      return !nonzero||delta.every(v=>v===0)?-1:walkSegment();
    }
    if(!bounds||(family!==1&&family!==2)||!index[offset+11]||
        ![5,6].includes(nativeRows[base])||nativeRows[base]!==index[offset+1]||
        nativeRows[base+1]!==family||nativeRows[base+5]>=0x80000000||index[offset+15]!==1||
        index[offset+7]!==Number(nativeRows[base+30]!==0)||
        index[offset+6]!==UINT&&index[offset+6]!==nativeRows[base+112]||
        [0,1,2].some(j=>index[offset+8+j]!==nativeRows[base+21+j]||
          index[offset+12+j]!==nativeRows[base+27+j]))return -1;
    // The established 128-word codec copies entry origin/direction at 117/114
    // and the actual hit at 21. No infinite miss ray, projected endpoint,
    // inferred bounce, or completed-camera total is turned into an interval.
    let nonzeroDirection=false;
    for(let j=0;j<3;j++) {
      const d=sourceFloats[base+114+j],width=bounds[1][j]-bounds[0][j];
      if(!Number.isFinite(d)||!Number.isFinite(width)||width<=0)return -1;
      nonzeroDirection ||= d!==0;
      start[j]=(sourceFloats[base+117+j]-bounds[0][j])/width*shape[j];
      end[j]=(sourceFloats[base+21+j]-bounds[0][j])/width*shape[j];
      delta[j]=end[j]-start[j];
      if(!Number.isFinite(start[j])||!Number.isFinite(end[j])||!Number.isFinite(delta[j]))return -1;
    }
    if(!nonzeroDirection||delta.every(v=>v===0))return -1;
    return walkSegment();
  }
  function walkSegment() {
    let enter=0,leave=1;
    for(let j=0;j<3;j++) {
      if(delta[j]===0) {if(start[j]<0||start[j]>=shape[j])return 0;}
      else {
        const a=-start[j]/delta[j],b=(shape[j]-start[j])/delta[j];
        enter=Math.max(enter,Math.min(a,b));leave=Math.min(leave,Math.max(a,b));
      }
    }
    // Equality can be a tangent or a positive interval collapsed by rounding.
    // Do not invent a tolerance/cell or report a proven empty intersection.
    if(enter===leave)return -1;
    if(enter>leave)return 0;
    let count=0;times[count++]=enter;times[count++]=leave;
    for(let j=0;j<3;j++)if(delta[j]!==0)for(let plane=1;plane<shape[j];plane++) {
      const t=(plane-start[j])/delta[j];if(t>enter&&t<leave)times[count++]=t;
    }
    times.subarray(0,count).sort();
    let used=0;
    for(let i=1;i<count;i++) {
      const a=times[i-1],b=times[i];if(a===b)continue;
      const middle=a+(b-a)/2;
      // Refuse unresolved floating-point intervals rather than inventing a
      // spatial epsilon, widening support, or returning a partial traversal.
      if(!(middle>a&&middle<b))return -1;
      for(let j=0;j<3;j++) {
        const p=start[j]+delta[j]*middle;
        if(!Number.isFinite(p)||p<0||p>=shape[j])return -1;
        end[j]=Math.floor(p);
      }
      const cell=linear(end);if(!used||cells[used-1]!==cell)cells[used++]=cell;
    }
    return used;
  }
  function visit(emit,measure) {
    for(let row=0;row<rows;row++) {
      const offset=row*20,point=pointCell(offset),segments=segmentCells(offset);
      let includedPoint=false,refs=0;
      for(let j=0;j<Math.max(0,segments);j++) {
        const cell=cells[j],support=(index[offset]===3?4:2)|(cell===point?1:0);
        includedPoint ||= cell===point;emit(cell*stride+row*8+support);refs++;
      }
      if(point>=0&&!includedPoint){emit(point*stride+row*8+1);refs++;}
      if(measure) {
        stats.pointRows+=Number(point>=0);stats.segmentRows+=Number(segments>0&&index[offset]!==3);
        stats.cameraQueryRows+=Number(segments>0&&index[offset]===3);
        stats.cameraQueryUnavailableRows+=Number(cameraQuerySegments&&index[offset]===3&&segments<0);
        stats.segmentUnavailableRows+=Number(segments<0);
        stats.unindexedRows+=Number(refs===0);
      }
    }
  }
  visit(()=>{if(++stats.references*8+scratchReserve>budget)
    throw Error('Scene-record index exceeds its explicit memory budget; no partial index published.');},true);
  const keys=new Float64Array(stats.references);let cursor=0;
  visit(key=>{keys[cursor++]=key;},false);
  // In-place heapsort keeps index storage bounded; no per-reference JS objects
  // or unaccounted sort-sized scratch array is allocated.
  const down=(root,size)=>{
    while(root*2+1<size) {
      let child=root*2+1;if(child+1<size&&keys[child]<keys[child+1])child++;
      if(keys[root]>=keys[child])break;
      const value=keys[root];keys[root]=keys[child];keys[child]=value;root=child;
    }
  };
  for(let i=Math.floor(keys.length/2)-1;i>=0;i--)down(i,keys.length);
  for(let i=keys.length-1;i>0;i--){const value=keys[0];keys[0]=keys[i];keys[i]=value;down(0,i);}
  const fields={family:0,semantic:1,context:2,recipientNamespace:4,recipientOrdinal:5,directionSemantics:15,directionBin:19};
  function* records(cell,filters={}) {
    if(!Array.isArray(cell)||cell.length!==3||cell.some((v,j)=>!integer(v)||v>=shape[j]))
      throw Error('Scene-record lookup cell is outside the declared grid.');
    for(const [key,value]of Object.entries(filters))if(!Object.hasOwn(fields,key)||!integer(value))
      throw Error('Unsupported scene-record lookup filter.');
    const lower=linear(cell)*stride,upper=lower+stride;let lo=0,hi=keys.length;
    while(lo<hi){const mid=Math.floor((lo+hi)/2);if(keys[mid]<lower)lo=mid+1;else hi=mid;}
    for(let i=lo;i<keys.length&&keys[i]<upper;i++) {
      const local=keys[i]-lower,row=Math.floor(local/8),support=local%8;
      if(Object.entries(filters).every(([key,value])=>index[row*20+fields[key]]===value))yield {row,support};
    }
  }
  return Object.freeze({records,stats:Object.freeze(stats),bytes:keys.byteLength,scratchReserve,
    applicationReady:false,energyStatements:false,cameraQuerySegments,
    semantics:'references to exact observations; endpoint=1, copied-finite-segment=2, derived-camera-query-segment=4; not deposition or coverage',
    directionSemantics:'filters use the copied record direction/bin, not an inferred segment-entry bin',
    limitations:'Only delivered records are indexed. Camera query geometry does not establish evaluated surface, traversal, valid medium or energy. Missing/unbounded camera intervals and producer-discarded paths remain unavailable.'});
}

// Receive actual forward deposits and traced throughput, never invented lights.
import {inverseAffine} from './marble_studio_subject_renderer.js';

const DIGEST=/^[0-9a-f]{64}$/, NAME=/^light-field-[0-9a-f]{32}\.bin$/;
const sha256=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
export const MAX_FIELD_BYTES=256*256*256*12;
// Dense subject bundles can exceed 85 MiB over the public connection. Newer
// publications never cancel a transfer; only a changed scene/request does.
export const fieldTransferTimeoutMs=byteLength=>Math.min(300000,Math.max(30000,15000+Math.ceil(byteLength/(512*1024))*1000));
export const FIELD_RECEIVER_CATEGORIES=Object.freeze({subject:1,stage:2,room:4});
const boundedField=metadata=>metadata?.schema==='studio.forward-field.v5'||
  metadata?.schema==='studio.forward-field.v6'&&metadata.directional?.region?.kind==='subject';
const directionalBounds=metadata=>boundedField(metadata)?metadata.directional:metadata;
export function fieldCellDimensions(metadata) {
  if(!metadata)return null;
  const dimensions=metadata.directional?.dimensions??metadata.dimensions;
  const bounds=directionalBounds(metadata);
  return dimensions.map((n,i)=>(bounds.bounds_max_m[i]-bounds.bounds_min_m[i])/n);
}
const weightedField=metadata=>['studio.forward-field.v3','studio.forward-field.v4','studio.forward-field.v5','studio.forward-field.v6'].includes(metadata?.schema);
export function fieldAtlasLayout(dimensions,bins) {
  const [width,height,depth]=dimensions,columns=Math.ceil(bins/Math.floor(256/depth)),slabs=Math.ceil(bins/columns);
  return {columns,slabs,dimensions:[width*columns,height,depth*slabs]};
}
export function fieldStencilDirections(count) {
  if(count===6)return [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
  if(count!==12)throw Error('Choose 6 or 12 field directions.');
  const p=(1+Math.sqrt(5))/2,n=Math.hypot(1,p);
  return [[0,1,p],[0,-1,p],[0,1,-p],[0,-1,-p],[1,p,0],[-1,p,0],[1,-p,0],[-1,-p,0],
    [p,0,1],[p,0,-1],[-p,0,1],[-p,0,-1]].map(v=>v.map(x=>x/n));
}
export function validateLightField(value,digest,revision) {
  const vector=x=>Array.isArray(x)&&x.length===3&&x.every(Number.isFinite);
  if(!value||!['studio.forward-field.v1','studio.forward-field.v2','studio.forward-field.v3','studio.forward-field.v4','studio.forward-field.v5','studio.forward-field.v6'].includes(value.schema)||!DIGEST.test(digest)||value.digest!==digest||
      value.live_revision!==revision||!Number.isSafeInteger(value.sequence)||value.sequence<0||
      !NAME.test(value.name)||value.format!=='rgb-f32le'||value.layout!=='zyx-interleaved-rgb'||
      value.color_space!=='linear-srgb'||!vector(value.dimensions)||
      !value.dimensions.every(n=>Number.isSafeInteger(n)&&n>=1&&n<=256)||
      !vector(value.bounds_min_m)||!vector(value.bounds_max_m)||
      value.bounds_max_m.some((n,i)=>n<=value.bounds_min_m[i])||
      !Number.isSafeInteger(value.byte_length)||value.byte_length>MAX_FIELD_BYTES)
    throw Error('Invalid or stale forward-field metadata.');
  if(value.surface_transport!==undefined&&value.surface_transport!=='studio-pbr-mixture-v1')throw Error('Invalid surface transport receipt.');
  const scalarBytes=value.dimensions.reduce((a,b)=>a*b,12);
  if(value.schema==='studio.forward-field.v1') {
    if(value.byte_length!==scalarBytes)throw Error('Invalid scalar field size.');
  } else {
    const d=value.directional,t=value.triangles,s=value.sources,m=value.materials,a=value.source_pbr16;
    const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
    const v6=value.schema==='studio.forward-field.v6',v5=boundedField(value),v4=v5||v6||value.schema==='studio.forward-field.v4',v3=weightedField(value),bins=v3?d?.bin_count:6;
    if(v4&&(!vector(d?.dimensions)||![32,64].includes(d.dimensions[0])||!d.dimensions.every(n=>n===d.dimensions[0])))
      throw Error('Invalid normalized field grid.');
    if(v5) {
      const region=d.region,pitch=d.cell_size_m,margin=d.margin_m;
      if(!Number.isFinite(pitch)||pitch<.00005||pitch>.01||!Number.isFinite(margin)||margin<0||margin>.032||
          !vector(d.bounds_min_m)||!vector(d.bounds_max_m)||region?.kind!=='subject'||
          !DIGEST.test(value.geometry_sha256)||region.subject_geometry_sha256!==value.geometry_sha256||
          !DIGEST.test(region.world_geometry_sha256)||!vector(region.actual_bounds_min_m)||!vector(region.actual_bounds_max_m)||
          !vector(region.padding_min_m)||!vector(region.padding_max_m)||
          region.actual_bounds_max_m.some((v,i)=>v<region.actual_bounds_min_m[i])||region.effective_margin_m!==Math.max(margin,pitch))
        throw Error('Invalid subject region identity or cell size.');
      const lower=region.actual_bounds_min_m.map(v=>Math.floor((v-region.effective_margin_m)/pitch));
      const upper=region.actual_bounds_max_m.map(v=>Math.ceil((v+region.effective_margin_m)/pitch));
      const required=upper.map((v,i)=>Math.max(1,v-lower[i])),grid=required.every(n=>n<=32)?32:64;
      const near=(a,b)=>Math.abs(a-b)<=Math.max(1e-9,1e-6*Math.max(Math.abs(a),Math.abs(b)));
      if(required.some(n=>n>64)||d.dimensions[0]!==grid||d.bounds_min_m.some((v,i)=>
          !near(v,(lower[i]-Math.floor((grid-required[i])/2))*pitch)||
          !near(d.bounds_max_m[i],(lower[i]-Math.floor((grid-required[i])/2)+grid)*pitch)||
          !near(region.padding_min_m[i],region.actual_bounds_min_m[i]-v)||
          !near(region.padding_max_m[i],d.bounds_max_m[i]-region.actual_bounds_max_m[i])||
          !(v<region.actual_bounds_min_m[i]&&d.bounds_max_m[i]>region.actual_bounds_max_m[i])))
        throw Error('Subject region does not preserve its exact cell size, margin and full bounds.');
    }
    if(v4) {
      const r=d.root_sampling;
      if(!r||r.version!=='per-complete-pump-inverse-replay-v1'||
          !Number.isSafeInteger(r.source_root_count)||r.source_root_count<1||r.source_root_count!==s?.count||
          !Number.isSafeInteger(r.completed_root_sweeps)||r.completed_root_sweeps<1||
          !Number.isSafeInteger(r.source_root_count*r.completed_root_sweeps)||
          !Number.isSafeInteger(r.recorded_root_jobs)||r.recorded_root_jobs<r.source_root_count*r.completed_root_sweeps||
          r.normalization_divisor!==r.completed_root_sweeps||
          !Number.isSafeInteger(r.excluded_partial_root_jobs)||r.excluded_partial_root_jobs<0||r.incomplete_dispatch!==false)
        throw Error('Invalid root-sweep normalization receipt.');
    }
    if(v6) {
      const p=d.source_populations,c=d.coverage;
      if(d.internal_media!==undefined) {
        const expected={version:'canonical-beer-segment-integral-v1',scope:'actual valid interior and exterior traced segments',
          entry_packet:true,integration:'per-recording-cell integral of canonical Beer transmittance',
          optical_authority:'installed medium/region/dye helper',invalid_ownership:'excluded',external_extinction:'unsupported; request rejected'};
        if(!d.internal_media||Object.entries(expected).some(([key,value])=>d.internal_media[key]!==value))
          throw Error('Invalid internal medium recording receipt.');
      }
      if(d.region&&!v5||d.quantization_divisor!==16777216||d.quantization_version!=='fixed24-track-length-v1'||
          !p||p.version!=='native-angular-refresh-mean-v1'||p.generator!=='native-emissive-triangle-roots-v1'||
          !Number.isSafeInteger(p.ray_budget)||p.ray_budget<128||p.ray_budget>65536||
          !Number.isSafeInteger(p.base_seed)||p.base_seed<0||p.base_seed>2147483647||
          !Number.isSafeInteger(p.first_index)||p.first_index<0||!Number.isSafeInteger(p.last_index)||p.last_index<p.first_index||
          !Number.isSafeInteger(p.count)||p.count!==p.last_index-p.first_index+1||p.count!==d.root_sampling.completed_root_sweeps||
          p.latest_seed!==(p.base_seed+p.last_index%2147483648)%2147483648||
          !DIGEST.test(p.latest_sources_sha256)||!DIGEST.test(p.rolling_sha256)||
          !c||!Number.isSafeInteger(c.nonzero_cells)||c.nonzero_cells<0||c.total_cells!==d.dimensions.reduce((a,b)=>a*b,1)||
          c.nonzero_cells>c.total_cells||typeof c.meaning!=='string'||!c.meaning)
        throw Error('Invalid cumulative source-population receipt.');
    }
    if(v3&&(![6,12].includes(bins)||d.stencil!==(bins===6?'axes6-v1':'icosahedron12-v1')||
        !vector(d.weight_from_linear_rgb)||!d.weight_from_linear_rgb.some(x=>x!==0)||
        !Array.isArray(d.bin_directions)||d.bin_directions.length!==bins||
        d.bin_directions.some((v,i)=>!vector(v)||v.some((x,j)=>Math.abs(x-fieldStencilDirections(bins)[i][j])>1e-12))))
      throw Error('Invalid directional stencil or energy weights.');
    if(!DIGEST.test(value.sha256)||!DIGEST.test(value.scene_sha256)||
        !d||!vector(d.dimensions)||!d.dimensions.every(n=>Number.isSafeInteger(n)&&n>=1&&n<=(v4?64:32))||
        d.format!=='rgb-direction-f32le'||d.layout!=='zyx-bin-channel'||
        d.estimator!==(v4?'track-length-root-sweep-v1':'track-length-throughput-v1')||d.direction_space!=='world'||
        !equal(d.bins,bins===6?['+X','-X','+Y','-Y','+Z','-Z']:Array.from({length:12},(_,i)=>`I${i}`))||
        !equal(d.channels,['R','G','B','mean_propagation_x','mean_propagation_y','mean_propagation_z'])||
        !t||t.format!=='u32le'||t.stride_bytes!==16||
        !equal(t.channels,['triangle_id','material_id','flags','class_mask'])||
        t.class_bits?.volumetric!==1||t.class_bits?.reflectance!==2||t.class_bits?.glass!==4||
        t.ordering!=='original packed scene triangles, before BVH reorder'||
        !s||s.format!=='f32le'||s.stride_bytes!==48||s.layout!=='baked-source-records'||
        !equal(s.channels,['position_x_m','position_y_m','position_z_m','amplitude_weight','direction_x','direction_y','direction_z','kind','frequency_hz','phase_radians','energy','packet_aux'])||
        !m||m.stride_bytes!==64||m.format!=='f32le'||m.layout!=='native-pbr16'||
        !Number.isSafeInteger(m.count)||m.count<1||m.count>65536||
        !a||a.stride_bytes!==64||a.format!=='f32le'||a.layout!=='authored-pbr16'||
        !Number.isSafeInteger(a.count)||a.count<0||!Array.isArray(a.material_ids)||a.material_ids.length!==a.count||
        !a.material_ids.every(id=>Number.isSafeInteger(id)&&id>=0&&id<m.count))
      throw Error('Invalid directional field metadata.');
    let offset=scalarBytes;
    for(const [section,expected] of [[d,d.dimensions.reduce((a,b)=>a*b,24*bins)],[t,t.count*16],[s,s.count*48],[m,m.count*64],[a,a.count*64]]) {
      if(!Number.isSafeInteger(section.count??0)||(section.count??0)<0||
          !Number.isSafeInteger(expected)||expected<0||section.offset_bytes!==offset||section.byte_length!==expected)
        throw Error('Invalid or overlapping field sections.');
      offset+=expected;
    }
    if(offset!==value.byte_length)throw Error('Incomplete field sections.');
  }
  return structuredClone(value);
}

export function fieldFromView(view,unitM,origin,boundsMin,boundsMax) {
  if(!Number.isFinite(unitM)||unitM<=0||![origin,boundsMin,boundsMax].every(v=>
      v?.length===3&&Array.from(v).every(Number.isFinite))||boundsMax.some((v,i)=>v<=boundsMin[i]))
    throw Error('Invalid forward-field coordinate mapping.');
  // The shared affine inverse already handles every Studio model/view frame.
  // vPosV includes each draw's object transform; inverse(view) recovers world.
  const inverse=inverseAffine(view),result=new Float32Array(inverse);
  for(let row=0;row<3;row++) {
    const span=boundsMax[row]-boundsMin[row],scale=unitM/span;
    for(let col=0;col<4;col++)result[col*4+row]=inverse[col*4+row]*scale;
    result[12+row]+=(origin[row]-boundsMin[row])/span;
  }
  if(!Array.from(result).every(Number.isFinite))throw Error('Forward-field coordinates exceed float32 range.');
  return result;
}

export function decodeLightField(buffer,metadata) {
  if(!(buffer instanceof ArrayBuffer)||buffer.byteLength!==metadata.byte_length)
    throw Error('Incomplete forward-field payload.');
  const view=new DataView(buffer),values=new Float32Array(metadata.dimensions.reduce((a,b)=>a*b,3));
  for(let i=0;i<values.length;i++) {
    const v=view.getFloat32(i*4,true);if(!Number.isFinite(v))throw Error('Nonfinite forward-field sample.');
    values[i]=v; // Signed out-of-gamut linear RGB is intentional.
  }
  if(metadata.schema==='studio.forward-field.v1')return values;
  const d=metadata.directional,t=metadata.triangles,s=metadata.sources;
  const v3=weightedField(metadata),bins=v3?d.bin_count:6;
  const atlas=fieldAtlasLayout(d.dimensions,bins),columns=atlas.columns;
  const [width,height,depth]=d.dimensions,voxels=width*height*depth;
  const atlasVoxels=atlas.dimensions.reduce((a,b)=>a*b,1);
  const radiance=new Float32Array(atlasVoxels*3),direction=new Float32Array(atlasVoxels*4);
  let nonzeroCells=0,nonzeroBins=0,lastOccupiedVoxel=-1;
  const wr=v3?d.weight_from_linear_rgb[0]:0,wg=v3?d.weight_from_linear_rgb[1]:0,wb=v3?d.weight_from_linear_rgb[2]:0;
  for(let voxel=0;voxel<voxels;voxel++)for(let bin=0;bin<bins;bin++) {
    const x=voxel%width,y=Math.floor(voxel/width)%height,z=Math.floor(voxel/(width*height));
    const dst=(((Math.floor(bin/columns)*depth+z)*height+y)*(width*columns)+(bin%columns)*width+x);
    const src=d.offset_bytes+(voxel*bins+bin)*24;
    const r=view.getFloat32(src,true),g=view.getFloat32(src+4,true),b=view.getFloat32(src+8,true);
    const dx=view.getFloat32(src+12,true),dy=view.getFloat32(src+16,true),dz=view.getFloat32(src+20,true);
    if(!Number.isFinite(r)||!Number.isFinite(g)||!Number.isFinite(b)||
        !Number.isFinite(dx)||!Number.isFinite(dy)||!Number.isFinite(dz))throw Error('Nonfinite directional sample.');
    if(Math.hypot(dx,dy,dz)>1.00001)throw Error('Invalid mean propagation direction.');
    if(r!==0||g!==0||b!==0){nonzeroBins++;if(lastOccupiedVoxel!==voxel){nonzeroCells++;lastOccupiedVoxel=voxel;}}
    // RGB may be signed: recover nonnegative line energy with the producer's
    // inverse color transform, not RGB luminance or absolute RGB components.
    const rawWeight=v3?((0+r*wr)+g*wg)+b*wb:1;
    const scale=v3?((0+Math.abs(r*wr))+Math.abs(g*wg))+Math.abs(b*wb):1;
    if(!Number.isFinite(rawWeight)||rawWeight < -1e-6*scale)throw Error('Invalid directional energy weight.');
    const weight=Math.max(0,rawWeight);
    const rgb=dst*3,rgba=dst*4;
    radiance[rgb]=r;radiance[rgb+1]=g;radiance[rgb+2]=b;
    direction[rgba]=dx*weight;direction[rgba+1]=dy*weight;direction[rgba+2]=dz*weight;direction[rgba+3]=weight;
    if(!Number.isFinite(direction[rgba])||!Number.isFinite(direction[rgba+1])||
        !Number.isFinite(direction[rgba+2])||!Number.isFinite(direction[rgba+3]))throw Error('Directional energy exceeds float32 range.');
  }
  const triangles=new Uint32Array(t.count*4),materialMasks=new Uint32Array(metadata.materials.count);
  for(let i=0;i<t.count;i++) {
    const row=Array.from({length:4},(_,c)=>view.getUint32(t.offset_bytes+i*16+c*4,true));
    if(row[0]!==i||row[1]>=materialMasks.length||(row[3]&~7)!==0)throw Error('Invalid triangle material identity.');
    triangles.set(row,i*4);materialMasks[row[1]]|=row[3];
  }
  const sources=new Float32Array(s.count*12);
  for(let i=0;i<sources.length;i++) {
    const v=view.getFloat32(s.offset_bytes+i*4,true);
    if(!Number.isFinite(v))throw Error('Nonfinite source record.');sources[i]=v;
  }
  const materials=new Float32Array(metadata.materials.count*16);
  for(let i=0;i<materials.length;i++) {
    const v=view.getFloat32(metadata.materials.offset_bytes+i*4,true);
    if(!Number.isFinite(v))throw Error('Nonfinite material record.');materials[i]=v;
  }
  const authoredMaterials=new Float32Array(metadata.source_pbr16.count*16);
  for(let i=0;i<authoredMaterials.length;i++) {
    const v=view.getFloat32(metadata.source_pbr16.offset_bytes+i*4,true);
    if(!Number.isFinite(v))throw Error('Nonfinite authored material record.');authoredMaterials[i]=v;
  }
  return {scalar:values,radiance,direction,atlasDimensions:atlas.dimensions,binCount:bins,atlasColumns:columns,triangles,materialMasks,sources,materials,
    coverage:{nonzeroCells,totalCells:voxels,nonzeroBins,totalBins:voxels*bins},
    authoredMaterials,authoredMaterialIds:metadata.source_pbr16.material_ids.slice()};
}

// IDs are local to each assembled scene. Match exact submitted PBR16 bytes,
// then follow recorded authored-to-native lineage to the native class masks.
// Engine-derived native rows are not evidence of an authored material identity.
export function mapFieldMaterials(decoded,local) {
  if(!local||local.stride!==16||!local.data||local.data.length%16||local.data.length>65536*16)
    throw Error('Current scene material identity is unavailable.');
  if(local.lookupData&&local.lookupData.length!==local.data.length)throw Error('Invalid projected material table.');
  const data=new Float32Array(local.lookupData??local.data);
  if(!Array.from(data).every(Number.isFinite)||!Array.from(local.data).every(Number.isFinite))throw Error('Invalid current material table.');
  const signature=(rows,i)=>Array.from(new Uint32Array(rows.buffer,rows.byteOffset+i*64,16)).join(',');
  const byRow=new Map(),masks=new Uint32Array(data.length/16);
  for(let i=0;i<decoded.authoredMaterialIds.length;i++) {
    const mask=decoded.materialMasks[decoded.authoredMaterialIds[i]];if(!mask)continue;
    const sig=signature(decoded.authoredMaterials,i);
    byRow.set(sig,(byRow.get(sig)??0)|mask);
  }
  const matched=new Set();
  for(let i=0;i<masks.length;i++) {
    const sig=signature(data,i);
    if(byRow.has(sig)){masks[i]=byRow.get(sig);matched.add(sig);}
  }
  if(!matched.size)throw Error('Ray field does not match current scene materials.');
  return masks;
}

export function createStudioLightField({gl,doc=document,getSceneKey,getSceneMaterials,fetcher=fetch,onChange=()=>{},fallbackTexture=null}) {
  const toggle=doc.getElementById('phong-light-field'),output=doc.getElementById('phong-light-field-status');
  const stencilControl=doc.getElementById('phong-field-stencil'),spatialControl=doc.getElementById('phong-field-spatial-linear');
  const transportControl=doc.getElementById('ray-surface-transport');
  const surfaceTransport=()=>transportControl?.checked?'studio-pbr-mixture-v1':null;
  const gridControl=doc.getElementById('phong-field-grid-size');
  const regionControl=doc.getElementById('phong-field-region'),cellControl=doc.getElementById('phong-field-cell-mm'),marginControl=doc.getElementById('phong-field-margin-mm');
  const recording=()=>{
    const region=regionControl?.value??'scene';
    if(!['scene','subject'].includes(region))throw Error('Choose whole scene or bounded subject recording.');
    if(region==='scene')return {region};
    const cellValue=cellControl?.value??'.5',marginValue=marginControl?.value??'1';
    const cellSizeM=String(cellValue).trim()?Number(cellValue)/1000:NaN,marginM=String(marginValue).trim()?Number(marginValue)/1000:NaN;
    if(!Number.isFinite(cellSizeM)||cellSizeM<.00005||cellSizeM>.01)throw Error('Subject cell size must be between 0.05 and 10 mm.');
    if(!Number.isFinite(marginM)||marginM<0||marginM>.032)throw Error('Subject margin must be between 0 and 32 mm.');
    return {region,cellSizeM,marginM};
  };
  const recordingKey=()=>JSON.stringify(recording());
  const syncRecordingControls=()=>{
    const subject=regionControl?.value==='subject';if(gridControl)gridControl.disabled=subject;
    if(cellControl)cellControl.disabled=!subject;if(marginControl)marginControl.disabled=!subject;
  };
  const divisionOutput=doc.getElementById('phong-field-division-status'),categoryOutput=doc.getElementById('phong-field-category-status');
  const categories=Object.entries(FIELD_RECEIVER_CATEGORIES).map(([name,bit])=>({name,bit,control:doc.getElementById(`phong-field-${name}`)}));
  const categoryMask=()=>disposed?0:categories.reduce((mask,{bit,control})=>mask|(control?.checked===true?bit:0),0);
  const gridSize=()=>{const n=Number(gridControl?.value??64);if(![32,64].includes(n))throw Error('Choose a 32 or 64 field grid.');return n;};
  const stencil=()=>{const n=Number(stencilControl?.value??12);if(![6,12].includes(n))throw Error('Choose 6 or 12 field directions.');return n;};
  const floatLinear=!!gl.getExtension?.('OES_texture_float_linear');
  const spatialLinear=()=>weightedField(metadata)&&spatialControl?.checked!==false&&floatLinear;
  let epoch=0,source=null,metadata=null,texture=null,pending=null,disposed=false,directionalTextures=null,coverage=null;
  const say=text=>{if(output)output.textContent=text;};
  const describeDivisions=()=>{
    if(!divisionOutput)return;
    const cell=fieldCellDimensions(metadata);
    const mm=values=>values.map(v=>Number((v*1000).toPrecision(6))).join(', ');
    const roi=boundedField(metadata)?` Requested margin: ${metadata.directional.margin_m*1000} mm; minimum guard: ${metadata.directional.region.effective_margin_m*1000} mm. Actual lower / upper padding (X, Y, Z): [${mm(metadata.directional.region.padding_min_m)}] / [${mm(metadata.directional.region.padding_max_m)}] mm. Padded recording bounds: [${mm(metadata.directional.bounds_min_m)}] to [${mm(metadata.directional.bounds_max_m)}] mm.`:'';
    divisionOutput.textContent=cell?`Received ${(metadata.directional?.dimensions??metadata.dimensions).join(' × ')} grid; cell dimensions: ${cell.map(n=>Number((n*1000).toPrecision(5))).join(' × ')} mm (world units).${roi} 64 divisions use 8× the spatial storage of 32.`:
      'World-space cell dimensions appear when a field bundle arrives. 64 divisions use 8× the spatial storage of 32.';
  };
  const categoryChange=()=>{
    if(categoryOutput)categoryOutput.textContent=categoryMask()?`Receiving categories: ${categories.filter(({bit})=>categoryMask()&bit).map(({name})=>name).join(', ')}. One global field; ray scene unchanged.`:
      'No receiving categories selected. Ordinary primary shading remains active.';
  };
  const drop=()=>{
    if(directionalTextures)for(const t of directionalTextures)gl.deleteTexture(t);
    directionalTextures=null;
    if(texture){
      const active=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);
      if(gl.getParameter(gl.TEXTURE_BINDING_3D)===texture)gl.bindTexture(gl.TEXTURE_3D,fallbackTexture);
      gl.deleteTexture(texture);gl.activeTexture(active);
    }
    texture=null;metadata=null;coverage=null;describeDivisions();
  };
  const requested=()=>!disposed&&toggle?.checked===true;
  const invalidate=(reason='Scene changed. Waiting for matching ray data.')=>{
    epoch++;source=null;pending?.controller.abort();pending=null;drop();if(requested())say(reason);
  };
  const key=context=>getSceneKey(context);
  const current=s=>requested()&&source===s&&s.epoch===epoch&&key(s.context)===s.key&&stencil()===s.stencil&&gridSize()===s.gridSize&&recordingKey()===s.recordingKey&&s.surfaceTransport===surfaceTransport();
  const capture=(context={})=>{
    if(!requested())return null;
    try{return {key:key(context),context:{...context},epoch,stencil:stencil(),gridSize:gridSize(),...recording(),recordingKey:recordingKey(),surfaceTransport:surfaceTransport(),
      materials:getSceneMaterials?structuredClone(getSceneMaterials(context)):null};}
    catch(error){say(`Field not requested: ${error.message}`);return null;}
  };
  const connect=(digest,ticket,revision=0)=>{
    if(!ticket||!requested()||ticket.epoch!==epoch||!DIGEST.test(digest))return;
    try{if(key(ticket.context)!==ticket.key||ticket.stencil!==stencil()||ticket.gridSize!==gridSize()||ticket.recordingKey!==recordingKey()||ticket.surfaceTransport!==surfaceTransport())return;
      pending?.controller.abort();pending=null;drop();source={...ticket,digest,revision};
      say("Opted in. Waiting for the ray tracer's field bundle.");
    }catch(error){invalidate(error.message);}
  };
  const observe=async(status,digest)=>{
    const s=source;if(!s||s.digest!==digest)return;
    let transferFailed=false;
    try {
      if(!current(s)){drop();return;}
      if(['error','canceled'].includes(status.state)){invalidate('Ray job ended without a current field.');return;}
      if(!status.light_field)return;
      const next=validateLightField(status.light_field,digest,s.revision);
      if((next.surface_transport??null)!==s.surfaceTransport)throw Error('Ray field surface transport does not match the captured request.');
      if(weightedField(next)&&next.directional.bin_count!==s.stencil)throw Error('Ray field stencil does not match the captured request.');
      if((next.schema==='studio.forward-field.v4'||next.schema==='studio.forward-field.v6'&&!boundedField(next))&&next.directional.dimensions[0]!==s.gridSize)throw Error('Ray field grid does not match the captured request.');
      if(boundedField(next)&&(s.region!=='subject'||next.directional.cell_size_m!==s.cellSizeM||next.directional.margin_m!==s.marginM))
        throw Error('Subject field pitch or margin does not match the captured request.');
      if(s.region==='subject'&&!boundedField(next)&&next.schema!=='studio.forward-field.v1')
        throw Error('Subject field response does not match the requested recording region.');
      if(metadata&&next.sequence<=metadata.sequence||pending)return;
      const controller=new AbortController(),work={controller};pending=work;
      let timedOut=false;
      const timer=setTimeout(()=>{timedOut=true;controller.abort();},fieldTransferTimeoutMs(next.byte_length));
      say(`Downloading field bundle ${next.sequence} (${(next.byte_length/1048576).toFixed(1)} MiB). ${metadata?'The last received field remains active.':'Phong remains visible while the field arrives.'}`);
      try {
        transferFailed=true;
        const response=await fetcher(`/api/marble_photo/field?digest=${digest}&name=${encodeURIComponent(next.name)}`,
          {cache:'no-store',signal:controller.signal});
        if(response.status===404||response.status===409){
          say(`Field bundle changed before transfer (HTTP ${response.status}). Retrying the latest publication.${metadata?' The last received field remains active.':''}`);
          return;
        }
        if(!response.ok)throw Error(`Forward-field HTTP ${response.status}`);
        const length=response.headers?.get?.('Content-Length');
        if(length!==null&&length!==undefined&&Number(length)!==next.byte_length)throw Error('Forward-field length changed.');
        const bytes=await response.arrayBuffer();transferFailed=false;if(!current(s))return;
        const values=decodeLightField(bytes,next);
        if(next.sha256){
          if(!DIGEST.test(next.sha256))throw Error('Invalid forward-field content hash.');
          const hash=await sha256(bytes);
          if(hash!==next.sha256)throw Error('Forward-field content hash mismatch.');
        }
        if(next.schema==='studio.forward-field.v6') {
          const population=next.directional.source_populations,section=next.sources;
          if(values.coverage.nonzeroCells!==next.directional.coverage.nonzero_cells)throw Error('Field coverage receipt does not match decoded energy.');
          const sourceHash=await sha256(new Uint8Array(bytes,section.offset_bytes,section.byte_length));
          if(sourceHash!==population.latest_sources_sha256)throw Error('Latest source population hash mismatch.');
          const previous=metadata?.directional?.source_populations;
          const priorHash=population.count===1?'0'.repeat(64):previous&&previous.first_index===population.first_index&&
            previous.last_index+1===population.last_index&&previous.count+1===population.count?previous.rolling_sha256:null;
          // Polling can skip publications. Check each adjacent link available
          // here; the API verifies the complete producer provenance receipt.
          if(priorHash!==null) {
            const chain=new Uint8Array(72),hex=hash=>Uint8Array.from(hash.match(/../g),part=>parseInt(part,16));
            chain.set(hex(priorHash));new DataView(chain.buffer).setBigUint64(32,BigInt(population.last_index),true);chain.set(hex(sourceHash),40);
            if(await sha256(chain)!==population.rolling_sha256)throw Error('Source population history hash mismatch.');
          }
        }
        if(!current(s))return;
        const isDirectional=next.schema!=='studio.forward-field.v1';
        const masks=isDirectional?mapFieldMaterials(values,s.materials):null;
        const previousActive=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);
        const previous=gl.getParameter(gl.TEXTURE_BINDING_3D),oldTexture=texture,unpack=gl.getParameter(gl.UNPACK_ALIGNMENT);
        let nextTexture=null,newDirectional=null;
        const owned=[];
        try {
          if(gl.isContextLost?.())throw Error('WebGL context lost.');
          if(next.dimensions.some(n=>n>gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)))throw Error('Forward volume exceeds this GPU’s texture limit.');
          nextTexture=gl.createTexture();if(!nextTexture)throw Error('Forward volume allocation failed.');
          owned.push(nextTexture);
          gl.bindTexture(gl.TEXTURE_3D,nextTexture);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);
          for(const parameter of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_3D,parameter,gl.NEAREST);
          for(const parameter of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,parameter,gl.CLAMP_TO_EDGE);
          gl.texImage3D(gl.TEXTURE_3D,0,gl.RGB32F,...next.dimensions,0,gl.RGB,gl.FLOAT,isDirectional?values.scalar:values);
          if(gl.getError()!==gl.NO_ERROR)throw Error('Forward volume upload failed.');
          if(isDirectional) {
            const dims=values.atlasDimensions;
            if(dims.some(n=>n>gl.getParameter(gl.MAX_3D_TEXTURE_SIZE))||gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)<14)
              throw Error('Directional field exceeds GPU texture limits.');
            newDirectional=[];
            for(const [index,data] of [values.radiance,values.direction].entries()) {
              const tex=gl.createTexture();if(!tex)throw Error('Directional texture allocation failed.');owned.push(tex);newDirectional.push(tex);
              gl.bindTexture(gl.TEXTURE_3D,tex);
              const filter=weightedField(next)&&spatialControl?.checked!==false&&floatLinear?gl.LINEAR:gl.NEAREST;
              for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_3D,p,filter);
              for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,p,gl.CLAMP_TO_EDGE);
              gl.texImage3D(gl.TEXTURE_3D,0,index?gl.RGBA32F:gl.RGB32F,...dims,0,index?gl.RGBA:gl.RGB,gl.FLOAT,data);
              if(gl.getError()!==gl.NO_ERROR)throw Error('Directional texture upload failed.');
            }
            const tex=gl.createTexture();if(!tex)throw Error('Material mask allocation failed.');owned.push(tex);newDirectional.push(tex);
            const previous2D=gl.getParameter(gl.TEXTURE_BINDING_2D);
            try {
              const limit=gl.getParameter(gl.MAX_TEXTURE_SIZE),width=Math.min(256,limit,masks.length),height=Math.ceil(masks.length/width);
              if(!width||height>limit)throw Error('Material mask exceeds texture limits.');
              const data=new Uint32Array(width*height);data.set(masks);
              gl.bindTexture(gl.TEXTURE_2D,tex);
              for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
              for(const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
              gl.texImage2D(gl.TEXTURE_2D,0,gl.R32UI,width,height,0,gl.RED_INTEGER,gl.UNSIGNED_INT,data);
              if(gl.getError()!==gl.NO_ERROR)throw Error('Material mask upload failed.');
            }finally{gl.bindTexture(gl.TEXTURE_2D,previous2D);}
          }
          if(gl.isContextLost?.()||!current(s))throw Error('Field scene changed during upload.');
          drop();texture=nextTexture;directionalTextures=newDirectional;metadata=next;coverage=isDirectional?values.coverage:null;owned.length=0;describeDivisions();
        }finally{
          for(const tex of owned)gl.deleteTexture(tex);
          gl.pixelStorei(gl.UNPACK_ALIGNMENT,unpack);gl.bindTexture(gl.TEXTURE_3D,previous===oldTexture&&oldTexture!==null?texture:previous);gl.activeTexture(previousActive);
        }
        say(`Ray field received: ${next.dimensions.join(' × ')} voxels · ${Number(next.ray_jobs??0).toLocaleString()} forward rays. Existing Phong volume glow; not directional lighting.`);
        if(isDirectional)describeDirectional();
        else if(next.directional_unavailable?.reason==='signed-accumulator-overflow')
          say('Directional recording reached its range limit. Scalar preview remains available; directional recording resumes after the ray field clears.');
      }catch(error){
        if(timedOut&&error.name==='AbortError')throw Error('Field download timed out. Retrying on the next update.');
        throw error;
      }finally{clearTimeout(timer);if(pending===work)pending=null;}
    }catch(error){
      if(source===s&&s.epoch===epoch&&error.name!=='AbortError'){
        if(!transferFailed)drop();
        say(`Field update unavailable: ${error.message}${metadata?' The last received field remains active.':''}`);
      }
    }
  };
  const bind=(program,view,unitM,origin)=>{
    const uniform=name=>gl.getUniformLocation(program,name);
    gl.uniform1f(uniform('uFieldGain'),0);gl.uniform1i(uniform('uFieldUseTransform'),0);
    if(!categoryMask()||!texture||!source)return false;
    try {
      if(gl.isContextLost?.()||!current(source)){drop();return false;}
      if(uniform('uFieldFromView')===null||uniform('uFieldUseTransform')===null)throw Error('This shader export does not support field coordinates yet.');
      const previous=gl.getParameter(gl.ACTIVE_TEXTURE);gl.activeTexture(gl.TEXTURE8);gl.bindTexture(gl.TEXTURE_3D,texture);gl.activeTexture(previous);
      gl.uniformMatrix4fv(uniform('uFieldFromView'),false,fieldFromView(view,unitM,origin,metadata.bounds_min_m,metadata.bounds_max_m));
      gl.uniform1i(uniform('uFieldVolume'),8);gl.uniform1i(uniform('uFieldUseTransform'),1);gl.uniform1f(uniform('uFieldGain'),1);
      return true;
    }catch(error){drop();say(`Field unavailable: ${error.message}`);return false;}
  };
  const directionalCurrent=()=>{
    if(!directionalTextures||!source)return false;
    if(gl.isContextLost?.()||!current(source)){drop();return false;}
    const local=getSceneMaterials?.(source.context),saved=source.materials;
    const sameRows=(a,b)=>a?.length===b?.length&&a!=null&&
      Array.from(a).every((v,i)=>Object.is(Math.fround(v),Math.fround(b[i])));
    if(!local||local.stride!==saved.stride||!sameRows(local.data,saved.data)||
        !sameRows(local.lookupData??local.data,saved.lookupData??saved.data)) {
      drop();say('Field unavailable: current materials changed.');return false;
    }
    return true;
  };
  const bindSecondary=(program,view,unitM,origin)=>{
    try {
      if(!categoryMask()){gl.uniform1f(gl.getUniformLocation(program,'uFieldGain'),0);return false;}
      if(!directionalCurrent())return false;
      const names=['uFieldRadiance','uFieldDirection','uFieldMaterialMask','uFieldGridDepth','uFieldViewFromWorld','uFieldFromView',
        'uFieldBinCount','uFieldAtlasColumns','uFieldSpatialLinear'];
      const u=Object.fromEntries(names.map(n=>[n,gl.getUniformLocation(program,n)]));
      if(Object.values(u).some(v=>v===null))throw Error('Secondary shader lacks directional field uniforms.');
      const bounds=directionalBounds(metadata);
      const mapping=fieldFromView(view,unitM,origin,bounds.bounds_min_m,bounds.bounds_max_m);
      const rotation=new Float32Array([view[0],view[1],view[2],view[4],view[5],view[6],view[8],view[9],view[10]]);
      if(!Array.from(rotation).every(Number.isFinite))throw Error('Invalid field direction transform.');
      const active=gl.getParameter(gl.ACTIVE_TEXTURE);
      try {for(let i=0;i<3;i++) {
        gl.activeTexture(gl.TEXTURE0+11+i);gl.bindTexture(i===2?gl.TEXTURE_2D:gl.TEXTURE_3D,directionalTextures[i]);
      }}finally{gl.activeTexture(active);}
      gl.uniform1i(u.uFieldRadiance,11);gl.uniform1i(u.uFieldDirection,12);gl.uniform1i(u.uFieldMaterialMask,13);
      gl.uniform1f(u.uFieldGridDepth,metadata.directional.dimensions[2]);
      const bins=weightedField(metadata)?metadata.directional.bin_count:6;
      gl.uniform1i(u.uFieldBinCount,bins);gl.uniform1i(u.uFieldAtlasColumns,fieldAtlasLayout(metadata.directional.dimensions,bins).columns);gl.uniform1i(u.uFieldSpatialLinear,spatialLinear()?1:0);
      gl.uniformMatrix3fv(u.uFieldViewFromWorld,false,rotation);
      gl.uniformMatrix4fv(u.uFieldFromView,false,mapping);
      gl.uniform1i(gl.getUniformLocation(program,'uFieldRestrictToBounds'),1);
      // The primary is deliberately field-free. Its copied zero gain must not
      // suppress the directional loop in the immediately following program.
      gl.uniform1f(gl.getUniformLocation(program,'uFieldGain'),1);
      return true;
    }catch(error){drop();say(`Field unavailable: ${error.message}`);return false;}
  };
  const bindMaterialMask=(program,fallback,view,unitM,origin)=>{
    let available=false;
    try {
      available=categoryMask()!==0&&directionalCurrent();
      if(available) {
        const bounds=directionalBounds(metadata);
        gl.uniformMatrix4fv(gl.getUniformLocation(program,'uFieldFromView'),false,
          fieldFromView(view,unitM,origin,bounds.bounds_min_m,bounds.bounds_max_m));
      }
    }catch(error){available=false;drop();say(`Field unavailable: ${error.message}`);}
    gl.uniform1i(gl.getUniformLocation(program,'uFieldRestrictToBounds'),available?1:0);
    const active=gl.getParameter(gl.ACTIVE_TEXTURE);
    try {gl.activeTexture(gl.TEXTURE0+13);gl.bindSampler(13,null);
      gl.bindTexture(gl.TEXTURE_2D,available?directionalTextures[2]:fallback);
      gl.uniform1i(gl.getUniformLocation(program,'uFieldMaterialMask'),13);
    }finally{gl.activeTexture(active);}
    return available;
  };
  const change=()=>{invalidate();say(requested()?'Opted in for the next RGB still or Live job.':'Ray field off. Ordinary Phong lighting is unchanged.');onChange();};
  const describeDirectional=()=>{
    const bins=weightedField(metadata)?metadata.directional.bin_count:6;
    const mode=spatialLinear()?'energy-weighted spatial interpolation':metadata.schema==='studio.forward-field.v2'?'nearest sampling (legacy v2)':
      spatialControl?.checked!==false&&!floatLinear?'nearest sampling (float-linear filtering unavailable on this GPU)':'nearest sampling';
    const estimator=['studio.forward-field.v4','studio.forward-field.v5','studio.forward-field.v6'].includes(metadata.schema)?'root-sweep normalized':'legacy unnormalized';
    const coverageText=coverage?.nonzeroCells?`${coverage.nonzeroCells.toLocaleString()} of ${coverage.totalCells.toLocaleString()} cells contain recorded directional energy (${(100*coverage.nonzeroCells/coverage.totalCells).toFixed(2)}%).`:
      'No recorded directional energy in this region yet.';
    const populations=metadata.directional.source_populations;
    const history=populations?` ${populations.count.toLocaleString()} source populations integrated; ${metadata.directional.root_sampling.recorded_root_jobs.toLocaleString()} recorded root jobs.`:'';
    const transport=metadata.surface_transport==='studio-pbr-mixture-v1'?'corrected opaque PBR transport':'legacy surface transport';
    say(`Directional ray field received: ${metadata.directional.dimensions.join(' × ')} grid, ${bins} directions, ${estimator}, ${mode}, ${transport}. ${coverageText}${history}`);
  };
  const spatialChange=()=>{
    if(!directionalTextures)return;
    const active=gl.getParameter(gl.ACTIVE_TEXTURE),previous=gl.getParameter(gl.TEXTURE_BINDING_3D);
    try {for(const texture of directionalTextures.slice(0,2)) {
      gl.bindTexture(gl.TEXTURE_3D,texture);
      for(const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_3D,p,spatialLinear()?gl.LINEAR:gl.NEAREST);
    }}finally{gl.bindTexture(gl.TEXTURE_3D,previous);gl.activeTexture(active);}
    describeDirectional();
  };
  toggle?.addEventListener('change',change);
  transportControl?.addEventListener('change',change);
  stencilControl?.addEventListener('change',change);spatialControl?.addEventListener('change',spatialChange);
  gridControl?.addEventListener('change',change);
  const regionChange=()=>{syncRecordingControls();change();};
  regionControl?.addEventListener('change',regionChange);cellControl?.addEventListener('change',change);marginControl?.addEventListener('change',change);
  for(const {control} of categories)control?.addEventListener('change',categoryChange);
  syncRecordingControls();describeDivisions();categoryChange();
  return {requested,capture,connect,observe,bind,bindSecondary,bindMaterialMask,invalidate,
    get categoryMask(){return categoryMask();},get hasReceivers(){return categoryMask()!==0;},
    get coverage(){return coverage&&{...coverage};},
    get hasDirectional(){try{return directionalCurrent();}catch(error){drop();say(`Field unavailable: ${error.message}`);return false;}},
    revise(digest,revision){if(source?.digest!==digest||source.revision===revision)return;
      const context=source.context;invalidate('Camera updated. Waiting for matching field data.');connect(digest,capture(context),revision);},
    stop(digest){if(source?.digest===digest)invalidate('Ray field stopped.');},
    get metadata(){return metadata&&structuredClone(metadata);},
    dispose(){disposed=true;invalidate();toggle?.removeEventListener('change',change);stencilControl?.removeEventListener('change',change);gridControl?.removeEventListener('change',change);spatialControl?.removeEventListener('change',spatialChange);
      transportControl?.removeEventListener('change',change);
      regionControl?.removeEventListener('change',regionChange);cellControl?.removeEventListener('change',change);marginControl?.removeEventListener('change',change);
      for(const {control} of categories)control?.removeEventListener('change',categoryChange);}};
}

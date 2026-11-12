import assert from 'node:assert/strict';
import {webcrypto} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {prepareOpticalSnapshot,OPTICAL_PROVENANCE} from '../examples/meowdoku/web/marble_optical_authoring.js';
import {createStudioLightField,validateLightField,decodeLightField,fieldFromView,mapFieldMaterials,fieldStencilDirections,fieldAtlasLayout,fieldCellDimensions,fieldTransferTimeoutMs,decodeFieldArrivals,decodeEvaluatedResponses,validateSceneResponseReceipt} from '../examples/meowdoku/web/marble_light_field.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

const A='a'.repeat(64),B='b'.repeat(64),I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
assert.equal(fieldTransferTimeoutMs(24),30000);
assert(fieldTransferTimeoutMs(89279056)>180000,'actual dense bundle gets time to cross public connection');
assert.equal(fieldTransferTimeoutMs(1e9),300000,'transfer budget remains bounded');
const meta=(sequence=1,revision=0)=>({schema:'studio.forward-field.v1',digest:A,live_revision:revision,sequence,
 name:`light-field-${String(sequence).padStart(32,'0')}.bin`,format:'rgb-f32le',layout:'zyx-interleaved-rgb',
 color_space:'linear-srgb',dimensions:[2,1,1],bounds_min_m:[1,2,3],bounds_max_m:[3,4,5],byte_length:24,ray_jobs:128});
const bytes=()=>new Float32Array([1,-.25,3,4,5,6]).buffer;
// Exercise the actual host callback with its canonical projector, not a second
// implementation of the optical rule. Tiny explicit geometry keeps the test CPU-only.
{
 const source=readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
 const callback=source.slice(source.indexOf('let fieldMaterialProjection=null;'),source.indexOf('function studioBuilderOptions('));
 const row=id=>[null,null,null,id],geometry={interior:Array.from({length:4},()=>row(1)),glass:Array.from({length:4},()=>row(0)),
  ribbons:[{start:0,count:4,material:1}]};
 const data=new Float32Array(48);for(const id of [0,1]){data[id*16+6]=1.5;data[id*16+7]=1;}
 data[32]=.3;const original=data.slice();let captures=0,projections=0;
 const recipe={version:1,mode:'glass_rod',provenance:OPTICAL_PROVENANCE,materials:[0,1].map(material=>({material,ior:1.5,
  wavelengths_nm:[450,550,650],transmittance:[.8,.8,.8],reference_thickness_mm:1,concentration:1}))};
 const make=new Function('tables','lastGeometry','computeFixtureRoles','writeStudioFixtureRows','writeShopLightEmitterRows',
  'FIXTURE_MAT_BASE','STUDIO_FIXED_MATERIAL_ROWS','shopLightState','shopEmissionProfiles','readSubjectDesign','opticalControls',
  'prepareOpticalSnapshot','$','ids',`let objectKey='a',studioSceneRevision=0;${callback};return {read:fieldMaterialRows,key:v=>objectKey=v,edit:()=>studioSceneRevision++};`);
 const host=make({pbr:{data,table:{floats_per_record:16}}},geometry,()=>({roleColor:[],roleStrength:[]}),()=>{},()=>{},2,0,()=>({}),{},
  ()=>({'base-shape':'marble',coverage:0,'band-count':0,shell:'glass'}),{capture:()=>{captures++;return structuredClone(recipe);}},
  (...args)=>{projections++;prepareOpticalSnapshot(...args);},()=>({value:'10'}),{glass:0});
 const first=host.read();assert.deepEqual(data,original,'projection cannot change displayed table');
 assert.equal(first.lookupData[5],1);assert.equal(first.lookupData[7],0);assert.equal(first.data[7],1);
 assert.equal(first.lookupData[32],data[32],'fixture/material rows outside optical membership stay unchanged');
 for(let i=0;i<10;i++)assert.equal(host.read().lookupData,first.lookupData);
 assert.equal(captures,1);assert.equal(projections,1,'unchanged frames never inspect geometry again');
 data[7]=.5;const edited=host.read();assert.deepEqual(edited.lookupData,first.lookupData);assert.notDeepEqual(edited.data,first.data);
 host.key('b');host.read();host.edit();host.read();assert.equal(projections,4,'raw rows, object key and edit revision each invalidate cache');
}
assert.equal(validateLightField(meta(),A,0).byte_length,24);
assert.deepEqual(fieldCellDimensions(meta()),[1,2,2]);assert.equal(fieldCellDimensions(null),null);
{
 const html=readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
 for(const name of ['subject','stage','room']) {
  const input=html.match(new RegExp(`<input[^>]*id="phong-field-${name}"[^>]*>`))?.[0];
  assert(input&&!/\bchecked\b/.test(input),'each category is explicitly opt-in by default');
 }
}
for(const edit of [{digest:B},{name:'../config.json'},{dimensions:[1,1,1]},{byte_length:NaN},
 {bounds_min_m:[NaN,0,0]},{bounds_max_m:[0,0,0]},{color_space:'srgb'},{live_revision:2},{sequence:-1}])
 assert.throws(()=>validateLightField({...meta(),...edit},A,0));
assert.deepEqual(Array.from(decodeLightField(bytes(),meta())),[1,-.25,3,4,5,6]);
assert.throws(()=>decodeLightField(new ArrayBuffer(20),meta()));
assert.throws(()=>decodeLightField(new Float32Array([1,NaN,3,4,5,6]).buffer,meta()));
const transform=(m,p)=>[0,1,2].map(r=>m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]);
const view=[0,1,0,0,-1,0,0,0,0,0,1,0,4,5,-6,1],world=[.5,.25,.75];
const mapping=fieldFromView(view,2,[1,2,3],[1,2,3],[3,4,5]);
assert.deepEqual(transform(mapping,transform(view,world)),world,'view mapping includes translation/rotation/units/bounds');
assert.throws(()=>fieldFromView(new Array(16).fill(0),1,[0,0,0],[0,0,0],[1,1,1]));
assert.throws(()=>fieldFromView(I,1e200,[0,0,0],[0,0,0],[1,1,1]),/float32/);

function harness(linear=false,categoryBits=1){
 const uploads=[],deleted=[],uniforms={},bindings=new Map();let active=108,unpack=4,id=0,error=0,key='scene-1',calls=0,hook=null;
 const gl={ACTIVE_TEXTURE:1,TEXTURE_BINDING_3D:2,UNPACK_ALIGNMENT:3,MAX_3D_TEXTURE_SIZE:4,TEXTURE8:108,
  TEXTURE_3D:5,TEXTURE_MIN_FILTER:6,TEXTURE_MAG_FILTER:7,TEXTURE_WRAP_S:8,TEXTURE_WRAP_T:9,TEXTURE_WRAP_R:10,
  NEAREST:11,CLAMP_TO_EDGE:12,RGB32F:13,RGB:14,FLOAT:15,NO_ERROR:0,TEXTURE0:100,
  TEXTURE_2D:16,TEXTURE_BINDING_2D:17,MAX_TEXTURE_SIZE:18,MAX_TEXTURE_IMAGE_UNITS:19,R32UI:20,RED_INTEGER:21,UNSIGNED_INT:22,
  RGBA32F:23,RGBA:24,LINEAR:25,getExtension:()=>linear?{}:null,
  getParameter(p){return p===1?active:p===2?bindings.get(active)??null:p===17?bindings.get(`2d:${active}`)??null:p===3?unpack:256;},
  activeTexture(v){active=v;},bindTexture(t,v){assert(!deleted.includes(v),'never rebind a deleted texture');bindings.set(t===16?`2d:${active}`:active,v);},
  pixelStorei(p,v){unpack=v;},createTexture(){return {id:++id};},deleteTexture(t){deleted.push(t);for(const [k,v]of bindings)if(v===t)bindings.set(k,null);},
  bindSampler(){},texParameteri(){},texImage3D(...args){uploads.push(args);},texImage2D(...args){uploads.push(args);},getError(){const v=error;error=0;return v;},
  getUniformLocation(p,n){return n;},uniform1f(n,v){uniforms[n]=v;},uniform1i(n,v){uniforms[n]=v;},uniformMatrix3fv(n,t,v){uniforms[n]=Array.from(v);},uniformMatrix4fv(n,t,v){uniforms[n]=Array.from(v);}};
 const toggle=new EventTarget();toggle.checked=false;const output={textContent:''};
 const responses=new EventTarget();responses.checked=false;
 const transport=new EventTarget();transport.checked=false;
 const stencil=new EventTarget();stencil.value='12';const spatial=new EventTarget();spatial.checked=true;
 const grid=new EventTarget();grid.value='64';
 const region=new EventTarget();region.value='scene';const cell=new EventTarget();cell.value='0.5';const margin=new EventTarget();margin.value='1';
 const categoryControls=Object.fromEntries(['subject','stage','room'].map((name,i)=>{const control=new EventTarget();control.checked=!!(categoryBits&(1<<i));return [name,control];}));
 const categoryOutput={textContent:''},divisionOutput={textContent:''};let changes=0;
 const doc={getElementById:id=>({'phong-light-field':toggle,'phong-light-field-status':output,
  'phong-field-responses':responses,
  'ray-surface-transport':transport,
  'phong-field-stencil':stencil,'phong-field-spatial-linear':spatial,'phong-field-grid-size':grid,
  'phong-field-region':region,'phong-field-cell-mm':cell,'phong-field-margin-mm':margin,
  'phong-field-category-status':categoryOutput,'phong-field-division-status':divisionOutput,
  ...Object.fromEntries(Object.entries(categoryControls).map(([name,control])=>[`phong-field-${name}`,control]))}[id]??null)};
 const fetcher=async(...args)=>{calls++;if(hook)return hook(...args);return {ok:true,status:200,headers:{get:()=> '24'},arrayBuffer:async()=>bytes()};};
 const fallbackTexture={id:'original-zero-volume'};bindings.set(gl.TEXTURE8,fallbackTexture);
 const materials={data:new Float32Array(32),stride:16};materials.data[16]=.5;
 const field=createStudioLightField({gl,doc,fallbackTexture,onChange:()=>changes++,getSceneMaterials:()=>materials,getSceneKey:()=>{if(key==='bad')throw Error('incompatible');return key;},fetcher});
 return {field,toggle,responses,transport,output,stencil,spatial,grid,region,cell,margin,categoryControls,categoryOutput,divisionOutput,gl,uploads,deleted,uniforms,bindings,fallbackTexture,materials,
  get changes(){return changes;},get calls(){return calls;},set key(v){key=v;},
  set hook(v){hook=v;},set error(v){error=v;},on(){toggle.checked=true;toggle.dispatchEvent(new Event('change'));},
  off(){toggle.checked=false;toggle.dispatchEvent(new Event('change'));},status(seq=1,rev=0){return {state:'live',light_field:meta(seq,rev)};}};
}
{
 const h=harness();assert.equal(h.field.capture(),null);await h.field.observe(h.status(),A);assert.equal(h.calls,0);
 h.on();const ticket=h.field.capture();h.field.connect(A,ticket);await h.field.observe(h.status(),A);
 assert.equal(h.uploads.length,1);assert.equal(h.field.bind({},I,1,[0,0,0]),true);assert.equal(h.uniforms.uFieldGain,1);
 assert.deepEqual(Array.from(h.uploads[0].at(-1)),[1,-.25,3,4,5,6]);
 await h.field.observe(h.status(),A);assert.equal(h.calls,1,'same snapshot is not reuploaded or accumulated');
 await h.field.observe(h.status(2),A);assert.equal(h.uploads.length,2);assert.equal(h.deleted.length,1,'snapshot replacement releases old texture');
 h.key='camera-2';assert.equal(h.field.bind({},I,1,[0,0,0]),false);assert.equal(h.uniforms.uFieldGain,0);
 h.field.revise(A,1);await h.field.observe(h.status(3,1),A);assert.equal(h.field.bind({},I,1,[0,0,0]),true,'camera revision can recover without restarting rays');
 h.off();assert.equal(h.field.bind({},I,1,[0,0,0]),false);assert.equal(h.field.metadata,null);
 assert.equal(h.bindings.get(h.gl.TEXTURE8),h.fallbackTexture,"off restores the complete original volume sampler");
 h.field.dispose();
}
{
 const h=harness();h.on();h.field.connect(A,h.field.capture());let resolve;
 h.hook=()=>new Promise(r=>resolve=r);const pending=h.field.observe(h.status(),A);
 h.off();resolve({ok:true,status:200,headers:{get:()=> '24'},arrayBuffer:async()=>bytes()});await pending;
 assert.equal(h.uploads.length,0,'an off toggle defeats an in-flight response');
 h.on();const stale=h.field.capture();h.field.invalidate();h.field.connect(A,stale);await h.field.observe(h.status(),A);
 assert.equal(h.calls,1,'a scene edit defeats a pending upload request');h.field.dispose();
}
{
 const h=harness();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:false,status:404});await h.field.observe(h.status(),A);assert.equal(h.uploads.length,0);
 h.hook=null;await h.field.observe(h.status(2),A);assert.equal(h.uploads.length,1,'superseded artifact recovers on next status');
 h.error=1285;await h.field.observe(h.status(3),A);assert.equal(h.field.metadata,null);assert.match(h.output.textContent,/upload failed/);
 h.key='bad';assert.equal(h.field.capture(),null);assert.match(h.output.textContent,/incompatible/);h.field.dispose();
}

// Compact two-voxel v2 fixture exercises the exact producer interleaving and
// intentionally reversed material IDs. No web/GPU/server dependency.
async function bundle(sequence=1) {
 const directional={offset_bytes:24,byte_length:288,dimensions:[2,1,1],bins:['+X','-X','+Y','-Y','+Z','-Z'],
  channels:['R','G','B','mean_propagation_x','mean_propagation_y','mean_propagation_z'],format:'rgb-direction-f32le',layout:'zyx-bin-channel',
  estimator:'track-length-throughput-v1',direction_space:'world'};
 const triangles={offset_bytes:312,byte_length:32,count:2,stride_bytes:16,format:'u32le',channels:['triangle_id','material_id','flags','class_mask'],
  class_bits:{volumetric:1,reflectance:2,glass:4},ordering:'original packed scene triangles, before BVH reorder'};
 const sources={offset_bytes:344,byte_length:48,count:1,stride_bytes:48,format:'f32le',layout:'baked-source-records',
  channels:['position_x_m','position_y_m','position_z_m','amplitude_weight','direction_x','direction_y','direction_z','kind','frequency_hz','phase_radians','energy','packet_aux']};
 const materials={offset_bytes:392,byte_length:128,count:2,stride_bytes:64,format:'f32le',layout:'native-pbr16'};
 const source_pbr16={offset_bytes:520,byte_length:128,count:2,stride_bytes:64,format:'f32le',layout:'authored-pbr16',material_ids:[1,0]};
 const payload=new ArrayBuffer(648),dv=new DataView(payload);new Float32Array(payload,0,6).set(new Float32Array(bytes()));
 for(let v=0;v<2;v++)for(let b=0;b<6;b++) {
  const offset=24+(v*6+b)*24;dv.setFloat32(offset,v*10+b,true);dv.setFloat32(offset+12,1,true);
 }
 new Uint32Array(payload,312,8).set([0,0,64,5,1,1,0,3]);dv.setFloat32(392,.75,true);dv.setFloat32(456,.25,true);
 dv.setFloat32(584,.5,true); // Authored bytes differ from actual engine-native rows.
 const sha256=Buffer.from(await webcrypto.subtle.digest('SHA-256',payload)).toString('hex');
 return {payload,metadata:{...meta(sequence),schema:'studio.forward-field.v2',scene_sha256:B,sha256,byte_length:648,directional,triangles,sources,materials,source_pbr16}};
}
async function bundleV3(bins=12) {
 const old=await bundle(),metadata=structuredClone(old.metadata),extra=2*(bins-6)*24;
 metadata.schema='studio.forward-field.v3';metadata.byte_length+=extra;
 Object.assign(metadata.directional,{bin_count:bins,stencil:bins===12?'icosahedron12-v1':'axes6-v1',
  bin_directions:fieldStencilDirections(bins),weight_from_linear_rgb:[1,1,1],
  bins:bins===6?['+X','-X','+Y','-Y','+Z','-Z']:Array.from({length:12},(_,i)=>`I${i}`),byte_length:2*bins*24});
 for(const name of ['triangles','sources','materials','source_pbr16'])metadata[name].offset_bytes+=extra;
 const payload=new ArrayBuffer(metadata.byte_length);new Uint8Array(payload).set(new Uint8Array(old.payload,0,24));
 new Uint8Array(payload).set(new Uint8Array(old.payload,312),312+extra);
 const dv=new DataView(payload);
 for(let voxel=0;voxel<2;voxel++)for(let bin=0;bin<bins;bin++) {
  const offset=24+(voxel*bins+bin)*24;dv.setFloat32(offset,1+bin+voxel*20,true);dv.setFloat32(offset+12,.5,true);
 }
 dv.setFloat32(24,-1,true);dv.setFloat32(28,3,true);
 metadata.sha256=Buffer.from(await webcrypto.subtle.digest('SHA-256',payload)).toString('hex');
 return {payload,metadata};
}
{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert.equal(h.field.coverage.nonzeroCells,2);
 const empty=b.payload.slice(0);new Uint8Array(empty,b.metadata.directional.offset_bytes,b.metadata.directional.byte_length).fill(0);
 const next={...b.metadata,sequence:2,sha256:Buffer.from(await webcrypto.subtle.digest('SHA-256',empty)).toString('hex')};
 let release,started;const ready=new Promise(r=>started=r);
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:()=>new Promise(r=>{release=r;started();})});
 const update=h.field.observe({state:'live',light_field:next},A);await ready;
 assert(h.field.hasDirectional);assert.equal(h.field.metadata.sequence,1);assert.equal(h.field.coverage.nonzeroCells,2,'last good atlas stays while next bundle downloads');
 release(empty);await update;assert.equal(h.field.coverage.nonzeroCells,0);assert.match(h.output.textContent,/No recorded directional energy/);
 h.field.dispose();
}
async function bundleV4(size,bins) {
 const old=await bundleV3(bins),metadata=structuredClone(old.metadata);
 const length=size**3*bins*24,extra=length-metadata.directional.byte_length;
 metadata.schema='studio.forward-field.v4';metadata.byte_length+=extra;
 Object.assign(metadata.directional,{dimensions:[size,size,size],byte_length:length,estimator:'track-length-root-sweep-v1',
  root_sampling:{version:'per-complete-pump-inverse-replay-v1',source_root_count:metadata.sources.count,completed_root_sweeps:1,
   recorded_root_jobs:metadata.sources.count,normalization_divisor:1,excluded_partial_root_jobs:0,incomplete_dispatch:false}});
 for(const name of ['triangles','sources','materials','source_pbr16'])metadata[name].offset_bytes+=extra;
 const payload=new ArrayBuffer(metadata.byte_length);new Uint8Array(payload).set(new Uint8Array(old.payload,0,24));
 new Uint8Array(payload).set(new Uint8Array(old.payload,old.metadata.triangles.offset_bytes),metadata.triangles.offset_bytes);
 const view=new DataView(payload);
 for(const voxel of [0,size-1,size**2,size**3-1])for(let bin=0;bin<bins;bin++) {
  const offset=24+(voxel*bins+bin)*24;view.setFloat32(offset,bin+1,true);view.setFloat32(offset+12,.5,true);
 }
 metadata.sha256=Buffer.from(await webcrypto.subtle.digest('SHA-256',payload)).toString('hex');
 return {payload,metadata};
}
async function bundleV5() {
 const b=await bundleV4(32,12),d=b.metadata.directional,pitch=.0005,margin=.001;
 b.metadata.schema='studio.forward-field.v5';b.metadata.geometry_sha256='c'.repeat(64);
 const lo=[-.0041,-.0031,.1751],hi=[.0042,.0032,.1842],lower=lo.map(v=>Math.floor((v-margin)/pitch)),upper=hi.map(v=>Math.ceil((v+margin)/pitch));
 const start=lower.map((v,i)=>v-Math.floor((32-(upper[i]-v))/2));
 Object.assign(d,{cell_size_m:pitch,margin_m:margin,bounds_min_m:start.map(v=>v*pitch),bounds_max_m:start.map(v=>(v+32)*pitch),
  region:{kind:'subject',subject_geometry_sha256:b.metadata.geometry_sha256,world_geometry_sha256:'d'.repeat(64),
   actual_bounds_min_m:lo,actual_bounds_max_m:hi,effective_margin_m:margin,
   padding_min_m:lo.map((v,i)=>v-start[i]*pitch),padding_max_m:hi.map((v,i)=>(start[i]+32)*pitch-v)}});
 return b;
}
async function bundleV6(index=0,previousHash='0'.repeat(64)) {
 const b=await bundleV5(),d=b.metadata.directional,section=b.metadata.sources;
 b.metadata.schema='studio.forward-field.v6';b.metadata.sequence=index+1;
 new DataView(b.payload).setFloat32(section.offset_bytes,index,true);
 const sourceHash=Buffer.from(await webcrypto.subtle.digest('SHA-256',new Uint8Array(b.payload,section.offset_bytes,section.byte_length))).toString('hex');
 const chain=new Uint8Array(72);chain.set(Buffer.from(previousHash,'hex'));new DataView(chain.buffer).setBigUint64(32,BigInt(index),true);chain.set(Buffer.from(sourceHash,'hex'),40);
 Object.assign(d.root_sampling,{completed_root_sweeps:index+1,normalization_divisor:index+1,recorded_root_jobs:index+1});
 Object.assign(d,{quantization_divisor:16777216,quantization_version:'fixed24-track-length-v1',coverage:{nonzero_cells:4,total_cells:32**3,meaning:'nonzero accumulated energy cells'},
  source_populations:{version:'native-angular-refresh-mean-v1',generator:'native-emissive-triangle-roots-v1',ray_budget:4096,base_seed:42,
   first_index:0,last_index:index,count:index+1,latest_seed:42+index,latest_sources_sha256:sourceHash,
   rolling_sha256:Buffer.from(await webcrypto.subtle.digest('SHA-256',chain)).toString('hex')}});
 b.metadata.sha256=Buffer.from(await webcrypto.subtle.digest('SHA-256',b.payload)).toString('hex');return b;
}
{
 const first=await bundleV6(),second=await bundleV6(1,first.metadata.directional.source_populations.rolling_sha256);
 validateLightField(first.metadata,A,0);validateLightField(second.metadata,A,0);
 const internal={version:'canonical-beer-segment-integral-v1',scope:'actual valid interior and exterior traced segments',entry_packet:true,
  integration:'per-recording-cell integral of canonical Beer transmittance',optical_authority:'installed medium/region/dye helper',
  invalid_ownership:'excluded',external_extinction:'unsupported; request rejected'};
 validateLightField({...first.metadata,directional:{...first.metadata.directional,internal_media:internal}},A,0);
 for(const key of Object.keys(internal))assert.throws(()=>validateLightField({...first.metadata,
  directional:{...first.metadata.directional,internal_media:{...internal,[key]:null}}},A,0),/internal medium/);
 for(const edit of [{count:2},{latest_seed:99},{ray_budget:0},{latest_sources_sha256:'bad'},{rolling_sha256:'bad'}])
  assert.throws(()=>validateLightField({...first.metadata,directional:{...first.metadata.directional,
   source_populations:{...first.metadata.directional.source_populations,...edit}}},A,0),/population/);
 for(const edit of [{quantization_divisor:65535},{quantization_version:'unknown'}])
  assert.throws(()=>validateLightField({...first.metadata,directional:{...first.metadata.directional,...edit}},A,0));
 const h=harness(true);h.region.value='subject';h.on();h.field.connect(A,h.field.capture());let active=first;
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>active.payload});
 await h.field.observe({state:'live',light_field:first.metadata},A);assert(h.field.hasDirectional,h.output.textContent);assert.equal(h.field.coverage.nonzeroCells,4);
 active=second;await h.field.observe({state:'live',light_field:second.metadata},A);assert(h.field.hasDirectional,h.output.textContent);
 assert.match(h.output.textContent,/2 source populations integrated; 2 recorded root jobs/);
 active=await bundleV6(2,second.metadata.directional.source_populations.rolling_sha256);
 active.metadata.directional.source_populations.rolling_sha256=A;
 await h.field.observe({state:'live',light_field:active.metadata},A);assert.equal(h.field.hasDirectional,false);assert.match(h.output.textContent,/history hash mismatch/);
 h.field.dispose();
 for(const failure of ['source','coverage']) {
  const h=harness(),bad=await bundleV6();h.region.value='subject';h.on();h.field.connect(A,h.field.capture());
  if(failure==='source')bad.metadata.directional.source_populations.latest_sources_sha256=A;
  else bad.metadata.directional.coverage.nonzero_cells=0;
  h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>bad.payload});
  await h.field.observe({state:'live',light_field:bad.metadata},A);assert.equal(h.uploads.length,0);h.field.dispose();
 }
}
{
 const b=await bundleV5();validateLightField(b.metadata,A,0);
 for(const edit of [{cell_size_m:.001},{margin_m:-1},{bounds_min_m:[0,0,0]},
  {region:{...b.metadata.directional.region,subject_geometry_sha256:A}},
  {region:{...b.metadata.directional.region,world_geometry_sha256:'bad'}},
  {region:{...b.metadata.directional.region,effective_margin_m:0}},
  {region:{...b.metadata.directional.region,padding_min_m:[0,0,0]}},
  {region:{...b.metadata.directional.region,actual_bounds_max_m:[1,1,1]}}])
  assert.throws(()=>validateLightField({...b.metadata,directional:{...b.metadata.directional,...edit}},A,0));
 const h=harness(true);assert.equal(h.cell.disabled,true);h.region.value='subject';h.region.dispatchEvent(new Event('change'));
 assert.equal(h.grid.disabled,true);assert.equal(h.cell.disabled,false);assert.equal(h.margin.disabled,false);h.on();
 const ticket=h.field.capture();assert.equal(ticket.region,'subject');assert.equal(ticket.cellSizeM,.0005);assert.equal(ticket.marginM,.001);
 h.field.connect(A,ticket);h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional,h.output.textContent);
 assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false,'legacy moments are not evaluated material responses');
 assert.equal(h.field.bindMaterialMask({}, {},I,1,[0,0,0]),false,'material flags are never sampling availability');
 assert.equal(h.field.bind({},I,1,[0,0,0]),true);
 assert.deepEqual(h.uniforms.uFieldFromView,Array.from(fieldFromView(I,1,[0,0,0],b.metadata.bounds_min_m,b.metadata.bounds_max_m)),
  'legacy scalar mapping retains global bounds');
 assert(fieldCellDimensions(b.metadata).every(v=>Math.abs(v-.0005)<1e-12));
 assert.match(h.divisionOutput.textContent,/0.5 × 0.5 × 0.5 mm/);assert.match(h.divisionOutput.textContent,/Requested margin: 1 mm/);
 assert.match(h.divisionOutput.textContent,/Padded recording bounds/);
 h.margin.value='2';h.margin.dispatchEvent(new Event('change'));assert.equal(h.field.hasDirectional,false);h.field.connect(A,ticket);assert.equal(h.field.metadata,null);
 assert.equal(h.field.bindMaterialMask({},{}),false);
 h.cell.value='.01';assert.equal(h.field.capture(),null,'unsupported pitch is refused');h.cell.value='.5';
 h.region.value='scene';h.region.dispatchEvent(new Event('change'));assert.equal(h.grid.disabled,false);assert.equal(h.cell.disabled,true);h.field.dispose();
}
for(const [size,bins,columns,slabs] of [[32,6,1,6],[32,12,2,6],[64,6,2,3],[64,12,3,4]]) {
 assert.deepEqual(fieldAtlasLayout([size,size,size],bins),{columns,slabs,dimensions:[size*columns,size,size*slabs]});
 const b=await bundleV4(size,bins);validateLightField(b.metadata,A,0);
 const h=harness(true);h.stencil.value=String(bins);h.grid.value=String(size);h.on();const ticket=h.field.capture();
 assert.equal(ticket.gridSize,size);h.field.connect(A,ticket);
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional,h.output.textContent);
 assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false);assert.equal(h.uniforms.uResponseCount,0);assert.match(h.output.textContent,/root-sweep normalized/);
 assert.deepEqual(h.uploads[1].slice(3,6),[size*columns,size,size*slabs]);
 for(const voxel of [0,size-1,size**2,size**3-1])for(let bin=0;bin<bins;bin++) {
  const x=voxel%size,y=Math.floor(voxel/size)%size,z=Math.floor(voxel/(size*size));
  const index=(((Math.floor(bin/columns)*size+z)*size+y)*(size*columns)+(bin%columns)*size+x);
  assert.equal(h.uploads[1].at(-1)[index*3],bin+1);assert.equal(h.uploads[2].at(-1)[index*4],.5*(bin+1));
  assert.equal(h.uploads[2].at(-1)[index*4+3],bin+1);
 }
 for(const edit of [{dimensions:[size,size,16]},{estimator:'track-length-throughput-v1'}])
  assert.throws(()=>validateLightField({...b.metadata,directional:{...b.metadata.directional,...edit}},A,0));
 for(const edit of [{version:'unknown'},{source_root_count:0},{source_root_count:2},{completed_root_sweeps:0},
  {normalization_divisor:2},{recorded_root_jobs:0},{excluded_partial_root_jobs:-1},{incomplete_dispatch:true}])
  assert.throws(()=>validateLightField({...b.metadata,directional:{...b.metadata.directional,
   root_sampling:{...b.metadata.directional.root_sampling,...edit}}},A,0),/normalization/);
 h.grid.value=String(size===32?64:32);h.grid.dispatchEvent(new Event('change'));assert.equal(h.field.hasDirectional,false);
 h.field.connect(A,ticket);assert.equal(h.field.metadata,null,'grid changes defeat old capture');h.field.dispose();
}
{
 const h=harness(false,0),b=await bundle();assert.equal(h.field.categoryMask,0);assert.equal(h.field.hasReceivers,false);
 h.on();const ticket=h.field.capture();h.field.connect(A,ticket);
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional,'data readiness independent of category application');
 assert.equal(h.field.bind({},I,1,[0,0,0]),false);assert.equal(h.uniforms.uFieldGain,0,'no accidental legacy glow while all off');
 assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false);assert.equal(h.field.bindMaterialMask({},{}),false);
 assert.match(h.divisionOutput.textContent,/1000 × 2000 × 2000 mm/);assert.match(h.divisionOutput.textContent,/8×/);
 const changes=h.changes,calls=h.calls;
 for(const [name,bit]of [['subject',1],['stage',2],['room',4]]) {
  for(const control of Object.values(h.categoryControls))control.checked=false;
  const control=h.categoryControls[name];control.checked=true;control.dispatchEvent(new Event('change'));
  assert.equal(h.field.categoryMask,bit);assert.equal(h.field.hasReceivers,true);assert(h.field.hasDirectional);
  assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false);assert.match(h.categoryOutput.textContent,new RegExp(name));
 }
 for(const control of Object.values(h.categoryControls))control.checked=true;
 assert.equal(h.field.categoryMask,7);assert.equal(h.changes,changes,'application-only category changes do not request ray restart');
 assert.equal(h.calls,calls);assert.equal(h.deleted.length,0);assert.equal(h.field.capture().epoch,ticket.epoch,'category changes preserve capture identity');
 for(const control of Object.values(h.categoryControls)){control.checked=false;control.dispatchEvent(new Event('change'));}
 assert.equal(h.field.bind({},I,1,[0,0,0]),false);assert(h.field.hasDirectional);h.field.dispose();
}
for(const bins of [6,12]) {
 const b=await bundleV3(bins),decoded=decodeLightField(b.payload,validateLightField(b.metadata,A,0));
 // Retain the former allocation-heavy math as a CPU reference for the hot
 // decoder's scalar/direct-write implementation, including f32 rounding.
 const referenceRGB=new Float32Array(decoded.radiance.length),referenceDirection=new Float32Array(decoded.direction.length);
 const referenceView=new DataView(b.payload);
 for(let voxel=0;voxel<2;voxel++)for(let bin=0;bin<bins;bin++) {
  const sample=Array.from({length:6},(_,c)=>referenceView.getFloat32(24+(voxel*bins+bin)*24+c*4,true));
  const weight=Math.max(0,sample.slice(0,3).reduce((sum,v,i)=>sum+v*b.metadata.directional.weight_from_linear_rgb[i],0));
  const dst=bin*2+voxel;
  referenceRGB.set(sample.slice(0,3),dst*3);
  referenceDirection.set([...sample.slice(3).map(v=>v*weight),weight],dst*4);
 }
 assert.deepEqual(decoded.radiance,referenceRGB);assert.deepEqual(decoded.direction,referenceDirection);
 assert.deepEqual(decoded.atlasDimensions,[2,1,bins]);
 assert.deepEqual(Array.from(decoded.direction.slice(0,4)),[1,0,0,2],'signed RGB recovers measured weight before weighting direction');
 assert.equal(decoded.radiance[0],-1,'signed RGB survives');
 assert.equal(decoded.radiance[6],2,'adjacent tile or slab starts with next bin');
 assert.equal(decoded.direction[4],10.5,'next spatial voxel retains actual weighted moment');
 for(const edit of [{bin_count:7},{stencil:'invented'},{bin_directions:[]},{weight_from_linear_rgb:[NaN,1,1]}])
  assert.throws(()=>validateLightField({...b.metadata,directional:{...b.metadata.directional,...edit}},A,0));
 const wrong=b.payload.slice(0);new DataView(wrong).setFloat32(24,-100,true);assert.throws(()=>decodeLightField(wrong,b.metadata),/weight/);
 for(const linear of [false,true]) {
  const h=harness(linear);h.stencil.value=String(bins);h.on();const ticket=h.field.capture();assert.equal(ticket.stencil,bins);h.field.connect(A,ticket);
  h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
  await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional,h.output.textContent);
  assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false);assert.equal(h.uniforms.uResponseCount,0);
  assert.equal(h.uploads[2][2],h.gl.RGBA32F);assert.deepEqual(h.uploads[2].slice(3,6),decoded.atlasDimensions);
  if(!linear)assert.match(h.output.textContent,/unavailable on this GPU/);
  h.spatial.checked=false;h.spatial.dispatchEvent(new Event('change'));assert(h.field.hasDirectional);
  h.field.bindSecondary({},I,1,[0,0,0]);assert.equal(h.uniforms.uResponseCount,0);
  h.stencil.value=String(bins===6?12:6);h.stencil.dispatchEvent(new Event('change'));assert.equal(h.field.hasDirectional,false);
  h.field.connect(A,ticket);assert.equal(h.field.metadata,null,'stencil changes defeat old capture');h.field.dispose();
 }
}
{
 const {payload,metadata}=await bundle(),decoded=decodeLightField(payload,validateLightField(metadata,A,0));
 assert.deepEqual(Array.from(decoded.radiance).filter((_,i)=>i%3===0),[0,10,1,11,2,12,3,13,4,14,5,15]);
 const local={data:new Float32Array(48),stride:16};local.data[16]=.5;local.data[32]=.8;
 assert.deepEqual(Array.from(mapFieldMaterials(decoded,local)),[3,5,0],'exact PBR rows rekey IDs, untraced rows stay zero');
 assert.throws(()=>mapFieldMaterials(decoded,{...local,stride:8,lookupData:local.data}),/identity/);
 assert.throws(()=>mapFieldMaterials(decoded,{...local,lookupData:new Float32Array(16)}),/projected/);
 const projected={data:new Float32Array(local.data).fill(.8),stride:16,lookupData:local.data};
 assert.deepEqual(Array.from(mapFieldMaterials(decoded,projected)),[3,5,0],'canonical lookup bytes can differ from display rows');
 const nativeOnly={data:decoded.materials,stride:16};
 assert.throws(()=>mapFieldMaterials(decoded,nativeOnly),/match/,'native-engine rows alone cannot establish authored identity');
 const repeated={...decoded,authoredMaterials:new Float32Array(32),authoredMaterialIds:[0,1]};
 assert.equal(mapFieldMaterials(repeated,local)[0],7,'one authored row mapped to multiple native IDs unions their masks');
 assert.throws(()=>mapFieldMaterials({...decoded,authoredMaterials:new Float32Array(),authoredMaterialIds:[]},local),/match/);
 assert.throws(()=>mapFieldMaterials(decoded,{data:new Float32Array(16).fill(.9),stride:16}),/match/);
 for(const edit of [{sha256:null},{scene_sha256:'bad'},{byte_length:524},
  {directional:{...metadata.directional,offset_bytes:20}},
  {directional:{...metadata.directional,estimator:undefined}},
  {directional:{...metadata.directional,estimator:'volume-splat-v1'}},
  {directional:{...metadata.directional,direction_space:'view'}},{triangles:{...metadata.triangles,count:3}},
  {materials:{...metadata.materials,stride_bytes:48}},{sources:{...metadata.sources,offset_bytes:348}},
  {source_pbr16:{...metadata.source_pbr16,material_ids:[2,0]}},{source_pbr16:{...metadata.source_pbr16,material_ids:[1]}},
  {source_pbr16:{...metadata.source_pbr16,offset_bytes:516}}])
  assert.throws(()=>validateLightField({...metadata,...edit},A,0));
 for(const [offset,value,uint]of [[24,NaN,false],[36,2,false],[344,Infinity,false],[392,NaN,false],[520,NaN,false],[312,1,true],[316,2,true],[324,8,true]]) {
  const corrupt=payload.slice(0),dv=new DataView(corrupt);dv[uint?'setUint32':'setFloat32'](offset,value,true);
  assert.throws(()=>decodeLightField(corrupt,metadata));
 }
}
{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>String(b.payload.byteLength)},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);
 assert.equal(h.field.hasDirectional,true,h.output.textContent);assert.equal(h.uploads.length,4);
 assert.equal(h.field.bindSecondary({},view,2,[1,2,3]),false);
 assert.equal(h.uniforms.uResponseCount,0,'legacy field cannot suppress ordinary Phong');
 const fallbackMask={id:'zero-material-mask'};
 assert.equal(h.field.bindMaterialMask({},fallbackMask,I,1,[0,0,0]),false);
 assert.equal(h.uniforms.uFieldMaterialMask,undefined,'no material-ID suppression sampler remains');
 assert.deepEqual(Array.from(h.uploads[3].at(-1)),[3,5]);
 assert.deepEqual(h.uploads[1].slice(3,6),[2,1,6]);
 h.materials.data[0]=.1;assert.equal(h.field.hasDirectional,false,'editing material defeats old bundle');assert.equal(h.deleted.length,4);
 assert.equal(h.field.bindMaterialMask({},fallbackMask),false);
 h.field.dispose();
}
{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:{...b.metadata,sha256:A}},A);
 assert.equal(h.uploads.length,0);assert.match(h.output.textContent,/hash mismatch/);
 h.gl.texImage2D=()=>{h.error=1285;};await h.field.observe({state:'live',light_field:b.metadata},A);
 assert.equal(h.field.hasDirectional,false);assert.equal(h.field.metadata,null);assert.equal(h.deleted.length,4,'partial directional upload frees every allocation');
 h.field.dispose();
}

{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional);
 await h.field.observe({state:'live',light_field:{...b.metadata,sequence:2}},A);
 assert.equal(h.deleted.length,4,'replacement deletes all four previous textures');assert(h.field.hasDirectional);
 h.hook=null;await h.field.observe(h.status(3),A);
 assert.equal(h.deleted.length,8);assert.equal(h.field.hasDirectional,false,'legacy snapshots clear directional textures');
 assert.equal(h.field.bind({},I,1,[0,0,0]),true);h.field.dispose();
}
{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());let release,started;
 const ready=new Promise(r=>started=r);
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:()=>new Promise(r=>{release=r;started();})});
 const work=h.field.observe({state:'live',light_field:b.metadata},A);
 await ready;h.off();release(b.payload);await work;
 assert.equal(h.uploads.length,0,'v2 cancellation prevents every texture upload');h.field.dispose();
}
{
 const h=harness(),b=await bundle();h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);h.gl.isContextLost=()=>true;
 assert.equal(h.field.bindSecondary({},I,1,[0,0,0]),false);assert.equal(h.field.hasDirectional,false);assert.equal(h.deleted.length,4);h.field.dispose();
}

{
 const h=harness(),b=await bundle();h.materials.lookupData=h.materials.data.slice();h.materials.data[7]=1;
 h.on();h.field.connect(A,h.field.capture());
 h.hook=async()=>({ok:true,status:200,headers:{get:()=>null},arrayBuffer:async()=>b.payload});
 await h.field.observe({state:'live',light_field:b.metadata},A);assert(h.field.hasDirectional,h.output.textContent);
 h.materials.data[7]=.5;assert.equal(h.field.hasDirectional,false,'raw opacity edit invalidates even when lookup rows are unchanged');
 h.field.dispose();
}
const geometry={interior:[[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]]],glass:[]};
const unpack=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
const plain=unpack(packMarblePhoto(geometry,A));assert(!('receive_light_field'in plain));
{
 const h=harness();h.on();h.transport.checked=true;h.transport.dispatchEvent(new Event('change'));
 const ticket=h.field.capture();assert.equal(ticket.surfaceTransport,'studio-pbr-mixture-v1');h.field.connect(A,ticket);
 await h.field.observe(h.status(),A);assert.equal(h.calls,0);assert.match(h.output.textContent,/surface transport/);
 h.field.connect(A,h.field.capture());const status=h.status();status.light_field.surface_transport='studio-pbr-mixture-v1';
 await h.field.observe(status,A);assert.equal(h.calls,1);assert(h.field.metadata);
 h.transport.checked=false;h.transport.dispatchEvent(new Event('change'));assert.equal(h.field.metadata,null);
 h.field.connect(A,ticket);await h.field.observe(status,A);assert.equal(h.calls,1,'old transport ticket cannot reconnect');
 h.field.connect(A,h.field.capture());await h.field.observe(status,A);assert.equal(h.calls,1,'corrected receipt cannot be claimed by legacy request');
 assert.throws(()=>validateLightField({...meta(),surface_transport:'unknown'},A,0),/surface transport/);
 h.field.dispose();
}
assert.throws(()=>packMarblePhoto(geometry,A,{surfaceTransport:'studio-pbr-mixture-v1',transport:'fixed:1'}),/RGB/);
assert(!('surface_transport' in plain));
assert.equal(unpack(packMarblePhoto(geometry,A,{surfaceTransport:'studio-pbr-mixture-v1'})).surface_transport,'studio-pbr-mixture-v1');
for(const surfaceTransport of [null,false,'legacy','unknown'])assert.throws(()=>packMarblePhoto(geometry,A,{surfaceTransport}));
assert.deepEqual(unpack(packMarblePhoto(geometry,A,{receiveLightField:false})),plain,'false preserves default wire identity');
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true})).receive_light_field,true);
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true})).light_field_stencil,12);
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true})).light_field_grid_size,64);
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true})).light_field_estimator,'track-length-root-sweep-v1');
const subjectField=unpack(packMarblePhoto(geometry,A,{receiveLightField:true,lightFieldRegion:'subject'}));
const sampling={version:1,ray_budget:4096,seed:42,refresh_each_sweep:false};
assert.deepEqual(unpack(packMarblePhoto(geometry,A,{lightFieldInternalMedia:false})),plain);
for(const lightFieldInternalMedia of [true,1,'true'])assert.throws(()=>packMarblePhoto(geometry,A,{lightFieldInternalMedia}));
assert.deepEqual(unpack(packMarblePhoto(geometry,A,{emitterSampling:sampling})).emitter_sampling,sampling);
for(const emitterSampling of [{...sampling,ray_budget:127},{...sampling,seed:2147483648},{...sampling,refresh_each_sweep:1},
 {...sampling,refresh_each_sweep:true},{...sampling,gain:10}])assert.throws(()=>packMarblePhoto(geometry,A,{emitterSampling}));
assert.equal(subjectField.light_field_cell_size_m,.0005);assert.equal(subjectField.light_field_margin_m,.001);
assert.equal(subjectField.light_field_region,'subject');assert(!('light_field_grid_size' in subjectField),'subject pitch chooses grid server-side');
for(const options of [{lightFieldRegion:'subject'},{receiveLightField:true,lightFieldRegion:'stage'},
 {receiveLightField:true,lightFieldCellSizeM:.0005},{receiveLightField:true,lightFieldRegion:'subject',lightFieldCellSizeM:.000001},
 {receiveLightField:true,lightFieldRegion:'subject',lightFieldMarginM:.1}])assert.throws(()=>packMarblePhoto(geometry,A,options));
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true,lightFieldGridSize:32})).light_field_grid_size,32);
for(const options of [{lightFieldGridSize:64},{receiveLightField:true,lightFieldGridSize:33},{receiveLightField:true,lightFieldGridSize:'64'}])
 assert.throws(()=>packMarblePhoto(geometry,A,options));
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true,lightFieldStencil:6})).light_field_stencil,6);
for(const options of [{lightFieldStencil:12},{receiveLightField:true,lightFieldStencil:7},{receiveLightField:true,lightFieldStencil:'12'}])
 assert.throws(()=>packMarblePhoto(geometry,A,options));
for(const options of [{receiveLightField:1},{receiveLightField:true,transport:'fixed:1'},
 {receiveLightField:true,orbitFrames:8},{receiveLightField:true,tiled:{tile_size:1024}}])assert.throws(()=>packMarblePhoto(geometry,A,options));
console.log('PASS: real-field metadata/coordinates/packing; opt-in, replacement, revision, cancellation, stale response and upload-failure lifecycle');

{
 const payload=new ArrayBuffer(240),v=new DataView(payload);
 const section={format:'arrival-record-v1',layout:'sparse-zyx-bin',stride_bytes:120,wavelengths_nm:[650,550,450],offset_bytes:0,byte_length:240,count:2};
 v.setUint32(0,2,true);v.setUint32(52,1,true);v.setUint32(64,1,true); // Visited, quantized zero is not unvisited.
 v.setUint32(88,1,true);v.setUint32(92,1,true);v.setUint32(116,1,true);
 v.setUint32(120,7,true);v.setUint32(120+76,1,true); // Excluded evidence is retained independently.
 const limits={cellBinCount:12,sourceCount:1,triangleCount:1};
 const a=decodeFieldArrivals(payload,section,limits);
 assert.deepEqual([...a.indices],[2,7]);assert.deepEqual([...a.visits],[1,0,0,0,0,0]);assert.equal(a.roundedZero[0],1);assert.equal(a.excluded[3],1);
 assert(a.spectral.every(x=>x===0));assert.equal(a.context[0],1);
 for(const [offset,value] of [[120,2],[64,2],[88,2],[116,4]]) {
  const bad=payload.slice(0);new DataView(bad).setUint32(offset,value,true);assert.throws(()=>decodeFieldArrivals(bad,section,limits));
 }
 const bad=payload.slice(0);new DataView(bad).setFloat32(4,NaN,true);assert.throws(()=>decodeFieldArrivals(bad,section,limits));
 assert.throws(()=>decodeFieldArrivals(payload,{...section,byte_length:120},limits));
}
console.log('Arrival spectral moments, independent prequantization visits, excluded evidence and context validation PASS');

{
 const intervals=new ArrayBuffer(1024),responses=new ArrayBuffer(1024),i=new DataView(intervals),r=new DataView(responses);
 for(let row=0;row<2;row++) {
  const o=row*512;i.setUint32(o+6*4,3,true);i.setUint32(o+40*4,7,true);
  i.setUint32(o+115*4,41,true);i.setUint32(o+4*4,row?99:41,true); // Later hit is never the recipient.
  i.setUint32(o+15*4,13,true);i.setUint32(o+2*4,2,true); // Same original invocation/sample, not two samples.
 }
 r.setUint32(0,0x7fc01234,true);r.setUint32(96*4,1,true); // Preserve exact invalid float bits with evaluated presence.
 r.setUint32(109*4,1,true);r.setUint32(120*4,2,true); // Absence and rounding stay distinct from presence.
 const receipt={response_schema:'native-camera-response-copies-v1',response_weighting:'original_path_throughput_not_availability_premultiplied',
  denominator:'original_camera_samples_not_response_rows',application_ready:false,row_count:2};
 const d=decodeEvaluatedResponses(intervals,responses,receipt);
 assert.equal(d.responses[0],0x7fc01234);assert.equal(d.responses[96],1);assert.equal(d.responses[109],1);assert.equal(d.responses[120],2);
 assert.equal(d.intervals[128+115],41);assert.equal(d.intervals[128+4],99);assert.equal(d.applicationReady,false);
 r.setUint32(0,0,true);assert.equal(d.responses[0],0x7fc01234,'decoded copies retain original evidence');
 r.setUint32(108*4,2,true);assert.throws(()=>decodeEvaluatedResponses(intervals,responses,receipt),/Reserved/);
 assert.throws(()=>decodeEvaluatedResponses(intervals,responses,{...receipt,application_ready:true}));
 // Proof only: F is already producer-resolved for the original visible recipient.
 // No raw spectral row is used to construct it, and no attenuation is replayed.
 const P=[.1,.2,.3],A=.4,B=[.5,.6,.7],F=[.9,.8,.7];
 for(const c of [0,.25,1])for(let band=0;band<3;band++) {
  const rgb=c*F[band]+(1-c)*P[band],alpha=c+(1-c)*A;
  const composite=rgb+(1-alpha)*B[band],expected=c*F[band]+(1-c)*(P[band]+(1-A)*B[band]);
  assert(Math.abs(composite-expected)<1e-12);
 }
 const h=harness();assert.equal(h.field.responseInformationRequested,false);assert.equal(h.field.capture(),null);
 h.on();assert.equal(h.field.capture().information,undefined,'ordinary field opt-in does not request extra native copies');
 assert.equal(h.field.responseInformationRequested,false);
 h.responses.checked=true;h.responses.dispatchEvent(new Event('change'));
 assert.equal(h.field.capture().information,'evaluated-responses-v1');assert.equal(h.field.responseInformationRequested,true);h.field.dispose();
}
console.log('Native response copies preserve bits, provenance and independent evidence; first-recipient complementary blend algebra PASS');

// Exercise the existing receiver, with no rendering page or GPU context. Raw
// typed snapshots must never take the legacy scalar or directional upload path.
{
 const payload=new ArrayBuffer(0),sha=Buffer.from(await crypto.subtle.digest('SHA-256',payload)).toString('hex');
 const metadata={schema:'studio-scene-responses-v1',scene_revision:B,publication_sequence:0,byte_order:'little',
  payload_bytes:0,payload_sha256:sha,
  families:Object.fromEntries([[1,'forward','evaluated-forward-responses-v1'],[2,'reactive','evaluated-forward-responses-v1'],
   [3,'camera_rhs_summands','native-camera-response-pairs-v1'],[4,'camera_complete_sample','native-camera-scene-sample-v1']].map(([id,name,codec])=>[id,{name,codec}])),
  contexts:[],sections:[['scene_index',80],['native_rows',512],['camera_rhs_rows',1536],['scalar_cells',24]].map(([name,stride])=>({name,stride,count:0,offset:0})),
  scalar_snapshot:null,indexing:{semantics:'lookup buckets only; exact positions/directions retained',stencil:'icosahedron12-v1',bounds:null,dimensions:null},
  retention:{mode:'bounded-fifo',capacity:4,retained_entries:0,evicted_entries:2,duplicate_entries:0,invalidated_entries:0,incomplete_dispatches:0,unavailable_on_eviction:true},
  complete_retained_snapshot:true,physical_completeness_claim:false,application_ready:false};
 const receipt={schema:metadata.schema,digest:A,live_revision:0,sequence:1,name:`light-scene-response-${'1'.repeat(32)}.bin`,
  byte_length:0,sha256:sha,application_ready:false,metadata};
 assert.equal(validateSceneResponseReceipt(receipt,A,0).name,receipt.name);
 for(const edit of [{digest:B},{live_revision:1},{name:'../response.bin'},{application_ready:true},{byte_length:64*1024*1024+1}])
  assert.throws(()=>validateSceneResponseReceipt({...receipt,...edit},A,0));
 const h=harness();h.on();h.responses.checked=true;h.responses.dispatchEvent(new Event('change'));
 h.field.connect(A,h.field.capture());
 h.hook=async url=>{assert.match(url,/\/api\/marble_photo\/field\?digest=.*&name=light-scene-response-/);
  return {ok:true,status:200,headers:{get:()=> '0'},arrayBuffer:async()=>payload};};
 await h.field.observe({state:'live',scene_responses:receipt,light_field:meta()},A);
 assert.equal(h.uploads.length,0,'typed response data never becomes invented radiance');
 assert.equal(h.field.hasResponses,false,'unready native evidence never suppresses Phong');
 assert.equal(h.field.sceneResponseMetadata.metadata.retention.evicted_entries,2);
 assert.match(h.output.textContent,/Typed data only/);
 await h.field.observe({state:'live',scene_responses:receipt},A);assert.equal(h.calls,1,'one fetch per immutable snapshot');
 const replacement=structuredClone(receipt);replacement.sequence=2;replacement.metadata.retention.evicted_entries=3;
 await h.field.observe({state:'live',scene_responses:replacement},A);
 assert.equal(h.field.sceneResponseMetadata.metadata.retention.evicted_entries,3,'retained snapshots replace, never merge');
 h.hook=async()=>{throw Error('temporary transfer failure');};
 await h.field.observe({state:'live',scene_responses:{...replacement,sequence:3}},A);
 assert.equal(h.field.sceneResponseMetadata.sequence,2,'a transfer failure cannot erase the prior validated same-scene snapshot');
 assert.match(h.output.textContent,/previous validated scene snapshot/);
 h.field.revise(A,1);assert.equal(h.field.sceneResponseMetadata,null,'camera revision cannot retain stale query context');
 await h.field.observe({state:'live',scene_responses:receipt},A);assert.equal(h.calls,3,'stale revision is refused before transfer');
 h.off();assert.equal(h.field.sceneResponseMetadata,null);h.field.dispose();
 const host=readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
 assert.match(host,/const wantsSecondary=.*lightField\?\.hasResponses/);
 const html=readFileSync(new URL('../examples/meowdoku/web/marble_studio.html',import.meta.url),'utf8');
 assert(!/\bchecked\b/.test(html.match(/<input[^>]*id="phong-field-responses"[^>]*>/)[0]));
}
console.log('Scene response receiver PASS: explicit extra opt-in, immutable typed snapshots, missing/evicted data keep Phong, no GPU uploads');

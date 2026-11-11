import assert from 'node:assert/strict';
import {createStudioLightField,validateLightField,decodeLightField,fieldFromView} from '../examples/meowdoku/web/marble_light_field.js';
import {packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

const A='a'.repeat(64),B='b'.repeat(64),I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
const meta=(sequence=1,revision=0)=>({schema:'studio.forward-field.v1',digest:A,live_revision:revision,sequence,
 name:`light-field-${String(sequence).padStart(32,'0')}.bin`,format:'rgb-f32le',layout:'zyx-interleaved-rgb',
 color_space:'linear-srgb',dimensions:[2,1,1],bounds_min_m:[1,2,3],bounds_max_m:[3,4,5],byte_length:24,ray_jobs:128});
const bytes=()=>new Float32Array([1,-.25,3,4,5,6]).buffer;
assert.equal(validateLightField(meta(),A,0).byte_length,24);
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

function harness(){
 const uploads=[],deleted=[],uniforms={},bindings=new Map();let active=108,unpack=4,id=0,error=0,key='scene-1',calls=0,hook=null;
 const gl={ACTIVE_TEXTURE:1,TEXTURE_BINDING_3D:2,UNPACK_ALIGNMENT:3,MAX_3D_TEXTURE_SIZE:4,TEXTURE8:108,
  TEXTURE_3D:5,TEXTURE_MIN_FILTER:6,TEXTURE_MAG_FILTER:7,TEXTURE_WRAP_S:8,TEXTURE_WRAP_T:9,TEXTURE_WRAP_R:10,
  NEAREST:11,CLAMP_TO_EDGE:12,RGB32F:13,RGB:14,FLOAT:15,NO_ERROR:0,
  getParameter(p){return p===1?active:p===2?bindings.get(active)??null:p===3?unpack:256;},
  activeTexture(v){active=v;},bindTexture(t,v){assert(!deleted.includes(v),'never rebind a deleted texture');bindings.set(active,v);},
  pixelStorei(p,v){unpack=v;},createTexture(){return {id:++id};},deleteTexture(t){deleted.push(t);for(const [k,v]of bindings)if(v===t)bindings.set(k,null);},
  texParameteri(){},texImage3D(...args){uploads.push(args);},getError(){const v=error;error=0;return v;},
  getUniformLocation(p,n){return n;},uniform1f(n,v){uniforms[n]=v;},uniform1i(n,v){uniforms[n]=v;},uniformMatrix4fv(n,t,v){uniforms[n]=Array.from(v);}};
 const toggle=new EventTarget();toggle.checked=false;const output={textContent:''};
 const doc={getElementById:id=>id==='phong-light-field'?toggle:output};
 const fetcher=async(...args)=>{calls++;if(hook)return hook(...args);return {ok:true,status:200,headers:{get:()=> '24'},arrayBuffer:async()=>bytes()};};
 const fallbackTexture={id:'original-zero-volume'};bindings.set(gl.TEXTURE8,fallbackTexture);
 const field=createStudioLightField({gl,doc,fallbackTexture,getSceneKey:()=>{if(key==='bad')throw Error('incompatible');return key;},fetcher});
 return {field,toggle,output,gl,uploads,deleted,uniforms,bindings,fallbackTexture,get calls(){return calls;},set key(v){key=v;},
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

const geometry={interior:[[[0,0,0],[1,0,0],[0,1,0],0,[[0,0,1],[0,0,1],[0,0,1]]]],glass:[]};
const unpack=buffer=>JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,16,new DataView(buffer).getUint32(4,true))));
const plain=unpack(packMarblePhoto(geometry,A));assert(!('receive_light_field'in plain));
assert.deepEqual(unpack(packMarblePhoto(geometry,A,{receiveLightField:false})),plain,'false preserves default wire identity');
assert.equal(unpack(packMarblePhoto(geometry,A,{receiveLightField:true})).receive_light_field,true);
for(const options of [{receiveLightField:1},{receiveLightField:true,transport:'fixed:1'},
 {receiveLightField:true,orbitFrames:8},{receiveLightField:true,tiled:{tile_size:1024}}])assert.throws(()=>packMarblePhoto(geometry,A,options));
console.log('PASS: real-field metadata/coordinates/packing; opt-in, replacement, revision, cancellation, stale response and upload-failure lifecycle');

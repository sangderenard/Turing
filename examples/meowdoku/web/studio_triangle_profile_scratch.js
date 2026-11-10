// Actual Studio hit_tri, compiled from original GLSL into shared SSA.
// This host only packs records and reduces returned distances; no JS intersection law.
import {validateRayBatch} from './client_ray_kernel.js';
import {validateTriangleScene} from './client_triangle_kernel.js';
const MAX_PAIRS=1048576;
export async function verifiedComputeBytes(base,file,expected){const r=await fetch(new URL(file,base),{cache:'no-store'});if(!r.ok)throw Error('Kernel fetch '+r.status);const b=await r.arrayBuffer();if(expected){const h=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),x=>x.toString(16).padStart(2,'0')).join('');if(h!==expected)throw Error('Compiled kernel hash mismatch');}return b;}
export class StudioTriangleCompute {
 static async create(base,{backend='auto'}={}){
  if(!['auto','wasm','webgpu'].includes(backend))throw Error('Unknown compute backend');
  const m=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,'studio_hit_tri.json')));
  if(m.count!==256||m.input_order?.length!==15||m.output_order?.join(',')!=='valid,distance'||!m.artifacts)throw Error('Unsupported compiled kernel ABI');
  const q=new StudioTriangleCompute();q.manifest=m;q.closed=false;q.busy=false;
  if(backend!=='wasm'&&globalThis.navigator?.gpu){
   const adapter=await navigator.gpu.requestAdapter();
   if(adapter){q.device=await adapter.requestDevice();q.device.lost.then(info=>{q.lost=info.message||'device lost';});try{
    const source=new TextDecoder().decode(await verifiedComputeBytes(base,'studio_hit_tri.wgsl',m.artifacts.wgsl));const module=q.device.createShaderModule({code:source});
    const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(x=>x.message).join('; '));
    q.pipeline=await q.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});q.backend='webgpu';return q;
   }catch(e){q.device.destroy();throw e;}}
  }
  if(backend==='webgpu')throw Error('WebGPU unavailable');
  q.wasm=(await WebAssembly.instantiate(await verifiedComputeBytes(base,'studio_hit_tri.wasm',m.artifacts.wasm))).instance;q.backend='wasm';return q;
 }
 setScene(scene){if(this.closed||this.busy)throw Error('Closed or busy');validateTriangleScene(scene);this.scene={revision:scene.revision,triangles:scene.triangles.map(t=>({v0:[...t.v0],e1:[...t.e1],e2:[...t.e2],owner:t.owner}))};}
 async query(rays,revision,{signal}={}){
  const count=validateRayBatch(rays);rays=rays.slice();if(this.closed||this.busy||this.scene?.revision!==revision)throw Error('Unavailable or stale scene');
  const tris=this.scene.triangles;if(count*tris.length>MAX_PAIRS)throw Error('Query exceeds bounded pair budget; retain and split work');
  const check=()=>{if(signal?.aborted||this.closed||this.lost)throw Error('Client query canceled; retained work must be reissued');};check();this.busy=true;
  const start=performance.now(),hits=new Float32Array(count*4);for(let i=0;i<count;i++)hits.set([-1,-1,0,0],i*4);
  const n=256,total=count*tris.length;
  try{for(let begin=0;begin<total;begin+=n){check();const size=Math.min(n,total-begin),feed=new Float32Array(n*15);
    for(let j=0;j<size;j++){const pair=begin+j,ri=Math.floor(pair/tris.length),t=tris[pair%tris.length],r=ri*8;const values=[...rays.subarray(r,r+3),...rays.subarray(r+4,r+7),...t.v0,...t.e1,...t.e2];values.forEach((v,k)=>feed[k*n+j]=v);}
    const result=this.backend==='webgpu'?await this.gpuBatch(feed):this.wasmBatch(feed,size);check();
    for(let j=0;j<size;j++){const pair=begin+j,ri=Math.floor(pair/tris.length),ti=pair%tris.length,d=result.distance[j],h=ri*4,r=ri*8;
     if(![0,1].includes(result.valid[j])||(result.valid[j]&&!Number.isFinite(d)))throw Error('Invalid compiled result; retained work must be reissued');
     if(result.valid[j]&&d>rays[r+3]&&d<rays[r+7]&&(!hits[h+2]||d<hits[h]))hits.set([d,tris[ti].owner,1,ti],h);}
    await new Promise(r=>setTimeout(r,0));
   }
   return {revision,hits,count,backend:this.backend,kernel:this.manifest.source_sha256,elapsedMs:performance.now()-start};
  }finally{this.busy=false;}
 }
 wasmBatch(feed,size){const valid=new Uint32Array(size),distance=new Float32Array(size),memory=this.wasm.exports.memory;
  const flag=new BigInt64Array(memory.buffer,16,1),dist=new Float64Array(memory.buffer,24,1),fn=this.wasm.exports.hit_tri;for(let i=0;i<size;i++){fn(feed[i+0],feed[i+256],feed[i+512],feed[i+768],feed[i+1024],feed[i+1280],feed[i+1536],feed[i+1792],feed[i+2048],feed[i+2304],feed[i+2560],feed[i+2816],feed[i+3072],feed[i+3328],feed[i+3584],16,24);valid[i]=Number(flag[0]);distance[i]=dist[0];}return {valid,distance};
 }
 async gpuBatch(feed){const d=this.device,n=256;d.pushErrorScope('validation');let popped=false;const inputs=d.createBuffer({size:feed.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const outputs=[0,1].map(()=>d.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}));const read=d.createBuffer({size:n*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{d.queue.writeBuffer(inputs,0,feed);const group=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[inputs,...outputs].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const enc=d.createCommandEncoder(),p=enc.beginComputePass();p.setPipeline(this.pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();outputs.forEach((o,i)=>enc.copyBufferToBuffer(o,0,read,i*n*4,n*4));d.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const error=await d.popErrorScope();popped=true;if(error)throw Error(error.message);const buffer=read.getMappedRange(),valid=new Uint32Array(buffer,0,n).slice(),distance=new Float32Array(buffer,n*4,n).slice();read.unmap();return {valid,distance};
  }finally{[inputs,...outputs,read].forEach(b=>b.destroy());if(!popped)await d.popErrorScope();}
 }
 close(){this.closed=true;this.device?.destroy();this.wasm=null;this.scene=null;}
}

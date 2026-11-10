// Exact compiled slab law. The owning BVH host performs traversal, never slab math.
import {verifiedComputeBytes} from './studio_triangle_compute.js';
export class CompiledSlabAxis {
 static async create(base,triangle){
  const q=new CompiledSlabAxis();q.backend=triangle.backend;q.device=triangle.device;
  q.manifest=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,'studio_slab_axis.json')));
  const m=q.manifest;if(m.schema!=='studio.scalar-law.v1'||m.entry!=='slab_axis'||m.count!==256||m.input_order?.length!==6||m.output_order?.join(',')!=='valid,nt,ft')throw Error('Unsupported slab ABI');
  if(q.backend==='wasm')q.wasm=(await WebAssembly.instantiate(await verifiedComputeBytes(base,'studio_slab_axis.wasm',m.artifacts.wasm))).instance;
  else {const code=new TextDecoder().decode(await verifiedComputeBytes(base,'studio_slab_axis.wgsl',m.artifacts.wgsl));const module=q.device.createShaderModule({code});const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error');if(errors.length)throw Error(errors.map(e=>e.message).join(';'));q.pipeline=await q.device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});}
  return q;
 }
 async run(feed,size){
  if(this.backend==='wasm'){const valid=new Uint32Array(size),near=new Float32Array(size),far=new Float32Array(size),memory=this.wasm.exports.memory;
   const flag=new BigInt64Array(memory.buffer,16,1),nv=new Float64Array(memory.buffer,24,1),fv=new Float64Array(memory.buffer,32,1),fn=this.wasm.exports.slab_axis;for(let i=0;i<size;i++){fn(feed[i+0],feed[i+256],feed[i+512],feed[i+768],feed[i+1024],feed[i+1280],16,24,32);valid[i]=Number(flag[0]);near[i]=nv[0];far[i]=fv[0];}return {valid,near,far};}
  const d=this.device,n=256;d.pushErrorScope('validation');let popped=false;
  const input=d.createBuffer({size:feed.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),outputs=[0,1,2].map(()=>d.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC})),read=d.createBuffer({size:n*12,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{d.queue.writeBuffer(input,0,feed);const group=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[input,...outputs].map((buffer,binding)=>({binding,resource:{buffer}}))});const enc=d.createCommandEncoder(),p=enc.beginComputePass();p.setPipeline(this.pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();outputs.forEach((o,i)=>enc.copyBufferToBuffer(o,0,read,i*n*4,n*4));d.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const error=await d.popErrorScope();popped=true;if(error)throw Error(error.message);const buf=read.getMappedRange(),valid=new Uint32Array(buf,0,size).slice(),near=new Float32Array(buf,n*4,size).slice(),far=new Float32Array(buf,n*8,size).slice();read.unmap();return {valid,near,far};}
  finally{[input,...outputs,read].forEach(b=>b.destroy());if(!popped)await d.popErrorScope();}
 }
}

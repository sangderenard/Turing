import {verifiedComputeBytes} from './studio_triangle_compute.js';
// One compiler-owned counted loop per candidate batch. No scalar Wasm calls here.
export class StudioCompiledLawBatch {
 static async create(base,stem,sourceText){
  const q=new StudioCompiledLawBatch(),m=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,stem+'.json')));
  if(m.schema!=='studio.compiled-law-batch.v1'||m.source!==sourceText||m.capacity!==16384||m.wasm_parameters[0]?.[0]!=='i32'||m.buffer_order.length!==m.buffer_dtypes.length)throw Error('Unsupported compiled batch ABI');
  q.manifest=m;q.instance=(await WebAssembly.instantiate(await verifiedComputeBytes(base,stem+'.wasm',m.artifacts.wasm))).instance;q.offsets=new Map();let end=Math.ceil(m.reserved_bytes/16)*16;
  m.buffer_order.forEach((id,i)=>{if(!['float32','int32'].includes(m.buffer_dtypes[i]))throw Error('Unsupported batch buffer type');q.offsets.set(id,end);end+=m.capacity*4;});
  const memory=q.instance.exports.memory;if(memory.buffer.byteLength<end)memory.grow(Math.ceil((end-memory.buffer.byteLength)/65536));
  q.views=new Map(m.buffer_order.map((id,i)=>[id,new (m.buffer_dtypes[i]==='int32'?Int32Array:Float32Array)(memory.buffer,q.offsets.get(id),m.capacity)]));
  q.args=m.wasm_parameters.slice(1).map(([kind,id])=>{if(kind!=='buffer'||!q.offsets.has(id))throw Error('Unsupported batch parameter');return q.offsets.get(id)});return q;
 }
 run(feed,count,stride){const m=this.manifest;if(!Number.isInteger(count)||count<0||count>m.capacity||stride<count||feed.length<m.input_ids.length*stride)throw Error('Invalid compiled batch shape');m.input_ids.forEach((id,k)=>this.views.get(id).set(feed.subarray(k*stride,k*stride+count)));this.instance.exports[m.entry](count,...this.args);return m.output_ids.map(id=>this.views.get(id).slice(0,count));}
}

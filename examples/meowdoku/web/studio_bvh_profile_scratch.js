// Temporary host-orchestrated original Studio LIFO BVH traversal.
// All slab and triangle arithmetic runs in compiler-produced Wasm/WebGPU laws.
import {StudioTriangleCompute,verifiedComputeBytes} from './studio_triangle_profile_scratch.js';
import {CompiledSlabAxis} from './studio_slab_profile_scratch.js';
import {validateRayBatch} from './client_ray_kernel.js';
export class StudioBvhCompute {
 static async create(base,options={}){const q=new StudioBvhCompute();q.triangle=await StudioTriangleCompute.create(base,options);try{q.slab=await CompiledSlabAxis.create(base,q.triangle);}catch(e){q.triangle.close();throw e;}q.backend=q.triangle.backend;const contract=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,'studio_bvh_query.json')));if(contract.schema!=='studio.bvh-query.v1'||contract.traversal!=='original-lifo-host-v1'||contract.epsilon!==1e-7)throw Error('Unsupported traversal contract');q.manifest={...q.triangle.manifest,...contract};q.closed=false;q.busy=false;return q;}
 setScene(s){
  if(this.closed||this.busy)throw Error('Closed or busy');
  if(this.manifest.query_variant==='region-bvh'&&s?.query_variant!=='region-bvh')throw Error('Region query requires explicit primary-region identity');
  if(s?.query_variant==='region-bvh'&&this.manifest.query_variant!=='region-bvh')throw Error('Region query cannot use plain triangle law');
  if(typeof s?.revision!=='string'||!s.revision||s.unsupportedGeometry?.length)throw Error('Unsupported scene identity/geometry');
  if(!Array.isArray(s.triangles)||!s.triangles.length||s.triangles.length>200000||!Array.isArray(s.bvh_nodes)||!s.bvh_nodes.length||s.bvh_nodes.length>400000||!Array.isArray(s.bvh_ids)||s.bvh_epsilon!==1e-7)throw Error('Unsupported exact Studio BVH ABI');
  const ownerIndex=new Map(),tri=new Float32Array(s.triangles.length*9),owners=new Int32Array(s.triangles.length);
  s.triangles.forEach((t,i)=>{if(!Number.isSafeInteger(t.owner)||t.owner<0||t.owner>16777215||ownerIndex.has(t.owner))throw Error('Invalid/duplicate primitive ID');const v=[...t.v0,...t.e1,...t.e2];if(v.length!==9||!v.every(x=>Number.isFinite(x)&&Number.isFinite(Math.fround(x))))throw Error('Invalid primitive coordinates');tri.set(v,i*9);owners[i]=t.owner;ownerIndex.set(t.owner,i);});
  const nodes=new Float32Array(s.bvh_nodes.length*12);s.bvh_nodes.forEach((n,i)=>{if(!Array.isArray(n)||n.length!==12||!n.every(x=>Number.isFinite(x)&&Number.isFinite(Math.fround(x))))throw Error('Invalid node record');nodes.set(n,i*12);for(let k=0;k<3;k++)if(n[k]>n[k+4])throw Error('Inverted bounds');});
  const ids=Int32Array.from(s.bvh_ids.map(x=>{if(!Number.isSafeInteger(x)||!ownerIndex.has(x))throw Error('Invalid BVH primitive index');return ownerIndex.get(x);}));
  const seen=new Set(),covered=new Set(),stack=[[0,0]];
  while(stack.length){const [i,depth]=stack.pop();if(!Number.isSafeInteger(i)||i<0||i>=s.bvh_nodes.length||seen.has(i)||depth>=94)throw Error('Invalid/cyclic/overdeep BVH');seen.add(i);const b=i*12,l=nodes[b+3],r=nodes[b+7];if(l<0){const start=nodes[b+8],count=nodes[b+9];if(!Number.isSafeInteger(start)||!Number.isSafeInteger(count)||start<0||count<0||start+count>ids.length)throw Error('Invalid leaf range');for(let k=start;k<start+count;k++){if(covered.has(ids[k]))throw Error('Duplicate BVH primitive');covered.add(ids[k]);}}else{stack.push([l,depth+1],[r,depth+1]);}}
  if(seen.size!==s.bvh_nodes.length||covered.size!==s.triangles.length)throw Error('Incomplete BVH coverage');
  this.scene={revision:s.revision,tri,owners,nodes,ids,epsilon:s.bvh_epsilon};
 }
 async query(rays,revision,{signal}={}){
  const count=validateRayBatch(rays);rays=rays.slice();if(this.closed||this.busy||this.scene?.revision!==revision)throw Error('Unavailable/stale BVH scene');
  const check=()=>{if(this.closed||signal?.aborted||this.triangle.lost)throw Error('Canceled/lost BVH query; retained rays must be reissued');};check();this.busy=true;
  const scene=this.scene,stacks=Array.from({length:count},()=>[0]),hits=new Float32Array(count*4),best=new Float32Array(count);for(let i=0;i<count;i++){hits.set([0,-1,0,-1],i*4);best[i]=rays[i*8+7];}
  const start=performance.now();let lastYield=start;let nodeTests=0,triangleTests=0;const profile={slabMs:0,triangleMs:0,yieldMs:0,slabCalls:0,triangleCalls:0,wasmCalls:0,feedBytes:0,rounds:0};
  const yieldIfDue=async()=>{if(performance.now()-lastYield>=8){const yieldStart=performance.now();await new Promise(r=>setTimeout(r,0));lastYield=performance.now();profile.yieldMs+=lastYield-yieldStart;check();}};
  try{while(stacks.some(s=>s.length)){
    profile.rounds++;check();let active=[];for(let ri=0;ri<count;ri++)if(stacks[ri].length)active.push({ri,node:stacks[ri].pop(),near:scene.epsilon,far:best[ri]});nodeTests+=active.length;
    for(let axis=0;axis<3&&active.length;axis++){
     const surviving=[];for(let offset=0;offset<active.length;offset+=256){check();const group=active.slice(offset,offset+256),feed=new Float32Array(256*6);
      group.forEach((q,j)=>{const b=q.node*12,r=q.ri*8;[rays[r+axis],rays[r+4+axis],scene.nodes[b+axis],scene.nodes[b+4+axis],q.near,q.far].forEach((v,k)=>feed[k*256+j]=v);});
      profile.feedBytes+=feed.byteLength;profile.slabCalls++;if(this.backend==='wasm')profile.wasmCalls+=group.length;const slabStart=performance.now();const out=await this.slab.run(feed,group.length);profile.slabMs+=performance.now()-slabStart;check();group.forEach((q,j)=>{if(![0,1].includes(out.valid[j])||!Number.isFinite(out.near[j])||!Number.isFinite(out.far[j]))throw Error('Invalid slab result');if(out.valid[j])surviving.push({...q,near:out.near[j],far:out.far[j]});});await yieldIfDue();
     }active=surviving;
    }
    let leaves=[];
    for(const q of active){const b=q.node*12;if(scene.nodes[b+3]<0)leaves.push({...q,cursor:scene.nodes[b+8],end:scene.nodes[b+8]+scene.nodes[b+9]});else{const stack=stacks[q.ri];if(stack.length>=94)throw Error('BVH stack capacity exceeded');stack.push(scene.nodes[b+3],scene.nodes[b+7]);}}
    while(leaves.length){check();const pending=leaves.filter(q=>q.cursor<q.end);if(!pending.length)break;
     for(let offset=0;offset<pending.length;offset+=256){const group=pending.slice(offset,offset+256),feed=new Float32Array(256*15);group.forEach((q,j)=>{const r=q.ri*8,ti=scene.ids[q.cursor];q.ti=ti;[...rays.subarray(r,r+3),...rays.subarray(r+4,r+7),...scene.tri.subarray(ti*9,ti*9+9)].forEach((v,k)=>feed[k*256+j]=v);});triangleTests+=group.length;
      profile.feedBytes+=feed.byteLength;profile.triangleCalls++;if(this.backend==='wasm')profile.wasmCalls+=group.length;const triangleStart=performance.now();const out=this.backend==='webgpu'?await this.triangle.gpuBatch(feed):this.triangle.wasmBatch(feed,group.length);profile.triangleMs+=performance.now()-triangleStart;check();
      group.forEach((q,j)=>{const d=out.distance[j];if(![0,1].includes(out.valid[j])||(out.valid[j]&&!Number.isFinite(d)))throw Error('Invalid triangle result');if(out.valid[j]&&d>rays[q.ri*8+3]&&d<best[q.ri]){best[q.ri]=d;hits.set([d,scene.owners[q.ti],1,scene.owners[q.ti]],q.ri*4);}q.cursor++;});await yieldIfDue();
     }leaves=pending.filter(q=>q.cursor<q.end);
    }
    await yieldIfDue();
   }
   return {count,hits,revision,kernel:this.manifest.query_kernel_sha256,slab_kernel:this.slab.manifest.source_sha256,backend:this.backend,nodeTests,triangleTests,elapsedMs:performance.now()-start,traversal:'original-lifo-host-v1',profile:{...profile,hostMs:performance.now()-start-profile.slabMs-profile.triangleMs-profile.yieldMs}};
  }finally{this.busy=false;}
 }
 close(){this.closed=true;this.triangle?.close();this.slab=null;this.scene=null;}
}


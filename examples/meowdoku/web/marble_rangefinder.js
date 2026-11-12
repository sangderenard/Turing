// All intersections of one straight diagnostic ray with the captured Studio
// triangle mesh. Intersection arithmetic stays in the existing compiled
// original Spectral hit_tri; this host packs inputs and keeps every result.
import {StudioCompiledLawBatch} from './studio_compiled_law_batch.js';
import {verifiedComputeBytes} from './studio_triangle_compute.js';
import {dofCameraFromView} from './marble_studio_dof.js';

const MAX_TRIANGLES=200000;
const finite=(a,n)=>a&&a.length===n&&Array.from(a).every(Number.isFinite);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const unit=a=>{const n=Math.hypot(...a);if(!(n>0))throw Error('Degenerate rangefinder direction.');return a.map(x=>x/n);};
const check=signal=>{if(signal?.aborted)throw Error('Rangefinder canceled.');};

export function rangefinderKey(capture) {
  return JSON.stringify([capture?.revision,capture?.state?.camera,capture?.state?.subject?.model]);
}

export function rangefinderRay(capture,u,v) {
  const state=capture?.state,camera=state?.camera,frame=state?.frame;
  if(state?.schema!=='turing.studio-visible-scene.v1'||state.channel!=='studio'||
      !finite(camera?.view,16)||!finite(frame?.rayOriginM,3)||!(frame.unitM>0)||
      !(camera.aspect>0)||!(camera.fovDegrees>0&&camera.fovDegrees<179)||
      ![u,v].every(x=>Number.isFinite(x)&&x>=0&&x<=1))throw Error('Invalid displayed Studio camera or rangefinder point.');
  const view=camera.view,pose=dofCameraFromView(view,{unitM:frame.unitM,targetDistance:1,fovDegrees:camera.fovDegrees});
  if(view[3]!==0||view[7]!==0||view[11]!==0||view[15]!==1||[0,1,2].some(i=>Math.abs(Math.hypot(view[i],view[4+i],view[8+i])-1)>4e-6))
    throw Error('Rangefinder needs a rigid camera view.');
  const eye=pose.eye_m.map((x,i)=>x+frame.rayOriginM[i]);
  const forward=unit([0,1,2].map(i=>-view[i*4+2]));
  const right=unit([0,1,2].map(i=>view[i*4])),up=unit([0,1,2].map(i=>view[i*4+1]));
  // A rigid view is required; do not silently treat scaled/sheared cameras as rigid.
  if(Math.max(Math.abs(dot(right,up)),Math.abs(dot(right,forward)),Math.abs(dot(up,forward)))>1e-5)
    throw Error('Rangefinder needs an orthogonal camera basis.');
  const shift=camera.tilt_shift??[0,0];
  if(!finite(shift,2))throw Error('Invalid camera principal point.');
  const extent=Math.tan(camera.fovDegrees*Math.PI/360),x=(2*u-1+shift[0])*extent*camera.aspect,y=(1-2*v+shift[1])*extent;
  const direction=unit(forward.map((a,i)=>a+x*right[i]+y*up[i]));
  const ray=Float32Array.from([...eye,0,...direction,1e6]);
  if(!Array.from(ray).every(Number.isFinite))throw Error('Rangefinder ray exceeds native float precision.');
  return {ray,forward,camera:{...pose,eye_m:eye,target_m:eye.map((x,i)=>x+forward[i]*frame.unitM)},u,v};
}

export function rangefinderScene(capture) {
  const s=capture?.state;
  if(s?.schema!=='turing.studio-visible-scene.v1'||s.channel!=='studio'||!Number.isSafeInteger(capture.revision))
    throw Error('Rangefinder needs the exact current Studio capture.');
  const blocks=[{ray:s.ray,sources:s.triangleSources,subject:false}];
  if(s.subject)blocks.push({ray:s.subject.ray,subject:true});
  const pbr=s.ray?.pbr;
  if(!pbr||pbr.length%16||!Array.from(pbr).every(Number.isFinite))throw Error('Captured materials are incomplete.');
  const count=blocks.reduce((n,b)=>n+(b.ray?.material_ids?.length||0),0);
  if(count<1||count>MAX_TRIANGLES)throw Error(`Rangefinder supports 1–${MAX_TRIANGLES} captured triangles.`);
  const geometry=new Float32Array(count*9),owners=[];
  for(const block of blocks){
    const r=block.ray;
    if(!r||r.vertices?.length!==r.material_ids?.length*9||(!block.subject&&block.sources?.length!==r.material_ids.length))
      throw Error('Captured triangle identity is incomplete.');
    for(let i=0;i<r.material_ids.length;i++){
      const points=Array.from(r.vertices.slice(i*9,i*9+9)),material=r.material_ids[i],index=owners.length;
      if(!finite(points,9)||!Number.isSafeInteger(material)||material<0||material>=pbr.length/16)
        throw Error('Invalid captured triangle or material.');
      const source=block.subject?{partId:s.subject.id??'subject',bodyId:s.subject.id??'subject',triangle:i}:block.sources[i];
      if(typeof source?.partId!=='string'||typeof source?.bodyId!=='string'||!Number.isSafeInteger(source.triangle))
        throw Error('Captured source owner is missing.');
      const row=Array.from(pbr.slice(material*16,material*16+16));
      // Match scene_builder.append_saved_studio_mesh_to_gpu_scene at identity
      // scale: first quantize vertices to native f32, then form f32 edges.
      const native=points.map(Math.fround);
      const values=[...native.slice(0,3),...[0,1,2].map(k=>Math.fround(native[k+3]-native[k])),...[0,1,2].map(k=>Math.fround(native[k+6]-native[k]))];
      geometry.set(values,index*9);
      owners.push({...source,sourceKind:block.subject?'subject':'fixture',material,
        transmission:row[5],opacity:row[7],ior:row[6],opaque:row[7]>=.999||row[5]<=0});
    }
  }
  if(!Array.from(geometry).every(Number.isFinite))throw Error('Captured geometry exceeds native float precision.');
  return {key:rangefinderKey(capture),revision:capture.revision,geometry,owners,count,
    scope:'All captured Studio triangle surfaces, straight pinhole ray. Analytic/refraction transport is not queried.'};
}

export class StudioRangefinder {
  static async create(base) {
    const contract=JSON.parse(new TextDecoder().decode(await verifiedComputeBytes(base,'studio_bvh_query.json')));
    if(contract.schema!=='studio.bvh-query.v1'||!contract.laws?.hit_tri)throw Error('Original Studio triangle law contract is missing.');
    const law=await StudioCompiledLawBatch.create(base,'studio_hit_tri_batch',contract.laws.hit_tri);
    if(law.manifest.input_ids.length!==15||law.manifest.output_ids.length!==2||law.manifest.entry!=='hit_tri_batch')
      throw Error('Unsupported original triangle batch ABI.');
    return new StudioRangefinder(law,contract);
  }
  constructor(law,contract){this.law=law;this.contract=contract;this.busy=false;this.closed=false;}
  async query(capture,{u,v,signal,spread=0}={}) {
    if(spread!==0)throw Error('Rangefinder measures one selected straight ray, not a Gaussian average.');
    if(this.closed||this.busy)throw Error('Rangefinder is closed or busy.');check(signal);
    const scene=rangefinderScene(capture),launch=rangefinderRay(capture,u,v),ray=launch.ray;
    const width=this.law.manifest.capacity,feed=new Float32Array(width*15),results=[];
    const start=performance.now();this.busy=true;
    try {
      for(let offset=0;offset<scene.count;offset+=width){
        check(signal);if(this.closed)throw Error('Rangefinder closed.');
        const n=Math.min(width,scene.count-offset);
        for(let j=0;j<n;j++){
          const values=[...ray.subarray(0,3),...ray.subarray(4,7),...scene.geometry.subarray((offset+j)*9,(offset+j+1)*9)];
          values.forEach((x,k)=>{feed[k*width+j]=x;});
        }
        const [valid,distances]=this.law.run(feed,n,width);
        if(valid.length!==n||distances.length!==n)throw Error('Original query returned an invalid result shape.');
        for(let j=0;j<n;j++){
          const t=distances[j];if(![0,1].includes(valid[j])||(valid[j]&&!Number.isFinite(t)))throw Error('Original query returned an invalid hit.');
          if(!valid[j]||t<=ray[3]||t>=ray[7])continue;
          const index=offset+j,delta=Array.from(ray.subarray(4,7),x=>x*t);
          const point=delta.map((x,k)=>x+ray[k]);
          results.push({...scene.owners[index],owner:index,rayParameter:t,
            distanceM:Math.hypot(...delta),axialDistanceM:dot(delta,launch.forward),pointM:point});
        }
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      check(signal);results.sort((a,b)=>a.rayParameter-b.rayParameter||a.owner-b.owner);
      let opaqueDistance=Infinity;
      for(const hit of results){hit.behindOpaque=hit.rayParameter>opaqueDistance;if(hit.opaque)opaqueDistance=Math.min(opaqueDistance,hit.rayParameter);}
      return {kind:'range',key:scene.key,revision:scene.revision,camera:launch.camera,hits:results,
        triangleCount:scene.count,scope:scene.scope,elapsedMs:performance.now()-start,
        kernel:this.contract.query_kernel_sha256,sourceLaw:this.contract.laws.hit_tri,
        message:`${results.length} triangle intersections along the straight ray. Choose an axial surface distance; glass refraction and apparent focus are not included.`};
    } finally {this.busy=false;}
  }
  close(){this.closed=true;}
}

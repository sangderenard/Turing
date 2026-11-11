// Studio exposure admission and connection to the existing retained-ray participant.
import {mountClientRayParticipation} from './client_ray_participation.js';
import {StudioBvhCompute} from './studio_bvh_compute.js';
export function clientIntersectionEligible({transport='fixed:3',orbitFrames=1,resolution=256,live=false,oversampling=1,tiled=false}={}) {
 return !tiled && Number(oversampling)===1 && transport==='fixed:3' && Number(orbitFrames)===1 && [128,256,512,768,1024,2048,4096].includes(Number(resolution));
}
export function clientIntersectionBase(work) {
 if(work.query_variant==='region-bvh')return '/client-ray-region/';
 if(work.query_variant==='plain-bvh')return '/client-ray-plain/';
 throw Error('This intersection variant is not available on this device.');
}
function createOriginalCandidateBackend(work,backend='webgpu') {
 if(backend!=='wasm')throw Error('Shared WebGPU is pending. Choose WebAssembly explicitly for original candidate-only kernels and host BVH traversal.');
 return StudioBvhCompute.create(new URL(clientIntersectionBase(work),import.meta.url),{backend:'wasm'});
}
export function mountJoinedPhotoParticipation({doc,parent,join,digest,mount=mountClientRayParticipation,createBackend=createOriginalCandidateBackend}) {
 if(!join||typeof join.job!=='string'||join.job!==digest||typeof join.credential!=='string'||!join.credential||!parent)throw Error('Server did not supply this exposure’s participation connection.');
 const current=mount({doc,parent,job:join.job,credential:join.credential,createBackend});
 current.element.querySelector('[data-key="enabled"]').checked=true;
 current.loop.setEnabled(true);
 return current;
}
export function createPhotoParticipation(doc, {mount=mountClientRayParticipation,createBackend=createOriginalCandidateBackend}={}) {
 const input=doc.getElementById('photo-client-intersections'),parent=doc.getElementById('photo-client-participation');
 let current=null;
 const eligible=()=>clientIntersectionEligible({transport:doc.getElementById('photo-transport')?.value,orbitFrames:doc.getElementById('photo-orbit-frames')?.value,resolution:doc.getElementById('photo-resolution')?.value,oversampling:doc.getElementById('photo-oversampling')?.value||1,tiled:!!doc.getElementById('photo-tile-size')?.value&&doc.getElementById('photo-tile-size').value!=='auto'});
 const update=()=>{if(input){input.disabled=!eligible();input.title=input.disabled?'Available for ordinary RGB still photos. Tiled and oversampled photos use server compute.':'Run retained triangle intersection batches locally using WebAssembly.';}};
 for(const id of ['photo-transport','photo-orbit-frames','photo-resolution','photo-oversampling','photo-tile-size'])doc.getElementById(id)?.addEventListener('change',update);
 update();
 const stop=()=>{current?.destroy();current=null;};
 return {requested:()=>!!input?.checked&&eligible(),stop,
  connect(result,requested){
   stop();if(!requested)return;
   current=mountJoinedPhotoParticipation({doc,parent,join:result.client_join,digest:result.digest,mount,createBackend});
  }
 };
}


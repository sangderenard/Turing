// One visible settings owner per page, shared by actual job participants.
import {participationSettings} from './client_ray_settings.js';
const panels=new WeakMap();
export function sharingControls(doc=document) {
 if(panels.has(doc))return panels.get(doc);
 let root=doc.getElementById('work-sharing-controls');
 if(!root){root=doc.createElement('section');root.id='work-sharing-controls';const heading=doc.createElement('h2');heading.textContent='Work sharing settings';root.append(heading);const jobs=doc.getElementById('marble-jobs');if(jobs)jobs.before(root);else doc.body.append(root);}
 const form=doc.createElement('div');form.className='grid';form.innerHTML=`<label class="wide">Compute engine<select data-sharing="backend"><option value="webgpu" selected disabled>WebGPU unavailable: compiler integration pending</option><option value="wasm">WebAssembly: original candidate kernels, host BVH</option></select></label><p class="wide">Choose WebAssembly explicitly to use the currently available implementation. It evaluates candidate intersections; BVH traversal is host-side. The full shared-compiler WebGPU engine is not available. No silent engine fallback.</p><label>Rays per work burst (streamed)<input data-sharing="batch" type="number" min="16" max="10000000" step="1" value="1000000"></label><label>Idle polls/second<input data-sharing="hz" type="number" min="0.1" max="10" step="0.1" value="10"></label><label>Measured wall-time duty (%)<input data-sharing="duty" type="number" min="1" max="100" value="50"></label><p class="wide">These settings apply to this page’s actual server-job participants below, including resumed jobs. A work burst streams successive retained packets; its final packet may overshoot the target. Internal queries use at most 4096 rays, not a 10-million-ray allocation. Burst size, image resolution and forward-ray sweep size are independent.</p><p class="wide">Duty adds a rest after a completed burst, based on measured work time; it is not a hard hardware-utilization cap. Large bursts may run a long time before resting. Device work uses battery and can slow the page. Pause a job’s participation below at any time; hidden tabs pause device work. Missing results remain pending. No performance savings are promised.</p><output class="wide" role="status"></output>`;
 root.append(form);
 const subscribers=new Set(),output=form.querySelector('output');let value=participationSettings();
 const read=()=>participationSettings(Object.fromEntries(['batch','hz','duty','backend'].map(key=>[key,form.querySelector(`[data-sharing="${key}"]`).value])));
 const update=()=>{try{value=read();output.textContent='Settings applied to this page’s participating jobs.';for(const fn of subscribers)fn(value,null);}catch(error){output.textContent=error.message;for(const fn of subscribers)fn(null,error);}};
 for(const key of ['batch','hz','duty','backend'])form.querySelector(`[data-sharing="${key}"]`).addEventListener('change',update);
 for(const type of ['click','input','change'])form.addEventListener(type,event=>event.stopPropagation());
 const api={get:()=>({...value}),subscribe(fn){subscribers.add(fn);return ()=>subscribers.delete(fn);}};
 panels.set(doc,api);return api;
}
if(typeof document!=='undefined')sharingControls(document);

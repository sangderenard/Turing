// Opt-in owner-job transport. No registration, open work pool, or credit system.
// The host supplies an ephemeral credential only for its own issued job.
import {participationSettings,participationRest,PARTICIPATION_KERNEL_CHUNK} from './client_ray_settings.js';
import {sharingControls} from './client_ray_controls.js';
const identity=['token','revision','sample_prefix','geometry_hash','kernel','query_kernel'];
export class ClientRayParticipation {
 constructor({job,credential,createBackend,fetcher=(...args)=>globalThis.fetch(...args),onStatus=()=>{},clock=()=>performance.now(),schedule=(f,ms)=>setTimeout(f,ms),unschedule=id=>clearTimeout(id)}) {
  if(typeof job!=='string'||!job||typeof credential!=='string'||!credential||typeof createBackend!=='function')throw Error('An originating job credential and compiled backend are required.');
  Object.assign(this,{job,credential,createBackend,fetcher,onStatus,clock,schedule,unschedule});
  this.settings=participationSettings();this.burstRays=0;this.burstWorkMs=0;this.closed=false;this.enabled=false;this.hidden=false;this.busy=false;this.generation=0;this.timer=null;this.backend=null;this.residentScene=null;this.lastSubmitted=null;this.controller=null;
 }
 configure(value){const next=participationSettings(value),changed=next.backend!==this.settings.backend;this.settings=next;if(changed)this.restart();}
 setEnabled(value){if(this.closed)return;this.enabled=Boolean(value);this.restart();}
 setHidden(value){this.hidden=Boolean(value);this.restart();}
 restart(){this.generation++;if(this.timer!==null)this.unschedule(this.timer);this.timer=null;this.controller?.abort();if(this.enabled&&!this.hidden&&!this.busy)this.timer=this.schedule(()=>this.tick(),0);}
 async request(action,body,signal){
  const timeout=new AbortController(),abort=()=>timeout.abort();
  signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
  const timer=setTimeout(abort,15000);
  try {
   const response=await this.fetcher('/api/client_rays/'+action,{method:'POST',credentials:'same-origin',cache:'no-store',signal:timeout.signal,
    headers:{'Content-Type':'application/json','X-Meowdoku-Rays':'1',Authorization:'Bearer '+this.credential},body:JSON.stringify({job:this.job,...body})});
   const value=await response.json();if(!response.ok){const error=Error(value.error||'Ray relay '+response.status);error.status=response.status;error.action=action;throw error;}return value;
  } finally {clearTimeout(timer);signal?.removeEventListener('abort',abort);}
 }
 async tick(){
  if(this.busy||!this.enabled||this.hidden)return;
  const generation=this.generation,start=this.clock();let delay=1000/this.settings.hz;
  this.busy=true;this.controller=new AbortController();
  const current=()=>generation===this.generation&&this.enabled&&!this.hidden;
  try {
   const reply=await this.request('poll',{},this.controller.signal);if(!current())return;
   if(reply.job!==this.job)throw Error('Relay returned a different job.');
   if(['committed','uncertain','canceled'].includes(reply.state)&&!(reply.state==='committed'&&reply.continuous===true)){
    this.enabled=false;this.onStatus(reply.state==='committed'?'Engine committed the returned intersections.':reply.state==='uncertain'?'Engine outcome uncertain. Stopped; no automatic retry.':'Job canceled.');return;
   }
   if(reply.state!=='pending'){this.onStatus(reply.state==='expired'?'Lease expired; original ray strip remains pending. Waiting for the engine.':`Engine state: ${reply.state}.`);return;}
   const w=reply.work;
   if(!w||identity.some(k=>typeof w[k]!=='string'||!w[k])||!Array.isArray(w.rays)||!w.rays.length||w.rays.length>(w.parent_dispatch===true?65536:4096)||!Number.isFinite(reply.remaining_ms)||reply.remaining_ms<=0)throw Error('Unsupported work identity, lease or batch size. No rays consumed.');
   if(this.lastSubmitted===w.token){this.onStatus('Result already submitted; waiting for engine confirmation.');return;}
   const expires=this.clock()+reply.remaining_ms;
   if(!this.backend&&!Array.isArray(w.triangles)){
    await this.request('scene_reset',{},this.controller.signal);
    this.onStatus('Requesting the exact scene descriptor for this device.');return;
   }
   if(this.backend&&(this.backendMode!==this.settings.backend||(this.backend.manifest.query_kernel_sha256||this.backend.manifest.source_sha256)!==w.query_kernel)){
    this.backend.close();this.backend=null;this.residentScene=null;
    if(!Array.isArray(w.triangles)){await this.request('scene_reset',{},this.controller.signal);return;}
   }
   if(!this.backend){
    const mode=this.settings.backend,created=await this.createBackend(w,mode);
    if(!current()||this.closed){created.close();return;}
    this.backend=created;this.backendMode=mode;
   }
   if(!current())return;
   if((this.backend.manifest.query_kernel_sha256||this.backend.manifest.source_sha256)!==w.query_kernel)throw Error('Compiled query kernel identity differs from the job.');
   const revision=w.query_geometry_hash?('query:'+w.query_geometry_hash+':'+w.query_kernel):(w.revision+':'+w.geometry_hash);
   const residentKey=revision+':'+w.query_kernel;
   if(this.residentScene!==residentKey){
    if(!Array.isArray(w.triangles)){await this.request('scene_reset',{},this.controller.signal);this.onStatus('Requesting the exact scene descriptor for this device.');return;}
    this.backend.setScene({revision,triangles:w.triangles,unsupportedGeometry:w.unsupportedGeometry,
     bvh_nodes:w.bvh_nodes,bvh_ids:w.bvh_ids,bvh_epsilon:w.bvh_epsilon,query_variant:w.query_variant});
    this.residentScene=residentKey;
   }
   if(reply.scene_ack_required){
    await this.request('scene_ack',{scene:w.query_geometry_hash?{query_geometry_hash:w.query_geometry_hash,query_kernel:w.query_kernel}:{revision:w.revision,geometry_hash:w.geometry_hash,query_kernel:w.query_kernel}},this.controller.signal);
    if(!current())return;
   }
   const rays=new Float32Array(w.rays.flat());if(rays.length!==w.rays.length*8)throw Error('Malformed ray batch.');
   // A retained wire packet is not the user's compute batch. Query smaller
   // chunks without submitting a partial packet or losing its exact identity.
   const result={revision,kernel:w.query_kernel,count:w.rays.length,hits:new Float32Array(w.rays.length*4),elapsedMs:0,backend:'unknown'};
   for(let offset=0;offset<w.rays.length;){
    if(!current())return;
    const count=Math.min(PARTICIPATION_KERNEL_CHUNK,this.settings.batch,w.rays.length-offset);
    const chunk=await this.backend.query(rays.subarray(offset*8,(offset+count)*8),revision,{signal:this.controller.signal});if(!current())return;
    if(chunk.revision!==revision||chunk.kernel!==w.query_kernel||chunk.count!==count||chunk.hits.length!==count*4)throw Error('Query result identity/shape mismatch.');
    result.hits.set(chunk.hits,offset*4);result.elapsedMs+=Number.isFinite(chunk.elapsedMs)?chunk.elapsedMs:0;result.backend=chunk.backend||'unknown';offset+=count;
    if(this.clock()>=expires){this.onStatus('Query missed its lease. Original rays remain pending.');return;}
   }
   if(this.clock()>=expires){this.onStatus('Query missed its lease. Original rays remain pending.');return;}
   const payload=Object.fromEntries(identity.map(k=>[k,w[k]]));if(w.query_geometry_hash)payload.query_geometry_hash=w.query_geometry_hash;payload.profile={client_compute_ms:Number.isFinite(result.elapsedMs)?result.elapsedMs:0,rays:result.count,backend:result.backend||'unknown'};payload.hits=Array.from({length:result.count},(_,i)=>Array.from(result.hits.subarray(i*4,i*4+4)));
   // Once sent, never resend on a lost response: poll authoritative status.
   this.lastSubmitted=w.token;
   let accepted;
   try {accepted=await this.request('result',{result:payload},this.controller.signal);}
   catch(error){
    if(current())this.onStatus('Submission outcome unknown; polling engine confirmation without resending.');
    return;
   }
   if(!current())return;
   if(accepted.state!=='submitted'||accepted.committed!==false){this.onStatus('Unexpected acknowledgement; polling engine confirmation without resending.');return;}
   this.burstRays+=result.count;this.burstWorkMs+=Math.max(0,this.clock()-start);
   const burstDone=this.burstRays>=this.settings.batch;
   this.onStatus(`${result.count} intersections submitted; engine validation/commit pending. Streamed burst ${this.burstRays.toLocaleString()}/${this.settings.batch.toLocaleString()} rays${burstDone?' complete; duty rest follows.':'.'} No savings measured.`);
   // Continue real packets immediately inside a burst. Never split a retained
   // parent submission merely to hit the logical target exactly.
   delay=burstDone?participationRest(this.settings,this.burstWorkMs):0;
   if(burstDone){this.burstRays=0;this.burstWorkMs=0;}
  } catch(error){
   if(current()&&error.status===429&&error.action==='poll'){
    // Poll rejection did not submit or consume work. Keep the retained job and back off.
    delay=Math.max(delay,1000);this.onStatus('Polling rate limited; waiting one second before checking the same job.');
   }else if(current()){this.enabled=false;this.onStatus(`Participation stopped: ${error.message}. ${this.lastSubmitted?'Submission outcome requires engine confirmation; do not assume it was uncommitted.':'No result was submitted by this client.'}`);}
  } finally {
   this.busy=false;this.controller=null;
   if(this.enabled&&!this.hidden)this.timer=this.schedule(()=>this.tick(),generation===this.generation?delay:0);
  }
 }
 close(){this.closed=true;this.enabled=false;this.restart();this.backend?.close();this.credential='';}
}

export function mountClientRayParticipation({doc=document,parent,job,credential,createBackend}) {
 const settings=sharingControls(doc);
 const box=doc.createElement('section');box.className='wide work-sharing-job';box.innerHTML=`<h3>Device participation · ${String(job).slice(0,10).replace(/[^a-zA-Z0-9_-]/g,'')}</h3><label><input data-key="enabled" type="checkbox"> Participate in this job</label><output role="status"></output>`;
 parent.append(box);const el=k=>box.querySelector(`[data-key="${k}"]`);
 const loop=new ClientRayParticipation({job,credential,createBackend,onStatus:text=>{box.querySelector('output').textContent=text;el('enabled').checked=loop.enabled;}});
 el('enabled').addEventListener('change',()=>loop.setEnabled(el('enabled').checked));
 loop.configure(settings.get());
 const unsubscribe=settings.subscribe((value,error)=>{if(error){loop.setEnabled(false);box.querySelector('output').textContent=error.message;el('enabled').checked=false;}else loop.configure(value);});
 for(const type of ['click','input','change'])box.addEventListener(type,event=>event.stopPropagation());
 const visibility=()=>loop.setHidden(doc.hidden),close=()=>loop.close();doc.addEventListener('visibilitychange',visibility);doc.defaultView.addEventListener('pagehide',close);visibility();
 return {loop,element:box,destroy(){unsubscribe();close();doc.removeEventListener('visibilitychange',visibility);doc.defaultView.removeEventListener('pagehide',close);box.remove();}};
}

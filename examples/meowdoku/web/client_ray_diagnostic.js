import {ClientSphereQuery} from './client_ray_kernel.js';

export function querySettings(value={}) {
  const batch=Number(value.batch??256),hz=Number(value.hz??2),duty=Number(value.duty??10);
  if(!Number.isInteger(batch)||batch<16||batch>4096||!Number.isFinite(hz)||hz<.1||hz>10||!Number.isFinite(duty)||duty<1||duty>50)
    throw new Error('Use 16–4096 rays, 0.1–10 batches/s, and 1–50% duty.');
  return {batch,hz,duty};
}
export function nextQueryDelay(settings,elapsedMs) {
  const s=querySettings(settings);
  return Math.max(0,1000/s.hz-elapsedMs,elapsedMs*(100/s.duty-1));
}
// Pinhole launch geometry only. All intersections use the compiled original kernel.
export function cameraQueryBatch(camera,count) {
  if(!Number.isInteger(count)||count<1||count>4096)throw new Error('Invalid query count.');
  const eye=camera?.eye_m,target=camera?.target_m,fov=camera?.fov_degrees;
  if(![eye,target].every(v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite))||!Number.isFinite(fov)||fov<=0||fov>=179)throw new Error('Invalid diagnostic camera.');
  const unit=v=>{const n=Math.hypot(...v);if(n<1e-12)throw new Error('Degenerate diagnostic camera.');return v.map(x=>x/n);};
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const forward=unit(target.map((v,i)=>v-eye[i])),right=unit(cross(forward,Math.abs(forward[1])>.99?[0,0,1]:[0,1,0])),up=cross(right,forward);
  const width=Math.ceil(Math.sqrt(count)),height=Math.ceil(count/width),rays=new Float32Array(count*8),extent=Math.tan(fov*Math.PI/360);
  for(let i=0;i<count;i++) {const x=(2*((i%width)+.5)/width-1)*extent,y=(1-2*(Math.floor(i/width)+.5)/height)*extent;
    const d=unit(forward.map((v,k)=>v+right[k]*x+up[k]*y));rays.set([...eye,0,...d,1000],i*8);}
  return {rays,width,height};
}

export class LocalQueryLoop {
  constructor({createBackend,getWork,onResult,onStatus=()=>{},clock=()=>performance.now(),schedule=(fn,ms)=>setTimeout(fn,ms),unschedule=id=>clearTimeout(id)}) {
    Object.assign(this,{createBackend,getWork,onResult,onStatus,clock,schedule,unschedule});
    this.settings=querySettings();this.enabled=false;this.hidden=false;this.generation=0;this.timer=null;this.backend=null;
  }
  configure(settings){this.settings=querySettings(settings);this.restart();}
  setEnabled(value){this.enabled=!!value;this.restart();}
  setHidden(value){this.hidden=!!value;this.restart();}
  restart(){this.generation++;if(this.timer!==null)this.unschedule(this.timer);this.timer=null;
    this.backend?.close();this.backend=null;
    if(this.enabled&&!this.hidden)this.timer=this.schedule(()=>this.tick(this.generation),0);
    else this.onStatus(this.hidden&&this.enabled?'Paused while page is hidden.':'Off. No local ray queries running.');}
  async tick(generation){this.timer=null;const start=this.clock();let backend=this.backend;
    try {
      if(!backend){backend=await this.createBackend();if(generation!==this.generation){backend.close();return;}this.backend=backend;}
      const work=this.getWork(this.settings.batch);backend.setScene(work.scene);
      const result=await backend.query(work.rays,work.scene.revision);
      if(generation!==this.generation)return;
      if(this.getWork(this.settings.batch).key!==work.key){this.onStatus('Scene changed; discarded old diagnostic.');}
      else this.onResult(result,work);
      this.timer=this.schedule(()=>this.tick(generation),nextQueryDelay(this.settings,Math.max(0,this.clock()-start)));
    }catch(error){if(generation!==this.generation)return;this.enabled=false;this.restart();this.onStatus(`Stopped: ${error.message}`);}
  }
  close(){this.enabled=false;this.restart();}
}

export function mountClientRayDiagnostic({doc=document,getScene,parent=doc.querySelector('aside'),createBackend=()=>ClientSphereQuery.create()}) {
  const box=doc.createElement('details');box.id='client-ray-diagnostic';box.className='wide';
  box.innerHTML=`<summary>Advanced: local sphere-distance queries</summary><p>Opt-in device GPU diagnostic only. Pinhole nearest-sphere distances, not a photograph: no triangles, rods, refraction, lighting, or server contribution. This does not reduce server work. Batches may use battery and slow other graphics.</p>
  <label><input data-control="enabled" type="checkbox"> Enable local diagnostic</label><div class="grid">
  <label>Rays per batch<input data-control="batch" type="number" min="16" max="4096" value="256" step="16"></label>
  <label>Maximum batches/second<input data-control="hz" type="number" min=".1" max="10" step=".1" value="2"></label>
  <label>Duty budget (%)<input data-control="duty" type="number" min="1" max="50" value="10"></label>
  <label>Geometry<select data-control="target"><option value="all">Shell and bubbles</option><option value="bubbles">Bubbles only (shell omitted)</option></select></label></div>
  <p>Duty schedules rests based on measured batch wall time; it is not a hard GPU utilization cap. Automatically pauses when hidden. Brighter means nearer within this batch; black means no sphere hit.</p>
  <canvas width="64" height="64" style="width:192px;max-width:100%;image-rendering:pixelated"></canvas><output role="status"></output>`;
  parent.append(box);const el=name=>box.querySelector(`[data-control="${name}"]`),out=box.querySelector('output'),canvas=box.querySelector('canvas'),ctx=canvas.getContext('2d');
  let lastKey='',revision=0;
  const work=count=>{const raw=getScene();let spheres=raw.spheres;
    if(el('target').value==='bubbles')spheres=spheres.slice(1);
    if(!spheres.length)throw new Error('No bubble spheres in this design.');
    const key=JSON.stringify([raw.camera,spheres]);if(key!==lastKey){lastKey=key;revision++;}
    return {...cameraQueryBatch(raw.camera,count),scene:{revision:String(revision),spheres},key};};
  const loop=new LocalQueryLoop({createBackend,getWork:work,onStatus:text=>{out.textContent=text;if(!loop.enabled)el('enabled').checked=false;},onResult:(result,w)=>{
    canvas.width=w.width;canvas.height=w.height;const pixels=ctx.createImageData(w.width,w.height);let hits=0,near=Infinity,far=-Infinity;
    for(let i=0;i<result.count;i++)if(result.hits[i*4+2]>.5){hits++;near=Math.min(near,result.hits[i*4]);far=Math.max(far,result.hits[i*4]);}
    for(let i=0;i<w.width*w.height;i++){const hit=i<result.count&&result.hits[i*4+2]>.5;const c=hit?Math.round(64+191*(1-(result.hits[i*4]-near)/Math.max(far-near,1e-12))):0;pixels.data.set([c,c,c,255],i*4);}
    ctx.putImageData(pixels,0,0);out.textContent=`${result.count} local rays · ${hits} sphere hits · ${result.elapsedMs.toFixed(1)} ms batch${hits?` · distance ${(near*1000).toFixed(2)}–${(far*1000).toFixed(2)} mm`:''}. Server work unchanged.`;
  }});
  for(const name of ['input','change'])box.addEventListener(name,event=>event.stopPropagation());
  el('enabled').addEventListener('change',()=>loop.setEnabled(el('enabled').checked));
  for(const name of ['batch','hz','duty'])el(name).addEventListener('change',()=>{try{loop.configure(Object.fromEntries(['batch','hz','duty'].map(k=>[k,el(k).value])));}catch(e){loop.setEnabled(false);out.textContent=e.message;}});
  el('target').addEventListener('change',()=>loop.restart());
  const visibility=()=>loop.setHidden(doc.hidden),close=()=>loop.close();doc.addEventListener('visibilitychange',visibility);doc.defaultView.addEventListener('pagehide',close);visibility();
  return {loop,element:box,destroy(){close();doc.removeEventListener('visibilitychange',visibility);doc.defaultView.removeEventListener('pagehide',close);box.remove();}};
}

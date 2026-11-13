import {StudioRangefinder} from './marble_rangefinder.js';
import {dofParameters,dofFocusAt,dofCircleRadius} from './marble_studio_dof.js';
import {FOCUS_MAX_M} from './marble_lens_controls.js';

// Gaussian-weighted image-plane rays, not Gaussian lens physics. Intersections
// remain the original compiled hit_tri law owned by StudioRangefinder.
export function autofocusSamples() {
  const samples=[];
  for(const y of [-1,0,1])for(const x of [-1,0,1])
    samples.push({u:.5+x*.3,v:.5+y*.3,weight:Math.exp(-((x*.3)**2+(y*.3)**2)/(2*.28**2))});
  return samples;
}
export function autofocusCoverage(samples,camera,focus,height,aspect) {
  const p=dofParameters({...camera,focus_distance_m:focus},{unitM:1,near:.001,far:1e6,height});
  let covered=0,total=0;
  for(const s of samples){
    const local=dofFocusAt([s.u,1-s.v],p,camera,aspect);
    const radius=Math.abs(dofCircleRadius(s.depth,{...p,focus:local}));
    covered+=s.weight*(radius<=1+1e-9?1:0);total+=s.weight;
  }
  return total?covered/total:0;
}
export function autofocusCandidates(samples,camera,{height,aspect,min=.001,max=FOCUS_MAX_M}) {
  if(!samples.length||!(min>0&&max>=min))throw Error('No measurable focus range.');
  const p=dofParameters(camera,{unitM:1,near:.001,far:1e6,height});
  if(!p.aperture)throw Error('A pinhole has no focus search.');
  const lo=1/max,hi=1/min,points=[lo,hi];
  for(const s of samples){
    // The existing tilted focus plane gives local reciprocal focus d/focus.
    const d=p.focus/dofFocusAt([s.u,1-s.v],p,camera,aspect),band=1/(p.aperture*p.pixelScale);
    for(const q of [(1/s.depth-band)/d,(1/s.depth+band)/d,1/s.depth/d])
      points.push(Math.max(lo,Math.min(hi,q)));
  }
  const sorted=[...new Set(points)].sort((a,b)=>a-b),all=[...sorted];
  for(let i=1;i<sorted.length;i++)all.push((sorted[i-1]+sorted[i])/2);
  const ranked=all.map(q=>({focus:1/q,coverage:autofocusCoverage(samples,camera,1/q,height,aspect)}));
  const best=Math.max(...ranked.map(x=>x.coverage));
  // Exact interval boundaries cover every maximum-coverage plateau for these
  // sampled rays. Image contrast selects within that optical coverage set.
  return ranked.filter(x=>x.coverage>=best-1e-12).sort((a,b)=>a.focus-b.focus);
}
export function autofocusContrast({data,width,height}) {
  if(width<3||height<3||data.length!==width*height*4)throw Error('Invalid focus feedback image.');
  const l=(x,y)=>{const i=4*(y*width+x);return .2126*data[i]+.7152*data[i+1]+.0722*data[i+2];};
  let sum=0,weight=0;
  for(let y=1;y<height-1;y++)for(let x=1;x<width-1;x++){
    const w=Math.exp(-((x/width-.5)**2+(y/height-.5)**2)/(2*.28**2));
    sum+=w*((l(x+1,y)-l(x-1,y))**2+(l(x,y+1)-l(x,y-1))**2);weight+=w;
  }
  return sum/weight;
}

export function mountStudioAutofocus({doc,canvas,getScene,getKey,photo,
  createBackend=()=>StudioRangefinder.create(new URL('./client-ray-plain/',import.meta.url)),now=()=>performance.now()}) {
  const enabled=doc.getElementById('studio-autofocus'),button=doc.getElementById('studio-autofocus-run'),status=doc.getElementById('studio-autofocus-status');
  let backend=null,creating=null,run=null,requestedFrame=false,attempted=false,stableKey=null,stableSince=0,lastCamera=null,disposed=false;
  const preview=doc.createElement('canvas');preview.width=160;preview.height=160;
  const context=preview.getContext('2d',{willReadFrequently:true});
  const say=text=>{status.textContent=text;};
  function cancel(text){requestedFrame=false;if(run){run.abort.abort();run=null;}button.textContent='Focus now';if(text)say(text);}
  async function start(camera,key){
    if(run||disposed)return;attempted=true;
    if(!doc.getElementById('studio-dof')?.checked){say('Enable Phong depth of field to search the actual preview.');return;}
    const task={abort:new AbortController(),key,camera,phase:'rays',started:now(),candidate:null};run=task;button.textContent='Cancel focus search';
    say('Autofocus: measuring nine Gaussian-weighted scene rays.');
    try {
      const capture=getScene();creating??=createBackend();backend??=await creating;if(disposed){backend.close();return;}
      const samples=[];
      for(const s of autofocusSamples()){
        if(run!==task) return;
        const result=await backend.query(capture,{...s,signal:task.abort.signal});
        const hit=result.hits.find(h=>!h.behindOpaque&&h.axialDistanceM>0);
        if(hit)samples.push({...s,depth:hit.axialDistanceM});
      }
      if(run!==task)return;
      if(key!==getKey())throw Error('Scene moved during autofocus. Run Focus now again.');
      task.candidates=autofocusCandidates(samples,camera,{height:canvas.height,aspect:canvas.width/canvas.height,min:photo.lensMinimumFocus()});
      task.samples=samples;task.height=canvas.height;task.aspect=canvas.width/canvas.height;task.refinements=0;task.index=0;task.best=null;task.phase='image';task.candidate=task.candidates[0].focus;
      say(`Autofocus: comparing ${task.candidates.length} completed Phong frames at maximum sampled depth coverage.`);
    }catch(error){if(run===task)cancel(`Autofocus stopped: ${error.message}`);}
  }
  function manual(event){
    if(event.target?.closest?.('#studio-autofocus-controls'))return;
    // Any explicit user control wins over an automatic focus search.
    attempted=true;enabled.checked=false;cancel('Manual controls: automatic focus released. Use Focus now to search again.');
  }
  const clicked=event=>{if(event.target?.closest?.('button'))manual(event);};
  const hidden=()=>{if(doc.hidden)cancel('Autofocus paused because the page is hidden.');};
  const lost=()=>cancel('Autofocus stopped: WebGL context lost.');
  doc.addEventListener('visibilitychange',hidden);canvas.addEventListener('webglcontextlost',lost);
  doc.addEventListener('click',clicked);
  doc.addEventListener('input',manual);doc.addEventListener('change',manual);
  canvas.addEventListener('pointerdown',manual);canvas.addEventListener('wheel',manual);
  const toggle=()=>{cancel();attempted=!enabled.checked;stableSince=now();};
  enabled.addEventListener('change',toggle);
  const requested=()=>{if(run||requestedFrame){cancel('Autofocus canceled; original focus retained.');return;}requestedFrame=true;button.textContent='Cancel focus search';say('Autofocus: waiting for the next completed Studio frame.');};
  button.addEventListener('click',requested);
  return {
    camera(camera){lastCamera=camera;return run?.candidate?{...camera,focus_distance_m:run.candidate}:camera;},
    frame(camera){
      if(disposed)return;
      const key=getKey();
      if(requestedFrame){requestedFrame=false;start(camera,key);return;}
      if(run){
        const task=run;
        if(key!==task.key||now()-task.started>30000){cancel('Autofocus stopped: scene changed or the 30-second search budget expired.');return;}
        if(task.phase!=='image'||camera.focus_distance_m!==task.candidate)return;
        try {
          if(!context)throw Error('Phong image feedback is unavailable.');
          const scale=160/Math.max(canvas.width,canvas.height),w=Math.max(3,Math.round(canvas.width*scale)),h=Math.max(3,Math.round(canvas.height*scale));
          if(preview.width!==w)preview.width=w;
          if(preview.height!==h)preview.height=h;
          context.drawImage(canvas,0,0,w,h);
          const contrast=autofocusContrast(context.getImageData(0,0,preview.width,preview.height)),candidate=task.candidates[task.index];
          if(!task.best||contrast>task.best.contrast||(contrast===task.best.contrast&&Math.abs(candidate.focus-task.camera.focus_distance_m)<Math.abs(task.best.focus-task.camera.focus_distance_m)))task.best={...candidate,contrast};
          task.index++;
          if(task.index<task.candidates.length){task.candidate=task.candidates[task.index].focus;return;}
          if(task.refinements<2){
            task.refinements++;
            const sorted=task.candidates.map(c=>1/c.focus).sort((a,b)=>a-b),q=1/task.best.focus,i=sorted.indexOf(q);
            const next=[sorted[i-1],sorted[i+1]].filter(Number.isFinite).map(n=>1/((n+q)/2))
              .map(focus=>({focus,coverage:autofocusCoverage(task.samples,task.camera,focus,task.height,task.aspect)}))
              .filter(c=>c.coverage>=task.best.coverage-1e-12&&!task.candidates.some(x=>x.focus===c.focus));
            if(next.length){task.candidates.push(...next);task.candidate=task.candidates[task.index].focus;return;}
          }
          const best=task.best;run=null;button.textContent="Focus now";
          if(!(best.contrast>0)){say("No image contrast for autofocus; original focus retained.");return;}
          // Only the winner changes authored camera settings. Profile minimum
          // focus and fixed aperture remain validated by the existing controls.
          photo.focusLensAt(best.focus,{camera:task.camera,custom:doc.getElementById('photo-lens').value==='unchanged'});
          say(`Focus ${(best.focus*1000).toFixed(2)} mm; ${(best.coverage*100).toFixed(1)}% weighted sampled surface coverage within 1 px blur radius. Phong contrast search complete. Straight-ray / opaque-depth approximation, not refractive autofocus.`);
        }catch(error){cancel(`Autofocus stopped: ${error.message}`);}
        return;
      }
      if(key!==stableKey){stableKey=key;stableSince=now();}
      if(enabled.checked&&!attempted&&now()-stableSince>=750)start(lastCamera,key);
    },
    cancel,
    dispose(){disposed=true;cancel();backend?.close();enabled.removeEventListener('change',toggle);button.removeEventListener('click',requested);doc.removeEventListener('visibilitychange',hidden);canvas.removeEventListener('webglcontextlost',lost);doc.removeEventListener('click',clicked);doc.removeEventListener('input',manual);doc.removeEventListener('change',manual);canvas.removeEventListener('pointerdown',manual);canvas.removeEventListener('wheel',manual);preview.width=preview.height=1;}
  };
}

import {StudioRangefinder,rangefinderKey} from './marble_rangefinder.js';

// Separate selected-hit/apply boundary. No focus is changed merely by measuring.
export function createRangefinderPanel({doc,photo,getScene,createBackend=()=>StudioRangefinder.create(new URL('./client-ray-plain/',import.meta.url))}) {
  const make=(tag,text)=>{const e=doc.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
  const root=make('section');root.id='camera-rangefinder';root.style.cssText='display:grid;gap:5px';
  const note=make('p','Straight pinhole ray through the current Studio triangle mesh. Every triangle hit is retained, including coincident and hidden surfaces. These are geometric ranges, not refracted apparent depth or focal length. Distances are measured from this displayed Studio view; a held or different Exposure camera needs its own measurement.');
  note.style.cssText='font-size:11px;margin:0';
  const label=make('label','Choose intersection '),select=make('select');select.id='camera-range-hit';select.disabled=true;label.append(select);
  const customLabel=make('label'),custom=make('input');custom.type='checkbox';custom.id='camera-range-custom';
  customLabel.append(custom,make('span',' Explicitly switch to Custom analytical, preserving framing and pupil'));
  const apply=make('button','Send selected axial distance to lens');apply.type='button';apply.id='camera-range-apply';apply.disabled=true;
  const status=make('output','Choose Mesh rangefinder, then a point in Studio.');status.id='camera-range-status';status.setAttribute('role','status');
  root.append(note,label,customLabel,apply,status);
  let pending=null,backend=null,creating=null,disposed=false;
  const clear=()=>{pending=null;select.replaceChildren();select.disabled=true;apply.disabled=true;};
  const read=async request=>{
    if(typeof getScene!=='function')throw Error('Current Studio capture is unavailable.');
    const capture=getScene();
    if(!backend){creating??=createBackend();try{backend=await creating;}finally{creating=null;}}
    if(disposed){backend?.close();throw Error('Rangefinder closed.');}
    const result=await backend.query(capture,request);
    if(rangefinderKey(getScene())!==result.key)throw Error('Studio camera or geometry changed; measure the ray again.');
    return result;
  };
  const accept=result=>{
    clear();pending=result;
    result.hits.forEach((hit,index)=>{
      const e=make('option',`${index+1}. axial ${(hit.axialDistanceM*1000).toFixed(3)} mm; ray ${(hit.distanceM*1000).toFixed(3)} mm | ${hit.partId} / triangle ${hit.triangle} / material ${hit.material}${hit.behindOpaque?' [behind opaque surface]':''}${hit.opaque?' [opaque]':' [transmitting surface]'}`);
      e.value=String(index);select.append(e);
    });
    select.value='0';select.disabled=!result.hits.length;apply.disabled=!result.hits.length;
    status.textContent=`${result.hits.length} hits from ${result.triangleCount.toLocaleString()} triangles. Original compiled query: ${result.elapsedMs.toFixed(1)} ms. Select the desired surface before applying.`;
  };
  const click=()=>{
    try {
      if(!pending)throw Error('Measure a Studio ray first.');
      if(rangefinderKey(getScene())!==pending.key)throw Error('Studio camera or geometry changed; measure the ray again.');
      const hit=pending.hits[Number(select.value)];if(!hit||!(hit.axialDistanceM>0))throw Error('Choose an intersection in front of the camera.');
      if(!custom.checked&&doc.getElementById('photo-lens')?.value==='unchanged')
        throw Error('Choose a lens profile, or explicitly enable Custom analytical, before setting focus.');
      const camera=photo.applyLensCamera(pending.camera);
      photo.focusLensAt(hit.axialDistanceM,{camera,custom:custom.checked});
      status.textContent=`Sent displayed Studio axial surface distance ${(hit.axialDistanceM*1000).toFixed(3)} mm to lens focus. Refraction may move apparent focus; source profile limits still apply.`;
      clear();
    }catch(error){status.textContent=error.message;}
  };
  apply.addEventListener('click',click);
  return {root,read,accept,invalidate(){clear();},
    dispose(){disposed=true;backend?.close();creating?.then(b=>b.close(),()=>{});apply.removeEventListener('click',click);root.remove();}};
}

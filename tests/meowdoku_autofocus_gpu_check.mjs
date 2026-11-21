import {pixels} from './meowdoku_studio_dof_gpu_check.mjs';
import {autofocusSamples,autofocusCandidates,autofocusContrast} from '../examples/meowdoku/web/marble_autofocus.js';
document.getElementById('autofocus').onclick=()=>{try{
const camera={focus_distance_m:.05,aperture_radius_m:.006,fov_degrees:50};
const samples=autofocusSamples().map(s=>({...s,depth:s.u<.33?.05:s.u>.66?.2:.1}));
const candidates=autofocusCandidates(samples,camera,{height:192,aspect:2});
let best=null;const measured=[];
for(const c of candidates){const output=pixels(camera.aperture_radius_m,c.focus);const contrast=autofocusContrast({data:output.bytes,width:384,height:192});measured.push({...c,contrast});if(!best||contrast>best.contrast)best={...c,contrast};}
if(!(best.focus>.09&&best.focus<.12&&best.contrast>0))throw Error('Feedback did not select the dominant central depth.');
pixels(camera.aperture_radius_m,best.focus);
document.getElementById('result').textContent=JSON.stringify({passed:true,actualGpu:true,best,candidates:measured,limit:'Existing Studio DOF shader and authored three-plane QA geometry. Not a full Studio/browser integration acceptance.'},null,2);
}catch(error){document.getElementById('result').textContent=error.stack;}};

import {mountCalibrationWatch} from './marble_calibration.js';

const query=new URLSearchParams(location.search),digest=query.get('digest'),lane=Number(query.get('lane'));
const status=document.getElementById('watch-status');
if(!/^[a-f0-9]{64}$/.test(digest??'')||!Number.isSafeInteger(lane)||lane<0||lane>31){
  status.textContent='Open this page from a running Studio calibration job.';
}else{
  const viewer=mountCalibrationWatch(document,document.getElementById('watch'),
    {fixedLane:lane,laneCount:Math.max(2,lane+1)});
  let stopped=false;
  async function poll(){
    if(stopped)return;
    try{
      const response=await fetch(`/api/marble_photo?digest=${digest}`,{cache:'no-store'});
      const value=await response.json().catch(()=>({}));
      if(!response.ok)throw Error(value.error||`HTTP ${response.status}`);
      viewer.update(value);
      status.textContent=`Sensor lane ${lane} · ${value.state} · Studio tick ${value.epoch??0}`;
      if(['complete','error','canceled'].includes(value.state)){
        stopped=true;
        if(value.error)status.textContent+=` · ${value.error}`;
        viewer.stop();return;
      }
    }catch(error){status.textContent=`Waiting for calibration: ${error.message||error}`;}
    setTimeout(poll,1000);
  }
  poll();
}

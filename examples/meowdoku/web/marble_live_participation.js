import {mountJoinedPhotoParticipation} from './marble_photo_participation.js';
// User-enabled cooperative Live uses the retained parent dispatch.
export const LIVE_CLIENT_INTERSECTIONS_READY = true;
export function createLiveParticipation(doc,{ready=LIVE_CLIENT_INTERSECTIONS_READY,mount=mountJoinedPhotoParticipation}={}) {
 const input=doc.getElementById('live-client-intersections'),parent=doc.getElementById('live-client-participation'),status=doc.getElementById('live-client-status');
 let current=null,digest=null;
 if(input){input.disabled=!ready;input.title=ready?'Applies when Live next starts or its scene is rebuilt. Pause device work in the pane without stopping Live.':'Retained Live history validation is not yet enabled.';}
 const report=message=>{if(status)status.textContent=message;};
 const stop=()=>{current?.destroy();current=null;digest=null;};
 return {requested:()=>ready&&!!input?.checked,stop,
  connect(result,requested){
   if(!requested){stop();return;}
   if(current&&digest===result.digest)return;
   try{const next=mount({doc,parent,join:result.client_join,digest:result.digest});stop();current=next;digest=result.digest;report('Cooperative Live connected. Pause device work below without clearing its ray history.');}
   catch(error){report(`Live device participation unavailable: ${error.message}. The job remains available in My jobs.`);}
  }
 };
}

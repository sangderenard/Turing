// Browser transport for Spectral's original RoutingGridView. The PNG and
// RoutingGraph edits come from the same server-side Pygame panel.
import {LIGHTING_BUSES} from './marble_lighting_racks.js';

const STORE='studio.camera-trigger-graph.v1';
export function mountMarbleTriggerGrid({doc,getReceivers}) {
  const root=doc.getElementById('camera-trigger-grid');
  const picture=doc.getElementById('camera-trigger-grid-image');
  const status=doc.getElementById('camera-trigger-grid-status');
  const values=doc.getElementById('camera-trigger-values');
  const fire=doc.getElementById('camera-trigger-fire');
  const previewSpeed=doc.getElementById('camera-trigger-preview-speed');
  let graph=null,pulse=null,started=0,playbackRate=50,chain=Promise.resolve(),opened=false;
  const request=async body=>{
    const response=await fetch('/api/marble_camera/trigger_grid',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const value=await response.json();
    if(!response.ok)throw Error(value.error||`routing HTTP ${response.status}`);
    return value;
  };
  const show=value=>{
    if(value.image)picture.src=value.image;
    if(value.graph){graph=value.graph;localStorage.setItem(STORE,JSON.stringify(graph));}
  };
  const enqueue=body=>{chain=chain.then(()=>request(body).then(show)).catch(error=>{status.textContent=error.message;});return chain;};
  const open=()=>{
    graph=null;opened=false;fire.disabled=true;
    const receivers=getReceivers();
    const buses=LIGHTING_BUSES.map(bus=>`rack:${bus.id}`);
    let saved=null;
    try {saved=JSON.parse(localStorage.getItem(STORE));}
    catch {localStorage.removeItem(STORE);}
    return enqueue({action:'open',receivers,buses,...(saved?{graph:saved}:{})}).then(()=>{
      opened=!!graph;fire.disabled=!opened;
      if(opened)status.textContent='Spectral routing grid ready. Columns send to rows; Mix sets amplitude and Delay sets propagation time.';
    });
  };
  root.addEventListener('toggle',()=>{if(root.open&&!opened)open();});
  doc.getElementById('camera-trigger-reset').addEventListener('click',()=>{
    localStorage.removeItem(STORE);graph=null;opened=false;open();
  });
  const point=event=>{const box=picture.getBoundingClientRect();return {
    x:Math.max(0,Math.min(959,Math.floor((event.clientX-box.left)*960/box.width))),
    y:Math.max(0,Math.min(619,Math.floor((event.clientY-box.top)*620/box.height)))};};
  const pointer=(name,event)=>{if(!opened)return;event.preventDefault();
    enqueue({action:'event',name,...point(event),button:event.button===2?3:1});};
  picture.addEventListener('pointerdown',event=>{picture.setPointerCapture(event.pointerId);pointer('down',event);});
  picture.addEventListener('pointerup',event=>pointer('up',event));
  picture.addEventListener('pointermove',event=>{if(event.buttons)pointer('move',event);});
  picture.addEventListener('contextmenu',event=>event.preventDefault());
  picture.addEventListener('wheel',event=>{if(!opened)return;event.preventDefault();enqueue({action:'event',name:'wheel',wheel_y:-Math.sign(event.deltaY)});},{passive:false});
  fire.addEventListener('click',()=>{chain=chain.then(async()=>{
    const value=await request({action:'pulse'});show(value);pulse=value;playbackRate=Number(previewSpeed.value);started=performance.now();
    status.textContent=`Manual flash replaying at ${playbackRate}× slower in Phong; exposure timing is unchanged.`;
  }).catch(error=>{status.textContent=error.message;});});
  open();
  return {
    graph:()=>{
      if(!opened||!graph)return undefined;
      const current=["Sync","Flash","Gate",...LIGHTING_BUSES.map(bus=>`rack:${bus.id}`),...getReceivers()];
      if(JSON.stringify(current)!==JSON.stringify(graph.nodes))
        throw Error('Light receivers changed. Reopen camera flash routing and reset or reconnect the current lights.');
      return structuredClone(graph);
    },
    levels:now=>{
      if(!pulse)return null;
      const index=Math.floor((now-started)/(pulse.step_ms*playbackRate));
      if(index<0||index>=pulse.duration_ms){pulse=null;values.textContent='Manual flash complete.';return null;}
      const active=new Set((graph?.edges??[]).filter(edge=>Math.abs(edge.w)>1e-12).map(edge=>edge.dst)
        .filter(key=>Object.hasOwn(pulse.signals,key)));
      const gains=Object.fromEntries(Object.entries(pulse.signals).map(([key,series])=>[key,series[index]??0]));
      values.textContent=`Manual +${index} ms · `+[...active].map(key=>`${key} ×${gains[key].toFixed(2)}`).join(' · ');
      return {active,gains,shopCutoff:index<pulse.shop_cutoff_end_ms};
    }
  };
}

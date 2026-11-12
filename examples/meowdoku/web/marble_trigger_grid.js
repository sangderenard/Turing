// Browser transport for Spectral's original RoutingGridView. The PNG and
// RoutingGraph edits come from the same server-side Pygame panel.
import {LIGHTING_BUSES} from './marble_lighting_racks.js';

const STORE='studio.camera-trigger-graph.v1';
const GENERATOR_STORE='studio.camera-trigger-generators.v1';
const GENERATOR_KEYS=['mod:1','mod:2','mod:3'];
const DEFAULT_GENERATOR=key=>({key,enabled:false,kind:'sine',frequency_hz:30,phase_deg:0,
  tension:1,attack_ms:0,decay_ms:0,sustain:1,hold_ms:20,release_ms:0,
  repeat_count:1,repeat_interval_ms:20,level:1});
const GENERATOR_FIELDS=[
  ['kind','Shape','choice',['sine','warped','harmonic','ramp_up','ramp_down','triangle']],
  ['frequency_hz','Frequency (Hz)','number',0,200,1],
  ['phase_deg','Phase (degrees)','number',-360,360,1],
  ['tension','Tension','number',0,4,0.1],
  ['level','Amplitude','number',0,2,0.05],
  ['repeat_count','Triggers','number',1,16,1],
  ['repeat_interval_ms','Trigger spacing (ms)','number',1,200,1],
  ['attack_ms','Attack (ms)','number',0,200,1],
  ['decay_ms','Decay (ms)','number',0,200,1],
  ['sustain','Sustain level','number',0,1,0.05],
  ['hold_ms','Sustain time (ms)','number',0,200,1],
  ['release_ms','Release (ms)','number',0,200,1],
];
export function mountMarbleTriggerGrid({doc,getReceivers}) {
  const root=doc.getElementById('camera-trigger-grid');
  const picture=doc.getElementById('camera-trigger-grid-image');
  const status=doc.getElementById('camera-trigger-grid-status');
  const values=doc.getElementById('camera-trigger-values');
  const fire=doc.getElementById('camera-trigger-fire');
  const previewSpeed=doc.getElementById('camera-trigger-preview-speed');
  const generatorPanel=doc.getElementById('camera-trigger-generators');
  let graph=null,pulse=null,started=0,playbackRate=50,chain=Promise.resolve(),opened=false;
  let generatorSpecs=GENERATOR_KEYS.map(DEFAULT_GENERATOR);
  try {
    const saved=JSON.parse(localStorage.getItem(GENERATOR_STORE));
    if(Array.isArray(saved)&&saved.length===3&&saved.every((item,i)=>item.key===GENERATOR_KEYS[i]))
      generatorSpecs=saved;
  } catch {localStorage.removeItem(GENERATOR_STORE);}
  const generatorInputs=[];
  for(const [i,key] of GENERATOR_KEYS.entries()){
    const box=doc.createElement('fieldset');
    const legend=doc.createElement('legend');legend.textContent=`${key} · camera-triggered modulator`;box.append(legend);
    const fields={};
    const enable=doc.createElement('label');enable.textContent='Enabled ';
    const toggle=doc.createElement('input');toggle.type='checkbox';toggle.checked=generatorSpecs[i].enabled;
    enable.append(toggle);box.append(enable);fields.enabled=toggle;
    for(const [name,label,type,a,b,step] of GENERATOR_FIELDS){
      const wrapper=doc.createElement('label');wrapper.textContent=label+' ';
      const input=doc.createElement(type==='choice'?'select':'input');
      if(type==='choice')for(const choice of a){
        const option=doc.createElement('option');option.value=choice;option.textContent=choice.replaceAll('_',' ');input.append(option);
      } else {input.type='number';input.min=a;input.max=b;input.step=step;}
      input.value=generatorSpecs[i][name];wrapper.append(input);box.append(wrapper);fields[name]=input;
    }
    generatorPanel.append(box);generatorInputs.push(fields);
  }
  const currentGenerators=()=>generatorInputs.map((inputs,i)=>Object.fromEntries([
    ['key',GENERATOR_KEYS[i]],['enabled',inputs.enabled.checked],
    ...GENERATOR_FIELDS.map(([name,,type])=>[name,type==='choice'?inputs[name].value:Number(inputs[name].value)])]));
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
    return enqueue({action:'open',receivers,buses,generators:currentGenerators(),...(saved?{graph:saved}:{})}).then(()=>{
      opened=!!graph;fire.disabled=!opened;
      if(opened)status.textContent='Spectral routing grid ready. Columns send to rows; Mix sets amplitude and Delay sets propagation time.';
    });
  };
  root.addEventListener('toggle',()=>{if(root.open&&!opened)open();});
  doc.getElementById('camera-trigger-reset').addEventListener('click',()=>{
    localStorage.removeItem(STORE);graph=null;opened=false;open();
  });
  generatorPanel.addEventListener('change',()=>{
    generatorSpecs=currentGenerators();
    localStorage.setItem(GENERATOR_STORE,JSON.stringify(generatorSpecs));
    if(opened)enqueue({action:'configure',generators:generatorSpecs});
  });
  const point=event=>{const box=picture.getBoundingClientRect();return {
    x:Math.max(0,Math.min(959,Math.floor((event.clientX-box.left)*960/box.width))),
    y:Math.max(0,Math.min(619,Math.floor((event.clientY-box.top)*620/box.height)))};};
  const pointer=(name,event)=>{if(!opened)return;event.preventDefault();
    enqueue({action:'event',name,...point(event),button:event.button===2?3:1,shift:event.shiftKey});};
  picture.addEventListener('pointerdown',event=>{picture.setPointerCapture(event.pointerId);pointer('down',event);});
  picture.addEventListener('pointerup',event=>pointer('up',event));
  picture.addEventListener('pointermove',event=>{if(event.buttons)pointer('move',event);});
  picture.addEventListener('contextmenu',event=>event.preventDefault());
  picture.addEventListener('wheel',event=>{if(!opened)return;event.preventDefault();enqueue({action:'event',name:'wheel',wheel_y:-Math.sign(event.deltaY),shift:event.shiftKey});},{passive:false});
  fire.addEventListener('click',()=>{chain=chain.then(async()=>{
    const value=await request({action:'pulse'});show(value);pulse=value;playbackRate=Number(previewSpeed.value);started=performance.now();
    status.textContent=`Manual flash replaying at ${playbackRate}× slower in Phong; exposure timing is unchanged.`;
  }).catch(error=>{status.textContent=error.message;});});
  open();
  return {
    graph:()=>{
      if(!opened||!graph)return undefined;
      const current=["Sync","Flash","Gate",...LIGHTING_BUSES.map(bus=>`rack:${bus.id}`),...getReceivers(),...GENERATOR_KEYS];
      if(JSON.stringify(current)!==JSON.stringify(graph.nodes))
        throw Error('Light receivers changed. Reopen camera flash routing and reset or reconnect the current lights.');
      return structuredClone(graph);
    },
    generators:()=>structuredClone(currentGenerators()),
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

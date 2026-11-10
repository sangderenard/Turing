// Camera input arbitration and reticle UI only. A capability is operative only
// when its real reader callback and a matching surface are both supplied.
import {validateSpotSpread} from './marble_spot_sample.js';

const labels={hand:'Hand / drag',range:'Mesh rangefinder',focus:'Spot focus',exposure:'Exposure point',white:'White point'};
const fallback={focus:'Full-scene refractive focus reader is not available. No focus measurement will be fabricated.',
  range:'The original compiled all-triangle range reader is not available.',
  exposure:'Select a completed profiled HDR photo to sample exposure.',white:'Select a completed profiled HDR photo to sample a neutral point.'};

export function createCameraToolbox({doc=document,mount=doc.getElementById('photo-controls'),surfaces=[],
    focus,range,sample,capabilities={},onResult=()=>{},onStatus=()=>{},onInvalidate=()=>{}}={}) {
  if(!mount)throw new Error('Camera toolbox needs an exposure-controls mount.');
  if(doc.getElementById('camera-toolbox'))throw new Error('Camera toolbox is already mounted.');
  const make=(tag,text)=>{const el=doc.createElement(tag);if(text!==undefined)el.textContent=text;return el;};
  const root=make('section');root.id='camera-toolbox';root.className='wide camera-toolbox';
  root.setAttribute('aria-label','Camera tools');
  root.style.cssText='grid-column:1 / -1;border:1px solid #536b82;border-radius:6px;padding:8px;display:grid;gap:6px';
  const toolbar=make('div');toolbar.setAttribute('role','toolbar');toolbar.setAttribute('aria-label','Camera pointer tool');
  toolbar.style.cssText='display:flex;flex-wrap:wrap;gap:4px';
  const buttons=Object.fromEntries(Object.entries(labels).map(([key,label])=>{
    const button=make('button',label);button.id=`camera-tool-${key}`;button.type='button';
    button.setAttribute('aria-pressed',String(key==='hand'));toolbar.append(button);return [key,button];
  }));
  const spreadLabel=make('label','Reticle spread '),spreadSelect=make('select');spreadSelect.id='camera-tool-spread';
  for(const [value,label] of [['0','Point (one accumulated pixel for HDR)'],['.01','Gaussian σ 1%'],['.03','Gaussian σ 3%'],['.1','Gaussian σ 10%'],['.25','Gaussian σ 25%'],['custom','Custom Gaussian']]){
    const option=make('option',label);option.value=value;spreadSelect.append(option);
  }
  spreadSelect.value='0';spreadLabel.append(spreadSelect);
  const sigmaLabel=make('label','Gaussian σ (% of shorter image side) '),sigma=make('input');sigma.type='number';
  sigma.id='camera-tool-sigma';sigma.min='.1';sigma.max='50';sigma.step='.1';sigma.value='3';sigmaLabel.hidden=true;sigmaLabel.append(sigma);
  const status=make('output','Hand / drag is active.');status.id='camera-tool-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  const availability=make('p');availability.id='camera-tool-availability';availability.style.cssText='font-size:11px;margin:0';
  const note=make('p','HDR samples are existing accumulated pixels, not individual rays. Gaussian sampling is center-weighted and clipped to the image. White point assumes the selected subject should be neutral.');note.style.cssText='font-size:11px;margin:0';
  root.append(toolbar,spreadLabel,sigmaLabel,status,availability,note);mount.prepend(root);
  const reticle=make('div');reticle.setAttribute('aria-hidden','true');reticle.hidden=true;
  reticle.style.cssText='position:fixed;pointer-events:none;z-index:10000;border:1px solid #fff;box-shadow:0 0 0 1px #111;border-radius:50%;transform:translate(-50%,-50%);color:#fff;text-shadow:0 0 2px #000;display:grid;place-items:center';
  reticle.textContent='+';doc.body.append(reticle);
  let tool='hand',disposed=false,generation=0,pending=null,anchor=null;
  const listeners=[],cursors=new Map(surfaces.map(s=>[s.element,s.element.style.cursor]));
  const listen=(el,type,fn,opts)=>{el?.addEventListener(type,fn,opts);listeners.push([el,type,fn,opts]);};
  const report=text=>{status.textContent=text;onStatus(text);};
  const abort=()=>{generation++;pending?.abort();pending=null;onInvalidate();};
  const allowed=(surface,kind)=>surface.kinds?.includes(kind);
  const capability=kind=>{
    const c=capabilities[kind],reader=kind==='focus'?focus:kind==='range'?range:sample;
    return {available:c?.available===true&&typeof reader==='function'&&surfaces.some(s=>allowed(s,kind)),reason:c?.reason||fallback[kind]};
  };
  const spread=()=>{
    if(tool==='range')return 0;
    if(spreadSelect.value==='custom'&&(sigma.value.trim()===''||!Number.isFinite(Number(sigma.value))||Number(sigma.value)<.1))
      throw new RangeError('Custom Gaussian σ must be 0.1–50% of the shorter image side.');
    return validateSpotSpread(spreadSelect.value==='custom'?Number(sigma.value)/100:Number(spreadSelect.value));
  };
  const hide=()=>{anchor=null;reticle.hidden=true;reticle.style.display='none';};
  const bounds=surface=>surface.bounds?.()??surface.element.getBoundingClientRect();
  const show=(surface,u,v)=>{
    const rect=bounds(surface),s=spread(),size=Math.max(18,6*s*Math.min(rect.width,rect.height));
    anchor={surface,u,v};reticle.hidden=false;reticle.style.display='grid';reticle.style.left=`${rect.left+u*rect.width}px`;
    reticle.style.top=`${rect.top+v*rect.height}px`;reticle.style.width=reticle.style.height=`${size}px`;
    reticle.title=s?'Circle shows 3σ radius; sampling uses a clipped square support.':'One sample location';
  };
  const refresh=()=>{
    const unavailable=[];
    for(const kind of ['range','focus','exposure','white']){
      const c=capability(kind);buttons[kind].disabled=!c.available;buttons[kind].title=c.available?(capabilities[kind].description||labels[kind]):c.reason;
      if(!c.available)unavailable.push(`${labels[kind]}: ${c.reason}`);
    }
    availability.textContent=unavailable.join(' ');
    for(const [kind,button]of Object.entries(buttons))button.setAttribute('aria-pressed',String(kind===tool));
    for(const surface of surfaces)surface.element.style.cursor=tool==='hand'?cursors.get(surface.element):(allowed(surface,tool)?'crosshair':'not-allowed');
    spreadSelect.disabled=tool==='hand'||tool==='range';sigma.disabled=tool==='hand'||tool==='range';
  };
  const select=kind=>{
    if(disposed||!(kind in labels))return false;
    if(kind!=='hand'&&!capability(kind).available){report(capability(kind).reason);return false;}
    abort();hide();tool=kind;refresh();
    report(kind==='hand'?'Hand / drag is active.':`${labels[kind]}: choose a point on ${surfaces.filter(s=>allowed(s,kind)).map(s=>s.label||'the matching image').join(' or ')}.`);
    return true;
  };
  for(const [kind,button]of Object.entries(buttons))listen(button,'click',()=>select(kind));
  const spreadChanged=()=>{abort();sigmaLabel.hidden=spreadSelect.value!=='custom';try{spread();sigma.setCustomValidity?.('');if(anchor)show(anchor.surface,anchor.u,anchor.v);}catch(error){hide();sigma.setCustomValidity?.(error.message);report(error.message);}};
  listen(spreadSelect,'change',spreadChanged);listen(sigma,'input',spreadChanged);
  const suppress=event=>{event.preventDefault();event.stopImmediatePropagation();};
  const measure=async(surface,event)=>{
    if(disposed||tool==='hand')return;
    suppress(event);
    if(doc.hidden)return;
    if(event.button!==undefined&&event.button!==0)return;
    if(event.isPrimary===false)return;
    if(!allowed(surface,tool)){report(`Use ${labels[tool]} on its matching image, not this view.`);return;}
    abort();const token=generation,kind=tool,controller=new AbortController();pending=controller;
    try{
      const c=capability(kind);if(!c.available)throw new Error(c.reason);
      const rect=bounds(surface);
      if(!(rect.width>0&&rect.height>0))throw new Error('The selected image is not visible.');
      const u=(event.clientX-rect.left)/rect.width,v=(event.clientY-rect.top)/rect.height;
      if(u<0||u>1||v<0||v>1){report('Choose a point inside the displayed image, not its letterbox.');return;}
      const revision=surface.revision?.(),s=spread();show(surface,u,v);report(`Reading ${labels[kind].toLowerCase()}…`);
      const request={kind,u,v,spread:s,signal:controller.signal};
      const result=await (kind==='focus'?focus(request):kind==='range'?range(request):sample(request));
      if(disposed||token!==generation||controller.signal.aborted)return;
      if(surface.revision?.()!==revision)throw new Error('The image or scene changed; choose the point again.');
      await onResult(result,request);
      if(disposed||token!==generation||controller.signal.aborted)return;
      report(result?.message||`${labels[kind]} read. ${result?.sampling||''}`.trim());
    }catch(error){if(!disposed&&token===generation&&!controller.signal.aborted)report(error.message||String(error));}
    finally{if(token===generation)pending=null;}
  };
  for(const surface of surfaces){
    // Capture-phase arbitration precedes existing Studio orbit handlers.
    listen(surface.element,'pointerdown',event=>{void measure(surface,event);},true);
    for(const type of ['pointermove','pointerup','pointercancel','click','dblclick','wheel'])
      listen(surface.element,type,event=>{if(tool!=='hand'){suppress(event);if(type==='pointercancel'){abort();hide();}}}, {capture:true,passive:false});
  }
  listen(doc,'keydown',event=>{if(event.key==='Escape'&&tool!=='hand'){event.preventDefault();select('hand');}});
  listen(doc,'visibilitychange',()=>{if(doc.hidden){abort();hide();}});
  listen(doc.defaultView,'blur',()=>{abort();hide();});
  for(const type of ['scroll','resize'])listen(doc.defaultView,type,()=>{if(anchor){try{show(anchor.surface,anchor.u,anchor.v);}catch{hide();}}},true);
  hide();refresh();
  return {root,select,get tool(){return tool;},get pending(){return !!pending;},
    invalidate(reason='Image changed; choose the point again.'){abort();hide();if(tool!=='hand')report(reason);},
    setCapabilities(next){capabilities=next;abort();if(tool!=='hand'&&!capability(tool).available)select('hand');else refresh();},
    dispose(){if(disposed)return;disposed=true;abort();listeners.forEach(([el,type,fn,opts])=>el?.removeEventListener(type,fn,opts));
      for(const [el,cursor]of cursors)el.style.cursor=cursor;reticle.remove();root.remove();},
  };
}

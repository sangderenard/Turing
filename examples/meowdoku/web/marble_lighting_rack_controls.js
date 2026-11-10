/** Shortcut editor only: the lighting owner validates and atomically commits state.
 * Null values disable overrides; authored fixture values are never edited here. */
export function createLightingRackControls(container,{
  getState,setState,getMaxDropFeet=()=>30,onBuildRacks,
}={}) {
  if(typeof getState!=='function'||typeof setState!=='function')throw new TypeError('Rack controls require state accessors');
  const doc=container.ownerDocument,element=doc.createElement('section'),owned=new Set(),views=[];
  element.id='lighting-rack-controls';
  const heading=doc.createElement('strong');heading.textContent='Lighting rack dimmers';element.append(heading);
  const help=doc.createElement('p');help.textContent='Shortcut overrides preserve individual fixture settings. Clear an override to return to those settings.';element.append(help);
  const status=doc.createElement('output');status.id='lighting-rack-status';status.setAttribute('role','status');
  const clone=value=>structuredClone(value);
  const labels={'upper-wall':'Upper wall','middle-wall':'Middle wall',edge:'Edge ring','near-edge':'Inner ring',center:'Center crane'};
  function edit(change){
    try {const next=clone(getState());change(next);if(setState(next)===false)throw Error('Rack override was rejected.');sync();status.textContent='';}
    catch(error){sync();status.textContent=error.message;}
  }
  function button(parent,id,text,action){
    const node=doc.createElement('button');node.type='button';node.id='lighting-rack-'+id;node.textContent=text;
    owned.add(node.id);node.addEventListener('click',action);parent.append(node);return node;
  }
  function slider(parent,id,label,{read,write,min=0,max=1,step=.01,clear=true,fallback=1,unit=''}){
    const wrap=doc.createElement('label'),input=doc.createElement('input'),out=doc.createElement('output');
    input.id='lighting-rack-'+id;input.type='range';input.min=String(min);input.step=String(step);
    out.id=input.id+'-out';input.setAttribute('aria-describedby',out.id);wrap.textContent=label+' ';wrap.append(input,out);parent.append(wrap);owned.add(input.id);
    const upper=()=>typeof max==='function'?max():max;
    const set=()=>edit(next=>{const value=input.value.trim()===''?NaN:Number(input.value),limit=upper();if(!Number.isFinite(value)||!Number.isFinite(limit)||value<min||value>limit)throw RangeError(label+' is outside the supported range.');write(next,value);});
    input.addEventListener('input',set);
    let clearButton=null;if(clear)clearButton=button(parent,id+'-clear','Clear '+label,()=>edit(next=>write(next,null)));
    views.push(state=>{const value=read(state),active=value!==null&&value!==undefined,limit=upper();input.max=String(limit);input.disabled=!Number.isFinite(limit)||limit<min;input.value=String(active?value:Math.min(limit,Math.max(min,fallback)));out.value=active?Number(value).toFixed(2)+unit:'Individual settings';out.textContent=out.value;input.setAttribute('aria-label',label+(active?' override active':' override inactive'));if(clearButton)clearButton.disabled=!active;});
    return input;
  }
  slider(element,'shop-dimmer','Shop lights',{read:s=>s.shopDimmer,write:(s,v)=>{s.shopDimmer=v;}});
  slider(element,'house-dimmer','House lights',{read:s=>s.houseDimmer,write:(s,v)=>{s.houseDimmer=v;},fallback:0});
  for(const initial of getState().buses){
    const id=initial.id,row=doc.createElement('fieldset'),legend=doc.createElement('legend');legend.textContent=labels[id]??id;row.append(legend);element.append(row);
    const bus=s=>{const found=s.buses.find(b=>b.id===id);if(!found)throw Error('Lighting bus is unavailable: '+id);return found;};
    slider(row,id+'-dimmer','Bus dimmer',{read:s=>bus(s).dimmer,write:(s,v)=>{bus(s).dimmer=v;}});
    slider(row,id+'-drop','Chain drop (feet)',{read:s=>bus(s).dropFeet,write:(s,v)=>{bus(s).dropFeet=v;},max:()=>getMaxDropFeet(id),fallback:.5,unit:' ft'});
    const solo=button(row,id+'-solo','Solo '+(labels[id]??id),()=>edit(s=>{s.solo=s.solo===id?null:id;}));
    views.push(s=>solo.setAttribute('aria-pressed',String(s.solo===id)));
  }
  for(const [key,label]of [['craneX','Crane across'],['craneZ','Crane along']])if(Object.hasOwn(getState(),key))slider(element,key,label,{read:s=>s[key],write:(s,v)=>{s[key]=v;},min:-1,max:1,fallback:0,clear:false});
  const allOut=button(element,'all-out','All out',()=>edit(s=>{s.allOut=!s.allOut;}));views.push(s=>allOut.setAttribute('aria-pressed',String(s.allOut)));
  const clearSolo=button(element,'clear-solo','Clear solo',()=>edit(s=>{s.solo=null;}));views.push(s=>{clearSolo.disabled=s.solo===null;});
  button(element,'clear-all','Clear all overrides',()=>edit(s=>{
    s.shopDimmer=null;s.houseDimmer=null;s.allOut=false;s.solo=null;
    for(const bus of s.buses){bus.dimmer=null;bus.dropFeet=null;}
  }));
  if(typeof onBuildRacks==='function')button(element,'build','Build modular rack lights',()=>{
    try {if(onBuildRacks()===false)throw Error('Rack layout was rejected.');sync();status.textContent='';}
    catch(error){sync();status.textContent=error.message;}
  });
  element.append(status);container.append(element);
  function sync(){const state=clone(getState());for(const view of views)view(state);}
  sync();return {element,sync,ownsControl:id=>owned.has(id)};
}


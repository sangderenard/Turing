// Mechanical attachments belong to a light, not to its dimensions or shape menu.
// These records drive the same real triangles used by preview and ray tracing.
const RECTANGLES=new Set(['key-left','key-right','fill-lower','fill-upper','underlight']);
const ROUND_RADII={background:.3,'kicker-left':.2,'kicker-right':.2,'table-left':.09,'table-right':.09};
export const LIGHT_TOOL_KINDS={grid:'Grid',snoot:'Snoot'};
export const LEGACY_GRID_SETTINGS={cellsX:6,cellsY:8,depth:.20,thickness:.025};
export const DEFAULT_GRID_ACCENT={enabled:false,material:'housing',pattern:'checker',size:1,scope:'interiors',width:.2};
const clone=value=>JSON.parse(JSON.stringify(value));
export function compatibleLightTools(lightId){
 return RECTANGLES.has(lightId)?['grid']:Object.hasOwn(ROUND_RADII,lightId)?['snoot']:[];
}
export function lightToolFootprint(lightId,sizes,keyShape='bare'){
 const size=sizes?.lights?.find(light=>light.id===lightId)??{width:1,height:1};
 if(RECTANGLES.has(lightId)){
  const base=lightId==='underlight'?[1.8,2.4]:lightId.startsWith('key-')?
   (['grid','snoot'].includes(keyShape)?[.2,.2]:[.76,.76]):[.6,.6];
  return {width:base[0]*size.width,height:base[1]*size.height};
 }
 return {diameter:2*ROUND_RADII[lightId]*size.width};
}
export function createLightTool(lightId,kind,sizes,keyShape='bare'){
 if(!compatibleLightTools(lightId).includes(kind))throw Error('Grid requires a rectangular fixture; snoot requires a round fixture.');
 const footprint=lightToolFootprint(lightId,sizes,keyShape);
 return {id:`${lightId}-${kind}`,lightId,kind,enabled:true,offset:0,
  settings:kind==='grid'?{...LEGACY_GRID_SETTINGS,interiorMaterial:'polished_silver',outerMaterial:'housing'}: {length:.44,mountRatio:1.25,mountAperture:0,exitAperture:footprint.diameter}};
}
export function migrateLightTools(sizes){
 const attachments=[];
 for(const [lightId,length,mountRadius]of [['background',1.2,.375],['kicker-left',.44,null],['kicker-right',.44,null],['table-left',.36,.1125],['table-right',.36,.1125]]){
  const size=sizes?.lights?.find(light=>light.id===lightId),radius=ROUND_RADII[lightId]*(size?.width??1);
  const entrance=mountRadius??radius*1.1,tool=createLightTool(lightId,'snoot',sizes);
  tool.enabled=size?.snoot??true;tool.settings={length:size?.snootLength??length,mountRatio:mountRadius===null?1.1:1.25,mountAperture:mountRadius===null?0:mountRadius*2,exitAperture:entrance*2};
  attachments.push(tool);
 }
 const under=sizes?.lights?.find(light=>light.id==='underlight'),grid=createLightTool('underlight','grid',sizes);
 grid.enabled=under?.grid??true;grid.settings.interiorMaterial='housing';
 for(const key of Object.keys(LEGACY_GRID_SETTINGS))grid.settings[key]=under?.['grid'+key[0].toUpperCase()+key.slice(1)]??LEGACY_GRID_SETTINGS[key];
 attachments.push(grid);
 return {version:1,attachments};
}
export function defaultLightTools(sizes){
 const result=migrateLightTools(sizes),over=createLightTool('fill-upper','grid',sizes);
 over.settings={cellsX:8,cellsY:8,depth:1,thickness:.025,interiorMaterial:'polished_silver',outerMaterial:'housing'};result.attachments.push(over);
 for(const tool of result.attachments)if(tool.kind==='grid')tool.settings.interiorMaterial='polished_silver';
 return result;
}
export function validateLightTools(value,sizes,keyShape='bare',materialNames=null){
 if(value===undefined)value=migrateLightTools(sizes);
 if(!value||value.version!==1||!Array.isArray(value.attachments)||value.attachments.length>32)throw Error('Invalid light tools');
 const ids=new Set(),mounts=new Set();
 const attachments=value.attachments.map(tool=>{
  if(!tool||typeof tool.id!=='string'||!/^[a-zA-Z0-9-]{1,80}$/.test(tool.id)||ids.has(tool.id))throw Error('Each light tool needs a unique stable ID');
  ids.add(tool.id);
  if(!compatibleLightTools(tool.lightId).includes(tool.kind))throw Error('Grid requires a rectangular fixture; snoot requires a round fixture.');
  const mount=tool.lightId+'/'+tool.kind;if(mounts.has(mount))throw Error('Only one tool of each kind can occupy a light mount');mounts.add(mount);
  if(typeof tool.enabled!=='boolean'||!Number.isFinite(tool.offset)||tool.offset<0||tool.offset>10)throw Error('Invalid light tool enabled state or axial offset');
  const s=tool.settings,footprint=lightToolFootprint(tool.lightId,sizes,keyShape);let settings;
  if(tool.kind==='grid'){
   if(!s||![s.cellsX,s.cellsY].every(n=>Number.isInteger(n)&&n>=1&&n<=32)||!Number.isFinite(s.depth)||s.depth<.01||s.depth>2||
      !Number.isFinite(s.thickness)||s.thickness<.002||s.thickness>.2)throw Error('Grid needs 1-32 cells per axis, depth 0.01-2 and thickness 0.002-0.2 Studio units');
   if(s.thickness>=Math.min(footprint.width/s.cellsX,footprint.height/s.cellsY,footprint.width/4,footprint.height/4))
    throw Error('Grid wall thickness must fit each cell pitch and inside the can; reduce cells/thickness or enlarge the fixture');
   settings={cellsX:s.cellsX,cellsY:s.cellsY,depth:s.depth,thickness:s.thickness};
   for(const key of ['interiorMaterial','outerMaterial']){const name=s[key]??'housing';if(typeof name!=='string'||!name.length||(materialNames&&name!=='housing'&&!materialNames.includes(name)))throw Error('Unavailable grid material');settings[key]=name;}
   if(s.accent!==undefined){
    const a=s.accent;
    if(!a||typeof a.enabled!=='boolean'||!['checker','rows','columns'].includes(a.pattern)||!['interiors','walls','corners'].includes(a.scope)||!Number.isInteger(a.size)||a.size<1||a.size>32||!Number.isFinite(a.width)||a.width<=0||a.width>.5||typeof a.material!=='string'||!a.material.length||(materialNames&&a.material!=='housing'&&!materialNames.includes(a.material)))throw Error('Invalid grid accent pattern, material or size');
    settings.accent={enabled:a.enabled,material:a.material,pattern:a.pattern,size:a.size,scope:a.scope,width:a.width};
   }
  }else{
   if(!s||!Number.isFinite(s.length)||s.length<0||s.length>10||!Number.isFinite(s.mountRatio)||s.mountRatio<=0||s.mountRatio>20||
      !Number.isFinite(s.mountAperture)||s.mountAperture<0||s.mountAperture>64||!Number.isFinite(s.exitAperture)||s.exitAperture<.002||s.exitAperture>64)throw Error('Snoot needs length 0-10 and exit aperture 0.002-64 Studio units');
   settings={length:s.length,mountRatio:s.mountRatio,mountAperture:s.mountAperture,exitAperture:s.exitAperture};
  }
  return {id:tool.id,lightId:tool.lightId,kind:tool.kind,enabled:tool.enabled,offset:tool.offset,settings};
 });
 return {version:1,attachments};
}
export function applyLightTools(roles,state,sizes,keyShape='bare'){
 const tools=validateLightTools(state,sizes,keyShape);
 const ids=['key-left','key-right','background','kicker-left','kicker-right','fill-lower','fill-upper','table-left','table-right','underlight'];
 return roles.map(role=>{
  const next={...role,grid:false,gobo:false},lightId=ids[role.roleIndex];
  for(const tool of tools.attachments.filter(t=>t.lightId===lightId&&t.enabled)){
   const s=tool.settings;
   if(tool.kind==='grid'&&(role.round||!Number.isFinite(role.halfW)||!Number.isFinite(role.halfH)))throw Error('Grid requires a rectangular fixture with a real width and height');
   if(tool.kind==='snoot'&&(!role.round||!Number.isFinite(role.radius)))throw Error('Snoot requires a round fixture with a real radius');
   if(tool.kind==='grid')Object.assign(next,{grid:true,gridToolId:tool.id,gridOffset:tool.offset,gridCellsX:s.cellsX,gridCellsY:s.cellsY,gridDepth:s.depth,gridThickness:s.thickness,gridInteriorMaterial:s.interiorMaterial,gridOuterMaterial:s.outerMaterial,gridAccent:s.accent??null,gridAccentMaterial:s.accent?.material??'housing'});
   else Object.assign(next,{gobo:s.length>0,shroudToolId:tool.id,shroudOffset:tool.offset,shroudLength:s.length,shroudRadius:s.mountAperture>0?s.mountAperture/2:role.radius*s.mountRatio,shroudExitRadius:s.exitAperture/2});
  }
  return next;
 });
}

// All controls below edit the canonical attachment records. Absent tools have
// no settings in the DOM. Rebuilding is limited to attach/remove/profile load.
export function createLightToolControls(groups,{getSizes,materialNames=[],getKeyShape=()=> 'bare',onChange=()=>{},validateChange=()=>{}}={}){
 let state=defaultLightTools(getSizes());const views=[],owned=new Set();
 const valid=value=>validateLightTools(value,getSizes(),getKeyShape(),materialNames.length?materialNames:null);
 const commit=(next,id,rebuild=false)=>{
  const before=state;
  try{next=valid(next);validateChange(next,before,id);state=next;if(rebuild)render();else sync();onChange();return true;}
  catch(error){sync();views.find(v=>v.id===id).status.textContent=error.message;return false;}
 };
 const sync=()=>{
  for(const view of views){
   view.status.textContent='';
   for(const field of view.fields){
    const tool=state.attachments.find(t=>t.id===field.toolId);if(!tool)continue;
    const value=field.read(tool);if(field.input.type==='checkbox')field.input.checked=value;else field.input.value=String(value);
   }
   for(const output of view.outputs){const tool=state.attachments.find(t=>t.id===output.toolId);if(!tool)continue;
    const f=lightToolFootprint(tool.lightId,getSizes(),getKeyShape()),s=tool.settings,x=f.width/s.cellsX,y=f.height/s.cellsY;
    output.element.textContent=`Cell pitch ${x.toFixed(3)} x ${y.toFixed(3)} Studio units; clear opening ${(x-s.thickness).toFixed(3)} x ${(y-s.thickness).toFixed(3)}. Grid and sealed apron follow the can.`;
   }
  }
 };
 function render(){
  owned.clear();
  for(const view of views){
   const {doc,id,element}=view;view.fields=[];view.outputs=[];element.replaceChildren();
   const label=doc.createElement('label'),select=doc.createElement('select');label.textContent='Attach tool ';select.id=`light-tool-attach-${id}`;owned.add(select.id);
   const empty=doc.createElement('option');empty.value='';empty.textContent='Choose a tool';select.append(empty);
   for(const kind of compatibleLightTools(id))if(!state.attachments.some(t=>t.lightId===id&&t.kind===kind)){
    const option=doc.createElement('option');option.value=kind;option.textContent=LIGHT_TOOL_KINDS[kind];select.append(option);
   }
   select.value='';select.disabled=select.children.length===1;label.append(select);element.append(label);
   const compatibility=doc.createElement('p');compatibility.textContent=RECTANGLES.has(id)?'Rectangular mount: grid supported. Round snoots require a round fixture.':'Round mount: snoot supported. Rectangular grids require a rectangular fixture.';element.append(compatibility);
   select.addEventListener('change',()=>{if(!select.value)return;const next=clone(state);next.attachments.push(createLightTool(id,select.value,getSizes(),getKeyShape()));commit(next,id,true);select.value='';});
   for(const tool of state.attachments.filter(t=>t.lightId===id)){
    const panel=doc.createElement('fieldset'),legend=doc.createElement('legend');legend.textContent=LIGHT_TOOL_KINDS[tool.kind];panel.append(legend);
    const field=(key,text,min,max,step,read=t=>key in t?t[key]:t.settings[key],write=(t,v)=>{if(key in t)t[key]=v;else t.settings[key]=v;})=>{
     const label=doc.createElement('label'),input=doc.createElement('input');label.textContent=text;input.id=`light-tool-${tool.id}-${key}`;input.type=key==='enabled'?'checkbox':'number';input.min=String(min);input.max=String(max);input.step=String(step);label.append(input);panel.append(label);owned.add(input.id);view.fields.push({input,toolId:tool.id,read});
     input.addEventListener('change',()=>{const next=clone(state),changed=next.attachments.find(t=>t.id===tool.id),value=input.type==='checkbox'?input.checked:input.value.trim()===''?NaN:Number(input.value);try{write(changed,value);commit(next,id);}catch(error){sync();view.status.textContent=error.message;}});
    };
    field('enabled','Enabled',0,1,1);field('offset','Mount extension (Studio units)',0,10,.01);
    if(tool.kind==='grid'){
     for(const [key,title] of [['interiorMaterial','Tunnel interior material'],['outerMaterial','Rims and exterior material']]){
      const label=doc.createElement('label'),input=doc.createElement('select');label.textContent=title;input.id=`light-tool-${tool.id}-${key}`;
      for(const name of ['housing',...materialNames]){const option=doc.createElement('option');option.value=name;option.textContent=name==='housing'?'Use fixture housing':name.replaceAll('_',' ');input.append(option);}
      label.append(input);panel.append(label);owned.add(input.id);view.fields.push({input,toolId:tool.id,read:t=>t.settings[key]});
      input.addEventListener('change',()=>{const next=clone(state);next.attachments.find(t=>t.id===tool.id).settings[key]=input.value;commit(next,id);});
     }
     field('cellsX','Grid cells across',1,32,1);field('cellsY','Grid cells up',1,32,1);
     field('depth','Grid depth (Studio units)',.01,2,.01);field('thickness','Wall thickness (Studio units)',.002,.2,.001);
     for(const [axis,key]of [['width','cellsX'],['height','cellsY']])field('pitch-'+axis,'Target cell '+axis+' (Studio units)',0,64,'any',t=>lightToolFootprint(id,getSizes(),getKeyShape())[axis]/t.settings[key],(t,value)=>{
      const span=lightToolFootprint(id,getSizes(),getKeyShape())[axis];if(!Number.isFinite(value)||value<span/32||value>span)throw Error(`Requested pitch must be between ${span/32} and ${span} Studio units (1-32 whole cells).`);t.settings[key]=Math.round(span/value);
     });
     const advanced=doc.createElement('details'),summary=doc.createElement('summary');summary.textContent='Advanced grid accent regions';advanced.append(summary);panel.append(advanced);
     const accentFields=[['enabled','Accent enabled',null],['material','Accent material',['housing',...materialNames]],['pattern','Repeat pattern',['checker','rows','columns']],['scope','Accent region',['interiors','walls','corners']],['size','Pattern block size (cells)',null],['width','Corner strip fraction (0-0.5)',null]];
     for(const [key,title,choices] of accentFields){
      const label=doc.createElement('label'),input=doc.createElement(choices?'select':'input');label.textContent=title;input.id=`light-tool-${tool.id}-accent-${key}`;
      if(choices)for(const name of choices){const option=doc.createElement('option');option.value=name;option.textContent=name.replaceAll('_',' ');input.append(option);}
      else {input.type=key==='enabled'?'checkbox':'number';input.min=key==='size'?'1':'.001';input.max=key==='size'?'32':'.5';input.step=key==='size'?'1':'.01';}
      label.append(input);advanced.append(label);owned.add(input.id);view.fields.push({input,toolId:tool.id,read:t=>(t.settings.accent??DEFAULT_GRID_ACCENT)[key]});
      input.addEventListener('change',()=>{const next=clone(state),t=next.attachments.find(t=>t.id===tool.id);t.settings.accent={...(t.settings.accent??DEFAULT_GRID_ACCENT),[key]:choices?input.value:key==='enabled'?input.checked:input.value.trim()===''?NaN:Number(input.value)};commit(next,id);});
     }
     const accentHelp=doc.createElement('p');accentHelp.textContent='Checker, row or column blocks select alternating groups of cells, starting at the negative local corner. Interiors colors all four tunnel sides of selected cells; walls colors their front/back cutting caps; corners colors strips beside their wall intersections through the full depth. Cell openings remain empty; no alpha mask or optical filter is added.';advanced.append(accentHelp);
     const output=doc.createElement('output');output.id=`light-tool-${tool.id}-pitch`;panel.append(output);view.outputs.push({element:output,toolId:tool.id});
    }else{
     field('length','Snoot length (Studio units)',0,10,.01);field('exitAperture','Exit aperture diameter (Studio units)',.002,64,.01);
    }
    const remove=doc.createElement('button');remove.type='button';remove.id=`light-tool-remove-${tool.id}`;remove.textContent='Remove '+LIGHT_TOOL_KINDS[tool.kind].toLowerCase();owned.add(remove.id);panel.append(remove);
    remove.addEventListener('click',()=>{const next=clone(state);next.attachments=next.attachments.filter(t=>t.id!==tool.id);commit(next,id,true);});element.append(panel);
   }
   element.append(view.status);
  }
  sync();
 }
 for(const {id,group}of groups){const doc=group.ownerDocument,element=doc.createElement('section'),status=doc.createElement('p');status.setAttribute('role','status');group.append(element);views.push({doc,id,element,status,fields:[],outputs:[]});}
 render();return {getState:()=>valid(state),setState:value=>{state=valid(value);render();},validateForSizes:sizes=>validateLightTools(state,sizes,getKeyShape()),sync,ownsControl:id=>owned.has(id)};
}

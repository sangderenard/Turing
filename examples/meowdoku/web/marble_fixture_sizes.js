import {defaultStudioTable} from './marble_studio_table.js';
import {createLightToolControls} from './marble_light_tools.js';
// Dimensions author the existing fixture roles used by Phong, emission and rays.
export const SIZED_FIXTURES=[
 {id:'key-left',label:'Left key',rect:true},{id:'key-right',label:'Right key',rect:true},
 {id:'background',label:'Background'},{id:'kicker-left',label:'Left kicker',snoot:true},
 {id:'kicker-right',label:'Right kicker',snoot:true},{id:'fill-lower',label:'Lower fill',rect:true},
 {id:'fill-upper',label:'Overlight',rect:true},{id:'table-left',label:'Stage left'},
 {id:'table-right',label:'Stage right'},{id:'underlight',label:'Underlight',rect:true,grid:true},
];
export const DEFAULT_UNDERLIGHT_GRID={grid:true,gridCellsX:6,gridCellsY:8,gridDepth:.20,gridThickness:.025};
export function defaultFixtureSizes(legacy=false){const result={version:1,lights:SIZED_FIXTURES.map(f=>({id:f.id,width:!legacy&&f.rect?(f.id==='fill-upper'?8/.6:f.id==='fill-lower'?4/.6:2):1,...(f.rect?{height:!legacy?(f.id==='fill-upper'?8/.6:f.id==='fill-lower'?2/.6:2):1}:{}),...(f.snoot?{snoot:true,snootLength:.2*2.2}:{}),...(f.grid?DEFAULT_UNDERLIGHT_GRID:{})}))};return legacy?result:resolveUnderlightFit({...result,underlightFit:{shortMargin:.05,longMargin:0}},defaultStudioTable());}
// Margins are measured from the full can/apron, not the glowing panel.
// The upward underlight local width is world Z; local height is world X.
export function validateUnderlightFitPose(sizes,poses){
 if(!sizes?.underlightFit)return;
 const pose=poses?.poses?.find(p=>p.id==='underlight');
 if(pose&&(Math.abs(pose.pos[0])>1e-8||Math.abs(pose.pos[2])>1e-8||(!pose.aimAtSubject&&Math.abs(pose.pitch-90)>1e-8)))throw Error('Table-fitted underlight must stay centred and point straight up; saved fixed-size rigs keep their authored pose');
}
export function resolveUnderlightFit(value,table=defaultStudioTable()){
 if(!value.underlightFit)return value;
 const {shortMargin,longMargin}=value.underlightFit;
 if(![shortMargin,longMargin].every(n=>Number.isFinite(n)&&n>=0&&n<=10))throw Error('Underlight margins must be 0-10 Studio units');
 const width=((table.length-2*shortMargin)/2-.002)/(.9*1.25);
 const height=((table.width-.68-2*longMargin)/2-.002)/(1.2*1.25);
 if(![width,height].every(n=>n>=.1&&n<=32))throw Error('Underlight margins leave no usable fixture inside the table legs/skirt');
 return {...value,underlightFit:{shortMargin,longMargin},lights:value.lights.map(p=>p.id==='underlight'?{...p,width,height}:p)};
}
export function validateFixtureSizes(value,validateLegacyAccessories=true){
 if(value===undefined)return defaultFixtureSizes(true);
 if(!value||value.version!==1||!Array.isArray(value.lights)||value.lights.length!==SIZED_FIXTURES.length)throw Error('Invalid fixture dimensions');
 if(value.underlightFit&&!['shortMargin','longMargin'].every(k=>Number.isFinite(value.underlightFit[k])&&value.underlightFit[k]>=0&&value.underlightFit[k]<=10))throw Error('Invalid underlight margins');
 return {version:1,...(value.underlightFit?{underlightFit:{...value.underlightFit}}:{}),lights:SIZED_FIXTURES.map(f=>{
  const p=value.lights.find(x=>x?.id===f.id);
  if(!p||![p.width,...(f.rect?[p.height]:[])].every(v=>Number.isFinite(v)&&v>=.1&&v<=32))throw Error('Fixture dimension factors must be between 0.1 and 32');
  if(f.snoot&&(typeof p.snoot!=='boolean'||!Number.isFinite(p.snootLength)||p.snootLength<0||p.snootLength>10))throw Error('Invalid kicker snoot');
  let grid={};
  if(f.grid){
   grid=Object.fromEntries(Object.entries(DEFAULT_UNDERLIGHT_GRID).map(([key,initial])=>[key,p[key]===undefined?initial:p[key]]));
   if(typeof grid.grid!=='boolean'||![grid.gridCellsX,grid.gridCellsY].every(v=>Number.isInteger(v)&&v>=1&&v<=32)||
      !Number.isFinite(grid.gridDepth)||grid.gridDepth<.01||grid.gridDepth>2||
      !Number.isFinite(grid.gridThickness)||grid.gridThickness<.002||grid.gridThickness>.2)
     throw Error('Grid needs 1–32 cells per axis, depth 0.01–2 and thickness 0.002–0.2 Studio units');
   if(validateLegacyAccessories&&grid.gridThickness>=Math.min(1.8*p.width/grid.gridCellsX,2.4*p.height/grid.gridCellsY,.45*p.width,.6*p.height))
     throw Error('Grid wall thickness must fit each cell pitch and inside the can; reduce cells/thickness or enlarge the underlight');
  }
  return {id:f.id,width:p.width,...(f.rect?{height:p.height}:{}),...(f.snoot?{snoot:p.snoot,snootLength:p.snootLength}:{}),...grid};
 })};
}
export function applyFixtureSizes(roles,state){
 const sizes=validateFixtureSizes(state,false);
 return roles.map(role=>{
  const f=SIZED_FIXTURES[role.roleIndex],p=sizes.lights.find(x=>x.id===f.id),next={...role};
  if(role.round)next.radius=role.radius*p.width;else {next.halfW=role.halfW*p.width;next.halfH=role.halfH*p.height;}
  if(f.grid)for(const key of Object.keys(DEFAULT_UNDERLIGHT_GRID))next[key]=p[key];
  if(f.snoot){next.gobo=p.snoot&&p.snootLength>0;next.shroudLength=p.snootLength;}
  return next;
 });
}
export function createFixtureSizeControls(container,onChange=()=>{},validateChange=()=>{},toolOptions={}){
 const doc=container.ownerDocument,root=doc.createElement('details'),title=doc.createElement('summary');title.textContent='A (exposure): fixture sizes and attached tools';root.append(title);
 const help=doc.createElement('p');help.textContent='Size factors multiply the original fixture dimensions. New overlight emitters are 8 x 8 Studio units, lower fill 4 x 4. Saved profiles keep their dimensions and tools. Attach tools to each compatible mount; only attached tools have settings. Grids use real walls and a sealed apron matching the can. Snoots use an open tube with an adjustable exit aperture.';root.append(help);
 let state=defaultFixtureSizes(),tools;const getTable=toolOptions.getTable??defaultStudioTable;
 const effective=()=>resolveUnderlightFit(state,getTable());let fitFields=[],fitButton=null;const fields=[],groups=[],status=doc.createElement('p');status.setAttribute('role','status');
 const sync=()=>{for(const {input,key}of fitFields){input.disabled=!state.underlightFit;input.value=String(state.underlightFit?.[key]??0);}if(fitButton)fitButton.hidden=!!state.underlightFit;for(const {input,id,key}of fields)input.value=String(state.lights.find(x=>x.id===id)[key]);tools?.sync();};
 for(const f of SIZED_FIXTURES){
  const group=doc.createElement('fieldset'),legend=doc.createElement('legend');legend.textContent=f.label;group.append(legend);groups.push({id:f.id,group});
  if(f.id==='underlight'){
   const note=doc.createElement('p');note.textContent='Autosize the complete underlight can and grid. Short-axis margin is inside the skirt; long-axis margin starts at the inner leg faces. Studio units, per side. Saved fixed-size rigs stay unchanged until converted. Keep the fitted underlight centred and pointing straight up.';group.append(note);
   fitButton=doc.createElement('button');fitButton.type='button';fitButton.textContent='Fit saved underlight to table';group.append(fitButton);
   const commitFit=fit=>{const before=state;try{const next=validateFixtureSizes(resolveUnderlightFit({...state,underlightFit:fit},getTable()),false);tools.validateForSizes(next);validateChange(next,before,'underlight',tools.getState());state=next;status.textContent='';sync();onChange();}catch(e){status.textContent=e.message;sync();}};
   fitButton.addEventListener('click',()=>commitFit({shortMargin:.05,longMargin:0}));
   for(const [key,text]of [['shortMargin','Short-axis skirt margin'],['longMargin','Long-axis leg margin']]){const label=doc.createElement('label'),input=doc.createElement('input');input.id=`underlight-${key}`;input.type='number';input.min='0';input.max='10';input.step='.01';label.textContent=text;label.append(input);group.append(label);fitFields.push({key,input});input.addEventListener('change',()=>commitFit({...state.underlightFit,[key]:input.value.trim()===''?NaN:Number(input.value)}));}
  }
  for(const key of f.id==='underlight'?[]:['width',...(f.rect?['height']:[])]){
   const label=doc.createElement('label'),input=doc.createElement('input');input.id=`fixture-size-${f.id}-${key}`;
   label.textContent=key==='width'?(f.rect?'Width factor':'Diameter factor'):'Height factor';input.type='number';input.min='.1';input.max='32';input.step='.01';label.append(input);group.append(label);fields.push({input,id:f.id,key});
   input.addEventListener('change',()=>{
    const next=JSON.parse(JSON.stringify(state)),p=next.lights.find(x=>x.id===f.id);p[key]=input.value.trim()===''?NaN:Number(input.value);
    try{const valid=validateFixtureSizes(next,false);tools.validateForSizes(valid);validateChange(valid,state,f.id,tools.getState());state=valid;status.textContent='';sync();onChange();}catch(e){sync();status.textContent=e.message;}
   });
  }
  root.append(group);
 }
 tools=createLightToolControls(groups,{...toolOptions,getSizes:effective,onChange:()=>{onChange();toolOptions.onChange?.();}});
 root.append(status);container.append(root);sync();return {element:root,tools,getState:()=>validateFixtureSizes(effective(),false),setState:value=>{state=validateFixtureSizes(value,false);sync();},sync,validateForTable:value=>{const next=resolveUnderlightFit(state,value);tools.validateForSizes(next);return next;},ownsControl:id=>fitFields.some(f=>f.input.id===id)||fields.some(f=>f.input.id===id)||tools.ownsControl(id)};
}

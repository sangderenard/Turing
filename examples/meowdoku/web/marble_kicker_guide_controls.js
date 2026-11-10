import {snootSphereGuide,kickerSubjectRays,tableGuideScene,KickerTableQuery,summarizeKickerSamples,kickerTableClearanceBound} from './marble_kicker_guide.js';
import {TABLE_TOP_Y} from './marble_studio_table.js';

export function kickerGuideHeights({floorY,ceilingY,subject,table,role}){
 const max=Math.min(50-floorY,ceilingY-floorY),stage=TABLE_TOP_Y-floorY,current=role.pos[1]-floorY;
 const values=Array.from({length:9},(_,i)=>max*i/8);
 if(table.enabled)for(const factor of [-8,-4,-2,-1,-.5,0,.5,1,2,4,8])values.push(stage+factor*subject.radius);
 values.push(4,current);
 return [...new Set(values.filter(v=>Number.isFinite(v)&&v>=0&&v<=max))].sort((a,b)=>a-b);
}

// Floor-relative height remains a view of the existing absolute pose. Neither
// the samples nor their result are saved, and no automatic relocation occurs.
export function createKickerHeightGuides(container,{getSnapshots,setHeight,createQuery=()=>KickerTableQuery.create(),delay=120}={}){
 const doc=container.ownerDocument,root=doc.createElement('section');root.id='kicker-height-guides';
 const title=doc.createElement('h3');title.textContent='Kicker height and line of sight';
 const help=doc.createElement('p');help.textContent='Height is above the floor. Aperture: exact emitter-center cone versus the scaled subject sphere. Stage: sampled straight rays through the real table slab, legs and skirt. Surface crossings are not optical blockage: transmission, room walls, other fixtures and gobos are not checked.';
 root.append(title,help);container.append(root);
 const views=new Map();let timer=null,query=null,creating=null,running=false,pending=null,generation=0,abort=null,disposed=false,lastKey=null;
 for(const [id,label]of [['kicker-left','Left kicker'],['kicker-right','Right kicker']]){
  const group=doc.createElement('fieldset'),legend=doc.createElement('legend');legend.textContent=label;group.append(legend);
  const wrapper=doc.createElement('label'),input=doc.createElement('input');wrapper.textContent='Height above floor (Studio units) ';input.type='number';input.id=id+'-height';input.step='.01';input.min='0';wrapper.append(input);
  const aperture=doc.createElement('output');aperture.id=id+'-aperture-guide';aperture.setAttribute('role','status');
  const current=doc.createElement('output');current.id=id+'-table-guide';current.setAttribute('role','status');
  const bound=doc.createElement('output');bound.id=id+'-table-clearance-bound';
  const bandLabel=doc.createElement('p');bandLabel.textContent='Height samples: green = no crossings, amber = some, red = all sampled rays. Click a sample to set height. These are not exact transition boundaries.';
  const band=doc.createElement('div');band.id=id+'-height-samples';if(band.style)band.style.cssText='display:flex;gap:2px;flex-wrap:wrap';
  group.append(wrapper,aperture,current,bound,bandLabel,band);root.append(group);views.set(id,{input,aperture,current,bound,band});
  input.addEventListener('input',()=>{
   const value=input.value.trim()===''?NaN:Number(input.value);
   try{if(!Number.isFinite(value)||value<0||value>Number(input.max))throw Error('Enter a height within the displayed floor-to-ceiling range');setHeight(id,value);}
   catch(error){current.textContent=error.message;}
  });
 }
 const take=()=>getSnapshots();
 const key=snapshots=>JSON.stringify(snapshots);
 const schedule=()=>{
  if(timer!==null)clearTimeout(timer);
  timer=setTimeout(()=>{timer=null;void work();},delay);
 };
 async function work(){
  if(running||disposed||!pending)return;
  const request=pending;pending=null;running=true;const token=generation,controller=new AbortController();abort=controller;
  try{
   const scene=tableGuideScene(request.snapshots[0].table,request.snapshots[0].floorY,request.snapshots[0].unitM),rays=[],groups=[];
   for(const snapshot of request.snapshots){
    const role=snapshot.role,tests=[{height:null,role}];
    if(snapshot.locked)for(const height of kickerGuideHeights(snapshot)){
     const pos=role.pos.map((v,i)=>i===1?height+snapshot.floorY:v),direction=snapshot.subject.center.map((v,i)=>v-pos[i]),length=Math.hypot(...direction);
     if(length<=snapshot.subject.radius)continue;
     tests.push({height,role:{...role,pos,dir:direction.map(v=>v/length)}});
    }
    for(const test of tests){const sample=kickerSubjectRays(test.role,snapshot.subject,snapshot.unitM),first=rays.length;rays.push(...sample);groups.push({id:snapshot.id,height:test.height,first,count:sample.length});}
   }
   let result;
   if(scene.triangles.length){
    if(!query){creating??=createQuery();try{query=await creating;}finally{creating=null;}}
    if(controller.signal.aborted)throw Error('Kicker guide canceled');
    result=await query.query(scene,rays,{signal:controller.signal});
   }else result={hits:rays.map(()=>null)};
   if(disposed||token!==generation||key(take())!==request.key)return;
   for(const group of groups){
    const view=views.get(group.id),summary=summarizeKickerSamples(result.hits.slice(group.first,group.first+group.count));
    if(group.height===null)view.current.textContent=summary.message;
    else{
     const button=doc.createElement('button');button.type='button';button.textContent='●';
     const text=`${group.height.toFixed(3)} above floor: ${summary.crossed}/${summary.total} sampled rays cross table surfaces`;
     button.title=text;button.setAttribute('aria-label',text);button.dataset.coverage=summary.state;button.dataset.height=String(group.height);
     if(button.style)button.style.cssText=`padding:2px;min-width:16px;color:${{none:'#61d997',some:'#ffd166',all:'#ff7b7b'}[summary.state]}`;
     button.addEventListener('click',()=>{try{setHeight(group.id,group.height);}catch(error){view.current.textContent=error.message;}});view.band.append(button);
    }
   }
  }catch(error){if(!disposed&&token===generation)for(const view of views.values())view.current.textContent='Table guide unavailable: '+error.message;}
  finally{running=false;if(abort===controller)abort=null;if(pending&&!disposed)schedule();}
 }
 function refresh(){
  if(disposed)return;
  let snapshots;
  try{snapshots=take();}catch(error){for(const view of views.values())view.current.textContent=error.message;return;}
  const nextKey=key(snapshots);if(nextKey===lastKey)return;lastKey=nextKey;generation++;abort?.abort();
  for(const snapshot of snapshots){
   const view=views.get(snapshot.id);view.input.value=String(Number((snapshot.role.pos[1]-snapshot.floorY).toFixed(3)));view.input.max=String(Math.min(50-snapshot.floorY,snapshot.ceilingY-snapshot.floorY));view.band.replaceChildren();
   try{const cone=snootSphereGuide(snapshot.role,snapshot.subject);view.aperture.textContent='Aperture: '+cone.message+' Emitter-center rays only.';view.aperture.dataset.coverage=cone.state;}
   catch(error){view.aperture.textContent='Aperture guide unavailable: '+error.message;}
   try{view.bound.textContent=kickerTableClearanceBound(snapshot.role,snapshot.subject,snapshot.table,snapshot.floorY).message;}
   catch(error){view.bound.textContent='Whole-emitter table bound unavailable: '+error.message;}
   view.current.textContent=snapshot.locked?'Checking current height and subject-locked height samples…':'Checking table paths at the current height. Aim is manual; enable Aim at subject for height-region samples.';
  }
  if(snapshots.length){pending={snapshots,key:nextKey};schedule();}
 }
 return {element:root,refresh,ownsControl:id=>[...views.values()].some(v=>v.input.id===id),
  dispose(){disposed=true;generation++;abort?.abort();if(timer!==null)clearTimeout(timer);pending=null;}};
}

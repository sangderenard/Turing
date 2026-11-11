// Advanced studio flags. One closed solid per gobo gives both renderers an
// unambiguous nearest front/back surface, rather than two coincident sheets.
import {defaultStudioGrid,validateStudioGrid,studioGridCells,createStudioGridControls,validateStudioArtwork,studioArtworkFaces,studioGraphPaperArtwork} from './marble_studio_grid.js';
export const MAX_STUDIO_GOBOS = 8;
export const GOBO_SEGMENTS = 48;
// Matches the thin tabletop's scale. Even a 1 mm saved-radius scene retains
// more than the ray tracer's 0.1 mm self-intersection guard through the panel.
export const GOBO_THICKNESS = 0.12;
const clone = value => JSON.parse(JSON.stringify(value));
const numeric = {width:[0.2,64], height:[0.2,64], x:[-64,64], y:[-64,64], z:[-64,64],
  pitch:[-180,180], yaw:[-180,180], roll:[-180,180]};

export function studioGraphPaperGobo({id=1,unitMm=304.8,center=[0,0,-2.5],yaw=0}={}) {
  if(!Number.isFinite(unitMm)||unitMm<1||unitMm>320)throw Error('A 64 mm reference needs a Studio unit between 1 and 320 mm');
  const {artwork}=studioGraphPaperArtwork();
  return {id,enabled:true,shape:'rectangle',width:64/unitMm,height:64/unitMm,
    x:center[0],y:center[1],z:center[2],pitch:0,yaw,roll:0,
    front:'white_plaster',back:'white_plaster',artwork};
}

export function validateStudioGobos(value, materialNames) {
  if (value === undefined) return {version:1, gobos:[]};
  if (!value || value.version !== 1 || !Array.isArray(value.gobos) || value.gobos.length > MAX_STUDIO_GOBOS)
    throw new Error(`Gobos must contain at most ${MAX_STUDIO_GOBOS} panels`);
  const names = new Set(materialNames), seen = new Set();
  const gobos = value.gobos.map(gobo => {
    if (!gobo || !Number.isSafeInteger(gobo.id) || gobo.id < 1 || seen.has(gobo.id))
      throw new Error('Invalid or duplicate gobo identity');
    seen.add(gobo.id);
    if (typeof gobo.enabled !== 'boolean' || !['rectangle','ellipse'].includes(gobo.shape))
      throw new Error('Invalid gobo shape or visibility');
    for (const [key,[min,max]] of Object.entries(numeric))
      if (!Number.isFinite(gobo[key]) || gobo[key] < min || gobo[key] > max)
        throw new Error(`Gobo ${key} must be between ${min} and ${max}`);
    for (const side of ['front','back'])
      if (typeof gobo[side] !== 'string' || !names.has(gobo[side]))
        throw new Error(`Unknown gobo ${side} material`);
    const grid=validateStudioGrid(gobo.grid,materialNames);
    if(grid && gobo.shape!=='rectangle')throw new Error('A structured gobo grid requires a rectangle');
    const artwork=validateStudioArtwork(gobo.artwork,materialNames);
    if(artwork && (grid||gobo.shape!=='rectangle'))throw new Error('Surface artwork needs a solid rectangular panel');
    return {id:gobo.id, enabled:gobo.enabled, shape:gobo.shape,
      ...Object.fromEntries(Object.keys(numeric).map(key=>[key,gobo[key]])), front:gobo.front, back:gobo.back,
      ...(grid===undefined?{}:{grid}),...(artwork===undefined?{}:{artwork})};
  });
  return {version:1, gobos};
}

// Fixed world axes: pitch X, then yaw Y, then roll Z (Rz * Ry * Rx).
// At zero rotation front is +Z; positive yaw turns it toward +X.
export function goboRotation(gobo, [x,y,z]) {
  const [pitch,yaw,roll] = ['pitch','yaw','roll'].map(key=>gobo[key]*Math.PI/180);
  const cy=Math.cos(pitch), sy=Math.sin(pitch), cz=Math.cos(yaw), sz=Math.sin(yaw);
  const cr=Math.cos(roll), sr=Math.sin(roll);
  const a=y*cy-z*sy, b=y*sy+z*cy, c=x*cz+b*sz, d=-x*sz+b*cz;
  return [c*cr-a*sr, c*sr+a*cr, d];
}
export function studioGoboFaces(value, ids, onPart) {
  const state=validateStudioGobos(value,Object.keys(ids)), faces=[];
  for (const gobo of state.gobos) {
    if (!gobo.enabled) continue;
    const panels=gobo.grid?studioGridCells(gobo.grid,gobo.width,gobo.height):
      [{id:null,x:0,y:0,width:gobo.width,height:gobo.height,material:''}];
    for(const panel of panels) {
    const first=faces.length;
    const front=ids[panel.material||gobo.front], back=ids[panel.material||gobo.back];
    if (![front,back].every(id=>Number.isInteger(id)&&id>=0)) throw new Error('Unknown gobo material ID');
    const w=panel.width/2, h=panel.height/2, t=GOBO_THICKNESS/2;
    const xy=gobo.shape==='rectangle' ? [[-w,-h],[w,-h],[w,h],[-w,h]] :
      Array.from({length:GOBO_SEGMENTS},(_,i)=>[w*Math.cos(i*2*Math.PI/GOBO_SEGMENTS),h*Math.sin(i*2*Math.PI/GOBO_SEGMENTS)]);
    const point=([x,y,z])=>goboRotation(gobo,[x+panel.x,y+panel.y,z]).map((v,i)=>v+[gobo.x,gobo.y,gobo.z][i]);
    const frontRim=xy.map(([x,y])=>point([x,y,t])), backRim=xy.map(([x,y])=>point([x,y,-t]));
    const f=point([0,0,t]), b=point([0,0,-t]);
    const add=(a,b,c,material,normal)=>faces.push([a,b,c,material,[normal,normal,normal]]);
    const fn=goboRotation(gobo,[0,0,1]), bn=fn.map(v=>-v);
    for (let i=0;i<xy.length;i++) {
      const j=(i+1)%xy.length;
      add(f,frontRim[i],frontRim[j],front,fn);
      add(b,backRim[j],backRim[i],back,bn);
      const dx=xy[j][0]-xy[i][0], dy=xy[j][1]-xy[i][1], length=Math.hypot(dx,dy);
      const rimNormal=goboRotation(gobo,[dy/length,-dx/length,0]);
      add(backRim[i],backRim[j],frontRim[j],front,rimNormal);
      add(backRim[i],frontRim[j],frontRim[i],front,rimNormal);
    }
    const id=panel.id===null?String(gobo.id):`${gobo.id}-${panel.id}`;
    onPart?.({id,bodyId:`gobo-${id}`,construction:"convex-panel",closed:true,convex:true,faces:faces.slice(first)});
    }
    if(gobo.artwork) {
      const normal=goboRotation(gobo,[0,0,1]);
      const point=([x,y,z])=>goboRotation(gobo,[x,y,z+GOBO_THICKNESS/2+0.001]).map((v,i)=>v+[gobo.x,gobo.y,gobo.z][i]);
      faces.push(...studioArtworkFaces(gobo.artwork,gobo.width,gobo.height,ids,point,normal,part=>
        onPart?.({...part,id:`${gobo.id}-art-${part.id}`,bodyId:`gobo-${gobo.id}-art-${part.id}`})));
    }
  }
  return faces;
}

export function createStudioGoboControls(container, materialNames, onChange=()=>{}, options={}) {
  const doc=container.ownerDocument, names=[...materialNames].sort();
  let state=validateStudioGobos(undefined,names), serial=0;
  const owned=new Set();
  const el=(tag,id,text)=>{const node=doc.createElement(tag);if(id){node.id=id;owned.add(id);}if(text)node.textContent=text;return node;};
  const add=el('button','studio-gobo-add','Add gobo');add.type='button';
  const addGrid=el('button','studio-gobo-add-grid','Add fine grid behind marble');addGrid.type='button';
  const addReference=el('button','studio-gobo-add-reference','Add graph-paper reference');addReference.type='button';
  const referenceMaterials=['white_plaster','black_matte_paint',...studioGraphPaperArtwork().reference.swatches];
  let gridEditors=[];
  const rows=el('div','studio-gobo-rows');
  const status=el('p','studio-gobo-status');status.setAttribute('role','status');
  const help=el('p',null,'Free-standing flags in Studio units. Front is +Z at zero rotation; back is -Z. Angles are degrees around world X (pitch), Y (yaw), then Z (roll). Each panel is 0.12 units thick with its front material on the rim. Materials are independent; opaque flags block light in ray photos/live rays. Phong shows the geometry and materials but does not trace their shadows. Lighting profiles save these rows; saved marble objects stay unchanged.');
  container.append(add,addGrid,addReference,status,rows,help);
  function updateStatus(message='') {
    add.disabled=state.gobos.length>=MAX_STUDIO_GOBOS || !names.length;
    addGrid.disabled=add.disabled;
    addReference.disabled=add.disabled||!referenceMaterials.every(n=>names.includes(n));
    status.textContent=message || `${state.gobos.length} / ${MAX_STUDIO_GOBOS} gobos. ${state.gobos.length?'Uncheck a row to hide it without losing its settings.':'No gobos in this set.'}`;
  }
  function render() {
    owned.clear();owned.add(add.id);owned.add(addGrid.id);owned.add(addReference.id);rows.replaceChildren();gridEditors=[];
    for (const [index,gobo] of state.gobos.entries()) {
      const prefix=`studio-gobo-${gobo.id}`, row=el('fieldset');row.className='grid studio-gobo-row';
      row.append(el('legend',null,`Gobo ${index+1}`));
      const enabledLabel=el('label');enabledLabel.className='lighting-pose-lock';
      const enabled=el('input',`${prefix}-enabled`);enabled.type='checkbox';enabled.checked=gobo.enabled;
      enabledLabel.append(enabled,'Enabled');row.append(enabledLabel);
      const remove=el('button',`${prefix}-remove`,'Remove');remove.type='button';
      remove.setAttribute('aria-label',`Remove gobo ${index+1}`);row.append(remove);
      const edit=(key,input)=>{
        const value=input.type==='checkbox'?input.checked:input.type==='number'?(input.value.trim()===''?NaN:Number(input.value)):input.value;
        const next=clone(state);next.gobos[index][key]=value;
        try {
          const valid=validateStudioGobos(next,names);
          input.setAttribute('aria-invalid','false');input.setCustomValidity?.('');
          if (JSON.stringify(valid)===JSON.stringify(state)) {updateStatus();return;}
          state=valid;updateStatus();onChange();
        } catch(error) {
          input.setAttribute('aria-invalid','true');input.setCustomValidity?.(error.message);
          updateStatus(`${error.message}. The scene keeps the last valid value.`);
        }
      };
      enabled.addEventListener('change',()=>edit('enabled',enabled));
      remove.addEventListener('click',()=>{
        state={...state,gobos:state.gobos.filter(item=>item.id!==gobo.id)};
        render();add.focus?.();onChange();
      });
      for(const [key,label,options] of [
        ['shape','Shape',[['rectangle','Rectangular'],['ellipse','Ovoid / ellipse']]],
        ['front','Front material',names.map(name=>[name,name.replaceAll('_',' ')])],
        ['back','Back material',names.map(name=>[name,name.replaceAll('_',' ')])],
      ]) {
        const labelNode=el('label',null,label),select=el('select',`${prefix}-${key}`);
        if(key==='shape')labelNode.className='wide';
        for(const [value,text] of options){const option=el('option',null,text);option.value=value;select.append(option);}
        select.value=gobo[key];select.addEventListener('change',()=>edit(key,select));labelNode.append(select);row.append(labelNode);
      }
      for(const [key,[min,max]] of Object.entries(numeric)) {
        const labels={width:'Width',height:'Height',x:'Position X',y:'Position Y',z:'Position Z',pitch:'Pitch X (deg)',yaw:'Yaw Y (deg)',roll:'Roll Z (deg)'};
        const label=el('label',null,labels[key]),input=el('input',`${prefix}-${key}`);
        input.type='number';input.min=String(min);input.max=String(max);input.step='any';input.value=String(gobo[key]);
        input.addEventListener('input',()=>edit(key,input));label.append(input);row.append(label);
      }
      const grid=createStudioGridControls(row,`${prefix}-grid`,names,value=>{
        const next=clone(state);if(value===undefined)delete next.gobos[index].grid;else next.gobos[index].grid=value;
        state=validateStudioGobos(next,names);updateStatus();onChange();
      });
      grid.setState(gobo.grid);gridEditors.push(grid);rows.append(row);
      if(gobo.artwork)row.append(el('p',null,'Reflective reference surface with real material markings. The graph-paper preset is 64 mm square at creation, with 2 mm squares, 8 mm major lines, material swatches and QR payloads STUDIO;2mm;L and STUDIO;2mm;R. Resizing changes the scale. Rotate the front toward the camera.'));
    }
    updateStatus();
  }
  function addGobo(fineGrid=false){
    if (add.disabled) return;
    const material=names.includes('black_felt')?'black_felt':names[0];
    const next=Math.max(serial,...state.gobos.map(gobo=>gobo.id))+1;
    serial=Number.isSafeInteger(next)?next:1;
    while(state.gobos.some(gobo=>gobo.id===serial))serial++;
    const gobo={id:serial,enabled:true,shape:'rectangle',width:1,height:2,x:-2,y:0,z:0,
      pitch:0,yaw:0,roll:0,front:material,back:material};
    if(fineGrid)Object.assign(gobo,{width:8,height:8,x:0,y:0,z:-2.5,
      front:names.includes('white_plaster')?'white_plaster':material,grid:defaultStudioGrid()});
    state=validateStudioGobos({...state,gobos:[...state.gobos,gobo]},names);render();
    doc.getElementById(`studio-gobo-${serial}-shape`)?.focus?.();onChange();
  }
  add.addEventListener('click',()=>addGobo());addGrid.addEventListener('click',()=>addGobo(true));
  addReference.addEventListener('click',()=>{
    if(addReference.disabled)return;
    try {
      const next=Math.max(serial,...state.gobos.map(g=>g.id))+1;
      const gobo=studioGraphPaperGobo({id:next,...options.referencePlacement?.()});
      state=validateStudioGobos({...state,gobos:[...state.gobos,gobo]},names);serial=next;render();onChange();
    } catch(error){updateStatus(error.message);}
  });
  render();
  return {getState:()=>clone(state),ownsControl:id=>owned.has(id)||gridEditors.some(editor=>editor.ownsControl(id)),
    setState(value){const valid=validateStudioGobos(value,names);state=valid;render();}};
}

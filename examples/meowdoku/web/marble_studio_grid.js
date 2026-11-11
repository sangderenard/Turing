// Authored data shared by the existing gobo solids and backdrop sheets.
// No colors, spectra, lights, fetching or renderer live here. Cells select
// canonical material names; null is absent geometry, not transparent paint.
import {qrMatrix} from './tipjar.js';
import {qrWallLayout} from './qr_wall.js';
export const MAX_STUDIO_GRID_AXIS = 64;
export const MAX_STUDIO_GRID_CELLS = 1024;

// Surface markings are authored in normalized panel coordinates, with physical
// size supplied by the existing gobo/backdrop. This is material geometry, not
// a screen texture, light or replacement material table.
export function validateStudioArtwork(value, materialNames) {
  if(value===undefined)return undefined;
  if(!value||value.version!==1||!Array.isArray(value.rectangles)||value.rectangles.length>2048||
      Object.keys(value).some(k=>!['version','rectangles'].includes(k)))throw Error('Artwork needs version 1 and at most 2048 material rectangles');
  const names=new Set(materialNames),seen=new Set();
  const rectangles=value.rectangles.map(rect=>{
    if(!rect||typeof rect.id!=='string'||!rect.id||rect.id.length>96||seen.has(rect.id))throw Error('Artwork needs unique rectangle identities');
    seen.add(rect.id);
    if(!names.has(rect.material))throw Error('Unknown artwork material');
    if(Object.keys(rect).some(k=>!['id','x','y','width','height','material'].includes(k))||
        !['x','y','width','height'].every(k=>Number.isFinite(rect[k]))||rect.width<=0||rect.height<=0||
        Math.abs(rect.x)+rect.width/2>0.5+1e-12||Math.abs(rect.y)+rect.height/2>0.5+1e-12)
      throw Error('Artwork rectangles must fit the normalized panel [-0.5, 0.5]');
    return {id:rect.id,x:rect.x,y:rect.y,width:rect.width,height:rect.height,material:rect.material};
  });
  return {version:1,rectangles};
}

export function studioArtworkFaces(artwork,width,height,ids,point,normal,onPart) {
  const faces=[];
  for(const r of artwork.rectangles) {
    const x=r.x*width,y=r.y*height,w=r.width*width/2,h=r.height*height/2,material=ids[r.material];
    if(!Number.isInteger(material)||material<0)throw Error('Unknown artwork material ID');
    const a=point([x-w,y-h,0]),b=point([x+w,y-h,0]),c=point([x+w,y+h,0]),d=point([x-w,y+h,0]);
    const part=[[a,b,c,material,[normal,normal,normal]],[a,c,d,material,[normal,normal,normal]]];
    faces.push(...part);onPart?.({id:r.id,construction:'planar-sheet',closed:false,convex:true,faces:part});
  }
  return faces;
}

// A 64 mm square reference card: central 2 mm squares, heavier 8 mm lines,
// canonical non-emissive material swatches and public QR payloads.
// Use width/height = 64 / (Studio unit in mm) to retain the stated scale.
export function studioGraphPaperArtwork() {
  const rectangles=[],black='black_matte_paint';
  const add=(id,x,y,width,height,material=black)=>rectangles.push({id,x,y,width,height,material});
  // Split horizontals at verticals: no coplanar overlaps or material ambiguity.
  const xs=Array.from({length:17},(_,i)=>({x:(i-8)/32,width:i%4===0?.002:.001}));
  for(const [i,line] of xs.entries())add(`grid-v-${i}`,line.x,0,line.width,.876);
  for(let row=0;row<=28;row++)for(let col=0;col<16;col++) {
    const a=xs[col].x+xs[col].width/2,b=xs[col+1].x-xs[col+1].width/2;
    add(`grid-h-${row}-${col}`,(a+b)/2,(row-14)/32,b-a,row%4===2?.002:.001);
  }
  const payloads=['STUDIO;2mm;L','STUDIO;2mm;R'];
  for(const [side,text] of payloads.entries()) {
    const layout=qrWallLayout(qrMatrix(text)),size=.24,cx=side?.375:-.375,cy=side?-.28:.28,pitch=size/layout.total;
    // Existing QR encoder and standard four-module quiet zone are authoritative.
    for(let row=0;row<layout.total;row++)for(let col=0;col<layout.total;) {
      if(!layout.cells[row*layout.total+col].dark){col++;continue;}
      const start=col;while(col<layout.total&&layout.cells[row*layout.total+col].dark)col++;
      add(`qr-${side}-r${row}-c${start}`,cx-size/2+(start+col)*pitch/2,cy+size/2-(row+.5)*pitch,(col-start)*pitch,pitch);
    }
  }
  // The calibration_* catalog entries are emitters, not reflective inks.
  // Use the real surface palette unchanged; this is not a certified color chart.
  const swatches=['heart','marble_cat_eye_green','marble_cat_eye_blue','marble_cat_eye_amber',
    'metal_copper','metal_oxidized_copper','black_matte_paint','white_plaster'];
  swatches.forEach((material,i)=>add(`swatch-${i}`,i<4?-.375:.375,(1.5-i%4)*.055,.18,.045,material));
  for(let i=0;i<16;i+=2)add(`scale-2mm-${i}`,(i-7.5)/32,-.473,1/32,.018);
  return {artwork:{version:1,rectangles},reference:{widthMm:64,heightMm:64,squarePitchMm:2,majorPitchMm:8,
    qrPayloads:payloads,quietModules:4,swatches,role:'reflective-reference-card'}};
}

export function defaultStudioGrid() {
  // 16² closed cells leave room for the default rig inside the existing MRP1
  // 2 MiB config bound; denser authored grids still face that global bound.
  return {version:1,rows:16,columns:16,gap:0.05,cells:Array(256).fill('')};
}

export function validateStudioGrid(value, materialNames) {
  if (value === undefined) return undefined;
  if (!value || value.version !== 1 ||
      Object.keys(value).some(key=>!['version','rows','columns','gap','cells'].includes(key)))
    throw new Error('Grid needs version, rows, columns, gap and cells');
  for (const key of ['rows','columns'])
    if (!Number.isInteger(value[key]) || value[key] < 1 || value[key] > MAX_STUDIO_GRID_AXIS)
      throw new Error(`Grid ${key} must be an integer from 1 to ${MAX_STUDIO_GRID_AXIS}`);
  const count=value.rows*value.columns, names=new Set(materialNames);
  if (count > MAX_STUDIO_GRID_CELLS || !Array.isArray(value.cells) || value.cells.length !== count)
    throw new Error(`Grid needs exactly rows × columns cells, at most ${MAX_STUDIO_GRID_CELLS}`);
  // Positive separation preserves the incumbent closed, independent gobo
  // bodies: adjacent solid cells must not introduce coincident interfaces.
  if (!Number.isFinite(value.gap) || value.gap < 0.01 || value.gap > 0.9)
    throw new Error('Grid gap must be a fraction of cell pitch from 0.01 to 0.9');
  const cells=Array.from(value.cells, cell=>{
    if (cell !== null && (typeof cell !== 'string' || cell !== '' && !names.has(cell)))
      throw new Error('Each grid cell must be null (hole), empty string (inherit), or a known material name');
    return cell;
  });
  return {version:1,rows:value.rows,columns:value.columns,gap:value.gap,cells};
}

// Row 0 is the top (+Y), column 0 is the left (-X), viewed from +Z.
// Consumers validate the record once, then use these same parametric bounds.
export function studioGridCells(grid, width, height) {
  const dx=width/grid.columns,dy=height/grid.rows,cellWidth=dx*(1-grid.gap),cellHeight=dy*(1-grid.gap);
  return grid.cells.flatMap((material,index)=>material===null ? [] : [{
    id:`r${Math.floor(index/grid.columns)}-c${index%grid.columns}`,
    row:Math.floor(index/grid.columns),column:index%grid.columns,material,
    x:-width/2+(index%grid.columns+0.5)*dx,
    y:height/2-(Math.floor(index/grid.columns)+0.5)*dy,
    width:cellWidth,height:cellHeight,
  }]);
}

// The UI edits the very same record accepted by profiles and agent products.
// External producers can supply cells directly; no dataset is inferred here.
export function createStudioGridControls(container, prefix, materialNames, onChange) {
  const doc=container.ownerDocument,owned=new Set();
  let state, draft=defaultStudioGrid();
  const el=(tag,id,text)=>{const n=doc.createElement(tag);if(id){n.id=`${prefix}-${id}`;owned.add(n.id);}if(text)n.textContent=text;return n;};
  const field=el('fieldset');field.className='wide grid';field.append(el('legend',null,'Structured grid'));
  const label=el('label',null,'Use cell grid (rectangles only) '),enabled=el('input','enabled');enabled.type='checkbox';label.append(enabled);field.append(label);
  const inputs={};
  for(const [key,min,max,step] of [['rows',1,MAX_STUDIO_GRID_AXIS,1],['columns',1,MAX_STUDIO_GRID_AXIS,1],['gap',0.01,0.9,0.01]]) {
    const label=el('label',null,key==='gap'?'Gap / cell pitch':key[0].toUpperCase()+key.slice(1)),input=el('input',key);
    input.type='number';input.min=String(min);input.max=String(max);input.step=String(step);inputs[key]=input;label.append(input);field.append(label);
  }
  const resize=el('button','resize','Resize grid / apply gap');resize.type='button';field.append(resize);
  const dataLabel=el('label',null,'Row-major cells (JSON)');dataLabel.className='wide';
  const data=el('textarea','cells');data.rows=5;data.spellcheck=false;dataLabel.append(data);field.append(dataLabel);
  const apply=el('button','apply','Apply cell data');apply.type='button';field.append(apply);
  const status=el('p','status');status.className='wide';status.setAttribute('role','status');field.append(status);
  const help=el('p',null,'Top row first, left to right. Each cell is a canonical material name, "" to inherit the parent, or null for a hole. Gaps are real openings. Resize preserves overlapping cells; new cells inherit. Up to 1024 cells. Disable the grid before choosing an ellipse.');help.className='wide';field.append(help);container.append(field);
  function sync() {
    enabled.checked=state!==undefined;
    for(const key of Object.keys(inputs)){inputs[key].value=String(draft[key]);inputs[key].disabled=!state;}
    data.value=JSON.stringify(draft.cells);data.disabled=resize.disabled=apply.disabled=!state;
    status.textContent=state?`${state.rows} × ${state.columns}; ${state.cells.filter(c=>c!==null).length} material cells.`:'Solid parent shape; grid disabled.';
    data.setAttribute('aria-invalid','false');
  }
  function commit(next) {
    try {
      const valid=validateStudioGrid(next,materialNames);
      onChange(valid);state=valid;if(state)draft=validateStudioGrid(state,materialNames);sync();
    } catch(error) {
      enabled.checked=state!==undefined;status.textContent=`${error.message}. The scene keeps the last valid grid.`;
      data.setAttribute('aria-invalid','true');
    }
  }
  enabled.addEventListener('change',()=>commit(enabled.checked?draft:undefined));
  apply.addEventListener('click',()=>{
    try {commit({...draft,cells:JSON.parse(data.value)});}
    catch(error){status.textContent=`Invalid cell JSON. ${error.message}`;data.setAttribute('aria-invalid','true');}
  });
  resize.addEventListener('click',()=>{
    const rows=Number(inputs.rows.value),columns=Number(inputs.columns.value),gap=Number(inputs.gap.value);
    if(!Number.isInteger(rows)||rows<1||rows>MAX_STUDIO_GRID_AXIS||!Number.isInteger(columns)||columns<1||columns>MAX_STUDIO_GRID_AXIS||rows*columns>MAX_STUDIO_GRID_CELLS){
      status.textContent=`Grid dimensions must be 1..${MAX_STUDIO_GRID_AXIS}, with at most ${MAX_STUDIO_GRID_CELLS} cells.`;return;
    }
    const cells=Array.from({length:rows*columns},(_,i)=>{
      const row=Math.floor(i/columns),column=i%columns;
      return row<draft.rows&&column<draft.columns?draft.cells[row*draft.columns+column]:'';
    });
    commit({version:1,rows,columns,gap,cells});
  });
  sync();
  return {ownsControl:id=>owned.has(id),setState(value){state=validateStudioGrid(value,materialNames);draft=state??defaultStudioGrid();sync();}};
}

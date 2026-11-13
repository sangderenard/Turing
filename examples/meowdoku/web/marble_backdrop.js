// The existing Studio backdrop roll, with independently sized material sheets.
// Preview and photo use these same faces; the saved marble mesh is untouched.
import {validateStudioGrid,studioGridCells,createStudioGridControls,validateStudioArtwork,studioArtworkFaces} from './marble_studio_grid.js';
import {TABLE_TOP_Y} from './marble_studio_table.js';
import {fabricWallMesh,fabricTableDrapePose,fabricCycloramaPose,fabricMeshAtPositions,FABRIC_GRID_SIDE,FABRIC_GRID_MAX_SIDE} from './marble_backdrop_fabric.js';
export const MAX_BACKDROP_LAYERS = 6;
export const MAX_BACKDROP_SIZE = 64;
export const BACKDROP_CIRCLE_SEGMENTS = 96;

export function defaultBackdropLayers() {
  return Array.from({length: MAX_BACKDROP_LAYERS}, (_, i) => ({
    material: "", width: 8 - 0.44 * i, height: 8 - 0.44 * i,
  }));
}

export function tallMountedFabricLayer(layer) {
  return {...layer,material:'fabric_muslin_white',shape:'rectangle',
    width:14,height:18,heightOffGround:2.5,stageDistance:5,
    lockToBackWall:false,lockToTableBack:false,aspectLocked:false,pitch:0,
    surfaceBehavior:'fabric',fabricPose:'wall',fabricGridSide:16,
    fabricAttachmentPoints:Array.from({length:16},(_,column)=>240+column),
    fabricRingPoints:[],fabricTiebackPoints:[],fabricFrozen:false};
}

export function validateBackdropLayers(value, materialNames) {
  if (!Array.isArray(value) || value.length !== MAX_BACKDROP_LAYERS)
    throw new RangeError("Backdrop settings must contain six layers");
  const names = new Set(materialNames);
  return value.map((layer) => {
    if (!layer || typeof layer !== "object" || typeof layer.material !== "string" ||
        (layer.material !== "" && !names.has(layer.material)))
      throw new RangeError("Unknown backdrop layer material");
    for (const key of ["width", "height"])
      if (!Number.isFinite(layer[key]) || layer[key] < 0.4 || layer[key] > MAX_BACKDROP_SIZE)
        throw new RangeError(`Backdrop ${key} must be between 0.4 and ${MAX_BACKDROP_SIZE}`);
    if (layer.shape !== undefined && !["rectangle", "ellipse"].includes(layer.shape))
      throw new RangeError("Unknown backdrop shape");
    if(layer.surfaceBehavior!==undefined&&!['cardstock','fabric'].includes(layer.surfaceBehavior))
      throw new RangeError('Unknown backdrop surface behavior');
    if(layer.fabricPose!==undefined&&!['wall','table'].includes(layer.fabricPose))
      throw new RangeError('Unknown backdrop fabric pose');
    const fabricSide=layer.fabricGridSide??FABRIC_GRID_SIDE;
    if(!Number.isInteger(fabricSide)||fabricSide<2||fabricSide>FABRIC_GRID_MAX_SIDE)
      throw new RangeError('Fabric grid side must be between 2 and 16');
    if(layer.fabricAttachmentPoints!==undefined&&(!Array.isArray(layer.fabricAttachmentPoints)||
       layer.fabricAttachmentPoints.length>fabricSide**2||new Set(layer.fabricAttachmentPoints).size!==layer.fabricAttachmentPoints.length||
       !layer.fabricAttachmentPoints.every(node=>Number.isInteger(node)&&node>=0&&node<fabricSide**2)))
      throw new RangeError(`Fabric attachment points must be distinct node IDs from 0 to ${fabricSide**2-1}`);
    if(layer.fabricRingPoints!==undefined&&(!Array.isArray(layer.fabricRingPoints)||
       new Set(layer.fabricRingPoints).size!==layer.fabricRingPoints.length||
       !layer.fabricRingPoints.every(node=>Number.isInteger(node)&&node>=0&&node<fabricSide**2)))
      throw new RangeError('Curtain rings must name distinct fabric nodes');
    if(layer.fabricTiebackPoints!==undefined&&(!Array.isArray(layer.fabricTiebackPoints)||
       new Set(layer.fabricTiebackPoints.map(item=>item?.node)).size!==layer.fabricTiebackPoints.length||
       !layer.fabricTiebackPoints.every(item=>Number.isInteger(item?.node)&&item.node>=0&&item.node<fabricSide**2&&
         Array.isArray(item.anchor)&&item.anchor.length===3&&item.anchor.every(Number.isFinite))||
       layer.fabricTiebackPoints.some(item=>layer.fabricRingPoints?.includes(item.node))))
      throw new RangeError('Tiebacks need distinct nodes and finite xyz anchors, separate from rings');
    if(layer.fabricFrozen!==undefined&&typeof layer.fabricFrozen!=='boolean')
      throw new RangeError('Invalid backdrop fabric freeze state');
    if(layer.surfaceBehavior==='fabric'&&(layer.shape==='ellipse'||layer.grid||layer.artwork))
      throw new RangeError('Fabric requires a solid rectangular backdrop wall');
    if (layer.centerY !== undefined && (!Number.isFinite(layer.centerY) || Math.abs(layer.centerY) > 32))
      throw new RangeError("Backdrop centre height must be between -32 and 32");
    for (const [key, min, max] of [["heightOffGround", 0, 32], ["stageDistance", 0, 64],
      ["pitch", -30, 30], ["curveRadius", 0, 8]])
      if (layer[key] !== undefined && (!Number.isFinite(layer[key]) || layer[key] < min || layer[key] > max))
        throw new RangeError(`Backdrop ${key} must be between ${min} and ${max}`);
    for (const key of ["lockToBackWall", "lockToTableBack", "aspectLocked"])
      if (layer[key] !== undefined && typeof layer[key] !== "boolean") throw new RangeError(`Invalid backdrop ${key}`);
    if (layer.lockToBackWall && layer.lockToTableBack) throw new RangeError("Backdrop cannot lock to both back planes");
    const grid=validateStudioGrid(layer.grid,materialNames);
    if(grid && layer.shape==='ellipse')throw new RangeError('A structured backdrop grid requires a rectangle');
    const artwork=validateStudioArtwork(layer.artwork,materialNames);
    if(artwork && (grid||layer.shape==='ellipse'))throw new RangeError('Surface artwork needs a solid rectangular backdrop');
    // Optional fields keep legacy rectangular profiles byte-for-byte in shape.
    return {material: layer.material, width: layer.width, height: layer.height,
      ...(layer.shape === undefined ? {} : {shape: layer.shape}),
      ...(layer.surfaceBehavior===undefined?{}:{surfaceBehavior:layer.surfaceBehavior}),
      ...(layer.fabricPose===undefined?{}:{fabricPose:layer.fabricPose}),
      ...(layer.fabricGridSide===undefined?{}:{fabricGridSide:layer.fabricGridSide}),
      ...(layer.fabricAttachmentPoints===undefined?{}:{fabricAttachmentPoints:[...layer.fabricAttachmentPoints]}),
      ...(layer.fabricRingPoints===undefined?{}:{fabricRingPoints:[...layer.fabricRingPoints]}),
      ...(layer.fabricTiebackPoints===undefined?{}:{fabricTiebackPoints:layer.fabricTiebackPoints.map(item=>({node:item.node,anchor:[...item.anchor]}))}),
      ...(layer.fabricFrozen===undefined?{}:{fabricFrozen:layer.fabricFrozen}),
      ...(layer.centerY === undefined ? {} : {centerY: layer.centerY}),
      ...Object.fromEntries(["heightOffGround", "stageDistance", "pitch", "curveRadius",
        "lockToBackWall", "lockToTableBack", "aspectLocked"].filter(key => layer[key] !== undefined).map(key => [key, layer[key]])),
      ...(grid === undefined ? {} : {grid}),...(artwork===undefined?{}:{artwork})};
  });
}

export function backdropFaces({count, material, sweep, layers, wallZ = -3.1, floorY = -1.02, table,
    fabricPositions}, ids, onPart) {
  if (!Number.isInteger(count) || count < 0 || count > MAX_BACKDROP_LAYERS)
    throw new RangeError("Backdrop count must be between zero and six");
  if (!count) return [];
  if (!Number.isFinite(wallZ)) throw new RangeError("Backdrop mounting plane must be finite");
  if (!Number.isFinite(floorY)) throw new RangeError("Backdrop floor plane must be finite");
  if (table?.enabled && (!Number.isFinite(table.length) || table.length <= 0)) throw new RangeError("Invalid backdrop table length");
  const step = 0.012;
  const faces = [], n = [0, 0, 1];
  if (material === "calibration") {
    // Preserve the existing eight-patch chart as a separate global mode.
    const chartMats = ["calibration_white", "calibration_red", "calibration_green", "calibration_blue",
      "calibration_yellow", "calibration_magenta", "calibration_cyan", "calibration_gray20"];
    const z = wallZ + step;
    chartMats.forEach((name, i) => {
      const cx = (i % 4 - 1.5) * 2, cy = (0.5 - Math.floor(i / 4)) * 4;
      const a = [cx - 0.9, cy - 1.8, z], b = [cx + 0.9, cy - 1.8, z];
      const c = [cx + 0.9, cy + 1.8, z], d = [cx - 0.9, cy + 1.8, z];
      faces.push([a, b, c, ids[name], [n,n,n]], [a, c, d, ids[name], [n,n,n]]);
      onPart?.({id:`chart-${i}`,bodyId:`backdrop-chart-${i}`,construction:"planar-sheet",closed:false,convex:true,faces:faces.slice(-2)});
    });
    return faces;
  }
  const valid = validateBackdropLayers(layers, Object.keys(ids));
  if (!Number.isInteger(ids[material])) throw new RangeError("Unknown default backdrop material");
  for (let i = 0; i < count; i++) {
    const first=faces.length;
    const pendingParts=[];
    const flushParts=()=>{for(const part of pendingParts)onPart?.(part);pendingParts.length=0;};
    const layer = valid[i], w = layer.width / 2, h = layer.height / 2;
    const tableBack=!!layer.lockToTableBack&&!!table?.enabled;
    const planeZ=tableBack?-table.length/2:layer.lockToBackWall===false?-(layer.stageDistance??-wallZ):wallZ;
    const mat = ids[layer.material || material], z = planeZ + step * (i + 1);
    const cy = layer.centerY ?? 0;
    const pitch=(layer.pitch??0)*Math.PI/180, radius=layer.curveRadius??0;
    const legacySweep=radius===0&&pitch===0&&layer.heightOffGround===undefined&&
      layer.lockToTableBack!==true&&layer.lockToBackWall!==false;
    const baseY=(tableBack?TABLE_TOP_Y:floorY)+(layer.heightOffGround??0)+step*(i+1);
    const curved=sweep&&layer.shape!=='ellipse'&&radius>0;
    const bottomY=layer.heightOffGround!==undefined||curved||tableBack
      ?baseY+(curved?radius*(1-Math.sin(pitch)):0):cy-h;
    const wallPoint=([x,y,oldZ])=>[x,bottomY+y-(cy-h),oldZ-Math.tan(pitch)*(y-(cy-h))];
    const wallNormal=[0,Math.sin(pitch),Math.cos(pitch)];
    const sweepPoint=(x,depth)=>{
      const arc=radius*Math.cos(pitch);
      if(radius<=0||depth>=arc)return [x,baseY,(legacySweep?planeZ:z)+depth];
      const theta=Math.acos(Math.max(-1,Math.min(1,Math.cos(pitch)-depth/radius)));
      return [x,baseY+radius*(1-Math.sin(theta)),z+depth];
    };
    const addSweep=(left,right,start,end,cellMat,part)=>{
      const from=faces.length,arc=radius*Math.cos(pitch);
      const cuts=[start];
      if(radius>0){for(let j=1;j<=12;j++){const theta=pitch+(Math.PI/2-pitch)*j/12;
        const d=radius*(Math.cos(pitch)-Math.cos(theta));if(d>start+1e-7&&d<end-1e-7)cuts.push(d);}}
      if(arc>start+1e-7&&arc<end-1e-7)cuts.push(arc);
      cuts.push(end);cuts.sort((a,b)=>a-b);
      for(let j=1;j<cuts.length;j++){
        const d0=cuts[j-1],d1=cuts[j],a=sweepPoint(left,d0),b=sweepPoint(right,d0),
          c=sweepPoint(right,d1),d=sweepPoint(left,d1);
        const middle=(d0+d1)/2,theta=radius>0&&middle<arc
          ?Math.acos(Math.max(-1,Math.min(1,Math.cos(pitch)-middle/radius))):Math.PI/2;
        const normal=radius===0?[0,1,0]:[0,Math.sin(theta),Math.cos(theta)];
        faces.push([d,c,b,cellMat,[normal,normal,normal]],[d,b,a,cellMat,[normal,normal,normal]]);
      }
      onPart?.({...part,faces:faces.slice(from)});
    };
    const transformWalls=()=>{if(pitch===0&&layer.heightOffGround===undefined&&!curved&&!tableBack)return;
      for(const face of faces.slice(first))if(face[4]?.[0]===n){
      for(let v=0;v<3;v++){face[v]=wallPoint(face[v]);face[4][v]=wallNormal;}
    }};
    if(layer.grid) {
      for(const cell of studioGridCells(layer.grid,layer.width,layer.height)) {
        const w=cell.width/2,h=cell.height/2,x=cell.x,y=cy+cell.y,cellMat=ids[cell.material||layer.material||material];
        if(!Number.isInteger(cellMat)||cellMat<0)throw new RangeError('Unknown grid material ID');
        faces.push([[x-w,y-h,z],[x+w,y-h,z],[x+w,y+h,z],cellMat,[n,n,n]],
          [[x-w,y-h,z],[x+w,y+h,z],[x-w,y+h,z],cellMat,[n,n,n]]);
        pendingParts.push({id:`layer-${i}-${cell.id}-wall`,bodyId:`backdrop-layer-${i}-${cell.id}`,construction:'planar-sheet',closed:false,convex:true,faces:faces.slice(-2)});
      }
      transformWalls();
      flushParts();
      const depth=Math.max(1.2-(4-w)*0.3,radius*Math.cos(pitch)+.1);
      if(sweep && depth>0.1)for(const cell of studioGridCells(layer.grid,layer.width,depth)) {
        const cw=cell.width/2,ch=cell.height/2,x=cell.x,center=depth/2-cell.y;
        const cellMat=ids[cell.material||layer.material||material];
        addSweep(x-cw,x+cw,center-ch,center+ch,cellMat,
          {id:`layer-${i}-${cell.id}-sweep`,bodyId:`backdrop-layer-${i}-${cell.id}`,construction:'curved-sweep',closed:false,convex:false});
      }
      continue;
    }
    if (layer.shape === "ellipse") {
      // Real nested opaque mats, not a screen-space texture. Both render routes
      // consume these exact triangles and the existing physical depth order.
      const rim = angle => [w * Math.cos(angle), cy + h * Math.sin(angle), z];
      for (let segment = 0; segment < BACKDROP_CIRCLE_SEGMENTS; segment++) {
        const a = segment * 2 * Math.PI / BACKDROP_CIRCLE_SEGMENTS;
        const b = (segment + 1) * 2 * Math.PI / BACKDROP_CIRCLE_SEGMENTS;
        faces.push([[0,cy,z],rim(a),rim(b),mat,[n,n,n]]);
      }
    } else {
      const wallFirst=faces.length;
      faces.push([[-w,cy-h,z],[w,cy-h,z],[w,cy+h,z],mat,[n,n,n]],
        [[-w,cy-h,z],[w,cy+h,z],[-w,cy+h,z],mat,[n,n,n]]);
      if(layer.surfaceBehavior==='fabric'){
        transformWalls();
        const wall=fabricWallMesh(faces.slice(wallFirst),{side:layer.fabricGridSide??FABRIC_GRID_SIDE});
        const posed=layer.fabricPose==='table'?fabricTableDrapePose(wall,table):
          sweep?fabricCycloramaPose(wall,{floorY}):wall;
        faces.splice(wallFirst,2,...(fabricPositions?.[i]
          ?fabricMeshAtPositions(posed,fabricPositions[i]):posed).faces);
      }
    }
    pendingParts.push({id:`layer-${i}-wall`,bodyId:`backdrop-layer-${i}`,
      construction:layer.surfaceBehavior==='fabric'?'spring-sheet':'planar-sheet',
      closed:false,convex:layer.surfaceBehavior!=='fabric',faces:faces.slice(first)});
    if(layer.artwork)faces.push(...studioArtworkFaces(layer.artwork,layer.width,layer.height,ids,
      ([x,y])=>[x,cy+y,z+.001],n,part=>pendingParts.push({...part,id:`layer-${i}-art-${part.id}`,bodyId:`backdrop-layer-${i}-art-${part.id}`})));
    transformWalls();
    flushParts();
    if (sweep && layer.shape !== "ellipse" && layer.surfaceBehavior !== 'fabric') {
      // Keep the existing sweep proportions, now derived from this sheet's
      // authored size instead of an implicit layer-index inset.
      const sweepDepth = Math.max(1.2 - (4 - w) * 0.3,radius*Math.cos(pitch)+.1);
      if (sweepDepth > 0.1) {
        addSweep(-w,w,0,sweepDepth,mat,
          {id:`layer-${i}-sweep`,bodyId:`backdrop-layer-${i}`,construction:"curved-sweep",closed:false,convex:false});
      }
    }
  }
  return faces;
}

export function createBackdropLayerControls(container, materialNames, onChange = () => {}) {
  const doc = container.ownerDocument, names = [...materialNames].sort();
  let layers = defaultBackdropLayers();
  const editors = layers.map((layer, index) => {
    const panel = doc.createElement("fieldset"); panel.className = "wide grid";
    const legend = doc.createElement("legend"); legend.textContent = `Backdrop ${index + 1} (${index === 0 ? "back" : "toward subject"})`;
    panel.append(legend);
    const label = doc.createElement("label"); label.className = "wide"; label.textContent = "Material";
    const select = doc.createElement("select"); select.id = `backdrop-layer-${index}-material`;
    for (const name of ["", ...names]) {
      const option = doc.createElement("option"); option.value = name;
      option.textContent = name ? name.replaceAll("_", " ") : "Use default backdrop material";
      select.append(option);
    }
    select.addEventListener("change", () => { layers[index].material = select.value; onChange(); });
    label.append(select); panel.append(label);
    const shapeLabel = doc.createElement("label"); shapeLabel.textContent = "Mat shape";
    const shape = doc.createElement("select"); shape.id = `backdrop-layer-${index}-shape`;
    for (const [value, text] of [["rectangle", "Rectangle"], ["ellipse", "Ellipse / circle"]]) {
      const option = doc.createElement("option"); option.value = value; option.textContent = text; shape.append(option);
    }
    shape.value = "rectangle";
    shape.addEventListener("change", () => {
      if((layers[index].grid||layers[index].artwork||layers[index].surfaceBehavior==='fabric') && shape.value==='ellipse'){
        shape.value=layers[index].shape??'rectangle';return;
      }
      layers[index].shape = shape.value; onChange();
    });
    shapeLabel.append(shape); panel.append(shapeLabel);
    const behaviorLabel=doc.createElement('label'),behavior=doc.createElement('select');
    behaviorLabel.textContent='Surface behavior';behavior.id=`backdrop-layer-${index}-surfaceBehavior`;
    for(const [value,name] of [['cardstock','Cardstock (rigid)'],['fabric','Fabric mesh (native Nodus motion)']]){
      const option=doc.createElement('option');option.value=value;option.textContent=name;behavior.append(option);
    }
    behavior.addEventListener('change',()=>{
      if(behavior.value==='fabric'&&(layers[index].shape==='ellipse'||layers[index].grid||layers[index].artwork)){
        behavior.value=layers[index].surfaceBehavior??'cardstock';return;
      }
      layers[index].surfaceBehavior=behavior.value;onChange();
    });
    behaviorLabel.append(behavior);panel.append(behaviorLabel);
    const poseLabel=doc.createElement('label'),pose=doc.createElement('select');
    poseLabel.textContent='Fabric start pose';pose.id=`backdrop-layer-${index}-fabricPose`;
    for(const [value,name] of [['wall','Hang at back wall'],['table','Extend forward over stage table']]){
      const option=doc.createElement('option');option.value=value;option.textContent=name;pose.append(option);
    }
    pose.addEventListener('change',()=>{
      layers[index].fabricPose=pose.value;
      if(layers[index].fabricAttachmentPoints===undefined){
        const side=layers[index].fabricGridSide??FABRIC_GRID_SIDE;
        attachments.value=(pose.value==='table'
          ?Array.from({length:side},(_,column)=>column)
          :Array.from({length:side},(_,column)=>(side-1)*side+column)).join(',');
      }
      onChange();
    });poseLabel.append(pose);panel.append(poseLabel);
    const attachmentLabel=doc.createElement('label'),attachments=doc.createElement('input');
    attachmentLabel.textContent='Attachment points (node IDs)';
    attachments.type='text';attachments.id=`backdrop-layer-${index}-fabricAttachmentPoints`;
    attachments.value='72,80';
    attachments.addEventListener('change',()=>{
      const text=attachments.value.trim();
      const points=text?text.split(/[\s,]+/).map(Number):[];
      const candidate=layers.map((layer,i)=>i===index?{...layer,fabricAttachmentPoints:points}:layer);
      try{layers=validateBackdropLayers(candidate,names);attachments.setCustomValidity?.('');onChange();}
      catch(error){attachments.setCustomValidity?.(error.message);attachments.reportValidity?.();}
    });attachmentLabel.append(attachments);panel.append(attachmentLabel);
    const ringLabel=doc.createElement('label'),rings=doc.createElement('input');
    ringLabel.textContent='Curtain ring nodes (slide sideways on their mount)';
    rings.type='text';rings.id=`backdrop-layer-${index}-fabricRingPoints`;
    rings.value=(layer.fabricRingPoints??[]).join(',');
    rings.addEventListener('change',()=>{
      const text=rings.value.trim(),points=text?text.split(/[\s,]+/).map(Number):[];
      try{layers=validateBackdropLayers(layers.map((item,i)=>i===index?{...item,fabricRingPoints:points}:item),names);
        rings.setCustomValidity?.('');onChange();}
      catch(error){rings.setCustomValidity?.(error.message);rings.reportValidity?.();}
    });ringLabel.append(rings);panel.append(ringLabel);
    const tiebackLabel=doc.createElement('label'),tiebacks=doc.createElement('input');
    tiebackLabel.textContent='Tiebacks / hooks (node@x:y:z; …)';
    tiebacks.type='text';tiebacks.id=`backdrop-layer-${index}-fabricTiebackPoints`;
    tiebacks.value=(layer.fabricTiebackPoints??[]).map(item=>`${item.node}@${item.anchor.join(':')}`).join('; ');
    tiebacks.addEventListener('change',()=>{
      const text=tiebacks.value.trim(),points=text?text.split(';').map(part=>{
        const [node,anchor]=part.trim().split('@');
        return {node:Number(node),anchor:anchor?.split(':').map(Number)};
      }):[];
      try{layers=validateBackdropLayers(layers.map((item,i)=>i===index?{...item,fabricTiebackPoints:points}:item),names);
        tiebacks.setCustomValidity?.('');onChange();}
      catch(error){tiebacks.setCustomValidity?.(error.message);tiebacks.reportValidity?.();}
    });tiebackLabel.append(tiebacks);panel.append(tiebackLabel);
    const gridSideLabel=doc.createElement('label'),gridSide=doc.createElement('select');
    gridSideLabel.textContent='Fabric grid';gridSide.id=`backdrop-layer-${index}-fabricGridSide`;
    for(const side of [9,16]){const option=doc.createElement('option');option.value=String(side);
      option.textContent=`${side} × ${side} (${side*side} nodes)`;gridSide.append(option);}
    gridSide.addEventListener('change',()=>{
      layers[index].fabricGridSide=Number(gridSide.value);
      delete layers[index].fabricAttachmentPoints;
      delete layers[index].fabricRingPoints;delete layers[index].fabricTiebackPoints;
      rings.value='';tiebacks.value='';
      attachments.value=(layers[index].fabricPose==='table'
        ?Array.from({length:Number(gridSide.value)},(_,column)=>column)
        :Array.from({length:Number(gridSide.value)},(_,column)=>(Number(gridSide.value)-1)*Number(gridSide.value)+column)).join(',');
      onChange();
    });gridSideLabel.append(gridSide);panel.append(gridSideLabel);
    const freezeLabel=doc.createElement('label'),freeze=doc.createElement('input');
    freeze.type='checkbox';freeze.id=`backdrop-layer-${index}-fabricFrozen`;freeze.disabled=true;
    freezeLabel.append(freeze,' Saved freeze state (use Pause fabric for live motion)');panel.append(freezeLabel);
    freeze.addEventListener('change',()=>{layers[index].fabricFrozen=freeze.checked;onChange();});
    const sizes = {};
    for (const key of ["width", "height", "centerY", "heightOffGround", "stageDistance", "pitch", "curveRadius"]) {
      const sizeLabel = doc.createElement("label"); sizeLabel.textContent = ({centerY:"Centre height",heightOffGround:"Height off ground/table",stageDistance:"Distance from stage",pitch:"Pitch (degrees)",curveRadius:"Sweep curve radius"})[key]??`Mat ${key}`;
      const output = doc.createElement("output"), input = doc.createElement("input");
      input.type = "range"; input.min = ({centerY:-32,pitch:-30,heightOffGround:0,stageDistance:0,curveRadius:0})[key]??0.4;
      input.max = ({centerY:32,pitch:30,heightOffGround:32,stageDistance:64,curveRadius:8})[key]??MAX_BACKDROP_SIZE; input.step = "0.01";
      input.id = `backdrop-layer-${index}-${key}`;
      const initial=layer[key]??(key==='stageDistance'?3.1:0);
      input.value = String(initial); output.value = initial.toFixed(2);
      input.addEventListener("input", () => {
        const ratio=layers[index].width/layers[index].height;
        layers[index][key] = Number(input.value); output.value = layers[index][key].toFixed(2);
        if(layers[index].aspectLocked&&['width','height'].includes(key)){
          const other=key==='width'?'height':'width';
          layers[index][other]=Math.min(MAX_BACKDROP_SIZE,Math.max(.4,key==='width'?layers[index].width/ratio:layers[index].height*ratio));
          sizes[other].input.value=String(layers[index][other]);sizes[other].output.value=layers[index][other].toFixed(2);
        }
        onChange();
      });
      sizeLabel.append(output, input); panel.append(sizeLabel); sizes[key] = {input, output};
    }
    const switches={};
    for(const [key,label] of [["lockToBackWall","Lock to back wall"],["lockToTableBack","Lock to table back"],["aspectLocked","Link width and height aspect ratio"]]){
      const wrapper=doc.createElement('label'),input=doc.createElement('input');input.type='checkbox';
      input.id=`backdrop-layer-${index}-${key}`;input.checked=layer[key]??(key==='lockToBackWall');
      input.addEventListener('change',()=>{
        layers[index][key]=input.checked;
        if(input.checked&&key==='lockToBackWall'){layers[index].lockToTableBack=false;switches.lockToTableBack.checked=false;}
        if(input.checked&&key==='lockToTableBack'){layers[index].lockToBackWall=false;switches.lockToBackWall.checked=false;}
        onChange();
      });wrapper.append(input,label);panel.append(wrapper);switches[key]=input;
    }
    const tall=doc.createElement('button');tall.type='button';
    tall.textContent='Set up tall mounted fabric';
    tall.addEventListener('click',()=>{
      const next=tallMountedFabricLayer(layers[index]);
      layers=validateBackdropLayers(layers.map((layer,i)=>i===index?next:layer),names);
      select.value=next.material;shape.value=next.shape;behavior.value=next.surfaceBehavior;
      pose.value=next.fabricPose;gridSide.value='16';attachments.value=next.fabricAttachmentPoints.join(',');
      rings.value='';tiebacks.value='';freeze.checked=false;
      for(const [key,{input,output}] of Object.entries(sizes)){
        const value=next[key]??0;input.value=String(value);output.value=value.toFixed(2);
      }
      for(const [key,input] of Object.entries(switches))input.checked=next[key]??false;
      onChange();
    });panel.append(tall);
    const quick=doc.createElement('label'),factor=doc.createElement('input'),apply=doc.createElement('button');
    quick.textContent='Quick scale width and height';factor.type='number';factor.min='.1';factor.max='8';factor.step='.1';factor.value='1.2';
    apply.type='button';apply.textContent='Scale mat';apply.addEventListener('click',()=>{
      const value=Number(factor.value);if(!Number.isFinite(value)||value<.1||value>8)return;
      const width=layers[index].width*value,height=layers[index].height*value;
      if(width<.4||height<.4||width>MAX_BACKDROP_SIZE||height>MAX_BACKDROP_SIZE)return;
      layers[index].width=width;layers[index].height=height;
      for(const key of ['width','height']){sizes[key].input.value=String(layers[index][key]);sizes[key].output.value=layers[index][key].toFixed(2);}
      onChange();
    });quick.append(factor,apply);panel.append(quick);
    const grid=createStudioGridControls(panel,`backdrop-layer-${index}-grid`,names,value=>{
      const next={...layers[index]};if(value===undefined)delete next.grid;else {next.grid=value;next.surfaceBehavior='cardstock';behavior.value='cardstock';}
      const candidate=layers.map((layer,i)=>i===index?next:layer);
      layers=validateBackdropLayers(candidate,names);onChange();
    });
    container.append(panel);
    return {panel, select, shape, behavior, pose, gridSide, attachments, rings, tiebacks, freeze, sizes,switches,grid};
  });
  return {
    getState: () => validateBackdropLayers(layers,names),
    setState(value) {
      layers = validateBackdropLayers(value, names);
      editors.forEach(({select, shape, behavior, pose, gridSide, attachments, rings, tiebacks, freeze, sizes,switches,grid}, i) => {
        select.value = layers[i].material;
        shape.value = layers[i].shape ?? "rectangle";
        behavior.value=layers[i].surfaceBehavior??'cardstock';freeze.checked=layers[i].fabricFrozen??false;
        pose.value=layers[i].fabricPose??'wall';
        const side=layers[i].fabricGridSide??FABRIC_GRID_SIDE;
        gridSide.value=String(side);
        attachments.value=(layers[i].fabricAttachmentPoints??(pose.value==='table'
          ?Array.from({length:side},(_,column)=>column)
          :Array.from({length:side},(_,column)=>(side-1)*side+column))).join(',');
        rings.value=(layers[i].fabricRingPoints??[]).join(',');
        tiebacks.value=(layers[i].fabricTiebackPoints??[]).map(item=>`${item.node}@${item.anchor.join(':')}`).join('; ');
        freeze.disabled=true;
        grid.setState(layers[i].grid);
        for (const [key, {input, output}] of Object.entries(sizes)) {
          const value=layers[i][key]??(key==='stageDistance'?3.1:0);
          input.value = String(value); output.value = value.toFixed(2);
        }
        for(const [key,input] of Object.entries(switches))input.checked=layers[i][key]??(key==='lockToBackWall');
      });
    },
    show(count, calibration = false) {
      editors.forEach(({panel}, i) => { panel.hidden = i >= count || calibration; });
      container.hidden = count === 0 || calibration;
    },
  };
}

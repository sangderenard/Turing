import {defaultLightingRacks,validateLightingRacks,layoutLightingRackFixtures} from './marble_lighting_racks.js';
import {applyLightingRackOverrides} from './marble_lighting_rack_override.js';
import {createLightingRackControls} from './marble_lighting_rack_controls.js';
import {validateSceneEngines,validateSceneMaterialEdits} from "./marble_scene_materials.js";
import {validateLightTools,migrateLightTools} from "./marble_light_tools.js";
import {createKickerHeightGuides} from "./marble_kicker_guide_controls.js";
import {createLightStandControls,validateLightStands} from "./marble_light_stand_settings.js";
import {validateShopLights,SHOP_LIGHT_MODELS} from "./marble_shop_lights.js";
import {createShopLightControls} from "./marble_shop_light_controls.js";
import {RADIAL_LIGHTS,radialLightPose,radialLightPosition} from "./marble_light_placement.js";
import {createFixtureSizeControls,validateFixtureSizes,resolveUnderlightFit,validateUnderlightFitPose} from "./marble_fixture_sizes.js";
import {createLightingPoseControls, createLightingPoseState, validateLightingPoseState} from "./marble_lighting.js";
import {tableLightRoles, studioPlacementWarnings} from "./marble_table_lights.js";
import {studioRoomBounds} from "./marble_room.js";
import {createLightingProfiles, roomSettingLeaves, validateRoomOverrides, resolveRoomProfile} from "./marble_lighting_profiles.js";
import {createBackdropLayerControls, validateBackdropLayers} from "./marble_backdrop.js";
import {STUDIO_LIGHTS, createLightStrengthControls, validateLightStrengths} from "./marble_light_strengths.js";
import {createStudioSurfaceControls, validateStudioSurfaces} from "./marble_studio_surfaces.js";
import {createStudioTableControls, validateStudioTable, tableFloorDrop, FLOOR_FIXTURE_INDICES} from "./marble_studio_table.js";

import {createStudioGoboControls, validateStudioGobos} from "./marble_studio_gobos.js";
import {createStudioScaleControls, defaultStudioScale, validateStudioScale, studioScaleTransform} from "./marble_studio_scale.js";

export const STUDIO_LIGHT_IDS = STUDIO_LIGHTS.map(light => light.id);
const labels = STUDIO_LIGHTS.map(light => light.label);
export const LIGHTING_PROFILE_FIELDS = ["key-shape", "light-x", "light-y", "light-z",
  "key-filter-left", "key-filter-right", "light-strength-left", "light-strength-right",
  "light-bg-strength", "background-filter", "light-kicker-left-strength", "kicker-left-filter",
  "light-kicker-right-strength", "kicker-right-filter", "light-fill-upper-strength", "fill-upper-filter",
  "light-fill-lower-strength", "fill-lower-filter", "floor-pattern", "floor-material-a", "floor-material-b",
  "light-table-left-strength", "table-left-filter", "light-table-right-strength", "table-right-filter", "light-under-strength", "under-filter",
  "wall-pattern", "wall-material-a", "wall-material-b", "ceiling-pattern", "ceiling-material-a", "ceiling-material-b",
  "backdrop-layers", "backdrop-material", "backdrop-sweep", "checker", "window-glass", "room-scale"];

// Validate against the live catalogue and the actual controls, not a second
// hand-maintained list of material IDs or slider bounds. No writes until valid.
function validateStudioLightingBase(value, fields, materialNames) {
  if (!value || value.version !== 1 || !value.controls || typeof value.controls !== "object")
    throw new Error("Unsupported Studio lighting profile");
  const legacy = Array.isArray(value.poses?.poses) && value.poses.poses.length === 7;
  const newFields = {"light-table-left-strength":0,"table-left-filter":"#ffffff",
    "light-table-right-strength":0,"table-right-filter":"#ffffff","light-under-strength":0,"under-filter":"#ffffff"};
  const controls = {};
  for (const id of LIGHTING_PROFILE_FIELDS) {
    const el = fields[id];
    // Older profiles retain the former backdrop mounting distance.
    const v = value.controls[id] === undefined ?
      (id === "room-scale" ? 1 : legacy ? newFields[id] : undefined) : value.controls[id];
    if (!el) throw new Error(`Lighting control unavailable: ${id}`);
    if (el.type === "checkbox") {
      if (typeof v !== "boolean") throw new Error(`Invalid ${id}`);
    } else if (el.type === "range") {
      if (!Number.isFinite(v) || v < Number(el.min) || v > Number(el.max) ||
          (id === "backdrop-layers" && !Number.isInteger(v))) throw new Error(`Invalid ${id}`);
    } else if (el.type === "color") {
      if (typeof v !== "string" || !/^#[0-9a-f]{6}$/i.test(v)) throw new Error(`Invalid ${id}`);
    } else if (typeof v !== "string" || !Array.from(el.options).some(option => option.value === v)) {
      throw new Error(`Unavailable option for ${id}`);
    }
    controls[id] = v;
  }
  const backdrops = validateBackdropLayers(value.backdrops, materialNames);
  const table = validateStudioTable(value.table, materialNames);
  const poses = legacy ? {version:1, poses:[
    ...validateLightingPoseState(value.poses, STUDIO_LIGHT_IDS.slice(0,7)).poses,
    ...createLightingPoseState(tableLightRoles().map((role,i)=>({id:STUDIO_LIGHT_IDS[7+i],
      pos:role.pos,dir:role.dir,aimAtSubject:true}))).poses,
  ]} : validateLightingPoseState(value.poses, STUDIO_LIGHT_IDS);
  const fixtureSizes=validateFixtureSizes(resolveUnderlightFit(validateFixtureSizes(value.fixtureSizes,false),table),value.lightTools===undefined);
  validateUnderlightFitPose(fixtureSizes,poses);
  const lightTools=validateLightTools(value.lightTools,fixtureSizes,controls["key-shape"],materialNames);
  return {version: 1, controls, poses, backdrops, fixtureSizes,lightTools,lightingRacks:validateLightingRacks(value.lightingRacks),
    sceneEngines:validateSceneEngines(value.sceneEngines),...(value.materialEdits===undefined?{}:{materialEdits:validateSceneMaterialEdits(value.materialEdits,materialNames)}),lightStands:validateLightStands(value.lightStands,materialNames),shopLights:validateShopLights(value.shopLights,materialNames),
    table, scale: validateStudioScale(value.scale), gobos: validateStudioGobos(value.gobos, materialNames),
    strengths: validateLightStrengths(value.strengths, fields),
    surfaces: validateStudioSurfaces(value.surfaces, fields, materialNames, {controls, backdrops, floorDrop:tableFloorDrop(table)})};
}

// The old v1 power split made Studio primary. Preserve both actual routes on
// load, but migrate its exposure values into A and Studio values into B overrides.
export function validateStudioLighting(value, fields, materialNames) {
  const base=validateStudioLightingBase(value,fields,materialNames);
  const inherited=[];
  base.strengths.lights.forEach((light,i)=>{
    if(light.exposure!==null) {
      const key=STUDIO_LIGHTS[i].strengthId;
      inherited.push({path:'controls/'+key,value:base.controls[key]});
      base.controls[key]=light.exposure*10;light.exposure=null;
    }
  });
  let given=value.roomOverrides?.values??[];
  if(value.lightTools===undefined){
    // Old Room accessory/dimension leaves described the same incumbent meshes.
    // Migrate both routes before exposing the one canonical tool source.
    const roomSizes=JSON.parse(JSON.stringify(base.fixtureSizes));
    for(const item of given){const match=/^fixtureSizes\/lights\/@([^/]+)\/([^/]+)$/.exec(item.path);
      if(match){const light=roomSizes.lights.find(v=>v.id===match[1]);if(light)light[match[2]]=item.value;}}
    const roomTools=migrateLightTools(roomSizes);
    given=given.filter(item=>!/^fixtureSizes\/lights\/@[^/]+\/(snoot|snootLength|grid|gridCellsX|gridCellsY|gridDepth|gridThickness)$/.test(item.path));
    const exposureLeaves=new Map(roomSettingLeaves(base).map(v=>[v.path,v.value]));
    for(const item of roomSettingLeaves({...base,lightTools:roomTools}).filter(v=>v.path.startsWith('lightTools/')))
      if(item.value!==exposureLeaves.get(item.path))given.push(item);
  }
  const roomOverrides=validateRoomOverrides({version:value.roomOverrides?.version??1,
    values:[...inherited.filter(item=>!given.some(v=>v.path===item.path)),...given]},base,
    candidate=>validateStudioLightingBase(candidate,fields,materialNames));
  return {...base,roomOverrides};
}

export function setupStudioLighting(doc, initialFixtures, materialNames, {onChange, onBackdropChange, onRoomChange, onGoboChange = () => {}, fixtureVertices = null, standWarnings = null, captureMaterials = () => undefined, restoreMaterials = () => {}, lightGuideRole = null, validateShopPlacement = () => {}}) {
  let renderCache=null;
  const invalidateRenderState=()=>{renderCache=null;};
  const originalChange=onChange, originalRoom=onRoomChange, originalBackdrop=onBackdropChange, originalGobo=onGoboChange;
  onChange=(...args)=>{invalidateRenderState();originalChange(...args);};
  onRoomChange=(...args)=>{invalidateRenderState();originalRoom(...args);};
  onBackdropChange=(...args)=>{invalidateRenderState();originalBackdrop(...args);};
  onGoboChange=(...args)=>{invalidateRenderState();originalGobo(...args);};
  // Measuring a point does not author lighting. Preserve the displayed scene
  // identity across the pointerdown/click pair and selected-hit controls.
  const invalidateRenderEvent=event=>{
    if(event.target?.closest?.('#camera-toolbox'))return;
    if(event.type==='click'&&['view','photo-image'].includes(event.target?.id))return;
    invalidateRenderState();
  };
  for(const kind of ['input','change','click'])doc.addEventListener?.(kind,invalidateRenderEvent,true);
  const $ = id => doc.getElementById(id);
  const fields = Object.fromEntries(LIGHTING_PROFILE_FIELDS.map(id => [id, $(id)]));
  let acceptedKeyShape=fields['key-shape'].value,keyShapeStatus=null;
  let applying = false;
  let syncPoseShortcuts = () => {}, syncPlacementWarnings = () => {};
  let scale, table;
  const subjectCenterY = () => studioScaleTransform(scale?.getState() ?? defaultStudioScale(),
    Number($("radius-mm").value), table?.getState().enabled ?? true).subjectCenter[1];
  const fixtureClearance = (poseId,state,sizes,profile=null,tools=null) => {
    if(!fixtureVertices||!table||!scale)return null;
    const tableState=profile?.table??table.getState(),subject=studioScaleTransform(profile?.scale??scale.getState(),Number($("radius-mm").value),tableState.enabled);
    const bounds=studioRoomBounds(profile?.controls?.["room-scale"]??Number($("room-scale").value),tableFloorDrop(tableState));
    const floor=poseId==='underlight'?bounds.floorY:subject.subjectCenter[1]+subject.subjectRadius;
    const ceiling=poseId==='underlight'&&tableState.enabled?-1-.12:bounds.ceilingY;
    const vertices=fixtureVertices(poseId,state,subject.subjectCenter,sizes,tools??profile?.lightTools??fixtureSizes.tools.getState());
    if(!vertices?.length)return null;
    const ys=vertices.map(p=>p[1]);
    const low=Math.min(...ys),high=Math.max(...ys);
    return {floor,ceiling,low,high,valid:low>=floor-1e-9&&high<=ceiling+1e-9,
      message:poseId==='underlight'?'Underlight housing and grid must fit above the floor and below the table underside.':'Overlight housing must fit above the subject top and below the ceiling.'};
  };
  const validateFixturePose = (next,id,sizes,tools=null) => {
    if(!['underlight','fill-upper'].includes(id))return;
    if(id==='underlight')validateUnderlightFitPose(sizes??fixtureSizes.getState(),next);
    const clearance=fixtureClearance(id,next,sizes,null,tools);if(clearance&&!clearance.valid)throw new RangeError(clearance.message);
  };
  const poses = createLightingPoseControls($("light-pose-controls"), initialFixtures.map((fixture, i) => ({
    id: STUDIO_LIGHT_IDS[i], label: labels[i], pos: fixture.pos, dir: fixture.dir,
    // New kickers track the scaled subject; keep only the authored background splash.
    aimAtSubject: i !== 2,
  })), () => {syncPoseShortcuts(); syncPlacementWarnings(); if (!applying) onChange();}, {validateChange:validateFixturePose});
  // A shortcut to the existing pose, not another saved source of truth.
  // Profile loads and Advanced edits update this field without moving a light.
  const shortcutIds = new Set(), shortcutSync = [], shortcutErrors = new Map();
  const error = (id, message) => {
    if (message) shortcutErrors.set(id, message); else shortcutErrors.delete(id);
    const output = $(id); if (output) output.textContent = message;
    syncPlacementWarnings();
  };
  for (const [id, poseId, label, min] of [
    ["underlight-height", "underlight", "Underlight height", -50],
    ["top-fill-height", "fill-upper", "Top fill height", 0],
  ]) {
    const input = $(id); if (!input) continue;
    shortcutIds.add(id);
    const offset = () => poseId === "fill-upper" ? subjectCenterY() : 0;
    shortcutSync.push(state => {
      try {const y = offset(); input.value = String(state.poses.find(p => p.id === poseId).pos[1] - y); input.max = String(50 - y);
        const p=state.poses.find(p=>p.id===poseId),c=fixtureClearance(poseId,state);
        if(c){input.min=String(Math.max(-50,c.floor-(c.low-p.pos[1]))-y);input.max=String(Math.min(50,c.ceiling-(c.high-p.pos[1]))-y);}}
      catch { /* Keep the last display during a partial radius edit. */ }
    });
    input.addEventListener("input", () => {
      let base;
      try {base = offset();} catch (e) {error(`${id}-status`, e.message); return;}
      const height = input.value.trim() === "" ? NaN : Number(input.value), y = height + base;
      if (!Number.isFinite(height) || height < min || y < -50 || y > 50) {
        error(`${id}-status`, `${label} must be between ${min} and ${50 - base} Studio units.`); return;
      }
      error(`${id}-status`, "");
      const state = poses.getState(), pose = state.poses.find(p => p.id === poseId);
      if (pose.pos[1] !== y) {
        pose.pos[1]=y;
        try{validateFixturePose(state,poseId);}catch(e){error(`${id}-status`,e.message);return;}
        poses.setState(state);
      }
    });
  }
  // Overhead shortcuts author the existing pose, shared by mesh and emitter paths.
  // Ceiling drop measures to the emitter origin; geometry clearance remains enforced.
  const overheadBounds=()=>studioRoomBounds(Number($("room-scale").value),tableFloorDrop(table?.getState()??{enabled:false}));
  for(const field of ['drop','x','z','yaw','pitch','lock']) {
    const input=$(`overhead-fill-${field}`);if(!input)continue;shortcutIds.add(input.id);
    shortcutSync.push(state=>{
      const p=state.poses.find(p=>p.id==='fill-upper');
      if(field==='lock')input.checked=p.aimAtSubject;
      else {
        const value=field==='drop'?overheadBounds().ceilingY-p.pos[1]:field==='x'?p.pos[0]:field==='z'?p.pos[2]:p[field];
        input.value=String(Number(value.toFixed(3)));
        if(field==='yaw'||field==='pitch')input.disabled=p.aimAtSubject;
        if(field==='drop')input.max=String(overheadBounds().ceilingY+50);
      }
    });
    input.addEventListener(field==='lock'?'change':'input',()=>{
      const state=poses.getState(),p=state.poses.find(p=>p.id==='fill-upper');
      const value=field==='lock'?input.checked:input.value.trim()===''?NaN:Number(input.value);
      const limits=field==='drop'?[0,overheadBounds().ceilingY+50]:field==='yaw'?[-180,180]:field==='pitch'?[-90,90]:[-50,50];
      if(field!=='lock'&&(!Number.isFinite(value)||value<limits[0]||value>limits[1])){error('top-fill-height-status',`Enter a value between ${limits[0]} and ${limits[1]}.`);return;}
      if(field==='lock')p.aimAtSubject=value;
      else if(field==='drop')p.pos[1]=overheadBounds().ceilingY-value;
      else if(field==='x'||field==='z')p.pos[field==='x'?0:2]=value;
      else {if(p.aimAtSubject)return;p[field]=value;}
      try{validateLightingPoseState(state,STUDIO_LIGHT_IDS);validateFixturePose(state,'fill-upper');poses.setState(state);error('top-fill-height-status','');}
      catch(e){if(field==='lock')input.checked=poses.getState().poses.find(p=>p.id==='fill-upper').aimAtSubject;error('top-fill-height-status',e.message);}
    });
  }
  // Lower fill edits the shared absolute pose; wall distance is a readout, not a second owner.
  const lowerBounds=()=>studioRoomBounds(Number($("room-scale").value),tableFloorDrop(table?.getState()??{enabled:false}));
  for(const field of ['wall-gap','height','yaw','pitch','lock']) {
    const id=`lower-fill-${field}`,input=$(id);if(!input)continue;shortcutIds.add(id);
    shortcutSync.push(state=>{
      const p=state.poses.find(p=>p.id==='fill-lower');
      if(field==='lock')input.checked=p.aimAtSubject;
      else {
        const value=field==='wall-gap'?lowerBounds().frontZ-p.pos[2]:field==='height'?p.pos[1]:p[field];
        input.value=String(Number(value.toFixed(3)));
        if(field==='yaw'||field==='pitch')input.disabled=p.aimAtSubject;
        if(field==='wall-gap')input.max=String(lowerBounds().frontZ-lowerBounds().backZ);
      }
    });
    input.addEventListener(field==='lock'?'change':'input',()=>{
      const state=poses.getState(),p=state.poses.find(p=>p.id==='fill-lower');
      const value=field==='lock'?input.checked:input.value.trim()===''?NaN:Number(input.value);
      const bounds=lowerBounds(),limits=field==='wall-gap'?[0,bounds.frontZ-bounds.backZ]:field==='yaw'?[-180,180]:field==='pitch'?[-90,90]:[-50,50];
      if(field!=='lock'&&(!Number.isFinite(value)||value<limits[0]||value>limits[1])){error('lower-fill-placement-status',`Enter a value between ${limits[0]} and ${limits[1]}.`);return;}
      if(field==='lock')p.aimAtSubject=value;
      else if(field==='wall-gap')p.pos[2]=bounds.frontZ-value;
      else if(field==='height')p.pos[1]=value;
      else {if(p.aimAtSubject)return;p[field]=value;}
      try{validateLightingPoseState(state,STUDIO_LIGHT_IDS);poses.setState(state);error('lower-fill-placement-status','');}
      catch(e){error('lower-fill-placement-status',e.message);}
    });
  }
  const placementCenter=()=>[0,subjectCenterY(),0];
  for (const poseId of RADIAL_LIGHTS) for (const coordinate of ['arc',...(poseId.startsWith('table-')?['elevation']:[]),'radius']) {
    const id=`${poseId}-${coordinate}`,input=$(id);if(!input)continue;
    shortcutIds.add(id);
    shortcutSync.push(state=>{const value=radialLightPose(state.poses.find(p=>p.id===poseId).pos,poseId,placementCenter())[coordinate];input.value=String(Number(value.toFixed(3)));});
    input.addEventListener('input',()=>{
      const value=input.value.trim()===''?NaN:Number(input.value),statusId=`${poseId}-placement-status`;
      if(!Number.isFinite(value)){error(statusId,'Enter a finite position value.');return;}
      const state=poses.getState(),pose=state.poses.find(p=>p.id===poseId),center=placementCenter(),view=radialLightPose(pose.pos,poseId,center);
      if(value===Number(view[coordinate].toFixed(3)))return;
      view[coordinate]=value;
      try{pose.pos=radialLightPosition(pose.pos,poseId,view,center);}catch(e){error(statusId,e.message);return;}
      error(statusId,'');poses.setState(state);
    });
  }
  for(const poseId of ['kicker-left','kicker-right','background']) {
    const id=`${poseId}-wall-gap`,input=$(id);if(!input)continue;shortcutIds.add(id);
    const baseRadius=()=>-studioRoomBounds(Number($("room-scale").value),tableFloorDrop(table?.getState()??{enabled:false})).backZ;
    shortcutSync.push(state=>{const pose=state.poses.find(p=>p.id===poseId);input.value=String(Number((baseRadius()-(poseId==='background'?-pose.pos[2]:Math.hypot(pose.pos[0],pose.pos[2]))).toFixed(3)));input.max=String(baseRadius());});
    input.addEventListener('input',()=>{
      const gap=input.value.trim()===''?NaN:Number(input.value),statusId=`${poseId}-placement-status`,r=baseRadius()-gap;
      if(!Number.isFinite(gap)||gap<0||r<0){error(statusId,'Back-wall inset must stay between zero and the subject-back wall distance.');return;}
      const state=poses.getState(),pose=state.poses.find(p=>p.id===poseId);
      if(poseId==='background')pose.pos[2]=-r;
      else {const view=radialLightPose(pose.pos,poseId,placementCenter());view.radius=r;pose.pos=radialLightPosition(pose.pos,poseId,view,placementCenter());}
      error(statusId,'');poses.setState(state);
    });
  }
  syncPoseShortcuts = () => {
    const state = poses.getState(); for (const sync of shortcutSync) sync(state);
    for (const id of shortcutErrors.keys()) {const output = $(id); if (output) output.textContent = "";}
    shortcutErrors.clear();
  };
  syncPoseShortcuts();
  let surfaces;
  const backdrops = createBackdropLayerControls($("backdrop-layer-controls"), materialNames, () => {
    surfaces?.advancedBackdropChanged(); onChange(); onBackdropChange();
  });
  const syncBackdropVisibility = () => backdrops.show(Number($("backdrop-layers").value), $("backdrop-material").value === "calibration");
  syncBackdropVisibility();
  // Retain the familiar paired-key controls as explicit shortcuts. Independent
  // light editing and saved profiles still own the final poses used by both views.
  for (const [axis, id] of ["light-x", "light-y", "light-z"].entries()) {
    $(id).addEventListener("input", () => {
      const state = poses.getState(), v = Number($(id).value) / 100;
      state.poses[0].pos[axis] = axis === 0 ? -Math.abs(v) : v;
      state.poses[1].pos[axis] = axis === 0 ? Math.abs(v) : v;
      poses.setState(state);
    });
  }
  const validate = value => validateStudioLighting(value, fields, materialNames);
  const validateBase = value => validateStudioLightingBase(value, fields, materialNames);
  let roomOverrides={version:1,values:[]}, roomEditor=null, editorChannel='A', calibrationDepth=0;
  let lightingRacks={...defaultLightingRacks(),shopDimmer:1,houseDimmer:0},rackControls=null;
  const lightStands=createLightStandControls($("lighting-profiles"),materialNames,{onChange:()=>{onChange();syncPlacementWarnings();}});
  const shopLights=createShopLightControls($("lighting-profiles"),materialNames,{
    getRoomBounds:()=>studioRoomBounds(Number(fields["room-scale"].value),table?tableFloorDrop(table.getState()):0),
    getUnitMm:()=>scale?studioScaleTransform(scale.getState(),Number($("radius-mm").value),table.getState().enabled).unitM*1000:defaultStudioScale().unitMm,
    validateChange:next=>{
      if(applying)return; // The complete restored A/B profiles were checked before any control changed.
      const profile=capture();profile.shopLights=next;
      const remaining=new Set(next.fixtures.map(f=>f.id));
      const overrides={...roomOverrides,values:roomOverrides.values.filter(item=>{const match=/^shopLights\/fixtures\/@([^/]+)\//.exec(item.path);return !match||remaining.has(match[1]);})};
      validateShopPlacement(profile);validateShopPlacement(resolveRoomProfile(profile,overrides,validateBase));
    },
    onChange:next=>{
      const remaining=new Set(next.fixtures.map(f=>f.id));
      roomOverrides={...roomOverrides,values:roomOverrides.values.filter(item=>{const match=/^shopLights\/fixtures\/@([^/]+)\//.exec(item.path);return !match||remaining.has(match[1]);})};
      onChange();
    },
  });
  // Fresh rigs use the existing maximum, explicitly uncalibrated emission units.
  // Loading historical profiles retains their authored power and exposure mode.
  const freshShops=shopLights.state();
  for(const fixture of freshShops.fixtures){fixture.sourceLevel=400;fixture.exposureMode='continuous';}

  const capture = () => validate({sceneEngines:{version:1,stage:$("scene-stage-engine")?.value??'legacy',room:$("scene-room-engine")?.value??'legacy'},
    lightingRacks,materialEdits:captureMaterials(),fixtureSizes:fixtureSizes.getState(),lightTools:fixtureSizes.tools.getState(),lightStands:lightStands.getState(),shopLights:shopLights.state(),version: 1, poses: poses.getState(), backdrops: backdrops.getState(), strengths: strengths.getState(), surfaces: surfaces.getState(), table: table.getState(), scale: scale.getState(), gobos: gobos.getState(), roomOverrides,
    controls: Object.fromEntries(Object.entries(fields).map(([id, el]) => [id,
      el.type === "checkbox" ? el.checked : el.type === "range" ? Number(el.value) : el.value]))});
  const selectChannel = channel => {
    editorChannel=channel;strengths.setSoloChannel(channel);
    if(fixtureSizes.element)fixtureSizes.element.hidden=channel==='B';
    lightStands.element.hidden=channel==='B';
    shopLights.element.hidden=channel==='B';
    if(roomEditor) {roomEditor.element.hidden=channel!=='B'; if(channel==='B')roomEditor.refresh();}
    const panel=$("studio-scene-controls");if(panel){panel.hidden=channel==='B';if(panel.style)panel.style.display=channel==='B'?'none':'';}
  };
  createLightingProfiles($("lighting-profiles"), {
    capture, onSelectChannel:selectChannel,
    validate,
    onChange,
    apply(value) {
      const valid = validate(value);
      validateShopPlacement(valid);validateShopPlacement(resolveRoomProfile(valid,valid.roomOverrides,validateBase));
      // Restore against the actual buffers before changing any lighting control.
      restoreMaterials(valid.materialEdits);
      for(const name of ['stage','room'])if($("scene-"+name+"-engine"))$("scene-"+name+"-engine").value=valid.sceneEngines[name];
      roomOverrides=valid.roomOverrides;lightingRacks=valid.lightingRacks; rackControls?.sync();
      for (const [id, v] of Object.entries(valid.controls)) {
        const el = fields[id];
        if (el.type === "checkbox") el.checked = v; else el.value = String(v);
        const out = $(id + "-out");
        if (out) out.value = id === "backdrop-layers" || id === "room-scale" ? String(v) :
          (Number(v) / (id.includes("strength") ? 10 : 100)).toFixed(2);
      }
      acceptedKeyShape=valid.controls['key-shape'];
      if(keyShapeStatus)keyShapeStatus.textContent='';
      applying = true;
      try {
        table.setState(valid.table);
        fixtureSizes.setState(valid.fixtureSizes);
        fixtureSizes.tools.setState(valid.lightTools);
        lightStands.setState(valid.lightStands);
        shopLights.setState(valid.shopLights);
        strengths.setState(valid.strengths);
        table.setState(valid.table); scale.setState(valid.scale); gobos.setState(valid.gobos);
        backdrops.setState(valid.backdrops); surfaces.setState(valid.surfaces); poses.setState(valid.poses);
        syncBackdropVisibility(); onRoomChange(); onBackdropChange(); onGoboChange();
        roomEditor?.refresh();
        doc.dispatchEvent?.(new Event("scene-material-profile-restored"));
      } finally {applying = false;}
    },
  });
  const fixtureSizes=createFixtureSizeControls($("lighting-profiles"),()=>{
    // Removing a tool removes its Room overrides too, so reattachment cannot
    // resurrect hidden settings. Surviving stable IDs retain their overrides.
    const ids=new Set(fixtureSizes.tools.getState().attachments.map(tool=>tool.id));
    if(fixtureSizes.getState().underlightFit)roomOverrides={...roomOverrides,values:roomOverrides.values.filter(item=>!/^fixtureSizes\/lights\/@underlight\/(width|height)$/.test(item.path))};
    roomOverrides={...roomOverrides,values:roomOverrides.values.filter(item=>{const match=/^lightTools\/attachments\/@([^/]+)\//.exec(item.path);return !match||ids.has(match[1]);})};
    onChange();syncPoseShortcuts();syncPlacementWarnings();
  },(next,before,id,tools)=>validateFixturePose(poses.getState(),id,next,tools),{
    materialNames,getTable:()=>table?.getState(),getKeyShape:()=>fields['key-shape'].value,
    validateChange:(next,before,id)=>validateFixturePose(poses.getState(),id,fixtureSizes.getState(),next),
  });
  keyShapeStatus=doc.createElement('output');keyShapeStatus.id='key-shape-status';
  keyShapeStatus.setAttribute('role','status');keyShapeStatus.setAttribute('aria-live','polite');
  (fields['key-shape'].parentElement??fixtureSizes.element).append(keyShapeStatus);
  const validateKeyShape=event=>{
    if(event?.target&&event.target!==fields['key-shape'])return;
    if(fields['key-shape'].value===acceptedKeyShape)return;
    // A dense grid can fit the .76-wide key but not its .20-wide hard shape.
    // Validate both channels before any ancestor handler can read that shape.
    try{
      capture();acceptedKeyShape=fields['key-shape'].value;
      keyShapeStatus.textContent='';fixtureSizes.tools.sync();invalidateRenderState();
    }catch(error){
      fields['key-shape'].value=acceptedKeyShape;fixtureSizes.tools.sync();invalidateRenderState();
      keyShapeStatus.textContent=error.message+' Previous key shape retained.';
      event?.stopImmediatePropagation?.();event?.stopPropagation?.();
    }
  };
  for(const kind of ['input','change']){
    // Document capture precedes the Studio's ancestor capture listener. The
    // target hook also supports isolated controls without a document dispatcher.
    doc.addEventListener?.(kind,validateKeyShape,true);
    fields['key-shape'].addEventListener(kind,validateKeyShape);
  }
  const strengths = createLightStrengthControls($("lighting-profiles"), fields, onChange, ()=>{onRoomChange();onBackdropChange();onGoboChange();});
  // All split settings now live in B; retain the old component for legacy loads
  // and solo ownership, not as a contradictory second channel editor.
  strengths.element.hidden=true;
  // One explicit edit of the existing source levels, not a renderer-only mask.
  const studioLightsOff=doc.createElement('button');studioLightsOff.type='button';
  studioLightsOff.id='studio-lights-off';studioLightsOff.textContent='Turn off Studio lights (A + B)';
  studioLightsOff.title='Set all ten Studio fixture levels to zero in exposure and room preview. Shop lights are unchanged.';
  studioLightsOff.addEventListener('click',()=>{
    const paths=new Set(STUDIO_LIGHTS.map(light=>'controls/'+light.strengthId));
    for(const {strengthId}of STUDIO_LIGHTS){fields[strengthId].value='0';const output=$(strengthId+'-out');if(output)output.value='0.00';}
    strengths.setState({version:1,lights:STUDIO_LIGHTS.map(({id})=>({id,exposure:null}))});
    roomOverrides={...roomOverrides,values:roomOverrides.values.filter(item=>!paths.has(item.path))};
    onChange();onRoomChange();roomEditor?.refresh();
  });
  $('lighting-profiles').append(studioLightsOff);
  table = createStudioTableControls($("studio-table-controls"), materialNames, (next, before) => {
    const delta = tableFloorDrop(before) - tableFloorDrop(next);
    if (delta) {
      const state = poses.getState();
      for (const index of FLOOR_FIXTURE_INDICES) state.poses[index].pos[1] += delta;
      applying = true;
      try {poses.setState(state);} finally {applying = false;}
      surfaces.roomBoundsChanged();
    }
    fixtureSizes.sync();scale?.sync(); syncPoseShortcuts(); syncPlacementWarnings();
    onRoomChange(); onBackdropChange(); onChange();
  }, (next, before) => {
    fixtureSizes.validateForTable(next);
    // Refuse an impossible floor translation before changing table or poses.
    // Width/length never move lights, even when the advisory warns of overhang.
    const delta = tableFloorDrop(before) - tableFloorDrop(next), state = poses.getState();
    for (const index of FLOOR_FIXTURE_INDICES) state.poses[index].pos[1] += delta;
    try {validateLightingPoseState(state, STUDIO_LIGHT_IDS);} catch {
      throw new RangeError("Table height would move a floor-mounted light outside Y = -50 to 50. Adjust that light in Advanced first.");
    }
  });
  scale = createStudioScaleControls($("studio-scale-controls"), {
    getRadiusMm: () => Number($("radius-mm").value),
    getTableEnabled: () => table.getState().enabled,
    onChange: () => {syncPoseShortcuts(); syncPlacementWarnings(); if (!applying) onChange();},
  });
  surfaces = createStudioSurfaceControls($("studio-surface-controls"), fields, backdrops, materialNames,
    {onChange, onRoomChange, onBackdropChange, syncBackdropVisibility, getRoomFloorDrop:()=>tableFloorDrop(table.getState())});
  const gobos = createStudioGoboControls($("studio-gobo-controls"), materialNames, () => {
    onGoboChange(); onChange();
  });
  const kickerGuides=lightGuideRole?createKickerHeightGuides(fixtureSizes.element,{
    getSnapshots:()=>{
      const tableState=table.getState(),subject=studioScaleTransform(scale.getState(),Number($("radius-mm").value),tableState.enabled);
      const bounds=studioRoomBounds(Number($("room-scale").value),tableFloorDrop(tableState)),state=poses.getState();
      return ['kicker-left','kicker-right'].map(id=>({id,table:tableState,floorY:bounds.floorY,ceilingY:bounds.ceilingY,
        unitM:subject.unitM,subject:{center:subject.subjectCenter,radius:subject.subjectRadius},
        locked:state.poses.find(p=>p.id===id).aimAtSubject,
        role:lightGuideRole(id,subject.subjectCenter,state,fixtureSizes.getState(),fixtureSizes.tools.getState())}));
    },
    setHeight:(id,height)=>{
      const state=poses.getState(),bounds=studioRoomBounds(Number($("room-scale").value),tableFloorDrop(table.getState()));
      const y=bounds.floorY+height;if(!Number.isFinite(height)||height<0||y>bounds.ceilingY||y>50)throw Error('Kicker height must fit between the floor and ceiling');
      state.poses.find(p=>p.id===id).pos[1]=y;poses.setState(state);
    },
  }):null;
  syncPlacementWarnings = () => {
    const state = table.getState();
    let subject;
    try {subject = studioScaleTransform(scale.getState(), Number($("radius-mm").value), state.enabled);}
    catch {return;}
    const warnings = studioPlacementWarnings(state, poses.getPoses(subject.subjectCenter),
      studioRoomBounds(Number($("room-scale").value), tableFloorDrop(state)),
      {center: subject.subjectCenter, radius: subject.subjectRadius});
    for (const [key, id] of [["table", "studio-table-placement-status"], ["table-left", "table-left-placement-status"],
      ["table-right", "table-right-placement-status"], ["fill-upper", "top-fill-height-status"]]) {
      const output = $(id); if (output) output.textContent = [shortcutErrors.get(id), warnings[key]].filter(Boolean).join(" ");
    }
    if(standWarnings)try{lightStands.setPlacementWarnings(standWarnings(capture()));}
    catch(error){lightStands.setPlacementWarnings([{lightId:'Floor stands',message:error.message}]);}
    kickerGuides?.refresh();
  };
  $("room-scale").addEventListener("input", () => {syncPoseShortcuts();syncPlacementWarnings();});
  $("radius-mm").addEventListener("input", () => {scale.sync(); syncPoseShortcuts(); syncPlacementWarnings();});
  scale.sync();
  syncPoseShortcuts();
  syncPlacementWarnings();
  roomEditor=createRoomOverrideControls($("lighting-profiles"),{
    capture,validateBase,fields,materialNames,
    getOverrides:()=>roomOverrides,
    setOverrides:value=>{
      const previous=new Map(roomOverrides.values.map(v=>[v.path,v.value]));
      const changed=value.values.filter(v=>previous.get(v.path)!==v.value).map(v=>v.path);
      changed.push(...roomOverrides.values.filter(v=>!value.values.some(n=>n.path===v.path)).map(v=>v.path));
      const candidate=resolveRoomProfile(capture(),value,validateBase);
      validateShopPlacement(candidate);
      for(const id of ['underlight','fill-upper'])if(changed.some(path=>path.includes('/@'+id+'/')||path.startsWith('lightTools/')||/^(table\/|scale\/|controls\/room-scale$)/.test(path))){
        const clearance=fixtureClearance(id,candidate.poses,candidate.fixtureSizes,candidate);
        if(clearance&&!clearance.valid)throw new RangeError('Room: '+clearance.message);
      }
      roomOverrides=value;onRoomChange();onBackdropChange();onGoboChange();
    },
  });
  applying=true;try{shopLights.setState(freshShops);}finally{applying=false;}
  const rackEnvironment=()=>({bounds:studioRoomBounds(Number(fields['room-scale'].value),tableFloorDrop(table.getState())),unitMm:studioScaleTransform(scale.getState(),Number($('radius-mm').value),table.getState().enabled).unitM*1000});
  const resolveRackProfile=(profile,next)=>{
    const environment={bounds:studioRoomBounds(profile.controls['room-scale'],tableFloorDrop(profile.table)),unitMm:studioScaleTransform(profile.scale,Number($('radius-mm').value),profile.table.enabled).unitM*1000};
    const resolved=structuredClone(profile);
    // Generated modules follow their rack when room dimensions or carriage
    // positions change. Authored modules remain untouched.
    if(next.buses.some(b=>b.enabled))resolved.shopLights=layoutLightingRackFixtures(resolved.shopLights,next,environment);
    return applyLightingRackOverrides(resolved,next);
  };
  const checkRacks=next=>{
    for(const candidate of [capture(),resolveRoomProfile(capture(),roomOverrides,validateBase)])validateShopPlacement(resolveRackProfile(candidate,next));
  };
  rackControls=createLightingRackControls($('light-solo-clear')?.parentElement??$('lighting-profiles'),{
    getState:()=>lightingRacks,
    getMaxDropFeet:()=>{const {bounds,unitMm}=rackEnvironment();return Math.max(0,(bounds.ceilingY-bounds.floorY)*unitMm/304.8-.28);},
    setState:next=>{next=validateLightingRacks(next);checkRacks(next);lightingRacks=next;onChange();onRoomChange();},
    onBuildRacks:()=>{
      const next=validateLightingRacks(lightingRacks);next.buses.forEach(b=>b.enabled=true);
      const shops=layoutLightingRackFixtures(shopLights.state(),next,rackEnvironment());
      const profile=capture();profile.shopLights=shops;validateShopPlacement(applyLightingRackOverrides(profile,next));
      shopLights.setState(shops);lightingRacks=next;rackControls.sync();onChange();onRoomChange();
    },
  });
  shopLights.refreshPlacement();
  for(const kind of ['input','change'])doc.addEventListener?.(kind,event=>{
    if(/^(room-scale|studio-unit-mm|studio-scale-mode|radius-mm|studio-table-)/.test(event.target?.id??''))shopLights.refreshPlacement();
  });
  selectChannel(editorChannel);
  for(const [kind,delta]of [['room-calibration-start',1],['room-calibration-end',-1]])$("lighting-profiles").addEventListener(kind,()=>{
    calibrationDepth=Math.max(0,calibrationDepth+delta);onRoomChange();onBackdropChange();onGoboChange();
  });
  return {poses, backdrops, strengths, surfaces, table, gobos, scale, fixtureSizes, lightTools:fixtureSizes.tools, lightStands, shopLights, syncBackdropVisibility,
    rackControls,invalidateRenderState,
    renderState:(channel='studio')=>{
      if(!renderCache){const base=capture();renderCache={exposure:resolveRackProfile(base,lightingRacks),studio:calibrationDepth?base:resolveRackProfile(resolveRoomProfile(base,roomOverrides,validateBase),lightingRacks)};}
      return renderCache[channel==='exposure'?'exposure':'studio'];
    },
    roomOverrides:roomEditor,
    kickerGuides,ownsPoseShortcut: id => shortcutIds.has(id)||!!kickerGuides?.ownsControl(id)};
}


function createRoomOverrideControls(container,{capture,validateBase,fields,materialNames,getOverrides,setOverrides}) {
  const doc=container.ownerDocument, element=doc.createElement('section');element.id='room-profile-overrides';
  const title=doc.createElement('h3');title.textContent='B (room): Phong-only overrides';
  const help=doc.createElement('p');help.textContent='Every setting follows A (exposure) until unlocked here. Lock it again to resume following A. These edits never change ray-traced photos or live-ray history. C and D save the complete exposure + room configuration. Add fixtures/gobos in A, then override their Room visibility or settings here.';
  const status=doc.createElement('p');status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  const rows=doc.createElement('div');element.append(title,help,status,rows);container.append(element);
  let refreshing=false;
  const pretty=path=>{
    const p=path.split('/');
    if(p[0]==='controls')return p[1].replaceAll('-',' ');
    if(p[0]==='poses')return p[2].slice(1).replaceAll('-',' ')+' '+(p[3]==='pos'?'position '+['X','Y','Z'][Number(p[4])]:p[3]==='aimAtSubject'?'on-subject lock':p[3]);
    if(p[0]==='backdrops')return 'Backdrop '+(Number(p[1])+1)+' '+p[2];
    if(p[0]==='gobos')return 'Gobo '+p[2].slice(1)+' '+p[3];
    return p.map(v=>v.replaceAll('-',' ')).join(' ');
  };
  function refresh() {
    if(refreshing)return;refreshing=true;
    try {
      const base=capture(), overrides=new Map(getOverrides().values.map(item=>[item.path,item.value]));
      const openGroups=new Set(Array.from(rows.children).filter(group=>group.open).map(group=>group.dataset.profileGroup));
      rows.replaceChildren();const groups=new Map();
      for(const {path,value} of roomSettingLeaves(base)) {
        // Pair shortcuts are redundant views; positions use the actual pose owner.
        if(['controls/light-x','controls/light-y','controls/light-z'].includes(path))continue;
        const root=path.split('/')[0];
        if(!groups.has(root)) {const group=doc.createElement('details');group.dataset.profileGroup=root;group.open=openGroups.has(root);const summary=doc.createElement('summary');summary.textContent={controls:'Light levels, filters and room surfaces',poses:'Light positions and directions',backdrops:'Backdrop layers',table:'Product table',scale:'Studio scale',gobos:'Gobos',fixtureSizes:'Fixture sizes',lightTools:'Attached light tools',lightStands:'Light supports',shopLights:'Ceiling shop lights'}[root];group.append(summary);rows.append(group);groups.set(root,group);}
        const row=doc.createElement('label');row.className='grid';
        const lock=doc.createElement('input');lock.type='checkbox';if(lock.style)lock.style.width='auto';lock.checked=overrides.has(path);lock.id='room-unlock-'+path.replaceAll('/','-');
        lock.setAttribute('aria-label','Unlock Room '+pretty(path));
        const text=doc.createElement('span');text.textContent=' Unlock '+pretty(path)+' ';
        const key=path.split('/').at(-1), source=path.startsWith('controls/')?fields[key]:null;
        const divisor=path.startsWith('controls/')&&key.includes('strength')?10:1;
        let choices=source?.options?Array.from(source.options).map(o=>[o.value,o.textContent||o.value]):null;
        if(['material','front','back','backingMaterial','topMaterial'].includes(key)) choices=[...(key==='material'&&root==='backdrops'?[['','Default backdrop material']]:[]),...materialNames.map(n=>[n,n.replaceAll('_',' ')])];
        if(root==='shopLights'&&key==='model')choices=Object.keys(SHOP_LIGHT_MODELS).map(name=>[name,name]);
        if(root==='shopLights'&&key==='emitterProfile')choices=[['fluorescent_cool_white','Fluorescent cool white'],['fluorescent_warm_white','Fluorescent warm white'],['led_cool_white','LED cool white'],['led_warm_white','LED warm white']];
        if(root==='shopLights'&&key==='source')choices=[['tubes','Emissive tubes'],['led-array','LED array']];
        if(root==='lightStands'&&key==='upperSupport')choices=[['ceiling','Ceiling cables'],['floor','Floor stand']];
        if(key==='shape')choices=[['rectangle','Rectangle'],['ellipse','Ovoid']];
        if(root==='scale'&&key==='mode')choices=[['fixed','Independent unit'],['legacy','Legacy radius-linked']];
        const input=doc.createElement(choices?'select':'input');input.id='room-value-'+path.replaceAll('/','-');
        if(choices)for(const [v,label]of choices){const o=doc.createElement('option');o.value=v;o.textContent=label;input.append(o);}
        else {input.type=typeof value==='boolean'?'checkbox':typeof value==='number'?'number':String(value).startsWith('#')?'color':'text';if(typeof value==='number')input.step='any';}
        input.setAttribute('aria-label','Room '+pretty(path));if(input.style)input.style.gridColumn='1/-1';
        const current=overrides.has(path)?overrides.get(path):value;
        if(typeof value==='boolean')input.checked=current;else input.value=String(typeof current==='number'?current/divisor:current);
        input.disabled=!lock.checked;
        const commit=next=>{
          try {
            const valid=validateRoomOverrides({version:1,values:Array.from(next,([path,value])=>({path,value}))},capture(),validateBase);
            setOverrides(valid);status.textContent='Room updated; exposure unchanged.';return true;
          }catch(e){status.textContent=e.message;return false;}
        };
        lock.addEventListener('change',()=>{
          const next=new Map(getOverrides().values.map(item=>[item.path,item.value]));
          if(lock.checked)next.set(path,value);else next.delete(path);
          if(commit(next)){
            // Keep the active DOM branch alive throughout the event. Replacing
            // rows here detaches the click/change target before bubbling and
            // can make ancestor outside-click handlers misclassify it.
            input.disabled=!lock.checked;
            const effective=roomSettingLeaves(capture()).find(item=>item.path===path)?.value;
            const selected=lock.checked?next.get(path):effective;
            if(typeof value==='boolean')input.checked=selected;else input.value=String(typeof selected==='number'?selected/divisor:selected);
          }else lock.checked=!lock.checked;
        });
        input.addEventListener('change',()=>{
          const v=typeof value==='boolean'?input.checked:typeof value==='number'?(input.value.trim()===''?NaN:Number(input.value)*divisor):input.value;
          const next=new Map(getOverrides().values.map(item=>[item.path,item.value]));next.set(path,v);
          commit(next);
        });
        row.append(lock,text,input);groups.get(root).append(row);
      }
    }finally{refreshing=false;}
  }
  return {element,refresh};
}

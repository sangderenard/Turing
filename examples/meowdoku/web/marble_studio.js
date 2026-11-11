import {computeStudioFixtureRoles,buildStudioFixtureFaces,createStudioSceneGeometry,buildStudioPhotoFixtures,initializeStudioFixtureRows} from './studio_scene_builder.js';
// Reuse the server loading panel; its boot hold covers Studio asset/GPU setup.
import "./loading_panel.js";
import {ensureObjAsset, importedMechanics} from "./marble_imported_object.js";
// A close-up on the same exported Pluck Phong shader and MaterialDatabase
// tables used by the game. All edits are local to this preview.
import { marbleMass } from "./marble_mass.js";
import { bubbleCount, validateBubbleHost, validateBubbleInclusions, validateBubbleRadius, phongInterior } from "./marble_bubbles.js";
import { rollingStep, rotationMatrix } from "./marble_roll.js";
import { ComputeRuntime } from "./compute.js";
import { caseModel } from "./gravity.js";
import { marblePreviewSpec, livePreviewParticle, previewLawToScene, cameraNudgeGravity } from "./marble_sim_preview.js";
import { bindMarbleDriveInput } from "./marble_drive_input.js";
import { MarbleChaseCamera, chaseViewMatrix } from "./marble_chase_camera.js";
import { previewMotionAngle } from "./marble_preview_motion.js";
import { createMarblePhoto, photoCameraFromStudio, validatePhotoFocus, OBJECT_CENTER } from "./marble_photo.js";
import { createLiveRayView } from "./marble_live.js";
import { createObjectAutosave } from "./marble_autosave.js";
import { prepareObjectSnapshot, objectSnapshotKey } from "./marble_object_snapshot.js";
import { downloadMarbleObject } from "./marble_downloads.js";
import { MARBLE_OBJECT_VERSION, readDesign, applyDesign, readPhysical, applyPhysical,
  buildMarbleObjectGeometry, materialIdsForDesign, geometrySha256, analyticSpheresForDesign,
  verifySavedMarbleObject } from "./marble_object.js";
import { initCalibration } from "./marble_calibration.js";
import { backdropFaces } from "./marble_backdrop.js";
import { setupStudioLighting } from "./marble_studio_lighting.js";
import { STUDIO_LIGHTS } from "./marble_light_strengths.js";
import { studioRoomBounds } from "./marble_room.js";
import { defaultStudioTable, tableFloorDrop } from "./marble_studio_table.js";
import {lightStandPlacementWarnings} from "./marble_light_stand_settings.js";
import {backdropSupportLayout} from "./marble_backdrop_supports.js";
import {shopLightEmitterAllocations,writeShopLightEmitterRows,studioShopLightGeometry} from "./marble_shop_lights.js";
import { defaultStudioScale, studioScaleTransform, studioFrameDistance, studioCameraClip, radiusFromSlider, sliderFromRadius } from "./marble_studio_scale.js";
import { isOpaquePreviewMaterial, triangleCentres, transparentTriangleOrder } from "./marble_glass_preview.js";
import { STUDIO_FIXED_MATERIAL_ROWS, STUDIO_RUNTIME_MATERIAL_ROWS, studioMaterialExtras, studioMaterialBase, savedMaterialRows } from "./marble_materials.js";
import {validateSceneMaterialEdits} from "./marble_scene_materials.js";
import {createOpticalMaterialControls} from "./marble_studio_optical.js";
import {restoreOpticalSurfaceRows,prepareOpticalSnapshot} from "./marble_optical_authoring.js";
import {packStudioScene,writeStudioFixtureRows} from "./marble_studio_scene.js";
import {stageStudioPhong} from "./marble_studio_phong.js";
import {createStudioDof,dofCameraFromView} from "./marble_studio_dof.js";
import {mountStudioCameraToolbox} from "./marble_camera_toolbox_mount.js";
import {PhotographicCameraBinding,mountPhotographicCameraBinding,linkedPhotoOrbit,studioDisplayedSubjectCenter} from "./marble_camera_binding.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view"), message = $("message");
let studioLighting = null;
let opticalControls = null;
let liveRay = null;
const photographicCameraBinding=new PhotographicCameraBinding();
document.addEventListener("studio-lens-change", () => liveRay?.cameraChanged());
let studioSceneRevision=0,lastStudioFrame=null;
// Conservative edit revision catches an edit and reversal during a hidden
// bake. Exact captured data is still compared before ownership transfers.
for(const event of ["input","change","studio-lens-change"])document.addEventListener(event,e=>{if(!e.target?.closest?.('#camera-toolbox'))studioSceneRevision++;},true);
let suppliedSubjectAsset=null;
let objectAutosave=null, savedObjectSnapshot=null, objectKey="", objectEditQueued=false, applyingObject=false, objectLoadIntent=0;
// Keep the ray-traced photo's "Current Studio orbit and zoom" camera option
// actually the same scene as the Phong preview, not just the same angle.
// Lighting is NOT sent as a separate hand-authored list: the server derives
// real NEE sources directly from `fixtures`' own emissive PBR geometry
// below (matching exposure_render_demo.py's canonical _build_tracer_scene
// pattern for its "orbiters" emissive-sphere demo) -- one source of truth.
const currentCameraOptions = (snapshot = false, {usePhotoCamera=true}={}) => ({
  ...studioPhotoCamera(!snapshot,{usePhotoCamera}),
  object_center_m: studioPhotoSubjectCenter(!snapshot),
  fixtures: sceneForPhoto(!snapshot),
});
const photo = createMarblePhoto(document, currentCameraOptions,
  () => prepareObjectSnapshot(captureObjectSnapshot()));
const cameraToolbox=mountStudioCameraToolbox({doc:document,photo,canvas,getScene:captureCurrentStudioScene});
const photographicCameraControls=mountPhotographicCameraBinding({doc:document,binding:photographicCameraBinding,
  captureLinked:()=>studioPhotoCamera($("photo-auto-snapshot")?.checked===false,{usePhotoCamera:false}),
  framePhoto:framePhotographicCamera,
  onChange:()=>{$("photo-camera").value="studio";photo.sceneChanged();}});
// preserveDrawingBuffer: the colour-calibration feature (marble_calibration.js)
// reads this canvas's own pixels via gl.readPixels() some time after a given
// frame rendered (not synchronously inside that frame's own draw call), which
// only returns valid content -- instead of whatever the browser already
// presented and cleared -- when the drawing buffer is preserved.
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, preserveDrawingBuffer: true });
const studioDof=gl?createStudioDof(gl):null;
let lastDofOptics=null;
function dofStatus(text){const out=$("studio-dof-status");if(out&&out.textContent!==text)out.textContent=text;}
if (!gl) message.textContent = "WebGL 2 is required for this studio";

function compile(type, source) {
  const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}
function program(vertex, fragment) {
  const p = gl.createProgram(); gl.attachShader(p, compile(gl.VERTEX_SHADER, vertex));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragment)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const identity = () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    for (let k = 0; k < 4; k++) o[c*4+r] += a[k*4+r] * b[c*4+k];
  return o;
}
const translate = (z) => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]);
const rotateY = (a) => new Float32Array([Math.cos(a),0,-Math.sin(a),0, 0,1,0,0, Math.sin(a),0,Math.cos(a),0, 0,0,0,1]);
const rotateX = (a) => new Float32Array([1,0,0,0, 0,Math.cos(a),Math.sin(a),0, 0,-Math.sin(a),Math.cos(a),0, 0,0,0,1]);
const rotateZ = (a) => new Float32Array([Math.cos(a),Math.sin(a),0,0, -Math.sin(a),Math.cos(a),0,0, 0,0,1,0, 0,0,0,1]);
function rotateQuaternion(q) {
  const r=rotationMatrix(q);
  return new Float32Array([r[0][0],r[1][0],r[2][0],0,r[0][1],r[1][1],r[2][1],0,
    r[0][2],r[1][2],r[2][2],0,0,0,0,1]);
}
const translate3=(x,y,z)=>new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1]);
const transformPoint=(m,[x,y,z])=>[m[0]*x+m[4]*y+m[8]*z+m[12],m[1]*x+m[5]*y+m[9]*z+m[13],m[2]*x+m[6]*y+m[10]*z+m[14]];
const scale3=(s)=>new Float32Array([s,0,0,0,0,s,0,0,0,0,s,0,0,0,0,1]);
function perspective(aspect, fovDegrees = 360 / 7, far = 30, near = 0.01) {
  const f = 1 / Math.tan(fovDegrees * Math.PI / 360);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}
const loc = (p, name) => gl.getUniformLocation(p, name);
const field = (p, name, value) => gl.uniform1f(loc(p, name), value);
const integer = (p, name, value) => gl.uniform1i(loc(p, name), value);
const hex = (rgb) => "#" + rgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2,"0")).join("");
const rgb = (value) => [1,3,5].map((k) => parseInt(value.slice(k,k+2),16)/255);

function floatTexture(table, unit, authoredRows = table.data.length / table.floats_per_record) {
  // 20 spare records reserved for runtime-authored materials (the Studio
  // light fixtures: 10 distinct emissive rows + 1 shared black housing row +
  // 1 bright debug-label row, then 8 shop emitters), beyond authored rows.
  const data = new Float32Array((authoredRows + STUDIO_RUNTIME_MATERIAL_ROWS)*table.floats_per_record), width = 256, height = Math.max(1, Math.ceil(data.length / width));
  data.set(table.data);
  const tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const update = () => { const padded = new Float32Array(width * height); padded.set(data);
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, padded); };
  update();
  return { data, table, update, original: data.slice() };
}
function flatArray(unit, pixel) {
  const tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA8, 1, 1, 2, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([...pixel,...pixel]));
}
function mesh(faces) {
  const pos = [], normal = [], material = [], group = [];
  for (const [a,b,c,id,n,gid=0] of faces) for (const [k,p] of [a,b,c].entries()) {
    pos.push(...p); normal.push(...n[k]); material.push(id); group.push(gid);
  }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buffers = [];
  for (const [index, data, integerAttribute] of [[0,new Float32Array(pos),false],[1,new Float32Array(normal),false],[2,new Int32Array(material),true],[4,new Int32Array(group),true]]) {
    const buffer = gl.createBuffer(); buffers.push(buffer); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); gl.enableVertexAttribArray(index);
    if (integerAttribute) gl.vertexAttribIPointer(index, 1, gl.INT, 0, 0);
    else gl.vertexAttribPointer(index, 3, gl.FLOAT, false, 0, 0);
  }
  gl.bindVertexArray(null);
  return { vao, buffers, count: pos.length/3 };
}
function dispose(mesh) {
  if (!mesh) return;
  gl.deleteVertexArray(mesh.vao); for (const buffer of mesh.buffers) gl.deleteBuffer(buffer);
}
function transparentMesh(faces) {
  const result=mesh(faces),buffer=gl.createBuffer();
  result.buffers.push(buffer);result.indexBuffer=buffer;
  result.centres=triangleCentres(faces);result.sortAxis=null;
  gl.bindVertexArray(result.vao);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,buffer);
  gl.bindVertexArray(null);
  return result;
}
function drawTransparent(result,modelView) {
  gl.bindVertexArray(result.vao);
  const axis=[modelView[2],modelView[6],modelView[10]];
  if(!result.sortAxis||axis.some((v,k)=>v!==result.sortAxis[k])) {
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,result.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,transparentTriangleOrder(result.centres,modelView),gl.DYNAMIC_DRAW);
    result.sortAxis=axis;
  }
  gl.drawElements(gl.TRIANGLES,result.count,gl.UNSIGNED_INT,0);
}
function digitFaces(char, center, dir, scale, materialId) {
  const g=glyphMeshes[char];
  if (!g) return [];
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const pos=g.positions;
  const faces=[];
  for (let t=0;t<g.triangles;t++) {
    const verts=[0,1,2].map((v)=>{
      const o=(t*3+v)*3, lx=pos[o], ly=pos[o+1], lz=pos[o+2];
      return center.map((c,i)=>c+right[i]*lx*scale+up[i]*ly*scale+dir[i]*lz*scale);
    });
    faces.push([verts[0],verts[1],verts[2],materialId,[dir,dir,dir]]);
  }
  return faces;
}
function stringFaces(text, basePos, dir, scale, materialId) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const faces=[];
  const spacing=scale*1.3;
  let offset=-(text.length-1)*spacing*0.5;
  for (const ch of text) {
    const charCenter=basePos.map((c,i)=>c+right[i]*offset);
    faces.push(...digitFaces(ch, charCenter, dir, scale, materialId));
    offset+=spacing;
  }
  return faces;
}

const dimensions = ["bubble-count","width","length","curve","strands","thickness","twist","tip","flare","phase","orientation-yaw","orientation-pitch","orientation-roll","tube-count","tube-radius","tube-turns","confetti-count","coverage","band-count","band-width","band-curve"];
const properties = { roughness:["pbr",3,100], metallic:["pbr",4,100], transmission:["pbr",5,100],
  opacity:["pbr",7,100], ior:["pbr",6,100],
  specular:["phong",1,100], shininess:["phong",2,1],
  "enamel-thickness":["enamel",0,1], "enamel-ior":["enamel",1,100] };
let tables, ids, P, interior, glass, glassOpaque, checkerMesh, backdropMesh, coatedId, paintAId, paintBId, lastGeometry, rollProps, lastMass,
  rollState, exteriorClass=new Map(), lastFrame=0, seed = 1, yaw = -0.5, pitch = 0.2, distance = 3.45, dieDefaultsApplied=false;
let FIXTURE_MAT_BASE=0, FIXTURE_HOUSING_MAT=0, FIXTURE_LABEL_MAT=0, fixtureMeshes=null, glyphMeshes={};
let opaqueSaved = null;
let tableOpaque=null, tableTransparentFaces=[], marbleTransparentFaces=[], sceneGlass=null, sceneGlassKey=null, savedPhotoRadius=null;
let standOpaque=null,standGlass=null,standTransparentFaces=[],standMeshKey=null;
let shopOpaque=null,shopGlass=null,shopTransparentFaces=[],shopMeshKey=null,shopGeometry=null;
let studioOrbitTarget=[0,0,0];
let lastPreviewSubjectTransform=null;
const sim = { base:null, runtime:null, spec:null, epoch:0, map:null, cageOpaque:null, cageGlass:null,
  status:"Particle simulation loads when selected", nextFlipStep:0, flipSeconds:0, motionKey:"" };
const simStatus=()=>{$("preview-sim-status").textContent=sim.runtime?.error?`Particle simulation error: ${sim.runtime.error}`:sim.status;};
const drive={direction:"neutral",keyboard:[0,0],active:false,input:null};
const DRIVE={neutral:[0,0],left:[-1,0],right:[1,0],up:[0,1],down:[0,-1]};
const chaseCamera=new MarbleChaseCamera();
const isMarbleChase=()=>$("preview-engine").value==="particle"&&$("preview-camera").value!=="case";
const driveViewYaw=()=>isMarbleChase()?chaseCamera.yaw:yaw;
function driveGravity() {
  const [dx,up]=drive.direction==="neutral"?drive.keyboard:DRIVE[drive.direction];
  const viewYaw=driveViewYaw();
  if(sim.runtime?.ready)sim.runtime.setGravity(...cameraNudgeGravity(dx,up,viewYaw));
  drive.yaw=viewYaw;
  drive.active=!!(dx||up);
  const direction=[up>0?"up":up<0?"down":"",dx>0?"right":dx<0?"left":""].filter(Boolean).join("/");
  $("drive-status").textContent=drive.active?`Tilting ${direction} in camera view`:
    "Click the view, then WASD/arrows, or hold the D-pad";
}
function simCage(spec) {
  const packed=spec.chamber_mesh;
  if(!packed)throw new Error("compiled chamber has no containing mesh");
  const decode=(base64)=>{const bytes=Uint8Array.from(atob(base64),(c)=>c.charCodeAt(0));return new Uint16Array(bytes.buffer);};
  const vertices=decode(packed.v),triangles=decode(packed.t),b=packed.bounds;
  const low=spec.view[2],high=spec.view[3],dz=b[5]-b[4],tile=Math.max(0.025,(b[1]-b[0])/9);
  const opaque=[],glass=[];
  const vertex=(id)=>[b[0]+vertices[3*id]/65535*(b[1]-b[0]),
    b[2]+vertices[3*id+1]/65535*(b[3]-b[2]),b[4]+vertices[3*id+2]/65535*dz];
  for(let i=0;i<packed.nt;i++) {
    const law=[vertex(triangles[3*i]),vertex(triangles[3*i+1]),vertex(triangles[3*i+2])];
    const scene=law.map(sim.map.point),[a,c,d]=scene;
    const ab=[c[0]-a[0],c[1]-a[1],c[2]-a[2]],ad=[d[0]-a[0],d[1]-a[1],d[2]-a[2]];
    const n=[ab[1]*ad[2]-ab[2]*ad[1],ab[2]*ad[0]-ab[0]*ad[2],ab[0]*ad[1]-ab[1]*ad[0]];
    const len=Math.max(1e-9,Math.hypot(...n)),normal=n.map((v)=>v/len);
    const center=law[0].map((_,k)=>(law[0][k]+law[1][k]+law[2][k])/3);
    const checked=center[2]<b[4]+0.12*dz || center[1]>high-0.12*(high-low);
    const parity=(Math.floor(center[0]/tile)+Math.floor(center[1]/tile)+Math.floor(center[2]/tile))&1;
    // The SDF toy's real contact mesh supplies the ramps, throat and pegs.
    // Give its obstacles distinct finishes without implying per-wall friction.
    const obstacle=Math.abs(normal[1])<0.72 && !checked;
    const id=checked ? (parity?ids.stone_onyx:ids.ceramic_porcelain)
      : obstacle ? (parity?ids.metal_copper:ids.metal_gold) : ids.marble_clear_glass;
    (checked||obstacle?opaque:glass).push([a,c,d,id,[normal,normal,normal]]);
  }
  return { opaque:mesh(opaque), glass:mesh(glass) };
}
function configureSimMotion() {
  const runtime=sim.runtime;if(!runtime?.ready)return;
  const kind=$("preview-motion").value,axis=$("preview-axis").value,hz=+$("preview-hz").value;
  const phase=+$("preview-phase").value,arc=+$("preview-arc").value;
  runtime.setSpin(0);runtime.setGravity(0,0,-1);
  if(kind==="rotate"||kind==="rock") {
    const result=runtime.setCasePeriodic(kind==="rotate"?"spin":"rock",hz,arc,phase,axis);
    if(!result)throw new Error(`this compiled product cannot rotate about ${axis.toUpperCase()}`);
    sim.status=`Compiled toy course · ${result.hz.toFixed(2)} Hz ${kind} about ${axis.toUpperCase()}${result.limited?" (bounded by contacts)":""} · physical angular mesh pose`;
  } else {
    runtime.setCasePeriodic(null);runtime.setCaseTilt(0);
    sim.nextFlipStep=runtime.steps+Math.round((phase/360)/(hz*runtime.manifest.law.dt));
    sim.status=`Compiled toy course · ${kind==="flip"?`physical Flip about ${axis.toUpperCase()}`:"case at rest"} · physical angular mesh pose`;
  }
  driveGravity();
  simStatus();
}
async function startSim() {
  if(lastGeometry?.imported)throw new Error("Supplied OBJ mechanics are not certified; sphere-contact simulation is unavailable.");
  if(!sim.base) {
    const count=Math.max(1,Math.min(16,Math.floor(+$('preview-count').value||1)));
    $('preview-count').value=String(count);
    const response=await fetch(`/api/marble_preview_spec?count=${count}`,{cache:"no-store"});
    if(!response.ok)throw new Error(`compiled preview spec HTTP ${response.status}`);
    sim.base=await response.json();
  }
  if(!sim.runtime) sim.runtime=new ComputeRuntime({onStatus:simStatus,onSamples:()=>{},onSupport:()=>{}});
  sim.spec=marblePreviewSpec(sim.base,{...readPhysical(document),"sim-omega":Number($("sim-omega").value)},++sim.epoch);
  sim.runtime.setGravity(0,0,-1);
  sim.runtime.sync(sim.spec);
  await sim.runtime.queue;
  if(!sim.runtime.ready)throw new Error(sim.runtime.error||"compiled particle unavailable");
  sim.map=previewLawToScene(sim.spec.view,sim.spec.depth,sim.spec.uniforms.r0);
  dispose(sim.cageOpaque);dispose(sim.cageGlass);
  const cage=simCage(sim.spec);sim.cageOpaque=cage.opaque;sim.cageGlass=cage.glass;
  configureSimMotion();
  sim.status=`Compiled toy course · ${sim.spec.studio_count} identical marble${sim.spec.studio_count===1?"":"s"} · real soft-sphere contacts`;
  simStatus();
}
const resetRoll=()=>{rollState={position:[0,0,0],orientation:[1,0,0,0],angularVelocity:[0,0,0]};};
const objectStatus=(message)=>{$("object-status").textContent=message;};
async function refreshObjects(selected="", valid=()=>true) {
  const response=await fetch("/api/marble_objects",{cache:"no-store"});
  if(!response.ok)throw new Error(`object library HTTP ${response.status}`);
  const entries=(await response.json()).entries||[],box=$("saved-object");
  if(!valid())return entries.length;
  box.textContent="";
  const empty=document.createElement("option");empty.value="";empty.textContent="Choose a saved object";box.append(empty);
  for(const item of entries) {
    const option=document.createElement("option");option.value=item.digest;
    option.textContent=`${item.name} · ${item.shape.toUpperCase()} · ${item.digest.slice(0,8)}`;
    box.append(option);
  }
  if(selected)box.value=selected;
  return entries.length;
}
let ribbonMaterialNames=Array.from({length:7},(_,i)=>`ribbon_${i+1}`);
function readRibbonDesign() {
  const design=readDesign(document,seed);design["ribbon-materials"]=[...ribbonMaterialNames];return design;
}
function readSubjectDesign() {
  const design=readRibbonDesign();
  if(design['base-shape']==='obj') {
    if(!suppliedSubjectAsset)throw new Error('Choose a supplied OBJ subject');
    design.source_asset=suppliedSubjectAsset;design.material_mode='studio-shell';
    design.kind='clear';design['bubble-count']=0;design.coverage=0;design['band-count']=0;
  }
  return design;
}
function captureObjectSnapshot() {
  if(!lastGeometry||!lastMass)throw new Error("the object is not ready yet");
  const design=readSubjectDesign();design.opaqueSaved=opaqueSaved;
  const optical=opticalControls?.capture();if(optical)design.optical_authoring=optical;
  return structuredClone({v:MARBLE_OBJECT_VERSION,name:$("object-name").value.trim(),
    generator:design["base-shape"]==="obj"?"meowdoku-imported-object-v1":"meowdoku-marble-object-v1",design,
    materials:{index:{...ids},extra:{coated:coatedId,paintA:paintAId,paintB:paintBId},
      tables:Object.fromEntries(Object.entries(tables).map(([name,table])=>[name,
        {stride:table.table.floats_per_record,data:Array.from(table.data)}]))},
    physical:readPhysical(document)});
}
function objectEdited() {
  if(applyingObject||!objectAutosave||objectEditQueued)return;
  objectEditQueued=true;
  // Target listeners finish material/recipe edits before we capture them.
  queueMicrotask(()=>{
    objectEditQueued=false;if(applyingObject)return;
    try {
      const key=objectSnapshotKey(captureObjectSnapshot());
      if(key===objectKey)return;
      objectKey=key;objectAutosave.changed();
    } catch(error){objectAutosave.changed();objectStatus(`Changes are not saved: ${error.message}`);liveRay?.pendingSave("Object is not saved. Phong preview shown.");}
  });
}
async function storeObjectSnapshot({value}) {
  const response=await fetch("/api/marble_objects",{method:"POST",
    headers:{"Content-Type":"application/json","X-Meowdoku-Marble":"1"},body:JSON.stringify(value),signal:AbortSignal.timeout(15000)});
  const result=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(result.error||`save HTTP ${response.status}`);
  return result;
}
function acceptSavedObject(result,prepared) {
  savedObjectSnapshot={...prepared,digest:result.digest};
  savedPhotoRadius=prepared.value.physical["radius-mm"]/1000;
  photo.saved(result.digest);liveRay?.saved(result.digest);
}
async function saveObject() {
  objectEdited();await Promise.resolve();
  const result=await objectAutosave.flush();return result.digest;
}
async function loadObject() {
  const intent=++objectLoadIntent;
  const digest=$("saved-object").value;
  if(!digest)throw new Error("select a saved design first");
  await Promise.resolve();
  if(objectAutosave.dirty){
    if(objectAutosave.enabled)await objectAutosave.flush();
    else if(!confirm("Load this object and discard the unsaved edits? Save a version or file first to keep them."))return;
  }
  if(intent!==objectLoadIntent)return;
  const loadRevision=objectAutosave.revision;
  const response=await fetch(`/api/marble_objects/${digest}`,{cache:"no-store",signal:AbortSignal.timeout(15000)});
  if(!response.ok)throw new Error(`load HTTP ${response.status}`);
  const value=await response.json();
  const restoredRows=savedMaterialRows(value.materials,ids,
    {coated:coatedId,paintA:paintAId,paintB:paintBId},tables);
  let verified=true,geometry;
  try { ({geometry}=await verifySavedMarbleObject(value)); }
  catch(error) {
    if(!/^saved mesh recipe no longer matches its triangles$|^saved mass or moment of inertia no longer matches the mesh$/.test(String(error.message)))throw error;
    verified=false;
  }
  if(intent!==objectLoadIntent)return;
  if(loadRevision!==objectAutosave.revision)throw new Error("Object changed while loading. Your edits are kept; load again when ready.");
  objectAutosave.acceptExternal(verified?{digest}:undefined);
  applyingObject=true;
  try {
  // Restore editable legacy surface values, not the opt-in snapshot override.
  // Re-saving applies the optical profile to a fresh immutable copy.
  const editable=structuredClone(value);restoreOpticalSurfaceRows(editable);
  if(value.design.optical_authoring?.mode!=='legacy'&&value.design.optical_authoring)
    for(const row of restoredRows.pbr)row.data=editable.materials.tables.pbr.data.slice(row.offset,row.offset+row.data.length);
  for(const [name,rows] of Object.entries(restoredRows)) {
    for(const row of rows)tables[name].data.set(row.data,row.offset);
    tables[name].update();
  }
  suppliedSubjectAsset=value.design.source_asset||null;
  seed=applyDesign(document,value.design);applyPhysical(document,value.physical);syncSubjectSizeControls();
  ribbonMaterialNames=value.design["ribbon-materials"]?.slice() ?? Array.from({length:7},(_,i)=>i%2?value.design.accent:value.design.eye);
  opticalControls?.restore(value.design);
  opaqueSaved=value.design.opaqueSaved||null;dieDefaultsApplied=value.design["base-shape"]!=="marble";
  $("object-name").value=value.name;
  rebuild();syncMaterial();
  for(const field of ["dice-detail","dice-bevel","dice-face-relief","mark-scale","relief-depth","die-inclusion-size"])
    $(field).dispatchEvent(new Event("input"));
  } finally {applyingObject=false;}
  objectKey=objectSnapshotKey(captureObjectSnapshot());
  if(verified) {
    acceptSavedObject({digest},{value,geometry});
    objectStatus(`Loaded ${value.name} · verified ${geometry.triangleCount.toLocaleString()} triangles and mass moments`);
  } else {
    objectKey="";objectEdited();
    objectStatus(`${value.name}: the stored mesh receipt no longer matches this recipe. The recipe is open for inspection; save it as a new object to verify its current mesh. The original is preserved.`);
  }
}
function objectLibraryControls() {
  const run=(action)=>action().catch(error=>objectStatus(error.message||String(error)));
  const toggle=$("object-autosave");
  try{toggle.checked=localStorage.getItem("meowdoku.object-autosave")!=="off";}catch{}
  objectAutosave=createObjectAutosave({capture:captureObjectSnapshot,prepare:prepareObjectSnapshot,store:storeObjectSnapshot,
    enabled:toggle.checked,
    onPending(){liveRay?.pendingSave(toggle.checked?"Saving object changes. Phong preview shown.":"Unsaved object changes. Save a version to update rays; Phong preview shown.");},
    onSaved(result,prepared,{revision}){
      acceptSavedObject(result,prepared);
      objectStatus(result.duplicate?`Existing version retained (${result.digest.slice(0,12)}). Identical recipes keep their original library name; exports use the current name.`:`Saved new version: ${prepared.value.name} (${result.digest.slice(0,12)}). Earlier versions are unchanged.`);
      void refreshObjects(result.digest,()=>objectAutosave.revision===revision).catch(error=>objectStatus(`Saved, but library refresh failed: ${error.message}`));
      if(typeof BroadcastChannel!=="undefined"){
        const channel=new BroadcastChannel("meowdoku.marble_objects");channel.postMessage({digest:result.digest});channel.close();
      }
    },
    onError(error){objectStatus(`Changes not saved: ${error.message}. Retry Save version or save a file backup.`);liveRay?.pendingSave("Object save failed. Your edits and Phong preview remain; retry Save version.");},
    onStatus(state){$("object-autosave-status").textContent=state.dirty?(state.pending?"Saving changes…":"Unsaved changes"):(state.phase==="saved"?"Saved":"Ready");},
  });
  toggle.addEventListener("change",()=>{
    objectAutosave.setEnabled(toggle.checked);
    try{localStorage.setItem("meowdoku.object-autosave",toggle.checked?"on":"off");}catch{}
    if(objectAutosave.dirty)liveRay?.pendingSave(toggle.checked?"Saving object changes. Phong preview shown.":"Autosave off. Save a version to update rays; Phong preview shown.");
  });
  $("save-object").addEventListener("click",()=>run(saveObject));
  $("download-object").addEventListener("click",()=>run(async()=>{
    const prepared=await prepareObjectSnapshot(captureObjectSnapshot());
    const filename=await downloadMarbleObject(prepared.value);objectStatus(`Downloaded ${filename}`);
  }));
  $("load-object").addEventListener("click",()=>run(loadObject));
  $("refresh-objects").addEventListener("click",()=>run(async()=>{
    const count=await refreshObjects();objectStatus(`${count} saved object${count===1?"":"s"}`);
  }));
  window.addEventListener("beforeunload",event=>{if(objectAutosave.dirty){event.preventDefault();event.returnValue="";}});
  run(refreshObjects);
  setupObjSubjects(run);
}
async function refreshObjSubjects(selected='') {
  const response=await fetch('/api/obj_assets',{cache:'no-store'});
  if(!response.ok)throw new Error(`OBJ library HTTP ${response.status}`);
  const {entries=[]}=await response.json(),select=$('obj-asset');select.textContent='';
  const empty=document.createElement('option');empty.value='';empty.textContent='Choose a supplied OBJ';select.append(empty);
  for(const item of entries) {
    const option=document.createElement('option');option.value=item.digest;
    option.textContent=`${item.name} (${item.visibility})`;select.append(option);
  }
  select.value=selected;return entries;
}
function setupObjSubjects(run) {
  if(!$('obj-asset'))return;
  const assertIdle=()=>{
    if(!$('photo-cancel').disabled||$('live-rays')?.checked||$('preview-engine').value==='particle')
      throw new Error('Finish the exposure and select the visual preview with live rays off before replacing the subject.');
  };
  $('obj-refresh').addEventListener('click',()=>run(refreshObjSubjects));
  $('obj-use').addEventListener('click',()=>run(async()=>{
    assertIdle();const digest=$('obj-asset').value;if(!digest)throw new Error('Select an OBJ from your library');
    if(objectAutosave.dirty) {
      if(objectAutosave.enabled)await objectAutosave.flush();
      else if(!confirm('Replace the subject and discard unsaved edits? Save a version or file first to keep them.'))return;
    }
    const before=objectSnapshotKey(captureObjectSnapshot()),asset=await ensureObjAsset(digest);
    assertIdle();if(before!==objectSnapshotKey(captureObjectSnapshot()))throw new Error('Subject changed while loading. Current edits kept.');
    const candidate={...readSubjectDesign(),'base-shape':'obj',source_asset:digest,material_mode:'studio-shell'};
    buildMarbleObjectGeometry(candidate,materialIdsForDesign(candidate,ids,{coated:coatedId,paintA:paintAId,paintB:paintBId}));
    suppliedSubjectAsset=digest;$('base-shape').value='obj';$('kind').value='clear';$('bubble-count').value='0';
    $('coverage').value='0';$('band-count').value='0';$('object-name').value=asset.name;
    opticalControls?.restore({});
    rebuild();syncMaterial();objectEdited();
    objectStatus(`Loaded ${asset.name}. Supplied geometry uses the Studio shell material override. Original file is unchanged; solid optics and mechanics are not certified.${asset.attribution ? ` Source credit and terms: ${asset.attribution}` : ""}`);
  }));
  $('obj-upload').addEventListener('click',()=>run(async()=>{
    const file=$('obj-file').files[0],mtl=$('obj-mtl').files[0];
    if(!file)throw new Error('Choose an OBJ file');
    if(file.size>8*1024*1024||(mtl&&mtl.size>1024*1024))throw new Error('OBJ or MTL exceeds upload limit');
    const response=await fetch('/api/obj_assets',{method:'POST',headers:{'Content-Type':'application/json','X-Meowdoku-Marble':'1'},
      body:JSON.stringify({name:file.name,obj:await file.text(),...(mtl?{mtl:await mtl.text()}:{})})});
    const result=await response.json();if(!response.ok)throw new Error(result.error||`OBJ upload HTTP ${response.status}`);
    await refreshObjSubjects(result.digest);objectStatus('OBJ saved privately in your library. Select Use subject to replace the preview.');
  }));
  run(refreshObjSubjects);
}
function repartitionExterior(force=false) {
  if(!lastGeometry)return;
  if(tableOpaque) rebuildTable();
  const classes=new Map();
  for(const face of [...lastGeometry.interior,...lastGeometry.glass]) if(!classes.has(face[3])) {
    const id=face[3],transmission=read("pbr",id,5),opacity=read("pbr",id,7);
    classes.set(id,isOpaquePreviewMaterial(transmission,opacity));
  }
  if(!force&&classes.size===exteriorClass.size&&[...classes].every(([id,value])=>exteriorClass.get(id)===value))return;
  exteriorClass=classes;
  dispose(interior);dispose(glass);dispose(glassOpaque);
  const opaque=[],transparent=[],insideOpaque=[];
  for(const face of phongInterior(lastGeometry)) (classes.get(face[3])?insideOpaque:transparent).push(face);
  for(const face of lastGeometry.glass) (classes.get(face[3])?opaque:transparent).push(face);
  interior=mesh(insideOpaque);glassOpaque=mesh(opaque);glass=transparentMesh(transparent);
  marbleTransparentFaces=transparent; sceneGlassKey=null;
}
const idFor = (layer) => layer.startsWith("ribbon:") ? ids[ribbonMaterialNames[Number(layer.split(":")[1])]] : layer === "enamel" ? coatedId : layer === "paintA" ? paintAId : layer === "paintB" ? paintBId :
  ids[$({glass:"shell",eye:"eye",accent:"accent",confetti:"confetti-material",mark:"mark-material"}[layer]).value];
const slot = (table, id, offset) => id * tables[table].table.floats_per_record + offset;
const read = (table, id, offset) => tables[table].data[slot(table,id,offset)];
const write = (table, id, offset, value) => { tables[table].data[slot(table,id,offset)] = value; tables[table].update(); };
const selectedSceneGroup=()=>/^(stage|room):/.exec($("layer").value)?.[1];
const selected = () => selectedSceneGroup()?Number($("layer").value.split(':')[1]):idFor($("layer").value);
const sceneEditedMaterials=new Set();
function captureSceneMaterialEdits() {
  return {version:1,materials:[...sceneEditedMaterials].map(name=>({name,
    tables:Object.fromEntries(Object.entries(tables).map(([key,t])=>[key,{stride:t.table.floats_per_record,
      data:Array.from(t.data.slice(ids[name]*t.table.floats_per_record,(ids[name]+1)*t.table.floats_per_record))}])),
    ...(opticalControls?.sceneRecord(ids[name])?{optical:opticalControls.sceneRecord(ids[name])}:{})}))};
}
function restoreSceneMaterialEdits(value) {
  if(value===undefined)return; // Historical lighting profiles never owned material edits.
  const valid=validateSceneMaterialEdits(value,Object.keys(ids));
  // Validate every actual buffer stride before committing any row.
  for(const m of valid.materials)for(const [name,row] of Object.entries(m.tables))
    if(row.stride!==tables[name].table.floats_per_record)throw new Error(`Shared ${name} material stride changed.`);
  for(const name of sceneEditedMaterials)for(const t of Object.values(tables)) {
    const first=ids[name]*t.table.floats_per_record;t.data.set(t.original.slice(first,first+t.table.floats_per_record),first);
  }
  sceneEditedMaterials.clear();
  for(const m of valid.materials) {sceneEditedMaterials.add(m.name);for(const [name,row]of Object.entries(m.tables))tables[name].data.set(row.data,ids[m.name]*row.stride);}
  for(const t of Object.values(tables))t.update();
  opticalControls?.restoreSceneRecords(valid.materials.filter(m=>m.optical).map(m=>({...m.optical,material:ids[m.name]})));
  syncMaterial();rebuildTable();objectEdited();
}
function markSceneMaterialEdited() {
  if(!selectedSceneGroup())return;
  const name=Object.keys(ids).find(name=>ids[name]===selected());if(name)sceneEditedMaterials.add(name);
  studioLighting?.invalidateRenderState();photo.sceneChanged();liveRay?.sceneChanged();
}
function refreshSceneMaterialTargets() {
  const layer=$("layer"),chosen=layer.value;
  for(const group of layer.querySelectorAll('optgroup[data-scene-materials]'))group.remove();
  const roles=computeFixtureRoles(studioSubjectTransform(false,"exposure").subjectCenter,"exposure").fixtureRoles;
  const targets={stage:[...tableFaces("exposure"),...lightStandFacesForStudio("exposure",roles)],
    room:[...checkerboardFaces("exposure"),...backdropRollFaces("exposure"),...backdropSupportFacesForStudio("exposure"),...goboFaces("exposure"),...shopLightFacesForStudio("exposure")]};
  for(const [name,faces]of Object.entries(targets)) {
    const group=document.createElement('optgroup');group.dataset.sceneMaterials=name;group.label=name==='stage'?'Product stage materials':'Remaining room materials';
    for(const id of [...new Set(faces.map(f=>f[3]))]) {
      const material=Object.keys(ids).find(name=>ids[name]===id);if(!material)continue;
      const option=document.createElement('option');option.value=`${name}:${id}`;option.textContent=material.replaceAll('_',' ');group.append(option);
    }
    layer.append(group);
  }
  if(Array.from(layer.options).some(o=>o.value===chosen))layer.value=chosen;else layer.value='glass';
}
function refreshRibbonMaterialTargets() {
  const layer=$("layer"),chosen=layer.value;
  layer.querySelector('optgroup[data-ribbon-materials]')?.remove();
  if(lastGeometry?.ribbons?.length) {
    const group=document.createElement('optgroup');group.dataset.ribbonMaterials='1';group.label='Ribbons';
    lastGeometry.ribbons.forEach((_,i)=>{const option=document.createElement('option');
      option.value=`ribbon:${i}`;option.textContent=`Ribbon ${i+1}`;group.append(option);});
    layer.append(group);
  }
  layer.value=Array.from(layer.options).some(o=>o.value===chosen)?chosen:'glass';
}
function makeRibbonIndependent() {
  const layer=$("layer").value;if(!layer.startsWith('ribbon:'))return;
  const i=Number(layer.split(':')[1]),name=`ribbon_${i+1}`,source=ids[ribbonMaterialNames[i]],target=ids[name];
  if(source===target)return;
  if(!Number.isInteger(target))throw new Error('Ribbon authoring rows are missing from the exported material library.');
  for(const table of Object.values(tables)) {const f=table.table.floats_per_record;
    table.data.set(table.data.slice(source*f,(source+1)*f),target*f);table.update();}
  ribbonMaterialNames[i]=name;rebuild();objectEdited();
}
function copyCoating() {
  for (const table of Object.values(tables)) {
    const f=table.table.floats_per_record, source=idFor("glass")*f;
    table.data.set(table.data.subarray(source,source+f),coatedId*f); table.update();
  }
  if (+$("coverage").value > 0) {
    write("enamel",coatedId,0,Math.max(300,read("enamel",coatedId,0)));
  }
}
function copyPaint() {
  for (const table of Object.values(tables)) {
    const f=table.table.floats_per_record, source=ids.ceramic_porcelain*f;
    for (const id of [paintAId,paintBId]) table.data.set(table.data.subarray(source,source+f),id*f);
    table.update();
  }
  for (const [id,color] of [[paintAId,$("paint-a-color").value],[paintBId,$("paint-b-color").value]]) {
    rgb(color).forEach((v,k)=>{tables.pbr.data[slot("pbr",id,k)]=v;});
    write("pbr",id,5,0); write("pbr",id,7,1);
  }
}
function reportMass(geometry) {
  // Other material/geometry controls can reach this while a radius is being
  // retyped. Keep the prior physical result until the real radius is valid.
  if (!syncSubjectSizeControls()) return;
  if(geometry.imported) {
    lastMass=importedMechanics();rollProps=lastMass;resetRoll();
    $("mass-report").textContent='Supplied OBJ geometry. Mass and inertia are unavailable; closure and mechanical volume have not been certified.';
    $("roll-status").textContent='Roll unavailable: imported mesh mechanics have not been certified';
    return;
  }
  const densityByRole={shell:+$("shell-density").value,eye:+$("eye-density").value,
    accent:+$("accent-density").value,confetti:+$("confetti-density").value,enamel:+$("enamel-density").value,
    mark:+$("mark-density").value};
  const result=marbleMass(geometry,{radiusM:+$("radius-mm").value/1000,densityByRole,
    enamelMaterial:coatedId,enamelThicknessM:read("enamel",coatedId,0)*1e-9});
  lastMass=result;
  rollProps=result; resetRoll();
  const tensor=result.inertiaKgM2.map((row)=>row.map((x)=>x.toExponential(3)).join("  ")).join("\n");
  const moments=result.principalMoments.map((p,i)=>`I${i+1} ${p.momentKgM2.toExponential(3)} kg·m² on (${p.axis.map((x)=>x.toFixed(2)).join(", ")})`).join("\n");
  const grouped=new Map();
  for(const c of result.components) {const g=grouped.get(c.name)||{count:0,massKg:0,volumeM3:0};
    g.count++;g.massKg+=c.massKg;g.volumeM3+=c.volumeM3;grouped.set(c.name,g);}
  const components=[...grouped].map(([name,c])=>`${name} ×${c.count}: ${(c.massKg*1000).toFixed(3)} g / ${(c.volumeM3*1e9).toFixed(2)} mm³`).join("\n");
  $("mass-report").style.whiteSpace="pre-line";
  $("mass-report").textContent=`${result.massExact?"Calculated":"Estimated"} mass ${(result.massKg*1000).toFixed(3)} g · center of mass (${result.centerM.map((x)=>(x*1000).toFixed(2)).join(", ")}) mm\nPrincipal moments of inertia:\n${moments}\nFull inertia tensor (kg·m²):\n${tensor}\n${components}\n${result.mechanicsNote}`;
  $("roll-status").textContent=result.validForRoll?"Locked · release to roll on the calculated inertia tensor":`Roll unavailable: ${result.mechanicsNote}`;
}
// Pure (no WebGL calls) face builder for the room's floor+wall+ceiling,
// shared by the Phong frame() draw loop AND the ray-traced photo bridge
// below -- the real room, not a placeholder checker target standing in
// for it. Each surface has its own independently picked material(s): a
// pattern (solid/checker) plus material A/B, instead of one shared
// hardcoded pair -- so wall, ceiling, and floor can each be isolated
// (e.g. dead matte black) to judge a glass subject's own colour without
// the room's own reflections dominating the photo.
function studioRenderState(channel="studio") { return studioLighting?.renderState?.(channel) ?? null; }
function studioControlValue(id, channel="studio") {
  const state=studioRenderState(channel);
  return state ? state.controls[id] : $(id)?.type==="checkbox" ? $(id).checked : $(id)?.value;
}
// Capture actual page values; the shared builders do not read controls.
function studioBuilderLighting(channel="studio") {
  return studioRenderState(channel)??{
    controls:Object.fromEntries(LIGHTING_PROFILE_FIELDS.map(id=>[id,studioControlValue(id,channel)])),
    poses:studioLighting?.poses?.getState?.(),backdrops:studioLighting?.backdrops?.getState(),
    fixtureSizes:studioLighting?.fixtureSizes?.getState(),lightTools:studioLighting?.lightTools?.getState(),
    table:studioTableState(channel),scale:studioLighting?.scale?.getState()??defaultStudioScale(),
    gobos:studioLighting?.gobos?.getState(),lightStands:studioLighting?.lightStands?.getState(),shopLights:shopLightState(channel),
  };
}
function studioBuilderOptions(channel="studio",saved=false) {
  return {lighting:studioBuilderLighting(channel),subject:studioSubjectTransform(saved,channel),geometrySubject:studioSubjectTransform(false,channel),
    materials:{ids,fixtureMaterialBase:FIXTURE_MAT_BASE,pbr:{data:tables.pbr.data,stride:tables.pbr.table.floats_per_record}}};
}
function studioRoomScale(channel="studio") { return Number(studioControlValue("room-scale",channel) || 3); }
function studioTableState(channel="studio") { return studioRenderState(channel)?.table ?? studioLighting?.table?.getState() ?? defaultStudioTable(); }
function studioFloorDrop(channel="studio") { return tableFloorDrop(studioTableState(channel)); }
function studioSubjectTransform(saved = false, channel = saved ? "exposure" : "studio") {
  const radiusMm = saved && savedPhotoRadius !== null ? savedPhotoRadius * 1000 : Number($("radius-mm").value);
  return studioScaleTransform(studioRenderState(channel)?.scale ?? studioLighting?.scale?.getState() ?? defaultStudioScale(), radiusMm, studioTableState(channel).enabled);
}
function studioPreviewSubjectTransform() {
  try {lastPreviewSubjectTransform = studioSubjectTransform();}
  catch { /* A partially typed number must not stop the animation loop. */ }
  return lastPreviewSubjectTransform;
}
function syncSubjectSizeControls() {
  let valid = true;
  try {
    const radiusMm = Number($("radius-mm").value);
    studioSubjectTransform();
    $("radius-mm").setCustomValidity?.("");
    $("marble-scale-slider").value = String(sliderFromRadius(radiusMm));
    $("marble-scale-out").textContent = `${(radiusMm * 2).toFixed(2)} mm diameter`;
    studioPreviewSubjectTransform();
  } catch (error) {
    valid = false;
    $("radius-mm").setCustomValidity?.(error.message);
    $("marble-scale-out").textContent = `${error.message}. Preview holds the last valid size.`;
    $("mass-report").textContent = "Mass and inertia are waiting for a valid marble radius.";
  }
  studioLighting?.scale?.sync();
  return valid;
}
function studioPhotoScale(saved = true) {
  return studioSubjectTransform(saved, "exposure").unitM;
}
function studioPhotoSubjectCenter(saved = true) {
  const transform = studioSubjectTransform(saved, "exposure");
  return OBJECT_CENTER.map((v,i) => v + transform.subjectCenter[i] * transform.unitM);
}
function studioPhotoCamera(saved = true,{usePhotoCamera=true}={}) {
  // A saved Exposure stays independent of unfinished Studio radius/profile edits.
  const photoTransform=studioSubjectTransform(saved,"exposure");
  const linked=linkedPhotoOrbit({yaw,pitch,distance,photoTransform});
  const scene=usePhotoCamera?photographicCameraBinding.resolve(linked):linked;
  scene.camera = photo.applyLensCamera(scene.camera);
  return scene;
}
function framePhotographicCamera() {
  const saved=$("photo-auto-snapshot")?.checked===false,subject=studioSubjectTransform(saved,"exposure");
  const current=studioPhotoCamera(saved).camera,delta=current.eye_m.map((v,i)=>v-current.target_m[i]);
  const radius=Math.hypot(...delta),photoYaw=Math.atan2(-delta[0],delta[2]),photoPitch=Math.asin(delta[1]/radius);
  const nextDistance=studioFrameDistance({...subject,fovDegrees:photo.lensFov(subject.fovDegrees)},1,photo.lensMinimumFocus());
  photo.focusLensAt(nextDistance*subject.unitM);
  const scene=photoCameraFromStudio(photoYaw,photoPitch,nextDistance,subject.unitM,subject.fovDegrees,subject.subjectCenter);
  scene.camera=photo.applyLensCamera(scene.camera);return scene;
}
function tableFaces(channel="studio",onPart) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).tableFaces(onPart);
}
function goboFaces(channel="studio",onPart) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).goboFaces(onPart);
}
function lightStandFacesForStudio(channel="studio",fixtureRoles,onPart) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).lightStandFaces(fixtureRoles,onPart);
}
function backdropSupportFacesForStudio(channel="studio",onPart) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).backdropHolderFaces(onPart);
}
function rebuildStandMeshes(fixtureRoles) {
  const state=studioRenderState("studio")?.lightStands??studioLighting?.lightStands?.getState();
  const material=state?.enabled?ids[state.material]:undefined;
  const key=JSON.stringify([state,fixtureRoles,studioRoomScale(),studioFloorDrop(),studioSubjectTransform(false).unitM,
    studioRenderState("studio")?.backdrops??studioLighting?.backdrops?.getState(),
    ["backdrop-layers","backdrop-material","backdrop-sweep"].map(id=>studioControlValue(id)),
    material===undefined?null:[read("pbr",material,5),read("pbr",material,7)]]);
  if(key===standMeshKey)return;
  const opaque=[],transparent=[];
  for(const face of [...lightStandFacesForStudio("studio",fixtureRoles),...backdropSupportFacesForStudio("studio")])
    (isOpaquePreviewMaterial(read("pbr",face[3],5),read("pbr",face[3],7))?opaque:transparent).push(face);
  dispose(standOpaque);dispose(standGlass);standOpaque=mesh(opaque);standGlass=transparentMesh(transparent);
  standTransparentFaces=transparent;standMeshKey=key;sceneGlassKey=null;
}
let shopEmissionProfiles;
function shopLightState(channel="studio") {
  return studioRenderState(channel)?.shopLights??studioLighting?.shopLights?.state();
}
function shopLightGeometryForStudio(channel="studio") {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).shopLightGeometry();
}
function shopLightFacesForStudio(channel="studio",onPart) {
  const geometry=shopLightGeometryForStudio(channel);
  for(const part of geometry.parts)onPart?.(part);
  return geometry.faces;
}
function rebuildShopMeshes() {
  const state=shopLightState(),materials=[state?.backingMaterial,state?.topMaterial].map(name=>ids[name]);
  const key=JSON.stringify([state,studioRenderState()?.lightingRacks,studioRoomScale(),studioFloorDrop(),studioSubjectTransform(false).unitM,
    materials.map(id=>id===undefined?null:[read("pbr",id,5),read("pbr",id,7)])]);
  if(key===shopMeshKey)return shopGeometry;
  const geometry=shopLightGeometryForStudio(),opaque=[],transparent=[];
  for(const face of geometry.faces)
    (isOpaquePreviewMaterial(read("pbr",face[3],5),read("pbr",face[3],7))?opaque:transparent).push(face);
  dispose(shopOpaque);dispose(shopGlass);shopOpaque=mesh(opaque);shopGlass=transparentMesh(transparent);
  shopTransparentFaces=transparent;shopMeshKey=key;shopGeometry=geometry;sceneGlassKey=null;
  return geometry;
}
function rebuildTable() {
  dispose(tableOpaque); const opaque=[]; tableTransparentFaces=[];
  // Free-standing gobos share the existing furniture partition and world-space
  // glass sort, including when their front and back use different materials.
  for(const face of [...tableFaces(),...goboFaces()])
    (isOpaquePreviewMaterial(read("pbr",face[3],5),read("pbr",face[3],7))?opaque:tableTransparentFaces).push(face);
  tableOpaque=mesh(opaque); sceneGlassKey=null;
}
// Furniture and subject glass share one world-space triangle sort. Drawing a
// whole transparent table before/after the marble is wrong at oblique views.
// Cache geometry until the subject transform or one of those meshes changes.
function transparentSet(model) {
  const key=Array.from(model).join(",");
  if(sceneGlassKey!==key) {
    const normal=n=>{
      const p=[model[0]*n[0]+model[4]*n[1]+model[8]*n[2],model[1]*n[0]+model[5]*n[1]+model[9]*n[2],
        model[2]*n[0]+model[6]*n[1]+model[10]*n[2]];
      return p.map(v=>v/Math.hypot(...p));
    };
    const faces=marbleTransparentFaces.map(([a,b,c,id,ns,gid=0])=>
      [transformPoint(model,a),transformPoint(model,b),transformPoint(model,c),id,ns.map(normal),gid]);
    dispose(sceneGlass); sceneGlass=transparentMesh([...tableTransparentFaces,...standTransparentFaces,...shopTransparentFaces,...faces]); sceneGlassKey=key;
  }
  return sceneGlass;
}
function checkerboardFaces(channel="studio",onSurface) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).roomFaces(onSurface);
}
function checkerboard() { return mesh(checkerboardFaces()); }
// A real backdrop roll: a stack of flat material panels mounted just in
// front of the wall, narrower by default and independently sized, set
// slightly further forward by `step` (just enough to clear the previous
// layer, not enough to look detached) -- the classic studio technique of
// stacking seamless paper/gel layers so each edge reads as a visible
// colour-change border around the subject. "Sweep" drapes the whole stack
// down across the wall/floor bevel and out along the floor, like a real
// cyclorama, instead of hanging as a flat vertical sheet.
function backdropRollFaces(channel="studio",onPart) {
  return createStudioSceneGeometry(studioBuilderOptions(channel)).backdropRollFaces(onPart);
}
function backdropRoll() { return mesh(backdropRollFaces()); }
// Pure (no WebGL calls) fixture-role builder, shared by the Phong frame()
// draw loop AND the ray-traced photo bridge (marble_photo.js), so both are
// built from the same validated profile. A owns exposure; B resolves explicit
// Phong-only overrides. No Room override is read by the photo/live bridge.
function computeFixtureRoles(subjectPosition = [0,0,0], strengthChannel = "studio", poseOverride = null, sizeOverride = null, toolsOverride = null) {
  const lighting={...studioBuilderLighting(strengthChannel)};
  if(poseOverride)lighting.poses=poseOverride;if(sizeOverride)lighting.fixtureSizes=sizeOverride;if(toolsOverride)lighting.lightTools=toolsOverride;
  return computeStudioFixtureRoles({lighting,ids,fixtureMaterialBase:FIXTURE_MAT_BASE,subjectPosition,solo:studioLighting?.strengths.getSolo?.(strengthChannel)});
}
// Pure (no WebGL calls) fixture mesh builder: the real 3D housing (box or
// tube) plus the glowing panel/disc face for each role, shared by frame()'s
// draw loop and the ray-traced photo bridge. `housingMat`/`labelMat` select
// which material ID the housing/label faces carry; frame() passes the live
// FIXTURE_HOUSING_MAT, the photo bridge passes its own compact table index.
function buildFixtureFaces(fixtureRoles, housingMat, onPart) {
  return buildStudioFixtureFaces(fixtureRoles,housingMat,onPart,ids);
}
// The ray-traced photo's entire non-marble scene, as one buffer, captured
// from the exact same pure face-builders frame() itself calls right now --
// floor, plaster wall (with its window), backdrop roll, and every light
// fixture (housings + panels), all in a single mesh+material upload. The
// marble keeps its own separate SHA256-verified path; this is everything
// else the Phong preview is currently showing, as it is, not a second,
// independently-maintained description of part of it.
//
// Spectral's own append_saved_studio_mesh_to_gpu_scene() already treats
// ANY uploaded triangle whose PBR row has nonzero emission as a genuine
// area light (MAT_FLAG_EMISSIVE) -- the same mechanism the saved marble's
// own glass/inclusions use. Room surfaces carry their real catalog PBR
// rows, read straight off the live MaterialDatabase texture (tables.pbr --
// the exact bytes the Phong shader already samples); fixtures carry their
// emission-only rows built the same way Studio's own FIXTURE_MAT_BASE rows
// are, since those live texture rows are per-session GL state, not
// something meaningful to re-key for an upload. Positions are Studio's own
// world units; the only transform needed is the same linear unit/centre
// conversion. The rig unit is independent of the saved object's physical
// radius. Explicit legacy profiles retain their original photo conversion.
export function sceneForPhoto(saved = true) {
  return buildStudioPhotoFixtures({...studioBuilderOptions("exposure",saved),rayOriginM:OBJECT_CENTER,shopEmissionProfiles,
    solo:studioLighting?.strengths.getSolo?.("exposure"),getOpticalRecord:id=>opticalControls?.sceneRecord(id)});
}
// Read-only capture for hidden native preparation. This deliberately does NOT
// use sceneForPhoto: Exposure is a separate saved profile/canonical pose. No
// simulation selection, camera change, light edit, GPU work or ray job occurs.
function captureCurrentStudioScene() {
  const displayed=lastStudioFrame;
  if(!displayed||displayed.engine!=="visual"||$("preview-engine").value!=="visual")
    throw new Error("Current-scene capture needs a displayed Studio scene, not the compiled toy course");
  if(displayed.revision!==studioSceneRevision||displayed.geometry!==lastGeometry||displayed.lighting!==studioRenderState("studio"))
    throw new Error("Wait for the latest Studio edit to reach its displayed frame before capturing");
  if(yaw!==displayed.orbit?.yaw||pitch!==displayed.orbit?.pitch||distance!==displayed.orbit?.distance||
      studioOrbitTarget.some((v,i)=>v!==displayed.orbit?.target[i]))
    throw new Error("Wait for the current camera movement to reach its displayed frame before capturing");
  if($("debug-coords")?.checked)throw new Error("Switch off coordinate glyphs before capturing the current scene");
  const parts=[],collect=prefix=>part=>parts.push({...part,id:`${prefix}/${part.id}`});
  if(studioControlValue("checker","studio")) {
    checkerboardFaces("studio",collect("room"));backdropRollFaces("studio",collect("backdrop"));
  }
  tableFaces("studio",collect("table"));goboFaces("studio",collect("gobo"));
  buildFixtureFaces(displayed.fixtureRoles,FIXTURE_HOUSING_MAT,collect("fixture"));
  lightStandFacesForStudio("studio",displayed.fixtureRoles,collect("stand"));
  backdropSupportFacesForStudio("studio",collect("backdrop-holder"));
  shopLightFacesForStudio("studio",collect("shop"));
  const data=tables.pbr.data.slice(),stride=tables.pbr.table.floats_per_record;
  writeStudioFixtureRows(data,stride,FIXTURE_MAT_BASE,displayed.roleColor,displayed.roleStrength);
  writeShopLightEmitterRows(data,stride,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS,shopLightState("studio"),{profiles:shopEmissionProfiles});
  return {revision:studioSceneRevision,state:packStudioScene({channel:"studio",unitM:displayed.subject.unitM,
    rayOriginM:OBJECT_CENTER,room:displayed.room,parts,pbr:{data,stride},
    subject:{id:"subject",faces:[...lastGeometry.interior,...lastGeometry.glass],model:displayed.model,
      radius:displayed.subject.subjectRadius,motion:displayed.motion,
      phong:{interiorCount:lastGeometry.interior.length,glassCount:lastGeometry.glass.length,bubbles:structuredClone(lastGeometry.bubbles??[])}},
    camera:{view:displayed.view,fovDegrees:displayed.fovDegrees,aspect:displayed.aspect},
    authoring:{object:captureObjectSnapshot(),lighting:displayed.lighting}})};
}
function syncMaterial() {
  const id = selected();
  $("albedo").value = hex([0,1,2].map((k) => read("pbr",id,k)));
  $("inner").value = hex([4,5,6].map((k) => read("phong",id,k)));
  $("emission").value = hex([8,9,10].map((k) => read("pbr",id,k)));
  $("enamel-color").value = hex([4,5,6].map((k) => read("enamel",id,k)));
  for (const [name,[table,offset,scale]] of Object.entries(properties)) {
    const actual = read(table,id,offset);
    $(name).value = String(actual*scale);
    $(name+"-out").value = actual.toFixed(name === "shininess" || name === "enamel-thickness" ? 0 : 2);
  }
  opticalControls?.refresh();
}
function rebuild(options = {}) {
  // Initial geometry construction must not discard a resumed finished photo.
  if (options.invalidatePhoto !== false && !applyingObject) { photo.stale(); objectEdited(); }
  const design=readSubjectDesign();
  if(opticalControls&&opticalControls.mode!=='legacy')design.optical_authoring={version:1,mode:opticalControls.mode};
  for (const name of dimensions) $(name+"-out").value = $(name).value;
  let geometry;
  try {
    validateBubbleRadius(design,Number($("radius-mm").value));
    geometry=buildMarbleObjectGeometry(design,materialIdsForDesign(design,ids,
      {coated:coatedId,paintA:paintAId,paintB:paintBId}));
    let validationPbr={stride:tables.pbr.table.floats_per_record,data:tables.pbr.data};
    if(design.optical_authoring) {
      // Validate exactly the opted-in saved surfaces without mutating the
      // interactive legacy surface table or starting any ray work.
      validationPbr={stride:validationPbr.stride,data:Array.from(validationPbr.data)};
      prepareOpticalSnapshot({design:{...design,optical_authoring:opticalControls.capture(geometry)},
        materials:{tables:{pbr:validationPbr}},physical:{'radius-mm':Number($('radius-mm').value)}},geometry,ids[design.shell]);
    }
    validateBubbleHost(design,ids[design.shell],validationPbr);
    validateBubbleInclusions(geometry,validationPbr);
  } catch(error) {
    message.textContent=error.message+" Previous valid preview retained; current controls have not been saved.";
    return;
  }
  const {interior:inside,glass:outside,triangleCount}=geometry;
  lastGeometry=geometry;
  refreshRibbonMaterialTargets();
  opticalControls?.refresh();
  repartitionExterior(true);
  for (const name of dimensions) $(name+"-out").value = $(name).value;
  message.textContent = `${triangleCount.toLocaleString()} triangles · pattern #${seed}`+
    (geometry.inclusionFit?` · inclusion fit ×${geometry.inclusionFit.factor.toFixed(2)}`:"");
  if(geometry.bubbles?.length)message.textContent+=` | ${geometry.bubbles.length} air bubbles (live rays/photos only)`;
  reportMass(geometry);
}
function controls() {
  const skipPhotoInvalidation=new Set(["object-name","saved-object","obj-asset","obj-file","obj-mtl","layer","preview-engine","preview-camera","preview-motion",
    "preview-axis","preview-hz","preview-phase","preview-arc","inertia-lock","roll-force-x","roll-force-z"]);
  const designChanged=(event)=>{
    if(event.target.closest?.('#camera-toolbox'))return;
    if(event.target.closest?.('#photo-camera-binding'))return;
    if(applyingObject||event.target.closest("#optical-material-controls")||event.target.closest("#room-profile-overrides")||event.target.closest("#live-ray-controls")||event.target.id==="object-autosave")return;
    if(event.target.id==="object-name"){objectEdited();return;}
    // This component invalidates photos/ray history only for exposure changes.
    // An unlinked Studio-only edit must not disturb an existing exposure.
    if(studioLighting?.strengths.ownsControl(event.target.id))return;
    if(studioLighting?.ownsPoseShortcut(event.target.id))return;
    if(studioLighting?.table.ownsControl(event.target.id))return;
    if(studioLighting?.gobos.ownsControl(event.target.id))return;
    if(studioLighting?.scale.ownsControl(event.target.id))return;
    if(studioLighting?.surfaces.ownsControl(event.target.id))return;
    if(event.target.closest("#studio-scene-controls, #lighting-profiles, #light-pose-controls, #backdrop-layer-controls")||["checker","window-glass"].includes(event.target.id)){
      liveRay?.sceneChanged();photo.sceneChanged();return;
    }
    else if(!event.target.closest("#photo-controls") && !skipPhotoInvalidation.has(event.target.id))objectEdited();
    if(!event.target.closest("#photo-controls, #light-pose-controls, #lighting-profiles, #backdrop-layer-controls") && !skipPhotoInvalidation.has(event.target.id)) photo.stale();
  };
  for(const id of ["reroll","reset-material"]) $(id).addEventListener("click",()=>{photo.stale();objectEdited();},true);
  $("photo-start").addEventListener("click",()=>photo.start(lastGeometry)
    .catch((error)=>photo.status(error.message||String(error))));
  const diceCost=()=>{const kind=$("base-shape").value,
    factor={d4:12,d6:24,d8:24,d10:40,d12:60,d20:60}[kind]||0;
    $("dice-detail-cost").value=kind==="marble"?"Die detail applies to polyhedra":
      `Estimated ${kind.toUpperCase()}: ${(factor*(+$("dice-detail").value)**2).toLocaleString()} triangles`;
  };
  $("base-shape").addEventListener("change",()=>{
    if(!["marble","obj"].includes($("base-shape").value)&&!dieDefaultsApplied) {
      $("shell").value="ceramic_porcelain";$("mark-material").value="stone_onyx";
      $("die-shell-preset").value="porcelain";
      $("layer").value="mark";$("opaque").checked=false;opaqueSaved=null;
      yaw=-0.65;pitch=0.42;dieDefaultsApplied=true;copyCoating();
    }
    diceCost();rebuild();syncMaterial();
  });
  $("die-shell-preset").addEventListener("change",()=>{
    const preset=$("die-shell-preset").value;
    if(preset==="custom")return;
    $("shell").value={porcelain:"ceramic_porcelain",clear:"marble_clear_glass",milk:"marble_milk_glass"}[preset];
    $("mark-material").value="stone_onyx";$("opaque").checked=false;opaqueSaved=null;
    copyCoating();rebuild();syncMaterial();
  });
  for(const field of ["marking-mode","mark-material"]) $(field).addEventListener("change",()=>{rebuild();syncMaterial();});
  for(const field of ["dice-detail","dice-bevel","dice-face-relief","mark-scale","relief-depth"]) $(field).addEventListener("change",()=>{rebuild();syncMaterial();});
  $("die-inclusion-size").addEventListener("input",()=>{$("die-inclusion-size-out").value=`${$("die-inclusion-size").value}%`;});
  $("die-inclusion-size").addEventListener("change",()=>{rebuild();syncMaterial();});
  for(const field of ["dice-detail","dice-bevel","dice-face-relief","mark-scale","relief-depth"]) $(field).addEventListener("input",()=>{
    $(field+"-out").value=field==="dice-detail"?$(field).value:
      field==="mark-scale"?(+$(field).value/100).toFixed(2):(+$(field).value/1000).toFixed(3);
    if(field==="dice-detail")diceCost();
  });
  for(const field of ["dice-detail","dice-bevel","dice-face-relief","mark-scale","relief-depth"])
    $(field).dispatchEvent(new Event("input"));
  $("compensate-raised").addEventListener("change",()=>{rebuild();syncMaterial();});
  diceCost();
  const lightValue=(name)=>{const divisor=name.includes("strength")?10:100;
    return (+$(name).value/divisor).toFixed(2);};
  const lightFields=["light-strength-left","light-strength-right","light-x","light-y","light-z",
    "light-bg-strength","light-kicker-left-strength","light-kicker-right-strength","light-fill-upper-strength","light-fill-lower-strength",
    "light-table-left-strength","light-table-right-strength","light-under-strength"];
  for(const name of lightFields)
    $(name).addEventListener("input",()=>{$(name+"-out").value=lightValue(name);});
  for(const name of lightFields)
    $(name+"-out").value=lightValue(name);
  $("reset-roll").addEventListener("click",resetRoll);
  const syncPreviewMotion=()=>{
    const motion=$("preview-motion").value;
    $("preview-camera").disabled=$("preview-engine").value!=="particle";
    $("preview-arc").disabled=motion!=="rock";
    $("preview-hz-out").value=`${(+$("preview-hz").value).toFixed(2)} Hz`;
    $("preview-phase-out").value=`${$("preview-phase").value}°`;
    $("preview-arc-out").value=`${$("preview-arc").value}°`;
    if($("preview-engine").value==="particle"&&sim.runtime?.ready) {
      try {configureSimMotion();} catch(error) {sim.status=String(error.message||error);simStatus();}
    }
  };
  for(const name of ["preview-motion","preview-axis","preview-hz","preview-phase","preview-arc"])
    $(name).addEventListener("input",syncPreviewMotion);
  $("preview-engine").addEventListener("change",()=>{
    chaseCamera.clear();
    $("chase-recenter").disabled=!isMarbleChase();
    $("preview-camera").disabled=$("preview-engine").value!=="particle";
    $("drive-pad").hidden=$("preview-engine").value!=="particle";
    if($("preview-engine").value==="particle") {
      sim.status="Loading compiled particle and native containing case…";simStatus();
      startSim().catch((error)=>{sim.status=`Particle simulation unavailable: ${error.message||error}`;simStatus();});
    } else {drive.direction="neutral";drive.input?.clear();driveGravity();sim.runtime?.sync(null);sim.status="Visual preview (cheap): object motion only";simStatus();}
  });
  $("preview-camera").addEventListener("change",()=>{
    chaseCamera.clear();driveGravity();
    $("preview-camera").title=$("preview-camera").value==="case"?"Shows the whole containing course":
      "Stable chase: drag to orbit, scroll to zoom, R to recenter. Heading stays fixed while steering.";
    $("chase-recenter").disabled=!isMarbleChase();
  });
  $("chase-recenter").addEventListener("click",()=>{chaseCamera.recenter(performance.now()/1000);driveGravity();});
  canvas.addEventListener("keydown",event=>{
    if(event.code==="KeyR"&&!event.repeat&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&!event.isComposing&&isMarbleChase()) {
      event.preventDefault();chaseCamera.recenter(performance.now()/1000);driveGravity();
    }
  });
  drive.input=bindMarbleDriveInput(canvas,{
    enabled:()=>$("preview-engine").value==="particle",
    onChange:axes=>{
      drive.keyboard=axes;
      if(axes.some(Boolean)&&$("preview-motion").value!=="off") {
        $("preview-motion").value="off";syncPreviewMotion();
      }
      driveGravity();
    },
  });
  const releaseDrive=()=>{drive.direction="neutral";drive.input.clear();driveGravity();};
  window.addEventListener("blur",releaseDrive);
  document.addEventListener("visibilitychange",()=>{if(document.hidden)releaseDrive();});
  for(const button of $("drive-pad").querySelectorAll("button[data-drive]")) {
    const direction=button.dataset.drive;
    button.addEventListener("pointerdown",(event)=>{
      event.preventDefault();button.setPointerCapture(event.pointerId);
      if($("preview-motion").value!=="off") {$("preview-motion").value="off";syncPreviewMotion();}
      drive.direction=direction;if(direction==="neutral")drive.input.clear();driveGravity();
    });
    const release=()=>{if(drive.direction===direction){drive.direction="neutral";driveGravity();}};
    for(const type of ["pointerup","pointercancel","lostpointercapture"])button.addEventListener(type,release);
  }
  $("preview-camera").dispatchEvent(new Event("change"));
  for(const id of ["sim-radius","sim-rho","sim-mu","sim-e","sim-roll","sim-irr"])
    $(id).addEventListener("change",()=>{if($("preview-engine").value==="particle")startSim().catch((error)=>{sim.status=String(error.message||error);simStatus();});});
  $("preview-count").addEventListener("change",()=>{
    sim.base=null;
    if($("preview-engine").value==="particle")startSim().catch((error)=>{sim.status=String(error.message||error);simStatus();});
  });
  $("sim-contact-preset").addEventListener("change",()=>{
    if($("sim-contact-preset").value!=="custom")$("sim-omega").value=$("sim-contact-preset").value;
    if($("preview-engine").value==="particle")startSim().catch((error)=>{sim.status=String(error.message||error);simStatus();});
  });
  $("sim-omega").addEventListener("change",()=>{
    $("sim-contact-preset").value="custom";
    if($("preview-engine").value==="particle")startSim().catch((error)=>{sim.status=String(error.message||error);simStatus();});
  });
  syncPreviewMotion();
  $("inertia-lock").addEventListener("change",()=>{if(!$("inertia-lock").checked && !rollProps?.validForRoll) $("inertia-lock").checked=true;});
  for (const id of ["floor-pattern","floor-material-a","floor-material-b",
                     "wall-pattern","wall-material-a","wall-material-b",
                     "ceiling-pattern","ceiling-material-a","ceiling-material-b"])
    $(id)?.addEventListener("change",()=>{dispose(checkerMesh);checkerMesh=checkerboard();});
  for (const id of ["room-scale","window-glass","backdrop-layers","backdrop-material","backdrop-sweep"]) {
    const el=$(id); if (!el) continue;
    el.addEventListener(id==="window-glass"||id==="backdrop-sweep"?"change":"input",()=>{
      if ($(id+"-out")) $(id+"-out").value=el.value;
      if (id==="window-glass" || id==="room-scale") { dispose(checkerMesh); checkerMesh=checkerboard(); }
      studioLighting?.syncBackdropVisibility();
      dispose(backdropMesh); backdropMesh=backdropRoll();
    });
  }
  for (const name of dimensions) $(name).addEventListener("input", rebuild);
  for (const name of ["eye","accent","confetti-material"]) $(name).addEventListener("change", () => { rebuild(); syncMaterial(); });
  const syncKind=()=>{const hybrid=$("kind").value==="hybrid";
    for(const name of ["orientation-yaw","orientation-pitch","orientation-roll"]) {if(hybrid) $(name).value=0;$(name).disabled=hybrid;$(name+"-out").value=$(name).value;}};
  $("confetti-enabled").addEventListener("change",()=>{rebuild();syncMaterial();});
  $("kind").addEventListener("change", () => {if(["confetti","hybrid"].includes($("kind").value))$("confetti-enabled").checked=true;syncKind();rebuild();syncMaterial();});
  $("ribbon-layout").addEventListener("change", rebuild);
  $("shell").addEventListener("change", () => {$("die-shell-preset").value="custom";$("opaque").checked=false;opaqueSaved=null;copyCoating(); rebuild(); syncMaterial(); });
  for(const field of ["radius-mm","shell-density","eye-density","accent-density","confetti-density","enamel-density","mark-density"])
    $(field).addEventListener("input",()=>{
      // Do not enter mass or bubble geometry work for a partial number edit.
      // The unmodified input remains invalid for the save/photo validation.
      if (!syncSubjectSizeControls()) return;
      if(field==="radius-mm" && Number($("bubble-count").value)>0)rebuild();
      else if(lastGeometry) reportMass(lastGeometry);
    });
  $("radius-mm").addEventListener("input", syncSubjectSizeControls); syncSubjectSizeControls();
  $("marble-scale-slider").addEventListener("input", () => {
    $("radius-mm").value = String(Number(radiusFromSlider($("marble-scale-slider").value).toFixed(3)));
    $("radius-mm").dispatchEvent(new Event("input", {bubbles:true}));
  });
  $("photo-focus-target")?.addEventListener("click", () => {
    try {
      const camera = studioPhotoCamera($("photo-auto-snapshot")?.checked === false).camera;
      const focusM = Math.hypot(...camera.eye_m.map((v,i)=>v-camera.target_m[i]));
      photo.focusLensAt(focusM,{camera,custom:true});
      $("photo-camera").value = "studio";
      photo.status(`Custom analytical focus set to the camera target at ${(focusM*1000).toFixed(2)} mm; camera and aperture are unchanged. This does not certify a real lens or autofocus through glass.`);
      liveRay?.cameraChanged();
    } catch (error) {photo.status(error.message);}
  });
  $("frame-marble").addEventListener("click", () => {
    try {
      if(isMarbleChase()){chaseCamera.recenter(performance.now()/1000);driveGravity();return;}
      const subject = studioSubjectTransform(false, "studio");
      const framed = {...subject, fovDegrees:photo.lensFov(subject.fovDegrees)};
      const nextDistance = studioFrameDistance(framed, Math.min(1, canvas.clientWidth / canvas.clientHeight), photo.lensMinimumFocus());
      if(photographicCameraBinding.followUser)photo.focusLensAt(nextDistance * subject.unitM);
      studioOrbitTarget = studioDisplayedSubjectCenter(subject,rollState?.position??null);
      distance = nextDistance;
      if(photographicCameraBinding.followUser)$("photo-camera").value = "studio";
      photo.status(`User view analytical framing is centred on the displayed marble at ${(distance*subject.unitM*1000).toFixed(2)} mm. ${photographicCameraBinding.followUser?"Linked photographic camera follows.":"Held photographic camera is unchanged."}`);
      liveRay?.cameraChanged();
    } catch (error) {photo.status(error.message);}
  });
  for(const field of ["paint-a-color","paint-b-color"]) $(field).addEventListener("input",()=>{copyPaint();syncMaterial();});
  $("preset").addEventListener("change",()=>{
    const preset=$("preset").value;
    if(preset==="classic") {$("kind").value="cat_eye";$("shell").value="marble_clear_glass";$("ribbon-layout").value="axial";$("strands").value=2;$("curve").value=0;$("phase").value=45;$("twist").value=90;$("coverage").value=0;$("band-count").value=0;}
    if(preset==="galaxy") {$("kind").value="confetti";$("shell").value="marble_clear_glass";$("coverage").value=55;$("band-count").value=0;$("confetti-count").value=400;}
    if(preset==="uniform") {$("kind").value="confetti";$("shell").value="marble_clear_glass";$("coverage").value=0;$("band-count").value=0;$("confetti-count").value=500;}
    if(preset==="spiral" || preset==="rings") {$("kind").value=preset;$("shell").value="marble_clear_glass";$("coverage").value=0;$("band-count").value=0;}
    if(preset==="bare") {$("kind").value="clear";$("coverage").value=0;$("band-count").value=0;}
    if(preset==="porcelain") {$("kind").value="clear";$("shell").value="ceramic_porcelain";$("coverage").value=0;$("band-count").value=7;$("band-width").value=24;}
    $("confetti-enabled").checked=["confetti","hybrid"].includes($("kind").value);
    syncKind();copyCoating();rebuild();syncMaterial();
  });
  $("coverage").addEventListener("input", () => { if (+$("coverage").value > 0 && !read("enamel",coatedId,0)) write("enamel",coatedId,0,300); syncMaterial(); });
  $("opaque").addEventListener("change", () => {
    const id=idFor("glass");
    if ($("opaque").checked) {
      opaqueSaved=[read("pbr",id,5),read("pbr",id,7),read("pbr",coatedId,5),read("pbr",coatedId,7)];
      for (const target of [id,coatedId]) { write("pbr",target,5,0); write("pbr",target,7,1); }
    } else if (opaqueSaved) {
      for (const [k,target] of [id,coatedId].entries()) { write("pbr",target,5,opaqueSaved[k*2]); write("pbr",target,7,opaqueSaved[k*2+1]); }
      opaqueSaved=null;
    }
    repartitionExterior();
    syncMaterial();
  });
  document.addEventListener("scene-material-profile-restored",()=>{refreshSceneMaterialTargets();syncMaterial();});
  $("layer").addEventListener("focus",refreshSceneMaterialTargets);
  $("layer").addEventListener("change", () => {makeRibbonIndependent();syncMaterial();});
  for(const id of ['scene-stage-engine','scene-room-engine'])$(id).addEventListener('change',()=>{
    opticalControls?.refresh();photo.sceneChanged();liveRay?.sceneChanged();
  });
  $("reroll").addEventListener("click", () => { seed = (Math.random() * 0x7fffffff) | 0; rebuild(); });
  $("reset-material").addEventListener("click", () => { const id = selected();
    if (id === coatedId) { copyCoating(); repartitionExterior();syncMaterial(); return; }
    if (id === paintAId || id === paintBId) { copyPaint(); repartitionExterior();syncMaterial(); return; }
    markSceneMaterialEdited();
    for (const table of Object.values(tables)) { const f = table.table.floats_per_record, start = id*f;
      table.data.set(table.original.subarray(start,start+f),start); table.update(); } repartitionExterior();syncMaterial(); });
  for (const [name,[table,offset,scale]] of Object.entries(properties)) $(name).addEventListener("input", () => {
    if ((name==="transmission" || name==="opacity") && ["glass","enamel"].includes($("layer").value)) {
      $("opaque").checked=false;opaqueSaved=null;
    }
    const value = +$(name).value; write(table,selected(),offset,value/scale);markSceneMaterialEdited();
    if(name==="transmission"||name==="opacity") {repartitionExterior();if(selectedSceneGroup())rebuildTable();opticalControls?.refresh();}
    $(name+"-out").value = (value/scale).toFixed(name === "shininess" || name === "enamel-thickness" ? 0 : 2);
    if(name==="enamel-thickness" && lastGeometry) reportMass(lastGeometry);
  });
  for (const [name,table,offset] of [["albedo","pbr",0],["inner","phong",4],["emission","pbr",8],["enamel-color","enamel",4]])
    $(name).addEventListener("input", () => { const id = selected(); rgb($(name).value).forEach((v,k) => { tables[table].data[slot(table,id,offset+k)] = v; }); tables[table].update();markSceneMaterialEdited(); });
  let last = null;
  // Match app.js camera gestures: track each pointer, zoom by the log of the
  // two-finger span ratio, and leave a lifted pinch's remaining finger idle.
  const cameraPointers=new Map();let cameraGesture="idle",pinchSpan=0;
  const cameraSpan=()=>{const [a,b]=cameraPointers.values();return Math.hypot(a[0]-b[0],a[1]-b[1]);};
  const zoomCamera=logScale=>{
    if(isMarbleChase())chaseCamera.zoomBy(logScale);
    else {distance=Math.max(photo.lensMinimumFocus()/studioSubjectTransform().unitM,Math.min(100000,distance*Math.exp(logScale)));liveRay?.cameraChanged();}
  };
  canvas.addEventListener("pointerdown",event=>{
    if(event.pointerType==="mouse"&&event.button!==0)return;
    cameraPointers.set(event.pointerId,[event.clientX,event.clientY]);
    if(cameraPointers.size===1){cameraGesture="orbit";last=[event.clientX,event.clientY];chaseCamera.drag(true,performance.now()/1000);}
    else if(cameraPointers.size===2&&cameraGesture==="orbit"){cameraGesture="pinch";pinchSpan=cameraSpan();last=null;}
    else {cameraGesture="dead";last=null;}
    event.preventDefault();try{canvas.setPointerCapture(event.pointerId);}catch(_){/* Pointer may already have been canceled. */}
  });
  canvas.addEventListener("pointermove",event=>{
    if(!cameraPointers.has(event.pointerId))return;
    cameraPointers.set(event.pointerId,[event.clientX,event.clientY]);
    if(cameraGesture==="pinch"&&cameraPointers.size===2){
      const span=cameraSpan();if(span>0&&pinchSpan>0&&span!==pinchSpan)zoomCamera(Math.log(pinchSpan/span));pinchSpan=span;
    }else if(cameraGesture==="orbit"&&cameraPointers.size===1&&last){
      const dx=(event.clientX-last[0])*.008,dy=(event.clientY-last[1])*.008;
      if(isMarbleChase())chaseCamera.orbit(dx,dy,performance.now()/1000);
      else {yaw+=dx;pitch+=dy;liveRay?.cameraChanged();}
      last=[event.clientX,event.clientY];
    }
    event.preventDefault();
  });
  const releaseCameraPointer=event=>{
    if(!cameraPointers.delete(event.pointerId))return;
    last=null;pinchSpan=0;cameraGesture=cameraPointers.size?"dead":"idle";
    if(!cameraPointers.size)chaseCamera.drag(false,performance.now()/1000);
    try{if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);}catch(_){}
  };
  const releaseCamera=()=>{for(const pointerId of [...cameraPointers.keys()])releaseCameraPointer({pointerId});};
  for(const type of ["pointerup","pointercancel","lostpointercapture"])canvas.addEventListener(type,releaseCameraPointer);
  window.addEventListener("blur",releaseCamera);
  document.addEventListener("visibilitychange",()=>{if(document.hidden)releaseCamera();});
  canvas.addEventListener("wheel",event=>{event.preventDefault();zoomCamera(event.deltaY*.001);},{passive:false});
  // Initialization dispatches synthetic inputs to populate labels, not edits.
  for(const eventName of ["input","change"]) $("view").closest("main").querySelector("aside")
    .addEventListener(eventName,designChanged,true);
}
function frame(now=0) {
  const dt=lastFrame?Math.min(0.025,(now-lastFrame)/1000):0;lastFrame=now;
  const simOn=$("preview-engine").value==="particle";
  if(simOn&&sim.runtime?.ready) {
    if(drive.active&&drive.yaw!==driveViewYaw())driveGravity();
    sim.runtime.frame(now/1000);
    if($("preview-motion").value==="flip"&&sim.runtime.shown!=null&&sim.runtime.steps>=sim.nextFlipStep) {
      const hz=+$("preview-hz").value;
      const turn=sim.runtime.startCaseFlip(Math.min(0.7/hz,1/hz),1,$("preview-axis").value);
      if(turn) {sim.flipSeconds=turn.seconds;sim.nextFlipStep=sim.runtime.steps+Math.ceil(Math.max(turn.seconds,1/hz)/sim.runtime.manifest.law.dt);
        sim.status=`Compiled toy course · physical Flip ${turn.seconds.toFixed(2)} s about ${turn.axis.toUpperCase()} · physical angular mesh pose`;simStatus();}
    }
  }
  if(!simOn&&dt>0 && rollState && rollProps?.validForRoll && !$("inertia-lock").checked) {
    try {rollState=rollingStep(rollState,rollProps,[+$("roll-force-x").value/1000,0,+$("roll-force-z").value/1000],
      dt,+$("radius-mm").value/1000);
      $("roll-status").textContent=`Rolling · angular speed ${Math.hypot(...rollState.angularVelocity).toFixed(2)} rad/s · contact force ${Math.hypot(...rollState.contactForce).toFixed(3)} N`;
    } catch(error) {$("inertia-lock").checked=true;$("roll-status").textContent=`Roll unavailable: ${error.message}`;}
  }
  const w=Math.max(1,Math.round(canvas.clientWidth*devicePixelRatio)), h=Math.max(1,Math.round(canvas.clientHeight*devicePixelRatio));
  if (canvas.width!==w || canvas.height!==h) { canvas.width=w; canvas.height=h; }
  let dofActive=false;
  if(P&&interior&&$("studio-dof")?.checked)try{studioDof.begin(w,h);dofActive=true;}
  catch(error){$("studio-dof").checked=false;dofStatus(`Phong DOF unavailable: ${error.message}`);}
  try {
  gl.viewport(0,0,w,h); gl.clearColor(0.035,0.075,0.12,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if (P && interior) {
    const follow=isMarbleChase();
    let cameraDistance=simOn?Math.max(distance,7):distance;
    const baseView=multiply(translate(-cameraDistance),multiply(rotateX(pitch),rotateY(yaw)));
    const subject=studioPreviewSubjectTransform();
    // The actual view below feeds both the focus warning and raster blur,
    // including player/chase mode. Saved photo-camera controls stay unchanged.
    const roll=rollState?rotateQuaternion(rollState.orientation):identity();
    const subjectPosition=studioDisplayedSubjectCenter(subject,rollState?.position??null);
    if(!simOn)studioOrbitTarget=[...subjectPosition];
    let object=multiply(translate3(...subjectPosition),multiply(roll,scale3(subject.subjectRadius)));
    const angle=previewMotionAngle($("preview-motion").value,+$("preview-hz").value,+$("preview-phase").value,+$("preview-arc").value,now/1000);
    const axis=$("preview-axis").value;
    const preview=axis==="x"?rotateX(angle):axis==="z"?rotateZ(angle):rotateY(angle);
    let caseM=identity(),haveParticle=true,particleWorld=null,particleRadius=null,particleObjects=[];
    if(simOn) {
      const runtime=sim.runtime,points=runtime?.points(),lead=livePreviewParticle(points);
      haveParticle=!!(runtime?.ready&&lead&&sim.map);
      if(runtime?.ready&&runtime.shown!=null) caseM=caseModel(runtime.caseAngleAt(runtime.shown*runtime.snap),0,0,runtime.caseAxis||axis,0);
      if(haveParticle) for(let id=0;id<(sim.spec.studio_count||1);id++) {
        const pt=livePreviewParticle(points,id);if(!pt)continue;
        const p=sim.map.point([pt.x,pt.y,pt.z]);
        if(id===0) {particleWorld=transformPoint(caseM,p);particleRadius=pt.radius*sim.map.scale;}
        const q=(id===0?runtime.rotation():runtime.rotationFor?.(id))?.quaternion;
        const pose=q?multiply(rotateX(-Math.PI/2),multiply(rotateQuaternion([q[3],q[0],q[1],q[2]]),rotateX(Math.PI/2))):identity();
        particleObjects.push(multiply(caseM,multiply(translate3(...p),multiply(pose,scale3(pt.radius*sim.map.scale)))));
      }
    }
    // The law position is in the case frame. Apply the same physical case
    // transform as the containing mesh before anchoring the camera to it.
    const chase=follow&&particleWorld?chaseCamera.update({position:particleWorld,radius:particleRadius,
      sampleSeconds:sim.runtime.shown*sim.runtime.snap*sim.runtime.manifest.law.dt,nowSeconds:now/1000,
      epoch:sim.epoch,steering:drive.active,autoHeading:$("preview-camera").value==="marble"}):null;
    if(chase)cameraDistance=chase.distance;
    const view=chase?chaseViewMatrix(chase):simOn?baseView:multiply(baseView,translate3(...studioOrbitTarget.map(v=>-v)));
    const mv=multiply(view,simOn?object:multiply(object,preview));
    gl.useProgram(P);
    let dofCamera=dofCameraFromView(view,{unitM:subject.unitM,targetDistance:cameraDistance,
      fovDegrees:simOn?360/7:subject.fovDegrees});
    let opticsError=null;
    try {dofCamera=photo.applyLensCamera(dofCamera,w/h);lastDofOptics={fov_degrees:dofCamera.fov_degrees,
      focus_distance_m:dofCamera.focus_distance_m,aperture_radius_m:dofCamera.aperture_radius_m};}
    catch(error){opticsError=error.message;if(lastDofOptics)dofCamera={...dofCamera,...lastDofOptics};}
    const viewFovDegrees=dofCamera.fov_degrees;
    const clip=studioCameraClip(subject,cameraDistance,studioRoomBounds(studioRoomScale(),studioFloorDrop()),studioOrbitTarget);
    const viewFar=simOn?30:clip.far,viewNear=chase?Math.min(.01,chase.radius*.2):simOn ? .01 : clip.near;
    const setTransform=(matrix)=>{ gl.uniformMatrix4fv(loc(P,"uMV"),false,matrix);
      gl.uniformMatrix4fv(loc(P,"uMVP"),false,multiply(perspective(w/h,viewFovDegrees,
        viewFar,viewNear),matrix)); };
    setTransform(mv);
    // Real room fixtures, each a fixed scene object nailed in place in the
    // room, exactly like the checkerboard backdrop/floor -- none of them
    // move with the camera. Positions/colours shared with studioLightRig()
    // via computeFixtureRoles()/ROOM_FIXTURES so the ray-traced photo
    // bridge renders the identical rig, not an independently-guessed one.
    const {fixtureRoles, roleColor, roleStrength}=computeFixtureRoles(simOn&&particleWorld?particleWorld:[object[12],object[13],object[14]]);
    rebuildStandMeshes(fixtureRoles);
    writeShopLightEmitterRows(tables.pbr.data,tables.pbr.table.floats_per_record,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS,shopLightState(),{profiles:shopEmissionProfiles});
    const displayedShop=rebuildShopMeshes();
    // Real emitter registration (the same process spectral-analyzer's own
    // camera uses to derive lights from emissive material groups:
    // _derive_gl_lights in demo_pluck_gl.py -- position = the emitter's own
    // centroid, intensity = its physical area, so the shader's existing
    // inverse-square + emitter-radius-softening falloff (uLightIntensity
    // interpreted as area, "emitter_r2 = area/(4*pi)") is genuinely
    // physically driven, not an arbitrary point-light guess). Each fixture
    // is tagged with its own group id (shared with its mesh's aGroupId, see
    // quadFace/circleFace calls below) so it excludes itself from its own
    // light per the shader's `uLightGroupId[i] == vGroupId` self-skip.
    // Spot cone angles come from the authored shape vocabulary shared with
    // the ray-traced photo path (ray_order.py's scrim/grid/snoot/bare).
    const coneAngles={bare:null, scrim:[70,40], grid:[45,25], snoot:[18,8]};
    const lightPos=[], lightColor=[], lightIntensity=[], lightGroup=[], lightSpotDir=[], lightSpotCone=[];
    // Shop fixtures enter the same registration loop as the original disk/panel
    // fixtures. Their position and area come from the actual emissive mesh;
    // no separate triangle sample lights or invented directional cones.
    const registeredFixtures=[...fixtureRoles.map((role,i)=>({...role,color:roleColor[i],strength:roleStrength[i]})),
      ...displayedShop.emitters.map(emitter=>({pos:emitter.position,area:emitter.surfaceAreaStudio,
        mat:emitter.groupId,shape:"bare",strength:1,
        color:Array.from(tables.pbr.data.slice(emitter.materialId*tables.pbr.table.floats_per_record+8,
          emitter.materialId*tables.pbr.table.floats_per_record+11))}))];
    for (const role of registeredFixtures) {
      const area=role.area??(role.round?Math.PI*role.radius*role.radius:4*role.halfW*role.halfH);
      lightPos.push(transformPoint(view, role.pos));
      lightColor.push(role.color);
      lightIntensity.push(area*role.strength);
      lightGroup.push(role.mat);
      const cone=coneAngles[role.shape];
      if (cone) {
        const d=role.dir, viewDir=[view[0]*d[0]+view[4]*d[1]+view[8]*d[2],
          view[1]*d[0]+view[5]*d[1]+view[9]*d[2], view[2]*d[0]+view[6]*d[1]+view[10]*d[2]];
        lightSpotDir.push(viewDir);
        lightSpotCone.push([Math.cos(cone[1]*Math.PI/180), Math.cos(cone[0]*Math.PI/180)]);
      } else { lightSpotDir.push([0,0,0]); lightSpotCone.push([0,0]); }
    }
    gl.uniform1i(loc(P,"uNumLights"),lightPos.length);
    for (let i=0;i<lightPos.length;i++) {
      gl.uniform3fv(loc(P,`uLightPos[${i}]`),lightPos[i]);
      gl.uniform3fv(loc(P,`uLightColor[${i}]`),lightColor[i]);
      gl.uniform1f(loc(P,`uLightIntensity[${i}]`),lightIntensity[i]);
      gl.uniform1i(loc(P,`uLightGroupId[${i}]`),lightGroup[i]);
      gl.uniform3fv(loc(P,`uLightSpotDir[${i}]`),lightSpotDir[i]);
      gl.uniform2fv(loc(P,`uLightSpotCone[${i}]`),lightSpotCone[i]);
    }
    field(P,"uLightCalibration",1); field(P,"uExposure",1); field(P,"uFieldGain",0);
    field(P,"uAmbientFloorGain",0); // no invisible indirect-light floor: pure emitter lighting only
    gl.uniformMatrix3fv(loc(P,"uCatCcmMatrix"),false,new Float32Array([1,0,0,0,1,0,0,0,1]));
    integer(P,"uEnableSpecular",1); integer(P,"uEnableEmissionDirect",0); integer(P,"uRenderPass",0);
    integer(P,"uThermalEmission",0); field(P,"uThermalGain",0); field(P,"uThermalRefK",1000); field(P,"uThermalFloorK",720);
    integer(P,"uShadowEnabled",0); integer(P,"uShadowLightCount",0);
    gl.uniformMatrix4fv(loc(P,"uShadowMatrices[0]"),false,identity());
    gl.vertexAttribI4i(4,0,0,0,0); gl.vertexAttribI4i(5,1,0,0,0); gl.vertexAttrib3f(6,0,0,0); gl.vertexAttrib2f(3,0,0);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND); gl.depthMask(true);
    if (!simOn && tableOpaque?.count) {
      setTransform(view); gl.bindVertexArray(tableOpaque.vao);
      gl.drawArrays(gl.TRIANGLES,0,tableOpaque.count); setTransform(mv);
    }
    if(standOpaque?.count) {
      setTransform(view);gl.bindVertexArray(standOpaque.vao);gl.drawArrays(gl.TRIANGLES,0,standOpaque.count);setTransform(mv);
    }
    if (simOn&&sim.cageOpaque) {
      setTransform(multiply(view,caseM));gl.bindVertexArray(sim.cageOpaque.vao);
      gl.drawArrays(gl.TRIANGLES,0,sim.cageOpaque.count);setTransform(mv);
    } else if (!simOn&&studioControlValue("checker") && checkerMesh) {
      setTransform(view); gl.bindVertexArray(checkerMesh.vao);
      gl.drawArrays(gl.TRIANGLES,0,checkerMesh.count);
      if (backdropMesh && backdropMesh.count) {
        gl.bindVertexArray(backdropMesh.vao);
        gl.drawArrays(gl.TRIANGLES,0,backdropMesh.count);
      }
      setTransform(mv);
    }
    // Real emissive light-fixture geometry: visible quads, sized and shaped
    // per photographic role, each with its own PBR row (not shared) so its
    // glow brightness is genuinely linked to its own power slider below --
    // rewritten every frame as baseColor*liveStrength, not a fixed boot-time
    // value. A slightly larger, slightly recessed black "housing" quad sits
    // behind each glowing panel (its own non-emissive matte material), so
    // every fixture reads as a real light body with a visible lit face, not
    // a bare floating rectangle. These are what the user actually sees as
    // "the lights" -- not an invisible point the uLightPos/uLightIntensity
    // uniforms above still separately drive for the marble's own shading.
    // Each panel faces the subject at the room's origin (aimAt below).
    {
      const pbrStride=tables.pbr.table.floats_per_record;
      writeStudioFixtureRows(tables.pbr.data,pbrStride,FIXTURE_MAT_BASE,roleColor,roleStrength);
      tables.pbr.update();
    }
    const fixtureFaces=buildFixtureFaces(fixtureRoles, FIXTURE_HOUSING_MAT);
    // Debug-only coordinate labels: real extruded glyph geometry (not an
    // HTML overlay) floated above each fixture, spelling its world X/Y/Z so
    // the position is readable directly off the rendered scene. No
    // minus-sign glyph exists in the exported alphabet, so a negative value
    // is spelled with a leading "N" (e.g. "Y" "N34" means Y = -34).
    if ($("debug-coords") && $("debug-coords").checked) {
      const fmt=(n)=>(n<0?"N":"")+String(Math.round(Math.abs(n)));
      for (const role of fixtureRoles) {
        const labelPos=role.pos.map((x,i)=>x+(i===1?0.6:0));
        const text=`X${fmt(role.pos[0])}Y${fmt(role.pos[1])}Z${fmt(role.pos[2])}`;
        fixtureFaces.push(...stringFaces(text, labelPos, role.dir, 0.3, FIXTURE_LABEL_MAT));
      }
    }
    if (fixtureMeshes) dispose(fixtureMeshes);
    fixtureMeshes=mesh(fixtureFaces);
    integer(P,"uEnableEmissionDirect",1);
    // key/background/kickerL/kickerR/fill are real world-space coordinates
    // (fixed room constants, or the live camera axis for fill), exactly
    // like the checkerboard backdrop and the marble itself -- drawn with
    // the same `view` matrix, not an identity-transform shortcut.
    setTransform(view); gl.bindVertexArray(fixtureMeshes.vao);
    gl.drawArrays(gl.TRIANGLES,0,fixtureMeshes.count);
    if(shopOpaque?.count){gl.bindVertexArray(shopOpaque.vao);gl.drawArrays(gl.TRIANGLES,0,shopOpaque.count);}
    integer(P,"uEnableEmissionDirect",0);
    setTransform(mv);
    if(haveParticle) for(const model of simOn?particleObjects:[object]) {
      const modelView=simOn?multiply(view,model):mv;
      setTransform(modelView);
      gl.disable(gl.BLEND);gl.depthMask(true);
      gl.bindVertexArray(interior.vao); gl.drawArrays(gl.TRIANGLES,0,interior.count);
      gl.disable(gl.BLEND);gl.depthMask(true);
      gl.bindVertexArray(glassOpaque.vao);gl.drawArrays(gl.TRIANGLES,0,glassOpaque.count);
      gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      if(!simOn&&(tableTransparentFaces.length||standTransparentFaces.length||shopTransparentFaces.length)) {
        setTransform(view); drawTransparent(transparentSet(multiply(object,preview)),view);
      } else drawTransparent(glass,modelView);
    }
    if(simOn&&sim.cageGlass) {
      setTransform(multiply(view,caseM));gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.bindVertexArray(sim.cageGlass.vao);gl.drawArrays(gl.TRIANGLES,0,sim.cageGlass.count);
    }
    if(simOn&&standGlass?.count) {
      setTransform(view);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);drawTransparent(standGlass,view);
    }
    if(simOn&&shopGlass?.count) {
      setTransform(view);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);drawTransparent(shopGlass,view);
    }
    gl.depthMask(true); gl.disable(gl.BLEND); gl.bindVertexArray(null);
    const lighting=studioRenderState("studio");
    if(lastStudioFrame&&(lastStudioFrame.geometry!==lastGeometry||lastStudioFrame.lighting!==lighting))studioSceneRevision++;
    const mode=$("preview-motion").value,rollLocked=$("inertia-lock").checked;
    // Reuse rollingStep's existing no-slip velocity at the displayed angular
    // state. Visual preview motion has no native velocity handoff yet.
    const omega=rollState&&!rollLocked?rollState.angularVelocity:[0,0,0],r=subject.radiusM;
    const motion={mode,axis,angle,roll:rollState?structuredClone(rollState):null,rollLocked,age:0,
      velocityM:mode==="off"?[-r*omega[2],0,r*omega[0]]:null,
      angularVelocity:mode==="off"?[...omega]:null};
    lastStudioFrame={revision:studioSceneRevision,engine:simOn?"particle":"visual",geometry:lastGeometry,lighting,
      subject,model:simOn?null:multiply(object,preview),view,fovDegrees:viewFovDegrees,aspect:w/h,fixtureRoles,roleColor,roleStrength,
      orbit:{yaw,pitch,distance,target:[...studioOrbitTarget]},
      room:studioControlValue("checker","studio")?studioRoomBounds(studioRoomScale("studio"),studioFloorDrop("studio")):null,motion};
    if(dofActive)try {
      studioDof.finish(dofCamera,{unitM:subject.unitM,near:viewNear,far:viewFar});
      dofStatus(opticsError?`Showing last valid Phong optics: ${opticsError}`:
        `Phong DOF: focus ${(dofCamera.focus_distance_m*1000).toFixed(1)} mm, pupil diameter ${(dofCamera.aperture_radius_m*2000).toFixed(2)} mm. Opaque-depth approximation; 64 px blur-radius limit.`);
    }catch(error){studioDof.cancel();$("studio-dof").checked=false;dofStatus(`Phong DOF unavailable: ${error.message}`);}
    else if(!$("studio-dof")?.checked&&!$("studio-dof-status")?.textContent.startsWith("Phong DOF unavailable"))dofStatus("Phong DOF off. Photos and live rays retain their lens settings.");
  }
    if(message.textContent.startsWith('Studio preview unavailable: '))message.textContent='';
  } catch(error) {
    lastStudioFrame=null;message.textContent='Studio preview unavailable: '+(error.message||String(error));
  } finally {if(dofActive)studioDof.cancel();}
  requestAnimationFrame(frame);
}
async function boot() {
  const loading = window.meowdokuLoading;
  try {
    if (!gl) return;
    const [vertex,fragment,assets]=await Promise.all([
      fetch("shaders/base_material.webgl.vert.glsl").then((r)=>{ if(!r.ok) throw Error(`vertex shader ${r.status}`); return r.text(); }),
      fetch("shaders/base_material.webgl.shadow.frag.glsl").then((r)=>{ if(!r.ok) throw Error(`fragment shader ${r.status}`); return r.text(); }),
      fetch("assets.json").then((r)=>{ if(!r.ok) throw Error(`materials ${r.status}`); return r.json(); })]);
    loading?.initializing("Marble Studio shaders and materials");
    shopEmissionProfiles=assets.emitterProfiles;
    ids=assets.material_index || {};
    glyphMeshes=assets.meshes||{};
    for (const name of ["marble_clear_glass","marble_milk_glass","marble_cat_eye_blue","marble_cat_eye_green","marble_cat_eye_amber"])
      if (!Number.isInteger(ids[name])) throw Error(`material ${name} is absent from assets.json; export the catalogue`);
    const source=fragment.replace(/^(#version[^\n]*\n)/,"$1#define PLUCK_TRANSMISSION 1\n#define PLUCK_PREMULTIPLIED_GLASS 1\n");
    P=program(vertex,source);
    gl.useProgram(P);
    const extra=studioMaterialExtras(assets), authoredRows=studioMaterialBase(assets,extra);
    coatedId=extra.coated; paintAId=extra.paintA; paintBId=extra.paintB;
    tables=Object.fromEntries(Object.entries(assets.materials).map(([name,table])=>[name,floatTexture(table,{pbr:0,phong:1,enamel:2,texstack:3}[name],authoredRows)]));
    // Twelve reserved spare PBR rows floatTexture() already over-allocates
    // beyond every named and private authored row -- real material-table
    // slots, not a side channel: 10 distinct emissive rows, one per light
    // role, plus one shared matte-black "housing" row. Pure emitters: black
    // albedo (nothing to reflect), emission holds the glow colour; frame()
    // rewrites each row's emission every frame as baseColor*liveStrength, so
    // the fixture's own glow is actually linked to its power slider, not a
    // fixed boot-time value. Housing: non-zero albedo but zero emission, a
    // visible dark body behind each glowing panel.
    const pbrStride=tables.pbr.table.floats_per_record;
    FIXTURE_MAT_BASE=authoredRows;
    FIXTURE_HOUSING_MAT=FIXTURE_MAT_BASE+STUDIO_LIGHTS.length;
    initializeStudioFixtureRows(tables.pbr.data,pbrStride,FIXTURE_MAT_BASE);
    FIXTURE_LABEL_MAT=FIXTURE_MAT_BASE+STUDIO_LIGHTS.length+1;
    tables.pbr.update();
    // The PBR table's emission colour is gated by the texstack table's own
    // per-material emit_gain (tx_emit_gain, texstack offset 8): zero by
    // default for these spare rows, which silently zeroed every fixture's
    // glow no matter what the PBR emission or the power sliders said. Set
    // emit_gain=1 and emit_layer=-1 (no UV emission texture override) for
    // all reserved rows so the emission written to pbr.data above actually
    // reaches the screen.
    {
      const texStride=tables.texstack.table.floats_per_record;
      const texBase=(slot)=>(FIXTURE_MAT_BASE+slot)*texStride;
      for (let slot=0;slot<STUDIO_RUNTIME_MATERIAL_ROWS;slot++) {
        const b=texBase(slot);
        for (let i=0;i<texStride;i++) tables.texstack.data[b+i]=0;
        tables.texstack.data[b+0]=-1; tables.texstack.data[b+1]=-1; tables.texstack.data[b+3]=-1; // emit/color/remit layer: no UV texture
        tables.texstack.data[b+8]=1; // emit_gain
        // The 10 light-role rows (0-9; housing/label stay plain diffuse)
        // use the shader's real emissive-lobe model (profile 1): emission
        // shaped by NdotV^direct_lobe_power, so each fixture actually reads
        // as a real light bulb/panel -- brightest face-on, falling off
        // toward its edges -- instead of a flat, uniformly-lit card.
        if (slot<STUDIO_LIGHTS.length) { tables.texstack.data[b+10]=2.2; tables.texstack.data[b+11]=1; }
      }
      tables.texstack.update();
    }
    const names=Object.keys(ids).filter((name)=>Number.isInteger(ids[name])).sort();
    for (const [select,initial] of [["shell","marble_clear_glass"],["confetti-material","metal_gold"],["mark-material","metal_gold"],["backdrop-material","stone_onyx"],
      ["floor-material-a","white_plaster"],["floor-material-b","white_plaster"],
      ["wall-material-a","white_plaster"],["wall-material-b","white_plaster"],
      ["ceiling-material-a","white_plaster"],["ceiling-material-b","white_plaster"]]) {
      if (!$(select)) continue;
      for (const name of names) { const option=document.createElement("option"); option.value=name; option.textContent=name.replaceAll("_"," "); $(select).append(option); }
      $(select).value=initial;
    }
    studioLighting=setupStudioLighting(document,computeFixtureRoles().fixtureRoles,names,{
      captureMaterials:captureSceneMaterialEdits,restoreMaterials:restoreSceneMaterialEdits,
      validateShopPlacement:profile=>{
        const subject=studioScaleTransform(profile.scale,Number($("radius-mm").value),profile.table.enabled);
        studioShopLightGeometry(profile.shopLights,ids,shopLightEmitterAllocations(profile.shopLights,FIXTURE_MAT_BASE+STUDIO_FIXED_MATERIAL_ROWS),{
          bounds:studioRoomBounds(Number(profile.controls["room-scale"]),tableFloorDrop(profile.table)),unitMm:subject.unitM*1000});
      },
      lightGuideRole:(id,subject,poses,sizes,tools)=>computeFixtureRoles(subject,"exposure",poses,sizes,tools).fixtureRoles[STUDIO_LIGHTS.findIndex(light=>light.id===id)],
      standWarnings:profile=>{
        const subject=studioScaleTransform(profile.scale,Number($("radius-mm").value),profile.table.enabled);
        const bounds=studioRoomBounds(Number(profile.controls["room-scale"]),tableFloorDrop(profile.table));
        const roles=computeFixtureRoles(subject.subjectCenter,"exposure",profile.poses,profile.fixtureSizes,profile.lightTools).fixtureRoles;
        const warnings=lightStandPlacementWarnings(profile.lightStands,roles,bounds.floorY,{bounds,unitMm:subject.unitM*1000,
          fixtureVertices:role=>buildFixtureFaces([role],FIXTURE_HOUSING_MAT).flatMap(face=>face.slice(0,3))});
        if(profile.lightStands.enabled&&profile.lightStands.backdropSupport){
          const faces=backdropFaces({count:Number(profile.controls['backdrop-layers']),material:profile.controls['backdrop-material'],
            sweep:profile.controls['backdrop-sweep'],layers:profile.backdrops,wallZ:bounds.backZ,floorY:bounds.floorY},ids);
          const layout=backdropSupportLayout(faces.flatMap(face=>face.slice(0,3)),bounds.floorY,{bounds,unitMm:subject.unitM*1000});
          if(!layout.available)warnings.push({lightId:'Backdrop holder',message:layout.reason});
        }
        return warnings;
      },
      fixtureVertices:(poseId,state,subject,sizes,tools)=>{
        const index=STUDIO_LIGHTS.findIndex(light=>light.id===poseId);
        const role=computeFixtureRoles(subject,"exposure",state,sizes,tools).fixtureRoles[index];
        return buildFixtureFaces([role],0).flatMap(face=>face.slice(0,3));
      },
      onChange:()=>{photo.sceneChanged();liveRay?.sceneChanged();},
      onBackdropChange:()=>{dispose(backdropMesh);backdropMesh=backdropRoll();},
      onGoboChange:()=>rebuildTable(),
      onRoomChange:()=>{dispose(checkerMesh);checkerMesh=checkerboard();rebuildTable();},
    });
    copyCoating(); copyPaint(); checkerMesh=checkerboard(); backdropMesh=backdropRoll();rebuildTable();
    if ($("backdrop-layers-out")) $("backdrop-layers-out").value=$("backdrop-layers").value;
    if ($("room-scale-out")) $("room-scale-out").value=$("room-scale").value;
    for (const [name,unit] of Object.entries({turing_ssbo_pbr:0,turing_ssbo_phong:1,turing_ssbo_enamel:2,turing_ssbo_texstack:3,
      uEmitUv:4,uColorUv:5,uDepthUv:6,uRemitUv:7,uFieldVolume:8,uShadowMaps:9,turing_ssbo_thermal_k:10})) integer(P,name,unit);
    flatArray(4,[0,255,128,255]); flatArray(5,[255,255,255,255]); flatArray(6,[128,255,255,255]); flatArray(7,[0,0,0,0]);
    const volume=gl.createTexture(); gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_3D,volume);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,1,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
    const shadow=gl.createTexture(); gl.activeTexture(gl.TEXTURE9); gl.bindTexture(gl.TEXTURE_2D_ARRAY,shadow);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY,1,gl.DEPTH_COMPONENT24,1,1,1);
    floatTexture({data:[0],floats_per_record:1},10);
    controls(); syncMaterial(); rebuild({invalidatePhoto:false});
    opticalControls=createOpticalMaterialControls({getContext:()=>({geometry:lastGeometry,
      design:readSubjectDesign(),shell:ids[$('shell').value],selectedMaterial:selected(),
      selectedGroup:selectedSceneGroup(),sceneEngines:{stage:$('scene-stage-engine').value,room:$('scene-room-engine').value},
      updateMaterialTables:()=>tables.pbr.update(),
      names:Object.fromEntries(Object.entries(ids).map(([name,id])=>[id,name.replaceAll('_',' ')])),
      pbr:{stride:tables.pbr.table.floats_per_record,data:tables.pbr.data}}),
      onChange(change){if(change?.modeChanged)rebuild();else{if(change?.sceneChanged)markSceneMaterialEdited();photo.stale();objectEdited();}}});
    refreshSceneMaterialTargets();
    objectLibraryControls();
    objectKey=objectSnapshotKey(captureObjectSnapshot());
    liveRay=createLiveRayView({doc:document,canvas,getGeometry:()=>savedObjectSnapshot?.geometry??lastGeometry,
      getOpticalMode:()=>opticalControls?.mode??'legacy',
      ensureSaved:async()=>{if(objectAutosave.enabled)await saveObject();else if(objectAutosave.dirty)throw new Error("Save your object changes or enable autosave first.");},
      getScene:()=>currentCameraOptions(false,{usePhotoCamera:false}),getPose:()=>validatePhotoFocus(studioPhotoCamera(true,{usePhotoCamera:false}).camera),
      getFilmResponse:()=>photo.filmResponse()});
    frame();
    initCalibration(document, { saveObject, getGeometry: () => lastGeometry,
      getCameraOptions: () => ({...currentCameraOptions(), filmResponse:photo.filmResponse()}), gl, canvas });
  } catch (error) { message.textContent=String(error.message||error); console.error("[marble studio]",error); }
  finally { loading?.done(); loading?.release("boot"); }
}
boot();
// Allocates and verifies invisible buffers only. A caller still needs the
// real ray acknowledgment and the final unchanged-scene ownership transfer.
const stageCurrentStudioPhong=(capture,dynamic,{signal}={})=>{
  if(!P||!gl.getProgramParameter(P,gl.LINK_STATUS))throw new Error("The actual Studio Phong program is not ready");
  return stageStudioPhong({gl,mesh,transparentMesh,dispose,capture,dynamic,signal});
};
export {captureCurrentStudioScene,stageCurrentStudioPhong};

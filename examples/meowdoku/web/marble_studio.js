// A close-up on the same exported Pluck Phong shader and MaterialDatabase
// tables used by the game. All edits are local to this preview.
import { marbleMass } from "./marble_mass.js";
import { rollingStep, rotationMatrix } from "./marble_roll.js";
import { ComputeRuntime } from "./compute.js";
import { caseModel } from "./gravity.js";
import { marblePreviewSpec, livePreviewParticle, previewLawToScene, cameraNudgeGravity } from "./marble_sim_preview.js";
import { previewMotionAngle } from "./marble_preview_motion.js";
import { createMarblePhoto, photoCameraFromStudio, photoLightsFromStudio } from "./marble_photo.js";
import { MARBLE_OBJECT_VERSION, readDesign, applyDesign, readPhysical, applyPhysical,
  buildMarbleObjectGeometry, materialIdsForDesign, geometrySha256,
  verifySavedMarbleObject } from "./marble_object.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view"), message = $("message");
// Keep the ray-traced photo's "Current Studio orbit and zoom" camera option
// actually the same scene as the Phong preview, not just the same angle:
// bundle the live five-role light rig (background/2 kickers/key/fill) in
// alongside the camera pose. marble_photo.py validates power in [0.01, 100];
// the Phong strength sliders go well past that for visual range, so clamp.
const studioLightRig = () => ({
  keyX: +$("light-x").value / 100, keyY: +$("light-y").value / 100, keyZ: +$("light-z").value / 100,
  keyStrength: Math.min(100, Math.max(0.01, +$("light-strength").value / 10)),
  keyShape: $("key-shape").value,
  keyFilterRgb: rgb($("key-filter").value),
  bgStrength: Math.min(100, Math.max(0.01, +$("light-bg-strength").value / 10)),
  kickerStrength: Math.min(100, Math.max(0.01, +$("light-kicker-strength").value / 10)),
  fillStrength: Math.min(100, Math.max(0.01, +$("light-fill-strength").value / 10)),
});
const photo = createMarblePhoto(document, () => ({
  ...photoCameraFromStudio(yaw, pitch, distance),
  ...photoLightsFromStudio(yaw, pitch, distance, studioLightRig()),
}));
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
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
function perspective(aspect) {
  const f = 1 / Math.tan(Math.PI / 7), near = 0.01, far = 30;
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}
const loc = (p, name) => gl.getUniformLocation(p, name);
const field = (p, name, value) => gl.uniform1f(loc(p, name), value);
const integer = (p, name, value) => gl.uniform1i(loc(p, name), value);
const hex = (rgb) => "#" + rgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2,"0")).join("");
const rgb = (value) => [1,3,5].map((k) => parseInt(value.slice(k,k+2),16)/255);

function floatTexture(table, unit) {
  // 6 spare records reserved for runtime-authored materials (the Studio
  // light fixtures: 5 distinct emissive rows + 1 shared black housing row),
  // beyond the catalogue's own rows.
  const data = new Float32Array(table.data.length + 6*table.floats_per_record), width = 256, height = Math.max(1, Math.ceil(data.length / width));
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
  const pos = [], normal = [], material = [];
  for (const [a,b,c,id,n] of faces) for (const [k,p] of [a,b,c].entries()) {
    pos.push(...p); normal.push(...n[k]); material.push(id);
  }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buffers = [];
  for (const [index, data, integerAttribute] of [[0,new Float32Array(pos),false],[1,new Float32Array(normal),false],[2,new Int32Array(material),true]]) {
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
// A flat quad facing `dir` (unit vector, the direction toward the subject),
// as two triangles for mesh(). This is the actual visible shape of a light
// fixture -- real geometry with a real emissive material, not a point.
function quadFace(center, dir, halfW, halfH, materialId) {
  let worldUp=[0,1,0];
  if (Math.abs(dir[1])>0.95) worldUp=[1,0,0];
  const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm3=(v)=>{const n=Math.max(1e-9,Math.hypot(...v));return v.map((x)=>x/n);};
  const right=norm3(cross3(dir,worldUp));
  const up=norm3(cross3(right,dir));
  const add=(a,b,s)=>a.map((x,i)=>x+b[i]*s);
  const c00=add(add(center,right,-halfW),up,-halfH), c10=add(add(center,right,halfW),up,-halfH);
  const c11=add(add(center,right,halfW),up,halfH), c01=add(add(center,right,-halfW),up,halfH);
  const n=[dir,dir,dir];
  return [[c00,c10,c11,materialId,n],[c00,c11,c01,materialId,n]];
}

const dimensions = ["width","length","curve","strands","thickness","twist","tip","flare","phase","orientation-yaw","orientation-pitch","orientation-roll","tube-count","tube-radius","tube-turns","confetti-count","coverage","band-count","band-width","band-curve"];
const properties = { roughness:["pbr",3,100], metallic:["pbr",4,100], transmission:["pbr",5,100],
  opacity:["pbr",7,100], ior:["pbr",6,100], ambient:["phong",0,100],
  specular:["phong",1,100], shininess:["phong",2,1],
  "enamel-thickness":["enamel",0,1], "enamel-ior":["enamel",1,100] };
let tables, ids, P, interior, glass, glassOpaque, checkerMesh, coatedId, paintAId, paintBId, lastGeometry, rollProps, lastMass,
  rollState, exteriorClass=new Map(), lastFrame=0, seed = 1, yaw = -0.5, pitch = 0.2, distance = 3.45, followDistance = 0.12, dieDefaultsApplied=false;
let FIXTURE_MAT_BASE=0, FIXTURE_HOUSING_MAT=0, fixtureMeshes=null;
let opaqueSaved = null;
const sim = { base:null, runtime:null, spec:null, epoch:0, map:null, cageOpaque:null, cageGlass:null,
  status:"Particle simulation loads when selected", nextFlipStep:0, flipSeconds:0, motionKey:"" };
const simStatus=()=>{$("preview-sim-status").textContent=sim.runtime?.error?`Particle simulation error: ${sim.runtime.error}`:sim.status;};
const drive={direction:"neutral"};
const DRIVE={neutral:[0,0],left:[-1,0],right:[1,0],up:[0,1],down:[0,-1]};
function driveGravity() {
  const [dx,up]=DRIVE[drive.direction];
  if(sim.runtime?.ready)sim.runtime.setGravity(...cameraNudgeGravity(dx,up,yaw));
  drive.yaw=yaw;
  $("drive-status").textContent=drive.direction==="neutral"?"Hold an arrow to tilt the toy course":
    `Tilting ${drive.direction} in camera view`;
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
async function refreshObjects(selected="") {
  const response=await fetch("/api/marble_objects",{cache:"no-store"});
  if(!response.ok)throw new Error(`object library HTTP ${response.status}`);
  const entries=(await response.json()).entries||[],box=$("saved-object");
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
async function saveObject() {
  if(!lastGeometry||!lastMass)throw new Error("the object is not ready yet");
  const design=readDesign(document,seed);design.opaqueSaved=opaqueSaved;
  // Save the geometry regenerated from this exact recipe, even if a control
  // changed after the last preview rebuild. The digest must describe the
  // recipe that the simulator will regenerate on load.
  const geometry=buildMarbleObjectGeometry(design,materialIdsForDesign(design,ids,
    {coated:coatedId,paintA:paintAId,paintB:paintBId}));
  const geometryHash=await geometrySha256(geometry);
  design.mesh_lods={};
  for(const detail of ["medium","low","proxy"]) {
    const lod=buildMarbleObjectGeometry(design,materialIdsForDesign(design,ids,
      {coated:coatedId,paintA:paintAId,paintB:paintBId}),{detail});
    design.mesh_lods[detail]=await geometrySha256(lod);
  }
  if(geometryHash!==await geometrySha256(lastGeometry))rebuild();
  reportMass(geometry);
  const value={v:MARBLE_OBJECT_VERSION,name:$("object-name").value.trim(),
    generator:"meowdoku-marble-object-v1",design,
    materials:{index:{...ids},extra:{coated:coatedId,paintA:paintAId,paintB:paintBId},
      tables:Object.fromEntries(Object.entries(tables).map(([name,table])=>[name,
        {stride:table.table.floats_per_record,data:Array.from(table.data)}]))},
    physical:readPhysical(document),derived:lastMass,
    geometry_sha256:geometryHash};
  await verifySavedMarbleObject(value);
  const response=await fetch("/api/marble_objects",{method:"POST",
    headers:{"Content-Type":"application/json","X-Meowdoku-Marble":"1"},body:JSON.stringify(value)});
  const result=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(result.error||`save HTTP ${response.status}`);
  await refreshObjects(result.digest);
  if(typeof BroadcastChannel!=="undefined") {
    const channel=new BroadcastChannel("meowdoku.marble_objects");
    channel.postMessage({digest:result.digest});channel.close();
  }
  photo.saved(result.digest);
  objectStatus(`${result.duplicate?"Already saved":"Saved"} ${value.name} · ${result.digest.slice(0,12)} · choose it beside Look in Fountain or Fountain3D, or beside Restart in Marble Madness; the game tab refreshes its library when you return`);
}
async function loadObject() {
  const digest=$("saved-object").value;
  if(!digest)throw new Error("select a saved design first");
  const response=await fetch(`/api/marble_objects/${digest}`,{cache:"no-store"});
  if(!response.ok)throw new Error(`load HTTP ${response.status}`);
  const value=await response.json();
  let verified=true,geometry;
  try { ({geometry}=await verifySavedMarbleObject(value)); }
  catch(error) {
    if(!/^saved mesh recipe no longer matches its triangles$|^saved mass or moment of inertia no longer matches the mesh$/.test(String(error.message)))throw error;
    verified=false;
  }
  if(Object.keys(value.materials.index).length!==Object.keys(ids).length||
      Object.entries(ids).some(([name,id])=>value.materials.index[name]!==id)||
      value.materials.extra.coated!==coatedId||value.materials.extra.paintA!==paintAId||
      value.materials.extra.paintB!==paintBId)
    throw new Error("saved material IDs differ from this catalogue; migration is required");
  for(const [name,saved] of Object.entries(value.materials.tables)) {
    const table=tables[name];
    if(!table||saved.stride!==table.table.floats_per_record||saved.data.length!==table.data.length)
      throw new Error("saved material layout differs from this catalogue");
  }
  for(const [name,saved] of Object.entries(value.materials.tables)) {
    tables[name].data.set(saved.data);tables[name].update();
  }
  seed=applyDesign(document,value.design);applyPhysical(document,value.physical);
  opaqueSaved=value.design.opaqueSaved||null;dieDefaultsApplied=value.design["base-shape"]!=="marble";
  $("object-name").value=value.name;
  rebuild();syncMaterial();
  for(const field of ["dice-detail","dice-bevel","dice-face-relief","mark-scale","relief-depth","die-inclusion-size"])
    $(field).dispatchEvent(new Event("input"));
  if(verified) {
    photo.saved(digest);
    objectStatus(`Loaded ${value.name} · verified ${geometry.triangleCount.toLocaleString()} triangles and mass moments`);
  } else {
    objectStatus(`${value.name}: the stored mesh receipt no longer matches this recipe. The recipe is open for inspection; save it as a new object to verify its current mesh. The original is preserved.`);
  }
}
function objectLibraryControls() {
  const run=(action)=>action().catch(error=>objectStatus(error.message||String(error)));
  $("save-object").addEventListener("click",()=>run(saveObject));
  $("load-object").addEventListener("click",()=>run(loadObject));
  $("refresh-objects").addEventListener("click",()=>run(async()=>{
    const count=await refreshObjects();objectStatus(`${count} saved object${count===1?"":"s"}`);
  }));
  run(refreshObjects);
}
function repartitionExterior(force=false) {
  if(!lastGeometry)return;
  const classes=new Map();
  for(const face of lastGeometry.glass) if(!classes.has(face[3])) {
    const id=face[3],transmission=read("pbr",id,5),opacity=read("pbr",id,7);
    classes.set(id,Math.min(opacity,1-0.8*transmission)>=0.999);
  }
  if(!force&&classes.size===exteriorClass.size&&[...classes].every(([id,value])=>exteriorClass.get(id)===value))return;
  exteriorClass=classes;
  dispose(glass);dispose(glassOpaque);
  const opaque=[],transparent=[];
  for(const face of lastGeometry.glass) (classes.get(face[3])?opaque:transparent).push(face);
  glassOpaque=mesh(opaque);glass=mesh(transparent);
}
const idFor = (layer) => layer === "enamel" ? coatedId : layer === "paintA" ? paintAId : layer === "paintB" ? paintBId :
  ids[$({glass:"shell",eye:"eye",accent:"accent",confetti:"confetti-material",mark:"mark-material"}[layer]).value];
const slot = (table, id, offset) => id * tables[table].table.floats_per_record + offset;
const read = (table, id, offset) => tables[table].data[slot(table,id,offset)];
const write = (table, id, offset, value) => { tables[table].data[slot(table,id,offset)] = value; tables[table].update(); };
const selected = () => idFor($("layer").value);
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
function checkerboard() {
  const faces=[], pattern=$("scene-pattern").value;
  const material=(a,b)=>pattern==="checker" ? ((a+b)&1?ids.stone_onyx:ids.ceramic_porcelain)
    : pattern==="porcelain" ? ((a+b)&1?ids.ceramic_porcelain:ids.stone_jade)
      : (a%3===0?ids.metal_brass:ids.metal_steel);
  for (let x=-4;x<4;x++) for (let y=-4;y<4;y++) {
    const z=-2.1, a=[x,y,z],b=[x+1,y,z],c=[x+1,y+1,z],d=[x,y+1,z];
    const n=[0,0,1], id=material(x,y);
    faces.push([a,b,c,id,[n,n,n]],[a,c,d,id,[n,n,n]]);
  }
  for (let x=-4;x<4;x++) for (let z=-4;z<4;z++) {
    const y=-1.02,a=[x,y,z],b=[x,y,z+1],c=[x+1,y,z+1],d=[x+1,y,z];
    const n=[0,1,0],id=material(x,z);
    faces.push([a,b,c,id,[n,n,n]],[a,c,d,id,[n,n,n]]);
  }
  return mesh(faces);
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
}
function rebuild() {
  photo.stale();
  dispose(interior);
  const design=readDesign(document,seed);
  const geometry=buildMarbleObjectGeometry(design,materialIdsForDesign(design,ids,
    {coated:coatedId,paintA:paintAId,paintB:paintBId}));
  const {interior:inside,glass:outside,triangleCount}=geometry;
  interior = mesh(inside); lastGeometry=geometry;
  repartitionExterior(true);
  for (const name of dimensions) $(name+"-out").value = $(name).value;
  message.textContent = `${triangleCount.toLocaleString()} triangles · pattern #${seed}`+
    (geometry.inclusionFit?` · inclusion fit ×${geometry.inclusionFit.factor.toFixed(2)}`:"");
  reportMass(geometry);
}
function controls() {
  const skipPhotoInvalidation=new Set(["object-name","saved-object","layer","preview-engine","preview-camera","preview-motion",
    "preview-axis","preview-hz","preview-phase","preview-arc","inertia-lock","roll-force-x","roll-force-z"]);
  const designChanged=(event)=>{
    if(!event.target.closest("#photo-controls") && !skipPhotoInvalidation.has(event.target.id)) photo.stale();
  };
  for(const eventName of ["input","change"]) $("view").closest("main").querySelector("aside")
    .addEventListener(eventName,designChanged,true);
  for(const id of ["reroll","reset-material"]) $(id).addEventListener("click",()=>photo.stale(),true);
  $("photo-start").addEventListener("click",()=>photo.start(lastGeometry)
    .catch((error)=>photo.status(error.message||String(error))));
  const diceCost=()=>{const kind=$("base-shape").value,
    factor={d4:12,d6:24,d8:24,d10:40,d12:60,d20:60}[kind]||0;
    $("dice-detail-cost").value=kind==="marble"?"Die detail applies to polyhedra":
      `Estimated ${kind.toUpperCase()}: ${(factor*(+$("dice-detail").value)**2).toLocaleString()} triangles`;
  };
  $("base-shape").addEventListener("change",()=>{
    if($("base-shape").value!=="marble"&&!dieDefaultsApplied) {
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
  const lightValue=(name)=>{const divisor=name.endsWith("strength")?10:100;
    return (+$(name).value/divisor).toFixed(2);};
  const lightFields=["light-strength","light-x","light-y","light-z",
    "light-bg-strength","light-kicker-strength","light-fill-strength"];
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
    $("preview-camera").disabled=$("preview-engine").value!=="particle";
    $("drive-pad").hidden=$("preview-engine").value!=="particle";
    if($("preview-engine").value==="particle") {
      sim.status="Loading compiled particle and native containing case…";simStatus();
      startSim().catch((error)=>{sim.status=`Particle simulation unavailable: ${error.message||error}`;simStatus();});
    } else {drive.direction="neutral";driveGravity();sim.runtime?.sync(null);sim.status="Visual preview (cheap): object motion only";simStatus();}
  });
  $("preview-camera").addEventListener("change",()=>{
    $("preview-camera").title=$("preview-camera").value==="marble"?
      "Follows the simulated particle position and physical angular pose":"Shows the whole containing course";
  });
  for(const button of $("drive-pad").querySelectorAll("button[data-drive]")) {
    const direction=button.dataset.drive;
    button.addEventListener("pointerdown",(event)=>{
      event.preventDefault();button.setPointerCapture(event.pointerId);
      if($("preview-motion").value!=="off") {$("preview-motion").value="off";syncPreviewMotion();}
      drive.direction=direction;driveGravity();
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
  $("scene-pattern").addEventListener("change",()=>{dispose(checkerMesh);checkerMesh=checkerboard();});
  for (const name of dimensions) $(name).addEventListener("input", rebuild);
  for (const name of ["eye","accent","confetti-material"]) $(name).addEventListener("change", () => { rebuild(); syncMaterial(); });
  const syncKind=()=>{const hybrid=$("kind").value==="hybrid";
    for(const name of ["orientation-yaw","orientation-pitch","orientation-roll"]) {if(hybrid) $(name).value=0;$(name).disabled=hybrid;$(name+"-out").value=$(name).value;}};
  $("kind").addEventListener("change", () => {syncKind();rebuild();syncMaterial();});
  $("shell").addEventListener("change", () => {$("die-shell-preset").value="custom";$("opaque").checked=false;opaqueSaved=null;copyCoating(); rebuild(); syncMaterial(); });
  for(const field of ["radius-mm","shell-density","eye-density","accent-density","confetti-density","enamel-density","mark-density"])
    $(field).addEventListener("input",()=>{ if(lastGeometry) reportMass(lastGeometry); });
  for(const field of ["paint-a-color","paint-b-color"]) $(field).addEventListener("input",()=>{copyPaint();syncMaterial();});
  $("preset").addEventListener("change",()=>{
    const preset=$("preset").value;
    if(preset==="classic") {$("kind").value="cat_eye";$("shell").value="marble_clear_glass";$("coverage").value=0;$("band-count").value=0;}
    if(preset==="galaxy") {$("kind").value="confetti";$("shell").value="marble_clear_glass";$("coverage").value=55;$("band-count").value=0;$("confetti-count").value=400;}
    if(preset==="uniform") {$("kind").value="confetti";$("shell").value="marble_clear_glass";$("coverage").value=0;$("band-count").value=0;$("confetti-count").value=500;}
    if(preset==="spiral" || preset==="rings") {$("kind").value=preset;$("shell").value="marble_clear_glass";$("coverage").value=0;$("band-count").value=0;}
    if(preset==="bare") {$("kind").value="clear";$("coverage").value=0;$("band-count").value=0;}
    if(preset==="porcelain") {$("kind").value="clear";$("shell").value="ceramic_porcelain";$("coverage").value=0;$("band-count").value=7;$("band-width").value=24;}
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
  $("layer").addEventListener("change", syncMaterial);
  $("reroll").addEventListener("click", () => { seed = (Math.random() * 0x7fffffff) | 0; rebuild(); });
  $("reset-material").addEventListener("click", () => { const id = selected();
    if (id === coatedId) { copyCoating(); repartitionExterior();syncMaterial(); return; }
    if (id === paintAId || id === paintBId) { copyPaint(); repartitionExterior();syncMaterial(); return; }
    for (const table of Object.values(tables)) { const f = table.table.floats_per_record, start = id*f;
      table.data.set(table.original.subarray(start,start+f),start); table.update(); } repartitionExterior();syncMaterial(); });
  for (const [name,[table,offset,scale]] of Object.entries(properties)) $(name).addEventListener("input", () => {
    if ((name==="transmission" || name==="opacity") && ["glass","enamel"].includes($("layer").value)) {
      $("opaque").checked=false;opaqueSaved=null;
    }
    const value = +$(name).value; write(table,selected(),offset,value/scale);
    if(name==="transmission"||name==="opacity") repartitionExterior();
    $(name+"-out").value = (value/scale).toFixed(name === "shininess" || name === "enamel-thickness" ? 0 : 2);
    if(name==="enamel-thickness" && lastGeometry) reportMass(lastGeometry);
  });
  for (const [name,table,offset] of [["albedo","pbr",0],["inner","phong",4],["emission","pbr",8],["enamel-color","enamel",4]])
    $(name).addEventListener("input", () => { const id = selected(); rgb($(name).value).forEach((v,k) => { tables[table].data[slot(table,id,offset+k)] = v; }); tables[table].update(); });
  let last = null;
  canvas.addEventListener("pointerdown", (event) => { last=[event.clientX,event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!last) return;
    // Clamped to match pitch: the backdrop is two finite planes (a back
    // wall, a floor), not an enclosing room. Unclamped yaw could orbit the
    // camera past the wall and look at its unlit backface up close -- a
    // solid-black frame that looks like a renderer bug but is just the
    // camera standing behind a flat backdrop with nothing behind it.
    yaw = Math.max(-1.2,Math.min(1.2,yaw+(event.clientX-last[0])*0.008)); pitch = Math.max(-1.2,Math.min(1.2,pitch+(event.clientY-last[1])*0.008)); last=[event.clientX,event.clientY]; });
  for (const type of ["pointerup","pointercancel"]) canvas.addEventListener(type, () => { last=null; });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    if ($("preview-engine").value==="particle" && $("preview-camera").value==="marble")
      followDistance=Math.max(0.06,Math.min(1.5,followDistance*Math.exp(event.deltaY*0.001)));
    else distance=Math.max(1.2,Math.min(14,distance*Math.exp(event.deltaY*0.001)));
  }, { passive:false });
}
function frame(now=0) {
  const dt=lastFrame?Math.min(0.025,(now-lastFrame)/1000):0;lastFrame=now;
  const simOn=$("preview-engine").value==="particle";
  if(simOn&&sim.runtime?.ready) {
    if(drive.direction!=="neutral"&&drive.yaw!==yaw)driveGravity();
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
  gl.viewport(0,0,w,h); gl.clearColor(0.035,0.075,0.12,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if (P && interior) {
    const follow=simOn&&$("preview-camera").value==="marble";
    const cameraDistance=follow?followDistance:simOn?Math.max(distance,7):distance;
    // The toy backplate normal maps to world +Y. Follow from that side of
    // the course, rather than looking through its opaque side rails.
    const baseView=multiply(translate(-cameraDistance),multiply(rotateX(pitch+(follow?Math.PI/2:0)),rotateY(yaw)));
    const radius=+$("radius-mm").value/1000;
    let object=rollState&&radius>0?multiply(translate3(rollState.position[0]/radius,0,rollState.position[2]/radius),rotateQuaternion(rollState.orientation)):identity();
    const angle=previewMotionAngle($("preview-motion").value,+$("preview-hz").value,+$("preview-phase").value,+$("preview-arc").value,now/1000);
    const axis=$("preview-axis").value;
    const preview=axis==="x"?rotateX(angle):axis==="z"?rotateZ(angle):rotateY(angle);
    let caseM=identity(),haveParticle=true,particleWorld=null,particleObjects=[];
    if(simOn) {
      const runtime=sim.runtime,points=runtime?.points(),lead=livePreviewParticle(points);
      haveParticle=!!(runtime?.ready&&lead&&sim.map);
      if(runtime?.ready&&runtime.shown!=null) caseM=caseModel(runtime.caseAngleAt(runtime.shown*runtime.snap),0,0,runtime.caseAxis||axis,0);
      if(haveParticle) for(let id=0;id<(sim.spec.studio_count||1);id++) {
        const pt=livePreviewParticle(points,id);if(!pt)continue;
        const p=sim.map.point([pt.x,pt.y,pt.z]);
        if(id===0)particleWorld=transformPoint(caseM,p);
        const q=(id===0?runtime.rotation():runtime.rotationFor?.(id))?.quaternion;
        const pose=q?multiply(rotateX(-Math.PI/2),multiply(rotateQuaternion([q[3],q[0],q[1],q[2]]),rotateX(Math.PI/2))):identity();
        particleObjects.push(multiply(caseM,multiply(translate3(...p),multiply(pose,scale3(pt.radius*sim.map.scale)))));
      }
    }
    // The law position is in the case frame. Apply the same physical case
    // transform as the containing mesh before anchoring the camera to it.
    const view=follow&&particleWorld?multiply(baseView,translate3(...particleWorld.map((v)=>-v))):baseView;
    const mv=multiply(view,simOn?object:multiply(object,preview));
    gl.useProgram(P);
    const setTransform=(matrix)=>{ gl.uniformMatrix4fv(loc(P,"uMV"),false,matrix);
      gl.uniformMatrix4fv(loc(P,"uMVP"),false,multiply(perspective(w/h),matrix)); };
    setTransform(mv);
    const key=[+$("light-x").value/100,+$("light-y").value/100,+$("light-z").value/100];
    // Fixed photographic roles, scaled to the current camera distance so
    // they track the subject exactly the way the key does: every light
    // here is a view-space position, which is what makes it "dynamic" --
    // it orbits with the camera with no extra transform needed.
    // Background: behind the subject, splashes the backdrop rather than
    // the subject's face. Kickers: behind and to each side, for rim
    // separation. Fill: large and low, under the camera.
    const background=[0, cameraDistance*0.20, -cameraDistance*1.65];
    const kickerL=[-cameraDistance*0.95, cameraDistance*0.30, -cameraDistance*1.05];
    const kickerR=[ cameraDistance*0.95, cameraDistance*0.30, -cameraDistance*1.05];
    const fill=[0, -cameraDistance*0.95, cameraDistance*0.30];
    const keyStrength=+$("light-strength").value/10;
    const bgStrength=+$("light-bg-strength").value/10;
    const kickerStrength=+$("light-kicker-strength").value/10;
    const fillStrength=+$("light-fill-strength").value/10;
    const keyFilter=rgb($("key-filter").value);
    // Roles: 0 key (filtered), 1 background (neutral), 2-3 kickers (cool rim), 4 fill (neutral-soft).
    const roleColor=[keyFilter,[0.85,0.88,0.95],[0.55,0.75,1.0],[0.55,0.75,1.0],[0.9,0.92,0.9]];
    const roleStrength=[keyStrength,bgStrength,kickerStrength,kickerStrength,fillStrength];
    // No uLightPos/uLightColor/uLightIntensity array at all: a synthetic
    // point light's Phong specular term (H=normalize(L+V)) tracks the
    // viewer, not the light's real position -- confirmed as the actual bug
    // behind the key highlight not lining up with the visible key fixture.
    // The five emissive fixture meshes below are the only lights. The
    // marble sees them exactly the way it already sees the checker
    // backdrop: through its own transmission/reflection, which is
    // genuinely position-correct because it IS the real geometry.
    gl.uniform1i(loc(P,"uNumLights"),0);
    field(P,"uLightCalibration",1); field(P,"uExposure",1); field(P,"uFieldGain",0);
    gl.uniformMatrix3fv(loc(P,"uCatCcmMatrix"),false,new Float32Array([1,0,0,0,1,0,0,0,1]));
    integer(P,"uEnableSpecular",1); integer(P,"uEnableEmissionDirect",0); integer(P,"uRenderPass",0);
    integer(P,"uThermalEmission",0); field(P,"uThermalGain",0); field(P,"uThermalRefK",1000); field(P,"uThermalFloorK",720);
    integer(P,"uShadowEnabled",0); integer(P,"uShadowLightCount",0);
    gl.uniformMatrix4fv(loc(P,"uShadowMatrices[0]"),false,identity());
    gl.vertexAttribI4i(4,0,0,0,0); gl.vertexAttribI4i(5,1,0,0,0); gl.vertexAttrib3f(6,0,0,0); gl.vertexAttrib2f(3,0,0);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    if (simOn&&sim.cageOpaque) {
      setTransform(multiply(view,caseM));gl.bindVertexArray(sim.cageOpaque.vao);
      gl.drawArrays(gl.TRIANGLES,0,sim.cageOpaque.count);setTransform(mv);
    } else if (!simOn&&$("checker").checked && checkerMesh) {
      setTransform(view); gl.bindVertexArray(checkerMesh.vao);
      gl.drawArrays(gl.TRIANGLES,0,checkerMesh.count); setTransform(mv);
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
    // Billboarded to face the camera (view-space +Z) rather than the
    // subject: a panel facing the subject goes edge-on (a near-invisible
    // sliver) whenever the camera's own view direction grazes its surface,
    // which is routinely true for kickers/background sitting behind the
    // subject. Facing the camera always shows the full fixture shape.
    const camFacing=[0,0,1];
    {
      const pbrStride=tables.pbr.table.floats_per_record;
      for (let i=0;i<5;i++) {
        const b=(FIXTURE_MAT_BASE+i)*pbrStride, c=roleColor[i], s=roleStrength[i];
        tables.pbr.data[b+8]=c[0]*s; tables.pbr.data[b+9]=c[1]*s; tables.pbr.data[b+10]=c[2]*s;
      }
      tables.pbr.update();
    }
    const keyIsHard=$("key-shape").value==="grid"||$("key-shape").value==="snoot";
    const fixtureRoles=[
      {pos:key,        halfW:cameraDistance*(keyIsHard?0.028:0.11), halfH:cameraDistance*(keyIsHard?0.028:0.11), mat:FIXTURE_MAT_BASE+0},
      {pos:background, halfW:cameraDistance*0.5,  halfH:cameraDistance*0.35, mat:FIXTURE_MAT_BASE+1},
      {pos:kickerL,    halfW:cameraDistance*0.08, halfH:cameraDistance*0.25, mat:FIXTURE_MAT_BASE+2},
      {pos:kickerR,    halfW:cameraDistance*0.08, halfH:cameraDistance*0.25, mat:FIXTURE_MAT_BASE+3},
      {pos:fill,       halfW:cameraDistance*0.45, halfH:cameraDistance*0.45, mat:FIXTURE_MAT_BASE+4},
    ];
    const fixtureFaces=[];
    for (const role of fixtureRoles) {
      const housingOffset=0.18*Math.max(role.halfW,role.halfH);
      const housingPos=role.pos.map((x,i)=>x-housingOffset*camFacing[i]);
      fixtureFaces.push(...quadFace(housingPos, camFacing, role.halfW*1.3, role.halfH*1.3, FIXTURE_HOUSING_MAT));
      fixtureFaces.push(...quadFace(role.pos, camFacing, role.halfW, role.halfH, role.mat));
    }
    if (fixtureMeshes) dispose(fixtureMeshes);
    fixtureMeshes=mesh(fixtureFaces);
    integer(P,"uEnableEmissionDirect",1);
    // key/background/kickerL/kickerR/fill are already view-space offsets
    // (that's what lets them orbit with the camera for free), so the quad
    // vertices built from them are already in eye space too. Passing the
    // real view matrix here would re-apply the camera's rotation/translation
    // on top of coordinates that are already past that transform, scattering
    // every fixture away from the camera the moment yaw/pitch left zero.
    setTransform(identity()); gl.bindVertexArray(fixtureMeshes.vao);
    gl.drawArrays(gl.TRIANGLES,0,fixtureMeshes.count);
    integer(P,"uEnableEmissionDirect",0);
    setTransform(mv);
    if(haveParticle) for(const model of simOn?particleObjects:[object]) {
      setTransform(simOn?multiply(view,model):mv);
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(interior.vao); gl.drawArrays(gl.TRIANGLES,0,interior.count);
      gl.disable(gl.BLEND);gl.depthMask(true);
      gl.bindVertexArray(glassOpaque.vao);gl.drawArrays(gl.TRIANGLES,0,glassOpaque.count);
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.bindVertexArray(glass.vao); gl.drawArrays(gl.TRIANGLES,0,glass.count);
    }
    if(simOn&&sim.cageGlass) {
      setTransform(multiply(view,caseM));gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
      gl.bindVertexArray(sim.cageGlass.vao);gl.drawArrays(gl.TRIANGLES,0,sim.cageGlass.count);
    }
    gl.depthMask(true); gl.disable(gl.BLEND); gl.bindVertexArray(null);
  }
  requestAnimationFrame(frame);
}
async function boot() {
  if (!gl) return;
  try {
    const [vertex,fragment,assets]=await Promise.all([
      fetch("shaders/base_material.webgl.vert.glsl").then((r)=>{ if(!r.ok) throw Error(`vertex shader ${r.status}`); return r.text(); }),
      fetch("shaders/base_material.webgl.shadow.frag.glsl").then((r)=>{ if(!r.ok) throw Error(`fragment shader ${r.status}`); return r.text(); }),
      fetch("assets.json").then((r)=>{ if(!r.ok) throw Error(`materials ${r.status}`); return r.json(); })]);
    ids=assets.material_index || {};
    for (const name of ["marble_clear_glass","marble_milk_glass","marble_cat_eye_blue","marble_cat_eye_green","marble_cat_eye_amber"])
      if (!Number.isInteger(ids[name])) throw Error(`material ${name} is absent from assets.json; export the catalogue`);
    const source=fragment.replace(/^(#version[^\n]*\n)/,"$1#define PLUCK_TRANSMISSION 1\n");
    P=program(vertex,source);
    gl.useProgram(P);
    coatedId=Math.max(...Object.values(ids))+1; paintAId=coatedId+1; paintBId=coatedId+2;
    tables=Object.fromEntries(Object.entries(assets.materials).map(([name,table])=>[name,floatTexture(table,{pbr:0,phong:1,enamel:2,texstack:3}[name])]));
    // Six reserved spare PBR rows floatTexture() already over-allocates
    // (table.data.length + 6*floats_per_record) -- real material-table
    // slots, not a side channel: 5 distinct emissive rows, one per light
    // role, plus one shared matte-black "housing" row. Pure emitters: black
    // albedo (nothing to reflect), emission holds the glow colour; frame()
    // rewrites each row's emission every frame as baseColor*liveStrength, so
    // the fixture's own glow is actually linked to its power slider, not a
    // fixed boot-time value. Housing: non-zero albedo but zero emission, a
    // visible dark body behind each glowing panel.
    const pbrStride=tables.pbr.table.floats_per_record;
    FIXTURE_MAT_BASE=tables.pbr.table.data.length/pbrStride;
    FIXTURE_HOUSING_MAT=FIXTURE_MAT_BASE+5;
    const base=(slot)=>(FIXTURE_MAT_BASE+slot)*pbrStride;
    for (let slot=0;slot<5;slot++) {
      const b=base(slot);
      for (let i=0;i<pbrStride;i++) tables.pbr.data[b+i]=0;
      tables.pbr.data[b+3]=1; tables.pbr.data[b+7]=1; // roughness, opacity
    }
    {
      const b=base(5);
      for (let i=0;i<pbrStride;i++) tables.pbr.data[b+i]=0;
      tables.pbr.data[b+0]=0.03; tables.pbr.data[b+1]=0.03; tables.pbr.data[b+2]=0.035; // near-black albedo
      tables.pbr.data[b+3]=0.8; tables.pbr.data[b+7]=1; // roughness, opacity; emission stays 0
    }
    tables.pbr.update();
    const names=Object.keys(ids).filter((name)=>Number.isInteger(ids[name])).sort();
    for (const [select,initial] of [["shell","marble_clear_glass"],["confetti-material","metal_gold"],["mark-material","metal_gold"]]) {
      for (const name of names) { const option=document.createElement("option"); option.value=name; option.textContent=name.replaceAll("_"," "); $(select).append(option); }
      $(select).value=initial;
    }
    copyCoating(); copyPaint(); checkerMesh=checkerboard();
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
    controls(); objectLibraryControls(); syncMaterial(); rebuild(); frame();
  } catch (error) { message.textContent=String(error.message||error); console.error("[marble studio]",error); }
}
boot();

// A close-up on the same exported Pluck Phong shader and MaterialDatabase
// tables used by the game. All edits are local to this preview.
import { catEyeMarble } from "./marble_mesh.js";
import { marbleMass } from "./marble_mass.js";

const $ = (id) => document.getElementById(id);
const canvas = $("view"), message = $("message");
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
function perspective(aspect) {
  const f = 1 / Math.tan(Math.PI / 7), near = 0.1, far = 30;
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,2*far*near/(near-far),0]);
}
const loc = (p, name) => gl.getUniformLocation(p, name);
const field = (p, name, value) => gl.uniform1f(loc(p, name), value);
const integer = (p, name, value) => gl.uniform1i(loc(p, name), value);
const hex = (rgb) => "#" + rgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2,"0")).join("");
const rgb = (value) => [1,3,5].map((k) => parseInt(value.slice(k,k+2),16)/255);

function floatTexture(table, unit) {
  const data = new Float32Array(table.data.length + 3*table.floats_per_record), width = 256, height = Math.max(1, Math.ceil(data.length / width));
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

const dimensions = ["width","length","curve","strands","thickness","twist","tip","flare","phase","orientation-yaw","orientation-pitch","orientation-roll","tube-count","tube-radius","tube-turns","confetti-count","coverage","band-count","band-width","band-curve"];
const properties = { roughness:["pbr",3,100], metallic:["pbr",4,100], transmission:["pbr",5,100],
  opacity:["pbr",7,100], ior:["pbr",6,100], ambient:["phong",0,100],
  specular:["phong",1,100], shininess:["phong",2,1],
  "enamel-thickness":["enamel",0,1], "enamel-ior":["enamel",1,100] };
let tables, ids, P, interior, glass, checkerMesh, coatedId, paintAId, paintBId, lastGeometry, seed = 1, yaw = -0.5, pitch = 0.2, distance = 3.45;
let opaqueSaved = null;
const idFor = (layer) => layer === "enamel" ? coatedId : layer === "paintA" ? paintAId : layer === "paintB" ? paintBId :
  ids[$({glass:"shell",eye:"eye",accent:"accent",confetti:"confetti-material"}[layer]).value];
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
    accent:+$("accent-density").value,confetti:+$("confetti-density").value,enamel:+$("enamel-density").value};
  const result=marbleMass(geometry,{radiusM:+$("radius-mm").value/1000,densityByRole,
    enamelMaterial:coatedId,enamelThicknessM:read("enamel",coatedId,0)*1e-9});
  const diag=[0,1,2].map((k)=>result.inertiaKgM2[k][k].toExponential(3)).join(", ");
  $("mass-report").textContent=`Nominal mass ${result.massKg.toFixed(6)} kg · center (${result.centerM.map((x)=>(x*1000).toFixed(2)).join(", ")}) mm · principal-axis tensor diagonal (${diag}) kg·m² · ${result.components.length-1} inclusions`;
}
function checkerboard() {
  const faces=[], dark=ids.stone_onyx, light=ids.ceramic_porcelain;
  for (let x=-4;x<4;x++) for (let y=-4;y<4;y++) {
    const z=-1.8, a=[x,y,z],b=[x+1,y,z],c=[x+1,y+1,z],d=[x,y+1,z];
    const n=[0,0,1], id=(x+y)&1 ? dark : light;
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
    const value = read(table,id,offset) * scale;
    $(name).value = String(value);
    $(name+"-out").value = value.toFixed(name === "shininess" || name === "enamel-thickness" ? 0 : 2);
  }
}
function rebuild() {
  dispose(interior); dispose(glass);
  const geometry = catEyeMarble({
    glass: idFor("glass"), enamel: coatedId, eye: idFor("eye"), accent: idFor("accent"), confetti: idFor("confetti"),
    paintA:paintAId,paintB:paintBId }, {
    rings:96, segments:128, eyeWidth:+$("width").value/100,
    eyeLength:+$("length").value/100, eyeCurve:+$("curve").value/100,
    strandCount:+$("strands").value, ribbonThickness:+$("thickness").value/1000,
    ribbonTwist:+$("twist").value/100, ribbonTip:+$("tip").value/100,
    ribbonFlare:+$("flare").value/100, ribbonPhase:+$("phase").value*Math.PI/180,
    orientationYaw:+$("orientation-yaw").value*Math.PI/180,
    orientationPitch:+$("orientation-pitch").value*Math.PI/180,
    orientationRoll:+$("orientation-roll").value*Math.PI/180,
    confettiCount:+$("confetti-count").value, enamelCoverage:+$("coverage").value/100,
    paintBandCount:+$("band-count").value,paintBandWidth:+$("band-width").value/100,
    paintBandCurve:+$("band-curve").value/100,
    tubeCount:+$("tube-count").value,tubeRadius:+$("tube-radius").value/1000,
    tubeTurns:+$("tube-turns").value/100,
    kind:$("kind").value, seed });
  const {interior:inside,glass:outside,triangleCount}=geometry;
  interior = mesh(inside); glass = mesh(outside); lastGeometry=geometry;
  for (const name of dimensions) $(name+"-out").value = $(name).value;
  message.textContent = `${triangleCount.toLocaleString()} triangles · pattern #${seed}`;
  reportMass(geometry);
}
function controls() {
  for (const name of dimensions) $(name).addEventListener("input", rebuild);
  for (const name of ["eye","accent","confetti-material","kind"]) $(name).addEventListener("change", () => { rebuild(); syncMaterial(); });
  $("shell").addEventListener("change", () => { copyCoating(); rebuild(); syncMaterial(); });
  for(const field of ["radius-mm","shell-density","eye-density","accent-density","confetti-density","enamel-density"])
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
    copyCoating();rebuild();syncMaterial();
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
    syncMaterial();
  });
  $("layer").addEventListener("change", syncMaterial);
  $("reroll").addEventListener("click", () => { seed = (Math.random() * 0x7fffffff) | 0; rebuild(); });
  $("reset-material").addEventListener("click", () => { const id = selected();
    if (id === coatedId) { copyCoating(); syncMaterial(); return; }
    for (const table of Object.values(tables)) { const f = table.table.floats_per_record, start = id*f;
      table.data.set(table.original.subarray(start,start+f),start); table.update(); } syncMaterial(); });
  for (const [name,[table,offset,scale]] of Object.entries(properties)) $(name).addEventListener("input", () => {
    const value = +$(name).value; write(table,selected(),offset,value/scale);
    $(name+"-out").value = (value/scale).toFixed(name === "shininess" || name === "enamel-thickness" ? 0 : 2);
    if(name==="enamel-thickness" && lastGeometry) reportMass(lastGeometry);
  });
  for (const [name,table,offset] of [["albedo","pbr",0],["inner","phong",4],["emission","pbr",8],["enamel-color","enamel",4]])
    $(name).addEventListener("input", () => { const id = selected(); rgb($(name).value).forEach((v,k) => { tables[table].data[slot(table,id,offset+k)] = v; }); tables[table].update(); });
  let last = null;
  canvas.addEventListener("pointerdown", (event) => { last=[event.clientX,event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (!last) return;
    yaw += (event.clientX-last[0])*0.008; pitch = Math.max(-1.2,Math.min(1.2,pitch+(event.clientY-last[1])*0.008)); last=[event.clientX,event.clientY]; });
  for (const type of ["pointerup","pointercancel"]) canvas.addEventListener(type, () => { last=null; });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); distance=Math.max(2.3,Math.min(7,distance*Math.exp(event.deltaY*0.001))); }, { passive:false });
}
function frame() {
  const w=Math.max(1,Math.round(canvas.clientWidth*devicePixelRatio)), h=Math.max(1,Math.round(canvas.clientHeight*devicePixelRatio));
  if (canvas.width!==w || canvas.height!==h) { canvas.width=w; canvas.height=h; }
  gl.viewport(0,0,w,h); gl.clearColor(0.035,0.075,0.12,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  if (P && interior) {
    const mv=multiply(translate(-distance),multiply(rotateX(pitch),rotateY(yaw)));
    gl.useProgram(P);
    const setTransform=(matrix)=>{ gl.uniformMatrix4fv(loc(P,"uMV"),false,matrix);
      gl.uniformMatrix4fv(loc(P,"uMVP"),false,multiply(perspective(w/h),matrix)); };
    setTransform(mv);
    const lights=[[-2.5,3,3],[3,1,2],[-1,-2,-3]];
    gl.uniform1i(loc(P,"uNumLights"),3);
    gl.uniform3fv(loc(P,"uLightPos[0]"),new Float32Array(lights.flat()));
    gl.uniform3fv(loc(P,"uLightColor[0]"),new Float32Array([1,0.9,0.78,0.52,0.72,1,0.5,0.6,1]));
    gl.uniform1fv(loc(P,"uLightIntensity[0]"),new Float32Array([22,10,7]));
    gl.uniform1iv(loc(P,"uLightGroupId[0]"),new Int32Array([-1,-1,-1]));
    gl.uniform3fv(loc(P,"uLightSpotDir[0]"),new Float32Array(9));
    gl.uniform2fv(loc(P,"uLightSpotCone[0]"),new Float32Array(6));
    field(P,"uLightCalibration",1); field(P,"uExposure",1); field(P,"uFieldGain",0);
    gl.uniformMatrix3fv(loc(P,"uCatCcmMatrix"),false,new Float32Array([1,0,0,0,1,0,0,0,1]));
    integer(P,"uEnableSpecular",1); integer(P,"uEnableEmissionDirect",0); integer(P,"uRenderPass",0);
    integer(P,"uThermalEmission",0); field(P,"uThermalGain",0); field(P,"uThermalRefK",1000); field(P,"uThermalFloorK",720);
    integer(P,"uShadowEnabled",0); integer(P,"uShadowLightCount",0);
    gl.uniformMatrix4fv(loc(P,"uShadowMatrices[0]"),false,identity());
    gl.vertexAttribI4i(4,0,0,0,0); gl.vertexAttribI4i(5,1,0,0,0); gl.vertexAttrib3f(6,0,0,0); gl.vertexAttrib2f(3,0,0);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
    if ($("checker").checked && checkerMesh) {
      setTransform(translate(-distance)); gl.bindVertexArray(checkerMesh.vao);
      gl.drawArrays(gl.TRIANGLES,0,checkerMesh.count); setTransform(mv);
    }
    gl.bindVertexArray(interior.vao); gl.drawArrays(gl.TRIANGLES,0,interior.count);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
    gl.bindVertexArray(glass.vao); gl.drawArrays(gl.TRIANGLES,0,glass.count);
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
    const names=Object.keys(ids).filter((name)=>Number.isInteger(ids[name])).sort();
    for (const [select,initial] of [["shell","marble_clear_glass"],["confetti-material","metal_gold"]]) {
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
    controls(); syncMaterial(); rebuild(); frame();
  } catch (error) { message.textContent=String(error.message||error); console.error("[marble studio]",error); }
}
boot();

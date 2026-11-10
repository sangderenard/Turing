// One versioned recipe for the Studio and the particle renderer. The recipe
// invokes the existing mesh generators; saved objects never carry a second
// approximation of their geometry.
import {ensureObjAsset, importedObjectGeometry, importedMechanics} from "./marble_imported_object.js";
import { catEyeMarble } from "./marble_mesh.js";
import { diceSolid, fitInclusionsInsideDie } from "./dice_mesh.js";
import { marbleMass } from "./marble_mass.js";
import { bubbleCount, appendAirBubbles, validateBubbleHost, validateBubbleInclusions, validateBubbleRadius } from "./marble_bubbles.js";
import {verifyOpticalSnapshot} from "./marble_optical_authoring.js";

export const MARBLE_OBJECT_VERSION = 1;
export const DESIGN_FIELDS = [
  "base-shape","die-shell-preset","kind","preset","shell","eye","accent","confetti-material",
  "width","length","curve","strands","ribbon-layout","thickness","twist","tip","flare","phase",
  "orientation-yaw","orientation-pitch","orientation-roll","tube-count","tube-radius","tube-turns",
  "confetti-enabled","confetti-count","coverage","band-count","band-width","band-curve","paint-a-color","paint-b-color",
  "marking-mode","mark-material","mark-scale","relief-depth","dice-detail","die-inclusion-size",
  "dice-bevel","dice-face-relief","compensate-raised","opaque","bubble-count"
];
export const PHYSICAL_FIELDS = ["radius-mm","shell-density","eye-density","accent-density",
  "confetti-density","enamel-density","mark-density","sim-rho","sim-radius",
  "sim-mu","sim-e","sim-roll","sim-irr"];

export function readDesign(root, seed) {
  const design={seed};
  for(const id of DESIGN_FIELDS) {
    const el=root.getElementById(id);
    if(id==="confetti-enabled") {design[id]=!!el?.checked;continue;}
    design[id]=id==="bubble-count"?Number(el?.value ?? 0):el.type==="checkbox"?el.checked:el.type==="range"?Number(el.value):el.value;
  }
  return design;
}
export function applyDesign(root, design) {
  for(const id of DESIGN_FIELDS) {
    if(id === "confetti-enabled" && !(id in design)) { const el=root.getElementById(id);if(el)el.checked=["confetti","hybrid"].includes(design.kind);continue; }
    if(id === "bubble-count" && !(id in design)) { root.getElementById(id).value="0"; continue; }
    if(id === "ribbon-layout" && !(id in design)) {
      root.getElementById(id).value="separated";
      continue;
    }
    if(!(id in design))throw new RangeError(`saved design missing ${id}`);
    const el=root.getElementById(id);
    if(el.type==="checkbox")el.checked=Boolean(design[id]); else el.value=String(design[id]);
  }
  return Number(design.seed);
}
export function readPhysical(root) {
  return Object.fromEntries(PHYSICAL_FIELDS.map(id=>[id,Number(root.getElementById(id).value)]));
}
export function applyPhysical(root, physical) {
  for(const id of PHYSICAL_FIELDS) {
    if(!(id in physical))throw new RangeError(`saved physical properties missing ${id}`);
    root.getElementById(id).value=String(physical[id]);
  }
}

export function buildMarbleObjectGeometry(design, materialIds, {detail="full"}={}) {
  const d=design,m=materialIds;
  if(d["base-shape"]==="obj")return importedObjectGeometry(d,m.shell);
  const bubbles=bubbleCount(d);
  if(bubbles) {
    // The canonical host owns placement; view-dependent LOD never owns it.
    const base={...d,'bubble-count':0};
    const canonical=buildMarbleObjectGeometry(base,m,{detail:'full'});
    // Proxy cores are visual substitutes, not real medium boundaries.
    const mesh=detail==='full'?canonical:buildMarbleObjectGeometry(base,m,{detail:detail==='proxy'?'low':detail});
    return appendAirBubbles(mesh,canonical,d,m.shell);
  }
  if(detail==="proxy"&&d["base-shape"]==="marble") {
    // The distant particle still has two nested, lit triangle surfaces. At a
    // pixel or two, its full ribbons and shell tessellation are subpixel.
    const t=(1+Math.sqrt(5))/2, raw=[[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]];
    const verts=raw.map((v)=>{const n=Math.hypot(...v);return v.map((x)=>x/n);});
    const indices=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
      [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]];
    const shell=[],core=[];
    const add=(out,radius,material)=>{for(const ids of indices){
      const [a,b,c]=ids.map((i)=>verts[i].map((x)=>x*radius));
      const u=b.map((x,i)=>x-a[i]),v=c.map((x,i)=>x-a[i]);
      const n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
      const outward=n.reduce((s,x,i)=>s+x*a[i],0)>0;
      const order=outward?ids:[ids[0],ids[2],ids[1]];
      const points=order.map((i)=>verts[i].map((x)=>x*radius));
      out.push([...points,material,order.map((i)=>verts[i])]);
    }};
    add(shell,1,m.shell);
    if(d.kind!=="clear" || d["confetti-enabled"])add(core,0.72,["clear","confetti"].includes(d.kind)?m.confetti:(m.ribbons?.[0] ?? m.eye));
    return {interior:core,glass:shell,triangleCount:core.length+shell.length};
  }
  if(detail==="proxy")return buildMarbleObjectGeometry(d,m,{detail:"low"});
  const meshDetail={full:[96,128],medium:[24,48],low:[12,24]}[detail];
  if(!meshDetail)throw new RangeError(`unknown mesh detail ${detail}`);
  const inclusions=catEyeMarble({glass:m.shell,enamel:m.coated,eye:m.eye,accent:m.accent,
    confetti:m.confetti,paintA:m.paintA,paintB:m.paintB,ribbons:m.ribbons},{
    rings:meshDetail[0],segments:meshDetail[1],eyeWidth:+d.width/100,eyeLength:+d.length/100,eyeCurve:+d.curve/100,
    strandCount:+d.strands,ribbonLayout:d["ribbon-layout"] ?? "separated",
    ribbonThickness:+d.thickness/1000,ribbonTwist:+d.twist/100,
    ribbonTip:+d.tip/100,ribbonFlare:+d.flare/100,ribbonPhase:+d.phase*Math.PI/180,
    orientationYaw:+d["orientation-yaw"]*Math.PI/180,
    orientationPitch:+d["orientation-pitch"]*Math.PI/180,
    orientationRoll:+d["orientation-roll"]*Math.PI/180,
    confettiEnabled:d["confetti-enabled"],confettiCount:+d["confetti-count"],enamelCoverage:+d.coverage/100,
    paintBandCount:+d["band-count"],paintBandWidth:+d["band-width"]/100,
    paintBandCurve:+d["band-curve"]/100,tubeCount:+d["tube-count"],
    tubeRadius:+d["tube-radius"]/1000,tubeTurns:+d["tube-turns"]/100,
    kind:d.kind,seed:+d.seed});
  if(d["base-shape"]==="marble")return inclusions;
  return fitInclusionsInsideDie(diceSolid(d["base-shape"],{
    shellMaterial:m.shell,markMaterial:m.mark,markingMode:d["marking-mode"],
    reliefDepth:+d["relief-depth"]/1000,markScale:+d["mark-scale"]/100,
    subdivisions:detail==="full"?+d["dice-detail"]:detail==="medium"?24:12,
    bevelRadius:+d["dice-bevel"]/1000,
    faceRelief:+d["dice-face-relief"]/1000,compensateRaised:!!d["compensate-raised"]
  }),inclusions,{scale:+d["die-inclusion-size"]/100});
}

export function materialIdsForDesign(design,index,extra) {
  for(const name of [design.shell,design.eye,design.accent,design["confetti-material"],design["mark-material"]])
    if(!Number.isInteger(index[name]))throw new RangeError(`unknown saved material ${name}`);
  const names=design["ribbon-materials"];
  if(names!==undefined && (!Array.isArray(names)||names.length!==7||names.some(name=>!Number.isInteger(index[name]))))
    throw new RangeError("invalid saved ribbon materials");
  return {ribbons:names?.map(name=>index[name]),shell:index[design.shell],eye:index[design.eye],accent:index[design.accent],
    confetti:index[design["confetti-material"]],mark:index[design["mark-material"]],
    coated:extra.coated,paintA:extra.paintA,paintB:extra.paintB};
}

export function geometryFromSavedMarbleObject(snapshot,{detail="full"}={}) {
  if(snapshot.v!==MARBLE_OBJECT_VERSION||!["meowdoku-marble-object-v1","meowdoku-imported-object-v1"].includes(snapshot.generator))
    throw new RangeError("unsupported saved marble object");
  if((snapshot.generator==="meowdoku-imported-object-v1")!==(snapshot.design["base-shape"]==="obj"))throw new Error("Subject source and generator disagree");
  return buildMarbleObjectGeometry(snapshot.design,
    materialIdsForDesign(snapshot.design,snapshot.materials.index,snapshot.materials.extra),{detail});
}

export async function geometrySha256(geometry) {
  const faces=[...geometry.interior,...geometry.glass],buffer=new ArrayBuffer(faces.length*76),view=new DataView(buffer);
  let at=0;
  for(const [a,b,c,id,n] of faces) {
    for(const vector of [a,b,c,...n])for(const x of vector) {
      if(!Number.isFinite(x))throw new RangeError("nonfinite mesh coordinate");
      view.setFloat32(at,x,true);at+=4;
    }
    view.setUint32(at,id,true);at+=4;
  }
  const hash=await crypto.subtle.digest("SHA-256",buffer);
  return [...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,"0")).join("");
}

export function massFromSavedMarbleObject(snapshot,geometry) {
  if(geometry.imported)return importedMechanics();
  const p=snapshot.physical,m=snapshot.materials.extra,t=snapshot.materials.tables.enamel;
  const densityByRole={shell:+p["shell-density"],eye:+p["eye-density"],accent:+p["accent-density"],
    confetti:+p["confetti-density"],enamel:+p["enamel-density"],mark:+p["mark-density"]};
  return marbleMass(geometry,{radiusM:+p["radius-mm"]/1000,densityByRole,enamelMaterial:m.coated,
    enamelThicknessM:t.data[m.coated*t.stride]*1e-9});
}

// Optional saved analytic descriptors do not select an exposure rendering mode.
// Both endpoints come from the same recipe that produced the triangle receipt.
export function analyticSpheresForDesign(design,geometry,shell,pbr) {
  if(design['base-shape']!=='marble' || +design.coverage!==0 || +design['band-count']!==0 ||
      !geometry.glass.length || geometry.glass.some(face=>face[3]!==shell))return undefined;
  const row=pbr.data.slice(shell*pbr.stride,(shell+1)*pbr.stride);
  if(row.length<11 || !row.slice(8,11).every(Number.isFinite) || row.slice(8,11).some(x=>x!==0))return undefined;
  return {version:1,spheres:[
    {start:geometry.interior.length,count:geometry.glass.length,center:[0,0,0],radius:1},
    ...(geometry.bubbles||[]).map(b=>({start:b.start,count:b.count,center:b.center.slice(),radius:b.radius})),
  ]};
}

export async function verifySavedMarbleObject(snapshot) {
  if(snapshot.generator==="meowdoku-imported-object-v1")await ensureObjAsset(snapshot.design.source_asset);
  const geometry=geometryFromSavedMarbleObject(snapshot),digest=await geometrySha256(geometry);
  if(bubbleCount(snapshot.design)) {
    validateBubbleRadius(snapshot.design,Number(snapshot.physical["radius-mm"]));
    validateBubbleInclusions(geometry,snapshot.materials.tables.pbr);
    validateBubbleHost(snapshot.design,snapshot.materials.index[snapshot.design.shell],snapshot.materials.tables.pbr);
    if(!snapshot.design.region_media && JSON.stringify(snapshot.design.boundary_media)!==JSON.stringify(geometry.boundaryMedia))
      throw new Error('saved air-bubble boundary media no longer match the mesh');
  }
  verifyOpticalSnapshot(snapshot,geometry,snapshot.materials.index[snapshot.design.shell]);
  if(Object.hasOwn(snapshot.design,'analytic_spheres')) {
    const expected=analyticSpheresForDesign(snapshot.design,geometry,
      snapshot.materials.index[snapshot.design.shell],snapshot.materials.tables.pbr);
    if(!expected || JSON.stringify(snapshot.design.analytic_spheres)!==JSON.stringify(expected))
      throw new Error('saved analytic sphere descriptors no longer match the mesh recipe');
  }
  if(digest!==snapshot.geometry_sha256)throw new Error("saved mesh recipe no longer matches its triangles");
  const lods={};
  for(const detail of ["medium","low","proxy"]) {
    const mesh=geometry.imported?geometry:geometryFromSavedMarbleObject(snapshot,{detail});
    const expected=snapshot.design.mesh_lods?.[detail];
    if(expected && (geometry.imported?digest:await geometrySha256(mesh))!==expected)
      throw new Error(`saved ${detail} mesh recipe no longer matches its triangles`);
    lods[detail]=mesh;
  }
  const mass=massFromSavedMarbleObject(snapshot,geometry),expected=snapshot.derived;
  if(geometry.imported) {
    if(Object.keys(expected).length!==Object.keys(mass).length||Object.keys(mass).some(k=>JSON.stringify(expected[k])!==JSON.stringify(mass[k])))throw new Error("Imported subject must not claim certified mechanics");
    return {geometry,mass,lods};
  }
  const near=(a,b)=>Math.abs(a-b)<=Math.max(1e-12,Math.abs(b)*1e-6);
  if(!near(mass.massKg,expected.massKg)||
      mass.centerM.some((x,i)=>!near(x,expected.centerM[i]))||
      mass.inertiaKgM2.some((row,i)=>row.some((x,j)=>!near(x,expected.inertiaKgM2[i][j])))||
      mass.massExact!==expected.massExact)
    throw new Error("saved mass or moment of inertia no longer matches the mesh");
  return {geometry,mass,lods};
}

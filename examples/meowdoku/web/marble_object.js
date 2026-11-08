// One versioned recipe for the Studio and the particle renderer. The recipe
// invokes the existing mesh generators; saved objects never carry a second
// approximation of their geometry.
import { catEyeMarble } from "./marble_mesh.js";
import { diceSolid, fitInclusionsInsideDie } from "./dice_mesh.js";
import { marbleMass } from "./marble_mass.js";

export const MARBLE_OBJECT_VERSION = 1;
export const DESIGN_FIELDS = [
  "base-shape","die-shell-preset","kind","preset","shell","eye","accent","confetti-material",
  "width","length","curve","strands","thickness","twist","tip","flare","phase",
  "orientation-yaw","orientation-pitch","orientation-roll","tube-count","tube-radius","tube-turns",
  "confetti-count","coverage","band-count","band-width","band-curve","paint-a-color","paint-b-color",
  "marking-mode","mark-material","mark-scale","relief-depth","dice-detail","die-inclusion-size",
  "dice-bevel","dice-face-relief","compensate-raised","opaque"
];
export const PHYSICAL_FIELDS = ["radius-mm","shell-density","eye-density","accent-density",
  "confetti-density","enamel-density","mark-density","sim-rho","sim-radius",
  "sim-mu","sim-e","sim-roll","sim-irr"];

export function readDesign(root, seed) {
  const design={seed};
  for(const id of DESIGN_FIELDS) {
    const el=root.getElementById(id);
    design[id]=el.type==="checkbox"?el.checked:el.type==="range"?Number(el.value):el.value;
  }
  return design;
}
export function applyDesign(root, design) {
  for(const id of DESIGN_FIELDS) {
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
  const meshDetail={full:[96,128],medium:[24,48],low:[12,24]}[detail];
  if(!meshDetail)throw new RangeError(`unknown mesh detail ${detail}`);
  const inclusions=catEyeMarble({glass:m.shell,enamel:m.coated,eye:m.eye,accent:m.accent,
    confetti:m.confetti,paintA:m.paintA,paintB:m.paintB},{
    rings:meshDetail[0],segments:meshDetail[1],eyeWidth:+d.width/100,eyeLength:+d.length/100,eyeCurve:+d.curve/100,
    strandCount:+d.strands,ribbonThickness:+d.thickness/1000,ribbonTwist:+d.twist/100,
    ribbonTip:+d.tip/100,ribbonFlare:+d.flare/100,ribbonPhase:+d.phase*Math.PI/180,
    orientationYaw:+d["orientation-yaw"]*Math.PI/180,
    orientationPitch:+d["orientation-pitch"]*Math.PI/180,
    orientationRoll:+d["orientation-roll"]*Math.PI/180,
    confettiCount:+d["confetti-count"],enamelCoverage:+d.coverage/100,
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
  return {shell:index[design.shell],eye:index[design.eye],accent:index[design.accent],
    confetti:index[design["confetti-material"]],mark:index[design["mark-material"]],
    coated:extra.coated,paintA:extra.paintA,paintB:extra.paintB};
}

export function geometryFromSavedMarbleObject(snapshot,{detail="full"}={}) {
  if(snapshot.v!==MARBLE_OBJECT_VERSION||snapshot.generator!=="meowdoku-marble-object-v1")
    throw new RangeError("unsupported saved marble object");
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
  const p=snapshot.physical,m=snapshot.materials.extra,t=snapshot.materials.tables.enamel;
  const densityByRole={shell:+p["shell-density"],eye:+p["eye-density"],accent:+p["accent-density"],
    confetti:+p["confetti-density"],enamel:+p["enamel-density"],mark:+p["mark-density"]};
  return marbleMass(geometry,{radiusM:+p["radius-mm"]/1000,densityByRole,enamelMaterial:m.coated,
    enamelThicknessM:t.data[m.coated*t.stride]*1e-9});
}

export async function verifySavedMarbleObject(snapshot) {
  const geometry=geometryFromSavedMarbleObject(snapshot),digest=await geometrySha256(geometry);
  if(digest!==snapshot.geometry_sha256)throw new Error("saved mesh recipe no longer matches its triangles");
  const lods={};
  for(const detail of ["medium","low"]) {
    const mesh=geometryFromSavedMarbleObject(snapshot,{detail});
    const expected=snapshot.design.mesh_lods?.[detail];
    if(expected && await geometrySha256(mesh)!==expected)
      throw new Error(`saved ${detail} mesh recipe no longer matches its triangles`);
    lods[detail]=mesh;
  }
  const mass=massFromSavedMarbleObject(snapshot,geometry),expected=snapshot.derived;
  const near=(a,b)=>Math.abs(a-b)<=Math.max(1e-12,Math.abs(b)*1e-6);
  if(!near(mass.massKg,expected.massKg)||
      mass.centerM.some((x,i)=>!near(x,expected.centerM[i]))||
      mass.inertiaKgM2.some((row,i)=>row.some((x,j)=>!near(x,expected.inertiaKgM2[i][j])))||
      mass.massExact!==expected.massExact)
    throw new Error("saved mass or moment of inertia no longer matches the mesh");
  return {geometry,mass,lods};
}

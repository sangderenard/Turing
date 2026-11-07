import { buildMarbleObjectGeometry, geometryFromSavedMarbleObject, geometrySha256,
  massFromSavedMarbleObject, verifySavedMarbleObject } from "../examples/meowdoku/web/marble_object.js";
const assert=(ok,message)=>{if(!ok)throw new Error(message);};
const index={marble_clear_glass:1,marble_cat_eye_blue:2,marble_milk_glass:3,
  metal_gold:4,stone_onyx:5};
const extra={coated:6,paintA:7,paintB:8};
const design={seed:17,"base-shape":"d6","die-shell-preset":"clear",kind:"cat_eye",preset:"custom",
  shell:"marble_clear_glass",eye:"marble_cat_eye_blue",accent:"marble_milk_glass",
  "confetti-material":"metal_gold","mark-material":"stone_onyx",
  width:28,length:75,curve:18,strands:3,thickness:35,twist:90,tip:135,flare:18,phase:0,
  "orientation-yaw":0,"orientation-pitch":0,"orientation-roll":0,"tube-count":2,
  "tube-radius":24,"tube-turns":220,"confetti-count":220,coverage:0,"band-count":0,
  "band-width":18,"band-curve":120,"paint-a-color":"#3769a8","paint-b-color":"#e7b35a",
  "marking-mode":"raised","mark-scale":72,"relief-depth":20,"dice-detail":24,
  "die-inclusion-size":100,"dice-bevel":40,"dice-face-relief":10,"compensate-raised":true,opaque:false};
const physical={"radius-mm":10,"shell-density":2500,"eye-density":2800,
  "accent-density":2600,"confetti-density":7800,"enamel-density":3000,"mark-density":2500,
  "sim-rho":1,"sim-radius":0.01,"sim-mu":0.35,"sim-e":0.05,"sim-roll":0,"sim-irr":0};
const tables=Object.fromEntries(["pbr","phong","enamel","texstack"].map(name=>
  [name,{stride:8,data:Array(9*8).fill(0)}]));
const snapshot={v:1,name:"Clear D6",generator:"meowdoku-marble-object-v1",design,
  materials:{index,extra,tables},physical};
const ids={shell:1,eye:2,accent:3,confetti:4,mark:5,...extra};
const geometry=buildMarbleObjectGeometry(design,ids),digest=await geometrySha256(geometry);
snapshot.geometry_sha256=digest;
snapshot.derived=massFromSavedMarbleObject(snapshot,geometry);
const restored=JSON.parse(JSON.stringify(snapshot));
const verified=await verifySavedMarbleObject(restored);
assert(verified.geometry.triangleCount===geometry.triangleCount,"full mesh recipe roundtrip");
assert(verified.mass.massKg===snapshot.derived.massKg,"mass receipt roundtrip");
const low=geometryFromSavedMarbleObject(restored,{detail:"low"});
assert(low.triangleCount<geometry.triangleCount,"low detail retains source design");
assert(new Set(low.interior.map(f=>f[3])).size===new Set(geometry.interior.map(f=>f[3])).size,
  "low detail retains material boundaries");
const altered=JSON.parse(JSON.stringify(restored));altered.design["dice-bevel"]=80;
let rejected=false;try{await verifySavedMarbleObject(altered);}catch(e){rejected=/mesh recipe/.test(e.message);}
assert(rejected,"changed die geometry must fail digest verification");
console.log(`saved object roundtrip: ${geometry.triangleCount} full / ${low.triangleCount} low triangles, verified moments`);

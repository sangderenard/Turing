import {ensureObjAsset} from "./marble_imported_object.js";
// Build one immutable save from captured authoring state. No DOM reads occur
// after hashing starts, so an edit cannot mix a new name/material with old mesh.
import {buildMarbleObjectGeometry, materialIdsForDesign, geometrySha256,
  analyticSpheresForDesign, massFromSavedMarbleObject, verifySavedMarbleObject} from "./marble_object.js";
import {bubbleCount, validateBubbleRadius, validateBubbleHost, validateBubbleInclusions} from "./marble_bubbles.js";
import {prepareOpticalSnapshot,restoreOpticalSurfaceRows} from "./marble_optical_authoring.js";

export function objectSnapshotKey(snapshot) {
  const {design,physical,name,materials}=snapshot;
  const used=[...new Set(Object.values(materialIdsForDesign(design,materials.index,materials.extra)))];
  const rows=Object.fromEntries(Object.entries(materials.tables).map(([name,table])=>[name,
    used.map(id=>[id,table.data.slice(id*table.stride,(id+1)*table.stride)])]));
  // Room/light rows are deliberately excluded from autosave identity.
  return JSON.stringify({design,physical,name,rows,...(materials.authoring?{materialAuthoring:materials.authoring}: {})});
}

export async function prepareObjectSnapshot(captured) {
  const value=structuredClone(captured),{design,materials}=value;
  if(design.optical_authoring?.legacy_pbr)restoreOpticalSurfaceRows(value);
  if(design["base-shape"]==="obj")await ensureObjAsset(design.source_asset);
  const ids=materialIdsForDesign(design,materials.index,materials.extra),pbr=materials.tables.pbr;
  validateBubbleRadius(design,value.physical["radius-mm"]);
  const geometry=buildMarbleObjectGeometry(design,ids);
  if(bubbleCount(design))design.boundary_media=geometry.boundaryMedia;
  prepareOpticalSnapshot(value,geometry,ids.shell);
  validateBubbleHost(design,ids.shell,pbr);
  validateBubbleInclusions(geometry,pbr);
  const spheres=analyticSpheresForDesign(design,geometry,ids.shell,pbr);
  if(spheres)design.analytic_spheres=spheres;
  value.geometry_sha256=await geometrySha256(geometry);
  design.mesh_lods={};
  for(const detail of ["medium","low","proxy"])
    design.mesh_lods[detail]=geometry.imported?value.geometry_sha256:await geometrySha256(buildMarbleObjectGeometry(design,ids,{detail}));
  value.derived=massFromSavedMarbleObject(value,geometry);
  await verifySavedMarbleObject(value);
  return {value,geometry};
}

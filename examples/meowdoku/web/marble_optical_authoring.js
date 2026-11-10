// Pure authoring bridge. Runtime admission remains a separate server decision.
// Input spectra are authored internal transmittance, not RGB-derived physics.
export function regionMediaFromGeometry(mesh, shell, mode) {
  if (!['glass_rod','injection_dye'].includes(mode)) throw new RangeError('Choose glass rod or injection dye.');
  if (!Number.isInteger(shell) || shell<0 || !mesh.glass?.length) throw new RangeError('Missing glass host.');
  const regions=[{start:mesh.interior.length,count:mesh.glass.length,mode:'host',material:shell}];
  const owners=new Uint8Array(mesh.interior.length+mesh.glass.length);
  const parts=[...(mesh.ribbons||[]),...(mesh.tubes||[]),...(mesh.confetti||[])];
  for(const p of parts)regions.push({start:p.start,count:p.count,mode,material:p.material});
  for(const p of mesh.bubbles||[])regions.push({start:p.start,count:p.count,mode:'air_bubble',material:-1});
  const faces=[...mesh.interior,...mesh.glass];
  if(regions.length>1024)throw new RangeError('Too many volume regions.');
  for(const r of regions) {
    if(![r.start,r.count,r.material].every(Number.isInteger) || r.start<0 || r.count<4 || r.start+r.count>faces.length)
      throw new RangeError('Invalid generated volume membership.');
    for(let i=r.start;i<r.start+r.count;i++) {
      if(owners[i]++)throw new RangeError('Duplicate generated volume membership.');
      if(faces[i][3] !== (r.mode==='air_bubble'?shell:r.material))throw new RangeError('Generated volume material disagrees with faces.');
    }
  }
  if(owners.some(n=>n!==1))throw new RangeError('Every generated face needs explicit volume ownership.');
  // Geometric closure, intersections and optical regime are separately checked
  // by server admission. This bridge does not certify arbitrary geometry.
  const result={version:1,regions};
  const o=mesh.parameters;
  if(mode==='glass_rod' && o?.ribbonLayout==='axial' && mesh.ribbons?.length>1) {
    if(o.kind!=='cat_eye' || mesh.tubes?.length || (mesh.confetti?.length && o.confettiEnabled!==true))
      throw new RangeError('Axial glass rod bundles require cat-eye ribbons with optional additive confetti.');
    if(mesh.ribbons.length!==o.strandCount || !mesh.inclusionFrame)
      throw new RangeError('Missing generated axial bundle frame.');
    result.axial_bundle={regions:mesh.ribbons.map((_,i)=>i+1),
      eye_length:o.eyeLength,eye_curve:o.eyeCurve,twist:o.ribbonTwist,phase:o.ribbonPhase,
      frame:structuredClone(mesh.inclusionFrame)};
  }
  return result;
}

export function authoredOpticalMedia(records, regions) {
  const required=new Set(regions.regions.filter(r=>r.material>=0).map(r=>r.material));
  if(!Array.isArray(records)||records.length!==required.size)throw new RangeError('Author a spectrum for every volume material.');
  const seen=new Set();
  for(const r of records) {
    if(!Number.isInteger(r.material)||!required.has(r.material)||seen.has(r.material))throw new RangeError('Invalid optical material membership.');
    seen.add(r.material);
    const w=r.wavelengths_nm,t=r.transmittance;
    if(!Array.isArray(w)||!Array.isArray(t)||w.length<2||w.length>32||w.length!==t.length ||
        !w.every((x,i)=>Number.isFinite(x)&&x>0&&(!i||x>w[i-1])) ||
        !t.every(x=>Number.isFinite(x)&&x>0&&x<=1))throw new RangeError('Use increasing wavelengths and internal transmission in (0, 1].');
    if(!Number.isFinite(r.reference_thickness_mm)||r.reference_thickness_mm<=0 ||
       !Number.isFinite(r.concentration)||r.concentration<0)throw new RangeError('Specify positive thickness and nonnegative concentration.');
  }
  return {version:1,materials:structuredClone(records)};
}

export const OPTICAL_PROVENANCE = 'ILLUSTRATIVE UNCALIBRATED';
export const OPTICAL_WAVELENGTHS = [450,550,650];
export function opticalMaterialIds(mesh) {
  return [...new Set([...mesh.interior,...mesh.glass].map(face=>face[3]))].sort((a,b)=>a-b);
}
export function illustrativeOpticalRecord(material,pbr,{neutralHost=false}={}) {
  const row=pbr.data.slice(material*pbr.stride,(material+1)*pbr.stride);
  // This is only a visibly labelled starting palette. RGB channels cannot
  // identify a physical spectrum or remove surface/Fresnel losses.
  return {material,wavelengths_nm:[...OPTICAL_WAVELENGTHS],
    // A nominal clear host starts as an ideal nonabsorbing dielectric.
    // Interface Fresnel loss remains; a tiny RGB display tint must not become
    // repeated bulk absorption per millimetre. Existing authored records stay intact.
    transmittance:neutralHost?[1,1,1]:[2,1,0].map(i=>Number(Math.max(.02,Math.min(1,row[i]||.02)).toFixed(4))),
    reference_thickness_mm:1,concentration:1,ior:row[6]};
}
export function validateOpticalAuthoring(value) {
  if(value?.version!==1 || !['legacy','glass_rod','injection_dye'].includes(value.mode) ||
      value.provenance!==OPTICAL_PROVENANCE || !Array.isArray(value.materials))
    throw new RangeError('Invalid optical authoring profile.');
  const seen=new Set();
  for(const record of value.materials) {
    if(!Number.isInteger(record.material)||record.material<0||seen.has(record.material) ||
        JSON.stringify(record.wavelengths_nm)!==JSON.stringify(OPTICAL_WAVELENGTHS) ||
        !Number.isFinite(record.ior)||record.ior<1||record.ior>3)
      throw new RangeError('Optical materials require unique IDs, 450/550/650 nm samples and IOR from 1 to 3.');
    seen.add(record.material);
  }
  authoredOpticalMedia(value.materials,{regions:value.materials.map(r=>({material:r.material}))});
  return value;
}
export function restoreOpticalSurfaceRows(snapshot) {
  const a=snapshot.design.optical_authoring;
  if(!a || a.mode==='legacy')return;
  const pbr=snapshot.materials.tables.pbr;
  if(!Array.isArray(a.legacy_pbr)||a.legacy_pbr.length!==a.materials.length)
    throw new RangeError('Missing original optical surface rows.');
  for(const [i,row] of a.legacy_pbr.entries()) {
    if(!Array.isArray(row)||row.length!==pbr.stride||!row.every(Number.isFinite))
      throw new RangeError('Invalid original optical surface row.');
    pbr.data.splice(a.materials[i].material*pbr.stride,pbr.stride,...row);
  }
}
export function prepareOpticalSnapshot(snapshot,mesh,shell) {
  const d=snapshot.design,a=d.optical_authoring;
  if(!a)return;
  validateOpticalAuthoring(a);
  if(a.mode==='legacy') {
    if(d.region_media||d.optical_media)throw new RangeError('Legacy mode cannot contain active optical media.');
    return;
  }
  if(!['marble','d4','d6','d8','d10','d12','d20'].includes(d['base-shape'])||+d.coverage!==0||+d['band-count']!==0||+snapshot.physical['radius-mm']<1)
    throw new RangeError('Optical volumes require an uncoated generated host with radius at least 1 mm.');
  if(d['base-shape']!=='marble'&&+d['bubble-count']>0)throw new RangeError('Optical die bubbles are not yet supported.');
  const regions=regionMediaFromGeometry(mesh,shell,a.mode);
  const {materials}=authoredOpticalMedia(a.materials,regions);
  const pbr=snapshot.materials.tables.pbr,host=materials.find(r=>r.material===shell);
  if(!host||pbr.stride<11)throw new RangeError('Optical volumes require a complete glass host material.');
  a.legacy_pbr=materials.map(r=>pbr.data.slice(r.material*pbr.stride,(r.material+1)*pbr.stride));
  for(const r of materials) {
    const offset=r.material*pbr.stride,row=pbr.data.slice(offset,offset+pbr.stride);
    if(row.length!==pbr.stride||!row.every(Number.isFinite)||row[4]!==0||row.slice(8,11).some(x=>x!==0))
      throw new RangeError('Optical volumes require nonmetal, nonemissive materials.');
    // Explicit opt-in changes the captured copy only. Legacy surface alpha
    // does not encode absorption; internal transmittance owns Beer-Lambert.
    pbr.data[offset+5]=1;pbr.data[offset+7]=0;
    pbr.data[offset+6]=a.mode==='injection_dye'?host.ior:r.ior;
  }
  d.region_media=regions;
  d.optical_media=authoredOpticalMedia(materials.map(({ior,...record})=>record),regions);
  delete d.boundary_media;
}
export function verifyOpticalSnapshot(snapshot,mesh,shell) {
  if(!snapshot.design.optical_authoring)return;
  validateOpticalAuthoring(snapshot.design.optical_authoring);
  const expected=structuredClone(snapshot);
  restoreOpticalSurfaceRows(expected);
  prepareOpticalSnapshot(expected,mesh,shell);
  for(const key of ['region_media','optical_media'])
    if(JSON.stringify(expected.design[key])!==JSON.stringify(snapshot.design[key]))
      throw new Error(`saved ${key} no longer matches its optical authoring recipe`);
  if(JSON.stringify(expected.materials.tables.pbr)!==JSON.stringify(snapshot.materials.tables.pbr))
    throw new Error('saved optical surface rows no longer match their authoring recipe');
}

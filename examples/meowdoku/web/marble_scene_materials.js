import {authoredOpticalMedia} from './marble_optical_authoring.js';

// These are selections and snapshots of the existing material buffers, not a
// second material database. Material IDs and PBR offsets stay authoritative.
export function validateSceneEngines(value) {
  if(value===undefined)return {version:1,stage:'legacy',room:'legacy'};
  if(value?.version!==1||!['legacy','dielectric'].includes(value.stage)||!['legacy','dielectric'].includes(value.room))
    throw new Error('Select Legacy or Dielectric for each scene material group.');
  return {version:1,stage:value.stage,room:value.room};
}

export function validateSceneMaterialEdits(value,names) {
  if(value===undefined)return undefined;
  if(value?.version!==1||!Array.isArray(value.materials)||value.materials.length>names.length)
    throw new Error('Invalid shared scene material edits.');
  const seen=new Set();
  for(const material of value.materials) {
    if(!names.includes(material.name)||seen.has(material.name)||!material.tables||typeof material.tables!=='object')
      throw new Error('Unknown or duplicate shared scene material.');
    seen.add(material.name);
    for(const [name,row] of Object.entries(material.tables))
      if(!['pbr','phong','enamel','texstack'].includes(name)||!Number.isInteger(row?.stride)||row.stride<1||row.stride>64||
        !Array.isArray(row.data)||row.data.length!==row.stride||!row.data.every(Number.isFinite))
        throw new Error('Invalid shared material buffer row.');
    if(!material.tables.pbr||material.tables.pbr.stride<16)throw new Error('Missing shared PBR row.');
    if(material.optical) {
      const r=material.optical;
      if(JSON.stringify(r.wavelengths_nm)!=='[450,550,650]'||!Array.isArray(r.transmittance)||r.transmittance.length!==3||
        !r.transmittance.every(x=>x===null||Number.isFinite(x)&&x>0&&x<=1)||
        !(r.reference_thickness_mm===null||Number.isFinite(r.reference_thickness_mm)&&r.reference_thickness_mm>0)||
        !Number.isFinite(r.concentration)||r.concentration<0)
        throw new Error('Invalid scene spectrum. Use transmittance in (0, 1] and positive reference thickness.');
    }
  }
  return structuredClone(value);
}

export function sceneMaterialEngineContract({engines,parts,materialIndex,pbr,getOpticalRecord}) {
  engines=validateSceneEngines(engines);
  const groups=['stage','room'].map(name=>({name,engine:engines[name],ranges:[]})),volumes=[],required=new Map();
  let start=0;
  for(const part of parts) {
    const group=groups.find(g=>g.name===part.group);
    if(!group)throw new Error('Every scene part needs its actual stage or room owner.');
    const count=part.faces.length;
    if(!count)continue;
    const ranges=group.ranges,last=ranges.at(-1);
    if(last&&last.start+last.count===start)last.count+=count;else ranges.push({start,count});
    const materials=[...new Set(part.faces.map(f=>f[3]))];
    const transmissive=materials.filter(id=>pbr.data[id*pbr.stride+5]>0||pbr.data[id*pbr.stride+7]<.999);
    if(group.engine==='dielectric'&&transmissive.length) {
      if(!part.closed||materials.length!==1)
        throw new Error(`${part.label??part.id}: Dielectric needs a closed, single-material volume. Open sheets and mixed-material solids cannot silently use Legacy.`);
      const id=materials[0],local=materialIndex.get(id),r=getOpticalRecord?.(id);
      if(!r)throw new Error(`${part.label??part.id}: enter internal transmittance and reference thickness in the shared material editor before Dielectric capture.`);
      const {ior,...record}=r;
      try{authoredOpticalMedia([{...record,material:local}],{regions:[{material:local}]});}
      catch(error){throw new Error(`${part.label??part.id}: complete internal transmittance and reference thickness in the shared material editor. ${error.message}`);}
      required.set(local,{...record,material:local});volumes.push({start,count,material:local});
    }
    start+=count;
  }
  return {version:1,groups,volumes,optical_media:{version:1,materials:[...required.values()]}};
}

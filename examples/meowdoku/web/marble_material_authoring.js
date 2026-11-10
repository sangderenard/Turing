// The canonical YAML is the authoring source. Exported definitions are only a
// derived view for choosing existing paths; native compilation belongs to the
// server. Never persist the six compiled chunks or process-local profile IDs.
export const CANONICAL_VOLUME_OPTICS = Object.freeze({version:1,source:'native_spectral_bands'});
const plain=value=>value!==null&&typeof value==='object'&&!Array.isArray(value);
const exact=(value,keys)=>plain(value)&&Object.keys(value).length===keys.length&&keys.every(key=>Object.hasOwn(value,key));
const keyPattern=/^[A-Za-z0-9_][A-Za-z0-9_.-]*$/;
const revisionPattern=/^[a-f0-9]{64}$/;
const clone=value=>structuredClone(value);
function finiteTree(value,depth=0,budget={left:100000}) {
  if(depth>48||--budget.left<0)throw new RangeError('Material override is too deeply nested or large.');
  if(value===null||typeof value==='string'||typeof value==='boolean')return;
  if(typeof value==='number'&&Number.isFinite(value))return;
  if(Array.isArray(value)){for(const item of value)finiteTree(item,depth+1,budget);return;}
  if(plain(value)){for(const item of Object.values(value))finiteTree(item,depth+1,budget);return;}
  throw new RangeError('Material overrides must contain finite JSON values.');
}
export function validateCanonicalSource(source) {
  if(!exact(source,['key','revision','source_yaml'])||!keyPattern.test(source.key)||
      !revisionPattern.test(source.revision)||typeof source.source_yaml!=='string'||
      new TextEncoder().encode(source.source_yaml).length>65536)
    throw new RangeError('Invalid canonical material source or revision.');
  return source;
}
function validateOverrides(overrides) {
  if(!Array.isArray(overrides)||overrides.length>256)throw new RangeError('Material overrides must be a list of at most 256 edits.');
  for(const override of overrides) {
    if(!exact(override,['path','value'])||!Array.isArray(override.path)||!override.path.length||
        !override.path.every(key=>typeof key==='string'||Number.isSafeInteger(key)&&key>=0))
      throw new RangeError('An override requires an existing key/index path and value.');
    finiteTree(override.value);
  }
}
export function validateMaterialAuthoring(authoring,{materialCount=Infinity,requiredIds=null}={}) {
  if(!exact(authoring,['version','records'])||authoring.version!==1||!Array.isArray(authoring.records)||authoring.records.length>128)
    throw new RangeError('Invalid canonical material authoring contract.');
  const seen=new Set();
  for(const record of authoring.records) {
    if(!exact(record,['material','source','overrides'])||!Number.isSafeInteger(record.material)||record.material<0||
        record.material>=materialCount||seen.has(record.material))throw new RangeError('Invalid or duplicate canonical material ID.');
    seen.add(record.material);validateCanonicalSource(record.source);validateOverrides(record.overrides);
  }
  if(requiredIds!==null) {
    const required=new Set(requiredIds);
    if(required.size!==seen.size||[...required].some(id=>!seen.has(id)))
      throw new RangeError('Canonical sources must cover exactly all volume materials.');
  }
  return authoring;
}
function child(node,key) {
  if(Array.isArray(node)) {
    if(!Number.isSafeInteger(key)||key<0||key>=node.length)throw new RangeError('Material override path does not exist.');
  } else if(!plain(node)||typeof key!=='string'||!Object.hasOwn(node,key))
    throw new RangeError('Material override path does not exist.');
  return node[key];
}
export function applyCanonicalOverrides(definition,overrides) {
  if(!plain(definition))throw new RangeError('Canonical material definition must be a mapping.');
  finiteTree(definition);validateOverrides(overrides);const result=clone(definition);
  for(const {path,value} of overrides) {
    let node=result;for(const key of path.slice(0,-1))node=child(node,key);
    const key=path.at(-1);child(node,key);
    // Define an own property rather than invoking Object.prototype setters.
    Object.defineProperty(node,key,{value:clone(value),enumerable:true,writable:true,configurable:true});
  }
  return result;
}
export function canonicalMaterialRecord(exported,material) {
  validateCanonicalSource(exported?.source);
  if(exported.key!==exported.source.key||!plain(exported.definition))throw new RangeError('Invalid canonical library record.');
  const record={material,source:clone(exported.source),overrides:[]};
  validateMaterialAuthoring({version:1,records:[record]});return record;
}
export function canonicalMaterialView(record,exported) {
  validateMaterialAuthoring({version:1,records:[record]});
  if(record.source.key!==exported?.key||JSON.stringify(record.source)!==JSON.stringify(exported?.source))
    throw new RangeError('The exported material view does not match this saved source revision.');
  return applyCanonicalOverrides(exported.definition,record.overrides);
}
export function setCanonicalMaterialField(record,exported,path,value) {
  const current=canonicalMaterialView(record,exported);
  applyCanonicalOverrides(current,[{path,value}]);
  let old=current;for(const key of path)old=child(old,key);
  if(JSON.stringify(old)===JSON.stringify(value))return clone(record);
  const result=clone(record),last=result.overrides.at(-1);
  // Coalesce consecutive edits to the same knob without reordering ancestor
  // edits or changing the meaning of a previously saved override sequence.
  if(last&&JSON.stringify(last.path)===JSON.stringify(path))last.value=clone(value);
  else result.overrides.push({path:clone(path),value:clone(value)});
  validateOverrides(result.overrides);return result;
}
export function canonicalBandEditor(record,exported) {
  const definition=canonicalMaterialView(record,exported);
  const field=Object.hasOwn(definition,'bands')?'bands':Object.hasOwn(definition,'spectral_bands')?'spectral_bands':null;
  const result={editable:false,requiresServerValidation:true,sourceNativeCompiles:exported.native_compiles===true,fields:[]};
  if(!field||!Array.isArray(definition[field])||!definition[field].length)
    return {...result,reason:'This source has no explicit native spectral bands. No bands are synthesized.'};
  for(const [index,band] of definition[field].entries()) {
    if(!plain(band)||!Object.hasOwn(band,'ior_real')||!Object.hasOwn(band,'ior_imag'))
      return {...result,reason:'Native volume bands need explicit real n and imaginary k fields.'};
    if(!Number.isFinite(band.ior_real)||band.ior_real<1||!Number.isFinite(band.ior_imag)||band.ior_imag<0||band.ior_imag>.001)
      return {...result,reason:'This volume path requires n >= 1 and weak absorption 0 <= k <= 0.001.'};
    for(const name of ['ior_real','ior_imag'])result.fields.push({path:[field,index,name],value:band[name],
      label:`Band ${index+1} ${name==='ior_real'?'n':'k'}`});
  }
  return {...result,editable:true,reason:'Existing source n/k fields are editable. This does not certify Studio transport; the server validates the source and its dependencies.'};
}
export async function verifyCanonicalSourceRevisions(authoring) {
  validateMaterialAuthoring(authoring);
  for(const {source} of authoring.records) {
    const bytes=new TextEncoder().encode(source.source_yaml),hash=await crypto.subtle.digest('SHA-256',bytes);
    const revision=[...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('');
    if(revision!==source.revision)throw new RangeError('Canonical material source revision mismatch.');
  }
  return authoring;
}
export function captureNativeOpticalAuthoring(mode,materialIds) {
  if(!['glass_rod','injection_dye'].includes(mode))throw new RangeError('Choose glass rod or injection dye.');
  if(!Array.isArray(materialIds)||!materialIds.length||new Set(materialIds).size!==materialIds.length||
      !materialIds.every(id=>Number.isSafeInteger(id)&&id>=0))throw new RangeError('Native volume material IDs must be unique.');
  return {version:2,mode,source:'native_spectral_bands',materials:materialIds.map(material=>({material}))};
}

// An explicit user interpretation of original ColorProfile samples as internal
// T. No RGB fitting, normalization, extrapolation or invented profile occurs.
export function applyColorProfileTransmittance(authoring,material,imported,referenceThicknessMM) {
  if(authoring?.version!==1||!Array.isArray(authoring.materials))throw new RangeError('Profile transmittance applies to reference-T authoring only.');
  if(!plain(imported)||typeof imported.profile_name!=='string'||!imported.profile_name||
      !revisionPattern.test(imported.source_revision)||imported.interpretation!=='authored_internal_transmittance'||
      JSON.stringify(imported.wavelengths_nm)!=='[450,550,650]'||!Array.isArray(imported.transmittance)||
      imported.transmittance.length!==3||!imported.transmittance.every(x=>Number.isFinite(x)&&x>0&&x<=1)||
      !Number.isFinite(referenceThicknessMM)||referenceThicknessMM<=0)
    throw new RangeError('Use original exported ColorProfile samples and an explicit positive reference thickness.');
  finiteTree(imported);
  const result=clone(authoring),record=result.materials.find(r=>r.material===material);
  if(!record)throw new RangeError('The selected material is not a volume material.');
  record.wavelengths_nm=clone(imported.wavelengths_nm);record.transmittance=clone(imported.transmittance);
  record.reference_thickness_mm=referenceThicknessMM;
  result.color_profile_sources??={};
  result.color_profile_sources[String(material)]=clone(imported);
  return result;
}
export function detachColorProfileSource(authoring,material) {
  const result=clone(authoring);
  if(result.color_profile_sources) {
    delete result.color_profile_sources[String(material)];
    if(!Object.keys(result.color_profile_sources).length)delete result.color_profile_sources;
  }
  return result;
}

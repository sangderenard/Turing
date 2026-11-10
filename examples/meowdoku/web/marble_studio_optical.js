import {OPTICAL_PROVENANCE, illustrativeOpticalRecord, opticalMaterialIds,
  validateOpticalAuthoring} from './marble_optical_authoring.js';

// Authoring controls must not lift the independent backend transport gate.
export const OPTICAL_RUNTIME_ENABLED = true;
export function createOpticalMaterialControls({doc=document,getContext,onChange=()=>{},
  runtimeEnabled=OPTICAL_RUNTIME_ENABLED}={}) {
  const $=id=>doc.getElementById(id),panel=$('optical-material-controls');
  if(!panel)return null;
  const mode=$('optical-mode'),rows=$('optical-material-rows'),status=$('optical-material-status');
  const sceneRecordIds=new Set();
  let profile=runtimeEnabled?{version:1,mode:'glass_rod',provenance:OPTICAL_PROVENANCE}:null,records=new Map(),touched=runtimeEnabled;
  const seedRecord=(id,context)=>illustrativeOpticalRecord(id,context.pbr,
    {neutralHost:id===context.shell&&context.design?.shell==='marble_clear_glass'});
  const numberInput=(parent,label,value,{min,max,step='any'},changed)=>{
    const box=doc.createElement('label'),input=doc.createElement('input');
    box.textContent=label;input.type='number';input.value=value===null?'':String(value);input.min=String(min);
    if(max!==undefined)input.max=String(max);
    input.step=step;input.required=true;input.setAttribute('aria-label',label);
    input.addEventListener('input',()=>{
      changed(input.value.trim()===''?(getContext().selectedGroup?null:NaN):Number(input.value));touched=true;onChange({sceneChanged:!!getContext().selectedGroup});
    });
    box.append(input);parent.append(box);return input;
  };
  function refresh() {
    const context=getContext();if(!context?.geometry)return;
    const ids=opticalMaterialIds(context.geometry);
    for(const id of ids)if(!records.has(id))records.set(id,seedRecord(id,context));
    const selected=profile?.mode??'legacy';mode.value=selected;
    const material=Number.isInteger(context.selectedMaterial)?context.selectedMaterial:ids[0];
    const sceneGroup=context.selectedGroup,sceneEngine=context.sceneEngines?.[sceneGroup]??'legacy';
    const transmissive=context.pbr.data[material*context.pbr.stride+5]>0||context.pbr.data[material*context.pbr.stride+7]<.999;
    const isVolume=sceneGroup?transmissive:ids.includes(material);
    const optical=(sceneGroup?sceneEngine==='dielectric':selected!=='legacy')&&runtimeEnabled&&isVolume;
    if(sceneGroup&&!records.has(material))sceneRecord(material,true);
    const disable=(id,value)=>{const input=$(id);if(input){input.disabled=value;input.setAttribute('aria-disabled',String(value));}};
    for(const id of ['opacity','transmission','roughness','metallic','albedo','inner','emission',
      'specular','shininess','enamel-thickness','enamel-ior','enamel-color'])disable(id,optical);
    disable('opaque',selected!=='legacy'&&runtimeEnabled);
    disable('ior',!sceneGroup&&optical&&selected==='injection_dye'&&material!==context.shell);
    for(const option of mode.options)if(option.value!=='legacy')option.disabled=!runtimeEnabled;
    rows.replaceChildren();
    // One material editor and one selected material. Disabled fields retain data.
    const r=records.get(material)??{wavelengths_nm:[450,550,650],transmittance:[1,1,1],reference_thickness_mm:1,concentration:1};
    const add=(label,value,limits,changed)=>{
      const input=numberInput(rows,label,value,limits,changed);input.disabled=!optical;
      input.setAttribute('aria-disabled',String(!optical));
    };
    for(const [i,wave] of r.wavelengths_nm.entries())
      add(`Internal transmittance ${wave} nm`,r.transmittance[i],{min:.000001,max:1},v=>r.transmittance[i]=v);
    add('Reference thickness (mm)',r.reference_thickness_mm,{min:.000001},v=>r.reference_thickness_mm=v);
    add('Dye concentration',r.concentration,{min:0},v=>r.concentration=v);
    if(status){status.textContent=sceneGroup&&optical?'Internal transmittance is measured through the reference thickness, excluding surface losses. Unset values block ray capture; no RGB or opacity conversion.':sceneGroup?'Shared catalogue material: changes affect every use of this material. Dielectric spectral fields apply to transmissive closed volumes.':'';status.hidden=!status.textContent;}

  }
  mode.addEventListener('change',()=>{
    if(!runtimeEnabled&&mode.value!=='legacy'){mode.value=profile?.mode??'legacy';return;}
    profile={version:1,mode:mode.value,provenance:OPTICAL_PROVENANCE};touched=true;refresh();onChange({modeChanged:true});
  });
  function capture(geometry=getContext().geometry) {
    if(!touched&&!profile)return undefined;
    const ids=opticalMaterialIds(geometry);
    for(const id of ids)if(!records.has(id))records.set(id,seedRecord(id,getContext()));
    // Shared PBR IOR is the authoring value; snapshot engine adapters may
    // resolve injection dye to the host without rewriting this source row.
    const shared=getContext().pbr;
    for(const id of ids)records.get(id).ior=shared.data[id*shared.stride+6];
    const result={version:1,mode:profile?.mode??'legacy',provenance:OPTICAL_PROVENANCE,
      materials:ids.map(id=>structuredClone(records.get(id)))};
    return structuredClone(validateOpticalAuthoring(result));
  }
  function restore(design) {
    const value=design.optical_authoring;
    if(value)validateOpticalAuthoring(value);
    profile=value?{version:1,mode:value.mode,provenance:value.provenance}:null;
    records=new Map([...Array.from(sceneRecordIds).filter(id=>records.has(id)).map(id=>[id,records.get(id)]),...(value?.materials??[]).map(r=>[r.material,structuredClone(r)])]);touched=!!value;
    // Import the already-authored optical IOR into the shared editable view.
    // Saved source files and inactive stored spectra are never modified.
    if(value&&value.mode!=='legacy') {
      const shared=getContext().pbr;
      for(const r of value.materials)shared.data[r.material*shared.stride+6]=r.ior;
      getContext().updateMaterialTables?.();
    }
    refresh();
  }
  function sceneRecord(id,create=false) {
    if(create&&!records.has(id))records.set(id,{material:id,wavelengths_nm:[450,550,650],
      transmittance:[null,null,null],reference_thickness_mm:null,concentration:1});
    const r=records.get(id);if(!r)return undefined;
    sceneRecordIds.add(id);const shared=getContext().pbr;
    return {...structuredClone(r),ior:shared.data[id*shared.stride+6]};
  }
  function restoreSceneRecords(values) {
    for(const r of values){records.set(r.material,structuredClone(r));sceneRecordIds.add(r.material);}
    refresh();
  }
  refresh();return {capture,restore,refresh,sceneRecord,restoreSceneRecords,get mode(){return profile?.mode??'legacy';}};
}

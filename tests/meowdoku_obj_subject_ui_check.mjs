import assert from 'node:assert/strict';import fs from 'node:fs';import vm from 'node:vm';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8');
const code=source.slice(source.indexOf('async function refreshObjSubjects('),source.indexOf('function repartitionExterior('));
function fixture() {
 const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:true,textContent:'',listeners:{},addEventListener(k,f){this.listeners[k]=f;},append(){},files:[]});return elements.get(id);};
 $('obj-asset').value='a'.repeat(64);$('preview-engine').value='visual';$('base-shape').value='marble';
 const state={revision:0,built:0,rebuilds:0};
 const ctx={$,suppliedSubjectAsset:null,objectAutosave:{dirty:false},objectSnapshotKey:JSON.stringify,
 captureObjectSnapshot:()=>({revision:state.revision}),ensureObjAsset:async digest=>({digest,name:'Test bunny'}),
 readSubjectDesign:()=>({shell:'glass'}),ids:{glass:0},coatedId:0,paintAId:0,paintBId:0,
 materialIdsForDesign:()=>({shell:0}),buildMarbleObjectGeometry:()=>{state.built++;},
 opticalControls:{restore(){}},rebuild(){state.rebuilds++;},syncMaterial(){},objectEdited(){},objectStatus(){},
 fetch:async()=>({ok:true,json:async()=>({entries:[]})}),document:{createElement:()=>({})},confirm:()=>false};
 vm.createContext(ctx);vm.runInContext(code,ctx);ctx.setupObjSubjects(fn=>fn());
 return {ctx,$,state,use:()=> $('obj-use').listeners.click()};
}
let f=fixture();f.$('photo-cancel').disabled=false;await assert.rejects(f.use(),/Finish the exposure/);assert.equal(f.state.built,0);
f=fixture();f.$('live-rays').checked=true;await assert.rejects(f.use(),/Finish the exposure/);
f=fixture();f.$('preview-engine').value='particle';await assert.rejects(f.use(),/Finish the exposure/);
f=fixture();f.ctx.ensureObjAsset=async()=>{f.state.revision++;return {name:'Later'};};await assert.rejects(f.use(),/changed while loading/);assert.equal(f.ctx.suppliedSubjectAsset,null);
f=fixture();f.ctx.buildMarbleObjectGeometry=()=>{throw Error('degenerate');};await assert.rejects(f.use(),/degenerate/);assert.equal(f.$('base-shape').value,'marble');assert.equal(f.ctx.suppliedSubjectAsset,null);
f=fixture();await f.use();assert.equal(f.$('base-shape').value,'obj');assert.equal(f.state.rebuilds,1);assert.equal(f.$('object-name').value,'Test bunny');
console.log('OBJ subject UI preserves active jobs, simulation, newer edits and invalid inputs; valid subject applies');

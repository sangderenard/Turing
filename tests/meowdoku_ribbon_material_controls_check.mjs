import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.value='';}
 append(child){child.parent=this;this.children.push(child);}
 remove(){this.parent.children=this.parent.children.filter(c=>c!==this);}
 get options(){return this.children.flatMap(c=>c.tag==='option'?[c]:c.options);}
 querySelector(){return this.children.find(c=>c.dataset.ribbonMaterials)||null;}
}
const layer=new Element('select');const glass=new Element('option');glass.value='glass';layer.append(glass);layer.value='glass';
const document={createElement:tag=>new Element(tag)};
const ids={eye:1,accent:2,ribbon_1:3,ribbon_2:4,ribbon_3:5,ribbon_4:6,ribbon_5:7,ribbon_6:8,ribbon_7:9};
const tables=Object.fromEntries(['pbr','phong','enamel','texstack'].map(name=>[name,{table:{floats_per_record:2},data:Float32Array.from({length:20},(_,i)=>i),update(){}}]));
const context=vm.createContext({document,$:()=>layer,ids,tables,ribbonMaterialNames:['eye','accent','eye','accent','eye','accent','eye'],lastGeometry:{ribbons:[{},{}]},rebuild(){},objectEdited(){}});
for(const root of ['examples/meowdoku/web','build/dot-marble-composition-20261110/stage']){
 const source=fs.readFileSync(`${root}/marble_studio.js`,'utf8');
 vm.runInContext(source.slice(source.indexOf('function refreshRibbonMaterialTargets()'),source.indexOf('function copyCoating()')),context);
 vm.runInContext('refreshRibbonMaterialTargets()',context);
 assert.deepEqual(layer.options.map(o=>o.textContent).filter(Boolean),['Ribbon 1','Ribbon 2']);
 layer.value='ribbon:1';vm.runInContext('makeRibbonIndependent()',context);
 assert.equal(context.ribbonMaterialNames[1],'ribbon_2');
 assert.deepEqual(Array.from(tables.pbr.data.slice(8,10)),[4,5]);
 assert.deepEqual(Array.from(tables.pbr.data.slice(2,4)),[2,3]);
 context.lastGeometry={ribbons:Array(7).fill({})};vm.runInContext('refreshRibbonMaterialTargets()',context);
 assert.equal(layer.options.length,8);assert.equal(layer.value,'ribbon:1');
 context.lastGeometry={ribbons:[{}]};vm.runInContext('refreshRibbonMaterialTargets()',context);
 assert.equal(layer.options.length,2);assert.equal(layer.value,'glass');
 context.lastGeometry={ribbons:[]};vm.runInContext('refreshRibbonMaterialTargets()',context);assert.equal(layer.options.length,1);
 context.lastGeometry={ribbons:[{},{}]};
}
console.log('PASS source/staged ribbon material UI: dynamic 1/2/7/0 entries, selection fallback, copy-on-edit isolation and all four shared tables');

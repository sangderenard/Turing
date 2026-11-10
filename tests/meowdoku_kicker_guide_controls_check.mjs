import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createKickerHeightGuides,kickerGuideHeights} from '../examples/meowdoku/web/marble_kicker_guide_controls.js';
import {KickerTableQuery} from '../examples/meowdoku/web/marble_kicker_guide.js';
import {defaultStudioTable,tableFloorDrop} from '../examples/meowdoku/web/marble_studio_table.js';
import {studioScaleTransform} from '../examples/meowdoku/web/marble_studio_scale.js';
class Element{
 constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.dataset={};this.style={};this.value='';this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;}
 setAttribute(key,value){this[key]=value;}
 addEventListener(kind,callback){(this.events[kind]??=[]).push(callback);}
 fire(kind){for(const callback of this.events[kind]??[])callback();}
}
const walk=node=>[node,...node.children.filter(n=>typeof n==='object').flatMap(walk)];
const doc={createElement:tag=>new Element(tag,doc)},container=doc.createElement('div'),find=id=>walk(container).find(e=>e.id===id);
const table=defaultStudioTable(),floorY=-1.02-tableFloorDrop(table),scale=studioScaleTransform({version:1,mode:'fixed',unitMm:304.8},10,true);
const state={leftY:-2,rightY:-2,locked:true},initial=structuredClone(state);
const snapshots=()=>['left','right'].map(side=>{
 const pos=[side==='left'?-6:6,state[side+'Y'],-6],direction=scale.subjectCenter.map((v,i)=>v-pos[i]),n=Math.hypot(...direction);
 return {id:'kicker-'+side,table,floorY,ceilingY:23.1,unitM:scale.unitM,subject:{center:scale.subjectCenter,radius:scale.subjectRadius},locked:state.locked,
  role:{pos,dir:direction.map(v=>v/n),radius:.2,gobo:true,shroudLength:.44,shroudRadius:.22,shroudExitRadius:.22}};
});
assert.ok(kickerGuideHeights(snapshots()[0]).includes(4),'Human-work height is sampled exactly');
const oldFetch=globalThis.fetch;globalThis.fetch=async url=>{try{return new Response(await readFile(url));}catch{return new Response('',{status:404});}};
let controls;
try{
 const query=await KickerTableQuery.create(new URL('../build/dot-client-ray-20261109/studio-compute/',import.meta.url));
 controls=createKickerHeightGuides(container,{getSnapshots:snapshots,createQuery:async()=>query,delay:0,
  setHeight:(id,height)=>{state[id.endsWith('left')?'leftY':'rightY']=floorY+height;controls.refresh();}});
 const settle=async()=>{for(let i=0;i<200;i++){if(!['left','right'].some(s=>find('kicker-'+s+'-table-guide').textContent.startsWith('Checking')))return;await new Promise(r=>setTimeout(r,5));}throw Error('Guide did not settle');};
 controls.refresh();await settle();assert.deepEqual(state,initial,'Guide never relocates saved poses');
 assert.match(find('kicker-left-table-guide').textContent,/49\/49 sampled rays/);
 assert.equal(find('kicker-left-aperture-guide').dataset.coverage,'clear');
 const samples=find('kicker-left-height-samples');assert.ok(samples.children.some(e=>e.dataset.coverage==='some'));assert.ok(samples.children.some(e=>e.dataset.coverage==='none'));assert.ok(samples.children.some(e=>e.dataset.coverage==='all'));
 samples.children.find(e=>Number(e.dataset.height)===4).fire('click');await settle();
 assert.equal(state.leftY,-1);assert.match(find('kicker-left-table-guide').textContent,/0\/49 sampled rays/);assert.equal(state.rightY,-2);
 const input=find('kicker-left-height');assert.equal(input.value,'4');assert.ok(controls.ownsControl(input.id));
 input.value='';input.fire('input');assert.equal(state.leftY,-1);assert.match(find('kicker-left-table-guide').textContent,/Enter a height/);
 input.value='3';input.fire('input');input.value='4';input.fire('input');await settle();await new Promise(r=>setTimeout(r,30));
 assert.match(find('kicker-left-table-guide').textContent,/0\/49 sampled rays/,'Stale lower-height result cannot overwrite latest height');
 state.locked=false;controls.refresh();await settle();assert.equal(find('kicker-left-height-samples').children.length,0);
 assert.match(find('kicker-left-table-guide').textContent,/sampled geometric coverage/);
 controls.dispose();state.leftY=-2;controls.refresh();assert.equal(find('kicker-left-height').value,'4');
 console.log('Kicker height UI: original compiled table samples, all/some/none bands, exact floor-relative pose editing, manual-aim gating, invalid input, stale-result rejection and disposal passed');
}finally{controls?.dispose();globalThis.fetch=oldFetch;}

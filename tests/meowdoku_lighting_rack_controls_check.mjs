import assert from 'node:assert/strict';
import {createLightingRackControls} from '../examples/meowdoku/web/marble_lighting_rack_controls.js';
class Element {
 constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.textContent='';this.disabled=false;}
 append(...nodes){this.children.push(...nodes);}
 setAttribute(key,value){this[key]=value;}
 addEventListener(kind,fn){(this.events[kind]??=[]).push(fn);}
 fire(kind){for(const fn of this.events[kind]??[])fn({target:this});}
}
const walk=e=>[e,...e.children.flatMap(walk)];
const doc={createElement:tag=>new Element(tag,doc)},container=new Element('div',doc);
let state={version:1,buses:['upper-wall','middle-wall','edge','near-edge','center'].map(id=>({id,enabled:false,dropFeet:null,dimmer:null})),shopDimmer:.15,houseDimmer:1,allOut:false,solo:null,craneX:0,craneZ:0};
let calls=0,reject=false,builds=0,maxDrop=12,received;
const controls=createLightingRackControls(container,{getState:()=>state,setState:next=>{received=next;if(reject)throw Error('Blocked fixture clearance');calls++;state=structuredClone(next);},getMaxDropFeet:()=>maxDrop,onBuildRacks:()=>{builds++;}});
const find=id=>walk(container).find(e=>e.id==='lighting-rack-'+id);
const change=(id,value)=>{find(id).value=String(value);find(id).fire('input');};
assert.equal(find('shop-dimmer').value,'0.15');assert.equal(find('house-dimmer').value,'1');assert.ok(walk(container).some(e=>e.tagName==='label'&&e.textContent.includes('Studio lights')));assert.equal(find('edge-drop-clear').disabled,true);
const authored=structuredClone(state);change('shop-dimmer',.4);assert.equal(state.shopDimmer,.4);assert.deepEqual(state.buses,authored.buses);assert.notEqual(received,state);
find('shop-dimmer-clear').fire('click');assert.equal(state.shopDimmer,null);assert.equal(find('shop-dimmer-out').textContent,'Individual settings');
change('edge-dimmer',.7);assert.equal(state.buses[2].dimmer,.7);assert.equal(state.buses[1].dimmer,null);
change('edge-drop',5);assert.equal(state.buses[2].dropFeet,5);assert.equal(find('edge-drop-clear').disabled,false);
find('edge-drop-clear').fire('click');assert.equal(state.buses[2].dropFeet,null);
find('edge-solo').fire('click');assert.equal(state.solo,'edge');assert.equal(find('edge-solo')['aria-pressed'],'true');find('edge-solo').fire('click');assert.equal(state.solo,null);
find('center-solo').fire('click');find('clear-solo').fire('click');assert.equal(state.solo,null);
find('all-out').fire('click');assert.equal(state.allOut,true);find('all-out').fire('click');assert.equal(state.allOut,false);
change('craneX',.8);change('craneZ',-.6);change('center-drop',2);find('all-out').fire('click');find('clear-all').fire('click');
assert.equal(state.shopDimmer,null);assert.equal(state.houseDimmer,null);assert.equal(state.allOut,false);assert.equal(state.solo,null);
assert.equal(state.craneX,.8);assert.equal(state.craneZ,-.6);assert.ok(state.buses.every(b=>b.dimmer===null&&b.dropFeet===null&&!b.enabled));
const before=structuredClone(state),count=calls;for(const [id,value]of [['shop-dimmer',2],['edge-drop',13],['craneZ',-2],['shop-dimmer','']]){change(id,value);assert.deepEqual(state,before);assert.equal(calls,count);assert.match(find('status').textContent,/outside/);}
reject=true;change('house-dimmer',.2);assert.deepEqual(state,before);assert.equal(find('house-dimmer-out').textContent,'Individual settings');assert.match(find('status').textContent,/Blocked/);reject=false;
maxDrop=4;controls.sync();assert.equal(find('center-drop').max,'4');change('center-drop',4);assert.equal(state.buses[4].dropFeet,4);
find('build').fire('click');assert.equal(builds,1);assert.equal(find('status').textContent,'');
assert.equal(controls.ownsControl('lighting-rack-center-drop'),true);assert.equal(controls.ownsControl('not-owned'),false);
const ids=walk(container).map(e=>e.id).filter(Boolean);assert.equal(new Set(ids).size,ids.length);
console.log('Lighting rack controls: dimmers, nullable overrides, chain limits, solo/all-out, clear preservation, detached updates, rejected edits, crane offsets and build action passed');

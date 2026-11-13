import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
class Element {
  constructor(){this.children=[];this.handlers={};this.value='50';}
  append(...children){this.children.push(...children);}
  after(child){this.afterElement=child;}
  setAttribute(key,value){this[key]=value;}
  addEventListener(name,fn){this.handlers[name]=fn;}
}
const elements=new Map();
const win=new Element();
const doc={defaultView:win,createElement:()=>new Element(),getElementById:id=>{
  if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
}};
globalThis.localStorage={getItem:()=>null,setItem(){},removeItem(){}};
const calls=[];let resolveRadio;
const graph={nodes:['Sync','Flash','Gate','studio:0','mod:1','mod:2','mod:3'],edges:[]};
globalThis.fetch=async (_url,options)=>{
  const body=JSON.parse(options.body);calls.push(body);
  const value=body.action==='radio'?await new Promise(resolve=>resolveRadio=resolve):{graph};
  return {ok:true,json:async()=>value};
};
const source=(await fs.readFile(new URL('../examples/meowdoku/web/marble_trigger_grid.js',import.meta.url),'utf8'))
  .replace("import {LIGHTING_BUSES} from './marble_lighting_racks.js';",'const LIGHTING_BUSES=[];');
const {mountMarbleTriggerGrid}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const controller=mountMarbleTriggerGrid({doc,getReceivers:()=>['studio:0']});
await new Promise(resolve=>setImmediate(resolve));
assert.equal(calls.length,1);
assert.equal(controller.levels(performance.now()),null);
await win.handlers['nodus-radio-signal']({detail:{status:'live',rms:999}});
assert.equal(calls.length,1,'no connection until opt-in');
const jack=elements.get('camera-trigger-grid-status').afterElement;
jack.handlers.click();
const pending=win.handlers['nodus-radio-signal']({detail:{status:'live',rms:999}});
assert.deepEqual(calls.at(-1),{action:'radio'},'never send untrusted browser measurements');
resolveRadio({status:'live',active:['studio:0'],gains:{'studio:0':.3}});await pending;
assert.equal(controller.levels(performance.now()).gains['studio:0'],.3);
assert.equal(controller.levels(performance.now()+2500),null,'stale data releases the override');
const late=win.handlers['nodus-radio-signal']({detail:{status:'live'}});
jack.handlers.click();resolveRadio({status:'live',active:['studio:0'],gains:{'studio:0':.8}});await late;
assert.equal(controller.levels(performance.now()),null,'late request cannot reconnect');
jack.handlers.click();
const again=win.handlers['nodus-radio-signal']({detail:{status:'live'}});
resolveRadio({status:'live',active:['studio:0'],gains:{'studio:0':.4}});await again;
await win.handlers['nodus-radio-signal']({detail:{status:'unavailable'}});
assert.equal(controller.levels(performance.now()),null,'closing widget releases override');
win.handlers.pagehide();assert.equal(controller.levels(performance.now()),null);
console.log('PASS: opt-in, server-source only, stale/disconnect/late-response radio lighting lifecycle');

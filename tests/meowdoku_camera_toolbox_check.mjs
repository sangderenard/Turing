import assert from 'node:assert/strict';
import {createCameraToolbox} from '../examples/meowdoku/web/marble_camera_toolbox.js';
import {mountStudioCameraToolbox,containedCanvasRect} from '../examples/meowdoku/web/marble_camera_toolbox_mount.js';
import {execFileSync} from 'node:child_process';

class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.style={};this.handlers=new Map();this.attributes={};this.value='';}
  append(...nodes){this.children.push(...nodes);nodes.forEach(n=>{n.parent=this;});}
  prepend(...nodes){this.children.unshift(...nodes);nodes.forEach(n=>{n.parent=this;});}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(n=>n!==this);}
  setAttribute(k,v){this.attributes[k]=v;}
  setCustomValidity(v){this.validity=v;}
  addEventListener(type,fn,opts){const list=this.handlers.get(type)||[];list.push({fn,opts});this.handlers.set(type,list);}
  removeEventListener(type,fn){this.handlers.set(type,(this.handlers.get(type)||[]).filter(r=>r.fn!==fn));}
  getBoundingClientRect(){return {left:10,top:20,width:200,height:100};}
  fire(type,props={}){const e={type,button:0,isPrimary:true,clientX:110,clientY:70,
    preventDefault(){this.prevented=true;},stopImmediatePropagation(){this.stopped=true;},...props};
    for(const {fn}of this.handlers.get(type)||[]){fn(e);if(e.stopped)break;}return e;}
}
const body=new Element('body'),mount=new Element(),stage=new Element('canvas'),photo=new Element('canvas');
stage.style.cursor='grab';photo.style.cursor='auto';body.append(mount,stage,photo);
const find=(el,id)=>el.id===id?el:el.children.map(n=>find(n,id)).find(Boolean);
const doc=new Element('document');doc.body=body;doc.hidden=false;doc.defaultView=new Element('window');doc.createElement=tag=>new Element(tag);doc.getElementById=id=>find(body,id);
const flush=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
let revision=1,reader=null,requests=[],results=[],statuses=[];
const caps={focus:{available:false,reason:'Exact mixed-geometry focus has no native reader.'},exposure:{available:true},white:{available:true}};
const toolbox=createCameraToolbox({doc,mount,surfaces:[{element:stage,kinds:['focus'],label:'Studio',revision:()=>revision},
  {element:photo,kinds:['exposure','white'],label:'completed photo',revision:()=>revision}],
  capabilities:caps,sample:request=>{requests.push(request);return reader?reader.promise:Promise.resolve({sampling:'One accumulated pixel',rgb:[1,1,1]});},
  onResult:(result,request)=>results.push({result,request}),onStatus:message=>statuses.push(message)});
assert.equal(mount.children[0],toolbox.root);assert.equal(toolbox.tool,'hand');
assert.equal(stage.fire('pointerdown').prevented,undefined,'hand leaves existing camera handlers alone');
assert.equal(doc.getElementById('camera-tool-focus').disabled,true);
assert.match(doc.getElementById('camera-tool-availability').textContent,/no native reader/);
assert.equal(toolbox.select('focus'),false);assert.equal(toolbox.tool,'hand');
assert.equal(toolbox.select('exposure'),true);assert.equal(photo.style.cursor,'crosshair');
assert.equal(stage.fire('pointerdown').prevented,true);assert.equal(requests.length,0,'wrong surface never measured');
assert.equal(photo.fire('pointerdown').prevented,true);await flush();assert.equal(results.length,1);
assert.equal(requests[0].u,.5);assert.equal(requests[0].v,.5);assert.equal(requests[0].spread,0);
assert.match(statuses.at(-1),/One accumulated pixel/);
const spread=doc.getElementById('camera-tool-spread');spread.value='.25';spread.fire('change');
photo.fire('pointerdown');await flush();assert.equal(requests.at(-1).spread,.25);
spread.value='custom';spread.fire('change');const sigma=doc.getElementById('camera-tool-sigma');
sigma.value='';sigma.fire('input');assert.match(statuses.at(-1),/0.1–50/);
const priorRequests=requests.length;photo.fire('pointerdown');await flush();assert.equal(requests.length,priorRequests);
sigma.value='50';sigma.fire('input');photo.fire('pointerdown');await flush();assert.equal(requests.at(-1).spread,.5);
sigma.value='51';sigma.fire('input');assert.match(statuses.at(-1),/spread/);
spread.value='0';spread.fire('change');
reader=deferred();photo.fire('pointerdown');const stale=requests.at(-1);assert.equal(toolbox.pending,true);
const latest=deferred();reader=latest;photo.fire('pointerdown');assert.equal(stale.signal.aborted,true);
latest.resolve({message:'Latest result'});await flush();const count=results.length;
assert.equal(toolbox.pending,false);
// The abandoned promise may still resolve, but cannot invoke the result owner.
// Keep its handle separately for a direct out-of-order test.
const old=deferred();reader=old;photo.fire('pointerdown');reader=deferred();photo.fire('pointerdown');
old.resolve({message:'Old result'});await flush();assert.equal(results.length,count);
toolbox.select('hand');assert.equal(requests.at(-1).signal.aborted,true);reader.resolve({});await flush();assert.equal(results.length,count);
toolbox.select('white');reader=deferred();photo.fire('pointerdown');revision++;reader.resolve({});await flush();
assert.match(statuses.at(-1),/changed/);assert.equal(results.length,count);
reader=deferred();photo.fire('pointerdown');doc.fire('keydown',{key:'Escape'});assert.equal(toolbox.tool,'hand');
assert.equal(requests.at(-1).signal.aborted,true);reader.resolve({});await flush();
toolbox.select('white');reader=deferred();photo.fire('pointerdown');doc.hidden=true;doc.fire('visibilitychange');
assert.equal(requests.at(-1).signal.aborted,true);doc.hidden=false;
reader.resolve({});await flush();assert.equal(results.length,count);
reader=deferred();photo.fire('pointerdown');toolbox.invalidate();reader.resolve({});await flush();assert.equal(results.length,count);
toolbox.setCapabilities({...caps,white:{available:false,reason:'No selected HDR'}});assert.equal(toolbox.tool,'hand');
toolbox.select('exposure');reader=deferred();photo.fire('pointerdown');toolbox.dispose();
assert.equal(requests.at(-1).signal.aborted,true);reader.resolve({});await flush();assert.equal(results.length,count);
assert.equal(doc.getElementById('camera-toolbox'),undefined);assert.equal(stage.style.cursor,'grab');assert.equal(photo.style.cursor,'auto');
assert.equal((photo.handlers.get('pointerdown')||[]).length,0);toolbox.dispose();

// Real HDR parser and proposal path connected to an owner stub. No adjustment
// reaches that owner until its separate Apply button is explicitly clicked.
const npz=execFileSync('python',['-c',`import io,sys,numpy as np
b=io.BytesIO();np.savez_compressed(b,linear_srgb_exposure=np.array([[[1.,2.,3.]]]))
sys.stdout.buffer.write(b.getvalue())`],{timeout:20000});
const bytes=npz.buffer.slice(npz.byteOffset,npz.byteOffset+npz.byteLength),oldFetch=globalThis.fetch;
let source={id:'fixture:hdr',url:'/hdr-fixture',revision:'one',complete:true,profiled:true,params:{},whiteReference:10};
const observers=new Set(),applied=[];
const controller={completedHdr:()=>source,subscribeCompletedHdr(fn){observers.add(fn);fn(source);return()=>observers.delete(fn);},
  applySpotAdjustment(proposal,binding){assert.equal(binding.sourceId,source.id);assert.equal(binding.sourceRevision,source.revision);
    applied.push(proposal);source=null;observers.forEach(fn=>fn(source));}};
try{
  photo.width=1;photo.height=1;assert.deepEqual(containedCanvasRect(photo),{left:60,top:20,width:100,height:100});
  photo.width=4;photo.height=1;assert.deepEqual(containedCanvasRect(photo),{left:10,top:45,width:200,height:50});
  photo.width=1;photo.height=1;
  globalThis.fetch=async()=>({ok:true,headers:new Headers(),arrayBuffer:async()=>bytes.slice(0)});
  const mounted=mountStudioCameraToolbox({doc,photo:controller,canvas:stage,photoCanvas:photo,mount});
  mounted.toolbox.select('white');photo.fire('pointerdown',{clientX:30});await flush();
  assert.match(doc.getElementById('camera-tool-status').textContent,/letterbox/);
  assert.equal(doc.getElementById('camera-tool-apply').disabled,true);
  photo.fire('pointerdown');
  // DecompressionStream crosses actual asynchronous I/O boundaries.
  for(let i=0;i<50&&doc.getElementById('camera-tool-apply').disabled;i++)await new Promise(r=>setTimeout(r,5));
  assert.equal(applied.length,0);assert.equal(doc.getElementById('camera-tool-apply').disabled,false);
  assert.match(doc.getElementById('camera-tool-proposal').textContent,/Treat the selected point as neutral/);
  doc.getElementById('camera-tool-apply').fire('click');assert.equal(applied.length,1);
  assert.equal(doc.getElementById('camera-tool-apply').disabled,true);
  assert.equal(doc.getElementById('camera-tool-focus').disabled,true);
  assert.equal(doc.getElementById('camera-tool-white').disabled,true);
  mounted.dispose();assert.equal(observers.size,0);
}finally{globalThis.fetch=oldFetch;}
console.log('Camera toolbox: hand default, honest capabilities, matched surfaces, Gaussian choice, stale/cancel/repeat/dispose tests PASS');

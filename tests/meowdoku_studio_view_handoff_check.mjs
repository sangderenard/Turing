import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {createStudioGameProtocol,validateStudioGameProtocol,validateStudioGameReceipt} from '../examples/meowdoku/web/marble_studio_game_protocol.js';

let source=readFileSync(new URL('../examples/meowdoku/web/marble_studio_game_layer.js',import.meta.url),'utf8');
source=source.replace(/^import .*;\r?\n/gm,'').replace(/^export function /gm,'function ');
const nodes=[],calls=[],photoCamera={view:[1,2,3,4],yaw:.2,pitch:.3,focus:7},before=JSON.stringify(photoCamera);
const protocol=createStudioGameProtocol('preserved-session');
const host={readProtocol:()=>protocol,readSnapshot:()=>null,update(){throw Error('No authored frame was requested');},
  readQualification:()=>({state:'pending',simulatedSeconds:5}),
  setActive(value){calls.push(['active',value]);},connect(){calls.push(['connect']);return ()=>calls.push(['disconnect']);},
  dispose(){calls.push(['dispose']);}};
const node=tag=>{
  const value={tag,style:{},dataset:{},children:[],handlers:{},hidden:false,removed:false,
    append(...children){this.children.push(...children);},addEventListener(name,handler){this.handlers[name]=handler;},
    remove(){this.removed=true;},contentWindow:tag==='iframe'?{mechanicalCreatureStudioLayer:host}:undefined};
  nodes.push(value);return value;
};
const doc={createElement:node,getElementById:()=>null,body:node('body'),location:{href:'https://same.test/marble_studio_layer.html',origin:'https://same.test'}};
const context=vm.createContext({document:doc,URL,console,Error,TypeError,JSON,validateStudioGameProtocol,validateStudioGameReceipt,
  createStudioSceneAsset(){throw Error('No capture expected');},prepareStudioAmmunition(){throw Error('No ammunition expected');},
  createMechanicalCreatureStudioReturn(){throw Error('No return renderer expected');}});
vm.runInContext(source,context);
const layer=context.mountStudioGameLayer({doc,canvas:{focus(){calls.push(['photo-focus']);}},capture:()=>new Promise(()=>{})});
const button=text=>nodes.find(n=>n.tag==='button'&&n.textContent===text);
const open=button('Open original game layer'),back=button('Return to Studio view'),stop=button('Stop game layer');
open.handlers.click();const frame=nodes.find(n=>n.tag==='iframe'),url=frame.src;
assert.equal(nodes.filter(n=>n.tag==='iframe').length,1);assert.equal(back.hidden,false);
// Return while its document is still loading must not reactivate hidden input.
back.handlers.click();frame.handlers.load();
assert.equal(calls.filter(c=>c[0]==='connect').length,1);assert.deepEqual(calls.filter(c=>c[0]==='active').at(-1),['active',false]);
assert.equal(frame.style.pointerEvents,'none');assert.equal(frame.removed,false);
open.handlers.click();assert.deepEqual(calls.filter(c=>c[0]==='active').at(-1),['active',true]);
assert.equal(frame.style.visibility,'visible','original qualification gate is visible before any Studio receipt');
assert.equal(layer.read().sentRevision,-1);
host.readQualification=()=>({state:'failed',simulatedSeconds:20,failures:['native tire crossing']});
layer.frame({revision:1});
assert.equal(frame.style.visibility,'visible','failure gate remains visible without launching the world');
assert.match(nodes.find(n=>n.id==='studio-game-layer-status').textContent,/qualification failed/);
back.handlers.click();assert.deepEqual(calls.filter(c=>c[0]==='active').at(-1),['active',false]);
open.handlers.click();
assert.equal(nodes.filter(n=>n.tag==='iframe').length,1);assert.equal(frame.src,url);
assert.equal(calls.filter(c=>c[0]==='connect').length,1);assert.equal(calls.filter(c=>c[0]==='dispose').length,0);
assert.equal(calls.filter(c=>c[0]==='photo-focus').length,2);assert.equal(JSON.stringify(photoCamera),before);
stop.handlers.click();assert.equal(frame.removed,true);assert.equal(calls.filter(c=>c[0]==='dispose').length,1);
assert.equal(calls.filter(c=>c[0]==='disconnect').length,1);assert.equal(JSON.stringify(photoCamera),before);
console.log(JSON.stringify({pass:true,returnWhileLoading:true,sameIframeOnReopen:true,originalControlHandoff:true,photoCameraUnchanged:true}));

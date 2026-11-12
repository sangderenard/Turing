// Offline lifecycle test: real header module, minimal DOM boundary, no network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
class Node {
  constructor(tag='node') { this.tag=tag; this.children=[]; this.attrs={}; this.handlers={}; this.hidden=false; this.classes=new Set(); this.classList={toggle:(k,on)=>on?this.classes.add(k):this.classes.delete(k)}; this.style={setProperty(k,v){this[k]=v;}}; }
  setAttribute(k,v) {this.attrs[k]=v;if(k==='hidden')this.hidden=true;}
  getAttribute(k){return this.attrs[k];}
  append(...nodes){for(const n of nodes){this.children.push(n);n.parent=this;}}
  prepend(...nodes){for(const n of nodes.reverse()){this.children.unshift(n);n.parent=this;}}
  getBoundingClientRect(){return {bottom:48};}
  addEventListener(k,fn){(this.handlers[k] ||= []).push(fn);}
  removeEventListener(k,fn){this.handlers[k]=(this.handlers[k]||[]).filter(f=>f!==fn);}
  fire(k,event={}){for(const fn of [...this.handlers[k]||[]])fn(event);}
  click(){this.fire('click');}
  focus(){focused=this;}
  remove(){this.parent.children=this.parent.children.filter(n=>n!==this);}
  attachShadow(){return this.shadow=new Node('shadow');}
}
let focused=null,nextFrame=0;const frames=new Map();
const document=new Node('document');document.body=new Node('body');document.createElement=t=>new Node(t);
document.querySelectorAll=()=>[{src:'test',getAttribute:()=> 'allow'}];
const window=new Node('window');window.dispatchEvent=e=>window.fire(e.type,e);
const context={document,window,console,Event:class{constructor(type){this.type=type;}},requestAnimationFrame:fn=>{frames.set(++nextFrame,fn);return nextFrame;},cancelAnimationFrame:id=>frames.delete(id),fetch:async()=>({ok:false,status:503}),Set,Map,Number,TypeError,Error};
vm.createContext(context);
const source=fs.readFileSync(new URL('../examples/meowdoku/web/site-nav.js',import.meta.url),'utf8').replace('export function registerControlSuite','function registerControlSuite').replace('import.meta.url','"test"');
vm.runInContext(source,context);
const register=window.MeowdokuSiteNav.registerControlSuite;
assert.throws(()=>register({identity:'bad',policy:{bindings:[{action:'jump'}]}}),/Invalid/);
let value=1,opens=0,closes=0,runs=0;
const spec={identity:'real-policy',label:'Game controls',policy:{bindings:[{action:'jump',inputs:['keyboard:Space','gamepad:button-0']}]},sample:()=>value,onOpen:()=>opens++,onClose:()=>closes++,actions:[{label:'Existing action',run:()=>runs++}]};
const dispose=register(spec);assert.throws(()=>register(spec),/already registered/);
const host=document.body.children[0],nav=host.shadow.children[1],wrap=nav.children[0],toggle=nav.children[1];
assert.equal(host.style.height,'56px');
const chip=wrap.children.find(n=>n.textContent==='Game controls'),panel=wrap.children.find(n=>n.tag==='section');
assert.ok(panel.hidden);chip.click();assert.equal(opens,1);assert.equal(panel.hidden,false);assert.equal(chip.attrs['aria-expanded'],'true');
const row=panel.children.find(n=>n.className==='binding');assert.ok(row.classes.has('active'));
panel.children.find(n=>n.textContent==='Existing action').click();assert.equal(runs,1);
panel.fire('keydown',{key:'Escape',stopPropagation(){}});assert.equal(closes,1);assert.ok(panel.hidden);assert.equal(focused,chip);assert.equal(frames.size,0);
chip.click();document.fire('pointerdown',{composedPath:()=>[document]});assert.ok(panel.hidden);assert.equal(closes,2);
chip.click();chip.click();assert.ok(panel.hidden);assert.equal(frames.size,0);
chip.click();dispose();dispose();assert.equal(frames.size,0);assert.ok(!wrap.children.includes(chip));assert.equal((document.handlers.pointerdown||[]).length,0);
const again=register(spec);
const againChip=wrap.children.find(n=>n.textContent==='Game controls');
againChip.click();assert.ok(frames.size);
toggle.click();assert.ok(wrap.hidden);assert.equal(frames.size,0);assert.equal(toggle.textContent,'Show site menu');assert.equal(toggle.attrs['aria-expanded'],'false');assert.equal(focused,toggle);
toggle.click();assert.ok(!wrap.hidden);assert.equal(toggle.attrs['aria-expanded'],'true');
again();
await Promise.resolve();await Promise.resolve();
console.log('PASS: registration, validation, existing callbacks, live telemetry, Escape, outside click, repeated toggles, unregister, API failure independence');

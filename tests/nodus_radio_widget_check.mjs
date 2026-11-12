// Offline lifecycle test: real header module, minimal DOM boundary, no network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
class Node {
  constructor(tag='node') { this.tag=tag; this.children=[]; this.attrs={}; this.handlers={}; this.hidden=false; this.classes=new Set(); this.classList={toggle:(k,on)=>on?this.classes.add(k):this.classes.delete(k)}; this.style={setProperty(k,v){this[k]=v;}}; }
  setAttribute(k,v) {this.attrs[k]=v;if(k==='hidden')this.hidden=true;}
  getAttribute(k){return this.attrs[k];}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
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
let polls=0, timerId=0;const timers=new Map();
Object.assign(context,{URLSearchParams, location:{search:'',pathname:'/'}, setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id)});
context.fetch=async path=>({ok:true,json:async()=>path==='/api/whoami'?{signed_in:true,username:'dot',capabilities:['view_site']}:(polls++,{status:'live',title:'Tempest',rms:.25,bands:[.1,.2,.3,.4,.5]})});
vm.createContext(context);
const source=fs.readFileSync(new URL('../examples/meowdoku/web/site-nav.js',import.meta.url),'utf8').replace('export function registerControlSuite','function registerControlSuite').replace('import.meta.url','"test"');
vm.runInContext(source,context);
for(let i=0;i<12;i++)await Promise.resolve();
const host=document.body.children[0],nav=host.shadow.children[1],actions=nav.children[0],toggle=nav.children[1];
const chip=actions.children.find(n=>n.textContent==='Nodus Radio');assert.ok(chip);
const panel=actions.children.find(n=>n.attrs['aria-label']==='Nodus Radio');assert.ok(panel.hidden);assert.equal(polls,0);
chip.click();for(let i=0;i<12;i++)await Promise.resolve();
assert.equal(polls,1);assert.equal(panel.hidden,false);assert.equal(timers.size,1);
assert.equal(panel.children.find(n=>n.className==='radio-bands').children.length,5);
assert.equal(panel.children.find(n=>n.tag==='strong').textContent,'Tempest');
toggle.click();assert.ok(panel.hidden);assert.equal(timers.size,0);
toggle.click();chip.click();chip.click();for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);assert.ok(panel.hidden);
chip.click();for(let i=0;i<12;i++)await Promise.resolve();panel.fire('keydown',{key:'Escape',stopPropagation(){}});assert.equal(timers.size,0);assert.equal(focused,chip);
chip.click();document.fire('pointerdown',{composedPath:()=>[document]});for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);
chip.click();window.fire('pagehide');for(let i=0;i<12;i++)await Promise.resolve();assert.equal(timers.size,0);
console.log('PASS: Radio lazy polling, real packet display, collapse, rapid toggles, Escape, outside close, unload, stale-response protection');

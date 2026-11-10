// Actual Wasm and actual lens-controls code, with a minimal DOM test adapter.
// This is not a claim that a real browser was exercised.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this.style={};this.dataset={};this.handlers=new Map();this.value='';this.textContent='';this.disabled=false;this.checked=false;}
 append(...items){this.children.push(...items);items.forEach(x=>x.parent=this);}
 replaceChildren(...items){this.children=[];this.append(...items);}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(x=>x!==this);}
 setAttribute(k,v){this[k]=v;}
 addEventListener(k,fn){const a=this.handlers.get(k)||[];a.push(fn);this.handlers.set(k,a);}
 removeEventListener(k,fn){this.handlers.set(k,(this.handlers.get(k)||[]).filter(x=>x!==fn));}
 dispatchEvent(event){for(const fn of this.handlers.get(event.type)||[])fn(event);}
 click(){if(!this.disabled)this.dispatchEvent({type:'click'});}
 get options(){return this.children;}
}
const find=(el,id)=>el.id===id?el:el.children.map(e=>find(e,id)).find(Boolean);
const doc=new Element('document');doc.body=new Element('body');doc.createElement=tag=>new Element(tag);doc.getElementById=id=>find(doc.body,id);doc.defaultView={Event};
for(const id of ['result','photo-lens-controls','photo-lens','photo-focus-mm','photo-focal-mm','photo-f-number','photo-focus-slider','photo-focal-slider','photo-aperture-slider','photo-lens-status','photo-lens-inventory','photo-focus-diagnostic']){
 const e=new Element(id==='photo-lens'?'select':'input');e.id=id;doc.body.append(e);
}
doc.getElementById('photo-lens').value='unchanged';doc.getElementById('photo-focus-mm').value='110';doc.getElementById('photo-focal-mm').value='50';doc.getElementById('photo-f-number').value='8';
const oldDocument=globalThis.document,oldFetch=globalThis.fetch;globalThis.document=doc;
globalThis.fetch=async url=>{try{return new Response(await readFile(url));}catch{return new Response('',{status:404});}};
try {
 await import('./meowdoku_rangefinder_browser.mjs');
 const result=JSON.parse(doc.getElementById('result').textContent);
 assert.equal(result.passed,true,result.error);console.log(JSON.stringify({...result,dom:'test adapter; not browser QA'}));
} finally {globalThis.document=oldDocument;globalThis.fetch=oldFetch;}

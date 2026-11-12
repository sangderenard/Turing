import assert from 'node:assert/strict';
import {defaultFilmResponse, normalizeFilmResponse, filmResponseFwhmNm, createFilmResponseControls, FILM_RESPONSE_STORAGE_KEY} from '../examples/meowdoku/web/marble_film_response.js';
import {packMarblePhoto,createMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';
const defaults=defaultFilmResponse();
assert.deepEqual(normalizeFilmResponse(),defaults);
const copy=v=>JSON.parse(JSON.stringify(v));
for(const value of [null,[],{}, {...defaults,version:true},{...defaults,version:2},{...defaults,extra:0},
  {...defaults,centers_nm:[650,550]}, {...defaults,centers_nm:[379,550,450]}, {...defaults,centers_nm:[650,550,781]},
  {...defaults,centers_nm:['650',550,450]}, {...defaults,sigmas_oct:[true,.06,.06]}, {...defaults,sigmas_oct:[NaN,.06,.06]},
  {...defaults,sigmas_oct:[Infinity,.06,.06]}, {...defaults,sigmas_oct:[.009,.06,.06]}, {...defaults,sigmas_oct:[1.01,.06,.06]}])
  assert.throws(()=>normalizeFilmResponse(value));
assert.deepEqual(normalizeFilmResponse({version:1,centers_nm:[380,780,380],sigmas_oct:[.01,1,.01]}),{version:1,centers_nm:[380,780,380],sigmas_oct:[.01,1,.01]});
assert.ok(filmResponseFwhmNm(650,.06)>63 && filmResponseFwhmNm(650,.06)<64);
const readConfig=options=>{const raw=packMarblePhoto({interior:[],glass:[]},'a'.repeat(64),options),view=new DataView(raw);return JSON.parse(new TextDecoder().decode(new Uint8Array(raw,16,view.getUint32(4,true))));};
assert.deepEqual(readConfig({}).film_response,defaults);
assert.equal(readConfig({}).exposure_seconds,30);
assert.equal(readConfig({seconds:600}).exposure_seconds,600);
const custom={version:1,centers_nm:[620,540,465],sigmas_oct:[.1,.2,.03]};
assert.deepEqual(readConfig({filmResponse:custom}).film_response,custom);
assert.equal(readConfig({transport:'fixed:1'}).film_response,undefined);
assert.throws(()=>readConfig({transport:'fixed:1',filmResponse:custom}));

class Element {
  constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.classList={toggle(){}};}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=nodes;}
  setAttribute(k,v){this[k]=v;}
  addEventListener(k,v){(this.events[k]??=[]).push(v);}
  fire(k){for(const f of this.events[k]??[])f();}
}
const doc={createElement:tag=>new Element(tag,doc)},all=node=>[node,...node.children.flatMap(all)];
const data=new Map(),storage={getItem:k=>data.get(k)??null,setItem:(k,v)=>data.set(k,v)};
const transport=new Element('select',doc);transport.value='fixed:3';
const container=new Element('details',doc),editor=createFilmResponseControls(container,transport,{storage});
const nodes=all(container),input=id=>nodes.find(n=>n.id===id);
input('film-red-centers_nm').value='620';input('film-red-centers_nm').fire('input');
input('film-green-sigmas_oct').value='.2';input('film-green-sigmas_oct').fire('input');
assert.equal(editor.getState().centers_nm[0],620);assert.equal(JSON.parse(data.get(FILM_RESPONSE_STORAGE_KEY)).sigmas_oct[1],.2);
const prior=data.get(FILM_RESPONSE_STORAGE_KEY);input('film-red-centers_nm').value='';input('film-red-centers_nm').fire('input');
assert.throws(()=>editor.getState());assert.equal(data.get(FILM_RESPONSE_STORAGE_KEY),prior);
transport.value='fixed:1';transport.fire('change');assert.equal(editor.forTransport(),undefined);assert.ok(nodes.filter(n=>n.tagName==='fieldset').every(n=>n.disabled));
transport.value='fixed:3';transport.fire('change');assert.throws(()=>editor.forTransport());
const restored=createFilmResponseControls(new Element('details',doc),transport,{storage});assert.equal(restored.getState().centers_nm[0],620);
nodes.find(n=>n.textContent==='Restore film defaults').fire('click');assert.deepEqual(editor.getState(),defaults);
const blocked=createFilmResponseControls(new Element('details',doc),transport,{storage:{getItem(){throw Error('blocked');},setItem(){throw Error('full');}}});assert.deepEqual(blocked.getState(),defaults);
console.log('custom film response: strict validation, wire format, local persistence, monochrome and invalid-input protection passed');

// Exercise actual start(), with every network call replaced by a local test double.
// Invalid sensitivity must reject BEFORE canceling the currently running exposure.
const fields={};
const photoDoc={defaultView:{localStorage:storage,location:{search:''}},createElement:tag=>new Element(tag,photoDoc),
  getElementById(id){return id==='photo-lens-controls'?null:(fields[id]??=new Element('div',photoDoc));}};
for(const [id,value] of Object.entries({'photo-transport':'fixed:3','photo-camera':'builtin','photo-resolution':'128',
  'photo-seconds':'0.1','photo-substeps':'1800','photo-iso':'100','photo-highlight-clip':'25','photo-orbit-frames':'1','photo-gamma':'100','photo-bounces':'4','photo-bounce-mode':'auto','photo-triangle-budget':'250000','photo-sphere-coverage':'exact'}))photoDoc.getElementById(id).value=value;
const calls=[],realFetch=globalThis.fetch;
globalThis.fetch=async(url,options)=>{
  calls.push({url,options});
  if(url.includes('?digest='))return new Promise(()=>{});
  return {ok:true,json:async()=>({digest:'b'.repeat(64)})};
};
try {
  const photo=createMarblePhoto(photoDoc);photo.saved('a'.repeat(64));
  const geometry={interior:[],glass:[],triangleCount:0};
  await photo.start(geometry);
  const before=calls.length,filmNodes=all(fields['photo-film-response']);
  const red=filmNodes.find(n=>n.id==='film-red-centers_nm');red.value='900';red.fire('input');
  assert.equal(calls.length,before,'editing response must not contact or cancel active job');
  await assert.rejects(photo.start(geometry),/centers_nm/);
  assert.equal(calls.length,before,'invalid response must not cancel or upload');
  red.value='620';red.fire('input');
  await assert.rejects(photo.start(geometry),/already running/);
  assert.equal(calls.length,before,'valid edits still cannot replace an active exposure');
  await fields['photo-cancel'].events.click[0]();
  await photo.start(geometry);
  assert.equal(calls.filter(c=>c.url.endsWith('/cancel')).length,1);
  const body=calls.filter(c=>c.url==='/api/marble_photo').at(-1).options.body,view=new DataView(body);
  const config=JSON.parse(new TextDecoder().decode(new Uint8Array(body,16,view.getUint32(4,true))));
  assert.equal(config.film_response.centers_nm[0],620);
  assert.equal(photo.filmResponse().centers_nm[0],620,'calibration must capture current response too');
  photo.stale();
} finally {globalThis.fetch=realFetch;}
console.log('photo start: edited response is captured; invalid inputs preserve active exposure');

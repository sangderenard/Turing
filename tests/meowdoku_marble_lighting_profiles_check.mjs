import assert from 'node:assert/strict';
import {STORAGE_KEY, QUICK_STORAGE_KEY, createLightingProfiles, readLightingProfiles, saveLightingProfile, loadLightingProfile} from '../examples/meowdoku/web/marble_lighting_profiles.js';

const memory = () => {
  const data = new Map();
  return {getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value)};
};
const validate = state => {
  assert.ok(state && Array.isArray(state.poses) && state.poses.length === 3, 'Invalid light poses');
  assert.ok(state.poses.every(Number.isFinite) && Number.isFinite(state.backdrop), 'Nonfinite lighting');
  return {poses: [...state.poses], backdrop: state.backdrop};
};
const state = {poses: [1, 2, 3], backdrop: 0.4};
const storage = memory();
assert.deepEqual(readLightingProfiles(storage, validate), []);
saveLightingProfile(storage, 'Studio', state, validate);
state.poses[0] = 9;
assert.deepEqual(loadLightingProfile(storage, 'Studio', validate), {poses: [1, 2, 3], backdrop: 0.4});
const reloaded = memory();
reloaded.setItem(STORAGE_KEY, storage.getItem(STORAGE_KEY));
assert.deepEqual(readLightingProfiles(reloaded, validate), readLightingProfiles(storage, validate));
saveLightingProfile(storage, 'Studio', state, validate);
assert.equal(readLightingProfiles(storage, validate).length, 1);
assert.equal(loadLightingProfile(storage, 'Studio', validate).poses[0], 9);
saveLightingProfile(storage, '__proto__', state, validate);
saveLightingProfile(storage, '<img onerror=alert(1)>', state, validate);
assert.equal(readLightingProfiles(storage, validate).length, 3);
for (const bad of ['{', 'null', '[]', '{"version":2,"profiles":[]}', '{"version":1,"profiles":{}}', JSON.stringify({version:1, profiles:[{name:'x',state},{name:'x',state}]}), JSON.stringify({version:1,profiles:[{name:' x ',state}]}), JSON.stringify({version:1,profiles:[{name:'x',state:{}}]})]) {
  const corrupt = memory(); corrupt.setItem(STORAGE_KEY, bad);
  assert.throws(() => readLightingProfiles(corrupt, validate));
  assert.throws(() => saveLightingProfile(corrupt, 'Studio', state, validate));
  assert.equal(corrupt.getItem(STORAGE_KEY), bad);
}
for (const name of ['', '   ', 'a'.repeat(81), null, 42]) assert.throws(() => saveLightingProfile(storage, name, state, validate));
assert.throws(() => saveLightingProfile(storage, 'bad', {poses: [NaN, 1, 2], backdrop: 1}, validate));
assert.throws(() => saveLightingProfile(storage, 'bad', {poses: [1, 1, 2], backdrop: Infinity}, validate));
assert.throws(() => readLightingProfiles(null, validate));
assert.throws(() => saveLightingProfile(null, 'Studio', state, validate));
assert.throws(() => readLightingProfiles({getItem(){throw Error('Denied');},setItem(){}}, validate));
const before = storage.getItem(STORAGE_KEY);
assert.throws(() => saveLightingProfile({getItem: storage.getItem, setItem(){throw Error('Quota');}}, 'New', state, validate));
assert.equal(storage.getItem(STORAGE_KEY), before);
const full = memory();
for(let i = 0; i < 30; i++) saveLightingProfile(full, `Profile ${i}`, state, validate);
assert.throws(() => saveLightingProfile(full, 'Overflow', state, validate));
saveLightingProfile(full, 'Profile 0', state, validate);
assert.equal(readLightingProfiles(full, validate).length, 30);
full.setItem(STORAGE_KEY, JSON.stringify({version:1,profiles:Array.from({length:31}, (_,i) => ({name:`Profile ${i}`,state}))}));
assert.throws(() => readLightingProfiles(full, validate));
console.log('marble lighting profiles: round-trip, update, corrupt preservation, limits and storage failures passed');

class Element {
  constructor(tag,doc) {this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';}
  append(...nodes) {this.children.push(...nodes);}
  replaceChildren(...nodes) {this.children=nodes;}
  setAttribute(k,v) {this[k]=v;}
  addEventListener(k,f) {(this.events[k]??=[]).push(f);}
  click() {for(const fn of this.events.click??[])fn();}
}
const doc={createElement:tag=>new Element(tag,doc)};
const walk=node=>[node,...node.children.flatMap(walk)];
const slots=memory();let current={poses:[1,2,3],backdrop:4},applied=null,changes=0;
const mount=()=>{const root=doc.createElement('div');createLightingProfiles(root,{storage:slots,capture:()=>current,validate,apply:s=>{applied=s;},onChange:()=>changes++});return root;};
let root=mount();const find=id=>walk(root).find(el=>el.id===id);
assert.equal(slots.getItem(QUICK_STORAGE_KEY),null);
for(const slot of ['A','B','C','D']) {assert.equal(find('lighting-quick-'+slot).disabled,true);find('lighting-store-'+slot).click();}
assert.equal(slots.getItem(STORAGE_KEY),null,'Slots cannot overwrite named profiles');
assert.equal(changes,0,'Storing cannot invalidate an exposure');
root=mount();assert.equal(find('lighting-quick-A').disabled,false);
current.poses[0]=9;find('lighting-quick-A').click();assert.equal(applied.poses[0],1);
assert.equal(changes,1);
const beforeSlot=slots.getItem(QUICK_STORAGE_KEY);current={poses:[NaN,2,3],backdrop:4};
find('lighting-store-A').click();assert.equal(slots.getItem(QUICK_STORAGE_KEY),beforeSlot);
slots.setItem(QUICK_STORAGE_KEY,'{');root=mount();
assert.equal(find('lighting-quick-A').disabled,true);find('lighting-store-A').click();
assert.equal(slots.getItem(QUICK_STORAGE_KEY),'{','Corrupt slots are preserved');
console.log('Quick slots: empty, persistence, capture isolation, separate storage, invalid input and corrupt preservation passed');

const {runCalibrationSet}=await import('../examples/meowdoku/web/marble_calibration.js');
const events=[];const calDoc={getElementById:id=>id==='lighting-profiles'?{dispatchEvent:e=>events.push(e.type)}:null};
await assert.rejects(runCalibrationSet(calDoc,{}, {subject:{shell:'missing'}}),/missing control/);
assert.deepEqual(events,['room-calibration-start','room-calibration-end']);
console.log('Failed calibration restores Room channel in finally');

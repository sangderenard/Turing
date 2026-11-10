import {installScenePackerInputs} from './meowdoku_scene_packer_harness.mjs';
import {applyLightTools,defaultLightTools,migrateLightTools} from '../examples/meowdoku/web/marble_light_tools.js';
import {applyFixtureSizes,defaultFixtureSizes} from "../examples/meowdoku/web/marble_fixture_sizes.js";
// Exercise the production fixture builder, emissive-material writer and photo
// wire payload. No renderer or substitute light calculation is involved.
import assert from 'node:assert/strict';
import {writeStudioFixtureRows} from '../examples/meowdoku/web/marble_studio_scene.js';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {STUDIO_LIGHTS, createLightStrengthControls, validateLightStrengths} from '../examples/meowdoku/web/marble_light_strengths.js';
import {tableLightRoles, tableLightModifierFaces} from '../examples/meowdoku/web/marble_table_lights.js';
import {packMarblePhoto, OBJECT_CENTER} from '../examples/meowdoku/web/marble_photo.js';

const html = fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.html', import.meta.url), 'utf8');
const source = fs.readFileSync(process.env.STUDIO_SOURCE ?? new URL('../examples/meowdoku/web/marble_studio.js', import.meta.url), 'utf8');
const copy = value => JSON.parse(JSON.stringify(value));
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
class Element {
  constructor(tag, doc) { this.tagName = tag; this.ownerDocument = doc; this.children = []; this.events = {}; this.value = ''; this.dataset = {}; }
  appendChild(node) { this.children.push(node); return node; }
  append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
  setAttribute(key, value) { this[key] = value; }
  addEventListener(type, fn) { (this.events[type] ??= []).push(fn); }
  click() { this.fire('click'); }
  fire(type) { for (const fn of this.events[type] ?? []) fn(); }
}
const nodes = {};
const walk = node => [node, ...node.children.flatMap(walk)];
const doc = {createElement: tag => new Element(tag, doc), getElementById: id => nodes[id] ?? walk(root).find(node => node.id === id)};
const root = doc.createElement('div');
for (const match of html.matchAll(/<input\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const el = doc.createElement('input');
  for (const a of match[1].matchAll(/([\w-]+)="([^"]*)"/g)) el[a[1]] = a[2];
  nodes[el.id] = el;
}
for (const match of html.matchAll(/<select\b[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  const el = doc.createElement('select'); el.id = match[1];
  for (const option of match[2].matchAll(/<option([^>]*)>(.*?)<\/option>/g))
    if (!el.value || option[1].includes('selected')) el.value = /value="([^"]*)"/.exec(option[1])?.[1] ?? option[2];
  nodes[el.id] = el;
}
const defaults = STUDIO_LIGHTS.map(({strengthId}) => Number(nodes[strengthId].value) / 10);
let invalidations = 0;
const strengths = createLightStrengthControls(root, nodes, () => invalidations++);
assert.equal(strengths.element.open, undefined); // details remains collapsed
assert.deepEqual(strengths.getStrengths('studio'), defaults);
assert.deepEqual(strengths.getStrengths('exposure'), defaults);
assert.equal(invalidations, 0);
strengths.element.open = true; strengths.element.fire('toggle'); strengths.element.open = false; strengths.element.fire('toggle');
assert.equal(invalidations, 0);
assert.ok(strengths.getState().lights.every(light => light.exposure === null));

function fixtureContext(text) {
  const context = vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),$, writeStudioFixtureRows, STUDIO_LIGHTS, OBJECT_CENTER, tableLightRoles, tableLightModifierFaces, studioLighting: null,
    studioTableState:()=>({enabled:false}),tableFaces:()=>[],goboFaces:()=>[],studioSubjectTransform:()=>({subjectCenter:[0,0,0]}),studioPhotoScale:()=>.11/3.45,
    FIXTURE_MAT_BASE: 0, tables: {pbr: {table: {floats_per_record: 16}, data: new Float32Array(STUDIO_LIGHTS.length * 16), update() {}}},
    rgb: value => [1, 3, 5].map(k => parseInt(value.slice(k, k + 2), 16) / 255),
    checkerboardFaces: () => [], backdropRollFaces: () => [],
  });
  function $(id) { return nodes[id]; }
  vm.runInContext(text.slice(text.indexOf('const ROOM_FIXTURES ='), text.indexOf('const currentCameraOptions =')), context);
  vm.runInContext(text.slice(text.indexOf('function quadFace('), text.indexOf('function digitFaces(')), context);
  vm.runInContext(text.slice(text.indexOf('function studioRenderState('), text.indexOf('function surfaceMaterial(')), context);
  vm.runInContext(text.slice(text.indexOf('function computeFixtureRoles('), text.indexOf('\nfunction syncMaterial()')).replace('export function', 'function'), context);
  installScenePackerInputs(context);
  return context;
}
const context = fixtureContext(source);
const run = code => copy(vm.runInContext(code, context, {timeout: 1000}));
const poses = run('computeFixtureRoles().fixtureRoles').map(({pos, dir}) => ({pos, dir}));
context.studioLighting = {strengths, poses: {getPoses: () => poses}};
const scene = () => run('sceneForPhoto()');
const roles = () => run('computeFixtureRoles()');
function preview(text = source, target = context) {
  const frame = text.indexOf('function frame(');
  const start = text.indexOf('const pbrStride=tables.pbr.table.floats_per_record;', frame);
  const end = text.indexOf('tables.pbr.update();', start) + 'tables.pbr.update();'.length;
  const values = vm.runInContext('computeFixtureRoles()', target);
  target.roleColor = values.roleColor; target.roleStrength = values.roleStrength; target.fixtureRoles = values.fixtureRoles;
  vm.runInContext(`{${text.slice(start, end)}}`, target, {timeout: 1000});
  const lightStart = text.indexOf('const coneAngles=', frame), lightEnd = text.indexOf('gl.uniform1i(loc(P,"uNumLights")', lightStart);
  target.view = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; target.transformPoint = (_, p) => p;
  const intensity = vm.runInContext(`(() => { ${text.slice(lightStart, lightEnd)} return lightIntensity; })()`, target, {timeout: 1000});
  return {pbr: Array.from(target.tables.pbr.data), intensity: Array.from(intensity)};
}
const initialScene = scene(), initialPreview = preview();
// Default fixture state is now the requested ten-light white-filter rig.
assert.equal(roles().fixtureRoles.length,10);
assert.deepEqual(roles().roleColor,Array.from({length:10},()=>[1,1,1]));
assert.deepEqual(defaults.slice(7),[6,6,1.5]);
assert.equal(initialScene.pbr.length,11*16);

for (const [index, light] of STUDIO_LIGHTS.entries()) {
  const link = doc.getElementById(`${light.id}-levels-linked`);
  const studio = doc.getElementById(`${light.id}-studio-strength`);
  const exposure = doc.getElementById(`${light.id}-exposure-strength`);
  assert.ok(strengths.ownsControl(light.strengthId));
  assert.ok(strengths.ownsControl(link.id) && strengths.ownsControl(studio.id) && strengths.ownsControl(exposure.id));
  const originalStudio = roles(), originalScene = scene();
  let before = invalidations;
  link.checked = false; link.fire('change');
  assert.equal(invalidations, before); assert.equal(exposure.disabled, false);
  assert.deepEqual(scene(), originalScene);
  // Every original slider now changes only the Studio route, including the
  // real preview direct-light uniform values and the fixture's PBR emission.
  nodes[light.strengthId].value = '175'; nodes[light.strengthId].fire('input');
  assert.equal(invalidations, before); assert.equal(roles().roleStrength[index], 17.5);
  assert.deepEqual(scene(), originalScene);
  let actualPreview = preview();
  assert.equal(actualPreview.pbr[index * 16 + 8], Math.fround(originalStudio.roleColor[index][0] * 17.5));
  assert.ok(actualPreview.intensity[index] > 0);
  // Optional numeric Studio editing owns the same original slider, not a
  // disconnected duplicate value. Zero must black out this lane only.
  studio.value = '0'; studio.fire('input');
  assert.equal(nodes[light.strengthId].value, '0'); assert.equal(invalidations, before);
  assert.deepEqual(scene(), originalScene);
  actualPreview = preview();
  assert.deepEqual(actualPreview.pbr.slice(index * 16 + 8, index * 16 + 11), [0, 0, 0]);
  assert.equal(actualPreview.pbr[index * 16], Math.fround(roles().roleColor[index][0])); assert.equal(actualPreview.intensity[index], 0);
  const previewBeforeExposure = preview();
  exposure.value = '9.3'; exposure.fire('input');
  assert.equal(invalidations, ++before); assert.deepEqual(preview(), previewBeforeExposure);
  const changedScene = scene();
  assert.deepEqual(changedScene.vertices, originalScene.vertices);
  assert.deepEqual(changedScene.material_ids, originalScene.material_ids);
  for (let row = 0; row < STUDIO_LIGHTS.length+1; row++) {
    if (row !== index) assert.deepEqual(changedScene.pbr.slice(row * 16, (row + 1) * 16), originalScene.pbr.slice(row * 16, (row + 1) * 16));
  }
  assert.deepEqual(changedScene.pbr.slice(index * 16 + 8, index * 16 + 11), originalStudio.roleColor[index].map(c => Math.fround(c * 9.3)));
  const wire = packMarblePhoto({interior: [], glass: [[[0, 0, 0], [1, 0, 0], [0, 1, 0], 0, [[0, 0, 1], [0, 0, 1], [0, 0, 1]]]]}, 'a'.repeat(64), {fixtures: changedScene});
  const view = new DataView(wire), header = JSON.parse(new TextDecoder().decode(new Uint8Array(wire, 16, view.getUint32(4, true))));
  assert.deepEqual(header.fixtures, changedScene);
  // Off exposure does not turn off a separately bright Studio fixture.
  studio.value = '12'; studio.fire('input'); assert.equal(invalidations, before);
  exposure.value = '0'; exposure.fire('input'); assert.equal(invalidations, ++before);
  assert.equal(roles().roleStrength[index], 12);
  assert.deepEqual(scene().pbr.slice(index * 16 + 8, index * 16 + 11), [0, 0, 0]);
  assert.equal(scene().pbr[index * 16], Math.fround(roles().roleColor[index][0]));
  const stable = scene();
  for (const value of ['', 'NaN', '-1', String(Number(exposure.max) + 1)]) {
    exposure.value = value; exposure.fire('input'); assert.equal(invalidations, before); assert.deepEqual(scene(), stable);
    exposure.fire('change'); assert.equal(exposure.value, '0');
  }
  link.checked = true; link.fire('change'); assert.equal(invalidations, ++before);
  assert.equal(strengths.getStrengths('exposure')[index], 12); assert.equal(exposure.disabled, true);
  nodes[light.strengthId].value = String(defaults[index] * 10); nodes[light.strengthId].fire('input');
  assert.equal(invalidations, ++before); assert.equal(strengths.getStrengths('exposure')[index], defaults[index]);
}
assert.deepEqual(scene(), initialScene); assert.deepEqual(preview(), initialPreview);
// Solo masks both routes but never rewrites linked/unlinked authored levels.
const fullState = strengths.getState();
for (const [index, light] of STUDIO_LIGHTS.entries()) {
  let before = invalidations;
  doc.getElementById(`${light.id}-solo`).click();
  assert.equal(strengths.getSolo(), light.id);
  assert.equal(invalidations, before + 1);
  assert.deepEqual(strengths.getState(), fullState);
  assert.deepEqual(strengths.getStrengths('studio'), defaults.map((v,i) => i === index ? v : 0));
  assert.deepEqual(strengths.getStrengths('exposure'), defaults.map((v,i) => i === index ? v : 0));
  const soloScene = scene(), soloPreview = preview();
  for (let i=0; i<STUDIO_LIGHTS.length; i++) if (i !== index) {
    assert.deepEqual(soloScene.pbr.slice(i*16+8,i*16+11), [0,0,0]);
    assert.deepEqual(soloPreview.pbr.slice(i*16+8,i*16+11), [0,0,0]);
    assert.equal(soloPreview.intensity[i], 0);
  }
  doc.getElementById(`${light.id}-solo`).click();
  assert.equal(strengths.getSolo(), null); assert.deepEqual(scene(), initialScene);
}
doc.getElementById('key-left-solo').click();
doc.getElementById('key-right-solo').click();
assert.equal(strengths.getSolo(), 'key-right');
doc.getElementById('light-solo-clear').click();
assert.equal(strengths.getSolo(), null);
const clearCount = invalidations; doc.getElementById('light-solo-clear').click();
assert.equal(invalidations, clearCount);
doc.getElementById('key-left-solo').click();
strengths.setState(fullState); assert.equal(strengths.getSolo(), null);
assert.deepEqual(scene(), initialScene);
const valid = strengths.getState();
for (const mutate of [s => s.version = 2, s => s.lights.pop(), s => s.lights[0].id = 'wrong', s => s.lights[0].exposure = undefined,
  s => s.lights[0].exposure = NaN, s => s.lights[0].exposure = Infinity, s => s.lights[0].exposure = -1, s => s.lights[0].exposure = 401]) {
  const bad = copy(valid); mutate(bad); assert.throws(() => validateLightStrengths(bad, nodes));
  assert.throws(() => strengths.setState(bad)); assert.deepEqual(strengths.getState(), valid);
}
assert.deepEqual(validateLightStrengths(undefined, nodes), valid);
// Global invalidation delegates exactly these owned controls to the component.
const invalidationBody = source.slice(source.indexOf('const designChanged=(event)=>{'), source.indexOf('\n  };', source.indexOf('const designChanged=(event)=>{')) + 5);
let photoStale = 0, rayChanged = 0;
const eventContext = vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),applyingObject:false,objectEdited:()=>{},studioLighting: {strengths}, skipPhotoInvalidation: new Set(), photo: {stale: () => photoStale++,sceneChanged:()=>photoStale++},
  liveRay: {sceneChanged: () => rayChanged++, stale: () => rayChanged++}});
vm.runInContext(invalidationBody + '\nglobalThis.changed=designChanged;', eventContext);
for (const light of STUDIO_LIGHTS) for (const id of [light.strengthId, `${light.id}-studio-strength`, `${light.id}-exposure-strength`, `${light.id}-levels-linked`])
  eventContext.changed({target: {id, closest: selector => selector.includes('#studio-scene-controls')}});
assert.equal(photoStale, 0); assert.equal(rayChanged, 0);
// Real single-light calibration cases turn off every added emitter and remove
// exposure overrides, including when launched after a split profile.
const calibration=fs.readFileSync(new URL('../examples/meowdoku/web/marble_calibration.js',import.meta.url),'utf8');
const calContext=vm.createContext({applyLightTools,defaultLightTools:migrateLightTools,wallRelativeFixtureDefaults:roles=>roles,applyFixtureSizes,defaultFixtureSizes:()=>defaultFixtureSizes(true),Event:class {constructor(type){this.type=type;}}});
vm.runInContext(calibration.slice(calibration.indexOf('const ROLE_IDS ='),calibration.indexOf('const settle =')),calContext);
for (const light of STUDIO_LIGHTS) {
  const link=doc.getElementById(`${light.id}-levels-linked`);link.checked=false;link.fire('change');
  const exposure=doc.getElementById(`${light.id}-exposure-strength`);exposure.value='2';exposure.fire('input');
  nodes[light.strengthId].value='30';
}
const calDoc={getElementById(id){const el=doc.getElementById(id);if(el)el.dispatchEvent=e=>el.fire(e.type);return el;}};
calContext.doc=calDoc;
doc.getElementById('key-right-solo').click();
vm.runInContext("applyCase(doc,{}, {role:'key_left',strength:50,filter:'#ffffff'})",calContext);
assert.deepEqual(strengths.getStrengths('studio'),[5,0,0,0,0,0,0,0,0,0]);
assert.deepEqual(strengths.getStrengths('exposure'),[5,0,0,0,0,0,0,0,0,0]);
console.log(`Ten-light strengths: exact defaults, isolated geometry/PBR/uniform lanes, zero, linked toggles, invalid inputs and actual photo wire payload passed (default scene SHA256 ${hash(initialScene)}).`);

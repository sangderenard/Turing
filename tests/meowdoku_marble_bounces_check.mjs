// Production bounce mapping and controls, without a server or GPU.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bounceSliderPosition, bouncesFromSlider, createPhotoBounceControls, photoBounceCount,
  packMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

assert.equal(bouncesFromSlider(0), 1);
assert.equal(bouncesFromSlider(100), 200);
assert.equal(bounceSliderPosition(1), 0);
assert.equal(bounceSliderPosition(200), 100);
const seen = new Set(), widths = new Map();
let previous = 1;
for (let step = 0; step <= 10000; step++) {
  const count = bouncesFromSlider(step / 100);
  assert.ok(count >= previous && count <= previous + 1, 'monotonic range never skips a count');
  seen.add(count); widths.set(count, (widths.get(count) || 0) + 1); previous = count;
}
assert.equal(seen.size, 200, 'every integer is reachable using the declared 0.01 step');
assert.ok(widths.get(4) > widths.get(50) && widths.get(50) > widths.get(190), 'low counts have wider pointer intervals');
for (let count = 1; count <= 200; count++) {
  assert.equal(bouncesFromSlider(Number(bounceSliderPosition(count).toFixed(2))), count,
    'inverse survives browser range-step rounding');
}
for (const value of [-1, 101, NaN, Infinity]) assert.throws(() => bouncesFromSlider(value), RangeError);
for (const value of [0, 201, 4.5, NaN, Infinity]) assert.throws(() => bounceSliderPosition(value), RangeError);

class Element extends EventTarget {
  constructor(id, value = '') { super(); this.id = id; this.value = value; this.textContent = ''; this.attributes = new Map(); }
  setAttribute(key, value) { this.attributes.set(key, value); }
  setCustomValidity(value) { this.validationMessage = value; }
  fire(type) { this.dispatchEvent(new Event(type, {bubbles: true})); }
  key(key) { const event = new Event('keydown', {cancelable: true}); Object.defineProperty(event, 'key', {value: key}); this.dispatchEvent(event); assert.equal(event.defaultPrevented, true); }
}
function controls(initial) {
  const input = new Element('photo-bounces', String(initial)), slider = new Element('photo-bounces-slider', '12.28');
  const output = new Element('photo-bounces-out');
  const elements = new Map([input, slider, output].map(element => [element.id, element]));
  const doc = {getElementById: id => elements.get(id), defaultView: {Event}};
  createPhotoBounceControls(doc);
  return {input, slider, output, doc};
}
for (const count of [1, 4, 7, 12, 13, 50, 200]) {
  const {input, slider, output, doc} = controls(count);
  assert.equal(input.value, String(count), 'initialization preserves actual legacy and current counts');
  assert.equal(photoBounceCount(doc), count);
  assert.equal(bouncesFromSlider(slider.value), count);
  assert.equal(output.textContent, String(count));
}
const {input, slider, output, doc} = controls(4);
let inputEvents = 0; slider.addEventListener('input', () => inputEvents++);
slider.key('ArrowUp'); assert.equal(photoBounceCount(doc), 5);
slider.key('PageUp'); assert.equal(photoBounceCount(doc), 15);
slider.key('Home'); assert.equal(photoBounceCount(doc), 1);
slider.key('End'); assert.equal(photoBounceCount(doc), 200);
slider.key('ArrowDown'); assert.equal(photoBounceCount(doc), 199);
assert.equal(inputEvents, 5, 'keyboard changes dispatch the input live rays observe');
slider.value = '100'; slider.fire('input');
assert.equal(input.value, '200'); assert.equal(output.textContent, '200');
assert.equal(slider.attributes.get('aria-valuetext'), '200 bounces');
input.value = '4'; input.fire('input');
assert.equal(bouncesFromSlider(slider.value), 4);
assert.equal(output.textContent, '4');
input.value = ''; input.fire('input'); assert.match(input.validationMessage, /whole number/);
assert.throws(() => photoBounceCount(doc), RangeError);
input.value = '12'; input.fire('change'); assert.equal(input.validationMessage, '');
assert.equal(bouncesFromSlider(slider.value), 12);

const unpack = bytes => JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 16, new DataView(bytes).getUint32(4, true))));
const geometry = {interior: [], glass: []}, digest = 'a'.repeat(64);
assert.equal(unpack(packMarblePhoto(geometry, digest)).max_bounces, 4);
for (const count of [1, 4, 12, 13, 50, 200]) {
  input.value = String(count); input.fire('input');
  assert.equal(unpack(packMarblePhoto(geometry, digest, {maxBounces: photoBounceCount(doc)})).max_bounces, count);
}
const html = readFileSync(new URL('../examples/meowdoku/web/marble_studio.html', import.meta.url), 'utf8');
for (const [id, attributes] of [
  ['photo-bounces', {type: 'number', min: 1, max: 200, step: 1, value: 4}],
  ['photo-bounces-slider', {type: 'range', min: 0, max: 100, step: 0.01, value: 12.28}],
]) {
  const tag = html.match(new RegExp(`<input id="${id}"[^>]*>`))?.[0];
  assert.ok(tag, `${id} exists`);
  for (const [key, value] of Object.entries(attributes)) assert.ok(tag.includes(`${key}="${value}"`), `${id} ${key}`);
}
assert.match(html, /High limits can substantially slow photos and live ray updates/);
console.log('Bounce controls: 200 reachable integers, low-count precision, legacy values, keyboard and payload checks PASS');

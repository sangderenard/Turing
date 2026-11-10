// Exercise the production photo lifecycle with network, image and clock boundaries mocked.
// No server or GPU is used. Run: node tests/meowdoku_marble_photo_retouch_check.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createMarblePhoto} from '../examples/meowdoku/web/marble_photo.js';

const A = 'a'.repeat(64), B = 'b'.repeat(64);
const flush = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
const reply = (body, status = 200) => ({ok: status >= 200 && status < 300, status, json: async () => body});
const complete = (extra = {}) => ({state: 'complete', image: 'frame-000000.png', hdr: 'frame-000000.npz',
  display_gamma: 1.2, frames: [], epoch: 10, ...extra});
const profile = {id: 'measured-profile-fixture', output_space: 'linear sRGB D65',
  reference_white: 'equal-power reference adapted to D65', condition_number: 2.5,
  sensor_to_linear_srgb: [[1.25, -.2, .04], [.1, 1, -.03], [-.01, .12, .9]],
  display_transform: 'luminance tone; gamut compression; sRGB transfer once',
  exposure_reference: '99th percentile of positive profiled luminance',
  limits: 'Three discrete lines only; not a measured camera calibration.'};

function harness(initial = complete()) {
  const original = Object.fromEntries(['fetch', 'Image', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'].map(key => [key, globalThis[key]]));
  const requests = [], draws = [], images = [], events = [], timers = new Map(), elements = new Map();
  let timerId = 0, now = 0, current = initial, fetchHook = null, delayImages = false, scene = null, sceneReads = 0, prepare = null;
  class Element {
    constructor(id = '') { this.id = id; this.value = '30'; this.hidden = true; this.textContent = ''; this.handlers = new Map(); this.children = []; }
    addEventListener(type, callback) { this.handlers.set(type, callback); }
    dispatchEvent(event) { events.push({target: this.id, type: event.type, bubbles: event.bubbles}); }
    fire(type) { return this.handlers.get(type)?.(); }
    replaceChildren(...nodes) { this.children = nodes; }
    append(...nodes) { this.children.push(...nodes); }
    getContext() { return {drawImage: image => draws.push(image.src)}; }
  }
  const element = id => { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); };
  const doc = {getElementById: id => ['photo-film-response','photo-lens-controls'].includes(id) ? null : element(id), createElement: () => new Element(),
    defaultView: {location: {search: `?photo=${A}`}, Event}};
  element('photo-transport').value = 'fixed:3'; element('photo-gamma').value = '100';
  element('photo-orbit-frames').value = '1'; element('photo-bounces').value = '4';
  element('photo-resolution').value = '256'; element('photo-bounce-mode').value = 'manual';
  element('photo-triangle-budget').value = '250000'; element('photo-sphere-coverage').value = 'mesh';
  globalThis.Image = class {
    constructor() { this.naturalWidth = 32; this.naturalHeight = 32; images.push(this); }
    set src(value) { this._src = value; if (!delayImages) queueMicrotask(() => this.onload?.()); }
    get src() { return this._src; }
    async decode() {}
  };
  const schedule = (callback, ms, interval = false) => { const id = ++timerId; timers.set(id, {callback, at: now + ms, ms, interval}); return id; };
  globalThis.setTimeout = (callback, ms) => schedule(callback, ms);
  globalThis.setInterval = (callback, ms) => schedule(callback, ms, true);
  globalThis.clearTimeout = globalThis.clearInterval = id => timers.delete(id);
  globalThis.fetch = async (url, options = {}) => {
    const request = {url, ...options, data: typeof options.body === 'string' ? JSON.parse(options.body) : null};
    requests.push(request);
    if (fetchHook) { const result = fetchHook(request); if (result !== undefined) return result; }
    if (url.endsWith('/retouch')) return reply({ok: true});
    if (options.method === 'POST') return reply({digest: B});
    return reply(current);
  };
  const photo = createMarblePhoto(doc, () => { sceneReads++; return scene; }, () => prepare());
  return {photo, element, requests, draws, images, events, timers,
    set prepare(value) { prepare = value; }, set scene(value) { scene = value; }, get sceneReads() { return sceneReads; },
    set status(value) { current = value; }, set fetchHook(value) { fetchHook = value; },
    set delayImages(value) { delayImages = value; },
    retouches: () => requests.filter(request => request.url.endsWith('/retouch')),
    async tick(ms) {
      const end = now + ms;
      while (true) {
        const entry = [...timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!entry) break;
        const [id, timer] = entry; now = timer.at; timers.delete(id);
        if (timer.interval) timers.set(id, {...timer, at: now + timer.ms});
        timer.callback(); await flush();
      }
      now = end; await flush();
    },
    input(field, value) { element(`retouch-${field}`).value = String(value); element(`retouch-${field}`).fire('input'); },
    restore() { Object.assign(globalThis, original); },
  };
}

const cases = [];
function check(name, body, initial) { cases.push([name, body, initial]); }

check('completed Apply refreshes the actual canvas and full-size link', async h => {
  await flush();
  assert.equal(h.element('photo-cancel').disabled, true);
  h.status = complete({retouched_image: 'preview-900001.png'});
  h.input('red', 180); h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.draws.at(-1), /preview-900001\.png/);
  assert.equal(h.element('photo-full').href, h.draws.at(-1));
  assert.equal(h.retouches().at(-1).data.red, 1.8);
  assert.equal(h.retouches().length, 1, 'viewing a finished HDR must not overwrite its saved retouch');
  await h.tick(5000);
  assert.equal(h.retouches().length, 1, 'Apply consumes the slider debounce');
  assert.equal(h.timers.size, 0, 'completed refresh must not restart exposure polling');
});

check('sliders debounce to one POST and refresh without Apply', async h => {
  await flush();
  h.input('red', 130); await h.tick(100);
  h.input('green', 160); await h.tick(100);
  h.input('red', 190); await h.tick(149);
  assert.equal(h.retouches().length, 0);
  h.status = complete({retouched_image: 'preview-900002.png'});
  await h.tick(1);
  assert.equal(h.retouches().length, 1);
  assert.equal(h.retouches()[0].data.red, 1.9);
  assert.equal(h.retouches()[0].data.green, 1.6);
  assert.match(h.draws.at(-1), /preview-900002\.png/);
});

check('Reset replaces a pending slider edit and restores baked gamma', async h => {
  await flush();
  h.input('gamma', 230); h.input('exposure', 150);
  h.status = complete({retouched_image: 'preview-900003.png'});
  h.element('retouch-reset').fire('click'); await flush(); await h.tick(200);
  assert.equal(h.retouches().length, 1);
  assert.equal(h.retouches()[0].data.gamma, 1.2);
  assert.equal(h.retouches()[0].data.exposure, 0);
  assert.equal(h.element('retouch-gamma').value, '120');
  assert.match(h.draws.at(-1), /preview-900003\.png/);
  assert.deepEqual(h.events, [
    {target: 'photo-retouch', type: 'retouch-controls-change', bubbles: true},
    {target: 'photo-retouch', type: 'retouch-controls-change', bubbles: true},
  ], 'programmatic activation and reset notify the film science controls');
});

check('one POST in flight and one latest pending edit bound repeated Apply', async h => {
  await flush();
  const first = deferred(), second = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ?
    (h.retouches().length === 1 ? first.promise : second.promise) : undefined;
  h.input('red', 110); h.element('retouch-apply').fire('click'); await flush();
  for (let value = 120; value <= 290; value += 10) {
    h.input('red', value); h.element('retouch-apply').fire('click');
  }
  assert.equal(h.retouches().length, 1);
  first.resolve(reply({ok: true})); await flush();
  assert.equal(h.retouches().length, 2);
  assert.equal(h.retouches()[1].data.red, 2.9);
  assert.equal(h.requests.filter(request => !request.method).length, 1, 'stale edit must not trigger a status refresh');
  h.status = complete({retouched_image: 'preview-900004.png'});
  second.resolve(reply({ok: true})); await flush();
  assert.match(h.draws.at(-1), /preview-900004\.png/);
  assert.equal(h.retouches().length, 2);
});

for (const [label, failed] of [
  ['HTTP', () => reply({error: 'saved HDR unavailable'}, 404)],
  ['network', () => Promise.reject(new Error('offline'))],
  ['malformed response', () => ({ok: true, status: 200, json: async () => { throw new Error('invalid JSON'); }})],
  ['rejected payload', () => reply({ok: false, error: 'retouch refused'})],
]) {
  check(`${label} retouch failure is visible and retry remains usable`, async h => {
    await flush(); const original = h.draws.at(-1);
    h.fetchHook = request => request.url.endsWith('/retouch') ? failed() : undefined;
    h.element('retouch-apply').fire('click'); await flush();
    assert.match(h.element('photo-status').textContent, /^Retouch failed:/);
    assert.equal(h.draws.at(-1), original);
    h.fetchHook = null; h.status = complete({retouched_image: 'preview-900005.png'});
    h.element('retouch-apply').fire('click'); await flush();
    assert.match(h.draws.at(-1), /preview-900005\.png/);
    assert.doesNotMatch(h.element('photo-status').textContent, /failed/i);
  });
}

check('successful POST followed by failed status GET reports the actual error', async h => {
  await flush();
  h.fetchHook = request => !request.method ? reply({error: 'status file unavailable'}, 503) : undefined;
  h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.element('photo-status').textContent, /Retouch failed: status file unavailable/);
  assert.equal(h.timers.size, 0);
});

check('saving another design drops debounce and never falls back to old lastDigest', async h => {
  await flush(); h.input('red', 160);
  h.photo.saved(B); await h.tick(1000);
  assert.equal(h.retouches().length, 0);
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches().length, 0);
  assert.match(h.element('photo-status').textContent, /No HDR photo/);
});

check('a resumed photo becomes stale even before any object has been saved', async h => {
  await flush(); h.input('red', 160); h.photo.stale(); await h.tick(1000);
  assert.equal(h.retouches().length, 0);
  assert.equal(h.element('photo-retouch').hidden, true);
  assert.match(h.element('photo-status').textContent, /Design changed/);
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches().length, 0);
});

check('old completed retouch response cannot replace a newly started photo', async h => {
  await flush(); const old = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ? old.promise : undefined;
  h.element('retouch-apply').fire('click'); await flush();
  h.photo.saved(B);
  h.status = complete({image: 'new-photo.png'});
  await h.photo.start({interior: [], glass: [], triangleCount: 0}); await flush();
  assert.match(h.draws.at(-1), /digest=bbbb.*new-photo\.png/);
  const count = h.draws.length, status = h.element('photo-status').textContent;
  old.resolve(reply({ok: true})); await flush();
  assert.equal(h.draws.length, count);
  assert.equal(h.element('photo-status').textContent, status);
});

check('old retouch image decode cannot replace newer settings', async h => {
  await flush(); h.delayImages = true;
  h.status = complete({retouched_image: 'preview-900006.png'});
  h.element('retouch-apply').fire('click'); await flush();
  const oldImage = h.images.at(-1), count = h.draws.length;
  h.input('red', 175);
  oldImage.onload(); await flush();
  assert.equal(h.draws.length, count);
  h.status = complete({retouched_image: 'preview-900007.png'});
  await h.tick(150); h.images.at(-1).onload(); await flush();
  assert.match(h.draws.at(-1), /preview-900007\.png/);
});

check('non-final selected orbit stays selected and refuses retouch', async h => {
  await flush();
  const frames = [0, 1].map(index => ({index, image: `frame-${index}.png`, hdr: `frame-${index}.npz`}));
  h.status = complete({frames, image: frames[1].image, hdr: frames[1].hdr});
  h.element('retouch-apply').fire('click'); await flush();
  const pending = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ? pending.promise : undefined;
  h.element('retouch-apply').fire('click'); await flush();
  h.element('photo-frames').children[0].fire('click'); await flush();
  assert.match(h.draws.at(-1), /name=frame-0\.png/);
  assert.equal(h.element('photo-retouch').hidden, true);
  const count = h.retouches().length;
  h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.element('photo-status').textContent, /final exposure only/);
  assert.equal(h.retouches().length, count);
  pending.resolve(reply({ok: true})); await flush();
  assert.match(h.draws.at(-1), /name=frame-0\.png/);
  assert.equal(h.element('photo-caption').textContent, 'Orbit frame 1 of 2');
  h.fetchHook = null;
  h.element('photo-frames').children[1].fire('click'); await flush();
  assert.equal(h.element('photo-retouch').hidden, false);
  h.status = complete({frames, image: frames[1].image, hdr: frames[1].hdr, retouched_image: 'preview-900008.png'});
  h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.draws.at(-1), /preview-900008\.png/);
  assert.equal(h.element('photo-caption').textContent, 'Orbit frame 2 of 2');
});

check('live retouch preserves the single polling loop through completion', async h => {
  await flush();
  assert.equal(h.element('photo-cancel').disabled, false);
  h.input('red', 150); await h.tick(150);
  assert.equal(h.retouches().length, 1);
  assert.equal(h.timers.size, 1);
  h.status = complete({state: 'exposing', retouched_image: 'preview-900010.png'});
  await h.tick(1850);
  assert.match(h.draws.at(-1), /preview-900010\.png/);
  assert.equal(h.timers.size, 1);
  h.status = complete({retouched_image: 'preview-900011.png'});
  await h.tick(2000);
  assert.match(h.draws.at(-1), /preview-900011\.png/);
  assert.equal(h.timers.size, 0);
  h.status = complete({retouched_image: 'preview-900012.png'});
  h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.draws.at(-1), /preview-900012\.png/);
}, complete({state: 'exposing'}));

check('late live status cannot undo retouch or stop the next polling tick', async h => {
  await flush(); const oldStatus = deferred(); let firstRead = true;
  h.fetchHook = request => {
    if (!request.method && firstRead) { firstRead = false; return oldStatus.promise; }
  };
  await h.tick(2000);
  h.status = complete({state: 'exposing', retouched_image: 'preview-900013.png'});
  h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.draws.at(-1), /preview-900013\.png/);
  oldStatus.resolve(reply(complete({state: 'exposing', image: 'older-status.png'}))); await flush();
  assert.match(h.draws.at(-1), /preview-900013\.png/);
  assert.equal(h.timers.size, 1);
  h.status = complete({retouched_image: 'preview-900014.png'}); await h.tick(2000);
  assert.match(h.draws.at(-1), /preview-900014\.png/);
  assert.equal(h.timers.size, 0);
}, complete({state: 'exposing'}));

check('live status refresh does not erase a retouch failure', async h => {
  await flush();
  h.fetchHook = request => request.url.endsWith('/retouch') ? reply({error: 'bad HDR'}, 422) : undefined;
  h.element('retouch-apply').fire('click'); await flush(); await h.tick(2000);
  assert.match(h.element('photo-status').textContent, /Retouch failed: bad HDR/);
  assert.equal(h.timers.size, 1);
}, complete({state: 'exposing'}));

check('Cancel invalidates live retouch response and queued sliders', async h => {
  await flush(); const old = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ? old.promise : undefined;
  h.element('retouch-apply').fire('click'); await flush();
  h.input('red', 175);
  await h.element('photo-cancel').fire('click'); await flush();
  const count = h.draws.length, status = h.element('photo-status').textContent;
  old.resolve(reply({error: 'old failure'}, 500)); await h.tick(5000);
  assert.equal(h.retouches().length, 1);
  assert.equal(h.draws.length, count);
  assert.equal(h.element('photo-status').textContent, status);
  assert.equal(h.element('photo-retouch').hidden, true);
  assert.equal(h.timers.size, 0);
}, complete({state: 'exposing'}));

check('stale saved design invalidates finished retouch and its status refresh', async h => {
  await flush(); h.photo.saved(B);
  await h.photo.start({interior: [], glass: [], triangleCount: 0}); await flush();
  const old = deferred();
  h.fetchHook = request => !request.method ? old.promise : undefined;
  h.element('retouch-apply').fire('click'); await flush();
  h.photo.stale(); const status = h.element('photo-status').textContent, count = h.draws.length;
  old.resolve(reply(complete({retouched_image: 'obsolete.png'}))); await flush();
  assert.equal(h.draws.length, count);
  assert.equal(h.element('photo-status').textContent, status);
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches().length, 1);
  assert.equal(h.element('photo-retouch').hidden, true);
});

check('live image without HDR still displays and controls wait for HDR', async h => {
  await flush();
  assert.match(h.draws.at(-1), /frame-000000\.png/);
  assert.equal(h.element('photo-retouch').hidden, true);
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches().length, 0);
  h.status = complete({state: 'exposing'}); await h.tick(2000);
  assert.equal(h.element('photo-retouch').hidden, false);
}, complete({state: 'exposing', hdr: null}));

check('postprocessing displays the saved sensor profile and its actual matrix', async h => {
  await flush();
  assert.match(h.element('retouch-profile-name').textContent, /measured-profile-fixture.*linear sRGB D65/);
  assert.match(h.element('retouch-profile-matrix').textContent, /1\.25  -0\.2  0\.04/);
  assert.match(h.element('retouch-profile-reference').textContent, /equal-power reference adapted to D65/);
  assert.match(h.element('retouch-profile-condition').textContent, /2\.5/);
  assert.match(h.element('retouch-profile-display').textContent, /sRGB transfer once/);
  assert.match(h.element('retouch-profile-display').textContent, /99th percentile/);
  assert.match(h.element('retouch-profile-display').textContent, /White: 4\.125/);
  assert.equal(h.element('retouch-profile-limits').textContent, profile.limits);
  assert.equal(h.element('retouch-profile-warning').hidden, true);
  assert.equal(h.retouches().length, 0, 'metadata inspection is read-only');
}, complete({color_profile: profile, display_white: 4.125}));

check('neutral-fallback warning is visible and old HDR has an explicit legacy label', async h => {
  await flush();
  assert.match(h.element('retouch-profile-condition').textContent, /not recorded; neutral fallback active/);
  assert.equal(h.element('retouch-profile-warning').hidden, false);
  assert.match(h.element('retouch-profile-warning').textContent, /cannot stably recover color/);
  h.status = complete(); h.element('retouch-apply').fire('click'); await flush();
  assert.match(h.element('retouch-profile-name').textContent, /Legacy \/ unprofiled HDR/);
  assert.equal(h.element('retouch-profile-matrix').textContent, 'Sensor matrix: not recorded.');
  assert.equal(h.element('retouch-profile-warning').hidden, true);
  assert.doesNotMatch(h.element('retouch-profile-display').textContent, /sRGB transfer once/);
  h.photo.saved(B);
  assert.match(h.element('retouch-profile-name').textContent, /waiting for exposure metadata/);
}, complete({color_profile: {...profile, condition_number: null, neutral_fallback: true,
  warning: 'Film response cannot stably recover color; neutral fallback is shown.'}}));

check('profiled normalization and tone controls post exact unscaled values after completion', async h => {
  await flush();
  assert.equal(h.element('retouch-normalization').disabled, false);
  assert.equal(h.element('retouch-tone-curve').disabled, false);
  assert.equal(h.element('retouch-normalization-white').disabled, true);
  h.element('retouch-normalization').value = 'manual'; h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-normalization-white').disabled, false);
  h.element('retouch-normalization-white').value = '2.75'; h.element('retouch-normalization-white').fire('input');
  h.element('retouch-tone-curve').value = 'linear'; h.element('retouch-tone-curve').fire('change');
  assert.match(h.element('retouch-display-summary').textContent, /manual scene-linear white 2\.75; linear luminance/);
  h.status = complete({color_profile: profile, retouched_image: 'preview-manual.png', retouch_display_transform: {
    normalization: 'manual', normalization_white: 2.75, effective_white: 2.75, tone_curve: 'linear',
    tone_formula: 'clip(Y/white,0,1)', transfer: 'standard sRGB once', gamut_mapping: 'neutral-axis compression',
  }});
  await h.tick(150);
  assert.equal(h.retouches().length, 1);
  assert.equal(h.retouches()[0].data.normalization, 'manual');
  assert.equal(h.retouches()[0].data.normalization_white, 2.75);
  assert.equal(h.retouches()[0].data.tone_curve, 'linear');
  assert.match(h.draws.at(-1), /preview-manual\.png/);
  assert.match(h.element('retouch-profile-rendered').textContent, /Last rendered: manual.*effective white 2\.75.*linear.*clip\(Y\/white,0,1\)/);
  assert.equal(h.element('retouch-normalization').value, 'manual');
  assert.equal(h.element('retouch-normalization-white').value, '2.75');
  assert.equal(h.element('retouch-tone-curve').value, 'linear');
  h.element('retouch-reset').fire('click'); await flush();
  assert.equal(h.retouches().at(-1).data.normalization, 'auto99');
  assert.equal(h.retouches().at(-1).data.normalization_white, 1);
  assert.equal(h.retouches().at(-1).data.tone_curve, 'exponential');
  assert.equal(h.element('retouch-normalization-white').value, '1');
  assert.equal(h.element('retouch-normalization-white').disabled, true);
}, complete({color_profile: profile}));

check('percentile and specular edits preserve HDR, restore tone, and reset default display', async h => {
  await flush();
  assert.equal(h.element('retouch-normalization').value, 'auto99');
  assert.equal(h.element('retouch-normalization-percentile').value, '99');
  assert.equal(h.element('retouch-normalization-percentile').disabled, true);
  h.element('retouch-normalization').value = 'percentile'; h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-normalization-percentile').disabled, false);
  h.input('normalization-percentile', '98.4'); await h.tick(150);
  assert.equal(h.retouches().at(-1).data.normalization_percentile, 98.4);
  assert.equal(h.retouches().at(-1).data.tone_curve, 'exponential');
  h.element('retouch-normalization').value = 'specular'; h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-normalization-percentile').value, '95');
  assert.equal(h.element('retouch-tone-curve').value, 'linear');
  assert.equal(h.element('retouch-tone-curve').disabled, true);
  h.input('normalization-percentile', '92.7');
  h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-normalization-percentile').value, '92.7', 'repeat selection preserves the edited percentile');
  h.status = complete({color_profile: profile, retouched_image: 'specular.png', retouch_display_transform: {
    normalization: 'specular', normalization_percentile: 92.7, effective_percentile: 92.7,
    effective_white: 6.2, tone_curve: 'linear', specular_clip_mode: true,
    highlight_policy: 'Clip luminance above the chosen white; retain raw HDR.',
  }});
  await h.tick(150);
  assert.equal(h.retouches().at(-1).data.normalization, 'specular');
  assert.equal(h.retouches().at(-1).data.normalization_percentile, 92.7);
  assert.equal(h.retouches().at(-1).data.tone_curve, 'linear');
  assert.match(h.element('retouch-profile-rendered').textContent, /Effective percentile: 92.7.*Clip luminance above the chosen white/);
  assert.match(h.element('retouch-display-summary').textContent, /highlights above the selected white clip/);
  h.element('retouch-normalization').value = 'percentile'; h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-tone-curve').value, 'exponential', 'leaving specular restores the prior user tone');
  assert.equal(h.element('retouch-tone-curve').disabled, false);
  h.element('retouch-tone-curve').value = 'linear'; h.element('retouch-tone-curve').fire('change');
  h.element('retouch-normalization').value = 'specular'; h.element('retouch-normalization').fire('change');
  h.element('retouch-normalization').value = 'manual'; h.element('retouch-normalization').fire('change');
  assert.equal(h.element('retouch-tone-curve').value, 'linear', 'an explicit earlier linear choice is preserved too');
  h.element('retouch-reset').fire('click'); await flush();
  assert.equal(h.retouches().at(-1).data.normalization, 'auto99');
  assert.equal(h.retouches().at(-1).data.normalization_percentile, 99);
  assert.equal(h.retouches().at(-1).data.tone_curve, 'exponential');
  assert.equal(h.element('retouch-normalization-percentile').disabled, true);
  assert.ok(h.requests.filter(r => r.method === 'POST').every(r => r.url.endsWith('/retouch')), 'display changes never retrace or cancel');
}, complete({color_profile: profile}));

check('invalid percentiles cancel queued retouch and endpoints stay inclusive', async h => {
  await flush();
  h.element('retouch-normalization').value = 'percentile'; h.element('retouch-normalization').fire('change');
  for (const value of ['', 'NaN', 'Infinity', '0', '.99', '100.01']) {
    h.input('normalization-percentile', value); await h.tick(150);
    assert.equal(h.retouches().length, 0);
    assert.match(h.element('photo-status').textContent, /Normalization percentile must be a finite value from 1 to 100/);
  }
  for (const value of ['1', '100']) {
    h.input('normalization-percentile', value); await h.tick(150);
    assert.equal(h.retouches().at(-1).data.normalization_percentile, Number(value));
  }
}, complete({color_profile: profile}));

check('legacy controls are disabled and all new fields are omitted from POST', async h => {
  await flush();
  for (const id of ['retouch-normalization', 'retouch-normalization-white', 'retouch-normalization-percentile', 'retouch-tone-curve',
    'retouch-saturation', 'retouch-vibrance', 'retouch-black-point',
    'retouch-firefly-filter', 'retouch-firefly-threshold', 'retouch-firefly-strength']) {
    assert.equal(h.element(id).disabled, true);
  }
  h.element('retouch-normalization').value = 'manual';
  h.element('retouch-normalization-white').value = 'NaN';
  h.element('retouch-tone-curve').value = 'linear';
  h.element('retouch-saturation').value = 'NaN'; h.element('retouch-vibrance').value = 'Infinity';
  h.element('retouch-black-point').value = '50';
  h.element('retouch-firefly-filter').checked = true;
  h.element('retouch-firefly-threshold').value = 'NaN'; h.element('retouch-firefly-strength').value = 'Infinity';
  h.input('red', 150); h.element('retouch-apply').fire('click'); await flush();
  const payload = h.retouches()[0].data;
  assert.equal(payload.red, 1.5);
  for (const key of ['normalization', 'normalization_white', 'normalization_percentile', 'tone_curve', 'saturation', 'vibrance', 'black_point',
    'firefly_filter', 'firefly_threshold', 'firefly_strength']) assert.equal(key in payload, false);
  assert.match(h.element('retouch-display-summary').textContent, /Legacy RGB corrections remain available/);
});

check('invalid manual white cancels queued edits and never reaches the API', async h => {
  await flush();
  h.element('retouch-normalization').value = 'manual'; h.element('retouch-normalization').fire('change');
  for (const value of ['', 'NaN', 'Infinity', '0', '0.0000001', '1000000000001']) {
    h.element('retouch-normalization-white').value = value;
    h.element('retouch-normalization-white').fire('input'); await h.tick(150);
    assert.equal(h.retouches().length, 0);
    assert.match(h.element('photo-status').textContent, /Manual white must be a finite scene-linear value/);
  }
  for (const value of ['0.000001', '1000000000000']) {
    h.element('retouch-normalization-white').value = value;
    h.element('retouch-normalization-white').fire('input'); await h.tick(150);
    assert.equal(h.retouches().at(-1).data.normalization_white, Number(value));
  }
  assert.equal(h.retouches().length, 2);
}, complete({color_profile: profile}));

check('live status records rendered settings without replacing newer local choices', async h => {
  await flush();
  h.element('retouch-normalization').value = 'manual'; h.element('retouch-normalization').fire('change');
  h.element('retouch-normalization-white').value = '3.5'; h.element('retouch-normalization-white').fire('input');
  h.element('retouch-tone-curve').value = 'linear'; h.element('retouch-tone-curve').fire('change');
  h.input('saturation', 145); h.input('vibrance', -35); h.input('black-point', 12);
  await h.tick(150);
  h.status = complete({state: 'exposing', color_profile: profile,
    retouch_params: {normalization: 'auto99', normalization_white: 1, tone_curve: 'exponential', saturation: 1, vibrance: 0, black_point: 0},
    retouch_display_transform: {normalization: 'auto99', normalization_white: 1, effective_white: 9,
      tone_curve: 'exponential', tone_formula: '1-exp(-2*Y/white)', transfer: 'standard sRGB once', saturation: 1, vibrance: 0, black_point: 0},
  });
  await h.tick(1850);
  assert.equal(h.element('retouch-normalization').value, 'manual');
  assert.equal(h.element('retouch-normalization-white').value, '3.5');
  assert.equal(h.element('retouch-tone-curve').value, 'linear');
  assert.equal(h.element('retouch-saturation').value, '145');
  assert.equal(h.element('retouch-vibrance').value, '-35');
  assert.equal(h.element('retouch-black-point').value, '12');
  assert.match(h.element('retouch-display-summary').textContent, /manual scene-linear white 3\.5; linear/);
  assert.match(h.element('retouch-profile-rendered').textContent, /auto99.*effective white 9.*exponential/);
  assert.match(h.element('retouch-profile-rendered').textContent, /Saturation: 1; vibrance: 0; black point: 0/);
}, complete({state: 'exposing', color_profile: profile}));

check('artistic controls start neutral, post scaled values, and Reset preserves raw HDR', async h => {
  await flush();
  const hdr = h.element('photo-hdr').href;
  assert.equal(h.retouches().length, 0, 'new controls must not write until edited');
  for (const [field, value] of [['saturation', '100'], ['vibrance', '0'], ['black-point', '0']]) {
    assert.equal(h.element(`retouch-${field}`).value, value);
    assert.equal(h.element(`retouch-${field}`).disabled, false);
  }
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches()[0].data.saturation, 1);
  assert.equal(h.retouches()[0].data.vibrance, 0);
  assert.equal(h.retouches()[0].data.black_point, 0);
  h.input('saturation', 172); h.input('vibrance', -43); h.input('black-point', 24);
  const recipe = {saturation: 1.72, vibrance: -.43, black_point: .24,
    black_point_formula: 'backend black point recipe', chroma_formula: 'backend chroma recipe',
    creative_controls: 'Artistic controls, not perceptually uniform.'};
  h.status = complete({color_profile: profile, retouched_image: 'creative.png', retouch_display_transform: recipe});
  await h.tick(150);
  assert.equal(h.retouches().at(-1).data.saturation, 1.72);
  assert.equal(h.retouches().at(-1).data.vibrance, -.43);
  assert.equal(h.retouches().at(-1).data.black_point, .24);
  assert.match(h.draws.at(-1), /creative\.png/);
  for (const [field, value] of [['saturation', '1.72'], ['vibrance', '-0.43'], ['black-point', '0.24']])
    assert.equal(h.element(`retouch-${field}-out`).textContent, value);
  for (const key of ['black_point_formula', 'chroma_formula', 'creative_controls'])
    assert.ok(h.element('retouch-profile-rendered').textContent.includes(recipe[key]));
  assert.match(h.element('retouch-display-summary').textContent, /Black point 0\.24 of normalized white before tone; saturation 1\.72; vibrance -0\.43/);
  h.input('saturation', 190); h.input('black-point', 50);
  h.element('retouch-reset').fire('click'); await flush(); await h.tick(150);
  assert.equal(h.retouches().length, 3, 'Reset consumes the pending artistic edit');
  assert.equal(h.retouches().at(-1).data.saturation, 1);
  assert.equal(h.retouches().at(-1).data.vibrance, 0);
  assert.equal(h.retouches().at(-1).data.black_point, 0);
  assert.equal(h.retouches().at(-1).data.gamma, 1.2);
  assert.equal(h.element('retouch-saturation').value, '100');
  assert.equal(h.element('retouch-vibrance').value, '0');
  assert.equal(h.element('retouch-black-point').value, '0');
  assert.equal(h.element('photo-hdr').href, hdr, 'display edits must not redirect raw HDR');
}, complete({color_profile: profile}));

check('artistic slider bursts serialize and only the newest complete parameter set wins', async h => {
  await flush(); const first = deferred(), last = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ?
    (h.retouches().length === 1 ? first.promise : last.promise) : undefined;
  h.input('saturation', 110); h.element('retouch-apply').fire('click'); await flush();
  for (let n = 1; n <= 10; n++) {
    h.input('saturation', 100 + n * 5); h.input('vibrance', n * 5); h.input('black-point', n * 2);
    h.element('retouch-apply').fire('click');
  }
  assert.equal(h.retouches().length, 1);
  const draws = h.draws.length;
  first.resolve(reply({ok: true})); await flush();
  assert.equal(h.draws.length, draws, 'outdated response must not refresh the display');
  assert.equal(h.retouches().length, 2);
  assert.equal(h.retouches()[1].data.saturation, 1.5);
  assert.equal(h.retouches()[1].data.vibrance, .5);
  assert.equal(h.retouches()[1].data.black_point, .2);
  h.status = complete({color_profile: profile, retouched_image: 'latest-creative.png'});
  last.resolve(reply({ok: true})); await flush();
  assert.match(h.draws.at(-1), /latest-creative\.png/);
}, complete({color_profile: profile}));

check('artistic slider validation cancels queued edits and accepts exact endpoint bounds', async h => {
  await flush();
  for (const [field, key, min, max, neutral] of [
    ['saturation', 'saturation', 0, 200, 100], ['vibrance', 'vibrance', -100, 100, 0],
    ['black-point', 'black_point', 0, 50, 0],
  ]) {
    for (const bad of ['', 'NaN', 'Infinity', min - 1, max + 1]) {
      h.input(field, neutral); const before = h.retouches().length;
      h.input(field, bad); await h.tick(150);
      assert.equal(h.retouches().length, before);
      assert.match(h.element('photo-status').textContent, /must be a finite value/);
    }
    for (const bound of [min, max]) {
      h.input(field, bound); await h.tick(150);
      assert.equal(h.retouches().at(-1).data[key], bound / 100);
    }
    h.input(field, neutral); await h.tick(150);
  }
}, complete({color_profile: profile}));

check('isolated-highlight filter is opt-in, records actual render settings, and Reset restores defaults', async h => {
  await flush();
  assert.equal(h.retouches().length, 0);
  assert.equal(h.element('retouch-firefly-filter').checked, false);
  assert.equal(h.element('retouch-firefly-filter').disabled, false);
  for (const [field, value] of [['threshold', '8'], ['strength', '100']]) {
    assert.equal(h.element(`retouch-firefly-${field}`).value, value);
    assert.equal(h.element(`retouch-firefly-${field}`).disabled, true);
  }
  h.element('retouch-apply').fire('click'); await flush();
  assert.equal(h.retouches()[0].data.firefly_filter, false);
  assert.equal(h.retouches()[0].data.firefly_threshold, 8);
  assert.equal(h.retouches()[0].data.firefly_strength, 1);
  const raw = h.element('photo-hdr').href;
  h.element('retouch-firefly-filter').checked = true;
  h.element('retouch-firefly-filter').fire('change');
  assert.equal(h.element('retouch-firefly-threshold').disabled, false);
  assert.equal(h.element('retouch-firefly-strength').disabled, false);
  h.input('firefly-threshold', '12.5'); h.input('firefly-strength', 35);
  const recipe = {firefly_filter: true, firefly_threshold: 12.5, firefly_strength: .35,
    firefly_method: 'Y > threshold * max of 8 neighbours; blend after gains before tone',
    firefly_warning: 'May reduce genuine isolated glints or sharp highlight peaks.'};
  h.status = complete({color_profile: profile, retouched_image: 'filtered.png', retouch_display_transform: recipe});
  await h.tick(150);
  assert.equal(h.retouches().length, 2);
  assert.equal(h.retouches().at(-1).data.firefly_filter, true);
  assert.equal(h.retouches().at(-1).data.firefly_threshold, 12.5);
  assert.equal(h.retouches().at(-1).data.firefly_strength, .35);
  assert.match(h.draws.at(-1), /filtered\.png/);
  assert.match(h.element('retouch-display-summary').textContent, /filter on: threshold 12\.5×; strength 0\.35/);
  for (const key of ['firefly_method', 'firefly_warning'])
    assert.ok(h.element('retouch-profile-rendered').textContent.includes(recipe[key]));
  assert.match(h.element('retouch-profile-rendered').textContent, /filter: on; threshold 12\.5×; strength 0\.35/);
  assert.equal(h.element('photo-hdr').href, raw, 'the raw HDR download stays unchanged');
  h.element('retouch-firefly-filter').checked = false;
  h.element('retouch-firefly-filter').fire('change'); await h.tick(150);
  assert.equal(h.retouches().at(-1).data.firefly_filter, false);
  assert.equal(h.element('retouch-firefly-threshold').disabled, true);
  assert.equal(h.element('retouch-firefly-strength').disabled, true);
  assert.equal(h.element('retouch-firefly-threshold').value, '12.5', 'toggle off retains selected options');
  h.element('retouch-firefly-filter').checked = true;
  h.element('retouch-firefly-filter').fire('change');
  h.element('retouch-reset').fire('click'); await flush(); await h.tick(150);
  const reset = h.retouches().at(-1).data;
  assert.equal(h.retouches().length, 4, 'Reset consumes the pending toggle edit');
  assert.deepEqual([reset.firefly_filter, reset.firefly_threshold, reset.firefly_strength], [false, 8, 1]);
  assert.equal(h.element('retouch-firefly-filter').checked, false);
  assert.equal(h.element('retouch-firefly-threshold').value, '8');
  assert.equal(h.element('retouch-firefly-strength').value, '100');
  assert.equal(h.element('retouch-firefly-strength').disabled, true);
  assert.equal(reset.gamma, 1.2);
  assert.equal(h.events.at(-1).type, 'retouch-controls-change');
}, complete({color_profile: profile}));

check('isolated-highlight edits serialize and stale responses cannot replace the latest toggle or metadata', async h => {
  await flush(); const first = deferred(), last = deferred();
  h.fetchHook = request => request.url.endsWith('/retouch') ?
    (h.retouches().length === 1 ? first.promise : last.promise) : undefined;
  h.element('retouch-firefly-filter').checked = true;
  h.element('retouch-firefly-filter').fire('change');
  h.element('retouch-apply').fire('click'); await flush();
  for (const threshold of [10, 16, 32]) {
    h.input('firefly-threshold', threshold); h.input('firefly-strength', threshold);
    h.element('retouch-apply').fire('click');
  }
  h.element('retouch-firefly-filter').checked = false;
  h.element('retouch-firefly-filter').fire('change');
  h.element('retouch-apply').fire('click');
  assert.equal(h.retouches().length, 1);
  first.resolve(reply({ok: true})); await flush();
  assert.equal(h.retouches().length, 2);
  const newest = h.retouches()[1].data;
  assert.deepEqual([newest.firefly_filter, newest.firefly_threshold, newest.firefly_strength], [false, 32, .32]);
  assert.equal(h.requests.filter(request => !request.method).length, 1, 'obsolete filter response must not read status');
  h.status = complete({color_profile: profile, retouched_image: 'latest-filter-off.png',
    retouch_display_transform: {firefly_filter: false, firefly_threshold: 32, firefly_strength: .32}});
  last.resolve(reply({ok: true})); await flush();
  assert.match(h.draws.at(-1), /latest-filter-off\.png/);
  assert.equal(h.element('retouch-firefly-filter').checked, false);
  assert.match(h.element('retouch-profile-rendered').textContent, /filter: off; threshold 32×; strength 0\.32/);
}, complete({color_profile: profile}));

check('isolated-highlight validation rejects invalid options before POST and accepts exact endpoints', async h => {
  await flush();
  h.element('retouch-firefly-filter').checked = true;
  h.element('retouch-firefly-filter').fire('change');
  for (const [field, key, min, max, neutral, scale] of [
    ['firefly-threshold', 'firefly_threshold', 2, 64, 8, 1],
    ['firefly-strength', 'firefly_strength', 0, 100, 100, 100],
  ]) {
    for (const bad of ['', 'NaN', 'Infinity', min - 1, max + 1]) {
      h.input(field, neutral); const before = h.retouches().length;
      h.input(field, bad); await h.tick(150);
      assert.equal(h.retouches().length, before, 'invalid edit also cancels a pending valid edit');
      assert.match(h.element('photo-status').textContent, new RegExp(key.replaceAll('_', ' ')));
    }
    for (const bound of [min, max]) {
      h.input(field, bound); await h.tick(150);
      assert.equal(h.retouches().at(-1).data[key], bound / scale);
    }
    h.input(field, neutral); await h.tick(150);
  }
}, complete({color_profile: profile}));

for (const camera of ['builtin', 'studio']) check(`${camera} photo keeps the authored lights and room while selecting only camera framing`, async h => {
  await flush(); h.photo.saved(B);
  h.element('photo-camera').value = camera;
  h.scene = {camera: {eye_m: [3, 4, 5], target_m: [6, 7, 8], fov_degrees: 35},
    checker: {center_m: [2, 2, 2], width_m: .25, checks: 8},
    fixtures: {triangles: [[1, 2, 3]], materials: [{emission: [2, 3, 4]}]},
    lights: [{position_m: [1, 0, 0], intensity: 2}]};
  await h.photo.start({interior: [], glass: [], triangleCount: 0}); await flush();
  assert.equal(h.sceneReads, 1, 'the scene snapshot must be read once');
  const request = h.requests.find(r => r.url === '/api/marble_photo' && r.method === 'POST');
  const bytes = request.body, length = new DataView(bytes).getUint32(4, true);
  const config = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 16, length)));
  assert.deepEqual(config.fixtures, {triangles: [[1, 2, 3]], materials: [{emission: [2, 3, 4]}]});
  assert.deepEqual(config.lights, [{position_m: [1, 0, 0], intensity: 2}]);
  assert.deepEqual(config.camera.eye_m, camera === 'studio' ? [3, 4, 5] : [0, 0, .07]);
  assert.deepEqual(config.camera.target_m, camera === 'studio' ? [6, 7, 8] : [0, 0, .18]);
  assert.equal(config.camera.fov_degrees, camera === 'studio' ? 35 : 18);
  assert.deepEqual(config.checker, camera === 'studio' ? {center_m: [2, 2, 2], width_m: .25, checks: 8} :
    {center_m: [0, 0, .30], width_m: .15, checks: 4});
  assert.equal(config.camera.resolution, 256);
  assert.equal(config.camera.aperture_radius_m, .002);
  assert.equal(config.camera.focus_distance_m, .11);
});

check('photo uploads the actual nonlinear bounce count and rejects invalid counts before uploading', async h => {
  await flush(); h.photo.saved(B);
  h.element('photo-bounces-slider').value = '100';
  h.element('photo-bounces-slider').fire('input');
  await h.photo.start({interior: [], glass: [], triangleCount: 0}); await flush();
  const requests = () => h.requests.filter(r => r.url === '/api/marble_photo' && r.method === 'POST');
  const body = requests()[0].body;
  const config = JSON.parse(new TextDecoder().decode(new Uint8Array(body, 16, new DataView(body).getUint32(4, true))));
  assert.equal(config.max_bounces, 200);
  assert.equal(h.element('photo-bounces').value, '200');
  h.element('photo-bounces').value = '201';
  await assert.rejects(h.photo.start({interior: [], glass: [], triangleCount: 0}), /whole number from 1 to 200/);
  assert.equal(requests().length, 1);
});

check('geometry policy applies to the next photo and invalid edits never cancel an active exposure', async h => {
  await flush(); h.photo.saved(B);
  h.element('photo-triangle-budget').value = '123456';
  h.status = {state: 'exposing', frames: [], epoch: 1};
  await h.photo.start({interior: [], glass: [], triangleCount: 0}); await flush();
  const body = h.requests.find(r => r.url === '/api/marble_photo' && r.method === 'POST').body;
  const config = JSON.parse(new TextDecoder().decode(new Uint8Array(body, 16, new DataView(body).getUint32(4, true))));
  assert.equal(config.triangle_budget, 123456); assert.equal(config.sphere_coverage, 'mesh');
  const before = h.requests.length;
  for (const invalid of ['', '999', '1000001', '1000.5']) {
    h.element('photo-triangle-budget').value = invalid;
    await assert.rejects(h.photo.start({interior: [], glass: [], triangleCount: 0}), /Triangle budget/);
    assert.equal(h.requests.length, before, 'invalid budget cannot upload, cancel, or poll another exposure');
  }
  h.element('photo-triangle-budget').value = '1000';
  h.element('photo-sphere-coverage').value = 'polygon';
  await assert.rejects(h.photo.start({interior: [], glass: [], triangleCount: 0}), /Choose Saved triangle mesh/);
  assert.equal(h.requests.length, before, 'invalid coverage cannot cancel an exposure');
  h.element('photo-sphere-coverage').value = 'mesh';
  await assert.rejects(h.photo.start({interior: [], glass: [], triangleCount: 0}), /already running/);
  assert.equal(h.requests.length, before, 'even valid new geometry waits for the active photo');
  assert.equal(h.element('photo-cancel').disabled, false);
});

check('explicit Exact selection is sent and an unsupported design never silently falls back to mesh', async h => {
  await flush(); h.photo.saved(B);
  h.element('photo-sphere-coverage').value = 'exact';
  h.fetchHook = request => request.url === '/api/marble_photo' && request.method === 'POST' ?
    reply({error: 'Exact spheres require saved analytic sphere metadata. Save this design again or choose Saved triangle mesh.'}, 400) : undefined;
  await assert.rejects(h.photo.start({interior: [], glass: [], triangleCount: 0}), /Save this design again or choose Saved triangle mesh/);
  const requests = h.requests.filter(r => r.url === '/api/marble_photo' && r.method === 'POST');
  assert.equal(requests.length, 1, 'no silent mesh retry');
  const body = requests[0].body;
  const config = JSON.parse(new TextDecoder().decode(new Uint8Array(body, 16, new DataView(body).getUint32(4, true))));
  assert.equal(config.sphere_coverage, 'exact');
  assert.equal(h.element('photo-sphere-coverage').value, 'exact', 'the rejected choice is not silently rewritten');
});

check('scene-only edits keep the saved object ready without a redundant save', async h => {
  await flush();h.photo.saved(B);h.photo.sceneChanged();
  assert.equal(h.element('photo-start').disabled,false);
  assert.match(h.element('photo-status').textContent,/saved object is ready/);
  assert.ok(!h.requests.some(r=>r.url.endsWith('/cancel')));
  await h.photo.start({interior:[],glass:[],triangleCount:0});
  assert.ok(h.requests.some(r=>r.url==='/api/marble_photo'&&r.method==='POST'));
});
check('autosave and rig edits never interrupt the original still exposure', async h => {
  await flush();h.photo.saved(B);h.status={state:'exposing',frames:[],epoch:1};
  await h.photo.start({interior:[],glass:[],triangleCount:0});await flush();
  const starts=h.requests.filter(r=>r.url==='/api/marble_photo'&&r.method==='POST').length;
  h.photo.stale();h.photo.saved(A);h.photo.sceneChanged();await h.tick(1000);
  assert.equal(h.requests.filter(r=>r.url==='/api/marble_photo'&&r.method==='POST').length,starts);
  assert.ok(!h.requests.some(r=>r.url.endsWith('/cancel')));
  assert.equal(h.element('photo-start').disabled,true);assert.equal(h.element('photo-cancel').disabled,false);
  assert.ok(h.requests.some(r=>r.url===`/api/marble_photo?digest=${B}`));
});

const html = readFileSync(new URL('../examples/meowdoku/web/marble_studio.html', import.meta.url), 'utf8');
for (const prefix of ['retouch', 'live']) {
  const input = html.match(new RegExp(`<input id="${prefix}-normalization-percentile"[^>]*>`))?.[0];
  for (const [key, value] of Object.entries({type:'range', min:1, max:100, step:0.1, value:99}))
    assert.ok(input?.includes(`${key}="${value}"`));
  assert.match(input, /disabled/);
  const select = html.match(new RegExp(`<select id="${prefix}-normalization"[^>]*>(.*?)</select>`))?.[1];
  for (const mode of ['auto99', 'percentile', 'specular', 'manual']) assert.ok(select?.includes(`value="${mode}"`));
}
const filterToggle = html.match(/<input id="retouch-firefly-filter"[^>]*>/)?.[0];
assert.match(filterToggle, /type="checkbox"/); assert.match(filterToggle, /disabled/);
assert.doesNotMatch(filterToggle, /checked/, 'the filter is opt-in');
for (const [id, type, min, max, step, value] of [
  ['threshold', 'number', 2, 64, 'any', 8], ['strength', 'range', 0, 100, 1, 100],
]) {
  const tag = html.match(new RegExp(`<input id="retouch-firefly-${id}"[^>]*>`))?.[0];
  assert.ok(tag);
  for (const [key, expected] of Object.entries({type, min, max, step, value})) assert.ok(tag.includes(`${key}="${expected}"`));
  assert.match(tag, /disabled/); assert.match(tag, /aria-describedby="retouch-firefly-help"/);
}
assert.ok(html.includes('May reduce genuine isolated glints or sharp highlight peaks.'));
for (const [id, min, max, value] of [['saturation', 0, 200, 100], ['vibrance', -100, 100, 0], ['black-point', 0, 50, 0]]) {
  const tag = html.match(new RegExp(`<input id="retouch-${id}"[^>]*>`))?.[0];
  assert.ok(tag, `${id} control exists in production markup`);
  for (const [key, expected] of Object.entries({type: 'range', min, max, step: 1, value}))
    assert.ok(tag.includes(`${key}="${expected}"`), `${id} declares ${key}=${expected}`);
  assert.match(tag, /disabled/, 'profile-only controls must initially be disabled');
  assert.ok(html.includes(`id="retouch-${id}-out"`));
}

check('invalid rod samples remain prominent until a new valid image actually decodes', async h => {
  await flush();assert.equal(h.element('photo-optical-warning').hidden,false);
  assert.match(h.element('photo-optical-warning').textContent,/INVALID IMAGE: 6/);
  h.status={state:'running',image:'valid-rod-frame.png',rod_bundle_diagnostic:{unresolved_samples:0}};
  h.delayImages=true;await h.tick(2000);
  assert.equal(h.element('photo-optical-warning').hidden,false,'a status response alone cannot clear the warning');
  await h.images.at(-1).onload();await flush();
  assert.equal(h.element('photo-optical-warning').hidden,true);
}, {state:'running',image:'invalid-rod-frame.png',rod_bundle_diagnostic:{unresolved_samples:6,warning:'Native cut unresolved.'}});

check('exposure-only snapshot works unsaved, locks repeated clicks and freezes camera/options', async h => {
  await flush();h.photo.stale();h.element('photo-auto-snapshot').checked=true;h.element('photo-auto-snapshot').fire('change');
  assert.equal(h.element('photo-start').disabled,false);
  const gate=deferred();let captures=0;h.prepare=()=>{captures++;return gate.promise;};
  h.element('photo-seconds').value='30';
  const pending=h.photo.start(null);assert.equal(h.element('photo-start').disabled,true);
  await assert.rejects(h.photo.start(null),/already preparing/);
  h.element('photo-seconds').value='99';h.photo.stale();h.photo.saved(A);
  gate.resolve({geometry:{interior:[],glass:[],triangleCount:0},value:{name:'Captured object'}});
  await pending;await flush();assert.equal(captures,1);
  const posts=h.requests.filter(r=>r.method==='POST');assert.equal(posts.length,1);
  assert.equal(posts[0].url,'/api/marble_photo');
  const raw=posts[0].body, config=JSON.parse(new TextDecoder().decode(new Uint8Array(raw,16,new DataView(raw).getUint32(4,true))));
  assert.equal(config.exposure_seconds,30);assert.deepEqual(config.object_snapshot,{name:'Captured object'});
  assert.equal('object_digest' in config,false);
});
check('snapshot failure preserves saved state and can retry without an autosave request', async h=>{
  await flush();h.photo.saved(A);h.element('photo-auto-snapshot').checked=true;
  h.prepare=async()=>{throw new Error('invalid snapshot');};
  await assert.rejects(h.photo.start(null),/invalid snapshot/);
  assert.equal(h.requests.filter(r=>r.method==='POST').length,0);
  assert.equal(h.element('photo-start').disabled,false);
  h.element('photo-auto-snapshot').checked=false;h.element('photo-auto-snapshot').fire('change');
  await h.photo.start({interior:[],glass:[],triangleCount:0});
  const raw=h.requests.find(r=>r.method==='POST').body, config=JSON.parse(new TextDecoder().decode(new Uint8Array(raw,16,new DataView(raw).getUint32(4,true))));
  assert.equal(config.object_digest,A);assert.equal('object_snapshot' in config,false);
});
check('snapshot off still requires an explicit saved object', async h=>{
  await flush();h.photo.stale();h.element('photo-auto-snapshot').checked=false;h.element('photo-auto-snapshot').fire('change');
  assert.equal(h.element('photo-start').disabled,true);
  await assert.rejects(h.photo.start(null),/Save this design/);
  assert.equal(h.requests.filter(r=>r.method==='POST').length,0);
});
assert.match(html, /id="photo-auto-snapshot"[^>]*checked/);

check('completed HDR getter binds actual decoded final profiled frame and detached parameters', async h=>{
  const changes=[];const stop=h.photo.subscribeCompletedHdr(source=>changes.push(source));
  assert.equal(h.photo.completedHdr(),null,'no decoded image at construction');
  await flush();const source=h.photo.completedHdr();
  assert.equal(source.id,`${A}:frame-000000.npz`);assert.equal(source.complete,true);assert.equal(source.profiled,true);
  assert.equal(source.whiteReference,7);assert.match(source.url,/digest=a{64}&name=frame-000000\.npz/);
  source.params.red=123;assert.equal(h.photo.completedHdr().params.red,1,'getter cannot mutate controller state');
  assert.ok(changes.some(value=>value?.id===source.id));stop();
  h.photo.stale();assert.equal(h.photo.completedHdr(),null);
}, complete({color_profile:profile,display_white:5,display_transform:{effective_white:7}}));
check('unprofiled and incomplete exposures cannot be sampled as completed HDR', async h=>{
  await flush();assert.equal(h.photo.completedHdr(),null);
});
check('HDR getter waits for actual image decode rather than status or download link', async h=>{
  h.delayImages=true;await flush();assert.equal(h.photo.completedHdr(),null);
  for(const image of h.images)await image.onload?.();await flush();assert.ok(h.photo.completedHdr());
}, complete({color_profile:profile,display_white:1}));
check('spot proposal requires exact current source and sends only one existing retouch', async h=>{
  await flush();const source=h.photo.completedHdr(),binding={sourceId:source.id,sourceRevision:source.revision};
  assert.throws(()=>h.photo.applySpotAdjustment({params:{exposure:1}},{...binding,sourceRevision:'stale'}),/changed/);
  assert.throws(()=>h.photo.applySpotAdjustment({params:{exposure:7}},binding),/bounded/);
  assert.throws(()=>h.photo.applySpotAdjustment({params:{toString:1}},binding),/bounded/);
  assert.equal(h.retouches().length,0,'sampling/getters and rejected proposals never POST');
  h.status=complete({color_profile:profile,display_white:1,retouched_image:'preview-900321.png'});
  h.photo.applySpotAdjustment({params:{red:1.5,green:.75,blue:1,whitebalance:0,tint:0}},binding);
  assert.equal(h.photo.completedHdr(),null,'in-flight correction cannot be sampled with stale metadata');
  await flush();assert.equal(h.retouches().length,1);assert.equal(h.retouches()[0].data.red,1.5);
  assert.equal(h.element('retouch-red').value,'150');assert.ok(h.photo.completedHdr());
  assert.throws(()=>h.photo.applySpotAdjustment({params:{exposure:1}},binding),/changed/);
}, complete({color_profile:profile,display_white:1}));
check('HDR getter rejects non-final orbit selection and playback', async h=>{
  await flush();assert.ok(h.photo.completedHdr());
  h.element('photo-frames').children[0].fire('click');await flush();assert.equal(h.photo.completedHdr(),null);
  h.element('photo-frames').children[1].fire('click');await flush();assert.ok(h.photo.completedHdr());
  h.element('photo-play').fire('click');assert.equal(h.photo.completedHdr(),null);
  h.element('photo-play').fire('click');assert.ok(h.photo.completedHdr());
}, complete({color_profile:profile,display_white:1,frames:[
  {index:0,image:'frame-old.png',hdr:'frame-old.npz'},
  {index:1,image:'frame-000000.png',hdr:'frame-000000.npz'}]}));

let failures = 0;
for (const [name, body, initial] of cases) {
  const h = harness(initial);
  try { await body(h); console.log(`PASS ${name}`); }
  catch (error) { failures++; console.error(`FAIL ${name}\n${error.stack}`); }
  finally { h.restore(); }
}
if (failures) process.exitCode = 1;

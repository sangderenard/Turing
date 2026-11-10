// Production component, DOM construction and native event dispatch; DOM/canvas,
// fetch and clock boundaries are test doubles. No server, browser or GPU render.
// Run: node tests/meowdoku_film_science_check.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createFilmResponseControls} from '../examples/meowdoku/web/marble_film_response.js';
import {displayHueHistogram} from '../examples/meowdoku/web/marble_film_science.js';

const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };
const clone = value => JSON.parse(JSON.stringify(value));
// Captured from the actual CPU preview_film(default response) pipeline, not a
// handwritten schema. This catches contract mismatches such as patch-name case.
const actual = JSON.parse(readFileSync(new URL('./fixtures/marble_film_preview_default.json', import.meta.url), 'utf8'));
const actualManualLinear = JSON.parse(readFileSync(new URL('./fixtures/marble_film_preview_manual_linear.json', import.meta.url), 'utf8'));
const actualSpecular = JSON.parse(readFileSync(new URL('./fixtures/marble_film_preview_specular.json', import.meta.url), 'utf8'));
const sensor = actual.sensor_rgb, rgb8 = actual.input_display_rgb8;
const result = (id = actual.profile.id) => { const value = clone(actual); value.profile.id = id; return value; };
const reply = (body, status = 200) => ({ok: status >= 200 && status < 300, status, json: async () => body});

function harness({corrections = false, initiallyOpen = false} = {}) {
  const original = Object.fromEntries(['fetch', 'setTimeout', 'clearTimeout'].map(k => [k, globalThis[k]]));
  const timers = new Map(), requests = [], roots = [];
  let now = 0, timerId = 0;
  class Element extends EventTarget {
    constructor(tag, ownerDocument) { super(); Object.assign(this, {tagName: tag, ownerDocument, children: [], value: '', textContent: '', open: false, hidden: false, disabled: false, attrs: {}, paints: []}); }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getContext(kind) {
      assert.equal(kind, '2d', 'diagnostic canvas must not use WebGL/GPU');
      if (!this.context) this.context = {
        clearRect: (...args) => this.paints.push(['clear', ...args]),
        fillRect: (...args) => this.paints.push(['rect', this.context.fillStyle, ...args]),
        fillText: (...args) => this.paints.push(['text', ...args]),
      };
      return this.context;
    }
    fire(type) { this.dispatchEvent(new Event(type)); }
  }
  const all = node => [node, ...node.children.flatMap(all)];
  const doc = {createElement: tag => new Element(tag, doc), getElementById: id => roots.flatMap(all).find(n => n.id === id) ?? null};
  const element = (tag, id) => { const el = doc.createElement(tag); el.id = id; roots.push(el); return el; };
  const container = element('details', 'photo-film-response'); container.open = initiallyOpen;
  const transport = element('select', 'photo-transport'); transport.value = 'fixed:3';
  if (corrections) {
    for (const [key, value] of Object.entries({exposure: 0, gamma: 100, red: 100, green: 100, blue: 100, whitebalance: 0, tint: 0, intensity: 100})) element('input', `retouch-${key}`).value = String(value);
    element('select', 'retouch-normalization').value = 'auto99';
    element('input', 'retouch-normalization-white').value = '1';
    element('input', 'retouch-normalization-percentile').value = '99';
    element('select', 'retouch-tone-curve').value = 'exponential';
    for (const [key, value] of Object.entries({saturation: 100, vibrance: 0, 'black-point': 0}))
      element('input', `retouch-${key}`).value = String(value);
    element('input', 'retouch-firefly-filter').checked = false;
    element('input', 'retouch-firefly-threshold').value = '8';
    element('input', 'retouch-firefly-strength').value = '100';
    for (const id of ['retouch-reset', 'retouch-apply', 'photo-retouch']) element('button', id);
  }
  globalThis.setTimeout = (fn, ms) => { const id = ++timerId; timers.set(id, {fn, at: now + ms}); return id; };
  globalThis.clearTimeout = id => timers.delete(id);
  globalThis.fetch = (url, options) => {
    assert.equal(url, '/api/marble_photo/film_preview', 'no photo upload, status, cancel, retouch, or GPU endpoint');
    const promise = deferred(); requests.push({url, ...options, data: JSON.parse(options.body), ...promise}); return promise.promise;
  };
  const storage = {getItem: () => null, setItem() {}};
  const controls = createFilmResponseControls(container, transport, {storage});
  return {container, transport, controls, requests, doc, all, element,
    $: id => doc.getElementById(id),
    open() { container.open = true; container.fire('toggle'); },
    close() { container.open = false; container.fire('toggle'); },
    async tick(ms = 400) {
      const end = now + ms;
      while (true) {
        const next = [...timers].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        const [id, timer] = next; timers.delete(id); now = timer.at; timer.fn(); await flush();
      }
      now = end; await flush();
    },
    restore() { controls.science.dispose(); Object.assign(globalThis, original); },
  };
}

const histogram = displayHueHistogram(rgb8);
assert.equal(histogram.chromatic, 6); assert.equal(histogram.lowChroma, 2); assert.equal(histogram.dark, 0);
for (const bin of [0, 6, 12, 18, 24, 30]) assert.equal(histogram.bins[bin], 1);
assert.equal(histogram.bins.reduce((a, b) => a + b, 0), 6);
const circular = displayHueHistogram([[255,0,0],[255,0,1],[255,1,0],[0,0,0],[2,0,0],[128,127,128]]);
assert.equal(circular.bins[0], 3, 'red hues around 0/360 must share the same centred bin');
assert.equal(circular.dark, 2); assert.equal(circular.lowChroma, 1);
assert.equal(displayHueHistogram([[255,250,250]]).lowChroma, 1);
assert.equal(displayHueHistogram([[255,249,249]]).chromatic, 1);

const cases = [];
const check = (name, options, body) => cases.push([name, options, body]);

check('lazy open renders server bytes, equal patches, counts and numeric diagnostics without a photo', {}, async h => {
  assert.equal(h.requests.length, 0);
  h.open(); assert.equal(h.requests.length, 1);
  const request = h.requests[0];
  assert.equal(request.method, 'POST');
  assert.deepEqual(request.data.params, {display_look:"reference",exposure:0,gamma:1,red:1,green:1,blue:1,whitebalance:0,tint:0,intensity:1,normalization:'auto99',tone_curve:'exponential',normalization_white:1,normalization_percentile:99,saturation:1,vibrance:0,black_point:0,firefly_filter:false,firefly_threshold:8,firefly_strength:1});
  assert.deepEqual(request.data.film_response, h.controls.getState());
  const payload = result(); payload.linear_srgb = clone(sensor); payload.linear_srgb[1] = [1.2, -.1, 0]; payload.output_display_rgb8[1] = [201,12,38];
  request.resolve(reply(payload)); await flush();
  assert.match(h.$('film-preview-status').textContent, /is current/);
  assert.equal(h.$('film-preview-input').paints.filter(p => p[0] === 'rect').length, 8);
  assert.equal(h.$('film-preview-output').paints.filter(p => p[0] === 'rect')[1][1], 'rgb(201,12,38)', 'server display bytes are not re-tonemapped');
  const rectangles = h.$('film-preview-output').paints.filter(p => p[0] === 'rect');
  assert.ok(rectangles.every(p => p[4] === 80 && p[5] === 80));
  assert.deepEqual(rectangles.at(-1).slice(2), [240,80,80,80]);
  assert.match(h.$('film-hue-input-counts').textContent, /6\/8 chromatic.*0 dark and 2 low-chroma/);
  assert.match(h.$('film-preview-metrics').textContent, /1\/8 patches.*1\/24 channel values below 0, 1\/24 above 1/);
  assert.match(h.$('film-preview-profile').textContent, /Condition number: 1.00063/);
  assert.ok(h.all(h.container).some(n => n.textContent.includes('not spectral wavelength')));
  for (const id of ['film-hue-input', 'film-hue-output']) {
    const canvas = h.$(id);
    assert.equal(canvas.context.font, '20px sans-serif');
    const labels = canvas.paints.filter(p => p[0] === 'text');
    assert.deepEqual(labels.slice(3).map(p => p[1]), ['0°','60°','120°','180°','240°','300°']);
    for (const [, , x, y, maxWidth] of labels) {
      assert.ok(x >= 0 && x + maxWidth <= canvas.width, 'tick text must fit within the canvas');
      assert.ok(y - 20 >= 0 && y <= canvas.height, 'tick baseline must leave room for the larger font');
    }
    for (const [, , x, y, width, height] of canvas.paints.filter(p => p[0] === 'rect'))
      assert.ok(x >= 0 && x + width <= canvas.width && y >= 0 && y + height <= canvas.height);
  }
});

check('unmodified actual CPU pipeline payload paints every returned display RGB exactly', {}, async h => {
  h.open(); h.requests[0].resolve(reply(clone(actual))); await flush();
  assert.match(h.$('film-preview-status').textContent, /is current/);
  for (const [panel, key] of [['input', 'input_display_rgb8'], ['output', 'output_display_rgb8']]) {
    const paints = h.$(`film-preview-${panel}`).paints.filter(p => p[0] === 'rect');
    assert.deepEqual(paints.map(p => p[1]), actual[key].map(rgb => `rgb(${rgb.join(',')})`));
  }
  assert.match(h.$('film-preview-metrics').textContent, /7\/8 patches/);
  assert.match(h.$('film-preview-profile').textContent, /marble-fixed3-cie1931-d65-v1/);
  assert.ok(h.all(h.container).some(n => n.textContent.includes('8. Gray 20%: RGB8')));
});

check('actual specular CPU fixture paints exact clipped output and leaves input unchanged', {corrections:true}, async h => {
  h.$('retouch-normalization').value = 'specular';
  h.$('retouch-normalization-percentile').value = '50';
  h.$('retouch-tone-curve').disabled = true;
  h.open();
  assert.equal(h.requests[0].data.params.normalization_percentile, 50);
  assert.equal(h.requests[0].data.params.tone_curve, 'linear');
  h.requests[0].resolve(reply(clone(actualSpecular))); await flush();
  assert.deepEqual(actualSpecular.sensor_rgb, actual.sensor_rgb);
  assert.deepEqual(actualSpecular.linear_srgb, actual.linear_srgb);
  assert.deepEqual(actualSpecular.input_display_rgb8, actual.input_display_rgb8);
  assert.notDeepEqual(actualSpecular.output_display_rgb8, actual.output_display_rgb8);
  assert.deepEqual(h.$('film-preview-output').paints.filter(p => p[0] === 'rect').map(p => p[1]),
    actualSpecular.output_display_rgb8.map(rgb => `rgb(${rgb.join(',')})`));
  assert.match(h.$('film-preview-transform').textContent, /Effective percentile: 50.*Linear normalized luminance at or above white is clipped/);
});

check('rapid changes debounce, abort immediately, and only the newest request can paint', {}, async h => {
  h.open(); const older = h.requests[0];
  const red = h.$('film-red-centers_nm'); red.value = '620'; red.fire('input');
  assert.equal(older.signal.aborted, true);
  await h.tick(399); assert.equal(h.requests.length, 1);
  red.value = '610'; red.fire('input'); await h.tick(399); assert.equal(h.requests.length, 1);
  await h.tick(1); assert.equal(h.requests.length, 2);
  assert.equal(h.requests[1].data.film_response.centers_nm[0], 610);
  h.requests[1].resolve(reply(result('newest'))); await flush();
  older.resolve(reply(result('stale'))); await flush();
  assert.match(h.$('film-preview-profile').textContent, /newest/);
  assert.doesNotMatch(h.$('film-preview-profile').textContent, /stale/);
});

check('invalid sensitivity cancels pending work, reports errors, and preserves actual controls', {}, async h => {
  h.open(); const pending = h.requests[0];
  h.$('film-green-sigmas_oct').value = ''; h.$('film-green-sigmas_oct').fire('input');
  assert.equal(pending.signal.aborted, true); await h.tick(1000);
  assert.equal(h.requests.length, 1); assert.match(h.$('film-preview-status').textContent, /sigmas_oct/);
  pending.resolve(reply(result())); await flush();
  assert.match(h.$('film-preview-status').textContent, /sigmas_oct/);
  h.all(h.container).find(n => n.textContent === 'Restore film defaults').fire('click'); await h.tick();
  assert.equal(h.requests.length, 2); assert.equal(h.requests[1].data.film_response.sigmas_oct[1], .06);
});

check('monochrome and closing stop pending previews; reopening and RGB restoration work', {}, async h => {
  h.open(); const pending = h.requests[0];
  h.transport.value = 'fixed:1'; h.transport.fire('change');
  assert.equal(pending.signal.aborted, true); assert.equal(h.$('film-preview-refresh').disabled, true);
  h.controls.science.refresh(); await h.tick(1000); assert.equal(h.requests.length, 1);
  assert.match(h.$('film-preview-status').textContent, /inactive in monochrome/);
  h.transport.value = 'fixed:3'; h.transport.fire('change'); await h.tick(); assert.equal(h.requests.length, 2);
  assert.equal(h.$('film-preview-refresh').disabled, false);
  h.close(); assert.equal(h.requests[1].signal.aborted, true);
  h.$('film-red-centers_nm').value = '620'; h.$('film-red-centers_nm').fire('input'); await h.tick(1000);
  assert.equal(h.requests.length, 2); h.open(); assert.equal(h.requests.length, 3);
});

check('correction input, reset, Apply and programmatic sync use current DOM settings', {corrections: true}, async h => {
  h.open(); h.requests[0].resolve(reply(result())); await flush();
  h.$('retouch-red').value = '180'; h.$('retouch-whitebalance').value = '-50'; h.$('retouch-red').fire('input'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.red, 1.8); assert.equal(h.requests.at(-1).data.params.whitebalance, -.5);
  h.$('retouch-reset').addEventListener('click', () => { h.$('retouch-red').value = '100'; h.$('retouch-gamma').value = '125'; });
  h.$('retouch-reset').fire('click'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.red, 1); assert.equal(h.requests.at(-1).data.params.gamma, 1.25);
  h.$('retouch-tint').value = '75'; h.$('photo-retouch').fire('retouch-controls-change'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.tint, .75);
  h.$('retouch-intensity').value = '140'; h.$('retouch-apply').fire('click'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.intensity, 1.4);
  const count = h.requests.length; h.$('retouch-red').value = ''; h.$('retouch-red').fire('input'); await h.tick();
  assert.equal(h.requests.length, count); assert.match(h.$('film-preview-status').textContent, /Invalid current correction: red/);
});

check('profile normalization and tone selectors send raw white, caption server recipe, and validate choices', {corrections: true}, async h => {
  h.open();
  h.$('retouch-normalization').value = 'manual'; h.$('retouch-normalization').fire('change');
  h.$('retouch-normalization-white').value = '2.5'; h.$('retouch-normalization-white').fire('input');
  h.$('retouch-tone-curve').value = 'linear'; h.$('retouch-tone-curve').fire('change');
  await h.tick();
  assert.equal(h.requests.at(-1).data.params.normalization, 'manual');
  assert.equal(h.requests.at(-1).data.params.normalization_white, 2.5, 'normalization white must not be scaled by 100');
  assert.equal(h.requests.at(-1).data.params.tone_curve, 'linear');
  const payload = clone(actualManualLinear);
  h.requests.at(-1).resolve(reply(payload)); await flush();
  assert.match(h.$('film-preview-transform').textContent, /effective white 2.5.*Tone linear:.*sRGB once/);
  assert.ok(h.$('film-preview-transform').textContent.includes(payload.display_transform.tone_formula), 'caption must preserve the actual server formula, including its clipping bounds');
  assert.deepEqual(h.$('film-preview-output').paints.filter(p => p[0] === 'rect').map(p => p[1]), payload.output_display_rgb8.map(rgb => `rgb(${rgb.join(',')})`));
  for (const [id, bad, error] of [
    ['retouch-normalization-white', '', 'normalization_white'],
    ['retouch-normalization-white', '0.0000001', 'normalization_white'],
    ['retouch-normalization-white', '1000000000001', 'normalization_white'],
    ['retouch-normalization', 'bogus', 'normalization'],
    ['retouch-tone-curve', 'bogus', 'tone_curve'],
  ]) {
    h.$('retouch-normalization').value = 'manual'; h.$('retouch-normalization-white').value = '1'; h.$('retouch-tone-curve').value = 'linear';
    const before = h.requests.length; h.$(id).value = bad; h.$(id).fire('change'); await h.tick();
    assert.equal(h.requests.length, before); assert.match(h.$('film-preview-status').textContent, new RegExp(error));
  }
  h.$('retouch-tone-curve').value = 'linear';
  for (const white of ['0.000001', '1000000000000']) {
    h.$('retouch-normalization-white').value = white; h.$('retouch-normalization-white').fire('change'); await h.tick();
    assert.equal(h.requests.at(-1).data.params.normalization_white, Number(white));
  }
});

check('percentile and specular preview share effective display parameters and server metadata', {corrections: true}, async h => {
  h.open();
  h.$('retouch-normalization').value = 'percentile'; h.$('retouch-normalization').fire('change');
  h.$('retouch-normalization-percentile').value = '96.3'; h.$('retouch-normalization-percentile').fire('input');
  await h.tick();
  assert.equal(h.requests.at(-1).data.params.normalization_percentile, 96.3);
  assert.equal(h.requests.at(-1).data.params.tone_curve, 'exponential');
  h.$('retouch-normalization').value = 'specular'; h.$('retouch-normalization').fire('change');
  h.$('retouch-tone-curve').disabled = true;
  h.$('retouch-normalization-percentile').value = '93.8'; h.$('retouch-normalization-percentile').fire('input');
  await h.tick();
  assert.equal(h.requests.at(-1).data.params.normalization_percentile, 93.8);
  assert.equal(h.requests.at(-1).data.params.tone_curve, 'linear', 'specular forces linear even when tone selector is disabled');
  const payload = result();
  Object.assign(payload.display_transform, {normalization:'specular', normalization_percentile:93.8,
    effective_percentile:93.8, effective_white:4.6, tone_curve:'linear', specular_clip_mode:true,
    highlight_policy:'Clip only luminance above selected white; raw HDR unchanged.'});
  h.requests.at(-1).resolve(reply(payload)); await flush();
  assert.match(h.$('film-preview-transform').textContent, /Effective percentile: 93.8.*Clip only luminance above selected white/);
  h.$('retouch-normalization-percentile').disabled = true; h.controls.science.refresh();
  assert.equal(h.requests.at(-1).data.params.normalization_percentile, 95, 'mode-only specular defaults match backend');
  h.$('retouch-normalization-percentile').disabled = false;
  for (const value of ['', 'NaN', 'Infinity', '.99', '100.01']) {
    const before = h.requests.length;
    h.$('retouch-normalization-percentile').value = value; h.$('retouch-normalization-percentile').fire('input'); await h.tick();
    assert.equal(h.requests.length, before); assert.match(h.$('film-preview-status').textContent, /normalization_percentile/);
  }
  for (const value of ['1', '100']) {
    h.$('retouch-normalization-percentile').value = value; h.$('retouch-normalization-percentile').fire('input'); await h.tick();
    assert.equal(h.requests.at(-1).data.params.normalization_percentile, Number(value));
  }
});

check('disabled legacy profile controls use direct-preview defaults, not retained unsupported settings', {corrections: true}, async h => {
  for (const id of ['retouch-normalization', 'retouch-normalization-white', 'retouch-normalization-percentile', 'retouch-tone-curve',
    'retouch-saturation', 'retouch-vibrance', 'retouch-black-point',
    'retouch-firefly-filter', 'retouch-firefly-threshold', 'retouch-firefly-strength']) {
    h.$(id).value = 'not-valid-legacy'; h.$(id).disabled = true;
  }
  h.open();
  const params = h.requests[0].data.params;
  assert.equal(params.normalization, 'auto99'); assert.equal(params.normalization_white, 1); assert.equal(params.tone_curve, 'exponential');
  assert.deepEqual([params.firefly_filter, params.firefly_threshold, params.firefly_strength], [false, 8, 1]);
  assert.equal(params.saturation, 1); assert.equal(params.vibrance, 0); assert.equal(params.black_point, 0);
  h.$('retouch-normalization').disabled = false; h.$('retouch-normalization').value = 'manual';
  h.$('retouch-normalization-white').disabled = false; h.$('retouch-normalization-white').value = '4';
  h.$('retouch-tone-curve').disabled = false; h.$('retouch-tone-curve').value = 'linear';
  h.$('retouch-reset').addEventListener('click', () => {
    h.$('retouch-normalization').value = 'auto99'; h.$('retouch-normalization-white').value = '1'; h.$('retouch-tone-curve').value = 'exponential';
  });
  h.$('retouch-reset').fire('click'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.normalization, 'auto99');
  assert.equal(h.requests.at(-1).data.params.normalization_white, 1);
  assert.equal(h.requests.at(-1).data.params.tone_curve, 'exponential');
});

check('artistic edits forward current controls and display actual server recipes without browser colour math', {corrections: true}, async h => {
  h.open(); const old = h.requests[0];
  h.$('retouch-saturation').value = '180'; h.$('retouch-saturation').fire('input');
  h.$('retouch-vibrance').value = '-25'; h.$('retouch-vibrance').fire('input');
  h.$('retouch-black-point').value = '15'; h.$('retouch-black-point').fire('input');
  assert.equal(old.signal.aborted, true);
  await h.tick(399); assert.equal(h.requests.length, 1);
  await h.tick(1); assert.equal(h.requests.length, 2);
  const request = h.requests.at(-1), params = request.data.params;
  assert.equal(params.saturation, 1.8); assert.equal(params.vibrance, -.25); assert.equal(params.black_point, .15);
  const payload = result('creative');
  Object.assign(payload.display_transform, {saturation: 1.8, vibrance: -.25, black_point: .15,
    black_point_formula: 'server black point formula', chroma_formula: 'server chroma formula',
    creative_controls: 'Artistic controls, not calibrated or perceptually uniform.'});
  payload.output_display_rgb8[1] = [93, 21, 56];
  request.resolve(reply(payload)); await flush();
  assert.match(h.$('film-preview-transform').textContent, /Saturation 1\.8; vibrance -0\.25; black point 0\.15 of normalized white/);
  for (const key of ['black_point_formula', 'chroma_formula', 'creative_controls'])
    assert.ok(h.$('film-preview-transform').textContent.includes(payload.display_transform[key]));
  assert.equal(h.$('film-preview-output').paints.filter(p => p[0] === 'rect')[1][1], 'rgb(93,21,56)');
  old.resolve(reply(result('obsolete'))); await flush();
  assert.match(h.$('film-preview-profile').textContent, /creative/);
  h.$('retouch-reset').addEventListener('click', () => {
    h.$('retouch-saturation').value = '100'; h.$('retouch-vibrance').value = '0'; h.$('retouch-black-point').value = '0';
  });
  h.$('retouch-reset').fire('click'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.saturation, 1);
  assert.equal(h.requests.at(-1).data.params.vibrance, 0);
  assert.equal(h.requests.at(-1).data.params.black_point, 0);
  h.$('retouch-vibrance').value = '40'; h.$('photo-retouch').fire('retouch-controls-change'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.vibrance, .4);
  h.$('retouch-black-point').value = '20'; h.$('retouch-apply').fire('click'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.black_point, .2);
});

check('artistic validation uses endpoint ranges and never sends blank or nonfinite controls', {corrections: true}, async h => {
  h.open();
  for (const [id, key, min, max, neutral] of [
    ['retouch-saturation', 'saturation', 0, 200, 100], ['retouch-vibrance', 'vibrance', -100, 100, 0],
    ['retouch-black-point', 'black_point', 0, 50, 0],
  ]) {
    for (const bad of ['', 'NaN', 'Infinity', min - 1, max + 1]) {
      const before = h.requests.length;
      h.$(id).value = String(bad); h.$(id).fire('input'); await h.tick();
      assert.equal(h.requests.length, before);
      assert.match(h.$('film-preview-status').textContent, new RegExp(`Invalid current correction: ${key}`));
    }
    for (const value of [min, max]) {
      h.$(id).value = String(value); h.$(id).fire('input'); await h.tick();
      assert.equal(h.requests.at(-1).data.params[key], value / 100);
    }
    h.$(id).value = String(neutral);
  }
});

check('isolated-highlight parameters use real preview, explain flat patches, and only latest settings paint', {corrections: true}, async h => {
  h.open(); const original = h.requests[0];
  assert.deepEqual([original.data.params.firefly_filter, original.data.params.firefly_threshold, original.data.params.firefly_strength], [false, 8, 1]);
  h.$('retouch-firefly-filter').checked = true; h.$('retouch-firefly-filter').fire('change');
  h.$('retouch-firefly-threshold').value = '12.5'; h.$('retouch-firefly-threshold').fire('input');
  h.$('retouch-firefly-strength').value = '35'; h.$('retouch-firefly-strength').fire('input');
  assert.equal(original.signal.aborted, true);
  await h.tick(); const newest = h.requests.at(-1);
  assert.equal(h.requests.length, 2);
  assert.deepEqual([newest.data.params.firefly_filter, newest.data.params.firefly_threshold, newest.data.params.firefly_strength], [true, 12.5, .35]);
  const payload = result('filtered-current');
  Object.assign(payload.display_transform, {firefly_filter: true, firefly_threshold: 12.5, firefly_strength: .35,
    firefly_method: 'Luminance ratio to eight neighbours, after gains and before tone',
    firefly_warning: 'May reduce genuine isolated glints or sharp highlight peaks.'});
  newest.resolve(reply(payload)); await flush();
  assert.match(h.$('film-preview-transform').textContent, /filter: on; threshold 12\.5×; strength 0\.35/);
  for (const key of ['firefly_method', 'firefly_warning']) assert.ok(h.$('film-preview-transform').textContent.includes(payload.display_transform[key]));
  assert.ok(h.all(h.container).some(n => n.textContent.includes('no spatial effect on these flat patches')));
  assert.deepEqual(h.$('film-preview-output').paints.filter(p => p[0] === 'rect').map(p => p[1]),
    payload.output_display_rgb8.map(rgb => `rgb(${rgb.join(',')})`), 'no fabricated filtered preview');
  original.resolve(reply(result('obsolete-filter'))); await flush();
  assert.match(h.$('film-preview-profile').textContent, /filtered-current/);
  h.$('retouch-firefly-filter').checked = false; h.$('retouch-firefly-filter').fire('change');
  h.$('retouch-firefly-threshold').disabled = true; h.$('retouch-firefly-strength').disabled = true;
  await h.tick();
  assert.deepEqual([h.requests.at(-1).data.params.firefly_filter, h.requests.at(-1).data.params.firefly_threshold,
    h.requests.at(-1).data.params.firefly_strength], [false, 12.5, .35], 'toggle-off retains exact selected options');
  h.$('retouch-reset').addEventListener('click', () => {
    h.$('retouch-firefly-filter').checked = false;
    h.$('retouch-firefly-threshold').value = '8'; h.$('retouch-firefly-strength').value = '100';
  });
  h.$('retouch-reset').fire('click'); await h.tick();
  assert.deepEqual([h.requests.at(-1).data.params.firefly_filter, h.requests.at(-1).data.params.firefly_threshold,
    h.requests.at(-1).data.params.firefly_strength], [false, 8, 1]);
  h.$('retouch-firefly-filter').checked = true; h.$('photo-retouch').fire('retouch-controls-change'); await h.tick();
  assert.equal(h.requests.at(-1).data.params.firefly_filter, true);
});

check('isolated-highlight options validate actual endpoint bounds without silent clamping', {corrections: true}, async h => {
  h.open(); h.$('retouch-firefly-filter').checked = true;
  for (const [id, key, min, max, neutral, scale] of [
    ['retouch-firefly-threshold', 'firefly_threshold', 2, 64, 8, 1],
    ['retouch-firefly-strength', 'firefly_strength', 0, 100, 100, 100],
  ]) {
    for (const bad of ['', 'NaN', 'Infinity', min - 1, max + 1]) {
      const before = h.requests.length;
      h.$(id).value = String(bad); h.$(id).fire('input'); await h.tick();
      assert.equal(h.requests.length, before);
      assert.match(h.$('film-preview-status').textContent, new RegExp(`Invalid current correction: ${key}`));
    }
    for (const value of [min, max]) {
      h.$(id).value = String(value); h.$(id).fire('input'); await h.tick();
      assert.equal(h.requests.at(-1).data.params[key], value / scale);
    }
    h.$(id).value = String(neutral);
  }
});

check('busy, timeout, network, malformed JSON and malformed matrices show errors without substitute colours', {}, async h => {
  h.open();
  for (const status of [429, 503]) {
    h.requests.at(-1).resolve(reply({error: status === 429 ? 'busy' : 'timeout'}, status)); await flush();
    assert.match(h.$('film-preview-status').textContent, new RegExp(`HTTP ${status}`));
    h.$('film-preview-refresh').fire('click');
  }
  h.requests.at(-1).reject(Error('offline')); await flush(); assert.match(h.$('film-preview-status').textContent, /offline/);
  h.controls.science.refresh(); h.requests.at(-1).resolve({ok:true,status:200,json:async () => { throw Error('bad JSON'); }}); await flush();
  assert.match(h.$('film-preview-status').textContent, /invalid JSON/);
  for (const key of ['sensor_rgb', 'linear_srgb', 'input_display_rgb8', 'output_display_rgb8']) {
    h.controls.science.refresh(); const bad = result(); bad[key][0][1] = key.endsWith('rgb8') ? 256 : null;
    h.requests.at(-1).resolve(reply(bad)); await flush(); assert.match(h.$('film-preview-status').textContent, new RegExp(key));
  }
  assert.equal(h.$('film-preview-output').paints.length, 0, 'failures never fabricate preview colours');
  h.controls.science.refresh(); const fallback = result(); fallback.profile.neutral_fallback = true; fallback.profile.warning = 'Ill-conditioned response';
  h.requests.at(-1).resolve(reply(fallback)); await flush();
  assert.match(h.$('film-preview-profile').textContent, /Neutral fallback: yes/);
  assert.match(h.$('film-preview-warnings').textContent, /Ill-conditioned response/);
});

check('initially open details and disposal are safe', {initiallyOpen: true}, async h => {
  assert.equal(h.requests.length, 0); await h.tick(); assert.equal(h.requests.length, 1);
  h.controls.science.dispose(); assert.equal(h.requests[0].signal.aborted, true);
  const prior = h.$('film-preview-status').textContent;
  h.requests[0].resolve(reply(result())); await flush(); assert.equal(h.$('film-preview-status').textContent, prior);
  h.open(); await h.tick(); assert.equal(h.requests.length, 1);
});

let failures = 0;
for (const [name, options, body] of cases) {
  const h = harness(options);
  try { await body(h); console.log(`PASS ${name}`); }
  catch (error) { failures++; console.error(`FAIL ${name}\n${error.stack}`); }
  finally { h.restore(); }
}
if (failures) process.exitCode = 1;
else console.log('Film Science: hue accounting and all production component lifecycle checks passed.');

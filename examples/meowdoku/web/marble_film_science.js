// Display-only diagnostics. The server owns the sensor profile and tone mapping;
// this module never simulates an exposure or reconstructs a colour transform.
const PATCH_NAMES = ['White', 'Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Gray 20%'];
const CORRECTION_DEFAULTS = {exposure: 0, gamma: 1, red: 1, green: 1, blue: 1, whitebalance: 0, tint: 0, intensity: 1};
const COLOR_CORRECTIONS = {saturation: {value: 1, min: 0, max: 2},
  vibrance: {value: 0, min: -1, max: 1}, black_point: {value: 0, min: 0, max: .5}};
const FILTER_CORRECTIONS = {firefly_threshold: {value: 8, min: 2, max: 64, scale: 1},
  firefly_strength: {value: 1, min: 0, max: 1, scale: 100}};

// Centred, circular 10-degree bins: red at 359 degrees and red at 0 degrees
// belong together. Each of the eight equal-area patches contributes ONE count.
export function displayHueHistogram(rgb8) {
  const bins = Array(36).fill(0);
  let dark = 0, lowChroma = 0;
  for (const pixel of rgb8) {
    const [r, g, b] = pixel.map(v => v / 255);
    const max = Math.max(r, g, b), min = Math.min(r, g, b), chroma = max - min;
    if (max <= .01) { dark++; continue; }
    if (chroma < .02) { lowChroma++; continue; }
    let hue;
    if (max === r) hue = 60 * ((g - b) / chroma);
    else if (max === g) hue = 60 * ((b - r) / chroma + 2);
    else hue = 60 * ((r - g) / chroma + 4);
    hue = (hue % 360 + 360) % 360;
    bins[Math.floor((hue + 5) / 10) % 36]++;
  }
  return {bins, dark, lowChroma, chromatic: bins.reduce((a, b) => a + b, 0)};
}

function validatePreview(value) {
  if (!value || !Array.isArray(value.patch_names) || value.patch_names.join(',') !== PATCH_NAMES.join(','))
    throw new Error('Invalid preview response: expected the eight named test patches.');
  for (const key of ['sensor_rgb', 'linear_srgb', 'input_display_rgb8', 'output_display_rgb8']) {
    const display = key.endsWith('rgb8'), matrix = value[key];
    if (!Array.isArray(matrix) || matrix.length !== 8 || matrix.some(row => !Array.isArray(row) || row.length !== 3 ||
        row.some(v => typeof v !== 'number' || !Number.isFinite(v) || (display && (!Number.isInteger(v) || v < 0 || v > 255)))))
      throw new Error(`Invalid preview response: ${key} must contain eight finite RGB triplets${display ? ' in 0..255' : ''}.`);
  }
  return value;
}

function readCorrections(doc) {
  const params = Object.fromEntries(Object.entries(CORRECTION_DEFAULTS).map(([key, fallback]) => {
    const input = doc.getElementById?.(`retouch-${key}`);
    const value = input ? (input.value.trim() === '' ? NaN : Number(input.value) / 100) : fallback;
    if (!Number.isFinite(value)) throw new Error(`Invalid current correction: ${key}.`);
    return [key, value];
  }));
  // Old exposures disable profile-only controls. Direct synthetic preview still
  // has a profile, so use its documented defaults rather than stale legacy UI.
  for (const [id, key, choices] of [
    ['retouch-display-look', 'display_look', ['reference', 'color_priority']],
    ['retouch-normalization', 'normalization', ['auto99', 'manual', 'percentile', 'specular']],
    ['retouch-tone-curve', 'tone_curve', ['exponential', 'linear']],
  ]) {
    const input = doc.getElementById?.(id);
    const value = input && !input.disabled ? input.value : choices[0];
    if (!choices.includes(value)) throw new Error(`Invalid current correction: ${key}.`);
    params[key] = value;
  }
  if (params.normalization === 'specular') {params.tone_curve = 'linear';params.display_look='reference';}
  const percentile = doc.getElementById?.('retouch-normalization-percentile');
  params.normalization_percentile = percentile && !percentile.disabled ?
    (percentile.value.trim() === '' ? NaN : Number(percentile.value)) : params.normalization === 'specular' ? 95 : 99;
  if (!Number.isFinite(params.normalization_percentile) || params.normalization_percentile < 1 || params.normalization_percentile > 100)
    throw new Error('Invalid current correction: normalization_percentile must be between 1 and 100.');
  const white = doc.getElementById?.('retouch-normalization-white');
  params.normalization_white = white && !white.disabled ? (white.value.trim() === '' ? NaN : Number(white.value)) : 1;
  if (!Number.isFinite(params.normalization_white) || params.normalization_white < 1e-6 || params.normalization_white > 1e12)
    throw new Error('Invalid current correction: normalization_white must be between 0.000001 and 1000000000000.');
  for (const [key, {value: fallback, min, max}] of Object.entries(COLOR_CORRECTIONS)) {
    const input = doc.getElementById?.(`retouch-${key.replaceAll('_', '-')}`);
    const value = input && !input.disabled ? (input.value.trim() === '' ? NaN : Number(input.value) / 100) : fallback;
    if (!Number.isFinite(value) || value < min || value > max)
      throw new Error(`Invalid current correction: ${key} must be between ${min} and ${max}.`);
    params[key] = value;
  }
  const filter = doc.getElementById?.('retouch-firefly-filter');
  const filterAvailable = filter && !filter.disabled;
  params.firefly_filter = filterAvailable ? filter.checked === true : false;
  for (const [key, {value: fallback, min, max, scale}] of Object.entries(FILTER_CORRECTIONS)) {
    const input = doc.getElementById?.(`retouch-${key.replaceAll('_', '-')}`);
    // Disabled options retain their chosen values while the toggle is off.
    // Only an unavailable profile resets the direct-preview defaults.
    const value = filterAvailable && input ? (input.value.trim() === '' ? NaN : Number(input.value) / scale) : fallback;
    if (!Number.isFinite(value) || value < min || value > max)
      throw new Error(`Invalid current correction: ${key} must be between ${min} and ${max}.`);
    params[key] = value;
  }
  return params;
}

function rangeText(matrix) {
  return ['R', 'G', 'B'].map((name, c) => {
    const values = matrix.map(row => row[c]);
    return `${name} ${Math.min(...values).toFixed(4)}..${Math.max(...values).toFixed(4)}`;
  }).join('; ');
}

export function createFilmSciencePreview(container, transport, getResponse, {debounceMs = 400} = {}) {
  const doc = container.ownerDocument;
  const make = (tag, text, className) => {
    const node = doc.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  const root = make('section', undefined, 'film-science');
  root.setAttribute('aria-label', 'Direct film colour diagnostics');
  root.append(make('h3', 'Direct colour transfer'));
  root.append(make('p', 'Eight equal-area test patches feed intentional linear R/G/B sensor values directly into the current profile and display mapping. No exposure or GPU render is needed. Uses the current photo correction settings; existing photographs are unchanged. The isolated-highlight filter uses the same CPU pipeline, but has no spatial effect on these flat patches.'));
  const refresh = make('button', 'Refresh colour panels'); refresh.type = 'button'; refresh.id = 'film-preview-refresh';
  const status = make('p', 'Open Film Science to load the direct colour preview.');
  status.id = 'film-preview-status'; status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  const results = make('div', undefined, 'film-science-results'); results.hidden = true;
  const panels = make('div', undefined, 'film-science-panels');
  const createPanel = (name, title) => {
    const figure = make('figure');
    figure.append(make('figcaption', title));
    const canvas = make('canvas'); canvas.width = 320; canvas.height = 160;
    canvas.id = `film-preview-${name}`; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', title);
    const legend = make('ol', undefined, 'film-patch-legend');
    const range = make('p');
    const histogram = make('canvas'); histogram.width = 360; histogram.height = 150;
    histogram.id = `film-hue-${name}`; histogram.setAttribute('role', 'img');
    const counts = make('p'); counts.id = `film-hue-${name}-counts`;
    figure.append(canvas, legend, range, make('p', 'Encoded-sRGB HSV hue: patch counts (0..8)'), histogram, counts);
    panels.append(figure);
    return {canvas, legend, range, histogram, counts};
  };
  const input = createPanel('input', 'Input: intentional sensor RGB (sRGB encoding only)');
  const output = createPanel('output', 'Output: current profile + display mapping');
  const metrics = make('p'); metrics.id = 'film-preview-metrics';
  const profile = make('p'); profile.id = 'film-preview-profile';
  const warnings = make('p'); warnings.id = 'film-preview-warnings';
  const corrections = make('p'); corrections.id = 'film-preview-corrections';
  const recipe = make('p'); recipe.id = 'film-preview-transform';
  results.append(panels, metrics, profile, warnings, corrections, recipe);
  root.append(refresh, status, results, make('p', 'Hue uses encoded-sRGB HSV, not spectral wavelength or a perceptually uniform colour metric. 36 circular bins are centred every 10 degrees (0 and 360 share a bin). Dark patches (max ≤ 0.01) and low-chroma patches (max − min < 0.02) are excluded, in that order. Counts represent patches, not fabricated image pixels. The synthetic profile is not a calibrated ICC measurement.'));
  container.append(root);

  let timer = null, request = null, generation = 0, disposed = false;
  const listeners = [];
  const listen = (node, event, fn) => { if (node) { node.addEventListener(event, fn); listeners.push([node, event, fn]); } };
  const invalidate = () => {
    generation++;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    request?.abort(); request = null;
    results.hidden = true;
    root.setAttribute('aria-busy', 'false');
  };

  const drawPanel = (panel, display, linear) => {
    const ctx = panel.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D rendering is unavailable.');
    ctx.clearRect(0, 0, 320, 160);
    const items = display.map((rgb, i) => {
      ctx.fillStyle = `rgb(${rgb.join(',')})`;
      ctx.fillRect((i % 4) * 80, Math.floor(i / 4) * 80, 80, 80);
      return make('li', `${i + 1}. ${PATCH_NAMES[i]}: RGB8 ${rgb.join(', ')}`);
    });
    panel.legend.replaceChildren(...items);
    panel.canvas.setAttribute('aria-label', items.map(item => item.textContent).join('; '));
    panel.range.textContent = `Linear RGB range: ${rangeText(linear)}. Display RGB8 range: ${rangeText(display)}.`;
    const hist = displayHueHistogram(display), chart = panel.histogram.getContext('2d');
    if (!chart) throw new Error('Canvas 2D rendering is unavailable.');
    chart.clearRect(0, 0, 360, 150);
    chart.fillStyle = '#aebed0'; chart.font = '20px sans-serif';
    // Both panels have exactly the same y scale: eight patches at full height.
    for (const count of [0, 4, 8]) { chart.fillText(String(count), 1, 119 - count * 12, 16); }
    hist.bins.forEach((count, bin) => {
      chart.fillStyle = `hsl(${bin * 10}, 80%, 60%)`;
      chart.fillRect(20 + bin * 9, 116 - count * 12, 8, count * 12);
    });
    chart.fillStyle = '#aebed0';
    // The fixed 50px label bounds fit each 54px tick slot, even at this larger
    // font, and keep the last label within the 360px canvas at narrow widths.
    for (const hue of [0, 60, 120, 180, 240, 300]) chart.fillText(`${hue}°`, 20 + hue / 10 * 9, 136, 50);
    const binsText = hist.bins.map((count, i) => count ? `${i * 10}°: ${count}` : '').filter(Boolean).join('; ') || 'none';
    panel.counts.textContent = `${hist.chromatic}/8 chromatic patches; excluded ${hist.dark} dark and ${hist.lowChroma} low-chroma. Nonzero hue bins: ${binsText}.`;
    panel.histogram.setAttribute('aria-label', `Encoded-sRGB HSV hue counts. ${panel.counts.textContent}`);
  };

  const render = (data, params) => {
    drawPanel(input, data.input_display_rgb8, data.sensor_rgb);
    drawPanel(output, data.output_display_rgb8, data.linear_srgb);
    const values = data.linear_srgb.flat();
    const below = values.filter(v => v < 0).length, above = values.filter(v => v > 1).length;
    const outside = data.linear_srgb.filter(row => row.some(v => v < 0 || v > 1)).length;
    metrics.textContent = `Profiled linear output before display compression: ${outside}/8 patches outside [0, 1]; ${below}/24 channel values below 0, ${above}/24 above 1. Display white: ${Number.isFinite(data.display_white) ? data.display_white : 'unavailable'}.`;
    const meta = data.profile ?? {};
    const condition = Number.isFinite(meta.condition_number) ? meta.condition_number.toPrecision(6) : 'unavailable';
    profile.textContent = `Profile: ${typeof meta.id === 'string' ? meta.id : 'unavailable'}. Condition number: ${condition}. Neutral fallback: ${meta.neutral_fallback === true ? 'yes' : meta.neutral_fallback === false ? 'no' : 'unavailable'}.`;
    const limits = Array.isArray(meta.limits) ? meta.limits.join('; ') : typeof meta.limits === 'string' ? meta.limits : meta.limits ? JSON.stringify(meta.limits) : 'unavailable';
    warnings.textContent = `Profile warning: ${meta.warning || 'none reported'}. Limits: ${limits}.`;
    corrections.textContent = `Current corrections: ${Object.entries(params).map(([key, value]) => `${key} ${value}`).join('; ')}.`;
    const transform = data.display_transform;
    const text = value => typeof value === 'string' || (typeof value === 'number' && Number.isFinite(value)) ? value : 'unavailable';
    recipe.textContent = transform ? `Actual output recipe: normalization ${text(transform.normalization)}; selected white ${text(transform.normalization_white)}; effective white ${text(transform.effective_white)}. Effective percentile: ${transform.effective_percentile === null ? "manual white" : text(transform.effective_percentile)}. ${text(transform.highlight_policy)}. Tone ${text(transform.tone_curve)}: ${text(transform.tone_formula)}. Saturation ${text(transform.saturation)}; vibrance ${text(transform.vibrance)}; black point ${text(transform.black_point)} of normalized white. Black point: ${text(transform.black_point_formula)}. Chroma: ${text(transform.chroma_formula)}. ${text(transform.creative_controls)}. Isolated-highlight filter: ${transform.firefly_filter === true ? 'on' : transform.firefly_filter === false ? 'off' : 'unavailable'}; threshold ${text(transform.firefly_threshold)}×; strength ${text(transform.firefly_strength)}. ${text(transform.firefly_method)}. ${text(transform.firefly_warning)}. Transfer: ${text(transform.transfer)}. Gamut mapping: ${text(transform.gamut_mapping)}.` :
      'Actual output recipe metadata: unavailable.';
    results.hidden = false;
  };

  const run = async (token, response) => {
    timer = null;
    if (disposed || token !== generation || transport.value !== 'fixed:3') return;
    const controller = new AbortController(); request = controller;
    root.setAttribute('aria-busy', 'true');
    status.textContent = 'Computing direct colour transfer…';
    try {
      // Read here, after all reset/synchronisation listeners have run.
      const params = readCorrections(doc);
      const reply = await fetch('/api/marble_photo/film_preview', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({film_response: response, params}), signal: controller.signal});
      if (disposed || token !== generation) return;
      let data;
      try { data = await reply.json(); }
      catch { throw new Error(`Preview returned invalid JSON (HTTP ${reply.status}).`); }
      if (disposed || token !== generation) return;
      if (!reply.ok) throw new Error(`HTTP ${reply.status}: ${typeof data?.error === 'string' ? data.error : 'colour preview request failed'}`);
      render(validatePreview(data), params);
      status.textContent = 'Direct colour preview is current. No photo was exposed or modified.';
    } catch (error) {
      if (disposed || token !== generation) return;
      results.hidden = true;
      status.textContent = `Colour preview unavailable: ${error.message}`;
    } finally {
      if (token === generation) { request = null; root.setAttribute('aria-busy', 'false'); }
    }
  };

  const schedule = (immediate = false) => {
    if (disposed) return;
    invalidate();
    const rgb = ['fixed:3','fixed:8'].includes(transport.value); refresh.disabled = !rgb;
    if (!rgb) { status.textContent = 'Direct colour preview is inactive in monochrome. RGB settings are retained.'; return; }
    let response;
    try { response = getResponse(); }
    catch (error) { status.textContent = `Colour preview unavailable: ${error.message}`; return; }
    if (!container.open && !immediate) { status.textContent = 'Open Film Science or refresh to load the direct colour preview.'; return; }
    const token = generation;
    status.textContent = 'Colour settings changed; updating preview…';
    if (immediate) void run(token, response);
    else timer = setTimeout(() => void run(token, response), debounceMs);
  };
  listen(container, 'toggle', () => { if (container.open) schedule(true); else invalidate(); });
  listen(transport, 'change', () => schedule());
  listen(refresh, 'click', () => schedule(true));
  for (const key of Object.keys(CORRECTION_DEFAULTS)) listen(doc.getElementById?.(`retouch-${key}`), 'input', () => schedule());
  for (const key of Object.keys(COLOR_CORRECTIONS)) listen(doc.getElementById?.(`retouch-${key.replaceAll('_', '-')}`), 'input', () => schedule());
  for (const id of ['retouch-normalization', 'retouch-normalization-white', 'retouch-normalization-percentile', 'retouch-tone-curve',
    'retouch-firefly-filter', 'retouch-firefly-threshold', 'retouch-firefly-strength']) {
    for (const event of ['input', 'change']) listen(doc.getElementById?.(id), event, () => schedule());
  }
  for (const id of ['retouch-reset', 'retouch-apply']) listen(doc.getElementById?.(id), 'click', () => schedule());
  listen(doc.getElementById?.('photo-retouch'), 'retouch-controls-change', () => schedule());
  schedule();
  return {changed: () => schedule(), refresh: () => schedule(true), dispose() {
    disposed = true; invalidate();
    for (const [node, event, fn] of listeners) node.removeEventListener?.(event, fn);
  }};
}

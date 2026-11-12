// Custom Gaussian sensor response, not a calibrated RGB colour space.
// Layer order is R/G/B. Sigma is measured in log2-frequency octaves.
import {createFilmSciencePreview} from './marble_film_science.js';
export const FILM_RESPONSE_STORAGE_KEY = 'meowdoku.studio.film-response.v1';
export const defaultFilmResponse = () => ({version: 1, centers_nm: [650, 550, 450], sigmas_oct: [.06, .06, .06]});

export function normalizeFilmResponse(value = defaultFilmResponse()) {
  if (!value || Array.isArray(value) || typeof value !== 'object' || value.version !== 1 ||
      Object.keys(value).sort().join(',') !== 'centers_nm,sigmas_oct,version')
    throw new Error('Unsupported film response; expected version 1.');
  const result = {version: 1};
  for (const [key, min, max] of [['centers_nm', 380, 780], ['sigmas_oct', .01, 1]]) {
    const values = value[key];
    if (!Array.isArray(values) || values.length !== 3 ||
        values.some(v => typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max))
      throw new Error(`Film ${key} must contain three numbers between ${min} and ${max}.`);
    result[key] = [...values];
  }
  return result;
}

export function filmResponseFwhmNm(centerNm, sigmaOct) {
  const halfWidth = Math.sqrt(2 * Math.log(2)) * sigmaOct;
  return centerNm * (2 ** halfWidth - 2 ** -halfWidth);
}

export function createFilmResponseControls(container, transport, {storage} = {}) {
  const doc = container.ownerDocument;
  let science = null;
  const title = doc.createElement('h3'); title.textContent = 'Custom sensor response';
  const note = doc.createElement('p');
  note.textContent = 'Peak-normalized Gaussian R/G/B sensitivity, not a calibrated colour profile. Centres use nm; sigma uses log2-frequency octaves. Wider bands collect more neighbouring light. Edits update the direct colour panels below and the next RGB exposure, not the current photo or the Phong preview.';
  const grid = doc.createElement('div'); grid.className = 'grid';
  const status = doc.createElement('p'); status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  const reset = doc.createElement('button'); reset.type = 'button'; reset.textContent = 'Restore film defaults';
  let storageError = null;
  if (storage === undefined) {
    try { storage = doc.defaultView?.localStorage ?? globalThis.localStorage; }
    catch (error) { storageError = error; }
  }
  let initial = defaultFilmResponse();
  try {
    if (storageError || !storage) throw new Error('Browser storage is unavailable.');
    const raw = storage.getItem(FILM_RESPONSE_STORAGE_KEY);
    if (raw !== null) initial = normalizeFilmResponse(JSON.parse(raw));
    status.textContent = 'Saved in this browser; each exposure records its own response.';
  } catch (error) {
    status.textContent = `Using film defaults. ${error.message} Existing saved data was not changed.`;
  }
  const editors = ['Red', 'Green', 'Blue'].map((channel, i) => {
    const panel = doc.createElement('fieldset'); panel.className = 'wide grid';
    const legend = doc.createElement('legend'); legend.textContent = channel + ' sensor layer'; panel.append(legend);
    const fields = {};
    for (const [key, text, min, max, step] of [['centers_nm', 'Centre (nm)', 380, 780, 1], ['sigmas_oct', 'Sigma (octaves)', .01, 1, .01]]) {
      const label = doc.createElement('label'); label.textContent = text;
      const input = doc.createElement('input'); input.type = 'number'; input.min = String(min); input.max = String(max); input.step = String(step);
      input.id = `film-${channel.toLowerCase()}-${key}`; input.value = String(initial[key][i]);
      input.setAttribute('aria-label', `${channel} ${text}`);
      label.append(input); panel.append(label); fields[key] = input;
    }
    const width = doc.createElement('output'); width.className = 'wide'; panel.append(width); grid.append(panel);
    return {panel, fields, width};
  });
  const read = () => normalizeFilmResponse({version: 1,
    centers_nm: editors.map(e => e.fields.centers_nm.value.trim() === '' ? NaN : Number(e.fields.centers_nm.value)),
    sigmas_oct: editors.map(e => e.fields.sigmas_oct.value.trim() === '' ? NaN : Number(e.fields.sigmas_oct.value))});
  const showWidths = value => editors.forEach((e, i) => {
    e.width.value = `FWHM ${filmResponseFwhmNm(value.centers_nm[i], value.sigmas_oct[i]).toFixed(1)} nm`;
  });
  const save = value => {
    showWidths(value);
    try {
      if (storageError || !storage) throw new Error('Browser storage is unavailable.');
      storage.setItem(FILM_RESPONSE_STORAGE_KEY, JSON.stringify(value));
      status.textContent = 'Saved in this browser. Start a new exposure to use this response.';
    } catch (error) { status.textContent = `Response is usable for this page, but was not saved. ${error.message}`; }
  };
  for (const editor of editors) for (const field of Object.values(editor.fields)) field.addEventListener('input', () => {
    try { save(read()); } catch (error) { status.textContent = error.message; }
    science?.changed();
  });
  reset.addEventListener('click', () => {
    const value = defaultFilmResponse();
    editors.forEach((e, i) => { for (const key of Object.keys(e.fields)) e.fields[key].value = String(value[key][i]); });
    save(value);
    science?.changed();
  });
  const modeNote = doc.createElement('p');
  const syncTransport = () => {
    const rgb = transport.value === 'fixed:3' || transport.value === 'fixed:8';
    editors.forEach(e => { e.panel.disabled = !rgb; }); reset.disabled = !rgb;
    modeNote.textContent = rgb ? 'RGB transport samples fixed 650/550/450 nm lines. These controls change sensor sensitivity, not source wavelengths or dye spectra.' :
      'RGB response controls are inactive in monochrome; your RGB settings are retained.';
  };
  transport.addEventListener('change', syncTransport); syncTransport(); showWidths(initial);
  container.append(title, note, grid, reset, status, modeNote);
  science = createFilmSciencePreview(container, transport, read);
  return {getState: read, forTransport: () => ['fixed:3','fixed:8'].includes(transport.value) ? read() : undefined, science};
}

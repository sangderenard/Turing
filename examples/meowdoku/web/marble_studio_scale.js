import {TABLE_TOP_Y} from './marble_studio_table.js';
import {FOCUS_MIN_M, FOCUS_MAX_M} from './marble_lens_controls.js';

export const LEGACY_STUDIO_UNIT_M = .11 / 3.45;
export const defaultStudioScale = () => ({version: 1, mode: 'fixed', unitMm: 304.8});

// A lighting profile owns only the rig's unit, never the object's radius.
// Omitted v1 state retains the exact old photo rig and camera conversion.
export function validateStudioScale(value) {
  if (value === undefined) return {version: 1, mode: 'legacy'};
  if (!value || value.version !== 1 || !['fixed', 'legacy'].includes(value.mode))
    throw new Error('Unsupported Studio scale settings');
  if (value.mode === 'legacy') return {version: 1, mode: 'legacy'};
  if (!Number.isFinite(value.unitMm) || value.unitMm < .1 || value.unitMm > 304.8)
    throw new RangeError('Studio unit must be between 0.1 and 304.8 mm');
  return {version: 1, mode: 'fixed', unitMm: value.unitMm};
}

export function studioScaleTransform(state, radiusMm, tableEnabled) {
  state = validateStudioScale(state);
  if (!Number.isFinite(radiusMm) || radiusMm < .1 || radiusMm > 200)
    throw new RangeError('Marble radius must be between 0.1 and 304.8 mm');
  const radiusM = radiusMm / 1000;
  const unitM = state.mode === 'fixed' ? state.unitMm / 1000 : tableEnabled ? radiusM : LEGACY_STUDIO_UNIT_M;
  const subjectRadius = radiusM / unitM;
  // Table-on sets maintain bottom contact, not a centre lock. Table-off
  // retains its original origin, including legacy floor-level photographs.
  const subjectCenter = [0, tableEnabled ? TABLE_TOP_Y + subjectRadius : 0, 0];
  return {unitM, radiusM, subjectRadius, subjectCenter,
    fovDegrees: state.mode === 'legacy' && !tableEnabled ? 18 : 360 / 7};
}

export const radiusFromSlider = value => .1 * 2000 ** (Number(value) / 100);
export const sliderFromRadius = radiusMm => 100 * Math.log(radiusMm / .1) / Math.log(2000);

export function studioFrameDistance(transform, aspect = 1, minimumFocusM = FOCUS_MIN_M) {
  if (!Number.isFinite(aspect) || aspect <= 0) throw new RangeError('Invalid camera aspect ratio');
  if (!Number.isFinite(minimumFocusM) || minimumFocusM < FOCUS_MIN_M || minimumFocusM > FOCUS_MAX_M)
    throw new RangeError('Invalid lens focus bound');
  const halfFov = Math.atan(Math.tan(transform.fovDegrees * Math.PI / 360) * Math.min(1, aspect));
  let distance = Math.max(minimumFocusM / transform.unitM, transform.subjectRadius * 1.2 / Math.sin(halfFov));
  // Round outward if metres -> scene units -> metres crosses a strict lower bound.
  if (distance * transform.unitM < minimumFocusM) distance *= 1 + Number.EPSILON;
  if (distance * transform.unitM > FOCUS_MAX_M) throw new RangeError('The subject cannot fit inside the current lens 10 m focus range. Use a wider focal length.');
  return distance;
}

// Clip in scene units, keeping tiny physical marbles visible in a foot-sized shop.
export function studioCameraClip(transform, distance, bounds, target=transform.subjectCenter) {
  if(!Number.isFinite(distance)||distance<=0||!Number.isFinite(transform.subjectRadius)||transform.subjectRadius<=0)
    throw new RangeError('Invalid camera clipping geometry');
  const near=Math.max(1e-9,Math.min(.01,transform.subjectRadius*.25,Math.max(1e-9,distance-transform.subjectRadius)*.25));
  let reach=transform.subjectRadius*2;
  for(const x of [bounds.leftX,bounds.rightX])for(const y of [bounds.floorY,bounds.ceilingY])for(const z of [bounds.backZ,bounds.frontZ])
    reach=Math.max(reach,Math.hypot(x-target[0],y-target[1],z-target[2]));
  return {near,far:Math.max(30,distance+reach+1)};
}

export function createStudioScaleControls(container, {getRadiusMm, getTableEnabled, onChange}) {
  const doc = container.ownerDocument;
  let state = defaultStudioScale();
  const label = doc.createElement('label'); label.textContent = 'Studio scale mode';
  const mode = doc.createElement('select'); mode.id = 'studio-scale-mode';
  for (const [value, text] of [['fixed', 'Independent marble size'], ['legacy', 'Legacy radius-linked rig']]) {
    const option = doc.createElement('option'); option.value = value; option.textContent = text; mode.append(option);
  }
  label.append(mode);
  const unitLabel = doc.createElement('label'); unitLabel.textContent = 'One Studio unit (mm)';
  const unit = doc.createElement('input'); unit.id = 'studio-unit-mm'; unit.type = 'number';
  unit.min = '.1'; unit.max = '304.8'; unit.step = 'any'; unitLabel.append(unit);
  const status = doc.createElement('output'); status.id = 'studio-scale-status'; status.className = 'wide';
  status.setAttribute('role', 'status');
  container.append(label, unitLabel, status);
  const sync = () => {
    mode.value = state.mode; unit.disabled = state.mode === 'legacy';
    try {
      const transform = studioScaleTransform(state, getRadiusMm(), getTableEnabled());
      unit.value = String(transform.unitM * 1000);
      status.textContent = state.mode === 'legacy' ?
        'Legacy photo scale preserved; preview uses that physical size ratio. Select Independent to freeze this rig size before resizing the marble.' :
        `Marble diameter ${(2 * getRadiusMm()).toFixed(2)} mm, ${Number((2 * transform.subjectRadius).toFixed(3))} Studio units. Radius changes only the marble; room, table and camera stay fixed.`;
    } catch (error) {status.textContent = error.message;}
  };
  mode.addEventListener('change', () => {
    try {
      const unitMm = studioScaleTransform(state, getRadiusMm(), getTableEnabled()).unitM * 1000;
      state = validateStudioScale({version: 1, mode: mode.value, unitMm}); sync(); onChange();
    } catch (error) {status.textContent = error.message;}
  });
  unit.addEventListener('input', () => {
    try {
      state = validateStudioScale({version: 1, mode: 'fixed', unitMm: Number(unit.value)});
      sync(); onChange();
    } catch (error) {status.textContent = error.message;}
  });
  sync();
  return {getState: () => ({...state}), setState(value) {state = validateStudioScale(value); sync();}, sync,
    ownsControl: id => id === mode.id || id === unit.id};
}

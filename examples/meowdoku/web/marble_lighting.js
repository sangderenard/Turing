// Pose authoring only: fixture identities/defaults, power and materials belong to
// the Studio's existing fixture builder. Positions are fixed world-space units.
export const LIGHTING_POSE_VERSION = 1;
const radians = Math.PI / 180;
const fail = (message) => { throw new Error(`Lighting poses: ${message}`); };
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
function number(value, min, max, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) fail(`invalid ${name}`);
  return value;
}
function vector(value, limit, name) {
  if (!Array.isArray(value) || value.length !== 3) fail(`invalid ${name}`);
  return Array.from(value, v => number(v, -limit, limit, name));
}
function idsChecked(ids) {
  if (!Array.isArray(ids) || ids.length < 1 || ids.length > 64 ||
      Array.from(ids).some(id => typeof id !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(id)) ||
      new Set(ids).size !== ids.length) fail('invalid fixture IDs');
  return ids;
}
// The expected IDs come from the live fixture builder, never from a saved file.
// Return a fresh allowlisted record so callers cannot mutate the editor state.
export function validateLightingPoseState(state, expectedIds) {
  idsChecked(expectedIds);
  if (!object(state) || state.version !== LIGHTING_POSE_VERSION || !Array.isArray(state.poses) || state.poses.length !== expectedIds.length) fail('invalid version or fixture count');
  return {version: LIGHTING_POSE_VERSION, poses: Array.from(state.poses, (pose, index) => {
    if (!object(pose) || pose.id !== expectedIds[index]) fail('fixture ID or order does not match the studio');
    if (typeof pose.aimAtSubject !== 'boolean') fail('invalid subject lock');
    return {id: pose.id, pos: vector(pose.pos, 50, 'position'),
      yaw: number(pose.yaw, -180, 180, 'yaw'), pitch: number(pose.pitch, -90, 90, 'pitch'),
      aimAtSubject: pose.aimAtSubject};
  })};
}
export function createLightingPoseState(initialPoses) {
  if (!Array.isArray(initialPoses)) fail('fixtures must be an array');
  const ids = idsChecked(initialPoses.map(p => p?.id));
  const poses = initialPoses.map(p => {
    const d = vector(p.dir, Number.MAX_VALUE, 'direction');
    const scale = Math.max(...d.map(Math.abs));
    if (scale === 0) fail('direction must not be zero');
    const unit = d.map(v => v / scale);
    const length = Math.hypot(...unit);
    const yaw = Math.atan2(unit[0], -unit[2]) / radians;
    const pitch = Math.asin(Math.max(-1, Math.min(1, unit[1] / length))) / radians;
    return {id: p.id, pos: p.pos, yaw, pitch, aimAtSubject: p.aimAtSubject};
  });
  return validateLightingPoseState({version: LIGHTING_POSE_VERSION, poses}, ids);
}
export function resolveLightingPoses(state, expectedIds, target = [0,0,0]) {
  const subject = vector(target, Number.MAX_VALUE, 'subject target');
  return validateLightingPoseState(state, expectedIds).poses.map(p => {
    const yaw = p.yaw * radians, pitch = p.pitch * radians;
    let dir = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
    // At the subject position there is no look-at vector. Keep the manual aim,
    // including while locked, rather than emitting a zero or NaN direction.
    const delta = subject.map((v, i) => v - p.pos[i]);
    const scale = Math.max(...delta.map(Math.abs));
    if (p.aimAtSubject && scale > 0) {
      const towards = delta.map(v => v / scale);
      const length = Math.hypot(...towards);
      dir = towards.map(v => v / length);
    }
    return {...p, dir};
  });
}

export function createLightingPoseControls(container, initialPoses, onChange = () => {}, {validateChange = () => {}} = {}) {
  let state = createLightingPoseState(initialPoses);
  const ids = state.poses.map(p => p.id);
  const doc = container.ownerDocument;
  const root = doc.createElement('div'); root.className = 'grid lighting-pose-controls';
  const add = (tag, text, parent = root) => {
    const element = doc.createElement(tag);
    if (text !== undefined) element.textContent = text;
    parent.appendChild(element); return element;
  };
  const selectorLabel = add('label', 'Position / direction'); selectorLabel.className = 'wide';
  const selector = add('select', undefined, selectorLabel); selector.dataset.lightingPose = 'fixture';
  for (const p of initialPoses) { const option = add('option', p.label || p.id, selector); option.value = p.id; }
  selector.value = ids[0];
  let selected = 0;
  const controls = {};
  const status = doc.createElement('p'); status.className = 'wide'; status.setAttribute('role', 'status');
  const emit = () => onChange(resolveLightingPoses(state, ids));
  const render = () => {
    const p = state.poses[selected];
    for (const [name, fields] of Object.entries(controls)) {
      const value = 'XYZ'.includes(name) ? p.pos['XYZ'.indexOf(name)] : p[name];
      for (const field of fields) {
        // Keep the exact authored direction in state, but avoid filling a
        // small numeric input with trigonometric conversion noise.
        field.value = field.type === 'number' ? String(Number(value.toFixed('XYZ'.includes(name) ? 2 : 1))) : String(value);
        field.disabled = (name === 'yaw' || name === 'pitch') && p.aimAtSubject;
      }
    }
    lock.checked = p.aimAtSubject;
  };
  function numeric(name, label, min, max, step) {
    const wrapper = add('label', label);
    const fields = ['number', 'range'].map(type => {
      const input = add('input', undefined, wrapper);
      input.type = type; input.min = String(min); input.max = String(max); input.step = String(step);
      input.dataset.lightingPose = name;
      input.setAttribute('aria-label', `${label}${type === 'range' ? ' slider' : ''}`);
      input.addEventListener('input', () => {
        const value = input.value.trim() === '' ? NaN : Number(input.value);
        if (!Number.isFinite(value) || value < min || value > max) {
          status.textContent = `${label} must be between ${min} and ${max}.`; return;
        }
        const next=validateLightingPoseState(state,ids),p = next.poses[selected];
        if ((name === 'yaw' || name === 'pitch') && p.aimAtSubject) return;
        if ('XYZ'.includes(name)) p.pos['XYZ'.indexOf(name)] = value;
        else p[name] = value;
        try{validateChange(next,p.id);}catch(error){status.textContent=error.message;return;}
        state=next;
        for (const field of controls[name]) if (field !== input) field.value = String(value);
        status.textContent = ''; emit();
      });
      input.addEventListener('change', () => { render(); });
      return input;
    });
    controls[name] = fields;
  }
  for (const axis of 'XYZ') numeric(axis, `${axis} position`, -50, 50, 0.01);
  const lockLabel = add('label'); lockLabel.className = 'wide lighting-pose-lock';
  const lock = add('input', undefined, lockLabel); lock.type = 'checkbox'; lock.dataset.lightingPose = 'lock';
  add('span', ' Aim at subject', lockLabel);
  lock.addEventListener('change', () => { const next=validateLightingPoseState(state,ids);next.poses[selected].aimAtSubject=lock.checked;try{validateChange(next,next.poses[selected].id);}catch(error){render();status.textContent=error.message;return;}state=next;status.textContent='';render();emit(); });
  numeric('yaw', 'Manual yaw (degrees)', -180, 180, 0.1);
  numeric('pitch', 'Manual pitch (degrees)', -90, 90, 0.1);
  const help = add('p', 'World-space positions. Yaw 0 points toward -Z; positive yaw turns toward +X; positive pitch points up. Unlocking restores the saved manual aim. If a light coincides with the subject, it uses that manual aim.');
  help.className = 'wide';
  root.appendChild(status);
  selector.addEventListener('change', () => { selected = ids.indexOf(selector.value); if (selected < 0) selected = 0; status.textContent = ''; render(); });
  container.appendChild(root); render();
  return {
    getPoses: (target = [0,0,0]) => resolveLightingPoses(state, ids, target),
    getState: () => validateLightingPoseState(state, ids),
    setState(next) { const validated = validateLightingPoseState(next, ids); state = validated; status.textContent = ''; render(); emit(); },
  };
}

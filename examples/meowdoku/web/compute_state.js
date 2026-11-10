// Lossless float32 transport for the existing compute product's named state
// columns. The manifest owns the column names; this module invents no physics
// fields or replacement snapshot format for the product's recording.
function namesFor(stateNames) {
  if (!Array.isArray(stateNames) || !stateNames.length ||
      stateNames.some(name => typeof name !== "string" || !name || ["__proto__", "constructor", "prototype"].includes(name)) ||
      new Set(stateNames).size !== stateNames.length)
    throw new TypeError("A full state start requires the manifest's state column names");
  return stateNames;
}

function countFor(pad) {
  if (!Number.isSafeInteger(pad) || pad < 1) throw new RangeError("Invalid compiled state column length");
}

export function decodeStateStart(start, pad, stateNames) {
  countFor(pad);
  const names = namesFor(stateNames), packed = start.state_f32;
  if (!packed || typeof packed !== "object" || Array.isArray(packed) ||
      Object.keys(packed).length !== names.length || names.some(name => !Object.hasOwn(packed, name)))
    throw new TypeError("Initial state columns must exactly match the compiled manifest");
  if (["age_f32", "xyz", "ids", "rs"].some(key => Object.hasOwn(start, key)))
    throw new TypeError("A full state start cannot also contain a legacy particle start");
  const out = {};
  for (const name of names) {
    if (typeof packed[name] !== "string") throw new TypeError(`Initial ${name} column must be base64`);
    const raw = atob(packed[name]);
    if (raw.length !== pad * 4) throw new RangeError(`Initial ${name} column length differs from compiled particle count`);
    const bytes = Uint8Array.from(raw, c => c.charCodeAt(0)), view = new DataView(bytes.buffer);
    const column = new Float32Array(pad);
    for (let i = 0; i < pad; i++) {
      const value = view.getFloat32(i * 4, true);
      if (!Number.isFinite(value)) throw new RangeError(`Initial ${name} column contains a nonfinite value`);
      column[i] = value;
    }
    out[name] = column;
  }
  return out;
}

// The native spherical law solves spin, while its existing display recording
// integrates xyzw orientation. Keep that orientation alongside the law columns
// instead of restarting the visible marble at the identity quaternion.
export function decodeStateOrientations(start, count, columns) {
  const values = start?.orientations_xyzw;
  if (values === undefined) return [];
  if (!Array.isArray(values) || values.length !== count ||
      !["wx", "wy", "wz"].every(name => columns?.[name]?.length >= count))
    throw new RangeError("Initial orientations must match the tracked angular particle count");
  return values.map((q, id) => {
    if (!Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite) || Math.abs(Math.hypot(...q) - 1) > 1e-6)
      throw new RangeError("Initial orientations must be finite unit xyzw quaternions");
    return {frame: 0, quaternion: [...q], omega: [columns.wx[id], columns.wy[id], columns.wz[id]]};
  });
}

export function encodeStateStart(columns, pad, stateNames, orientations) {
  countFor(pad);
  const names = namesFor(stateNames);
  if (!columns || Object.keys(columns).length !== names.length || names.some(name => !Object.hasOwn(columns, name)))
    throw new TypeError("State columns must exactly match the compiled manifest");
  const count = columns[names[0]]?.length;
  if (!Number.isSafeInteger(count) || count < 1 || count > pad)
    throw new RangeError("State columns exceed the compiled particle count");
  const state_f32 = {};
  for (const name of names) {
    const values = columns[name];
    if (values?.length !== count) throw new RangeError("State columns have different lengths");
    const bytes = new Uint8Array(pad * 4), view = new DataView(bytes.buffer);
    for (let i = 0; i < count; i++) {
      if (!Number.isFinite(values[i]) || !Number.isFinite(Math.fround(values[i])))
        throw new RangeError(`State ${name} column contains a nonfinite float32 value`);
      view.setFloat32(i * 4, values[i], true);
    }
    let text = "";
    for (let i = 0; i < bytes.length; i += 0x8000)
      text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    state_f32[name] = btoa(text);
  }
  const start = {state_f32};
  if (orientations !== undefined) {
    start.orientations_xyzw = orientations;
    start.orientations_xyzw = decodeStateOrientations(start, orientations.length, columns).map(row => row.quaternion);
  }
  return start;
}

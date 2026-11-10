// Saved material IDs are part of the geometry digest. Restore authored rows
// in place, never renumber them or import another session's live light rig.
// Saved fixed fixture IDs (ten sources, housing, label) never move. Shop
// emitters append after those twelve slots, including when a fixture is off.
export const STUDIO_FIXED_MATERIAL_ROWS = 12;
export const STUDIO_RUNTIME_MATERIAL_ROWS = STUDIO_FIXED_MATERIAL_ROWS + 256;
const TABLE_NAMES = ["pbr", "phong", "enamel", "texstack"];
const EXTRA_NAMES = ["coated", "paintA", "paintB"];
const LEGACY_EXTRAS = {coated:32, paintA:33, paintB:34};
const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
const exactKeys = (value, names) => record(value) && Object.keys(value).length === names.length &&
  names.every(name => Object.hasOwn(value, name));
const validId = id => Number.isSafeInteger(id) && id >= 0;
function indexIds(index) {
  if (!record(index) || !Object.keys(index).length || !Object.values(index).every(validId) ||
      new Set(Object.values(index)).size !== Object.keys(index).length)
    throw new Error("invalid material catalogue IDs");
  return Object.values(index);
}
function extraIds(extra, namedIds) {
  if (!exactKeys(extra, EXTRA_NAMES) || !Object.values(extra).every(validId) ||
      new Set(Object.values(extra)).size !== EXTRA_NAMES.length ||
      Object.values(extra).some(id => namedIds.includes(id)))
    throw new Error("invalid Studio coating/paint IDs");
  return Object.values(extra);
}

export function studioMaterialExtras(assets) {
  const namedIds = indexIds(assets.material_index);
  const first = Math.max(...namedIds) + 1;
  // Old exports have no explicit private-slot metadata.
  const extra = assets.material_extras === undefined ?
    {coated:first, paintA:first+1, paintB:first+2} : assets.material_extras;
  extraIds(extra, namedIds);
  return {...extra};
}

export function studioMaterialBase(assets, extra) {
  const namedIds = indexIds(assets.material_index);
  const privateIds = extraIds(extra, namedIds);
  if (!exactKeys(assets.materials, TABLE_NAMES)) throw new Error("missing material float table");
  let count;
  for (const table of Object.values(assets.materials)) {
    if (!record(table) || !Number.isSafeInteger(table.floats_per_record) || table.floats_per_record <= 0 ||
        !Number.isSafeInteger(table.records) || table.records <= 0 || !Array.isArray(table.data) ||
        table.data.length !== table.records * table.floats_per_record ||
        !Array.from(table.data).every(value => typeof value === "number" && Number.isFinite(Math.fround(value))))
      throw new Error("invalid material float table");
    if (count !== undefined && table.records !== count) throw new Error("material table row counts differ");
    count = table.records;
  }
  if (Math.max(...namedIds) >= count ||
      (assets.material_extras !== undefined && Math.max(...privateIds) >= count))
    throw new Error("material ID exceeds exported float tables");
  // A legacy export has 32 rows and its extras were allocated in the browser.
  // Give those extras their own space before the current fixture rows.
  return Math.max(count, ...privateIds.map(id => id + 1));
}

export function savedMaterialRows(materials, index, extra, tables) {
  const currentIds = indexIds(index), currentExtras = extraIds(extra, currentIds);
  if (!exactKeys(materials, ["index", "extra", "tables"]) && !exactKeys(materials, ["index", "extra", "tables", "authoring"])) throw new Error("invalid saved material library");
  const savedIds = indexIds(materials.index);
  extraIds(materials.extra, savedIds);
  if (Object.entries(materials.index).some(([name, id]) => !Object.hasOwn(index, name) || index[name] !== id) ||
      EXTRA_NAMES.some(name => materials.extra[name] !== extra[name]))
    throw new Error("saved material IDs differ from this catalogue; migration is required");
  const sameCatalogue = savedIds.length === currentIds.length;
  // The only older layout admitted here is the original named 0..31 catalogue
  // with the known private slots 32..34. Earlier incompatible IDs stay rejected.
  const legacy = savedIds.length === 32 && savedIds.every(id => id < 32) &&
    EXTRA_NAMES.every(name => materials.extra[name] === LEGACY_EXTRAS[name]);
  // Appended disk profiles must not invalidate saved objects. Admit only a
  // complete old prefix (including private slots), with every name/ID already
  // checked above. A missing old name or renumbered row remains an error.
  const oldRows=[...savedIds,...Object.values(materials.extra)];
  const completePrefix=oldRows.length===Math.max(...oldRows)+1 &&
    new Set(oldRows).size===oldRows.length;
  if (!sameCatalogue && !legacy && !completePrefix)
    throw new Error("saved material IDs differ from this catalogue; migration is required");
  if (!exactKeys(materials.tables, TABLE_NAMES) || !exactKeys(tables, TABLE_NAMES))
    throw new Error("missing material float table");
  const authoredIds = [...savedIds, ...currentExtras].sort((a,b) => a-b);
  const savedBase = Math.max(...authoredIds) + 1;
  const staged = {};
  let savedCount;
  for (const name of TABLE_NAMES) {
    const saved = materials.tables[name], live = tables[name], stride = live?.table?.floats_per_record;
    if (!record(saved) || !Number.isSafeInteger(stride) || stride <= 0 || saved.stride !== stride ||
        !Array.isArray(saved.data) || saved.data.length % stride ||
        !live.data || live.data.length % stride || live.data.length < (Math.max(...currentIds, ...currentExtras)+1)*stride)
      throw new Error("saved material layout differs from this catalogue");
    const count = saved.data.length / stride;
    // Old Studio allocated 32+9 rows, overlapping extras with three fixtures.
    // Accept the bytes as saved, but only restore named rows and extras 32..34.
    // Newer saved catalogues may still contain the former nine-row light rig.
    if ((count !== savedBase + STUDIO_RUNTIME_MATERIAL_ROWS && count !== savedBase + STUDIO_FIXED_MATERIAL_ROWS + 16 && count !== savedBase + STUDIO_FIXED_MATERIAL_ROWS + 8 && count !== savedBase + STUDIO_FIXED_MATERIAL_ROWS && count !== savedBase + 9 && !(legacy && count === 41)) ||
        (savedCount !== undefined && savedCount !== count))
      throw new Error("saved material layout differs from this catalogue");
    savedCount = count;
    if (!Array.from(saved.data).every(value => typeof value === "number" && Number.isFinite(Math.fround(value))))
      throw new Error("nonfinite saved material float");
    staged[name] = authoredIds.map(id => ({offset:id*stride,
      data:new Float32Array(saved.data.slice(id*stride, (id+1)*stride))}));
  }
  // No live table or saved snapshot was changed, even if a later table failed.
  return staged;
}

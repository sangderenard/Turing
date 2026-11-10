// Simple room finishes author the existing Advanced controls. No parallel
// rendering state: preview, live rays and photos still read those same fields.
import {MAX_BACKDROP_LAYERS, validateBackdropLayers} from "./marble_backdrop.js";
import {studioRoomBounds} from "./marble_room.js";

export const ROOM_SURFACE_FIELDS = ["floor", "wall", "ceiling"].flatMap(surface =>
  ["pattern", "material-a", "material-b"].map(suffix => `${surface}-${suffix}`));
export const STUDIO_SURFACE_PRESETS = [
  {id: "white-matte", color: "White", finish: "Matte", material: "white_plaster", label: "White plaster"},
  {id: "white-gloss", color: "White", finish: "Gloss", material: "white_gloss_paint", label: "Glossy white paint"},
  {id: "black-matte", color: "Black", finish: "Matte", material: "black_felt", label: "Black felt"},
  {id: "black-gloss", color: "Black", finish: "Gloss", material: "black_gloss_paint", label: "Glossy black paint"},
];
const clone = value => JSON.parse(JSON.stringify(value));

export function concentricBackdropLayers(scale = 3, inverted = false, floorDrop = 0, shape = "ellipse") {
  if (typeof inverted !== "boolean") throw new TypeError("Backdrop inversion must be a boolean");
  if (!["ellipse", "rectangle"].includes(shape)) throw new TypeError("Invalid concentric outline");
  const {leftX, rightX, floorY, ceilingY} = studioRoomBounds(scale, floorDrop);
  const inner = ceilingY - floorY;
  const outer = 0.98 * Math.hypot(rightX - leftX, inner);
  return Array.from({length: MAX_BACKDROP_LAYERS}, (_, i) => {
    const diameter = outer + (inner - outer) * i / (MAX_BACKDROP_LAYERS - 1);
    // The default innermost mat is white. Invert swaps every actual material.
    const white = (i % 2 === 1) !== inverted;
    return {material: white ? "white_plaster" : "black_felt", width: diameter, height: diameter,
      shape, centerY: (floorY + ceilingY) / 2};
  });
}

function validateSurfaceBackup(value, fields) {
  if (value === null) return null;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid Advanced room backup");
  return Object.fromEntries(ROOM_SURFACE_FIELDS.map(id => {
    const v = value[id];
    if (typeof v !== "string" || !Array.from(fields[id].options).some(option => option.value === v))
      throw new Error(`Unavailable Advanced room option: ${id}`);
    return [id, v];
  }));
}

function validateBackdropBackup(value, fields, materialNames) {
  if (value === null) return null;
  if (!value || !Number.isInteger(value.count) || value.count < 0 || value.count > MAX_BACKDROP_LAYERS ||
      typeof value.sweep !== "boolean" || typeof value.material !== "string" ||
      !Array.from(fields["backdrop-material"].options).some(option => option.value === value.material))
    throw new Error("Invalid Advanced backdrop backup");
  return {count: value.count, material: value.material, sweep: value.sweep,
    layers: validateBackdropLayers(value.layers, materialNames)};
}

export function validateStudioSurfaces(value, fields, materialNames, current = null) {
  // Existing v1 profiles did not have simple controls. Never infer or apply a
  // preset merely because one is now available; old controls/layers remain law.
  if (value === undefined) return {version: 1, concentric: false, inverted: false, shape: "ellipse",
    surfaceBackup: null, backdropBackup: null};
  if (!value || value.version !== 1 || typeof value.concentric !== "boolean" || typeof value.inverted !== "boolean")
    throw new Error("Invalid simple Studio settings");
  const shape = value.shape ?? "ellipse";
  if (!["ellipse", "rectangle"].includes(shape)) throw new Error("Invalid concentric outline");
  const result = {version: 1, concentric: value.concentric, inverted: value.inverted, shape,
    surfaceBackup: validateSurfaceBackup(value.surfaceBackup, fields),
    backdropBackup: validateBackdropBackup(value.backdropBackup, fields, materialNames)};
  if (result.concentric && current) {
    const expected = concentricBackdropLayers(current.controls["room-scale"], result.inverted, current.floorDrop ?? 0, result.shape);
    if (!result.backdropBackup || current.controls["backdrop-layers"] !== MAX_BACKDROP_LAYERS ||
        current.controls["backdrop-material"] !== "white_plaster" || current.controls["backdrop-sweep"] ||
        JSON.stringify(current.backdrops) !== JSON.stringify(expected))
      throw new Error("Concentric backdrop settings do not match their authored layers");
  }
  return result;
}

export function createStudioSurfaceControls(container, fields, backdrops, materialNames,
  {onChange, onRoomChange, onBackdropChange, syncBackdropVisibility, getRoomFloorDrop = () => 0}) {
  const doc = container.ownerDocument, names = new Set(materialNames);
  let state = validateStudioSurfaces(undefined, fields, materialNames);
  const ownedIds = new Set();
  const el = (tag, id, text) => {
    const node = doc.createElement(tag);
    if (id) {node.id = id; ownedIds.add(id);}
    if (text) node.textContent = text;
    return node;
  };
  const grid = el("div"); grid.className = "studio-surface-matrix";
  grid.setAttribute("role", "group"); grid.setAttribute("aria-label", "All room surfaces: colour and finish");
  grid.append(el("span"), el("strong", null, "Matte"), el("strong", null, "Gloss"));
  const buttons = [];
  for (const color of ["White", "Black"]) {
    grid.append(el("strong", null, color));
    for (const preset of STUDIO_SURFACE_PRESETS.filter(item => item.color === color)) {
      const button = el("button", `studio-surface-${preset.id}`, preset.label); button.type = "button";
      button.disabled = !names.has(preset.material);
      button.title = button.disabled ? "This finish needs an updated material table." : `Set floor, four walls and ceiling to solid ${preset.label.toLowerCase()}.`;
      button.addEventListener("click", () => {
        if (button.disabled || currentPreset()?.id === preset.id) return;
        if (!currentPreset()) state.surfaceBackup = captureSurfaces();
        for (const id of ROOM_SURFACE_FIELDS) fields[id].value = id.endsWith("pattern") ? "solid" : preset.material;
        sync(); onRoomChange(); onChange();
      });
      grid.append(button); buttons.push({button, preset});
    }
  }
  const surfaceStatus = el("p", "studio-surface-status"); surfaceStatus.setAttribute("role", "status");
  const toggleLabel = el("label"); toggleLabel.className = "studio-surface-toggle";
  const toggle = el("input", "studio-concentric-enabled"); toggle.type = "checkbox";
  toggle.disabled = !names.has("white_plaster") || !names.has("black_felt");
  toggleLabel.append(toggle, "Concentric black / white backdrop");
  const invertLabel = el("label"); invertLabel.className = "studio-surface-toggle";
  const invert = el("input", "studio-concentric-invert"); invert.type = "checkbox";
  invert.disabled = toggle.disabled;
  invertLabel.append(invert, "Invert (black centre instead of white)");
  const shapeLabel = el("label");shapeLabel.append("Concentric outline ");
  const shape = el("select", "studio-concentric-shape");
  for (const [value,text] of [["ellipse","Ovoid"],["rectangle","Rectangular"]]) {const option=el("option",null,text);option.value=value;shape.append(option);}
  shapeLabel.append(shape);
  const backdropStatus = el("p", "studio-concentric-status"); backdropStatus.setAttribute("role", "status");
  const help = el("p", null, "Six nested mats with ovoid or rectangular outlines. Use Advanced layer width and height for tall pillar patterns. The smallest spans the back wall's height; larger rings reach its corners and are naturally clipped by the room. Turning the rig off restores the previous Advanced backdrop. No preset is applied until you choose one.");
  container.append(grid, surfaceStatus, toggleLabel, invertLabel, shapeLabel, backdropStatus, help);
  const advanced = doc.getElementById("studio-surface-restore-controls") ?? container;
  const restoreSurfaces = el("button", "studio-restore-surfaces", "Restore previous Advanced room"); restoreSurfaces.type = "button";
  const restoreBackdrop = el("button", "studio-restore-backdrop", "Restore previous Advanced backdrop"); restoreBackdrop.type = "button";
  advanced.append(restoreSurfaces, restoreBackdrop);

  function captureSurfaces() { return Object.fromEntries(ROOM_SURFACE_FIELDS.map(id => [id, fields[id].value])); }
  function currentPreset() {
    return STUDIO_SURFACE_PRESETS.find(preset => ROOM_SURFACE_FIELDS.every(id =>
      fields[id].value === (id.endsWith("pattern") ? "solid" : preset.material)));
  }
  function captureBackdrop() {
    return {count: Number(fields["backdrop-layers"].value), material: fields["backdrop-material"].value,
      sweep: fields["backdrop-sweep"].checked, layers: backdrops.getState()};
  }
  function applyBackdrop(value) {
    fields["backdrop-layers"].value = String(value.count);
    fields["backdrop-material"].value = value.material;
    fields["backdrop-sweep"].checked = value.sweep;
    backdrops.setState(value.layers);
    const output = doc.getElementById("backdrop-layers-out"); if (output) output.value = String(value.count);
    syncBackdropVisibility();
  }
  function authorConcentric() {
    applyBackdrop({count: MAX_BACKDROP_LAYERS, material: "white_plaster", sweep: false,
      layers: concentricBackdropLayers(Number(fields["room-scale"].value), state.inverted, getRoomFloorDrop(), state.shape)});
  }
  function backdropChanged() { sync(); onBackdropChange(); onChange(); }
  toggle.addEventListener("change", () => {
    if (toggle.disabled || toggle.checked === state.concentric) {sync(); return;}
    if (toggle.checked) {
      state.backdropBackup = captureBackdrop(); state.concentric = true; authorConcentric();
    } else {
      state.concentric = false; applyBackdrop(state.backdropBackup);
    }
    backdropChanged();
  });
  invert.addEventListener("change", () => {
    if (invert.disabled || invert.checked === state.inverted) return;
    state.inverted = invert.checked;
    if (state.concentric) {authorConcentric(); backdropChanged();} else sync();
  });
  shape.addEventListener("change", () => {
    if (!["ellipse","rectangle"].includes(shape.value)) {sync();return;}
    state.shape=shape.value;
    if(state.concentric){authorConcentric();backdropChanged();}else sync();
  });
  restoreSurfaces.addEventListener("click", () => {
    if (!state.surfaceBackup) return;
    for (const [id, value] of Object.entries(state.surfaceBackup)) fields[id].value = value;
    state.surfaceBackup = null; sync(); onRoomChange(); onChange();
  });
  restoreBackdrop.addEventListener("click", () => {
    if (!state.backdropBackup) return;
    state.concentric = false; applyBackdrop(state.backdropBackup); state.backdropBackup = null; backdropChanged();
  });
  for (const id of ROOM_SURFACE_FIELDS) fields[id].addEventListener("change", sync);
  for (const id of ["backdrop-layers", "backdrop-material", "backdrop-sweep"])
    for (const event of ["input", "change"]) fields[id].addEventListener(event, advancedBackdropChanged);
  fields["room-scale"].addEventListener("input", () => {
    if (state.concentric) {authorConcentric(); backdropChanged();} else sync();
  });

  function advancedBackdropChanged() { state.concentric = false; sync(); }
  function sync() {
    const preset = currentPreset();
    for (const {button, preset: item} of buttons) button.setAttribute("aria-pressed", String(preset?.id === item.id));
    surfaceStatus.textContent = preset ? `All room surfaces: ${preset.label.toLowerCase()}.` : "Custom room materials. Open Advanced to edit individual surfaces.";
    if (buttons.some(({button}) => button.disabled)) surfaceStatus.textContent += " Some finishes need an updated material table.";
    toggle.checked = state.concentric; invert.checked = state.inverted; shape.value=state.shape; shape.disabled=toggle.disabled;
    const layers = concentricBackdropLayers(Number(fields["room-scale"].value), state.inverted, getRoomFloorDrop(), state.shape);
    const inner = layers.at(-1), outer = layers[0];
    backdropStatus.textContent = toggle.disabled ? "The concentric rig needs an updated material table." :
      `${state.concentric ? "Concentric rig on" : Number(fields["backdrop-layers"].value) ? "Advanced backdrop active" : "Backdrop off"}. ` +
      `Default rig: ${state.inverted ? "black" : "white"} centre, diameter ${inner.width.toFixed(2)}; outer ${outer.width.toFixed(2)}; centre height ${inner.centerY.toFixed(2)} Studio units.`;
    restoreSurfaces.disabled = !state.surfaceBackup; restoreBackdrop.disabled = !state.backdropBackup;
  }
  sync();
  return {sync, advancedBackdropChanged, ownsControl: id => ownedIds.has(id),
    roomBoundsChanged() {if (state.concentric) {authorConcentric(); onBackdropChange();} sync();},
    getState: () => clone(state),
    setState(value) {state = validateStudioSurfaces(value, fields, materialNames); sync();},
  };
}

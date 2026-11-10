// Photography furniture, separate from the saved subject and particle course.
export const TABLE_FLOOR_DROP = 3;
export const TABLE_TOP_Y = -1;
export const TABLE_THICKNESS = 0.12;
export const TABLE_HALF_WIDTH = 2.6;
export const TABLE_HALF_DEPTH = 1.5;
// Missing dimensions in older profiles retain the original slab, independently
// of new-set authoring defaults.
export const LEGACY_TABLE_DIMENSIONS = {width:5.2,length:3,height:3.02};
export const TABLE_DIMENSIONS = {
  width: {label: "Width (X)", initial: 8, min: 2, max: 20},
  length: {label: "Length (Z)", initial: 4, min: 2, max: 20},
  height: {label: "Height (floor to top)", initial: 4, min: .30, max: 10},
};
export const FLOOR_FIXTURE_INDICES = [2, 3, 4, 5];

export function defaultStudioTable() {
  return {version: 1, enabled: true, material: "marble_milk_glass", skirt: {enabled:true,material:"black_felt"},
    ...Object.fromEntries(Object.entries(TABLE_DIMENSIONS).map(([key, spec]) => [key, spec.initial]))};
}
export function validateStudioTable(value, materialNames) {
  // Earlier profiles were floor-level sets. Loading one must not insert new
  // furniture or move its authored fixtures, camera target, or saved object.
  if (value === undefined) return {...defaultStudioTable(), ...LEGACY_TABLE_DIMENSIONS, enabled: false, material: "marble_clear_glass", skirt:{enabled:false,material:"black_felt"}};
  if (!value || value.version !== 1 || typeof value.enabled !== "boolean" ||
      typeof value.material !== "string" || !new Set(materialNames).has(value.material))
    throw new Error("Invalid photography table settings");
  const dimensions = Object.fromEntries(Object.entries(TABLE_DIMENSIONS).map(([key, spec]) => {
    // Missing dimensions are the exact original slab, never a new-sized set.
    const dimension = value[key] === undefined ? LEGACY_TABLE_DIMENSIONS[key] : value[key];
    if (!Number.isFinite(dimension) || dimension < spec.min || dimension > spec.max)
      throw new RangeError(`Table ${spec.label.toLowerCase()} must be between ${spec.min} and ${spec.max} Studio units`);
    return [key, dimension];
  }));
  const skirt=value.skirt??{enabled:false,material:"black_felt"};
  if(!skirt || typeof skirt.enabled!=="boolean" || typeof skirt.material!=="string" ||
      (skirt.enabled && !new Set(materialNames).has(skirt.material)))throw new Error("Invalid table skirt settings");
  return {version: 1, enabled: value.enabled, material: value.material, ...dimensions, skirt:{enabled:skirt.enabled,material:skirt.material}};
}
// The original room floor is Y=-1.02 and the contact plane is Y=-1.
// Height edits change the floor/supports, never the marble or tabletop.
export const tableFloorDrop = state => state.enabled ? (state.height ?? LEGACY_TABLE_DIMENSIONS.height) - .02 : 0;

// Closed outward-wound cuboids. The slab and each support are independent
// closed solids meeting at their boundaries, never overlapping volumes.
function cuboid(low, high, material) {
  const p = (x,y,z) => [x ? high[0] : low[0], y ? high[1] : low[1], z ? high[2] : low[2]];
  const faces = [];
  for (const [corners, normal] of [
    [[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[-1,0,0]],
    [[[1,0,1],[1,0,0],[1,1,0],[1,1,1]],[1,0,0]],
    [[[0,0,1],[0,0,0],[1,0,0],[1,0,1]],[0,-1,0]],
    [[[0,1,0],[0,1,1],[1,1,1],[1,1,0]],[0,1,0]],
    [[[1,0,0],[0,0,0],[0,1,0],[1,1,0]],[0,0,-1]],
    [[[0,0,1],[1,0,1],[1,1,1],[0,1,1]],[0,0,1]],
  ]) {
    const [a,b,c,d] = corners.map(corner => p(...corner)), ns = [normal,normal,normal];
    faces.push([a,b,c,material,ns],[a,c,d,material,ns]);
  }
  return faces;
}
export function studioTableFaces(state, ids, floorY = -1.02 - tableFloorDrop(state), onPart) {
  const valid = validateStudioTable(state, Object.keys(ids));
  if (!valid.enabled) return [];
  if (!Number.isFinite(floorY) || floorY >= TABLE_TOP_Y - TABLE_THICKNESS)
    throw new RangeError("Table supports require a floor below the tabletop");
  const material = ids[valid.material];
  if (!Number.isInteger(material) || material < 0) throw new RangeError("Unknown table material ID");
  const underside = TABLE_TOP_Y - TABLE_THICKNESS;
  const halfWidth = valid.width / 2, halfLength = valid.length / 2;
  const faces = cuboid([-halfWidth,underside,-halfLength],
    [halfWidth,TABLE_TOP_Y,halfLength],material);
  onPart?.({id:"slab",bodyId:"table",construction:"cuboid",closed:true,convex:true,faces:faces.slice()});
  // Keep the original .25 inset and .18-square legs inside the resized slab.
  for (const [xi,x] of [-halfWidth+.25,halfWidth-.25].entries()) for (const [zi,z] of [-halfLength+.25,halfLength-.25].entries()) {
    const leg=cuboid([x-.09,floorY,z-.09],[x+.09,underside,z+.09],material);faces.push(...leg);
    onPart?.({id:`leg-${xi}-${zi}`,bodyId:"table",construction:"cuboid",closed:true,convex:true,faces:leg});
  }
  if(valid.skirt.enabled) {
    const skirtMaterial=ids[valid.skirt.material];
    if(!Number.isInteger(skirtMaterial)||skirtMaterial<0)throw new RangeError("Unknown skirt material ID");
    // Thin closed strips give the common Phong/ray surface path distinct inward
    // and outward faces, not coincident duplicated triangles. Open underneath.
    const t=.01, bottom=floorY+.02;
    const panels=[
      [[-halfWidth-t,bottom,-halfLength-t],[halfWidth+t,underside,-halfLength]],
      [[-halfWidth-t,bottom,halfLength],[halfWidth+t,underside,halfLength+t]],
      [[-halfWidth-t,bottom,-halfLength],[-halfWidth,underside,halfLength]],
      [[halfWidth,bottom,-halfLength],[halfWidth+t,underside,halfLength]],
    ];
    panels.forEach(([low,high],i)=>{
      const panel=cuboid(low,high,skirtMaterial);faces.push(...panel);
      onPart?.({id:`skirt-${i}`,bodyId:"table",construction:"cuboid",closed:true,convex:true,faces:panel});
    });
  }
  return faces;
}

export function createStudioTableControls(container, materialNames, onChange, validateChange = () => {}) {
  const doc = container.ownerDocument;
  let state = validateStudioTable(defaultStudioTable(), materialNames);
  const enabledLabel = doc.createElement("label"); enabledLabel.className = "wide";
  const enabled = doc.createElement("input"); enabled.type = "checkbox"; enabled.id = "studio-table-enabled";
  enabledLabel.append(enabled, " Photography table");
  const materialLabel = doc.createElement("label"); materialLabel.textContent = "Table material";
  const material = doc.createElement("select"); material.id = "studio-table-material";
  for (const name of [...materialNames].sort()) {
    const option = doc.createElement("option"); option.value = name;
    option.textContent = name === "marble_milk_glass" ? "Milk glass (default)" : name.replaceAll("_", " ");
    material.append(option);
  }
  materialLabel.append(material);
  container.append(enabledLabel, materialLabel);
  const skirtLabel=doc.createElement("label"),skirt=doc.createElement("input");
  skirt.type="checkbox";skirt.id="studio-table-skirt-enabled";skirtLabel.append(skirt," Fabric skirt around all four sides");
  const skirtMaterialLabel=doc.createElement("label"),skirtMaterial=doc.createElement("select");
  skirtMaterial.id="studio-table-skirt-material";skirtMaterialLabel.append("Skirt material (both sides)",skirtMaterial);
  for(const name of [...materialNames].sort()){const option=doc.createElement("option");option.value=name;option.textContent=name==="black_felt"?"Black felt (default)":name.replaceAll("_"," ");skirtMaterial.append(option);}
  container.append(skirtLabel,skirtMaterialLabel);
  const dimensionInputs = {};
  const status = doc.createElement("output"); status.className = "wide";
  status.id = "studio-table-dimension-status"; status.setAttribute("role", "status");
  for (const [key, spec] of Object.entries(TABLE_DIMENSIONS)) {
    const label = doc.createElement("label"); label.textContent = `${spec.label} (Studio units)`;
    const input = doc.createElement("input"); input.type = "number"; input.id = `studio-table-${key}`;
    input.min = String(spec.min); input.max = String(spec.max); input.step = "0.01";
    input.setAttribute("aria-describedby", "studio-table-dimension-help studio-table-dimension-status");
    label.append(input); dimensionInputs[key] = input; container.append(label);
    input.addEventListener("input", () => {
      const value = input.value.trim() === "" ? NaN : Number(input.value);
      if (!Number.isFinite(value) || value < spec.min || value > spec.max) {
        status.textContent = `${spec.label} must be between ${spec.min} and ${spec.max} Studio units.`; return;
      }
      status.textContent = "";
      if (value !== state[key]) change({...state, [key]: value});
    });
  }
  const help = doc.createElement("p"); help.className = "wide";
  help.id = "studio-table-dimension-help";
  help.textContent = "Studio units use the lighting profile scale (10 mm per unit for new rigs). New tables are 8 x 4 x 4 Studio units (80 x 40 x 40 mm at 10 mm/unit). Width is X and length is Z, centred on the Studio origin. Height is floor-to-top (default 4): the floor and floor-mounted lights move, while the tabletop stays at Y = -1 and the marble/camera stay fixed. Legs follow the table edges. Stage lights keep their authored positions; check the placement warnings after resizing. The skirt uses one shared material on both sides, runs from the underside to just above the floor, and leaves the central underlight path open. It is static geometry, not a cloth simulation. The particle course is unchanged.";
  const placement = doc.createElement("output"); placement.className = "wide";
  placement.id = "studio-table-placement-status"; placement.setAttribute("role", "status");
  container.append(status, help, placement);
  const sync = () => {
    skirt.checked=state.skirt.enabled;skirt.disabled=!state.enabled;skirtMaterial.value=state.skirt.material;skirtMaterial.disabled=!state.enabled||!state.skirt.enabled;
    enabled.checked = state.enabled; material.value = state.material; material.disabled = !state.enabled;
    for (const [key, input] of Object.entries(dimensionInputs)) {input.value = String(state[key]); input.disabled = !state.enabled;}
    status.textContent = "";
  };
  const change = next => {
    const before = state, valid = validateStudioTable(next, materialNames);
    try {validateChange(valid, before);} catch (error) {sync(); status.textContent = error.message; return;}
    state = valid; sync(); onChange(state,before);
  };
  enabled.addEventListener("change", () => change({...state, enabled: enabled.checked}));
  material.addEventListener("change", () => change({...state, material: material.value}));
  skirt.addEventListener("change",()=>change({...state,skirt:{...state.skirt,enabled:skirt.checked}}));
  skirtMaterial.addEventListener("change",()=>change({...state,skirt:{...state.skirt,material:skirtMaterial.value}}));
  sync();
  return {getState: () => ({...state,skirt:{...state.skirt}}), setState(value) {state = validateStudioTable(value, materialNames); sync();},
    ownsControl: id => id === enabled.id || id === material.id || id===skirt.id || id===skirtMaterial.id || Object.values(dimensionInputs).some(input => id === input.id)};
}

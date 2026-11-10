// The existing Studio backdrop roll, with independently sized material sheets.
// Preview and photo use these same faces; the saved marble mesh is untouched.
export const MAX_BACKDROP_LAYERS = 6;
export const MAX_BACKDROP_SIZE = 64;
export const BACKDROP_CIRCLE_SEGMENTS = 96;

export function defaultBackdropLayers() {
  return Array.from({length: MAX_BACKDROP_LAYERS}, (_, i) => ({
    material: "", width: 8 - 0.44 * i, height: 8 - 0.44 * i,
  }));
}

export function validateBackdropLayers(value, materialNames) {
  if (!Array.isArray(value) || value.length !== MAX_BACKDROP_LAYERS)
    throw new RangeError("Backdrop settings must contain six layers");
  const names = new Set(materialNames);
  return value.map((layer) => {
    if (!layer || typeof layer !== "object" || typeof layer.material !== "string" ||
        (layer.material !== "" && !names.has(layer.material)))
      throw new RangeError("Unknown backdrop layer material");
    for (const key of ["width", "height"])
      if (!Number.isFinite(layer[key]) || layer[key] < 0.4 || layer[key] > MAX_BACKDROP_SIZE)
        throw new RangeError(`Backdrop ${key} must be between 0.4 and ${MAX_BACKDROP_SIZE}`);
    if (layer.shape !== undefined && !["rectangle", "ellipse"].includes(layer.shape))
      throw new RangeError("Unknown backdrop shape");
    if (layer.centerY !== undefined && (!Number.isFinite(layer.centerY) || Math.abs(layer.centerY) > 32))
      throw new RangeError("Backdrop centre height must be between -32 and 32");
    // Optional fields keep legacy rectangular profiles byte-for-byte in shape.
    return {material: layer.material, width: layer.width, height: layer.height,
      ...(layer.shape === undefined ? {} : {shape: layer.shape}),
      ...(layer.centerY === undefined ? {} : {centerY: layer.centerY})};
  });
}

export function backdropFaces({count, material, sweep, layers, wallZ = -3.1, floorY = -1.02}, ids, onPart) {
  if (!Number.isInteger(count) || count < 0 || count > MAX_BACKDROP_LAYERS)
    throw new RangeError("Backdrop count must be between zero and six");
  if (!count) return [];
  if (!Number.isFinite(wallZ)) throw new RangeError("Backdrop mounting plane must be finite");
  if (!Number.isFinite(floorY)) throw new RangeError("Backdrop floor plane must be finite");
  const step = 0.012;
  const faces = [], n = [0, 0, 1];
  if (material === "calibration") {
    // Preserve the existing eight-patch chart as a separate global mode.
    const chartMats = ["calibration_white", "calibration_red", "calibration_green", "calibration_blue",
      "calibration_yellow", "calibration_magenta", "calibration_cyan", "calibration_gray20"];
    const z = wallZ + step;
    chartMats.forEach((name, i) => {
      const cx = (i % 4 - 1.5) * 2, cy = (0.5 - Math.floor(i / 4)) * 4;
      const a = [cx - 0.9, cy - 1.8, z], b = [cx + 0.9, cy - 1.8, z];
      const c = [cx + 0.9, cy + 1.8, z], d = [cx - 0.9, cy + 1.8, z];
      faces.push([a, b, c, ids[name], [n,n,n]], [a, c, d, ids[name], [n,n,n]]);
      onPart?.({id:`chart-${i}`,bodyId:`backdrop-chart-${i}`,construction:"planar-sheet",closed:false,convex:true,faces:faces.slice(-2)});
    });
    return faces;
  }
  const valid = validateBackdropLayers(layers, Object.keys(ids));
  if (!Number.isInteger(ids[material])) throw new RangeError("Unknown default backdrop material");
  for (let i = 0; i < count; i++) {
    const first=faces.length;
    const layer = valid[i], w = layer.width / 2, h = layer.height / 2;
    const mat = ids[layer.material || material], z = wallZ + step * (i + 1);
    const cy = layer.centerY ?? 0;
    if (layer.shape === "ellipse") {
      // Real nested opaque mats, not a screen-space texture. Both render routes
      // consume these exact triangles and the existing physical depth order.
      const rim = angle => [w * Math.cos(angle), cy + h * Math.sin(angle), z];
      for (let segment = 0; segment < BACKDROP_CIRCLE_SEGMENTS; segment++) {
        const a = segment * 2 * Math.PI / BACKDROP_CIRCLE_SEGMENTS;
        const b = (segment + 1) * 2 * Math.PI / BACKDROP_CIRCLE_SEGMENTS;
        faces.push([[0,cy,z],rim(a),rim(b),mat,[n,n,n]]);
      }
    } else {
      faces.push([[-w,cy-h,z],[w,cy-h,z],[w,cy+h,z],mat,[n,n,n]],
        [[-w,cy-h,z],[w,cy+h,z],[-w,cy+h,z],mat,[n,n,n]]);
    }
    onPart?.({id:`layer-${i}-wall`,bodyId:`backdrop-layer-${i}`,construction:"planar-sheet",closed:false,convex:true,faces:faces.slice(first)});
    if (sweep && layer.shape !== "ellipse") {
      // Keep the existing sweep proportions, now derived from this sheet's
      // authored size instead of an implicit layer-index inset.
      const sweepDepth = 1.2 - (4 - w) * 0.3;
      if (sweepDepth > 0.1) {
        const y = floorY + step * (i + 1), up = [0,1,0];
        faces.push([[-w,y,wallZ+sweepDepth],[w,y,wallZ+sweepDepth],[w,y,wallZ],mat,[up,up,up]],
          [[-w,y,wallZ+sweepDepth],[w,y,wallZ],[-w,y,wallZ],mat,[up,up,up]]);
        onPart?.({id:`layer-${i}-sweep`,bodyId:`backdrop-layer-${i}`,construction:"planar-sheet",closed:false,convex:true,faces:faces.slice(-2)});
      }
    }
  }
  return faces;
}

export function createBackdropLayerControls(container, materialNames, onChange = () => {}) {
  const doc = container.ownerDocument, names = [...materialNames].sort();
  let layers = defaultBackdropLayers();
  const editors = layers.map((layer, index) => {
    const panel = doc.createElement("fieldset"); panel.className = "wide grid";
    const legend = doc.createElement("legend"); legend.textContent = `Backdrop ${index + 1} (${index === 0 ? "back" : "toward subject"})`;
    panel.append(legend);
    const label = doc.createElement("label"); label.className = "wide"; label.textContent = "Material";
    const select = doc.createElement("select"); select.id = `backdrop-layer-${index}-material`;
    for (const name of ["", ...names]) {
      const option = doc.createElement("option"); option.value = name;
      option.textContent = name ? name.replaceAll("_", " ") : "Use default backdrop material";
      select.append(option);
    }
    select.addEventListener("change", () => { layers[index].material = select.value; onChange(); });
    label.append(select); panel.append(label);
    const shapeLabel = doc.createElement("label"); shapeLabel.textContent = "Mat shape";
    const shape = doc.createElement("select"); shape.id = `backdrop-layer-${index}-shape`;
    for (const [value, text] of [["rectangle", "Rectangle"], ["ellipse", "Ellipse / circle"]]) {
      const option = doc.createElement("option"); option.value = value; option.textContent = text; shape.append(option);
    }
    shape.value = "rectangle";
    shape.addEventListener("change", () => { layers[index].shape = shape.value; onChange(); });
    shapeLabel.append(shape); panel.append(shapeLabel);
    const sizes = {};
    for (const key of ["width", "height", "centerY"]) {
      const sizeLabel = doc.createElement("label"); sizeLabel.textContent = key === "centerY" ? "Centre height " : `Mat ${key} `;
      const output = doc.createElement("output"), input = doc.createElement("input");
      input.type = "range"; input.min = key === "centerY" ? "-32" : "0.4";
      input.max = key === "centerY" ? "32" : String(MAX_BACKDROP_SIZE); input.step = "0.01";
      input.id = `backdrop-layer-${index}-${key}`;
      input.value = String(layer[key] ?? 0); output.value = (layer[key] ?? 0).toFixed(2);
      input.addEventListener("input", () => {
        layers[index][key] = Number(input.value); output.value = layers[index][key].toFixed(2); onChange();
      });
      sizeLabel.append(output, input); panel.append(sizeLabel); sizes[key] = {input, output};
    }
    container.append(panel);
    return {panel, select, shape, sizes};
  });
  return {
    getState: () => layers.map((layer) => ({...layer})),
    setState(value) {
      layers = validateBackdropLayers(value, names);
      editors.forEach(({select, shape, sizes}, i) => {
        select.value = layers[i].material;
        shape.value = layers[i].shape ?? "rectangle";
        for (const [key, {input, output}] of Object.entries(sizes)) {
          input.value = String(layers[i][key] ?? 0); output.value = (layers[i][key] ?? 0).toFixed(2);
        }
      });
    },
    show(count, calibration = false) {
      editors.forEach(({panel}, i) => { panel.hidden = i >= count || calibration; });
      container.hidden = count === 0 || calibration;
    },
  };
}

/* Additive startup data for the preserved MechanicalCreature world.
 * Call appendMechanicalCreatureStudioBuilding immediately after parsing the
 * original model, BEFORE its existing TuringWorldRegistry constructor. The
 * unchanged native boundary/solid realization owns contacts, picking, depth
 * and shadows. Captured fixtures remain appearance geometry, not fake bodies.
 */

function prepareMechanicalCreatureStudioBuilding(packet, {
  translationM = [32, 1.544, -18.00788329297247],
  identity = "studio:marble/instance:captured-room",
  assetIdentity = "studio:marble/captured-full-scene",
  entranceWidthM = 2, entranceHeightM = 2.4,
} = {}) {
  const scene = packet?.scene, room = scene?.room, scale = scene?.frame?.unitM;
  if (packet?.schema !== "turing.studio-class-world-import.v1" || !room ||
      !Array.isArray(scene.parts) || !(scale > 0) || !Number.isFinite(scale) ||
      typeof packet.classIdentity !== "string" || !packet.classIdentity.startsWith("python:"))
    throw new TypeError("A captured Studio class-world packet is required");
  if (!Array.isArray(translationM) || translationM.length !== 3 ||
      !translationM.every(Number.isFinite) || !identity || !assetIdentity)
    throw new TypeError("Studio placement needs one finite metre translation and identity");
  for (const key of ["leftX", "rightX", "floorY", "ceilingY", "backZ", "frontZ"])
    if (!Number.isFinite(room[key])) throw new TypeError("Studio room bounds must be finite");
  const minimum = [room.leftX, room.floorY, room.backZ].map((v, i) => v * scale + translationM[i]);
  const maximum = [room.rightX, room.ceilingY, room.frontZ].map((v, i) => v * scale + translationM[i]);
  if (minimum.some((v, i) => v >= maximum[i]) || entranceWidthM <= 0 ||
      entranceHeightM <= 0 || entranceWidthM >= maximum[0] - minimum[0] ||
      entranceHeightM >= maximum[1] - minimum[1])
    throw new RangeError("Studio room or entrance has invalid dimensions");
  const appearance = {studio_asset_identity: assetIdentity};
  const common = {floor_height: .02, wall_thickness: .04, radius: 0,
    palette_role: "building-face", wall_palette_role: "building-wall", appearance,
    physics: {body: "static", collider: "boundary-shell-plus-floor", enabled: true},
    placement: {custody: "placed", elevation: minimum[1] - .02, rotation: [0, 0, 0]},
    openings: []};
  const building = {...common, identity, kind: "building", label: "Marble Studio",
    source_class_identity: packet.classIdentity,
    center: [(minimum[0] + maximum[0]) / 2, (minimum[2] + maximum[2]) / 2],
    half_extent: [(maximum[0] - minimum[0]) / 2, (maximum[2] - minimum[2]) / 2],
    height: maximum[1] - minimum[1] + .02, hierarchy_depth: 1,
    openings: [{identity: `${identity}/opening:entry`, kind: "door", side: "north",
      offset: 0, width: entranceWidthM, height: entranceHeightM}],
  };
  const boxes = [building], admitted = [], decorative = [];
  for (const part of scene.parts) {
    if (part.construction !== "cuboid" || !part.closed || !part.convex) {
      if (part.construction !== "inward-room-plane") decorative.push(part.id);
      continue;
    }
    // The original source explicitly authors cuboids. Their axis-aligned
    // corner planes are checked before using the native solid-box recipe.
    const points = part.faces?.flatMap(face => face.slice(0, 3));
    if (!points?.length || points.some(p => !Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)))
      throw new TypeError(`Invalid captured cuboid ${part.id}`);
    const low = [0, 1, 2].map(i => Math.min(...points.map(p => p[i])));
    const high = [0, 1, 2].map(i => Math.max(...points.map(p => p[i])));
    if (low.some((v, i) => v >= high[i]) ||
        points.some(p => p.some((v, i) => v !== low[i] && v !== high[i])))
      throw new TypeError(`Captured cuboid ${part.id} is not an existing axis-aligned solid`);
    const lowM = low.map((v, i) => v * scale + translationM[i]);
    const highM = high.map((v, i) => v * scale + translationM[i]);
    boxes.push({identity: `${identity}/part:${part.id}`, kind: "static-solid", label: `Studio ${part.id}`,
      parent_identity: identity, hierarchy_depth: 2, geometry_mode: "solid",
      center: [(lowM[0] + highM[0]) / 2, (lowM[2] + highM[2]) / 2],
      half_extent: [(highM[0] - lowM[0]) / 2, (highM[2] - lowM[2]) / 2],
      height: highM[1] - lowM[1], floor_height: 0, wall_thickness: 0, radius: 0,
      palette_role: "building-face", wall_palette_role: "building-wall", openings: [],
      placement: {custody: "placed", elevation: lowM[1], rotation: [0, 0, 0]},
      physics: {body: "static", collider: "solid-contact-surface", enabled: true},
      appearance: {...appearance}, studio_source_part: part.id});
    admitted.push(part.id);
  }
  return {schema: "mechanical-creature-studio-building-v1", identity, assetIdentity,
    sourceClassIdentity: packet.classIdentity, sourceRevision: packet.revision,
    translationM: [...translationM], unitM: scale,
    placement: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...translationM, 1],
    renderEntrance: {partId: "room/front", widthM: entranceWidthM, heightM: entranceHeightM}, boxes,
    proxyIdentities: boxes.map(box => box.identity),
    boundsM: {minimum, maximum},
    entrance: {sourcePart: "room/front", side: "north", widthM: entranceWidthM,
      heightM: entranceHeightM, worldCenter: [building.center[0], minimum[1], maximum[2]],
      sourceBounds: {minimum: [(room.leftX + room.rightX) / 2 - entranceWidthM / (2 * scale), room.floorY, room.frontZ],
        maximum: [(room.leftX + room.rightX) / 2 + entranceWidthM / (2 * scale), room.floorY + entranceHeightM / scale, room.frontZ]},
      derivation: "ordinary-native-boundary-opening-in-imported-copy; source-capture-unchanged"},
    contacts: {room: "native-boundary-floor-with-openings", cuboids: admitted,
      decorativeParts: decorative, triangleDynamics: false,
      ceiling: "native-shell-visual-ceiling; no-new-contact-law"},
  };
}

function appendMechanicalCreatureStudioBuilding(model, definition) {
  if (definition?.schema !== "mechanical-creature-studio-building-v1")
    throw new TypeError("Unknown Studio building definition");
  const world = model.world, specialization = world?.identity_specialization;
  const envelope = model.document_geometry?.boxes.find(box => box.kind === "world-envelope");
  if (!envelope || !Array.isArray(world?.objects) || !Array.isArray(world.object_order) ||
      !Array.isArray(world.structural_object_order) || specialization?.authority !== "authored-string-identity" ||
      !Array.isArray(specialization.objects) || !Array.isArray(specialization.semantic_parts))
    throw new TypeError("Studio requires the original authored world registry");
  const identities = new Set(world.objects.map(object => object.identity));
  const existing = world.objects.find(object => object.identity === definition.identity);
  if (existing?.extensions?.["studio.capture"]?.assetIdentity === definition.assetIdentity) return existing;
  for (const box of definition.boxes) {
    if (!box.identity || identities.has(box.identity)) throw new Error(`Studio identity collision: ${box.identity}`);
    identities.add(box.identity);
  }
  let objectId = Math.max(0, ...specialization.objects.map(row => row.runtime_id));
  let partId = Math.max(0, ...specialization.semantic_parts.map(row => row.runtime_id));
  const appended = definition.boxes.map(source => {
    const box = structuredClone(source);
    box.parent_identity ||= envelope.identity;
    box.spatial_container = envelope.identity;
    const solid = box.geometry_mode === "solid";
    const semanticParts = solid ? [{identity: `${box.identity}/surface:body`, role: "body", material_role: box.palette_role}] :
      [{identity: `${box.identity}/surface:floor`, role: "floor", material_role: box.palette_role},
        ...["south", "north", "west", "east"].map(side => ({identity: `${box.identity}/surface:${side}`,
          role: "boundary-wall", side, material_role: box.wall_palette_role})),
        {identity: `${box.identity}/surface:ceiling`, role: "ceiling", material_role: box.wall_palette_role}];
    const object = {identity: box.identity, kind: box.kind, parent: box.parent_identity,
      label: box.label, name: box.label,
      transform: {position: [box.center[0], box.placement.elevation, box.center[1]],
        yaw_deg: 0, coordinate_space: "data-world", placed_in: envelope.identity},
      form: {recipe: solid ? "solid-box" : "boundary-floor-with-openings", half_extent: [...box.half_extent],
        height: box.height, floor_height: box.floor_height, wall_thickness: box.wall_thickness,
        radius: 0, openings: structuredClone(box.openings)},
      material_bindings: solid ? {body: box.palette_role} : {floor: box.palette_role, walls: box.wall_palette_role},
      capabilities: ["inspect", "publish-mesh", "collide-static"], semantic_parts: semanticParts,
      physics: {...box.physics, collision_authority: "world-physics"},
      persistence: {authority: "captured-Studio-asset", revision: 0},
      extensions: {"abstract_ui.document_geometry": box,
        "studio.capture": {assetIdentity: definition.assetIdentity, sourceClassIdentity: definition.sourceClassIdentity,
          revision: definition.sourceRevision, sourcePart: box.studio_source_part || "room",
          contacts: definition.contacts, entrance: definition.entrance}},
      interaction: {type: "inspect", destination: box.identity},
      dependencies: [{relationship: "contained-by", target: box.parent_identity}],
    };
    const runtimeId = ++objectId;
    return {box, object, specialization: {identity: box.identity, runtime_id: runtimeId},
      parts: semanticParts.map(part => ({identity: part.identity, runtime_id: ++partId, object_runtime_id: runtimeId}))};
  });
  for (const {box, object, specialization: entry, parts} of appended) {
    model.document_geometry.boxes.push(box);
    world.objects.push(object); world.object_order.push(object.identity);
    world.structural_object_order.push(object.identity);
    specialization.objects.push(entry); specialization.semantic_parts.push(...parts);
  }
  // Existing map selection and navigation consume the authored region/building
  // list. Keep the captured source class association; do not invent members or
  // replace the MechanicalCreature interpretation already in that list.
  const regionIdentity = `${definition.identity}/source-region`;
  model.regions.push({identity: regionIdentity, kind: "region", name: "Imported Studio",
    position: {column: 0, row: model.regions.length, width: 1, height: 1},
    dependencies: [{relationship: "contained-by", target: model.identity}],
    buildings: [{identity: definition.identity, kind: "building", name: "Marble Studio",
      source_kind: "python", module: definition.sourceClassIdentity.slice(7).split(".").slice(0, -1).join("."),
      source_class_identity: definition.sourceClassIdentity, metaphor: "captured Studio room",
      position: {column: 0, row: 0, width: 1, height: 1}, rooms: [],
      dependencies: [{relationship: "contained-by", target: regionIdentity}],
      interaction: {type: "inspect", destination: definition.identity}}]});
  return appended[0].object;
}

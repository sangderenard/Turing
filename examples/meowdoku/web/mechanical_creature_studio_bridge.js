/* Optional same-origin transport around the ORIGINAL game cycle. No physics
 * clock, camera controller, worker decoder or registry is created here.
 * Complete environment batches are consumed before the original controls;
 * copied results publish after the original render/camera/lifecycle work.
 */
const mechanicalCreatureStudioBridgeState = {
  pending: null, applied: null, definition: null, sink: null, epoch: null,
  slots: [null, null], publishedSlot: -1,
};

function mechanicalCreatureStudioFreeze(value) {
  if (value && typeof value === "object") {
    for (const child of Object.values(value)) mechanicalCreatureStudioFreeze(child);
    Object.freeze(value);
  }
  return value;
}

function mechanicalCreatureStudioVector(value, size, name) {
  if (!Array.isArray(value) || value.length !== size || !value.every(Number.isFinite))
    throw new TypeError(`${name} requires ${size} finite coordinates`);
  return [...value];
}

function mechanicalCreatureStudioRegistration() {
  const definition = globalThis.mechanicalCreatureStudioBuildingDefinition;
  if (definition?.schema !== "mechanical-creature-studio-building-v1")
    throw new Error("The original world must register its captured Studio building first");
  return definition;
}

function validateMechanicalCreatureStudioEnvironment(request) {
  const initial = mechanicalCreatureStudioRegistration(), state = mechanicalCreatureStudioBridgeState;
  const definition = request.definition;
  if (typeof request.assetIdentity !== "string" || !request.assetIdentity ||
      !Number.isSafeInteger(request.sequence) || request.sequence < 1 ||
      !Number.isSafeInteger(request.revision) || request.revision < 0 ||
      definition?.schema !== initial.schema || definition.identity !== initial.identity ||
      definition.assetIdentity !== request.assetIdentity || definition.sourceRevision !== request.revision ||
      definition.sourceClassIdentity !== initial.sourceClassIdentity ||
      !Array.isArray(definition.boxes) || definition.boxes.length !== initial.boxes.length ||
      !Array.isArray(definition.proxyIdentities) || definition.proxyIdentities.length !== initial.proxyIdentities.length ||
      new Set(definition.proxyIdentities).size !== initial.proxyIdentities.length ||
      definition.proxyIdentities.some(identity => !initial.proxyIdentities.includes(identity)) ||
      definition.unitM !== initial.unitM || !Array.isArray(definition.placement) ||
      definition.placement.length !== initial.placement.length ||
      definition.placement.some((value, i) => value !== initial.placement[i]))
    throw new TypeError("Environment update must retain its registered Studio owner and exact identity set");
  if (state.applied && request.sequence <= state.applied.sequence)
    throw new Error("Stale Studio environment sequence");
  if (request.commit !== undefined && typeof request.commit !== "function") throw new TypeError("commit must be synchronous callback");
  if (request.isCurrent !== undefined && typeof request.isCurrent !== "function") throw new TypeError("isCurrent must be callback");
  const expectedAsset = state.applied?.assetIdentity || initial.assetIdentity;
  const seen = new Set(), updates = [];
  for (const source of definition.boxes) {
    const authored = initial.boxes.find(box => box.identity === source.identity);
    const box = shaderViewer.geometry.find(box => box.identity === source.identity);
    const object = model.world.objects.find(object => object.identity === source.identity);
    const runtimeId = turingWorld.objectRuntimeId(source.identity);
    if (!authored || seen.has(source.identity) || !box || !object || !(runtimeId > 0) ||
        turingWorld.objectFromRuntimeId(runtimeId) !== object ||
        box.kind !== authored.kind || source.kind !== authored.kind ||
        source.geometry_mode !== authored.geometry_mode || box.geometry_mode !== authored.geometry_mode ||
        box.appearance?.studio_asset_identity !== expectedAsset ||
        object.extensions?.["studio.capture"]?.assetIdentity !== expectedAsset ||
        source.appearance?.studio_asset_identity !== request.assetIdentity)
      throw new Error(`Studio ownership or original runtime identity disagrees: ${source.identity}`);
    seen.add(source.identity);
    const center = mechanicalCreatureStudioVector(source.center, 2, "center");
    const half_extent = mechanicalCreatureStudioVector(source.half_extent, 2, "half_extent");
    const rotation = mechanicalCreatureStudioVector(source.placement?.rotation, 3, "placement.rotation");
    if (half_extent.some(value => value <= 0) || !(source.height > 0) || !Number.isFinite(source.height) ||
        !Number.isFinite(source.placement?.elevation) || rotation[0] !== 0 || rotation[2] !== 0 ||
        ![source.floor_height, source.wall_thickness, source.radius].every(value => Number.isFinite(value) && value >= 0) ||
        source.floor_height > source.height || !Array.isArray(source.openings) ||
        source.openings.length !== authored.openings.length)
      throw new TypeError(`Invalid native Studio box dimensions: ${source.identity}`);
    const openingIds = new Set();
    for (const opening of source.openings) {
      const old = authored.openings.find(value => value.identity === opening.identity);
      if (!old || openingIds.has(opening.identity) || opening.kind !== old.kind || opening.side !== old.side ||
          !Number.isFinite(opening.offset) || !(opening.width > 0) || !Number.isFinite(opening.width) ||
          !(opening.height > 0) || !Number.isFinite(opening.height) ||
          (opening.bottom !== undefined && (!Number.isFinite(opening.bottom) || opening.bottom < 0)))
        throw new TypeError(`Studio openings must retain their original identities: ${source.identity}`);
      openingIds.add(opening.identity);
    }
    updates.push({box, object, values: {center, half_extent, height: source.height,
      floor_height: source.floor_height, wall_thickness: source.wall_thickness, radius: source.radius,
      placement: {...box.placement, elevation: source.placement.elevation, rotation},
      openings: structuredClone(source.openings),
      appearance: {...box.appearance, studio_asset_identity: request.assetIdentity}}});
  }
  return updates;
}

function applyMechanicalCreatureStudioEnvironment() {
  const state = mechanicalCreatureStudioBridgeState, request = state.pending;
  if (!request) return;
  state.pending = null;
  let saved = null, previousRevision = shaderViewer.revision;
  try {
    if (request.isCurrent && !request.isCurrent()) throw new DOMException("Studio update superseded", "AbortError");
    const updates = validateMechanicalCreatureStudioEnvironment(request);
    saved = updates.map(({box, object, values}) => ({box, object,
      values: Object.fromEntries(Object.keys(values).map(key => [key, box[key]])),
      transform: object.transform, form: object.form, capture: object.extensions["studio.capture"]}));
    for (const {box, object, values} of updates) {
      Object.assign(box, values);
      object.transform = {...object.transform, position: [box.center[0], box.placement.elevation, box.center[1]],
        rotation: [...box.placement.rotation], yaw_deg: box.placement.rotation[1]};
      object.form = {...object.form, half_extent: [...box.half_extent], height: box.height,
        floor_height: box.floor_height, wall_thickness: box.wall_thickness, radius: box.radius,
        openings: structuredClone(box.openings)};
      object.extensions["studio.capture"] = {...object.extensions["studio.capture"],
        assetIdentity: request.assetIdentity, revision: request.revision};
    }
    shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
    // The original function owns static realization, identity spans, contacts,
    // mesh publication and the existing worker's `colliders` message.
    rebuildPortableSceneMesh();
    const receipt = {buildingIdentity: request.definition.identity, assetIdentity: request.assetIdentity,
      previousAssetIdentity: state.applied?.assetIdentity || mechanicalCreatureStudioRegistration().assetIdentity,
      revision: request.revision, sequence: request.sequence, geometryRevision: shaderViewer.revision,
      colliderPublication: stateLoopRuntime.ready ? "posted-to-original-worker" : "native-scene-ready-worker-pending"};
    if (request.commit) {
      const result = request.commit(mechanicalCreatureStudioFreeze({...receipt}));
      if (result === false || (result && typeof result.then === "function"))
        throw new Error("Studio renderer commit must complete synchronously");
    }
    state.applied = mechanicalCreatureStudioFreeze(receipt);
    state.definition = request.definition;
    request.resolve(state.applied);
  } catch (error) {
    if (saved) {
      for (const old of saved) {
        Object.assign(old.box, old.values); old.object.transform = old.transform;
        old.object.form = old.form; old.object.extensions["studio.capture"] = old.capture;
      }
      shaderViewer.revision = previousRevision; model.scene_mesh.revision = previousRevision;
      try { rebuildPortableSceneMesh(); } catch (rollbackError) { reportRuntimeFault("studio-environment-rollback", rollbackError); }
    }
    request.reject(error);
    if (error.name !== "AbortError") reportRuntimeFault("studio-environment", error);
  }
}

function readMechanicalCreatureStudioCamera() {
  if (!shaderViewer.cameraPosition || !shaderViewer.cameraFacing) return null;
  return {positionM: mechanicalCreatureStudioVector(shaderViewer.cameraPosition, 3, "game camera"),
    facing: mechanicalCreatureStudioVector(shaderViewer.cameraFacing, 3, "game facing")};
}

function markMechanicalCreatureStudioColliderPublication() {
  const state = mechanicalCreatureStudioBridgeState;
  if (state.applied) state.applied = mechanicalCreatureStudioFreeze({...state.applied,
    colliderPublication: "posted-to-original-worker"});
}

function publishMechanicalCreatureStudioSnapshot(now, dt) {
  const state = mechanicalCreatureStudioBridgeState;
  if (!state.epoch) return;
  try {
    const bodies = [], seen = new Set();
    for (const [states, pickup] of [[projectileState, false], [projectilePickupState, true]]) {
      for (const [identity, body] of states) {
        if (!body.appearanceAssetIdentity) continue;
        if (seen.has(identity) || body.identity !== identity || body.box.identity !== identity || body.record.identity !== identity)
          throw new Error("Original projectile lifecycle identity disagrees");
        seen.add(identity);
        const snapshot = stateLoopRuntime.bodies.get(identity);
        if (snapshot && (snapshot.identity !== identity || stateLoopRuntime.slotRecords[snapshot.slot] !== snapshot ||
            !Number.isSafeInteger(snapshot.generation) || snapshot.generation < 1))
          throw new Error("Original projectile snapshot membership disagrees");
        if (!(body.box.radius > 0) || !Number.isFinite(body.box.radius) ||
            body.record.geometry?.radius !== body.box.radius ||
            ((body.sleeping === true || pickup) && snapshot) ||
            (!body.sleeping && !pickup && stateLoopRuntime.ready && !snapshot))
          throw new Error("Original projectile radius or lifecycle membership disagrees");
        bodies.push({identity, appearanceAssetIdentity: body.appearanceAssetIdentity,
          ammunitionItemIdentity: body.ammunitionItemIdentity, status: body.record.status,
          sleeping: body.sleeping === true, pickup,
          positionM: mechanicalCreatureStudioVector(body.position, 3, "projectile position"),
          velocityMPerS: mechanicalCreatureStudioVector(body.velocity, 3, "projectile velocity"),
          radiusM: body.box.radius, membership: snapshot ? {slot: snapshot.slot, generation: snapshot.generation} : null});
      }
    }
    const packet = mechanicalCreatureStudioFreeze({schema: "mechanical-creature-studio-state-v1",
      epoch: state.epoch, frameTimeMs: now, frameDtSeconds: dt, dtAuthority: "original-viewport-slice",
      workerSequence: stateLoopRuntime.latestSequence, workerReady: stateLoopRuntime.ready,
      geometryRevision: shaderViewer.revision, environments: state.applied ? [{...state.applied}] : [],
      bodies, camera: readMechanicalCreatureStudioCamera()});
    // Single-thread double-slot publication. Never claim SharedArrayBuffer or
    // native solver synchronization: each complete owned packet swaps at once.
    const slot = state.publishedSlot === 0 ? 1 : 0;
    state.slots[slot] = packet; state.publishedSlot = slot;
    if (state.sink) state.sink(packet);
  } catch (error) { reportRuntimeFault("studio-state-publication", error); }
}

globalThis.mechanicalCreatureStudioBridge = Object.freeze({
  prepareEnvironment(packet, {assetIdentity} = {}) {
    if (typeof assetIdentity !== "string" || !assetIdentity) throw new TypeError("Captured scene assetIdentity is required");
    const initial = mechanicalCreatureStudioRegistration();
    const definition = prepareMechanicalCreatureStudioBuilding(packet, {
      identity: initial.identity, assetIdentity, translationM: [...initial.translationM],
      entranceWidthM: initial.entrance.widthM, entranceHeightM: initial.entrance.heightM});
    return {assetIdentity, revision: definition.sourceRevision, definition,
      building: {identity: definition.identity, assetIdentity, placement: [...definition.placement],
        entrance: {...definition.renderEntrance}, proxyIds: [...definition.proxyIdentities]}};
  },
  queueEnvironment(request) {
    const state = mechanicalCreatureStudioBridgeState;
    validateMechanicalCreatureStudioEnvironment(request);
    if (state.pending && request.sequence <= state.pending.sequence) throw new Error("Stale queued Studio sequence");
    // Own a complete native-shape snapshot; caller edits after enqueue cannot
    // tear the next frame's environment. No unregistered objects are admitted.
    const owned = {...request, definition: structuredClone(request.definition)};
    const promise = new Promise((resolve, reject) => { owned.resolve = resolve; owned.reject = reject; });
    if (state.pending) state.pending.reject(new DOMException("Studio update superseded", "AbortError"));
    state.pending = owned;
    return promise;
  },
  setSink(sink, {epoch} = {}) {
    if (sink !== null && typeof sink !== "function") throw new TypeError("State sink must be function or null");
    if (sink !== null && (typeof epoch !== "string" || !epoch)) throw new TypeError("Host must supply a nonempty session epoch");
    const state = mechanicalCreatureStudioBridgeState;
    if (epoch !== state.epoch) { state.slots = [null, null]; state.publishedSlot = -1; }
    state.sink = sink; state.epoch = sink === null ? null : epoch;
  },
  readSnapshot() {
    const state = mechanicalCreatureStudioBridgeState;
    return state.publishedSlot < 0 ? null : state.slots[state.publishedSlot];
  },
  readCamera() { return readMechanicalCreatureStudioCamera(); },
});

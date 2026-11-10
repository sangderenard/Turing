/* Optional same-origin transport around the ORIGINAL game cycle. No physics
 * clock, camera controller, worker decoder or registry is created here.
 * Complete environment batches are consumed before the original controls;
 * copied results publish after the original render/camera/lifecycle work.
 */
const mechanicalCreatureStudioBridgeState = {
  pending: null, applied: null, definition: null, sink: null, epoch: null,
  slots: [null, null], publishedSlot: -1, colliderBarrier: null,
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

function validateMechanicalCreatureStudioNewPart(box, initial) {
  if (typeof box.studio_source_part !== "string" || !box.studio_source_part ||
      box.identity !== `${initial.identity}/part:${box.studio_source_part}` ||
      box.parent_identity !== initial.identity || box.kind !== "static-solid" || box.geometry_mode !== "solid" ||
      (box.source_class_identity !== undefined && box.source_class_identity !== initial.sourceClassIdentity) ||
      box.placement?.custody !== "placed" || box.physics?.body !== "static" ||
      box.physics?.collider !== "solid-contact-surface" || box.physics?.enabled !== true ||
      box.radius !== 0 || box.floor_height !== 0 || box.wall_thickness !== 0 ||
      !Array.isArray(box.openings) || box.openings.length)
    throw new TypeError(`Only the owned authored static cuboid recipe can be admitted: ${box.identity}`);
}

function readMechanicalCreatureStudioRegisteredDefinition() {
  const initial = mechanicalCreatureStudioRegistration(), state = mechanicalCreatureStudioBridgeState;
  const base = state.definition || initial, assetIdentity = state.applied?.assetIdentity || initial.assetIdentity;
  const objects = initial.proxyIdentities.map(identity => turingWorld.object(identity));
  for (const object of model.world.objects) {
    if (initial.proxyIdentities.includes(object.identity)) continue;
    const capture = object.extensions?.["studio.capture"];
    if (object.parent === initial.identity && capture?.sourceClassIdentity === initial.sourceClassIdentity &&
        typeof capture.sourcePart === "string") objects.push(object);
  }
  const boxes = objects.map(object => {
    const box = object?.extensions?.["abstract_ui.document_geometry"], capture = object?.extensions?.["studio.capture"];
    if (!box || turingWorld.object(object.identity) !== object || box.identity !== object.identity ||
        capture?.sourceClassIdentity !== initial.sourceClassIdentity ||
        capture.assetIdentity !== box.appearance?.studio_asset_identity)
      throw new Error("Registered Studio box and canonical capture ownership disagree");
    const startup = initial.boxes.find(source => source.identity === object.identity);
    if (startup) {
      const parent = startup.parent_identity || model.document_geometry.boxes.find(source => source.kind === "world-envelope")?.identity;
      if (!parent || object.parent !== parent || box.parent_identity !== parent || object.kind !== box.kind ||
          capture.sourcePart !== (startup.studio_source_part || "room"))
        throw new Error("Startup Studio parent or source-part ownership changed");
    } else {
      validateMechanicalCreatureStudioNewPart(box, initial);
      if (capture.sourcePart !== box.studio_source_part || object.kind !== box.kind)
        throw new Error("Registered Studio source part disagrees");
    }
    const descriptor = structuredClone(box);
    // These are prospective descriptors for the next capture, not an applied
    // receipt. Inactive failed admissions can retain their previous asset tag
    // on the authoritative object; normalize only this copied projection.
    descriptor.appearance = {...descriptor.appearance, studio_asset_identity: assetIdentity};
    if (startup && startup.parent_identity === undefined) delete descriptor.parent_identity;
    return descriptor;
  });
  return {...base, assetIdentity, boxes, proxyIdentities: boxes.map(box => box.identity)};
}

function validateMechanicalCreatureStudioEnvironment(request) {
  const initial = mechanicalCreatureStudioRegistration(), state = mechanicalCreatureStudioBridgeState;
  const registered = readMechanicalCreatureStudioRegisteredDefinition();
  const definition = request.definition;
  if (typeof request.assetIdentity !== "string" || !request.assetIdentity ||
      !Number.isSafeInteger(request.sequence) || request.sequence < 1 ||
      !Number.isSafeInteger(request.revision) || request.revision < 0 ||
      definition?.schema !== initial.schema || definition.identity !== initial.identity ||
      definition.assetIdentity !== request.assetIdentity || definition.sourceRevision !== request.revision ||
      definition.sourceClassIdentity !== initial.sourceClassIdentity ||
      !Array.isArray(definition.boxes) || definition.boxes.length < registered.boxes.length ||
      !Array.isArray(definition.proxyIdentities) || definition.proxyIdentities.length !== definition.boxes.length ||
      new Set(definition.proxyIdentities).size !== definition.proxyIdentities.length ||
      registered.proxyIdentities.some(identity => !definition.proxyIdentities.includes(identity)) ||
      definition.unitM !== initial.unitM || !Array.isArray(definition.placement) ||
      definition.placement.length !== initial.placement.length ||
      definition.placement.some((value, i) => value !== initial.placement[i]))
    throw new TypeError("Environment update must retain its registered Studio owner and exact identity set");
  if (state.applied && request.sequence <= state.applied.sequence)
    throw new Error("Stale Studio environment sequence");
  if (request.commit !== undefined && typeof request.commit !== "function") throw new TypeError("commit must be synchronous callback");
  if (request.isCurrent !== undefined && typeof request.isCurrent !== "function") throw new TypeError("isCurrent must be callback");
  const expectedAsset = state.applied?.assetIdentity || initial.assetIdentity;
  const activeProxyIds = definition.activeProxyIdentities ?? definition.proxyIdentities;
  if (!Array.isArray(activeProxyIds) || !activeProxyIds.includes(initial.identity) ||
      new Set(activeProxyIds).size !== activeProxyIds.length ||
      activeProxyIds.some(identity => !definition.proxyIdentities.includes(identity)))
    throw new TypeError("Active Studio proxies must be a unique registered subset containing its building");
  const previousActive = state.definition?.activeProxyIdentities ?? state.definition?.proxyIdentities ?? initial.proxyIdentities;
  const seen = new Set(), updates = [];
  for (const source of definition.boxes) {
    const authored = registered.boxes.find(box => box.identity === source.identity);
    const originals = model.document_geometry.boxes.filter(box => box.identity === source.identity), box = originals[0];
    const visible = shaderViewer.geometry.filter(box => box.identity === source.identity);
    const object = model.world.objects.find(object => object.identity === source.identity);
    const runtimeId = turingWorld.objectRuntimeId(source.identity);
    if (seen.has(source.identity) || !definition.proxyIdentities.includes(source.identity) ||
        source.appearance?.studio_asset_identity !== request.assetIdentity)
      throw new Error(`Studio ownership or original runtime identity disagrees: ${source.identity}`);
    if (!authored) {
      validateMechanicalCreatureStudioNewPart(source, initial);
      if (typeof turingWorld.appendObjects !== "function") throw new Error("The original registry has no canonical append admission");
      if (originals.length || visible.length || object || runtimeId || !activeProxyIds.includes(source.identity))
        throw new Error(`New Studio identity collides with an existing owner or is inactive: ${source.identity}`);
    } else if (originals.length !== 1 || !object || !(runtimeId > 0) ||
        object.extensions?.["abstract_ui.document_geometry"] !== box ||
        (previousActive.includes(source.identity) ? visible.length !== 1 || visible[0] !== box : visible.length !== 0) ||
        turingWorld.objectFromRuntimeId(runtimeId) !== object ||
        box.kind !== authored.kind || source.kind !== authored.kind ||
        source.geometry_mode !== authored.geometry_mode || box.geometry_mode !== authored.geometry_mode ||
        source.placement?.custody !== authored.placement?.custody || box.placement?.custody !== authored.placement?.custody ||
        source.parent_identity !== authored.parent_identity || source.studio_source_part !== authored.studio_source_part ||
        object.extensions?.["studio.capture"]?.assetIdentity !== box.appearance?.studio_asset_identity ||
        (previousActive.includes(source.identity) && box.appearance?.studio_asset_identity !== expectedAsset))
      throw new Error(`Studio ownership or original runtime identity disagrees: ${source.identity}`);
    seen.add(source.identity);
    const center = mechanicalCreatureStudioVector(source.center, 2, "center");
    const half_extent = mechanicalCreatureStudioVector(source.half_extent, 2, "half_extent");
    const rotation = mechanicalCreatureStudioVector(source.placement?.rotation, 3, "placement.rotation");
    if (half_extent.some(value => value <= 0) || !(source.height > 0) || !Number.isFinite(source.height) ||
        !Number.isFinite(source.placement?.elevation) || rotation[0] !== 0 || rotation[2] !== 0 ||
        ![source.floor_height, source.wall_thickness, source.radius].every(value => Number.isFinite(value) && value >= 0) ||
        source.floor_height > source.height || !Array.isArray(source.openings) ||
        source.openings.length !== (authored?.openings.length ?? 0))
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
    updates.push({box, object, source, values: {center, half_extent, height: source.height,
      floor_height: source.floor_height, wall_thickness: source.wall_thickness, radius: source.radius,
      placement: {...(box?.placement || source.placement), elevation: source.placement.elevation, rotation},
      openings: structuredClone(source.openings),
      appearance: {...(box?.appearance || source.appearance), studio_asset_identity: request.assetIdentity}}});
  }
  return {updates, activeProxyIds};
}

function admitMechanicalCreatureStudioParts(definition, updates) {
  const missing = updates.filter(update => !update.object);
  if (!missing.length) return;
  const boxes = model.document_geometry.boxes;
  if (!Object.isExtensible(boxes) || !Object.getOwnPropertyDescriptor(boxes, "length")?.writable)
    throw new TypeError("Original document geometry cannot retain admitted Studio boxes");
  if (boxes.length + missing.length >= 0x100000000)
    throw new RangeError("Original document geometry capacity cannot retain admitted Studio boxes");
  const authored = createMechanicalCreatureStudioWorldObjects(model, definition, missing.map(update => update.source.identity));
  // This is the existing registry's canonical append API. It owns all runtime
  // IDs and clones. Admission alone never activates mesh, contacts or receipts.
  turingWorld.appendObjects(authored, {structural: true});
  for (const update of missing) {
    update.object = turingWorld.object(update.source.identity);
    update.box = update.object.extensions["abstract_ui.document_geometry"];
    boxes.push(update.box);
  }
}

function setMechanicalCreatureStudioActiveGeometry(updates, activeProxyIds) {
  const geometry = shaderViewer.geometry, active = new Set(activeProxyIds);
  for (const {box} of updates) if (!active.has(box.identity)) {
    const index = geometry.indexOf(box); if (index >= 0) geometry.splice(index, 1);
  }
  // Reinsert the exact registered box, after its preceding active registered
  // sibling. Do not construct another world object or move unrelated geometry.
  let previous = -1;
  for (const {box} of updates) if (active.has(box.identity)) {
    let index = geometry.indexOf(box);
    if (index < 0) { index = previous + 1; geometry.splice(index, 0, box); }
    previous = index;
  }
}

function applyMechanicalCreatureStudioEnvironment() {
  const state = mechanicalCreatureStudioBridgeState, request = state.pending;
  if (!request) return;
  state.pending = null;
  let saved = null, savedGeometry = null, previousRevision = shaderViewer.revision;
  try {
    if (request.isCurrent && !request.isCurrent()) throw new DOMException("Studio update superseded", "AbortError");
    const {updates, activeProxyIds} = validateMechanicalCreatureStudioEnvironment(request);
    admitMechanicalCreatureStudioParts(request.definition, updates);
    savedGeometry = [...shaderViewer.geometry];
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
    setMechanicalCreatureStudioActiveGeometry(updates, activeProxyIds);
    shaderViewer.revision += 1; model.scene_mesh.revision = shaderViewer.revision;
    // The original function owns static realization, identity spans, contacts,
    // mesh publication and the existing worker's `colliders` message.
    rebuildPortableSceneMesh();
    const receipt = mechanicalCreatureStudioColliderReceipt({buildingIdentity: request.definition.identity, assetIdentity: request.assetIdentity,
      previousAssetIdentity: state.applied?.assetIdentity || mechanicalCreatureStudioRegistration().assetIdentity,
      revision: request.revision, sequence: request.sequence, geometryRevision: shaderViewer.revision,
      colliderPublication: stateLoopRuntime.ready ? "posted-to-original-worker" : "native-scene-ready-worker-pending"});
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
      shaderViewer.geometry.splice(0, shaderViewer.geometry.length, ...savedGeometry);
      shaderViewer.revision = previousRevision; model.scene_mesh.revision = previousRevision;
      try {
        rebuildPortableSceneMesh();
        // The rejected edit was already posted. The restored field needs a new
        // barrier; an earlier observation cannot acknowledge this rollback.
        if (state.applied) state.applied = mechanicalCreatureStudioColliderReceipt(state.applied);
        else { stopMechanicalCreatureStudioColliderObservation(); state.colliderBarrier = null; }
      } catch (rollbackError) {
        invalidateMechanicalCreatureStudioColliderPublication();
        reportRuntimeFault("studio-environment-rollback", rollbackError);
      }
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
  if (state.applied) state.applied = mechanicalCreatureStudioColliderReceipt(state.applied);
}

function stopMechanicalCreatureStudioColliderObservation() {
  const barrier = mechanicalCreatureStudioBridgeState.colliderBarrier;
  if (barrier?.listener) barrier.worker.removeEventListener("message", barrier.listener);
  if (barrier) barrier.listener = null;
}

function observeMechanicalCreatureStudioColliderBarrier() {
  const state = mechanicalCreatureStudioBridgeState, barrier = state.colliderBarrier;
  stopMechanicalCreatureStudioColliderObservation();
  if (!barrier) return;
  barrier.observedSequence = null;
  const epoch = state.epoch;
  if (!epoch || typeof barrier.worker.addEventListener !== "function" ||
      typeof barrier.worker.removeEventListener !== "function") return;
  barrier.listener = event => {
    if (state.colliderBarrier !== barrier || state.epoch !== epoch) return;
    if (stateLoopRuntime.worker !== barrier.worker || !stateLoopRuntime.ready) {
      invalidateMechanicalCreatureStudioColliderPublication(); return;
    }
    const message = event.data;
    // Observe this exact Worker's original message after its existing decoder.
    // A late retired worker or a naked latestSequence assignment is no proof.
    if (message?.type === "snapshot-buffer" && Number.isSafeInteger(message.sequence) &&
        message.sequence >= barrier.requiredSequence &&
        message.sequence === stateLoopRuntime.latestSequence && barrier.observedSequence === null)
      barrier.observedSequence = message.sequence;
  };
  barrier.worker.addEventListener("message", barrier.listener);
}

function mechanicalCreatureStudioBarrierEvidence(barrier) {
  return {scope: "worker-collider-field", status: barrier.observedSequence === null ?
    "awaiting-original-recycle" : "observed-after-original-recycle",
    postedAfterSequence: barrier.postedAfterSequence, requiredSequence: barrier.requiredSequence,
    observedSequence: barrier.observedSequence};
}

function mechanicalCreatureStudioColliderReceipt(receipt) {
  const state = mechanicalCreatureStudioBridgeState;
  stopMechanicalCreatureStudioColliderObservation(); state.colliderBarrier = null;
  if (!stateLoopRuntime.ready || !stateLoopRuntime.worker)
    return mechanicalCreatureStudioFreeze({...receipt, colliderPublication: "native-scene-ready-worker-pending",
      workerColliderBarrier: null});
  const sequence = stateLoopRuntime.latestSequence;
  if (!Number.isSafeInteger(sequence) || sequence < 0 || sequence > Number.MAX_SAFE_INTEGER - 2)
    throw new Error("Original worker sequence cannot support a collider barrier");
  // Original protocol: at most one snapshot is in flight, even with three
  // pooled buffers. N+1 may already be queued before the collider post. N+2
  // requires the main handler's recycle of N+1, queued AFTER that post on the
  // same Worker channel. FIFO therefore proves the CPU collider assignment
  // happened before N+2 publication. This is NOT a solver-step, GPU completion
  // or persistent-current-field acknowledgment; later edits can supersede it.
  const barrier = state.colliderBarrier = {worker: stateLoopRuntime.worker,
    postedAfterSequence: sequence, requiredSequence: sequence + 2, observedSequence: null, listener: null};
  observeMechanicalCreatureStudioColliderBarrier();
  return mechanicalCreatureStudioFreeze({...receipt, colliderPublication: "posted-to-original-worker",
    workerColliderBarrier: mechanicalCreatureStudioBarrierEvidence(barrier)});
}

function invalidateMechanicalCreatureStudioColliderPublication() {
  const state = mechanicalCreatureStudioBridgeState;
  stopMechanicalCreatureStudioColliderObservation(); state.colliderBarrier = null;
  if (state.applied) state.applied = mechanicalCreatureStudioFreeze({...state.applied,
    colliderPublication: "native-scene-ready-worker-pending", workerColliderBarrier: null});
}

function refreshMechanicalCreatureStudioColliderReceipt() {
  const state = mechanicalCreatureStudioBridgeState, barrier = state.colliderBarrier;
  if (!barrier || !state.applied) return;
  if (!stateLoopRuntime.ready || stateLoopRuntime.worker !== barrier.worker ||
      stateLoopRuntime.latestSequence < barrier.postedAfterSequence ||
      (barrier.observedSequence !== null && stateLoopRuntime.latestSequence < barrier.observedSequence)) {
    invalidateMechanicalCreatureStudioColliderPublication(); return;
  }
  state.applied = mechanicalCreatureStudioFreeze({...state.applied,
    workerColliderBarrier: mechanicalCreatureStudioBarrierEvidence(barrier)});
}

function publishMechanicalCreatureStudioSnapshot(now, dt) {
  const state = mechanicalCreatureStudioBridgeState;
  if (!state.epoch) return;
  try {
    refreshMechanicalCreatureStudioColliderReceipt();
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
    const incoming = prepareMechanicalCreatureStudioBuilding(packet, {
      identity: initial.identity, assetIdentity, translationM: [...initial.translationM],
      entranceWidthM: initial.entrance.widthM, entranceHeightM: initial.entrance.heightM});
    const registered = readMechanicalCreatureStudioRegisteredDefinition();
    for (const box of incoming.boxes) if (!registered.proxyIdentities.includes(box.identity)) {
      validateMechanicalCreatureStudioNewPart(box, initial);
      if (typeof turingWorld.appendObjects !== "function") throw new Error("The original registry has no canonical append admission");
      if (model.world.objects.some(object => object.identity === box.identity) ||
          model.document_geometry.boxes.some(source => source.identity === box.identity) || turingWorld.object(box.identity))
        throw new Error(`New Studio identity collides with an existing owner: ${box.identity}`);
      registered.boxes.push({...structuredClone(box), appearance: {...box.appearance, studio_asset_identity: registered.assetIdentity}});
      registered.proxyIdentities.push(box.identity);
    }
    const plan = reconcileMechanicalCreatureStudioBuilding(registered, incoming, registered);
    const definition = {...plan.definition, activeProxyIdentities: [...plan.activeProxyIds]};
    return {assetIdentity, revision: definition.sourceRevision, definition,
      building: {identity: definition.identity, assetIdentity, placement: [...definition.placement],
        entrance: {...definition.renderEntrance}, proxyIds: [...plan.activeProxyIds]}};
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
    if (epoch !== state.epoch || sink === null) {
      stopMechanicalCreatureStudioColliderObservation();
      state.slots = [null, null]; state.publishedSlot = -1;
    }
    state.sink = sink; state.epoch = sink === null ? null : epoch;
    observeMechanicalCreatureStudioColliderBarrier();
  },
  readSnapshot() {
    const state = mechanicalCreatureStudioBridgeState;
    return state.publishedSlot < 0 ? null : state.slots[state.publishedSlot];
  },
  readCamera() { return readMechanicalCreatureStudioCamera(); },
});

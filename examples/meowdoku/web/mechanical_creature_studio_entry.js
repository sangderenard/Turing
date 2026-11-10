/* Optional entry data for the preserved player's native respawn/navigation.
 * No new player, input loop, camera or body transport. Only the initial import
 * enters the exterior doorway apron; later imports and view switches retain it.
 */
let mechanicalCreatureStudioEntryReceipt = null;
const mechanicalCreatureStudioQualificationWaiters = [];

function readMechanicalCreatureStudioQualification() {
  if (typeof vehicleQualification === "undefined") return null;
  const q = vehicleQualification;
  return {state:q.complete&&!q.passed?"failed":q.passed&&q.animationStarted?"ready":"pending",
    simulatedSeconds:Number(q.simulatedSeconds||0),failures:[...q.failures]};
}

function notifyMechanicalCreatureStudioQualification() {
  const q = readMechanicalCreatureStudioQualification();
  if (q?.state === "pending") return;
  for (const waiter of mechanicalCreatureStudioQualificationWaiters.splice(0)) {
    if (q?.state === "failed") waiter.reject(new Error("Original vehicle qualification failed: " + q.failures.join("; ")));
    else waiter.resolve();
  }
}

function prepareMechanicalCreatureStudioEntry(receipt) {
  const qualification = readMechanicalCreatureStudioQualification();
  if (qualification && qualification.state !== "ready")
    throw new Error("Original vehicle qualification must pass and launch the world before Studio entry");
  const applied = mechanicalCreatureStudioBridgeState.applied;
  if (!applied || (receipt !== null && (receipt?.state !== "visible" || receipt.identity !== applied.assetIdentity ||
      receipt.sequence !== applied.sequence || receipt.sourceRevision !== applied.revision ||
      receipt.building?.identity !== applied.buildingIdentity)))
    throw new Error("Player entry requires the current visible Studio environment receipt");
  const definition = readMechanicalCreatureStudioRegisteredDefinition();
  const building = definition.boxes.find(box => box.identity === definition.identity);
  const opening = building?.openings.find(item => item.identity === `${definition.identity}/opening:entry`);
  const actor = entityState.get(viewportControls.policy?.actor);
  const eye = Number(model.viewer.camera.eye_height), radius = Number(model.viewer.camera.collision_radius);
  const clearance = Number(model.navigation.grid.clearance), thickness = Number(building?.wall_thickness);
  if (!actor || !Array.isArray(actor.entity?.pose?.position || actor.record?.pose?.position) ||
      !opening || opening.side !== "north" || ![eye,radius,clearance,thickness,opening.width,opening.height].every(Number.isFinite) ||
      eye <= 0 || radius <= 0 || clearance < 0 || thickness < 0 ||
      opening.width <= radius * 2 || opening.height <= eye + radius)
    throw new Error("The original player capsule does not fit the Studio entry");
  // Use the native map's door-point calculation on the exterior side. One
  // half-door width leaves room to walk toward the doorway instead of spawning
  // in its wall plane. The native contact surface remains the floor authority.
  const inset = -Math.max(opening.width / 2, clearance + thickness * .5 + .025);
  const point = navigationOpeningPoint(building, opening, inset);
  const ground = cameraGroundHeight(point[0], point[1]);
  const support = sampleContactSurface(point[0], point[1], ground, ground, 0);
  const position = [point[0], Number(support?.height) + eye, point[1]];
  const yaw = Math.atan2(building.center[1] - point[1], building.center[0] - point[0]);
  if (!support || !position.every(Number.isFinite) || !Number.isFinite(yaw))
    throw new Error("Studio exterior entry has no finite native ground support");
  const envelope = shaderViewer.geometry.find(box => box.identity === model.document_geometry.representation_boundary?.inside);
  if (!envelope || Math.abs(point[0]-envelope.center[0])+radius >= envelope.half_extent[0] ||
      Math.abs(point[1]-envelope.center[1])+radius >= envelope.half_extent[1])
    throw new Error("Studio exterior entry is outside the original world boundary");
  if (shaderViewer.colliders.some(collider => !collider.surface &&
      playerColliderVerticalOverlap(position, collider) &&
      point[0] > collider.minimum[0]-radius && point[0] < collider.maximum[0]+radius &&
      point[1] > collider.minimum[2]-radius && point[1] < collider.maximum[2]+radius))
    throw new Error("Studio exterior entry is obstructed for the original player capsule");
  return {buildingIdentity:definition.identity,assetIdentity:applied.assetIdentity,
    revision:applied.revision,sequence:applied.sequence,positionM:position,yaw,pitch:0,
    supportIdentity:support.identity,location:"outside-studio-entrance",playerIdentity:viewportControls.policy.actor};
}

function mechanicalCreatureStudioRespawnPosition() {
  // Ordinary R remains the original respawn. Once this optional session has
  // entered Studio, resolve its current native doorway/support again rather
  // than retaining a coordinate that a later room edit could obstruct.
  if (!mechanicalCreatureStudioEntryReceipt || vehicleRuntime.active) return null;
  return prepareMechanicalCreatureStudioEntry(null).positionM;
}

globalThis.mechanicalCreatureStudioEntry = Object.freeze({
  readQualification:readMechanicalCreatureStudioQualification,
  whenReady() {
    const q = readMechanicalCreatureStudioQualification();
    if (!q || q.state === "ready") return null;
    if (q.state === "failed") return Promise.reject(new Error("Original vehicle qualification failed: " + q.failures.join("; ")));
    // The original gate calls the notifier on failure or its successful world
    // launch. No new polling loop, timer or qualification decision is created.
    return new Promise((resolve,reject)=>mechanicalCreatureStudioQualificationWaiters.push({resolve,reject}));
  },
  enter(receipt) {
    if (!receipt) throw new Error("Player entry requires a visible Studio receipt");
    if (mechanicalCreatureStudioEntryReceipt) return mechanicalCreatureStudioEntryReceipt;
    const entry = prepareMechanicalCreatureStudioEntry(receipt);
    // All destination validation precedes any control, vehicle or body write.
    setViewportControlHighlight(false);
    cancelEntityNavigation(entry.playerIdentity);
    clearActiveVehicle(); // Keep its native parked solver body and machine state.
    if (!respawnViewportActor(entry.positionM)) throw new Error("The original player respawn refused Studio entry");
    viewportControls.yaw = entry.yaw; viewportControls.pitch = entry.pitch;
    mechanicalCreatureStudioEntryReceipt = mechanicalCreatureStudioFreeze(entry);
    setPlacementStatus("Walking outside Marble Studio · original player and launcher · V returns to the parked vehicle");
    return mechanicalCreatureStudioEntryReceipt;
  },
  setActive(active) {
    const q = readMechanicalCreatureStudioQualification();
    if (q && q.state !== "ready") return; // Never disturb its active test vehicle.
    if (!active) {
      if (primaryActionState.down) endViewportPrimary("studio-view");
      if (secondaryActionState.down) endViewportSecondary("studio-view");
      setViewportControlHighlight(false);
      shaderViewer.element?.blur();
    } else if (mechanicalCreatureStudioEntryReceipt) {
      // This invokes the incumbent capture/focus path. A browser may require
      // a fresh click to grant pointer lock after asynchronous initial import.
      requestViewportControls();
    }
  },
  read() {
    return {entry:mechanicalCreatureStudioEntryReceipt,active:viewportControls.highlighted,
      positionM:viewportControls.position ? [...viewportControls.position] : null,
      yaw:viewportControls.yaw,pitch:viewportControls.pitch,mountedVehicle:vehicleRuntime.active?.identity ?? null};
  },
});

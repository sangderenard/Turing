/* Additive, classic-script hook for the pinned playable MechanicalCreature page.
 * The game owns bodies, colliders, camera, controls, clocks and lifecycle. This
 * hook changes only selected projectile COLOR draws. Depth/shadow proxies stay.
 *
 * Install globalThis.mechanicalCreatureImportedAppearance = {
 *   identities(context): Set<string>,
 *   draw({...context, identities, phase: "opaque" | "transparent"}): void
 * } after the imported appearance resources are ready. Return an empty Set
 * while unavailable. Both callbacks are synchronous. Each callback must restore every GL state it changes and
 * treat the original state Maps as read-only. Setting the adapter to null
 * restores the original appearance on the next existing game render.
 */

function mechanicalCreatureAppearanceSpans(identities) {
  if (!identities?.size) return [];
  const count = shaderViewer.vertexCount;
  return shaderViewer.identitySpans.filter(span => identities.has(span.identity) &&
    Number.isInteger(span.firstVertex) && Number.isInteger(span.vertexCount) &&
    span.firstVertex >= 0 && span.vertexCount > 0 &&
    span.firstVertex % 3 === 0 && span.vertexCount % 3 === 0 &&
    span.firstVertex + span.vertexCount <= count
  ).sort((a, b) => a.firstVertex - b.firstVertex);
}

function prepareMechanicalCreatureImportedAppearance(gl, cameraPosition, cameraFacing, width, height) {
  const adapter = globalThis.mechanicalCreatureImportedAppearance;
  if (!adapter || typeof adapter.identities !== "function" || typeof adapter.draw !== "function") return null;
  const context = {gl, cameraPosition, cameraFacing, width, height, sceneGeometry: shaderViewer.geometry,
    projectileState, projectilePickupState, stateLoopRuntime, snapshotBodies: stateLoopRuntime.bodies};
  try {
    const requested = adapter.identities(context);
    if (!(requested instanceof Set)) throw new TypeError("imported appearance identities must be a Set");
    // Only explicitly imported projectile or static asset identities can opt
    // out of their color proxy. The original world remains authoritative.
    // Vehicle spans share the index but live beyond this base VAO's vertices.
    const staticAssets = new Set((shaderViewer.geometry || []).filter(box =>
      box.appearance?.studio_asset_identity).map(box => box.identity));
    const candidates = new Set([...requested].filter(identity => staticAssets.has(identity) ||
      (projectileState.get(identity) || projectilePickupState.get(identity))?.appearanceAssetIdentity));
    const identities = new Set(mechanicalCreatureAppearanceSpans(candidates).map(span => span.identity));
    return identities.size || adapter.drawWhenEmpty === true ? {adapter, context, identities} : null;
  } catch (error) {
    reportRuntimeFault("imported-appearance", error);
    return null;
  }
}

function drawMechanicalCreatureBaseAppearance(gl, hiddenIdentities = null) {
  gl.bindVertexArray(shaderViewer.vao);
  const spans = mechanicalCreatureAppearanceSpans(hiddenIdentities);
  if (!spans.length) {
    gl.drawArrays(gl.TRIANGLES, 0, shaderViewer.vertexCount);
    return;
  }
  let first = 0;
  for (const span of spans) {
    if (span.firstVertex > first) gl.drawArrays(gl.TRIANGLES, first, span.firstVertex - first);
    first = Math.max(first, span.firstVertex + span.vertexCount);
  }
  if (first < shaderViewer.vertexCount) gl.drawArrays(gl.TRIANGLES, first, shaderViewer.vertexCount - first);
}

function drawMechanicalCreatureImportedAppearance(frame, requestedPhase = null) {
  if (!frame) return;
  try {
    for (const phase of requestedPhase ? [requestedPhase] : ["opaque", "transparent"]) {
      const result = frame.adapter.draw({...frame.context, identities: frame.identities, phase});
      if (result && typeof result.then === "function")
        throw new TypeError("imported appearance draw must be synchronous");
      if (result === false)
        throw new Error(`imported appearance declined ${phase} draw`);
    }
  } catch (error) {
    // A failed appearance must not interrupt the actual game, wheels or HUD.
    // Callbacks must restore GL state even on failure; rebind the original
    // program/VAO and show the existing proxy rather than an invisible body.
    reportRuntimeFault("imported-appearance", error);
    const gl = frame.context.gl;
    gl.useProgram(shaderViewer.program);
    gl.bindVertexArray(shaderViewer.vao);
    for (const span of mechanicalCreatureAppearanceSpans(frame.identities))
      gl.drawArrays(gl.TRIANGLES, span.firstVertex, span.vertexCount);
  }
}

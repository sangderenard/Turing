2026-11-09 STUDIO CHECKPOINT / RAY-SCENE INTEGRITY (MEASURED, source only):

The requested future path is native shared-room recording followed by
continuous all-triangle ray motion. The existing recording and ray systems
remain the owners. No alternate scheduler, interpolated body simulation or
new ray worker was added.

pauseAtDisplayedState now also retains its exact {dt,snapEvery,epoch} clock
and source-product/backend/adapter/count/pad identity. It already retains all
named padded state, recorded quaternion, uniforms, step/seconds and case
transforms. The new metadata does not change native stepping or capture order.

New web/marble_studio_checkpoint.js provides createStudioCheckpoint and
verifyStudioCheckpoint. It is an integrity envelope with scope explicitly
checkpoint-forward, not an archive of a run. It captures all inputs before
the first await and retains:
- The existing complete MRP1 scene packet byte-for-byte: saved subject
  triangles/normals/material IDs and room/fixture triangles/PBR/camera.
- Saved object, authoritative saved-object identity receipt, and native scene.
- Exact native clock/state and complete supplied product manifest.
- Native-to-ray translation from the shared frame, plus SHA-256 receipts.

It verifies the MRP1 subject geometry against the existing geometry SHA,
material references and finite PBR data. Saved-object identity includes design,
materials and physical fields, not just mesh geometry. marble_objects.py now
exposes source_receipt(value), the exact bytes previously local to
source_digest. source_digest hashes those unchanged bytes; Store get/put and
HTTP schemas have not changed. The envelope requires that authoritative
receipt, avoiding JavaScript/Python JSON number-spelling differences (tested
with Python 10.0 versus JavaScript 10). Receipt transport is deliberately not
yet added to any endpoint or UI lifecycle.

Native binding checks require the shared-scene manifest, the same coordinate
frame, exact float32 room/vertex uniforms and identical authored topology.
An old toy-product checkpoint cannot be presented as a shared Studio scene.
This does not independently prove that every optical triangle has an admitted
native collider: whole-scene admission remains a separate release gate.

Tests: 4 passed in 1.04 seconds for tests/test_meowdoku_studio_checkpoint_js.py
and tests/test_meowdoku_compute_state_js.py; the latter includes 18 actual
runtime bookkeeping/transport-double checks. Node checkpoint tests cover exact
time, Python identity bytes, unchanged mesh payload, immutable capture, old-toy
refusal, moved-room refusal, missing poses, changed product, changed materials,
nonnumeric material data and corruption. Independent review's two findings
(unbound saved-object material identity and nonnumeric PBR) were fixed and
reverified. No new GPU execution or native build was performed.

ARCHIVE / MOTION BOUNDARIES FOUND:
- Recording is a sliding in-memory window, not persistence. Keys are opaque
  product snapshots; input events prune with old keys. Endpoint capture is not
  a complete run and may truncate queued future when branching to the display.
- Current host quaternions integrate mean spin at display-snapshot cadence,
  not every native tick. A microframe consumer must replay native ticks from
  a checkpoint, not claim that interpolated display samples are exact physics.
- Exact loaded product bytes/backend identity, unpruned native input events,
  scene revision history and a recorded camera track still need persistence.
- The complete scene must be admitted atomically with its physics generation;
  ray subject poses must be bound to that timeline. sceneForPhoto currently
  targets the canonical saved subject rather than the transient particle pose.
- Reuse Spectral SceneCameraCoordinator/on_dt_request and existing scene
  versioning. Exact fixed native tick alignment or a proved partial-step
  contract is still needed for arbitrary exposure slices.
- The present envelope is bounded to a 64 MiB MRP1 payload and rejects excess;
  it does not decimate or silently omit triangles.

The chase owner documented the actual camera/ray consumer path in
examples/meowdoku/STUDIO_RECORDED_MOTION_HANDOFF_2026-11-09.md. Camera edits
remain separate. No checkpoint imports were added to Studio or web_export,
no backend endpoint was added, no file was downloaded/uploaded for the user,
and no server/live job was changed.

RELEASE NOTE: camera-only restart was separately completed by the visual
coordinator with --retain-fountain-products and a pinned artifact receipt.
That coordinator verified all 83 fountain files unchanged and retained old
compute.js. This checkpoint/native source is not thereby deployed or activated;
a later physics release must deliberately update that retained-product mode.

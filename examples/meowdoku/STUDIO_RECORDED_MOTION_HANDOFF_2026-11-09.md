# Recorded Studio motion and ray microframes

Status: chase-camera source is implemented and tested. The continuous,
full-scene ray-motion path below is an integration plan, not a delivered
renderer or an assertion that current live rays follow physics.

## Chase camera

`web/marble_chase_camera.js` consumes completed displayed positions and their
recorded times. It does not step the law or change particle state. The Studio
adapter takes `shown * snap * manifest.law.dt`, not the newest queued step.
The chase is Y-up with no marble-spin roll, uses radius-relative distance,
bounded damped aim and horizontal look-ahead, and resets on run changes,
backward seeks, teleports and long tab interruptions. Auto heading turns toward
travel only after steering is released. Manual drag temporarily suppresses
auto heading; fixed-heading mode never turns itself. Scroll changes distance,
R/the button recenters. Chase yaw supplies the existing camera-oriented gravity
mapper, so held controls cannot create an auto-camera steering feedback loop.

Static Studio orbit and `studioPhotoCamera()` keep their existing eye/target
and physical-unit conversion. The chase is not yet a ray-camera revision and
does not claim camera collision avoidance, occlusion removal, or room physics.

## Existing components to reuse

- `web/recording.js`: a bounded sliding point-snapshot window, full native
  product keyframes and an input log. `plan(frame)` gives exact native replay
  from a keyframe/start to a displayed frame. Keys are opaque device/product
  snapshots; old frames and inputs are evicted. This is not a portable archive.
- `web/compute.js`: the existing runtime executes and records the law.
  `pauseAtDisplayedState()` drains queued work, restores the displayed frame,
  verifies its positions, exports every manifest state column and its recorded
  orientation, and truncates a newer branch. It is an endpoint checkpoint, not
  a non-destructive recording export. Host quaternion recording currently
  integrates mean angular velocity at snapshot cadence, generally six native
  ticks. It must not be advertised as a native per-tick orientation history.
- `web/marble_studio_physics.js`: physical metre/second frame, exact authored
  parts and full-state start/handoff helpers. Both native and Studio axes are
  Y-up. Photo coordinates are `OBJECT_CENTER + Studio * unitM`, where
  `OBJECT_CENTER = [0, 0, .18]`. Thus native-to-photo translation is
  `OBJECT_CENTER - nativeOriginM`; do not reuse the old toy's quarter-turn.
- Spectral `camera_designer/studio_photo_geometry.py::prepare_photo_geometry`
  consumes the saved subject, captured room/fixtures, materials and physical
  camera housing. `marble_preview_worker.py` builds its tracer scene and GPU
  BVH. Exact subject/scene triangle identity already has a path; it is static.
- Spectral `camera_software/exposure_timing.py` already supplies exposure
  slices, immutable `CameraSliceJob` identities, `SceneSnapshot`, a
  `SceneFrameProvider` interface, canonical-result ordering/deduplication and
  a multireader release gate. `MutableSceneFrameProvider` wraps a mutable
  current scene; it is not an immutable, concurrently safe motion archive.
- Spectral `camera_software/managed_time_bridge.py` maps a slice to the real
  Turing `TimeWindowRequest` and checks exact landing/causality/generation.
  `ManagedProcessState`, `NativeWaveStateParticipant` and
  `NativeQueueStateParticipant` provide transactional rollback participants.
  This adapter contains no substitute timestep controller.
- The actual thick-lens consumer is already wired to this exposure scheduler:
  `thick_lens_focus_lab.py::ForwardCppLensBench.begin_scene_camera_step()` calls
  `SceneCameraCoordinator.ensure_step(exposure_time_s)`. Its camera loop's
  `_run_one_exposure` calls `on_dt_request(slice.dt)` before `_trace`, waits for
  that sensor stage, and drains/joins T5 at exposure completion. `_trace` passes
  separate `flash_exposure_weight` and `natural_exposure_weight` into
  `trace_forward`; a flash-off slice still receives natural emission/sensor
  work. The camera program, not display FPS, owns the exposure duration.
  `camera_manifest.py` already names burst duration, stage count, profile, duty
  cycle and energy scale. Reuse this complete interface, not a second scheduler.
- The current thick-lens upward callback calls `physics_backend.request_dt`,
  whose implementation is still a stub. That is the concrete integration seam,
  not proof that Studio physics is already stepping behind it. Existing
  `scene_version.py::SceneVersionCache` records centroid/group-range versions,
  estimates translation velocity and supplies a preview vertex-motion buffer;
  angular velocity remains zero pending quaternion history. This is not a
  complete archived triangle/BVH snapshot or moving-ray collision path. Its
  bounded versioning/rebuild interface can be extended instead of duplicated.

## Smallest honest integration sequence

1. Finish native whole-scene admission and shared-room pose transitions. Keep
   exact authored geometry, materials, object identities and units. Existing
   independent-convex/room products do not admit a concave compound course;
   do not promote its construction prisms into independent bodies. Captured
   geometry must cover floor, table, gobos, backdrop, fixtures and all subjects.
2. Capture an immutable run archive: manifest/product/source digest, admitted
   scene digest, full start/key states, all unpruned runtime inputs with steps,
   body/mesh identities and an explicit orientation-recording convention.
   Preserve the original run rather than using pause/handoff repeatedly as a
   destructive exporter. A validated checkpoint-plus-scene envelope is a useful
   first stage, but does not stand in for the whole time interval.
3. Replay the same native law, using the complete input log. Materialize pose
   samples at the microframe boundaries. Do not linearly interpolate sparse
   displayed points through collisions and call that native physics. Settle
   the orientation cadence contract before rotating every interior triangle.
   Radius-aware native `law.dt` is fixed in the current product: align exact
   boundaries to native ticks, or explicitly implement and prove a short-step
   product. A generic managed-time adapter does not make this WebGPU law
   variable-step merely by requesting a fractional boundary.
4. Implement a recording-backed `SceneFrameProvider` that resolves a specific
   immutable scene version for each slice. Transform the same object-local
   mesh, interior materials/media and camera housing by the recorded pose.
   Use full mesh policy when all triangles are requested; no silent LOD or
   analytic-sphere substitution. Rebuild/refit each affected acceleration
   structure before either forward or sensor rays use that version, and retain
   it until all readers release it. Avoid a shared mutable BVH for overlapping
   jobs. Record the actual chase eye/target timeline if chase replay is desired,
   rather than rerunning a browser-rate camera filter during offline rendering.
5. Dispatch small ray bundles through the existing slice job/collector and
   managed-time commit gate. Give every contribution its slice, scene version,
   program generation, sample time and weight. Apply shutter/exposure weights
   once and publish only completed, accepted causal prefixes. Preserve native
   wave/queue state in rollback; the in-progress wave-transport coupling audit
   must also cover any new continuation registry. Bundle size is a work budget,
   not the physical exposure duration or a guarantee of real-time speed.
6. Define the intended optical temporal model. Independent frozen-pose slices
   form a sampled shutter integral; that is not automatically exact moving-
   boundary photon transport. If rays/waves carry across slices, advance their
   real time and geometry/field ownership together. Prove energy and causality
   through that path before claiming continuous optical motion. Use the existing
   frame/AVI writer only after the accepted temporal exposures are assembled.

## Why the current live ring is not this pipeline

`camera_designer/marble_live_worker.py::run_live_sensor` uses a fresh fixed ray
budget for one complete sensor sweep of one static scene. It clears the forward
field at each sweep, retains a finite per-pixel sensor contribution ring (or
optional checkpoint), and changes camera/history settings at sweep boundaries.
Its retained contributions estimate a static view; they have no recorded-body
pose or shutter-time labels. Merely changing subject vertices inside this loop
would mix different physical times and potentially inconsistent BVHs/fields.
Reusing its budget control or display code is sensible; calling the existing
ring temporal motion blur is not.

## Required gates before calling motion rendered

- Native checkpoint/replay equality for all state columns and controls, with
  a documented orientation convention and non-destructive branching/eviction.
- One scene identity across physics and rays; exact triangle/material counts,
  transforms, interior-media ownership, camera frame and physical units.
- Static replay parity against the existing photograph, then a simple moving
  sphere/triangle scene with known occlusion and shutter exposure timing.
- Ordered versus out-of-order/retried slice equality, stale-generation rejection,
  multi-reader release, rollback of field/wave/queue state, and no double weights.
- Native-tick boundary checks, fast contacts, held controls, seek/rebirth,
  canceled capture, and a camera track with no replay-time filter drift.
- Actual GPU motion render with all captured triangles; measure BVH update,
  memory, ray coverage, convergence and throughput rather than imply a frame
  rate from microframe or bundle size alone.

The existing Spectral exposure/managed-time tests pass 22/22. They exercise
the real scheduling/collector/bridge logic with bounded fixtures, not an
integrated Studio physics plus moving-geometry optical render.

## Verification in this source batch

The focused chase/drive tests cover real coordinate mapping, steering lock,
follow/lead bounds, manual orbit/zoom/recenter, recording interruption, radius
scaling and 30/60/120 Hz camera behavior. An isolated actual Chrome Studio
loaded the retained compiled 4k toy and advanced from displayed frame 10 to 63
(408 queued native steps), verified control lock/release, zoom/recenter, fixed
heading and unchanged static/photo camera pose, with no browser exceptions.
A second expanded browser gate explicitly confirmed `GpuProduct` on the
NVIDIA Ampere adapter (displayed frames 7 to 57, 372 queued steps) and also
exercised real pointer dragging, held/released D-pad and the R key.
The screenshot was inspected. It is a Phong gameplay preview, not a ray render.
This did not save an object, run a photo/live job, export assets or restart the
production service. The chase source is a later, unpublished batch.

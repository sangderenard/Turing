# Camera toolbox and exact focus reader audit

## Implemented, isolated

`web/marble_camera_toolbox.js` exports `createCameraToolbox` and prepends its
section into the supplied mount (default `#photo-controls`). Hand/drag is the
default. Narrow photo-controller hooks now bind the exact completed image and
existing retouch queue. No Studio bootstrap, frame renderer, material container,
server, exported site or live user's state was changed by this worker.

The three reticle tools require both a real callback and an explicit available
capability. Unavailable tools are disabled and their reason remains visible.
Point sampling and Gaussian sigma presets (1, 3, 10 and 25 percent) plus a
custom 0.1–50 percent sigma are available. Sigma refers to the shorter image
dimension. The ring shows a 3-sigma radius; sampling has square 3-sigma support.

Capture-phase pointer arbitration prevents a reticle click from also orbiting
the camera. Hand mode leaves the incumbent handlers alone. New clicks, tool
changes, spread edits, Escape, hidden-page/blur, explicit invalidation and
disposal abort pending requests. Late callbacks and changed surface revisions
cannot invoke the result consumer. Disposal restores cursors and removes all
listeners and generated elements. The callback must honor its AbortSignal and
return a measurement rather than apply a change before the caller validates it.

## Integration contract

The ready-to-use initializer is
`mountStudioCameraToolbox({doc:document,photo,canvas})` from
`web/marble_camera_toolbox_mount.js`. Call it once after `photo` and `canvas`
exist. It chooses the actual `#photo-image` and mounts near the exposure
controls. It keeps focus disabled with a visible reason, enables HDR tools only
when their exact source is ready, displays each measured proposal, and applies
it only through the separate **Apply proposed photo adjustment** button. Its
`dispose()` unsubscribes, clears retained HDR and removes the toolbox. No
initializer call was added to the shared bootstrap by this worker.

Call `createCameraToolbox` with:

- `doc`, optional `mount`.
- `surfaces`: `{element, kinds, label, revision}` records. `revision()` returns
  a stable scalar identity for the exact displayed image/camera/scene. Use the
  Studio canvas only for a supported real focus reader, and the completed photo
  canvas for `['exposure','white']`. Coordinates are normalized top-left u/v of
  that canvas's image content; do not substitute the live overlay or a different
  scene. Optional `bounds()` supplies the actual content rectangle. The supplied
  mount helper accounts for the photo's centered `object-fit:contain`, rejects
  letterbox clicks, and verifies the HDR dimensions match the decoded canvas.
- `focus({kind,u,v,spread,signal})` and
  `sample({kind,u,v,spread,signal})`, each returning a measurement.
- `capabilities`: `{focus:{available:false,reason},exposure:{available:...},
  white:{available:...}}`. Absence of callback/surface also disables a tool.
- `onResult(measurement,request)`: called only after the toolbox checks its
  generation, signal and surface revision. Apply through the existing owner,
  not a second retouch/render path. If this callback is asynchronous it must
  check the same signal before writing.
- `onStatus(message)`, optional. The toolbox also owns its visible status.
- `onInvalidate()`, optional, for clearing proposed adjustments when a new
  click/tool/spread/source cancels the old request.

The returned object has `select`, `setCapabilities`, `invalidate`, `dispose`,
and read-only `tool`/`pending`. Call `invalidate` when changing scene/camera or
photo selection. Updating capabilities also cancels a pending request.

`web/marble_spot_sample.js` provides:

- `readSpotHdr(ArrayBuffer)`: uses the existing `parseNpz` selective reader to
  read only `linear_srgb_exposure`. It never substitutes unprofiled
  `linear_rgb_exposure` sensor channels or screenshot values.
- `sampleSpotHdr(array,{u,v,spread,signal})`: a weighted mean of actual completed
  HDR pixels. Spread zero is **one accumulated pixel, not one ray**. Gaussian
  support is truncated at 3 sigma on each axis, clipped and renormalized at
  image boundaries. Negative values remain negative. Nonfinite samples reject.
  Reports RGB, exact profile luminance coefficients, pixel count, negative
  pixel count, support bounds, sigma and explicit limitations.
- `createCompletedHdrSampler({getSource,fetchImpl})`: one cached immutable HDR;
  validates source identity before/after fetch/parse/sample. `getSource()` must
  return null unless the displayed final frame is complete and profiled, else
  `{id,url,revision,complete:true,profiled:true}`. `id` should include job digest
  and exact HDR filename. `revision` must bind final-frame selection and display
  identity. URL comes from the real photo controller, never a canvas guess.
  The actual getter also supplies decoded width/height for HDR shape checks.
  Bounds are 64 MiB compressed archive and 4,194,304 H×W×3 floating pixels.
- `proposeSpotAdjustment(sample,{kind,params,whiteReference,targetFraction})`
  returns a proposal and explanation only. White-point correction assumes the
  selected point is neutral, uses Y/R,Y/G,Y/B and resets old white-balance/tint
  gains to zero. Exposure correction uses the current RGB/WB/tint gains and an
  explicitly supplied recorded white reference to propose absolute display EV
  at targetFraction (default 0.18) before tone mapping. This is a metering
  convention, not physical radiometry or encoded 18-percent gray. Both helpers
  reject corrections outside incumbent control ranges rather than clamp them.

The current photo controller owns `photoComplete`, `hasColorProfile`,
`previewHdr`, `selectedFrame`, `generation`, `displayToken`, `retouchDigest`
and the precise HDR URL. Its new `completedHdr()` getter verifies actual decoded
final-frame identity, not merely that `#photo-hdr` is visible. It refuses active
jobs, playback, unprofiled frames, in-flight/pending/failed retouches and stale
selection, and returns detached current parameters plus the recorded effective
white. `subscribeCompletedHdr(fn)` updates capability state. The narrow
`applySpotAdjustment(proposal,{sourceId,sourceRevision})` validates source and
control ranges, synchronizes incumbent sliders and calls their existing queue
once; it does not issue a separate POST or alter the raw file.

The completed snapshot and both
worker/server retouch paths call `Image.fromarray(display)` without flipping,
so the HDR array row order matches the displayed PNG. No sensor basis inversion
or profile transformation is repeated here. Applying a proposal should use the
existing retouch controls/queue so stale-job and exactly-once logic remain owned
by `marble_photo.js`.

## Exact scene/focus audit

The actual photo path is:

1. `web/marble_photo.js::packMarblePhoto`: MRP1 contains exact saved subject
   triangle positions/normals/material IDs and the captured non-subject fixture
   mesh, camera and exposure options.
2. `marble_photo.py::decode_upload`: verifies saved/snapshot geometry SHA,
   immutable PBR table, medium adjacency, region media, analytic spheres and
   triangle-budget admission.
3. Spectral `camera_designer/studio_photo_geometry.py::prepare_photo_geometry`:
   the shared CPU-only actual preparation builds native camera housing, saved
   subject and real fixture geometry, explicit media and analytic spheres using
   `scene_builder`. It returns geom/shade/materials/triangles/contexts/sources
   plus packed spheres and geometry evidence.
4. `camera_designer/marble_preview_worker.py::run` calls that reader, builds the
   actual renderer BVH and creates the sensor accumulator. Refraction and
   parametric-region transport remain in that existing shader.

The displayed Studio has a different capture path:
`marble_studio.js::captureCurrentStudioScene` requires the last completed
displayed frame/current revision, actual subject model/view, authored room,
table/gobos/fixture parts, and `packStudioScene` exact source geometry/material
identity. `currentCameraOptions`/`sceneForPhoto` explicitly use the separate
Exposure profile and canonical subject pose. A reticle must not click one
surface while querying the other path and present that as a measured depth.

Available query code and limits:

- `web/studio_triangle_compute.js::StudioTriangleCompute` uses the actual
  Spectral `demo_pluck_gl.py::hit_tri`, compiled through existing shared SSA
  into Wasm/WebGPU by `studio_triangle_compute.py`. Source/artifact hashes are
  checked. It supports at most 4,096 rays and 4,096 triangles, with 1,048,576
  ray/triangle pairs per dispatch. Assets currently reside in isolated build
  output, not a deployed endpoint. Complete scene ownership/filter/tie ordering
  and analytic/parametric boundaries are not supplied by this host alone.
- Spectral `camera_designer/studio_first_hit_loan.py::FirstHitLoan` exports real
  sensor launches and actual BVH-ordered triangle data from the GL owner. It
  refuses explicit regions and requires one sample per site, no sprinkle or
  sample evidence, and at most 4,096 rays/triangles. It is a transport continuation
  seam, not a public spot-focus query. Capture/resume must remain on its GL
  context; do not reuse it by silently mutating an active photo.
- Spectral `ray_tracer_bridge.py::trace_single_ray`, `trace_cone_rays` and
  `trace_depth_map` accept actual triangles, but use a different absolute
  determinant tolerance (1e-9) from current Studio's scale-relative hit law.
  They are no-bounce triangle queries, not refraction-aware autofocus. They
  cannot be advertised as macro-scale renderer parity or full mixed geometry.

No verified mixed-geometry, arbitrary-scene, refraction-aware autofocus reader
was found in this audit. Focus remains explicitly unavailable until that real
reader is supplied. Do not replace it with a sphere-only scene, target distance,
synthetic depth, new CPU triangle solver or unverified HDR contrast sweep.
Any first-hit-only implementation must be called **surface focus**. For an
off-axis hit, lens focus is the axial projection dot(hit-eye,forward), not its
Euclidean slant distance. Surface focus does not focus an apparent glass interior.

## Verification

Both focused Node checks pass:

- `node tests/meowdoku_spot_sample_check.mjs`: actual compressed NumPy archive
  with Unicode metadata and different raw/profiled arrays, selection, known
  Gaussian weights/edges, row orientation, input preservation, negative/nonfinite
  values, correction bounds, stale source, cancellation and HTTP/size failure.
- `node tests/meowdoku_camera_toolbox_check.mjs`: generated controls, hand default,
  missing-reader reasons, matching surface dispatch, spread/custom bounds,
  repeated/out-of-order clicks, source-revision rejection, Escape, visibility,
  explicit invalidation, capability loss and listener/cursor disposal.

The existing `meowdoku_marble_photo_retouch_check.mjs` suite also passes with
five added focused source/proposal cases: decoded-frame ownership, missing
profile, delayed image decode, bounded exactly-once apply/stale rejection, and
final-orbit selection/playback. The toolbox integration test passes a real
compressed NumPy HDR through the complete reader/proposal path and proves no
owner write occurs before its Apply button.

These are isolated module and DOM-double checks. They are not browser visual
QA, real production HDR sampling, native focus verification, deployment,
publication or a server restart. Shared bootstrap mount and visual verification
belong to the coordinating worker.

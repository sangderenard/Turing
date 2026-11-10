# Studio autosave, exports and sample-coverage overlay

Local source is implemented and tested; this receipt is not publication proof.
No production server, active photo/live worker or GPU job was changed.

## Behavior

- Object edits capture a detached recipe, material tables, name and physical
  properties before hashing. Geometry, LOD receipts and mass derive only from
  that capture. A serialized 800 ms autosave queue coalesces edits, suppresses
  stale completion, retains dirty data after failures, and supports off/on and
  manual retry. Loads reject edits arriving during verification; a newer load
  supersedes an older load. Dirty pages warn before navigation.
- Autosave is on by default and its toggle persists per origin. Saved objects
  remain immutable content-addressed versions. Duplicate recipes retain their
  original library name; file exports use the current name. The existing
  128-version library limit is visible and unchanged. Full-library/network
  failures keep edits and the last live worker, showing Phong until retry.
- Lighting/room changes do not create object saves. Photo scene invalidation
  preserves the saved object identity; existing still exposures continue with
  their captured scene during both object autosave and rig edits.
- Phong continues drawing below a pointer-transparent canvas. Missing, pending,
  rejected and outdated ray revisions leave that canvas transparent. Only a
  current accepted frame can show. Real sample-validity alpha, supplied by the
  live worker, distinguishes sampled black from untouched pixels. Off is fully
  transparent. This is not brightness-keying or a Phong screenshot.
- Save object to file exports the complete versioned JSON recipe. Download last
  ray image exports the last accepted original PNG Blob, preserving alpha. It
  remains available after Off and is explicitly labeled as the last image.
- Live batch is 128–65,536 actual total primary forward rays per sensor sweep,
  default 8192, applied at a completed sweep boundary with history reset. An
  incomplete numeric edit keeps the previous worker rather than canceling it.
  Existing additive photon normalization is unchanged, so brightness may change.

## Verification

- Turing Python live/batch/object-store gates: 42 passed.
- Autosave Node queue: 20 cases; downloads: 8 cases. Snapshot isolation, mass,
  object-only fingerprint, material/load races, live lifecycle/memory/display,
  transparent fallback/batch, ten-light and photo/retouch suites pass. New photo
  regressions prove scene-only readiness and no still cancel during autosave.
- Spectral child receipt:137 CPU tests passed, 1 opt-in GPU test skipped. Tests
  exercise actual live loop/pump logic, partial sweep revisions and RGBA output.
- Actual Chrome UI on the separate 8883 fixture verified autosave, load, toggle
  off, failure/retry while live stays enabled, rig-only save counts unchanged,
  actual batch 12345 control, Phong opacity 1 and ray opacity 0 when pending/Off.
- Downloaded PNG exactly matches fixture response bytes: sampled black is
  RGBA(0,0,0,255), untouched alpha 0. Downloaded JSON exactly equals the saved
  recipe which Chrome verified and loaded. This browser fixture is generated
  test imagery, not a real GPU exposure.
- Detailed receipt: build/dot-live-ray-20261109/autosave-ui-verification.json.
  Screenshot: autosave-ui-off.png in that directory. Helper PID 19816; fake live
  off, errors/delay reset. QA tab closed. Parent owns staging/publication.

## Separate findings

The Chrome-verified recipe's medium-LOD exact hash differs when reverified in
Node. Full geometry and exported bytes match. Origin is unconfirmed; no mesh
determinism change was attempted here. The older generic object test separately
expects the d6 proxy to be smaller than low, while current d6 proxy uses low;
that existing assertion was left unchanged. Dedicated new snapshot tests pass.

Include the three new frontend modules marble_autosave.js, marble_downloads.js
and marble_object_snapshot.js alongside the changed Studio/live/photo assets.
The live batch/alpha backend change requires a coordinated server/worker release.
Do not restart an active user exposure to publish this batch.

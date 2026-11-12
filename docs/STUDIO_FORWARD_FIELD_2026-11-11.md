# Opt-in Studio forward field in Phong

The browser can opt in before an ordinary RGB still or Live request with
`receiveLightField: true`; the existing MRP1/API contract carries
`receive_light_field: true`. Default/false requests retain their former
canonical configuration and digest. The existing Node Studio agent product
accepts the same explicit option. Orbit/camera sequences, tiles and non-RGB
jobs are rejected by the API when the field is requested.

## Data and accumulation

This is the existing production forward field, not sensor/image pixels.
Fixed-three transport stores direct 650/550/450 nm voxel energy as R32UI
counts. The export divides by the existing 65535 quantization factor and
uses the existing CIE/D65 three-line conversion directly. It does not apply
film-response inversion, clipping, exposure normalization, gamma, denoising,
new ray work or a new estimator. Signed out-of-gamut linear sRGB is retained.

Still jobs retain their existing additive field and overflow-clear behavior.
Live retains its existing fresh fixed-budget field per sensor sweep. Camera
sensor history/checkpoints are independent. The browser replaces a snapshot;
it never adds a cumulative snapshot a second time or invents temporal field
accumulation. More forward rays can make the existing additive field brighter.

The existing `uFieldVolume` / `uFieldGain` shader hook adds the sampled voxel
energy as glow. This is **not** directional radiance, calibrated irradiance,
correct surface BRDF illumination, refracted reflections, or a second ray
renderer. The limitation is stated in the UI and response metadata.

## Ownership, bounds and coordinates

The GL owner exports at most once per second into immutable
`light-field-<32 lowercase hex>.bin` artifacts, retaining at most four owned
files, reconciled across worker restarts. Exact owned incomplete files are
cleaned; symlinks/reparse points and unrelated files are untouched. A held
Windows reader prevents further growth. Metadata under
`status.light_field` has schema `studio.forward-field.v1`, exact job digest,
Live revision, sequence, sweep, ray/sample counts, dimensions, metre bounds,
SHA256, byte count and explicit color/accumulation semantics. Payload layout
is little-endian float32 RGB interleaved in z/y/x order.

`GET /api/marble_photo/field?digest=...&name=...` uses the existing `photo_read`
and job ownership gate, with no legacy unindexed-job fallback. It streams
only the current advertised artifact of an explicitly opted-in job. Stale
names/revisions, canceled/yielded/inactive work, malformed bounds, wrong
length and symlinks are rejected. Completed still fields remain available to
their existing owner.

Studio validates metadata/hash/length/finite samples, uploads RGB32F at the
already reserved texture unit 8, and applies a per-view affine map from
`vPosV` through inverse(view), Studio physical units/origin and field bounds.
Each draw's existing model transform is already included in `vPosV`. Outside
the volume contributes zero. `uFieldUseTransform=false` preserves legacy
sampling for other callers. Off, stale scene and failed uploads restore the
original complete zero-volume sampler, avoiding a broken WebGL draw.

The UI admits only the same static, unrolled scene, with resolved Room B and
Exposure A matching; it never applies an Exposure field to different Room
geometry/materials/lighting. Camera revision renewal replaces stale data.
Opt-out, scene edits and asynchronous cancellation cannot install an old
response. Existing saved-scene consumers still reject active field resources
rather than silently exporting an incomplete texture packet.

## Verification and deployment

- 153 focused Turing Python tests passed, including strict opt-in and default
  digest preservation, real HTTP owner/capability checks, exact live payload
  streaming and JS receiver contract.
- Node controller tests cover signed data, physical/view mapping, immutable
  replacement, no duplicate accumulation, camera rekey, cancellation/stale
  response, 404 recovery, failed GPU upload and complete fallback sampler.
- Existing photo camera/room/retouch, Live lifecycle, Studio boot/loading,
  subject renderer and Node-only agent product checks pass.
- One unchanged lighting regression expects the obsolete fresh B half-power /
  A-off defaults. Current source sets fresh sources to continuous 400 units;
  that assertion also fails on untouched HEAD e92e60f9 in detached
  C:/dev/pf-field-baseline and was not changed here.
- Spectral: 55 focused worker tests passed, with one symlink test skipped
  because Windows denied test symlink creation; actual RTX 3060 production field test
  (128 rays, 8 cubed, exact 6144-byte conversion, unchanged producer bytes);
  45 real native Phong mapping checks in three shader variants; project IR
  export and all five GLSL exports validated. Four shader/glass pytest gates
  passed. This is not a full-repository-suite claim.

Actual Chrome Studio acceptance used real Sessions, authentication, the
existing API handler and real GPU worker in an isolated local fixture. Two
UI-opted-in still jobs completed with finite, nonzero signed RGB fields:
96 cubed, 10,616,832 bytes, 131,200 forward rays, with hashes verified. The
actual draw sampled unit 8 with `uFieldGain=1`, `uFieldUseTransform=true` and
a finite nonidentity view-to-field matrix. Opt-out returned both controls
to zero without WebGL console errors.

This proves routing, upload and shader binding, not visible improvement.
The saturated wide view had no on/off pixel difference. Subject on/off
variation was comparable to off/off variation even with DOF disabled, so
no isolated appearance effect is claimed. Live's browser lifecycle has not
been exercised; its producer, API and controller tests passed.

Detailed receipt, draw probes and screenshots are in
`build/dot-studio-field-20261111/receipt.json` and adjacent files. The test
tab was closed and isolated server stopped, with listener absence verified.
No production service restart or publication is performed by this change. Publish backend, worker, canonical shader export,
JS/HTML and the complete current material asset catalogue as one coordinated
release after confirming existing jobs. Do not publish a checkbox against an
old API or an old shader. The current built assets lacked the new ribbon
material rows, discovered and refreshed only in the isolated browser fixture
through the canonical material registrar/MaterialDatabase.

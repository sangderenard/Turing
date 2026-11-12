# Scene-attached native response consumer

Status: source and CPU/offline compiler checks only. The current producer,
API and decoder retain `application_ready: false`. No GPU context, render,
exposure, screenshot, deployment or restart was performed. This is not an
end-to-end rendering or completeness receipt.

## Transport and explicit recording

The separate, default-off `phong-field-responses` control adds
`light_field_information: evaluated-responses-v1` only when field receiving
is also on. Ordinary field requests omit this extra-data option. The existing
authenticated field endpoint carries immutable `status.scene_responses`
snapshots. The receiver validates the outer digest/revision/name/size, the
native manifest and SHA256, then replaces the retained snapshot atomically.
It does not merge evicted rows back from an older snapshot.

`marble_scene_responses.js` retains typed native camera, forward, reactive,
scalar and optional application rows. Raw flux, unsigned scalar deposits,
spectral summands and parent totals are not interpreted as material radiance.
Typed-only snapshots do not allocate a GPU application texture or suppress
Phong. The native store's discard/availability declarations remain explicit.

## Prepared application path

The native `native-evaluated-rgb-v1` application row is 48 bytes:
recipient ID; world cell XYZ; direction bin; query context; component;
available original camera samples; original camera samples; signed linear
RGB conditional mean. GPU packing only sorts the seven identity keys and
copies the words. It does not alter RGB, integrate samples, reconstruct
optics or replay boundary factors. The current implementation admits
components 0, 1 and 2, not a parent-total alternative beside those components.

The shader uses the existing copied recipient attribute 7 and the declared
world cells/stencil. Camera propagation toward the scene is not forward
packet propagation. A query must be explicitly selected by the producer's
current camera/transport receipt, agree with the live revision and match the
actual raster base-eye origin at float32 precision. Cache order is not a
context selector. Camera/lens and film edits invalidate the captured query.

The three existing slots are retained:

1. Volume: copy component 0, the native pre-first-recipient camera segment,
   into a separate float32 target. The ordinary primary has no matching
   camera-air term, so its baseline for this component is zero. Opaque depth
   is copied; transparent fragments then replace the volume target with the
   nearest actual recipient, including zero baseline when data is unavailable.
2. Reflectance: blend component 1, the first-opaque downstream response,
   against the untouched ordinary linear primary using sample availability.
3. Glass: retain the existing sorted transparent scene traversal. For the
   first actual recipient only, blend component 2 against its ordinary
   premultiplied fallback as `c*F + (1-c)*P`, with alpha `c + (1-c)*A`.

The volume pass also copies recipient and exact `gl_FragCoord.z` bits into
an integer attachment. Components 1 and 2 require both copied values to
match. A cached complete camera response cannot be consumed by a farther
opaque or glass layer underneath nearer glass. Other layers keep ordinary
fallback. The final canonical resolve adds the retained volume once and
performs display conversion once.

RGB is not multiplied by availability twice. A measured zero with positive
availability can replace Phong; unavailable or evicted samples cannot.
Copied RGB bypasses the Phong BRDF, Beer/Fresnel, emission and color-matrix
equations. Only the uncovered transparent fallback evaluates ordinary
material shading. Direction-moment-to-light reconstruction and material-mask
primary suppression have been removed from the canonical shader.

Targets use RGBA32F, with explicit EXT_color_buffer_float and EXT_float_blend
requirements. Unsupported devices retain ordinary Phong instead of silently
clipping finite native float32 values through a half-float target. The optional
separate Phong specular override is disabled and labelled unsupported because
the native downstream component does not split that contribution.

Attachment allocation also requires the explicit 128 MiB target budget. The
preflight reserves 64 bytes per pixel plus 64 KiB, conservatively above the
48 bytes per pixel of actual attachments. It runs before attachment allocation.
An unsupported viewport retains ordinary Phong without silently resizing it.
The final secondary resolve evaluates the same canonical ACES rational
equation in reciprocal form above one, after the component sum, to avoid
intermediate square overflow for finite HDR float32 data. This is not a
per-component tone map; ordinary shader branches keep their original code.

## Boundaries still in force

- The current schema is not application-ready; activation remains blocked.
- The pre-first-recipient volume path does not cover surface-free/miss-ray
  camera-air contributions, which have no recipient/world-hit application
  key. Their native typed evidence is preserved. The current worker disables
  authored camera air; no background contribution is fabricated.
- An omitted raster recipient, unknown identity, unmatched query or unsupported
  projection cannot acquire availability from another material or nearby face.
- No readiness or convergence claim follows from sample count, RGB brightness,
  successful download, shader compilation or these source checks.

## Permitted verification performed

- `node tests/meowdoku_studio_light_field_check.mjs`
- `node tests/meowdoku_scene_responses_check.mjs`
- `node tests/meowdoku_response_lineage_check.mjs`
- `node tests/meowdoku_secondary_field_check.mjs`
- Spectral: `python -m pytest tests/test_studio_secondary_field.py::test_secondary_field_project_ir_export_and_es_link tests/test_base_material_webgl_compilation.py -q`

The Node tests use CPU data and mocked GL call recording, not a GPU context or
substitute renderer. The selected Spectral tests use the actual canonical
project-IR export and offline glslang compilation/linking; the hidden-GPU test
was not run. Ordinary fragment preprocessor tokens match the prior HEAD for
base, transmission/premultiplied-glass and shadow variants. The hazards CLI
refused a preexisting malformed filename; newest valid entries and the legacy
opening were read directly before testing.

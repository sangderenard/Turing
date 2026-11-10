# Region ownership shader integration (opt-in; GPU gate pending)

The region state is now connected to the actual composed forward and camera
shaders, not only the CPU acceptance reference. Legacy source variants remain
unchanged; an explicitly tagged new saved region design selects new variants.
No default activation, saved-object mutation, server restart or publication.

Saved design.region_media version1 contains region descriptors with start/count,
mode (host, glass_rod, injection_dye, air_bubble) and material ID. Region zero is
the host. It must accompany optical_media and replaces fixed boundary_media.
Turing checks immutable geometry ownership, closed/outward surfaces, spherical
uncoated host, containment, nonemissive dielectric material rules, host IOR for
injected dye, and matching bubble count. Overlapping different rod materials or
rod/dye target ambiguity is conservatively rejected by bounds; no precedence
is invented. Same-material rods, injected dyes and air bubbles may cohabit.

Scene preparation projects spectra into existing native MatBuf, preserves each
region's own identity through degenerate filtering/exact-sphere packing, and
encodes region tags in unused geometry metadata. Existing sphere kernels remain
intersection authority. Binding15 carries native-mapped region records. Runtime
selects opt-in shader variants for both forward and camera paths; camera geometry
reload regenerates/validates records and updates binding15. Changed region count
requires a new worker. Legacy no-region buffers and shader selection stay as before.

GPU ownership derives initial membership from each closed region's nearest
boundary. At a boundary it resolves candidate geometry membership across all
regions, so non-LIFO and coincident crossings are not handled by a guessed stack.
It reuses the existing native-index Fresnel/Snell/TIR helper for the resolved
before/after media. Reflection/TIR restore membership. Bubble union is air,
same-material rods form a union, and injection absorption adds to host extinction.
Homogeneous attenuation is exact Beer integration along the owned segment.
It is not a heterogeneous spatial dye solver or complex-index Fresnel solver.

Reference implementation recomputes membership at crossings and may be expensive.
It uses existing 1um ray offsets and is admitted only for radius>=1mm. Geometry
classification/numerical grazing behavior requires GPU validation before UI use.
The existing opacity and transmission defaults were not remapped silently.

Checks after final lifecycle edits: 62 Spectral tests passed, one optional old
GPU test skipped; 31 Turing tests passed. Full shader variants assemble, but no
claim of GL compilation yet. Added tests/studio_region_gpu_check.py to compile
both complete variants and run nine actual-GPU cases using original sphere
intersection kernel: origin containment, additive attenuation, bubble cutting,
restoration, TIR and reflection membership. This gate was NOT RUN: read-only
nvidia-smi showed GPU100% with an active user live render. No competing GPU work.

Remaining before release: run numerical GPU gate idle; independent review of
new integration; real saved scene render and performance evidence; browser
controls and safe authoring/restore; no general conflicting-rod cut ownership
or spatially varying dye field yet. Do not advertise either mode as live.

## Follow-up exact-sphere admission and scale gate
Fixed `saved_analytic_spheres` to validate shell/air analytic memberships against versioned region descriptors when that contract is present. Added full immutable-store MRP1 decode regression with a 24,320-face host plus bubble; wrong region ownership rejects. Sphere and region suites: 13 passed.

The pending GPU gate now checks 1 m, 25 mm and 1 mm host radii and contains a strict additional 0.4 micrometre bubble case. That case is expected to expose the existing fixed 1 micrometre ray offset/minimum-hit limitation; it is not marked xfail. Current admission does not prove minimum feature thickness. No general overlapping-volume correctness or live readiness is claimed. Resolving the classifier alone would be insufficient because canonical continuation and exact-sphere intersection also use fixed offsets. GPU gate remains unexecuted while the user's live render occupies the GPU.

## Coordinated scale-aware traversal follow-up
The opt-in region shader now derives one shared positional budget from float32 world coordinates (two machine epsilons, with a 1pm near-origin floor). Origin reclassification, triangle minimum hit distance, exact-sphere minimum hit distance and forward/camera continuation all use that same budget. Triangle parallel rejection is relative to both edge lengths and ray length rather than a fixed 1e-8 determinant, which otherwise rejects tiny valid faces. Default shader variants remain unchanged.

The strict GPU thin-bubble gate now runs at z=0.18m as well as radius1mm, with bubble radius0.4um. It remains unexecuted while the user GPU is active. This is a candidate numerical fix, not proof of arbitrary sub-ULP/tangent/self-intersecting geometry. Finite-precision unresolved features still require fail-closed admission or explicit diagnostic before broad release. CPU suite:63 passed,1 skipped; Turing39 passed. Reviewer requested to inspect coordinated event assumptions.

## Offline shader and authoring receipts
Both complete generated region forward/camera shaders pass installed official Vulkan SDK `glslangValidator -S comp` with exit0. Reproduce via Spectral `python -m tests.studio_region_shader_compile_check`; this creates no GPU context and is not a substitute for NVIDIA linking or numerical transport tests.

Added isolated `web/marble_optical_authoring.js` (not wired into live UI) to derive region contracts only from canonical component membership and accept explicit authored reference-thickness spectra. `node tests/meowdoku_optical_authoring_check.mjs` passes both material modes, complete ownership, immutable geometry and independent spectrum copies. Snapshot/UI activation remains gated on transport validation. Latest active job remains live with GPU99%; no active work interrupted.

## Rod partition investigation after user clarification
User requests actual equal spatial partition of overlapping rods, not blended materials or precedence. Existing axial generator has identical superellipse cross-sections around a common curved local-Y centreline, rotated evenly through pi and twisted together.

A reproducible CPU experiment, Spectral `python -m tests.studio_rod_partition_geometry_check`, uses the exact superellipse equation and 2880 angular by128 equal-area radial samples. Nearest-blade angular ownership gives equal total sampled cross-sectional area for2–6 symmetric rods, but fails equal shares of each individual shared zone for4+ rods: a zone shared by rods0,1,2 can be wholly assigned to rod1. Thus angular bisectors are not equivalent to the strongest interpretation of equal partition among every zone's participants. Parent informed before implementation; no material partition policy activated. The rule also needs a generalization for tubes/confetti without a shared axial frame. Latest GPU95% remains active.

## Approved practical axial partition groundwork
User confirmed angular bisectors following twist and accepted practical equality for complex overlaps. Added Spectral `camera_designer/studio_rod_bundle.py`: actual occupied rod candidates determine one material owner; pairwise bisector surfaces follow the existing curved local-Y centreline and twist, with analytic gradients. A CPU derivative-bound interval isolator finds internal crossings without relying on a fixed sample grid; grazing/coincident ambiguity and work exhaustion explicitly reject rather than silently losing an interface.

`python -m pytest tests/test_studio_rod_bundle.py -q`:7 passed. Includes transition within two overlapping old volumes, analytic normal finite-difference agreement, five crossings of a strongly twisted divider and explicit coincident-ray failure. This is CPU geometry groundwork, not live transport: native/GPU nearest-hit event integration, common authoring frame serialization, Phong cut geometry and numerical GPU verification remain. Generic nonaxial bundle policies are not invented. Reviewer notified. Latest GPU40% remains active; no disruption.

## Native/GPU separator integration and actual worker receipt
Implemented bundle affine/recipe records on binding15, dynamic occupied-rod ownership and twist-following separator events in both forward and camera nearest-hit traversal. Actual internal cut hits reuse native dielectric Fresnel/Snell/TIR, eta-squared camera transport and existing Beer extinction. Unresolved root isolation or exhausted retry caps fail closed and increment a diagnostic counter; worker snapshots expose rod_bundle_diagnostic. Host triangle representative is precomputed and validated, avoiding per-event full-scene scanning.

NVIDIA GeForce RTX3060 verification after confirming user render canceled/no active photo worker:
- Region gate36 cases passed at1m/25mm/1mm and translated0.18m studio with0.4um bubble.
- Rod event gate8 cases passed at25mm/1mm: internal cut, transmission/reflection, eta-squared, nearest-bound clipping, explicit coincident failure.
- Canonical JS full96x128 meshes2/3/4/7 rods (72960 through194560 triangles) passed actual native packing/BVH/full-camera traversal probes.
- Actual production worker test, canonical3rod + real emissive rectangle,128px,2epochs: complete,0unresolved samples, nonblack image. First run12.703s preparation+18.813s rendering,35.14s wall. After pruning inactive/same-material separator pairs:11.203s preparation+18.875s rendering,33.438s wall. No demonstrated render speed improvement. Image was inspected: noisy low-sample smoke result, not calibration/convergence or visual parity proof. Receipts at spectral-analyzer/build/rod-bundle-worker-gate[-optimized]-20261109/receipt.json.

Authoring bubble mode now permits inclusion intersections only under explicit optical_authoring v1 glass_rod/injection_dye; legacy bubble recipes and digest remain unchanged. Opt-in cut geometry marks shared-volume mechanics approximate rather than falsely claiming exact mass/inertia. Live runtime admission remains gated pending review and persistent browser warning/UI completion.

## Experimental photo-first release and production verification
Independent optical review cleared explicit experimental photo-first opt-in after lifecycle/retry/diagnostic fixes. Validated axial bundle members are now admitted; unspecified conflicting rods still reject. Both UI and server reject optical LIVE requests. Legacy remains default and saved original surface rows restore on return to legacy. Illustrative color-derived starting spectra are explicitly uncalibrated, editable and never described as measured material data. Browser invalid-image warnings persist until a replacement valid image/revision is displayed.

Eleven optical static assets published through a scoped manifest, preserving other published/source differences. Source export/no-store lists updated. A later separately authorized local-ray diagnostic overlay changed bootstrap content; it was preserved. Verified HTTPS with build/meowdoku/tls/ca.crt, never disabled certificate checks.

Guarded production8766 restart: no active photo/preflight/test worker immediately before stop; exact original command and retained-fountain receipt used; oldPID45412 -> newPID23324. Unrelated8767 PID15712 remained untouched. HTTPS Studio200 and local edit tier verified.

Real production API check passed via standard X-Meowdoku-Marble header and bit-exact binary mesh transfer (JSON geometry transfer was unsuitable because it loses signed zero). Explicit optical LIVE rejected with photo-only error. Actual opt-in glass-rod photo3aa74cbeb8b1ed0567a3508ec9fa038fa9a75e33ad95b2826d7ad433323f352c completed with0unresolved cut samples,8epochs,0.579s rendering,2.937s preparation,6.14s API wait. PNG fetched and visually inspected; low-sample smoke only, not quality/convergence/calibration evidence. Only the newly created temporary verification object was archived out of the production object library, retaining its immutable record under the test receipt; no pre-existing object changed. Receipt: build/dot-optical-controls-20261109/production-check/receipt.json.

Known unchanged baseline: distant D6 proxy triangle-count assertion fails identically with prepublication modules. New optical save/load/re-save, browser desktop/mobile, legacy restoration,8bubble cohabitation, invalid-image lifecycle and autosave tests pass. Phong remains an approximation and shared-volume mechanical mass/inertia is not claimed exact.

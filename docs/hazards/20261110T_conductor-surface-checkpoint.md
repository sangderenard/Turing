# Conductor surface transport checkpoint (source only)

The staged ideal spectral mirror is restricted to authored roughness exactly
zero. Nonzero-roughness catalogue metals, including polished presets at .005,
retain their existing path. No preset or saved material was rewritten.

Actual production forward-field GPU proof: finite mirror wall and receiver,
unchanged baked source rays at650/550/450nm. Gold reflected-field ratios against
white: .965998/.716652/.274629. Blue: .132520/.318221/.865737. Whole direct bypass
field arrays are exactly equal. This proves forward reflected field tint at
zero roughness, not full camera image convergence or calibrated conductor data.

New studio_conductor_surface.py provides CPU and GLSL complex n/k Fresnel and
GGX NDF sampling with f*cos/pdf weight, null lower-hemisphere events and an exact
zero-roughness delta. Three CPU tests cover normal Fresnel, white-furnace bounds
and direction/color independence. An isolated GPU test passes1024 CPU/GPU cases.
Small-roughness float32 cancellation was corrected using tan-squared sampling.
This kernel is NOT yet attached to production transport.

Integration remaining: bind existing MaterialDatabase PBR tensors explicitly to
forward and sensor shaders, indexed by existing native material IDs. Do not
repurpose diffusion or remission fields to carry roughness. Route only the
explicitly selected advanced stage/room surfaces; keep diffuse cloth separate.
Preserve canonical n/k when authored; artistic RGB spectrum is not measured n/k.
Disk fibonacci_satin_gold has named color plus enamel, not a measured full n/k
curve, and must not be described as one. Camera accumulation must be tested in
addition to forward field transport before release.

## Coordinated integration, still unpublished

Explicit advanced stage/room ranges now flag opaque full-metal faces per triangle;
legacy faces stay unflagged even if the material ID is shared. Native canonical
PBR16 is uploaded at binding18 (16/17 belong to first-hit loans/verifier). The
buffer has separate ownership/destruction, is refreshed on camera/dynamic scene
rebuild, and is included in retained-work signature and byte hashes. Admission
checks precede allocation. No diffusion/remission fields were repurposed.

Forward, reactive and camera paths now share GGX direction sampling and weights.
Explicit canonical n/k authority has its own flag, including k<=.1. Canonical
full-metal sources without explicit n/k reject rather than treating the native
reflectance-amplitude field as a power coefficient. Artistic catalogue rows
use the same existing authored RGB spectral projection, not claimed measured.
Inclusive24-bit RNG is mapped to half-open uniform before sampling; exact1 is
covered by CPU/GPU parity. Rough cloth stays outside the conductor path.

Verified:64 spectral/material CPU,8 Turing scene-engine,11 dynamic-reader tests.
1024GPU law parity cases include exact RNG endpoints. Actual forward field has
12occupied receiver bins at roughness0 and625 at.4; direct bypass arrays remain
identical while reflected gold/blue paths tint. Actual camera-to-conductor-to-
emitter tests pass with field and NEE absent, both roughnesses. An authored
n1.5,k.01 synthetic fixture gives ~.0400 response rather than its .9 authored
reflectance, demonstrating explicit n/k authority. First-hit continuation GPU
bitwise parity and existing mutation/cancel/expiry tests pass. Dedicated new
native-buffer mutation coverage is being run separately.

Still awaiting independent final review and coordinated release. This is an
isotropic single-scatter surface model, not measured coating/cloth, microfacet
multiple scattering, polarization, or a fractional-metal blend. No published
assets, saved user materials, or running server were changed by this worker.

Release safety refinement: ADVANCED_CONDUCTOR_RUNTIME_ENABLED remains False in
studio_scene_engines.py because newly spawned workers import source immediately.
The explicit test masks exercise the complete implementation; normal job routing
remains on existing behavior until the reviewed release flips this switch.
Dedicated separate-native-buffer roughness mutation GPU test now passes as well.

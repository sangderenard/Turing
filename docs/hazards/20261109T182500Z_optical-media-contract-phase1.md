# Phase 1 optical media implementation (source only)

Added opt-in `design.optical_media` version 1. Each material record has
`material`, increasing `wavelengths_nm`, `transmittance` in (0,1],
`reference_thickness_mm`, and nonnegative `concentration`. Internal
transmittance excludes interface Fresnel losses. Existing PBR IOR remains the
refractive index. The definition is immutable/content-addressed with design;
the exposure request cannot substitute it.

Spectral validates exact coverage of all explicit volume materials, opacity
zero/nonmetal/nonemissive requirements, units, finiteness, native float range,
and wavelength coverage. It interpolates absorption over wavelength and feeds
existing native MatBuf n/k records. Material cache identity includes the new
optical definition. Legacy saved materials remain byte-for-byte on their old
conversion path. Surface roughness does not enter the new bulk coefficient.

Turing admits explicit ordinary no-bubble inclusions only when this optical
contract is present. Phase 1 conservatively requires contained inclusions with
disjoint axis-aligned bounding boxes; a rejection is not proof the actual
solids overlap. Existing bubble exclusions remain. This is groundwork, NOT
implementation of the requested general cut-volume or crossed-ribbon modes.

No authoring UI or default migration was added. Nothing was restarted,
published, or applied to a user's saved design. The feature is opt-in through
a properly authored saved record plus boundary map. It is not exposed by the
current Studio page yet. New user-facing glass-rod and injection-dye modes
remain pending volume ownership integration: rods are distinct n/k media;
injected dye changes host absorption without fictitious optical interfaces.

Checks: 22 Turing tests passed (optical admission, bubbles, photo bridge).
27 Spectral tests passed; one optional GPU test skipped. New tests verify
reference thickness, thickness doubling, concentration, clear limit, sigma
interpolation, no extrapolation, roughness independence, invalid/overflow
rejection, native MatBuf extinction, and cache identity. Turing tests cover
legacy isolation, no-bubble admission, conservative overlap and containment.
No real GPU slab image or browser test was run, so no rendered correctness claim.

Files:
- Spectral camera_designer/studio_optical_media.py (new)
- Spectral camera_designer/scene_builder.py (optical input + native conversion)
- Spectral camera_designer/studio_photo_geometry.py (forward immutable record)
- Spectral tests/test_studio_optical_media.py (new)
- Turing examples/meowdoku/marble_media.py (opt-in disjoint admission)
- Turing examples/meowdoku/marble_photo.py (forward saved optical contract)
- Turing tests/test_meowdoku_optical_media.py (new)

## Review fixes and next integration reference

Independent review found two defects in initial opt-in scope. Both are fixed:
- New optical maps must pass closed consistent manifold winding and outward
  signed volume per connected component. Duplicate parametric seam/pole vertices
  are welded only for admission at declared 1e-7 radius-unit precision; transport
  coordinates are untouched. This is not a self-intersection proof. Reversed,
  locally reversed and open surfaces reject; actual canonical ribbons pass.
- Real-index dielectric Fresnel is only a weakly absorbing interface model.
  Authored spectra with imaginary index k>0.001 anywhere over their piecewise
  interpolated range reject pending complex-index boundary transport. This is a
  declared approximation limit, not a claim of exact complex Fresnel.

Final review-fix checks: 26 Turing tests passed; 28 Spectral passed, one GPU skipped.
Reviewer separately reran focused suites and reproduced rejection of both defects.

Added spectral `studio_region_media.py` as a CPU acceptance reference (not wired
into renderer). It preserves per-volume, non-LIFO membership; same-material rods
form a union; injection dye memberships retain additive concentration entries;
air bubbles form an air union and mask dye while retaining underlying membership;
reflection/TIR do not change state. Coincident event groups resolve atomically.
Different-material rod overlap and rod/dye overlap require authored ownership or
injection target and reject instead of silently choosing a material. Nine tests
passed for these rules. No user state or existing renderer path changed.

Actual integration still needs geometry-derived initial membership, durable
per-volume identity in GPU records, shared forward/camera event state, and
resolved segment extinction. Existing native CPU/GPU paths both track one scalar
medium and fixed triangle side pairs. Existing ScaleContext sphere interval
walkers do not partition arbitrary overlapping material regions and cannot be
reused unchanged for this purpose. Reuse their intersection machinery and native
n/k laws, not their overlap handling.

Homogeneous segments use exact Beer integration. Heterogeneous injected dye
requires spatial concentration integration; this reference does not supply it.

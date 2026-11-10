2026-11-09 AXIAL ROD BUNDLE AUTHORING (source-only; runtime remains gated)

marble_mesh.js now records the exact common inclusionFrame already used by
orient: rotate about the triangle-weighted inclusion centroid, then apply the
existing origin-centered envelope fit. Face/normal arithmetic is unchanged.
The frame has origin, axes (generator X/Y/Z unit vectors in shell coordinates)
and scale, with p = origin + scale * sum(q[i] * axes[i]). Hybrid's existing
radial fit also updates the metadata, but hybrid bundle authoring is refused.

The pure marble_optical_authoring.js bridge emits optional region_media v1
axial_bundle only for axial cat_eye glass_rod families with 2..7 rods:
regions (ordered region indices), eye_length, eye_curve, twist, phase, frame.
Injection dye, single rods and separated legacy recipes have no bundle.
No live control, saved-object state, renderer selection or process changed.

saved_axial_bundle(design, regions, vertices) independently validates the
descriptor, recipe values, canonical ordered ranges, generator orientation,
centroid/fit frame, axial extent and common symmetric ribbon family. Actual
canonical j=0 ring vertices bind twist/phase to geometry, so changing both
recipe and descriptor cannot reorient cuts while retaining old triangles.
Geometry comparisons use the existing float32 shell-coordinate budget; this
is not proof of arbitrary sub-ULP surface resolution or transport accuracy.
The ordinary saved_region_media path validates a bundle then still raises
"axial rod bundle runtime is not enabled". No conflicting-rod ownership gate
has been relaxed. Native cut events and their validation belong to the
separate Spectral integration; authoring alone must never activate them.

Verification:
- Node optical authoring: exact transformed vertices, envelope fit, copied
  metadata, mode/shape restrictions, full face ownership, no mesh mutation.
- Existing Node marble mesh and axial-default checks pass, including the
  preserved legacy geometry digest and new/legacy saved-object roundtrips.
- 68 focused Python tests passed (axial authoring, region media, optical
  media, bubbles, exact sphere admission). Actual generated cases include
  2/3/4/7 rods, identity/oriented/fitted frames, full 96x128 canonical detail,
  and 128 longitudinal rings. Schema/recipe/frame/range/geometry corruption
  rejects; all valid bundles remain rejected by the runtime admission gate.
- JS syntax, Python compilation and scoped tracked whitespace checks pass.
No GPU test, browser session, exposure, service restart, publication, commit
or push. These are focused checks, not a full repository test result.

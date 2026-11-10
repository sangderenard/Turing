# Ribbon material contract audit (2026-11-09)

Read-only production audit; no renderer/default changes, server restart, or GPU job.
Current local sources, not remote history.

## Measured findings

`material_db.py::_pbr_to_ray_record` ignores PBR transmission. CPU-built
raymat_compat records for green ribbon transmission 0, 0.25 and 1 were bytewise
identical. Scene registration separately emits spectral transmittance and flags,
so this is specifically a mismatch in the Studio compatibility shader path,
not a claim that transmission is unused everywhere in Spectral Analyzer.

Ray absorption is derived from roughness and metallic:
refl = metal*0.85 + (1-metal)*(1-rough)*0.12;
diff = (1-metal)*rough; absorption = max(0,1-refl-diff).
Blue ribbon absorption is 0.7744; green/amber 0.7656. It has no authored
reference thickness or measured extinction spectrum.

Studio scales radius-mm / 1000 before transport. Applying its existing
exp(-absorption * distance_m * (1-color)) law to a 1 mm green-ribbon segment
transmits [0.9992959,0.99979331,0.99953309] in the reference RGB basis,
excluding Fresnel interfaces. At 10 mm: [0.99298123,0.99793501,0.99534073].
These are CPU calculations of current conversion and law, not measured renders.

Opacity on legacy refractive surfaces means p_reflect = F + opacity*(1-F),
plus opacity-weighted surface lighting. It is not pigment concentration.
At opacity >= .999 an opaque branch is used. Increasing opacity is not a
physically defensible repair for too little dye absorption.

Ordinary no-bubble materials use a legacy shader that applies tint using the
segment before the hit and the hit material, without tracking residence in
that material. Refraction assumes air/material at each surface. It cannot
correctly predict arbitrary nested glass ribbons. Explicit positive/negative
media and actual segment Beer attenuation already exist in
camera_designer/studio_boundary_media.py, but Turing's saved_boundary_media
admits them only with bubbles and disjoint interiors. Centered crossed ribbons
with multiple strands are explicitly rejected from that path.

## Existing components to reuse

OpticalMaterial already defines n and k; spectral MaterialDatabase contains
per-band ior_real/ior_imag. studio_medium_transmittance applies
exp(-4*pi*k*frequency*distance/c). No parallel bespoke material system needed.

For a predictable absorbing dye, author internal spectral transmittance T_ref
at declared reference thickness d_ref, excluding interface Fresnel losses.
Convert sigma_a = -log(T_ref)/d_ref and k = sigma_a*lambda/(4*pi).
Then T(2*d_ref) = T_ref squared. Concentration may multiply sigma_a for a
Beer-Lambert mixture within its applicable regime. Roughness must not change
bulk absorption. Keep interface IOR and roughness independent.

A truly cloudy/opaque inclusion needs separate scattering coefficients and
phase behavior, not a renamed alpha. Dissolved dye and scattering pigment
are distinct material targets; no universal real-world dye default exists.
Three-band authored values can be explicitly labeled an approximation rather
than a measured full spectrum.

## Required implementation order and acceptance

1. Author and serialize reference-thickness transmittance or native n/k bands
without repurposing the existing transmission/opacity fields silently.
2. Extend explicit medium ownership to ordinary shell/inclusion scenes and
resolve intersections consistently (shared volume or explicit composition
policy). Do not infer an air gap at a glass/glass boundary.
3. Use the existing n/k propagation for each segment in forward and camera
paths, including reflection/TIR residence. Preserve saved legacy definitions
under a versioned migration.
4. Gate with slab 1x/2x thickness, concentration scaling, zero absorption,
index-matched inclusion, host/inclusion/host, reflection/TIR, interface energy,
and separate forward/camera path tests. No claim of physical calibration until
these gates and a measured material reference are satisfied.

No default changed because a stronger ad hoc opacity/tint would hide rather
than fix the missing material contract and volume ownership.

## Bounded implementation proposal after geometry inspection

Existing mesh metadata already lists every closed ribbon's triangle range and
material. Axial geometry uses common centerline and evenly rotated bands;
materials alternate eye/accent. This gives usable volume identity, but not a
physical composition for their overlap.

A first release can extend saved boundary ownership beyond bubble-count>0 to
validated disjoint interiors, without changing ABI. Material extinction must
be saved as a digest-covered, versioned optical record and converted into the
existing MatBuf n/k bands. Test default legacy import separately from explicit
new optical definition. This release does not fix overlapping axial ribbons.

A second release needs per-volume membership, not merely a stack, since
intersecting volumes can exit in a different order from entry. Keep volume IDs
and active membership along each ray; use geometry identity, not material ID,
for crossing events. Equal-material overlap forms a union: internal crossings
change membership but cause no optical interface or double concentration.
Different-material overlap needs author-defined physical meaning. Additive dye
concentrations in a common base glass can sum absorption, but may not share the
same n; an arbitrary priority/last-entered rule cannot be called a dye model.
Alternatively construct nonoverlapping fused regions with real interfaces.
The choice changes the object and must not be silently invented.

Acceptance scenarios for membership: A-enter/B-enter/A-exit/B-exit,
A-enter/B-enter/B-exit/A-exit, camera initially inside A+B, tangent and coincident
surfaces, reflection at entry (membership unchanged), TIR at exit (unchanged),
same-material A+B union, mismatched indices, shell/air bubble/dye transitions,
and deterministic replay with identical frozen random values. Both forward
and camera shaders must consume the same boundary and spectral records.

Existing CPU boundary suite re-run: 8 passed, 1 optional GPU test skipped.
This checks current range validation and conversion only; it does not validate
intersecting ribbons or establish correct bulk dye behavior. No new GPU job ran.

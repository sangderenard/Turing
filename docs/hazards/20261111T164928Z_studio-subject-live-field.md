2026-11-11 STUDIO SUBJECT FIELD / PROGRESSIVE LIVE FIELD (local, not yet published):

The running release is still the v3 receipt documented in
20261111T161106Z_studio-directional-field-live.md. V4/v5 replay accounting,
category opt-ins and subject ROI were staged; v6 live integration is now source
frozen and awaiting real browser acceptance plus coherent publication. Do not
confuse source or staged assets with deployed assets.

Actual subject acceptance receipt:
build/dot-subject-field-20261111/subject-receipt.json.
Twelve directions, exact 0.5 mm pitch, 64 cubed cells, 32 mm subject cube,
requested 1 mm margin and actual 5.5-6.5 mm per-face padding. Full scene retained.
All 262144 cells were zero: 4464 fixed baked roots, 8192 recorded jobs, one
normalized complete sweep, 128 initial partial jobs excluded. Subject-only
Phong image is dark, with gain0/1 pixel difference zero. This is absent measured
coverage, not a successful relighting or evidence of a broken glass interface.
Primary-only context differs by up to11/255 between display paths; exact whole
image parity outside the ROI is not claimed.

The real source is native_emissive_sources -> RayTracer.submit_emissive_triangles:
144 ceiling fixture emission triangles, fluorescent_cool_white profiles,
gain400/white filter. Fixture-local materials17/18 map to native16/17; they are
not the similarly numbered subject catalogue slots. Preview-stage point source
is suppressed when physical fixtures exist. Repeating fixed roots does not
increase angular coverage. Native interaction-target sampling currently has an
inconsistent PDF/weight and was not used.

V6 exposes bounded source budget/seed and fresh angular populations through the
same live worker. The existing sensor still computes; field-only UI skips its
image fetch/overlay. Completed per-root-corrected raw energy/direction moments
are averaged across populations. Latest source table is explicitly latest-only;
metadata carries seed sequence, population count and rolling source hashes.
Unchanged completed populations are not redundantly republished.

New v6 fixed-point scale16777216 is decoded with the same divisor, not a gain.
At requested32768 roots the actual fixture allocation is32832; representative
0.5 mm crossings with two replays would be only0.31-0.62 old counts, now80-158.
Overflow still rejects the population. Legacy v3-v5 and scalar scale65535 stay
unchanged. Native proof96passes/1environment skip; API253 focused tests and
actual9,449,824-byte v6 stream/hash validation; browser client decoded same
artifact with3populations/34nonzero fixturecells. These fixture results do not
establish subject coverage in the upcoming live run.

Internal glass recording is a separate helper under test, not in frozen v6.
Existing region transport already uses Snell/Fresnel/TIR and Beer attenuation.
The observer must integrate entry throughput along actual internal segments,
not remove an exterior guard after whole-segment attenuation. No fog, artificial
illumination, material tint adjustment or particle scattering was introduced.

No commit, stash, forced restart or security change. Concurrent work preserved.

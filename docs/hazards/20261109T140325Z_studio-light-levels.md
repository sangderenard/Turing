2026-11-09 LINKED STUDIO / EXPOSURE LIGHT LEVELS (MEASURED, local Windows source):

All seven original sliders remain the authoritative Studio strengths and are
linked to photo exposure by default. The collapsed optional menu exposes a
per-light link checkbox plus Studio and exposure numeric fields. Unlinking
starts at the existing value; re-linking uses the current Studio value. Merely
opening or closing the menu never changes the rig. A null stored override is
linked; profiles without the optional strengths field normalize to all linked.
Existing v1 browser storage is retained. Profile validation occurs before any
control writes and bounds exposure values against the actual sliders.

computeFixtureRoles selects the relevant channel. The existing WebGL direct
light registration and real fixture PBR rows consume Studio power. The actual
sceneForPhoto fixture geometry/material packer consumes exposure power, so
photos and the live ray-feed callback use those real emitter rows. No extra
light rig, gain multiplier, or renderer was introduced. At zero, each lane
independently preserves its existing black-felt panel behavior. Geometry,
filters, poses, material IDs, housing and room paths remain shared.

The strength component owns invalidation for its controls. Only an effective
exposure change marks the photo stale and invalidates live ray history. An
unlinked Studio-only edit does neither. Existing global capture listeners skip
these owned IDs; unrelated controls keep their existing behavior.

Validation: the new meowdoku_marble_light_strengths_check.mjs executes the
production fixture builders, actual WebGL material writer and direct-light
registration loop, and actual photo wire packer. All seven lights passed
independent isolation, zeros, link/unlink, invalid numeric input, and event
ownership checks. Compared with the before-edit local source snapshot, default
scene JSON, preview PBR bytes and direct-light uniforms are exactly equal.
Default fixture-scene SHA256 is
67c81fedbd10ccbbe2e9ec87e5a8b17be229688c786d23b01fb431e8304b66a1.
The strengthened Studio integration test passed new/legacy profile save/load,
all-seven split values, zero override and atomic invalid-state rejection.
Existing lighting profiles, poses, photo room, room geometry and axial defaults
Node tests also passed. JS syntax and scoped tracked-file whitespace checks
passed. No full suite, GPU exposure, export, restart, commit or publication ran.
Real-browser integration QA is coordinated by the parent task separately.

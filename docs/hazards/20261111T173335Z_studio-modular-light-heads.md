2026-11-11 MODULAR LIGHT HEADS / KICKER SIDE POSITION (Windows, source only):

Current source HEAD 34eeeeea with concurrent uncommitted Studio work preserved.
The existing fixtureSizes records now optionally select native, round, or
softbox-1x4 heads. Omitted heads preserve saved geometry exactly. Fresh kickers
use 1 x 4 Studio-unit boxes at the existing one-foot default unit, with opaque
2 x 8 grids. Selecting the preset through the existing controls converts one
foot to the current Studio unit; subsequent dimension/scale edits remain
ordinary authored resizing. Unsupported extreme scales fail instead of
silently shrinking the preset. Head changes replace incompatible tools and
clear superseded per-head Room overrides. No new renderer, light source,
material law, backend, service, or scene builder was introduced.

The existing shared fixture role/material IDs, real box/quad/grid triangles,
pose records, scene packing and MRP1 route remain authoritative. The existing
radial view already supports -180 to 180 degrees; 90 degrees puts either
kicker at its own side while preserving height. New rectangular default
placement no longer asks for a nonexistent circular radius. Whole-emitter
table clearance now uses rectangular extents; the circular snoot guide
explicitly leaves grid aperture clearance unverified.

Eight focused Node checks passed: modular_light_heads, marble_light_tools,
marble_fixture_sizes, kicker_guide, kicker_guide_controls, marble_light_stands,
studio_fixture_components, and studio_agent_product (each under tests/ with
the meowdoku_ prefix except studio_agent_product). New tests exercise all ten
head mounts, physical preset conversion, failed-edit rollback, incompatible
tool replacement, saved-state reload, radial side poses, actual fresh floor
fit, finite meshes, preserved ownership/materials and exact scene/MRP1 equality.
The separate head test produced 5448 fixture triangles and a 6422010-byte
MRP1 payload. Each kicker has exactly two emitting triangles; its grid uses
the existing zero-emission housing row. No GPU exposure or browser screenshot
is claimed by this receipt. Scoped diff whitespace check passed.

Broader harnesses are not green: marble_studio_lighting stops at its existing
half-power shop-light expectation; marble_light_stand_settings passes its
all-ten-default-fit checks then fails at a VM missing withSceneCategory;
underlight_fit fails at a VM missing boxHousing. These harness issues were
not patched here, and this is not a whole-repository pass. The hazards-log
reader still refuses a preexisting malformed filename; newest valid entries
and the legacy opening were read directly.

Publication is coordinated by the existing field-release worker. This worker
did not restart, deploy, commit, push, change network/security settings, alter
the retained field-test profiles, or submit a new GPU job.

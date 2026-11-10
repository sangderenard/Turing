2026-11-09 ADVANCED TWO-SIDED GOBOS (MEASURED, local Windows source):

Advanced room controls contain a second collapsed Gobos disclosure, empty by
default. Add creates an editable row; at most eight independently enabled
rectangle/ovoid flags have full width/height, world XYZ, pitch X/yaw Y/roll Z
and separate catalogue front/back materials. The rotation convention is
Rz * Ry * Rx in degrees, front +Z at zero. Panels are closed 0.12-unit solids,
not coincident triangles with competing materials; the rim uses the front
material. This retains more than the tracer's 0.1 mm self-hit guard at the
minimum saved 1 mm radius. There is no new material identity or shader path.

Preview uses the same faces as sceneForPhoto (also live rays), partitioned
by the existing PBR opacity/transmission rule into opaque furniture and the
shared furniture/subject transparent sort. Materials come from the actual
live PBR rows. The UI discloses that Phong does not trace gobo shadows.
Number inputs retain the last valid scene while an intermediate/invalid edit
is visibly marked. Spawn/hide/remove refresh the scene and ray history.

Lighting profiles save optional versioned gobo state. Missing state loads
empty; older profiles are never rewritten on read. Validation finishes
before any scene field changes. Table, simple surfaces, Advanced backdrop
backups, light poses and split exposure/Studio strengths are preserved.
Saved marble recipes, geometry, digests and material IDs are unchanged.

Measured focused checks: gobo geometry/editor validation, closed topology,
front/back ray hits at zero and compound rotations, elliptical corner holes,
row limits, edit/hide/remove, and invalid/legacy state; exact profile roundtrip,
all-other-state preservation and corrupt atomic rejection; actual production
sceneForPhoto plus MRP1 packing and actual Python decode_upload. Eight ovoid
panels with all ten fixtures, six circular mats and the table produce 3600
triangles and 14 compact PBR rows, below unchanged limits 5000/64. The actual
preview furniture partition exactly matches those same face arrays.

Existing material-ID restoration, light poses/profiles/strengths, table,
room, photo/retouch, live-ray lifecycle, bounce mapping checks pass. Python
focused suite (gobos, room/table decode, bounce admission): 13 passed.
System Python311 also passes the existing Spectral ten-light source derivation
and new gobo decode tests (2 passed). Turing's .venv lacks torch and cannot
import camera_designer/neural_assembly.py (nn is undefined); the same tests
pass with the installed system interpreter. No dependencies were changed.

Isolated real Chrome QA passed: nested collapsed placement, multiple rows,
fractional edits, invalid-width warning, visible red front/blue back after
rotation, rectangular/ovoid silhouette, hide/remove, and exact profile reload.
Screenshots are build/dot-gobo-ui-qa-20261109/front.jpg and back.jpg. The
isolated server and QA tab were closed. Actual ray/photo GPU exposure was
not run for gobos; CPU ray intersections and real upload decode passed.

The final regression sweep reached the concurrent auto-bounce worker's
in-progress live harness (missing manual/auto mode) after all gobo/scene/
profile/material tests passed. That worker owns its harness update; this
failure is not attributed to the gobo change. No full suite, deployment,
production export, server restart, saved-state write, commit or push was run
by this worker. Only the new helper is added to the normal export list.

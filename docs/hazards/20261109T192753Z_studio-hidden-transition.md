2026-11-09 HIDDEN STUDIO TRANSITION AND REAL PHONG STAGING (MEASURED, canonical Windows):

Source-only, not published or activated. No production restart, exposure,
user-scene edit, camera movement or physics selection was performed. The
existing user tab was inspected read-only; the GPU check used its own loopback
test tab, now closed, and owned test server PID79296, now stopped.

marble_studio_transition.js owns a detached candidate only. Exact structured
state plus monotonic input/change/lens revision is checked after preparation,
after renderer staging and synchronously before transfer. Edit/revert,
unversioned state changes, cancellation/supersession, lost renderer readiness,
late native completion, cleanup failure and one-use transfer are tested.
Cancellation waits for in-flight work before releasing native resources.
takeReady never starts physics, changes the camera or installs renderer state.
Both actual consumers must acknowledge the same revision and native step.

captureCurrentStudioScene in marble_studio.js captures the actual last displayed
subject model/view/FOV and explicit Studio lighting, never sceneForPhoto's
separate Exposure profile/canonical pose. It refuses an edit/camera movement
not yet displayed, the old toy course, and debug glyph geometry. Optional
builder callbacks identify exact room planes, table slab/legs, gobo IDs,
backdrop layers/sweeps and fixture roles without changing any face. Compound
body IDs remain intact; these labels do not certify native contact admission.
Static faces/normals/material IDs, subject full geometry and original Phong
bubble partition, physical frame and actual room bounds are retained. Hidden
room means room=null, not an invented containing wall. Non-off visual preview
motion does not invent a velocity seed. The separate Exposure wrapper is
unchanged. Lens worker changes were preserved; capture stores the lens FOV
actually used by the frame projection.

marble_studio_phong.js reuses the actual mesh/transparentMesh/dispose functions,
existing Phong material classifier and bubble filter, and existing world-space
transparent sort. It allocates hidden buffers, restores GL bindings, awaits a
GPU fence, reads back all four vertex attributes and initial glass indices,
then issues a live ready getter plus revision/step/release. It does not draw or
replace the active buffers. A released/context-lost receipt cannot transfer.
Material rows are still the current Studio textures, so the final exact scene
check is mandatory. Static furniture glass joins the subject glass sort; the
ray geometry remains complete, including bubbles.

Measured real Chrome WebGL2/ANGLE NVIDIA RTX3060: 672 source triangles, 504
opaque-interior plus 168 transparent, 13 actual buffers; 29.6 ms for successful
staging plus cancellation test. All attribute/index readbacks matched source
float32/int32 bytes, framebuffer and previous VAO/array binding stayed unchanged,
and cancel waited for the fence then released all three allocated meshes.
This proves invisible upload/staging, not a rendered dynamic frame or native
contact correctness. Repro and result: build/dot-studio-hidden-20261109/.

Final focused gate: 12 Node scripts passed (transition, exact scene, Phong,
table, gobos, lighting, chase, drive, courses, room, dynamic geometry, native
frame); source JS syntax and server/web_export py_compile passed. Native bake
and source certificate regression tests passed 17 in 0.83 seconds. The exact
scene fixture contained 1,384 triangles from actual shared builders.

During test development, VM cross-realm prototypes caused two strict-equality
assertions to fail despite equal numeric contents; structuredClone now compares
the exact values, including signed zero. A VM fixture lacked structuredClone
and was corrected. A new export initially fell inside an older VM extraction
range; export moved to the module tail. The chase source assertion was updated
to preserve its orbit-pose check while allowing the concurrent lens wrapper.
No production geometry/admission rule was relaxed for these test issues.

Remaining: actual native scene-product delivery/admission, transport to the
Spectral dynamic reader, both-consumer atomic reveal and subsequent motion
updates. Ray worker reports only a source-level sweep-boundary staging API;
its runtime end-to-end test/HTTP route are not completed here. Whole-scene
compound/warped contacts and unsupported optical/media paths remain explicit
blockers. No toy fallback, invisible proxy collider or fabricated ray receipt.

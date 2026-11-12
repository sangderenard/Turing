2026-11-11 PHONG CANVAS CAPTURE AND BOUNDED INTERNAL HOST (Windows, source-only):

Added marble_phong_capture.js and Studio controls. Copies completed original
Phong frames after secondary/DOF. PNG + MediaRecorder WebM/MP4, fixed dimensions,
24/30/60 fps ceiling, 2-minute/128 MiB bounds, repeated/canceled/error cleanup.
No ray job, alternate renderer, audio or screen-capture permission is involved.
Source export closure contains the new module. Concurrent field edits preserved.

Actual Chrome PNG downloaded/decoded at 1062 x 799; framed marble inspected.
Two VP9 videos decoded, including 47 changing rotation frames through 40.582 s.
Instrumented fixture runs about 1 fps; no sustained 30 fps claim. Downloads
are in build/dot-phong-capture-20261111, copied from the browser's Downloads.
Node lifecycle/failure checks passed. Focused combined Python: 22 pass, 2 skips.
Studio boot, loading, DOF and 8 existing download checks pass.
Actual field+DOF coexistence: primary and passes 0/1/2/3, gain 1, GL error 0,
then 1062 x 855 PNG completed. No claim of visible field contribution.

studio_phong.py reuses unchanged standalone_studio full-scene captured renderer,
actual captured GLSL and resource contract, in bounded disposable software WebGL.
StudioAgentProducts optionally admits render-captured camera-only PNG requests.
Existing scene/ray operation semantics and ray commit authority are unchanged.
Actual host tests: 4 pass with GL opt-in, 6 existing agent tests pass, Nodus
adapter 9 pass/1 optional skip. Reference PNG is byte-identical to prior runner.

NOT COMPLETE: arbitrary settings-to-Phong; active field/DOF host replay; server
video; public endpoint/startup configuration. Captured host reference image
shows an exterior wall, not current live Studio parity. See docs/STUDIO_PHONG_HOST.md.
No publication, service restart, production access or full-suite claim.

Hazards reader tools/hazards_log.py refused an existing malformed entry name
20261109T_material-library-paused-checkpoint.md before tests; newest entries and
legacy opening were read directly instead. Existing log was not changed.

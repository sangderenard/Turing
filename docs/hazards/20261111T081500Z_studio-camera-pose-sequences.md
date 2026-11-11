2026-11-11 EXPLICIT STUDIO CAMERA POSE SEQUENCES (MEASURED, Windows, base ecb0265c):

The existing MRP1/API/Studio agent boundary accepts 1..32 camera overrides.
Every entry resolves through the still camera's bounded controls, source lens
selection and physical-back crop. The normalized config contains full cameras.
Orbit, tile/oversampling, live and cooperative modes are mutually exclusive.
Spectral's existing worker selects each pose for native housing plus sensor
arguments and keeps per-frame camera/image receipts and the existing AVI path.
No new renderer, scheduler, GPU run, service restart or lifecycle change.

Turing: 65 passed in 20.73 s across camera pose admission, marble photo, tiles and
Studio agent tests. Node camera_pose_sequence_check and studio_agent_product_check
passed. The isolated checkout needs SPEAKTOME_REPO=C:/dev/Powershell/speaktome
and STUDIO_TEST_SPECTRAL_ROOT=C:/dev/spectral-camera-poses-20261111 for those tests;
initial collection without the former stopped before tests, then passed with it.

Spectral: 34 targeted CPU tests passed in 44.53 s. They cover the actual worker serial frame loop with a
renderer double, exact per-frame optics/geometry calls, snapshot retention,
existing tile cleanup and encoding-error receipts. No GPU correctness claim.
The broader existing test_studio_camera_geometry.py has four failures reproduced
unchanged on canonical 2743d99 with the same Turing virtualenv: missing yaml,
old native_pbr_ssbo fixture, and two old glGetIntegerv fixture omissions.
They are not changed by this task. No full-suite claim.

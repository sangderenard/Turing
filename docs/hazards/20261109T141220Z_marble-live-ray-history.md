2026-11-09 LIVE RAY CAMERA WITH FINITE HISTORY (SOURCE ONLY, NOT PUBLISHED)

The optional Live ray-traced camera toggle replaces the main Phong display
with PNG frames from the real Spectral fixed-three-line thin-lens sensor.
It is disabled by default. It requires a saved static design, Visual preview,
and Motion Off. It follows orbit/zoom; animated/particle geometry is not
captured. Scene-light changes restart the owned live scene. Object geometry or
material edits require saving again. A running photograph or another tab's
live session is rejected with 409, not interrupted.

The sensor stores each pixel/channel contribution in an actual float32 GPU
circular array, with float64 totals and an adjustable 1..64 contribution
window. Expired values are subtracted before replacement. Full windows are
also summed from their bounded state after each replacement so extreme bright
paths cannot leave cancellation ghosts after expiration. This is not decay,
exponential smoothing, or blending of complete frames. Camera/lens/history
changes invalidate the ring, and publishing waits for every pixel's first
new-camera sample. Memory is checked before allocation and capped at 128 MiB
for ring plus totals; selected long-edge sizes are 128, 256, 512 pixels.

Forward illumination is finite too: at each complete sensor sweep the live
worker clears the voxel field and emits exactly 8192 fresh forward rays.
Every strip in that sweep samples the same field. No forward energy predates
its sweep; only the resulting sensor contributions persist for N samples.
The random sequence advances through field resets. Partial final rows are
covered. The ordinary still-photo accumulation path remains unchanged.

The browser has one serialized pump, revision checks around asynchronous
boundaries, and a fresh session identity on every enable. Shutdown is awaited
before a subsequent start. An old delayed cancellation cannot kill a new
session. Control updates and image publication are atomic. Windows control
file replacement retries bounded reader/delete-sharing collisions. The client
renews a 30-second lease; hiding/closing the tab stops live, and a lost client
expires. Only the latest three preview PNGs are retained. Output is limited
to at most 4 Hz and depends on tracing/PNG throughput; there is no promise of
video-rate convergence or noise-free frames.

Validation on the local Windows source:
- 31 Python tests passed across live configuration/API, photograph, completed
 retouch, and film response. One pre-existing cffi deprecation warning.
- 7 production JS lifecycle cases passed, including a delayed cancel/off-on
 race, stale pending start, busy-photo conflict, hidden tab, and invalid design.
- Spectral CPU tests: 2 passed; one opt-in GPU test skipped by default.
- The same GPU invariant test was executed directly outside pytest with the
 real OpenGL context: exact warmup/wrap/expiry, 1e20 bright-ray expiration,
 zeros, pixel/channel independence, reset, and full sensor shader link passed.
- A real 122352-triangle saved scene ran 128 live epochs with no production API.
 Eight snapshots were produced. Camera eye changed by .001 m and history 4 to 8
 during the run; final revision 1, 8/8 samples, 4301 visible pixels, finite and
 nonnegative output. Warmed red-channel mean ranged 603..648 rather than growing
 with elapsed history. Recorded preparation 3.437 s; last snapshot elapsed 2.25 s.
 Receipt: build/dot-live-ray-20261109/gpu-smoke-result.json. The temporary
 rendered images were not retained after the execution workspace cleaned up.
- A real Chrome source page booted and its isolated test-save action worked.
 End-to-end live-toggle browser QA is pending in the coordinator's stable
 local fixture server; this worker's helper unexpectedly exited between calls.
- Scoped whitespace checks and JS/Python syntax checks passed. No full suite,
 production export, production restart, commit, or publication ran here.

The first scene smoke exposed a PyOpenGL binding issue: glGetInteger64v on the
SSBO size query returned a denormal float instead of the reported integer.
The allocator now uses glGetIntegerv; the actual bound was 2147483647 bytes.
The successful scene smoke exercised allocation after that correction.

Primary source: examples/meowdoku/marble_live.py and web/marble_live.js, narrow
integration in marble_photo.py/server.py/web_export.py and the Studio page.
Spectral sibling source: camera_designer/sensor_history.py,
camera_designer/marble_live_worker.py, narrow SensorAccumulator integration in
demo_pluck_gl.py, and optional run integration in marble_preview_worker.py.
Other ongoing lighting/material/color changes have independent owners and
must be validated together before publication. Do not overwrite their edits.

2026-11-09 LIVE HISTORY, CHECKPOINT, AND DISPLAY CONTROLS (SOURCE ONLY)

This supersedes the 128 MiB / 64-sample limit in the earlier live-ray receipt.
The ring now has a 2 GiB total storage cap, including all linear accumulation
state. The UI calculates bytes from the actual rectangular sensor dimensions.
It displays both MiB and exact bytes, and turns the number bright red (#ff0000,
weight 900) strictly above 512 MiB. The slider and exact numeric field adapt to
resolution, checkpoint mode, and the worker's reported SSBO block limit.
At square 128px with checkpoints, the cap is 10,914 rays per pixel; at square
512px it is 674. Resizing or changing checkpoint mode clamps an overlarge value.
Default ring length remains 16; the new UI enables checkpointing by default.
Old configurations without the checkpoint field retain finite-window behavior.

The GPU ring stores float32 sensor contributions. Without checkpointing, the
float64 ring total is the only extra state (8 bytes per pixel/channel). With
checkpointing there are four float64 values per pixel/channel: ring total,
evicted sum, Neumaier compensation, and evicted count (32 bytes). On eviction,
the old contribution is added exactly once to the checkpoint before its slot
is overwritten. Display sensor energy is (checkpoint sum + ring sum) divided
by (checkpoint count + valid ring count). Profile conversion and tone mapping
occur afterwards, once. This is an all-sample mean, not an exponential decay,
nor an accumulation of nonlinear PNGs. Storage stays fixed as counts grow.

Checkpoint sums and counts have finite precision. Counts are exact integers
through the float64 range; the worker refuses the next sample at 2^53-1
completed sweeps rather than silently overflowing a count. A sum of finite
float32 inputs cannot overflow float64 before that count limit. The output
sensor texture remains float32. Checkpoints are session-local GPU state, not
saved images. Camera/scene/history-capacity/checkpoint-mode changes reset both
ring and checkpoint; tint and normalizer updates preserve them. Closing or
pausing the live session still stops its worker under the existing lease rules.

Allocation checks the actual GL_MAX_SHADER_STORAGE_BLOCK_SIZE for both buffers.
When GL_NVX_gpu_memory_info is advertised, the driver free-memory estimate is
checked with 128 MiB headroom, crediting only the old owned ring that will be
released. Driver estimates are not guarantees. The old buffers are finished
and deleted before replacement to avoid a temporary doubled multi-GiB peak.
Every new buffer's reported GL_BUFFER_SIZE is verified. Partial allocations
are deleted on failure, the live worker reports an error, and the browser
restores Phong. It never silently uses a shorter history or cancels a photo.
Other scene/GPU storage is additional to the displayed history budget.

New live Tint and Normalizer controls pass validated display-only parameters
to the existing render_linear_srgb path. Tint is the same green/magenta gain
as saved retouch (-3..3). Normalization is auto99 or a manual relative linear
white reference (1e-6..1e12). Switching to manual starts from the last reported
white when available. Display-only revisions do not blank the current image,
clear the ring, alter checkpoint state, or upload a new scene.

Completed checks:
- 31 targeted Turing Python tests passed (one existing cffi warning).
- 44 Spectral history/color-profile CPU tests passed, with the real GPU test
  intentionally skipped while the user's 1200-second photograph is exposing.
- 11 production JS lifecycle cases plus resolution/aspect/memory boundary
  matrix passed. Includes device caps, warning threshold, resize clamp, tint
  updates without scene restart/blanking, and invalid macro focus cleanup.
- Exact shared-pipeline PNG parity for live tint/manual white passed with
  mocked GL readback. Linear input samples remained unchanged.
- Allocation tests cover preflight device rejection, near-cap resize release
  order, partial-allocation cleanup, and exact-count overflow guard.
- The full production sensor shader, including checkpoint code, passed offline
  glslangValidator compilation. No OpenGL context was created for that check.
- Actual Chrome UI on the isolated 8883 fixture verified bright-red memory,
  rectangular sizing, resize clamping, tint/manual-white controls, one-session
  start/update/stop, and restored Phong. Its returned PNG was a generated color
  grid, not a real GPU exposure. Receipt:
  build/dot-live-ray-20261109/expanded-ui-fixture-verification.json.
- JS/Python syntax and scoped whitespace checks passed.

Remaining gate: run the small real OpenGL ring/checkpoint numeric test and
actual allocation/resize smoke only after the active user exposure ends and
no other GPU worker is active. The test covers exact eviction means, counts,
1e20 bright rays, reset, and capacity changes. No 2 GiB stress allocation has
been run or is required during user work. No production restart, export,
publication, or GPU interruption occurred for this increment.

Owned files: Turing marble_live.py, web/marble_live.js, the live-control HTML
region, and live tests. Spectral sensor_history.py, SensorAccumulator history
allocation/dispatch hooks, marble_live_worker.py, optional snapshot display
argument in marble_preview_worker.py, and history tests. Auto-bounce, bubbles,
firefly suppression and other lighting edits remain independently owned.

2026-11-09 STUDIO STATE HANDOFF FOUNDATION (MEASURED, canonical Windows,
base f998b80f, source only):

ComputeRuntime now accepts opt-in full float32 start columns whitelisted by
manifest.law.state, including the padded native span, linear/angular velocity,
age and current radius. The existing host angular recording can start from
explicit unit xyzw quaternions instead of identity. Legacy age-only and
fixed-point start payloads retain their existing decode path. Unknown/missing
columns, incompatible lengths, mixed formats, nonfinite state and invalid
orientations refuse before reset. No new integrator or force law is added.

pauseAtDisplayedState uses the incumbent Recording.plan/keyframe/input log to
restore the visible frame, not the newer queued native state. It drains queued
batches, reads every native state column including padding, checks displayed
positions exactly against the restored columns, and retains the recorded pose,
uniforms, step and case-frame motion/displacement. It returns no state on a
divergence, changed run, missing pose or invalid readback. It pauses explicitly;
Continue resumes the same native step. Replay/Bounce leave the pause, including
after a failed capture. Playback can cancel a pending readback; an equivalent
spec refresh is not mistaken for a new run. Repeated capture requests coalesce.

Replay work defaults to a bounded allowance of 360 native steps. A caller may
provide another nonnegative integer allowance. An older frame without a nearby
key refuses before replay instead of submitting an unbounded recovery job.
This does not make recording automatic; full Studio integration must choose
the existing keyframe recording settings appropriate to its product.

Owned runtime changes: web/compute.js, new web/compute_state.js, and the single
compute_state.js asset registration beside compute.js in web_export.py. The
export dependency must accompany compute.js. The export closure test executes
the real static asset-copy loops and imports their result with the existing
server-published WASM shell, without running export, server or atlas work.

Focused tests: tests/meowdoku_compute_state_check.mjs (18 checks),
tests/test_meowdoku_compute_state_js.py. Actual runtime reset/first-snapshot
orientation, replay, mode changes and cancellation are exercised with an
in-memory product double. Actual GPU/WASM readback methods are exercised with
mock device memory. These are not GPU physics execution. Existing preview,
angular integration, visual motion and replay-layer suites also pass. The
standalone old preview test initially could not import the unexported WASM
shell from source; the checked isolated fixture supplies the same real shell
that server.install_shell publishes, not a substitute implementation.

Native triangle groundwork is documented separately in
20261109T161143Z_studio-native-triangle-foundation.md. Neither foundation is
wired into the Studio physics selector. The current deployed particle mode
still uses the separate MarbleMadness course. Full shared-scene collision,
closed-solid/manifold ownership, compiled capacity/upload, scale/time mapping,
stage-sized routes and Pluck player integration remain unfinished. The shared
marble_studio_scale.js helper owns scene unitM/subjectRadius/subjectCenter;
no competing world scale or optical-to-friction convention was introduced.

Final combined gate: 12 passed in 2.02 seconds, including the retained delayed-
column readback interruption regression and the native contact suite. Independent
review found and verified fixes for export registration, Replay/Bounce recovery
and equivalent spec refresh cancellation; no residual issue in the bounded
review. Syntax and scoped tracked-file whitespace checks passed.

No GPU job, active-worker change, server restart, publication, commit or push.
Keep these files separate from an already-frozen visual release until the
integration owner deliberately includes and verifies them together.

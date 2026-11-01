2026-11-01 Control surface and Fortran deployment inspection

Read current compiler instructions and newest native barrier receipt. User
requested the total backend control surface and deployment/type inspection.
No compiler or runtime implementation changed; expanded the existing backend
threading operation reference rather than creating a second registry.

Observed sources: ssa_registry.Handler, control_source.ControlProgram blocks,
thread_operations, deployment_frame, deployment_classification,
deployment_lowering, deployment_ssa_binding, deployment_outlining,
ssa_fortran_backend and profiled_c_shell.

Recorded complete control families, existing deployment join modes, execution
classes and strategies, structural deploy token versus numerical dtypes, and
Fortran's actual serial profile/marker lowering. Kept missing Event/atomics/
deque mappings and pause/cancellation distinct from registered/verified work.
Found no native authored-thread/exception/mutex mappings in Fortran emitter.
Its legacy validation call maps to error stop; that is not caught-exception
support. The future pool integration profile note is not implemented support.
Binding's sibling SSA use check is not a complete shared-memory alias/effect
proof; outlining contains additional shared-store and sequence checks.

Validation on current checkout:
pytest -q tests/test_deployment_lowering.py tests/test_deployment_classification.py
 tests/test_deployment_ssa_binding.py tests/test_fortran_control_target.py
35passed,1skipped,1warning in0.84s. Fortran compiler absent, native compile test
skipped. Warning is optional nodus tensor core unavailable. No new standalone
Fortran execution or parallel support claimed. Exact next boundary is native
Fortran ABI/launch integration and per-family mapping/refusal with error,
capture, reentrancy, progress and lifetime contracts preserved.

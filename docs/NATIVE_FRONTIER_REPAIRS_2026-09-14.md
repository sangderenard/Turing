# Native compiler frontier repairs — 2026-09-14

## Scope and evidence

Both frontiers remain compiler pressure tests. No Python dispatch was added to
the native adaptive window or native recurrent training cycle. New compiler
identities continue to come from `GLOBAL_MONOTONIC_IDS.mint()`.

### Recurrent perforated training

The sum of per-transition scalar losses now works without the previous
sum-tensors-then-reduce workaround. Two independent compiler defects caused it:

1. A specialized `bw_add` unit adjoint reached an `unbroadcast` helper whose
   only surviving Phi input was the literal unit seed. Structural recovery
   failed to publish that identity. It now aliases the exact operand without
   inventing an SSA identity or treating a multi-edge Phi as an identity.
2. Explicit broadcasting allocated storage before applying the requested
   destination shape. A `(1, 1)` provisional result therefore allocated eight
   bytes for a `(2, 3)` double result. Shape settlement now precedes tensor
   descriptor registration.

Focused native coverage poisons gradient outputs and checks every element for
both unit and explicit 2.5 seeds. A separate reference test calculates gradients
by central finite differences through a three-transition recurrence, including
hidden history and predicted-state feedback, then independently applies clipped
Adam across two experiences and two updates. It compares every parameter and
both moment banks, not just loss reduction.

Evidence before the subsequent shape/optional changes:

- `build/perforated_frontier_regressions.log`: six native network/recurrent tests.
- `build/native_adjoint_final.log`: three native adjoint/reference tests passed.
- `build/native_frontier_metadata_regressions.log`: 31 metadata/control tests passed.

The real-engine pressure run uses 64 transitions, two training experiences, one
validation experience, hidden width four, and two epochs:

```powershell
python -u -m src.common.tensors.abstract_nn.demo_recurrent_engine_learning --output-dir build/recurrent_engine_frontier_20260914 --steps 64 --hidden 4 --training-runs 2 --validation-runs 1 --epochs 2
```

The optimizer is the existing LLVM native loop around the compiled trajectory
forward/loss/generated VJP. This is not a claim that the separate authored
functional Adam ProcessGraph has been linked. The real-engine run must produce
its own concrete result before native-engine success is claimed.

### Validator simulation

The saved pre-frame checkpoint now clears the former three physical input
conflicts. An explicit ABI scalar dtype must survive control lowering: using a
floating storage slot as a loop count cannot reinterpret its bits as an integer.

`build/validator_frontier_20260914_v6/repository-ssa.pkl` was produced, with one
optional result conflict and 38 other real audit findings. The old ID magnitude
heuristic also falsely reported 391 functions because its threshold equalled
the central issuer's initial ID. The heuristic now checks trillion-scale values;
it is a diagnostic, not an allocator or proof of identity ownership.

The v7 replay completed eight frame rounds and three result-type rounds. Its
audit (`build/validator_frontier_20260914_v7_audit.log`) has 13 formal-parity groups,
three definition-dominance findings, and the same optional result conflict.
This is repository SSA, not an executable validator DLL.

Additional general repairs made while replaying:

- Late unary/expression recovery updates producer positions after each insertion.
- Scalar expression recovery refuses authored aggregate/list replication even
  when its provisional storage handle has scalar shape.
- Exact source call-edge ABI propagation includes derived tensor descriptors.
- Callsite specialization invalidates dependent cached shapes; shape constants
  wait for their actual input descriptors.
- Optional record constructors publish explicit payload and presence fields.
  Presence column spelling now agrees between record and sequence-row layouts.
- Constructor-only parameters retain their authored ABI names.
- `Metrics.dt_limit` and `advanced_dt` are declared optional, matching their
  existing Python annotations/defaults. This changes representation, not physics
  or adaptive-controller tolerances.

`build/native_optional_record_constructor_v2.log` records three native DLL tests
distinguishing absent, present-zero, and present-nonzero values. The linked record
row group passes four tests in
`build/optional_record_constructor_regressions_v2.log`. Late source recovery and
the central-ID heuristic group passes ten tests in
`build/late_source_recovery_regressions.log`. Three shape regressions pass in
`build/call_edge_shape_regressions_v2.log`.

A fresh source build is required to incorporate the optional ABI declarations;
the old pickle stores the earlier declarations:

```powershell
python -u tools/build_vehicle_validator_simulation.py --output build/validator_frontier_20260914_v8 --lanes 8
```

The v8 log is `build/validator_frontier_20260914_v8.log`. Recheck its result before
starting another build. No full-suite gate was used.

After the shape and optional-constructor changes, the cross-frontier group
(`test_native_scalar_loss_adjoint`, `test_recurrent_native_history_reference`,
`test_ssa_optional_values`, `test_managed_native_output_contract`) passes 17 tests
in 117.35 seconds (`build/native_frontier_cross_regressions.log`). The optional
record/scalar group subsequently passes seven tests in 19.07 seconds
(`build/optional_record_native_final.log`), including a native zero/absence test
that resolves presence from the local record descriptor rather than stale
callee accounting IDs.

## Later control and storage replays

The v8 source build ended after 1903.31 seconds with a `FortranEmissionError`:
zero unmaterialized operations/calls, one undefined operand (`1583` in
`validator_simulation_advance` planned region 44), plus linked structural/type
findings. The full diagnostic is `build/validator_frontier_20260914_v8/failure.json`.
Its resolved graph is retained; a new source extraction is unnecessary.

Subsequent shared repairs and focused evidence:

- Remove uniquely defined, unused Phi joins while retaining their producers and
  effects, and protecting descriptor/publication identities. The control cleanup
  group passed 16 tests (`native_control_cleanup_regressions.log`).
- Settle fresh repeated-region projections through the actual Call/GEP/Load
  object chain. Use `source_output_id` as the **caller** output identity and the
  explicit call mapping; original aggregate indices can have gaps after dead
  outputs are removed. The result-type/native-loop group passed 11 tests
  (`native_projection_identity_regressions.log`). An intermediate positional
  version caused many false shape conflicts in v8 and has been corrected.
- Convert scalar call arguments into the callee's settled storage type, and
  track the actual C local's type rather than copied tensor buffer metadata.
  Fifteen loop/type/Boolean tests passed (`native_loop_storage_regressions_v2.log`).
- A terminal return inside an arm now owns its lexical edge. Its anchor is the
  returned value (or surviving tuple slot), matching the loop planner. When a
  return and its producer call share an anchor, source call placement orders
  the call first. The native early-return mutation test passed
  (`native_lexical_return_anchor_v3.log`). Strengthening it to count every loop
  visit exposed a further scalar-to-array Phi storage problem; follow the later
  `native_scalar_cells_regressions.log` result before claiming that case passes.

The current full resolved-graph replay is:

```powershell
python -u ../speaktome/AGENTS/tools/replay_turing_resolved_frontier.py build/validator_frontier_20260914_v8/resolved-process-graph.pkl --repo . --output build/validator_frontier_20260914_v9
```

It retains `pre-frame-link.pkl` as well as final repository SSA and audit output.
Its log is `build/validator_frontier_20260914_v9.log`. It began before the latest
lexical-return anchor/call-order edits; use its concrete audit to choose the next
checkpoint replay. No full native validator artifact is claimed by these tests.

The v9 replay reached pre-frame lowering but its diagnostic serializer rejected
the deployment's dynamically created class. The helper now uses the already
installed `joblib.externals.cloudpickle` for that checkpoint, as the older
working pre-frame helpers do. The corrected replay is **v10**, with log
`build/validator_frontier_20260914_v10.log`; v9 has no usable pre-frame checkpoint.

The strengthened native loop now verifies all 3/5 iterations, zero iterations,
and an early return after exactly one mutation. Its initial shaped zero needed
real storage: shaped scalar Const payloads are now emitted as filled buffers.
Two standalone native tensor-constant tests pass in 3.41 seconds
(`native_tensor_constant_storage.log`).

An adjacent existing DT test, `test_native_loop_call_argument`, reproduced a
stale timestep argument with all eight relevant modules loaded from HEAD in an
isolated process (`native_loop_call_full_head_baseline.log`). The scheduled
marker correctly held the loop Phi, but late aggregate/frame reconciliation
overwrote it from a source-seed receipt. Frame linking now preserves the marker's
bindings across rounds and records the selected physical input ID in the frame
receipt, leaving source provenance in `argument_bindings`. The native loop and
input receipt group passes six tests in 19.83 seconds
(`native_current_frame_receipts.log`). The v10 process predates this last repair;
replay its pre-frame checkpoint with current code when available.

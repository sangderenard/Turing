# Continuation handoff — wait for concrete compiler output

Date: 2026-09-14 local workspace time.

There are two active native-compilation frontiers:

1. Validator DT simulation: keep the Python viewer, compile the coupled physics
   and adaptive DT controller underneath it into a native DLL that owns the whole
   viewer-tick cycle without repeated Python dispatch for inner substeps.
2. Perforated engine training: get the engine/perforated network training path
   working as native-compiled history-respecting forward graph -> backward graph
   -> update execution.

Do not spend turns saying that a long compiler run is still running. These builds
can take 10 minutes, 20 minutes, 45 minutes, or over an hour. Wait patiently for
a specific compiler result before speaking again. A useful update has one of
these things:

- a completed native artifact and the exact command to run it;
- a concrete exception, failing gate, or mismatch with file paths and the first
  failing symbol/value/region;
- a focused test result after a code change;
- an explicit user question that cannot be answered from the repository state.

The most recent compiler defect was duplicate formal IDs in the validator native
simulation replay. It came from compiler-created `SSAValue` objects sharing a
numeric ID, not from valid reuse of one actual object across multiple parameters.
A central issuer now exists in `src/compiler/monotonic_ids.py`. New
compiler-created representational IDs must call `GLOBAL_MONOTONIC_IDS.mint()`.
There is no reservation protocol and no local counter should scan existing graph
or SSA IDs to allocate new identity.

The saved validator checkpoint
`build/validator_simulation_native_v3_20260914/pre-frame-link.pkl` has been
replayed to the old failure seam after the allocator repair. It reported zero
functions with duplicate formal IDs before `_complete_propagated_frame_tails`.
The focused receipt/dispatch tests passed:

```powershell
python -m pytest tests/test_native_call_input_receipts.py tests/test_scheduled_process_graph_dispatches.py -q
# 30 passed, 1 warning
```

This is not a finished native validator claim. The correct next action is to run
or continue the specific native build/checkpoint needed for one of the two
frontiers and wait for its actual result. If a build is quiet for a long time,
leave it alone unless there is external evidence that the process is dead.

Relevant reports:

- `docs/FRONTIER_REPORT_2026-09-14_NATIVE_COMPILER.md`
- `docs/CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md`
- `docs/CONTINUATION_2026-09-14_NATIVE_DT.md`
- `TEST_BASELINE_AND_HAZARDS.md`

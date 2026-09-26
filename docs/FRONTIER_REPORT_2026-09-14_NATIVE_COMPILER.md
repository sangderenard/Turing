# Native compiler frontier report — validator DT and perforated engine training

Date: 2026-09-14 local workspace time.

This repository is at a compiler-frontier crossroads with two active attempts.
Both attempts are trying to find whether the compiler can move work that is now
near the practical limit of the local Python/runtime machine into native code
without losing the semantics that made the Python version useful.

## Frontier 1: validator DT system as one native simulation cycle

The validator goal is to keep the Python viewer, input plumbing, and presentation
shell, but compile the coupled simulation below it. The native DLL should own the
whole validator adaptive-time cycle for a viewer tick: controller state,
sub-ticks, rollback/accept decisions, state feedback, and result publication.
Python should not repeatedly dispatch tiny native calls for each adaptive inner
step. Python remains only where the visualizer or still-unlowered application
code requires it.

The current validator path is documented in
`docs/CONTINUATION_2026-09-14_VALIDATOR_NATIVE.md`. The latest fresh v3 build
had already passed the earlier scheduling failures: instantiation, call topology,
complete control planning, and individual region lowering including
`run_superstep`, `pi_update`, `copy_shallow`, and `restore`. It saved the real
inputs to `_class_surface_ssa_program` as
`build/validator_simulation_native_v3_20260914/pre-frame-link.pkl`.

The next concrete failure was not a physical/numerical result. It was a compiler
identity failure at native call input binding:

```text
ValueError: invalid native input binding receipt for validator_simulation__validator_simulation_window -> validator_simulation__run_superstep__specialized_8e51a922f2e2: receipt=1845, operands=1845, unique_formals=1844
```

Read-only replay of the saved pre-frame-link checkpoint found the exact cause:
the callee had two distinct formal `SSAValue` objects carrying the same numeric
ID, `357`. One was the sequence `status_address_id`; the other was a child table
pool member for `Metrics.unresolved_report`. Because call binding is keyed by
formal ID, the two different formal objects collapsed to one binding slot. This
was not a valid case of one actual feeding multiple parameters. It was two
separate compiler-created objects sharing one representational ID.

## ID allocator repair made in this snapshot

A central compiler ID issuer now exists at `src/compiler/monotonic_ids.py`:
`GLOBAL_MONOTONIC_IDS`. The rule for compiler-created representational IDs is now
simple: call `GLOBAL_MONOTONIC_IDS.mint()`. Do not reserve ranges. Do not scan a
function, module, graph, or source table to invent the next ID. Do not seed a
local counter from source IDs. Source-side numbers may be correlation metadata or
references to already-defined values; they are not authority to allocate new SSA
identity.

The current change routes fresh compiler-created SSA/value IDs through the global
issuer in the main validator/native compile path and adjacent SSA lowering
passes:

- `src/compiler/fortran_c_shell.py`
- `src/compiler/precompile_to_ssa.py`
- `src/compiler/ir_sequence_tables.py`
- `src/compiler/hierarchical_plan.py`
- `src/compiler/deployment_ssa_binding.py`
- `src/compiler/ir_indexing.py`
- `src/compiler/ssa_call_input_adapters.py`
- `src/compiler/tensor_ssa_lowering.py`
- `src/compiler/ssa_optional_values.py`
- `src/compiler/ssa_primitive_lowering.py`
- `src/compiler/ir_identities.py`
- `src/compiler/ssa_record_return_state.py`

The misleading phrase "authored field" was removed from the touched
`fortran_c_shell.py` path. The intended meaning was only that multiple source
sites can prove the same field relationship. Those sites do not issue SSA IDs.

Verification performed after the allocator repair:

```powershell
python -m py_compile src/compiler/monotonic_ids.py src/compiler/ir_sequence_tables.py src/compiler/fortran_c_shell.py src/compiler/hierarchical_plan.py src/compiler/deployment_ssa_binding.py src/compiler/precompile_to_ssa.py src/compiler/ir_indexing.py src/compiler/ssa_call_input_adapters.py src/compiler/tensor_ssa_lowering.py src/compiler/ssa_optional_values.py src/compiler/ssa_primitive_lowering.py src/compiler/ir_identities.py src/compiler/ssa_record_return_state.py
python -m pytest tests/test_native_call_input_receipts.py tests/test_scheduled_process_graph_dispatches.py -q
```

Result: py_compile passed; focused pytest passed `30 passed, 1 warning`.
The saved validator checkpoint was replayed to the exact previous seam and
reported:

```text
{'duplicate_formal_function_count_before_complete': 0, 'examples': []}
```

This proves the previous duplicate-formal failure is removed at the saved seam.
It does not prove the full native validator DLL builds or runs. The next run must
wait for the compiler to produce its next concrete result.

## Frontier 2: perforated engine native history-respecting training

The perforated engine training frontier is separate from the validator DT target.
Its goal is native-compiled training over the engine/perforated network while
respecting history from forward graph to backward graph to update. The dirty tree
contains this line of work in the recurrent learning/demo files, LLVM/native
training machinery, tests, and documentation. Existing baseline notes record
that a headless LDT path compiled an 18-batch cyclic bank and executed forward,
loss, generated-VJP, and Adam motions in one native call, with training loss
moving from `1.11127` to `0.02979`. The later adaptive sample refresh completed
2 native epochs / 36 motions and refreshed real simulator transitions without
recompilation.

That is strong evidence that native work can reduce Python overhead for this
training frontier, but it is not yet a finished claim that the complete
history-respecting forward-to-backward-to-update training system is solved. The
next useful step is to let the current compiler/native attempt run until it
produces either a native artifact, a numerical mismatch, or a precise compiler
failure.

## Operational condition

The full test suite is not an appropriate gate here. `TEST_BASELINE_AND_HAZARDS.md`
records that full-suite behavior is hazardous and slow. Use focused tests for the
modules touched, plus saved checkpoint replays and the specific long native build
commands. Long compiles here normally take 10 minutes to more than an hour. Do
not interrupt them just because they are quiet.

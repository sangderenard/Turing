# Meowdoku compiler and direct Wasm handoff — 2026-11-04

## Scope and source

- Validation checkout: `C:\Users\alber\.codex\worktrees\compressor-validation\turing`. Canonical was `C:\dev\Powershell\turing` at `785b747a` when this lane began. Do not copy the untracked `examples/meowdoku/` fixture into a commit.
- The unchanged authored law is lowered through `src.compiler.fortran_c_shell.lower_ast_source_to_ssa` with its supplied nine float64 rank-one span arguments and the sanctioned C tensor reference. Saved **complete SSA** modules for batch 25 and 64 are `%TEMP%\meowdoku_validation_25_module.pkl` and `%TEMP%\meowdoku_validation_64_module.pkl`. Both have 35 functions; entry is `meowdoku_step_batched__meowdoku_step`. Boundary's float alias Call adapter change made SSA complete for both sizes. No further non-Wasm compiler refusal has been observed on these two source lowers.
- A diagnostic slice preserves the authored initial `cells`, `r1`, `r2`, and pairwise subtraction/absolute value, then returns a scalar sum. Its complete 10-function SSA is `build/meowdoku_pairwise25_ssa.pkl` (tuple of module, outputs, exports). The `.item()` variant stopped earlier in source-link output accounting and is not a direct Wasm acceptance result.

## Current direct Wasm state

- `src/compiler/ssa_wasm_backend.py` is modified but **WIP**, not ready to commit as an accepted Meowdoku backend. `python -m py_compile src/compiler/ssa_wasm_backend.py` passes. The saved pairwise module emits `complete=False`, 12 shortfalls: Call 6, unavailable operand 2, GetElementPtr 2, Load 2. Saved full batch 25 and 64 each emit `complete=False`; the latest batch-25 breakdown totals 124: Call 48, GetElementPtr 21, Load 21, Ret 5, unavailable operand 29. No Wasm/game output parity has been measured or claimed.
- Root host ABI in the batch-25 artifact is now exactly 13 buffer parameters: eight authored tensor inputs, then five output buffers (IDs 106, 108, 113, 119, 121). Four root `compiler_frame_storage` activation cells (IDs 2305843010213693952–3955) are allocated internally using `is_compiler_owned_storage` and `module_storage_requirements`; they are no longer exposed as game inputs. This is only ABI classification, not a completed executable.
- The pairwise `broadcast_double` output destinations `%2305843010213693966` and `%2305843010213693971` have same-frame `SSATensorDescriptor` temporary/static float64 shape `(25,25)`, data ID equal to the output ID, byte size 5000; `module_storage_requirements` independently gives count 625, dtype float64, shape `(25,25)`. Their minted `SSA_VALUE` physical fields are absent. The new `ensure_call_output_arena` allocates only when descriptor and storage obligation agree. Boundary inserted its one-line hook before Call actual/formal mapping. That cleared the previously first unallocated output refusal, exposing deeper kernel instructions. Emission still needs an auditable arena-placement receipt tied to the exact scoped destination/descriptor/storage requirement before this hunk is accepted as final.

## Changed files and ownership

- Boundary owns the `ssa_wasm_backend.py` `inline_call` / generic multi-block callee dispatcher, Call actual/formal pointer closure, and `tests/test_wasm_multiblock_callee.py`. Its CFG section is import-stable and tested. It preserves Phi edge copy, branches, and callee Ret identity without a kernel-specific substitute.
- Evidence agent owns the same file's `emit_straight_instruction` primitives, local typed aggregate/glyph/call-output arenas, hidden root frame allocation, root Ret span-copy draft, and `WasmCoreExecution` reserved-prefix/output-buffer draft. It added exact imported LLVM literal decoding through `ir_literals`, typed `Load` from proved GEP element layout, signed `SExt`, `sdiv`/`srem`, LLVM scalar conversions, select/bit operations, and exact `llvm.fabs.f64`/`llvm.floor.f64` spelling. These are **not** a blanket approval of the combined 696-line diff; the WIP primitive/arena/Ret hunks have only partial saved-SSA emission and existing regression gates.
- Other modified compiler files in this shared validation checkout belong to root or boundary and are outside this Wasm handoff. Do not overwrite whole files to transplant these overlapping hunks. No game law, dt system, app, or fused backend was changed by this lane.

## Measured gates and reproduction

- On 2026-11-04, `python -u -m pytest -q tests/test_wasm_core_int64.py tests/test_mobile_reference_app.py --tb=short` passed **8/8** in 13.02 seconds after the direct arena/Ret changes; the latest imported literal/bit primitive edits were subsequently `py_compile` checked and emitted saved pairwise/full modules, but that 8-test gate has not been rerun after the final bit-op edit. Boundary separately reported its three CFG tests plus the same existing tests **12/12**, and mapping audit **28/0**, before the final primitive edits.
- From the validation checkout, saved-SSA emission only: `$env:PYTHONPATH=(Get-Location).Path; python -u $env:TEMP\meowdoku_pairwise_emit_small.py` and `$env:PYTHONPATH=(Get-Location).Path; python -u $env:TEMP\meowdoku_wasm_shortfall_classes.py`. Both print inline and take about four seconds each. The complete source pickles above can be loaded directly with `pickle.load`; the pairwise pickle contains `(module, outputs, exports)`, the full ones contain an `IRModule` alone. No long source lower is needed to reproduce the current direct-emission limits.

## Pending exact gaps

- `pow` has no direct Wasm spelling in the current core. Other selected Call output/aggregate identities disagree, including `binary_value` output slots. The pairwise region2 unary output `%17` has a storage requirement and descriptor but fails the present temporary-arena predicate; inspect its actual storage kind and Call output ownership before changing the predicate.
- Imported pointer loads and GEPs still need exact typed layout for the listed 21/21 full-module shortfalls. Root Ret span copy still reports five missing exact source slots/static output counts. Do not mark `complete=True` by suppressing either class.
- The backend does not yet execute the authored game at batch 25 or 64. Direct Wasm acceptance remains: both emitted core artifacts `complete=True`, zero shortfalls, correct eight-input/five-output host ABI, and measured native Wasm/game parity. None of those completion claims is made here.

This handoff freezes the Wasm lane. Resume only on a new user instruction.

## Frozen transfer artifact

- Backend and CFG-test snapshot: `C:\dev\Powershell\turing\build\meowdoku_wasm_handoff_20261104`. The portable patch is `C:\dev\Powershell\turing\build\meowdoku_wasm_handoff_20261104\wasm_backend_and_cfg_tests.patch`; `git apply --check` passed against canonical main785b747a. It contains only the backend diff and new CFG regression, excluding compressor/SSA adapter edits. It has **not** been applied to canonical.
- Frozen backend SHA256: `88dc294b412a11b831b90517aea75dbc910f3b1ad6fb326ee7c4e2fba99b0fe0`. The full report is copied to canonical repository root for the game lane. Keep the WIP status above when resuming.

## Final SSA recheck after the shared compiler fixes

The unchanged authored law was re-lowered on the final validation core: batch 25 completed 35 functions with zero audit findings in 8.89s; batch 64 did the same in 8.84s. The core/alias canonical install gate passed 4/4 in 33.72s. Remaining measured Meowdoku work is direct Wasm emission.

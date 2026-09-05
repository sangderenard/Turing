# Compiler bootstrap structural frontier handoff — 2026-08-22

## Superseding autonomous route

The manual parent-unit procedure below is retained only as failure evidence.
Do not select or relaunch unit 47. Commit `6d8919a` makes the bootstrap creep a
compiler-owned fixed-point process: it discovers the authored catalogue,
compiles with isolated resource-bounded workers, emits and verifies native
products, exposes only receipt-proven deployments to the next worker pass,
reuses verified source-region seeds, and automatically crawls every persisted
failed-unit ProcessGraph subdivision plan. It records a terminal frontier
instead of asking a person to choose the next unit.

The unattended entry point is:

```powershell
python -m tools.compile_project_catalogue `
  --source src/compiler/precompile_to_ssa.py `
  --output build/compiler-bootstrap-creep `
  --creep-project `
  --jobs 4 `
  --max-total-gb 48 `
  --worker-reserve-gb 8 `
  --max-worker-gb 12 `
  --unit-timeout-seconds 1800
```

`creep-progress.json` is the live durable state and `manifest.json` is the
fixed-point product. The limits are policy inputs to the automatic scheduler,
not instructions to manually choose work.

Successful creep now atomically publishes installable products to
`build/compiler-bootstrap-registry.json`. The canonical
`lower_ast_source_to_ssa` entrypoint reads that pinned registry on first use in
each process, re-proves every receipt and artifact hash, and installs only
source-revision-matched deployments. Registry or source drift is a recorded
non-fatal refusal: authored Python remains active. The current local registry
contains the verified `plan_compute_dispatch` product, and a fresh canonical
compile followed by a WebGPU deployment decision has demonstrated one native
post-activation call with zero post-activation fallbacks.

## Terminal parent result

The parent compile terminated without publishing `unit.json` or `failure.json`.
It is no longer running. Do not interpret the absence of a Python traceback as
success: Windows terminated the process below Python's exception boundary.

- Windows PID: `4284`
- Started: `2026-08-22T04:50:39-05:00`
- At the last live sample: 44.9 minutes, 2357.8 CPU seconds,
  16.55 GiB working set, 29.23 GiB private memory.
- The user explicitly said the compile may reach 50 GiB without error.
- Last durable phase:
  `ssa-source: instantiating complete control/operator deployment`
- Live telemetry:
  `build/compiler-bootstrap-catalogue/lower_control_parent_v363/compile-progress.json`
- Terminal directory (telemetry only):
  `build/compiler-bootstrap-catalogue/lower_control_parent_v363`

Authoritative OS evidence:

- Windows Application Error event 1000 names `python.exe`, process id
  `0x10bc` (= 4284), exception `0xc0000005`, at 2026-08-22 17:14:16 local.
- Windows then reported "Virtual Memory Minimum Too Low" (Application Popup
  event 26) and a matching Windows Error Reporting BEX64 record.
- `tools.diagnose_translation --compilation-unit ...` correctly classifies the
  directory as interrupted before repository SSA, with the last durable phase
  `deployment-instantiation`.

Therefore v363 is a resource/crash frontier in eager deployment construction,
not evidence that the fresh mapping contracts failed and not a repository-SSA
semantic result. The next run must be parent/orchestrator monitored so an OS
crash becomes a durable terminal receipt.

Read-only status commands:

```powershell
Get-Process -Id 4284 -ErrorAction SilentlyContinue |
  Select-Object Id,CPU,WorkingSet64,PrivateMemorySize64,StartTime
Get-Content build/compiler-bootstrap-catalogue/lower_control_parent_v363/compile-progress.json
Get-ChildItem build/compiler-bootstrap-catalogue/lower_control_parent_v363 -Force
```

The command that launched it was:

```powershell
python -m tools.compile_project_catalogue `
  --resolved-process-graph build/compiler-bootstrap-catalogue/lower_control_plan_structural_v360/resolved-process-graph.pkl `
  --process-graph-plan build/compiler-bootstrap-catalogue/lower_control_plan_structural_v360/process-graph-units.json `
  --planned-unit 47 `
  --output build/compiler-bootstrap-catalogue/lower_control_parent_v363
```

This is a source-only baseline. It was not given the verified structural child
as an installed replacement. Consequently its peak memory is the baseline for
the unbootstrapped nine-function recursive compiler unit, not evidence about
the eventual installed artifact's memory cost.

## What was proved before the parent run

The exact blocked operation in `_ControlSSABuilder.lower_while` was the
resident mapping mutation corresponding to:

```python
self.external_values[initial_id] = previous
```

The mapping is authored as `dict[int, SSAValue | None]`. MapIR already retained
that annotation, but numerical subdivision discarded it and exposed anonymous
integer slots. The current worktree now carries the mapping key/value/record
contract through graph reduction, captures it as a structural resident-table
integral, and lowers it through repository SSA's existing sequence-table store
machinery.

The final verified child product is:

```text
build/compiler-bootstrap-catalogue/lower_control_structural_integral_v362/
  repository-ssa.pkl
  repository-verification.json
  unit.json
```

Important identities and hashes:

- Entry: `subdivision_region_376e90f7d1fadca79c81`
- Helper: `ssa_sequence_289_store`
- Identity token chain:
  `process-graph-subdivision / _ControlSSABuilder.lower_while /
  loop-control-owner / loop:288 / region:6`
- Repository SSA SHA-256:
  `9a033fe47fcbb2f14a72d7fe5598dbe7302603502c50376d4eeb1aa3a4873c73`
- Product status: `verified`
- Repository SSA complete: `true`
- Shortfalls: none

The exact SSA evaluator proved all observable mapping-store outcomes:

1. Empty-table insert: key `7`, record handle `42`, length becomes `1`, status
   is inserted (`1`).
2. Same-key update: record handle becomes `99`, length remains `1`, status is
   updated (`3`).
3. Full fixed-capacity table: length and all storage remain unchanged, status
   is capacity-exhausted (`2`).

The verifier also checks before execution that:

- the root and helper sequence descriptors agree;
- the helper call is resolved inside the artifact;
- table storage arguments preserve their exact physical IDs and positional
  dtype/shape ABI;
- the stored value retains `structural_record_identity=SSAValue` and
  `structural_record_handle=True`;
- mutation is caller-visible and the root publishes no fictitious numerical
  output.

This command now gives the correct meta-compilation verdict:

```powershell
python -m tools.diagnose_translation `
  --compilation-product build/compiler-bootstrap-catalogue/lower_control_structural_integral_v362
```

Expected result: one verified integral, zero failures/partials/unverified work,
three exact probes, clean catalogue frontier.

## Critical routing conclusion

Do **not** splice the verified child into the parent merely because it is
verified. The child is the body region's table mutation. The parent was
originally refused because loop 288 had no compiled control program. Linking
only the body would flatten it and silently erase iteration semantics. The
guard in `prepare_graph_precompile` is correct to reject that.

The v363 parent run is the decisive experiment using the fresh v360 resolved
graph after mapping-contract propagation:

- If it succeeds, immediately audit the repository SSA and then build an exact
  semantic/ABI verifier for the complete unit. Only after verification should
  it be installable, with authored source retained as recursive fallback.
- If it terminates with `CompilationSubdivisionRequired`, record the new exact
  boundary. If it is still loop 288 with `strategy=constant`, the next integral
  must be the **loop-control owner**, not another numerical/body region.
- If it resource-fails during deployment instantiation, the next deterministic
  cut belongs inside construction of the callsite-specialized shell tree. Unit
  47 contains nine mutually recursive compiler functions; the constructor
  eagerly expands callsite-specialized instances by call path. The existing
  shell-type cache avoids repeated graph planning but does not avoid the large
  invocation tree. Preserve per-callsite bindings while separating reusable
  shell/kernel structure from invocation records; do not incorrectly share
  mutable shell instances.

## Code changed or relied upon

The worktree is heavily dirty and has concurrent historical work. Do not
assume every line in these files belongs to this handoff, and do not stash,
pop, reset, checkout, or mass-rewrite them.

Relevant current changes are in:

- `src/common/tensors/topological_reducer.py`
  - propagates annotated class-field mapping contracts.
- `src/compiler/glsl_deployment_strategy.py`
  - captures stateful mapping regions as structural resident-table programs.
- `src/compiler/precompile_to_ssa.py`
  - lowers structural integrals through sequence tables and includes the
    generated table-store helper.
- `src/compiler/project_compilation_product.py`
  - structural exact verifier and verified subdivision receipt integration.
  - This path currently appears untracked in `git status`; preserve it.
- `src/compiler/ssa_reference_evaluator.py`
  - import audit repaired by removing evaluator-only `BitLength` semantics.
    Do not re-add an opcode absent from the authoritative LLVM likeness table.
- `tools/diagnose_translation.py`
  - `--compilation-product` now accepts standalone `unit.json` subdivision
    products and reports verified probe counts.
- `tools/TRANSLATION_DEBUGGING.md`
  - records the structural meta-compilation decision-tree branch.
- Corresponding focused tests in
  `tests/test_abstract_tensor_topological_reducer.py`,
  `tests/test_precompile_to_ssa.py`,
  `tests/test_project_compilation_product.py`, and
  `tests/test_diagnose_translation.py`.

## Verification already run

- `tests/test_project_compilation_product.py`: **50 passed**.
- `tests/test_diagnose_translation.py` plus
  `tests/test_project_compilation_product.py`: **70 passed**.
- Earlier combined lowering/reducer/project/diagnostic gate:
  **175 passed, 5 warnings**.
- `py_compile` passed for touched compiler sources.
- `git diff --check` showed no whitespace errors, only expected LF/CRLF
  warnings.

Known unrelated focused-suite noise:

- Four evaluator calibration tests choose the newest arbitrary build pickle;
  the newest was spectral SSA without
  `symbolic_fluid_control__symbolic_fluid_advance`, producing `KeyError`.
- A ProcessGraph fusion test still expects Python `id()` feeds instead of the
  newer deterministic IDs.
- A sequence-table test expects no C sidecar although current native emission
  publishes one.

Do not use those as evidence against the structural verifier without first
repairing or constraining their artifact fixtures.

## Historical next actions (superseded)

1. Preserve and inspect `unit.json`, `failure.json`, repository SSA, and the
   last `compile-progress.json`; do not rerun before reading them.
2. Run the decision tree against the terminal directory. A standalone
   `unit.json` is now supported.
3. On success, run repository SSA stages 1, 2, and 4 against the exported root,
   then implement exact semantics/ABI verification appropriate to this
   stateful compiler compartment. Do not mark it verified merely because it
   lowered.
4. Install only through a token-chain and ABI-matched receipt. Recursive
   compiler inspection must fall back to authored source; a baked compiler
   unit must never recursively call an opaque baked child.
5. Rerun the same unit and record selected implementation, elapsed time, peak
   resident/private memory, and exact output agreement against this baseline.
6. Run only focused compiler/bootstrap regression gates; the repository's full
   suite is not a safe baseline.

These steps describe the investigation that led to the autonomous controller;
they are not the current operating procedure. The structural work was
checkpointed in `839a40d`, the first automatic subdivision crawl in `de8ec68`,
and the self-feeding project/bootstrap controller in `6d8919a`.

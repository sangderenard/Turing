# Next-task intention: compiler bootstrap and wider integration

Date: 2026-11-01. This is the next-task contract and investigation order, not a
report that bootstrap has been attempted or completed on the current changes.
It follows the [work report](threading_control_deployment_work_report_2026-11-01.md).

## Intended outcome

Make the existing compiler bootstrap work against the current compiler, with
honest source/native coverage, concordance and verification receipts. Then use
wider testing to reconcile the new threading/control/deployment capabilities
with older, more complex source programs. Preserve both sets of behavior.

The initial results may look worse than the focused milestone: a large compiler
program can expose incompatible assumptions across pursuit, binding, layouts,
control ownership, exception paths, source partitions and deployment. Expect
that possibility without declaring the code broken in advance. A new test
passing is evidence for its case, not proof that the compiler understands its
own implementation. A bootstrap refusal is a concrete frontier to investigate,
not a reason to add a substitute implementation or suppress the refusal.

## Starting facts and uncertainty

| Observed on this workspace | Not yet established |
| --- | --- |
| Current work is a dirty cloud clone on `work`, base `3eeab1bf`; cumulative untracked files matter. | Which wider failures are pre-existing versus introduced by this work. |
| Focused native/mixed/deployment tests pass as recorded in the work report. | A full compiler bootstrap or full semantic-corpus pass on this revision. |
| Selected default bootstrap registry was absent; no installed native provenance core was demonstrated. | Whether an existing product elsewhere matches current source/toolchain fingerprints. |
| The plan declares provenance core as the blocking first unit. | Whether it now compiles, verifies, installs and survives fresh-process activation. |
| Ordinary source entry activates accepted products before ingestion. | Whether every later compiler stage actually uses the installed native product without fallback. |
| Native section and passive deployment admission are intentionally narrow. | Whether compiler-shaped aggregates, recursion and exceptional control fit those contracts. |

Historical hazards and old build logs are context only. Do not diagnose the
current checkout from them or treat a prior count of failures as current.

## Existing route to use

Read current root/compiler instructions and newest hazard first. Use the
existing bootstrap tools and their actual plan, not another bootstrap driver:

| Existing component | Responsibility |
| --- | --- |
| `tools/bootstrap_compiler_whole_program.py` | Plan and execute the provenance-first whole-program workflow. |
| `src/compiler/whole_program_compiler_bootstrap.py` | Versioned boundaries and blocking schedule: provenance core, then ProcessGraph-to-AOT outer partition. |
| `tools/bootstrap_compiler_concordance.py` | Existing staged/whole-module provenance build and durable failure rollup. |
| `src/compiler/compiler_bootstrap_runtime.py` | Receipt verification, installation and fresh-process selection. |
| `tools/compiler_bootstrap_status.py` | Current registry/coverage evidence, exercise route and accepted result-layout checks. |
| `tools/bootstrap_compiler_exponentially.py` | Existing joined fresh-process generation supervisor, when the published workflow reaches that route. |
| `tools/benchmark_bootstrapped_compiler.py` | Compare authored and installed compiler behavior after correctness/activation is established. |

The whole-module provenance product must satisfy its real module ABI. A raw
bind(C) library or collection of accessor wrappers is not an equivalent native
provenance core. Do not replace a live identity book midway through a process.
The existing workflow requires verification, installation and restart before
advancing the outer compiler partition.

## Investigation and implementation order

1. **Freeze the evidence, not the development.** Record revision, dirty and
   untracked paths, active work/extraction contracts, selected build target,
   toolchains, registry path and bootstrap product fingerprints. Preserve current
   files and old receipts. Use a new versioned output root for new attempts;
   do not reset the checkout or overwrite a failed run to make it disappear.

2. **Measure current behavior before repairs.** Run the mapping concordance
   audit and the focused regression gates, then the bootstrap/provenance test
   group. Record every failure, skip and refusal, including collection/import
   failures. Test successes produced by mocks verify orchestration contracts,
   not a native self-hosted compiler. Establish the wider semantic corpus's
   baseline in recorded batches; do not silently omit expensive/failing cases.

3. **Inspect the actual bootstrap plan and status.** Verify CLI/environment
   settings against current sources. Suggested existing entry points are:

   ```text
   python -m tools.bootstrap_compiler_whole_program --plan-only --output build/compiler-bootstrap-reconciliation
   ```

   Inspect the emitted plan.json directly. The status tool expects an
   exponential-bootstrap bootstrap-state.json and is not a reader for this
   whole-program plan-only directory. Use it only on the appropriate existing
   exponential run. Record the actual selected registry; do not create an
   apparently verified registry just to proceed.

4. **Build the blocking provenance unit first.** Start a bounded diagnostic run
   through the real CLI, for example:

   ```text
   python -m tools.bootstrap_compiler_whole_program --provenance-only --jobs 1 --output build/compiler-bootstrap-reconciliation
   ```

   This is an initial reproducibility setting, not a new permanent scheduling
   policy. Inspect available memory/toolchains and the existing scheduler's
   options before increasing concurrency. Keep the default stop-after-failure
   behavior until the first frontier is understood. Do not skip native
   provenance because a different leaf happens to compile.

5. **Trace each failure through the actual compile.** Keep the submitted source,
   source/policy hash, unit identity, last accepted stage, current graph/SSA
   cells, failure receipt and exact writer/reader disagreement. For topology
   failures, inspect actual operands and control/dependency edges. For type or
   layout failures, locate the producer that should have recorded the fact and
   the consumer that requires it. For stale products, distinguish fingerprint
   invalidation from a semantic miscompile. No AST reconstruction beyond the
   ingestion boundary, spelling-based identity repair, blanket mutation
   exemption, silent type coercion or audit suppression.

6. **Repair shared contracts and preserve regressions.** Reuse the existing
   writer, reader, native section ABI, outliner or synchronization composition.
   Route decisions and rewrites through concordance. Add the smallest real
   source regression that preserves the failing relationship, then rerun the
   original complex unit. A reduced example passing does not close the original
   bootstrap failure. Retain new native thread/exception/deployment tests while
   repairing old cases; do not undo supported semantics to obtain a green
   bootstrap summary.

7. **Verify and activate in a fresh process.** Require matching source/toolchain/
   library receipts, actual semantic verification, correct layouts and the
   declared native module surface. Use existing runtime routing counters and
   status tools to distinguish native calls from permitted source fallback.
   Respect the existing compiler coordination contract where authored Python
   is permitted; never mistake that for permission to implement fundamental
   native program operations by calling Python.

8. **Advance and broaden only after that prerequisite holds.** Run the outer
   ProcessGraph-to-AOT partition through the existing whole-program builder,
   then its current source catalogues/generations. Repeat wider regressions
   after shared repairs. Distinguish compiler worker processes, native authored
   application threads, and passive deployment workers; they are different
   concurrency contracts. Measure performance only after semantic and routing
   equivalence is established.

## First regression battery

Start with the real existing files, in bounded groups with durable results:

- Bootstrap/runtime/status: `test_whole_program_compiler_bootstrap.py`,
  `test_native_provenance_bootstrap.py`, `test_bootstrap_compiler_concordance.py`,
  `test_compiler_bootstrap_runtime.py`, `test_compiler_bootstrap_status.py`,
  `test_bootstrap_result_layout_check.py`, `test_bootstrap_expansion_policy.py`,
  `test_exponential_compiler_bootstrap.py`.
- Book semantics and writers: `test_copy_on_read_scopes.py`,
  `test_provenance_compartment_operations.py`,
  `test_provenance_storage_contracts.py`, `test_provenance_writer_sources.py`.
- Complex control/storage: `test_pythonic_control_torture.py`,
  `test_native_semantics_corpus.py`, call-order, record-layout, returned-list and
  sequence-capacity cases selected from actual failures. Expand from these;
  do not treat this list as a substitute for the wider suite.
- Retained new capabilities: source C exception/thread/Condition/barrier tests,
  dual-IR and structured-region tests, `test_source_c_shell_sections.py`,
  `test_source_native_thread_counts.py`, deployment outlining/lowering tests.

Use standalone AOT applications for new program-execution claims. Existing
bootstrap activation tests or legacy runtime harnesses may have different
contracts; report what they actually verify instead of relabeling them.
The external string-core source/layout setup is excluded per user direction.

## Completion criteria and required handoff

Report progress as a frontier, with exact denominators: discovered, lowered,
backend-emitted, compiled, semantically verified, installed, native-used and
still refused/failed. A plan, DLL/object file, lower-only result, sealed
metadata or passing mocked activation test alone does not prove bootstrap.

The target is a verified native provenance core selected in a fresh compiler
process, followed by a verified outer compiler partition using it, with wider
regressions reconciled and the new capabilities retained. If blocked, report
which prerequisite is unfulfilled and the exact source/graph/ABI contract at
that frontier; do not claim completion of the next stage.

Deliver source changes, meaningful regressions, repeatable commands, current
concordance results and a new hazard receipt. Include remaining shortfalls and
resource/toolchain limits. Do not claim general native LLVM/Fortran thread
operators, coarrays, full string-core compilation or self-hosting merely because
C-wrapper tests pass.

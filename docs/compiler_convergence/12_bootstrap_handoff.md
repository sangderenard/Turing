# Compiler bootstrap handoff

The 2026-10-16 continuation (failure neighborhoods, pursuit dedupe, batch
sort, the two remaining unit failures) is
[16_failure_neighborhood_dedupe_continuation.md](16_failure_neighborhood_dedupe_continuation.md).

The current 2026-10-15 continuation is
[15_provenance_sort_glyph_bootstrap_continuation.md](15_provenance_sort_glyph_bootstrap_continuation.md).
It records the resident sort and glyph work, the latest provenance/bootstrap
frontier, the static-method call-binding regression and the exact next run.

The topology-guided SSA growth continuation is documented in
[13_native_compilation_growth.md](13_native_compilation_growth.md), including
provenance interior candidates, stage/restart APIs, SSA composition and native execution evidence
and the remaining whole-module activation frontier.

This is the continuation brief for the user's work on `sangderenard/Turing`.
The goal is to make a practical, trustworthy self-bootstrap: remove avoidable
compiler overhead, repair incomplete lowering, use resource-aware compilation
units, and run the real bootstrap until it either verifies or exposes precise
remaining failures. Do not stop at faster partial lowering or cleaner reports.

## Non-negotiable user requirements

- Retain adaptive memory discipline: pressure sensing, admission, cache
  contraction/expansion, cold compression, spill and collection. Do not raise
  budgets or disable these mechanisms to claim performance success.
- Remove wasteful Python idioms and repeated broad compiler work where a
  simpler structure or fast path preserves exact semantics and identities.
- Investigate internal recursive subdivision and resizing of jobs. Balance
  optimization across larger pieces against faster compilation and lower RAM
  on smaller pieces, using existing dependency/control boundaries.
- Distinguish necessary identity/semantic accounting from optional diagnostic
  work. Never suppress a real correctness failure to obtain a completion seal.
- The user specifically says executable code outside a DLL has access to the
  native shells' information channels, and progress should be supplied there.
  Continue tracing that existing facility and its binding into compiler code.
  Do not assume progress must be a generic Python callback or invent a second
  channel because the isolated helper currently lacks a binding.

## Repository and review state

- Draft PR: https://github.com/sangderenard/Turing/pull/783
- Remote branch: `codex/bootstrap-efficiency-2026-10-13`.
- Base inspected and used for comparisons:
  `61a4818ae63c89cc160597f4550eec70d96ec0c4`.
- Remote implementation head before this handoff:
  `9cecf5dc9b0849b37a8225d62c4542b1282e889b`.
- Working checkout: `/workspace/Turing-perf`, local branch
  `codex/bootstrap-performance`. Untouched baseline: `/workspace/Turing`.
- Local accounting/shell commits have different hashes from their GitHub API
  equivalents but matching trees. Fetch the remote branch for a new checkout;
  do not infer missing changes from the different local commit hashes.
- All changes are reviewable on the draft branch; nothing was merged to main.
  No long-running bootstrap or test worker remains active.

Read `AGENTS.md`, the applicable directory instructions, and the current opening
receipts in `TEST_BASELINE_AND_HAZARDS.md`. Use the canonical public entry
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. Use existing compiler,
IdentityBook, shell and subdivision mechanisms; no parallel substitute system.
Trace the actual wrong writer with exact identities before changing it. Repro
and baseline in separate checkouts. Do not stash or reset source files to test
the baseline. Do not run the full test suite or install/use PyTorch.

Detailed measurements and history:
`docs/compiler_convergence/11_bootstrap_performance.md` and
`docs/compiler_convergence/evidence/11_bootstrap_performance.json`.

## Changes already implemented

1. `project_compilation_product.compile_project_call.report` measures RSS only
   for retained graph-detail snapshots, rather than for discarded events.
   Memory regulator and worker admission reads remain live.
2. `ExtractionContract.subject` caches source availability by exact definition
   identity, bounded to 256 entries. Classification still runs per occurrence;
   derived contracts receive fresh caches.
3. Topological reduction reuses the function-address membership set and finds
   terminal-merge candidates once. Copy-on-read projection counts no longer
   allocate a source-sized complement; members outside the source universe
   still count correctly.
4. Deployment progress samples whole-graph counts at most once per second
   during progress, with a measurement timestamp. Phase boundaries retain
   exact counts. An explicitly empty executable-node tuple immediately returns
   an empty control partition. The top-level source catalogue is intentionally
   non-executable; actual function graphs are planned separately.
5. Live storage takes an exact scalar-type fast path while preserving memo
   overrides, subclasses, aliases and mutable-object handling.
6. Disk compaction validates and copies existing LZMA records without encoding
   them again. Fresh/modified/zlib/raw tiles still receive the deeper bounded
   compression pass. No disk header/codec format or pressure tier changed.
7. Project completeness now consumes existing `closure_formals` receipts for
   exact surviving argument IDs, matching the existing SSA signature checker.
   A stale capture receipt cannot explain another argument.
8. `_record_unplanned_callable_inputs` in `fortran_c_shell.py` records surviving
   unplanned callable-input invocation edges in the existing unresolved
   `SSACallRecord` table, derived from source node cells. Without this, fixing
   closure accounting let a no-argument callback body falsely appear complete.
   Already resolved/represented calls are preserved. Records batch per function.
9. Two `device_ns` locals in `profiled_c_shell.py` now use the declared callback
   ABI's `unsigned long long`, fixing LP64 Linux incompatible-pointer errors.

Regression coverage lives in `test_project_compilation_product.py`,
`test_extraction_contract.py`, `test_copy_on_read_scopes.py`,
`test_deployment_progress_cost.py`, and `test_live_tiles.py`.

## What actual runs established

On Python 3.12.14, four CPU quota cores, a 16-GiB cgroup, no swap, and the pure
Python tensor backend, bootstrap used `--jobs 2 --min-workers 1` with the
existing default 10-GiB worker/aggregate budgets, 2-GiB reserve and 75-percent
decoded-front elasticity. Budgets were not raised for speed.

- Actual source closure: 133.78 seconds baseline versus 105.91 seconds changed.
- Actual tensor callsite specialization: 581.12 versus 185.07 seconds (3.14x).
- Before the empty-partition fix, arrival at catalogue control partitioning:
  1058.66 versus 591.62 seconds.
- With the empty-partition fix, a run progressed to roughly 160 of 621 function
  shells, crossed the worker's 10-GiB threshold, then spent substantial time
  adopting live storage and compacting disk tiles. It was stopped after about
  31 minutes to make consistent storage changes. It did not finish.
- The final storage run reconfirmed all four partial helper receipts and was
  deliberately stopped at 337 seconds during lexical extraction. The scalar
  and LZMA fast paths have pressure/native tests and component measurements;
  their whole-bootstrap benefit has not been measured.

There is no verified full bootstrap or whole-bootstrap speedup claim. The
memory budget is a regulation/admission policy, not a hard RSS cap. Running
workers are not terminated at the threshold; one indivisible large fact and
initial adoption can exceed it. Source-node limits also are not byte caps.

## Fresh correctness blockers

Fresh helper outputs are in
`build/bootstrap-helper-accounting-final/<helper>/unit.json`; each also retains
`compile-source.py`, `resolved-process-graph.pkl`, `process-graph-units.json`,
and `repository-ssa.pkl`. These local build artifacts are not the portable
handoff: regenerate them through `compile_project_call` when moving machines.

| Helper | Missing used parameters | Missing captures | Unexplained root IDs |
| --- | --- | --- | --- |
| `_report` | `message` | none after repair | none after repair |
| `close_region_feed_set` | `program` | none | 9, 8, 6, 5, 7 |
| `rooted_attribute_path` | `expression` | `root_parameters` | 4, 40 |
| `collect_hierarchy_values` | `closure` | `collect_hierarchy_values`, `executable_value_ids` | none |

All are nested functions of `_lower_process_graph_to_compilation` in
`src/common/tensors/accelerator_backends/aot_compile.py`. All still report
`repository_ssa_complete=False`. The builder requires every authored source
record, native status and native verification to be complete. Complete inner
planner regions must not upgrade an incomplete enclosing source method.

For `_report`, the chain is observed:

- Source graph Input 0 is `message`, Input 1 is `progress`.
- Call 4 has the exact callee edge to 1 and argument edge to 0.
- Conditional control retains `body_callsite_ids=(4,)`.
- An observation hook at `_schedule_loop_callsites` sees no hierarchy items;
  there is no `PlanCall` for the callback invocation.
- Final SSA contains the comparison/conditional but an empty true arm.
- Existing closure metadata correctly identifies `progress` as argument 1.
- The repaired publisher recognizes it. The unresolved-call table now records
  call 4, callee `progress`, with its input ID and missing-invocation diagnostic.

The other helpers read FusedProgram steps/feeds, inspect AST nodes and traverse
hierarchy records/captured mutable sets. Their missing representation and
anonymous formals are observed. The exact first wrong writer for each remaining
loss has not been identified; do not call an ABI annotation a proven fix.

## Priority 1: bind progress through the existing shell interface

Relevant mechanisms already exist:

- `shell_telemetry.py`: `TelemetryChannel`, five record kinds,
  `CompileProgress.progress_callback`, `record_launch_profile`.
- `profiled_c_shell.py`: `turing_launch_logger(user, profile)`,
  `turing_profiled_launch_ex`, optional bounded `TuringTraceRing`, and trace-site
  context. Ring records carry sequence, shell/device timing, region and status.
- `fortran_c_shell.emit_native_c_shell_source`: standalone C main passes the
  existing native logger/site context around C/LLVM/GLSL/Fortran compute.
  Trace is compiled out when disabled; retaining that zero-cost option matters.
- `shell_io.py`, `_native_shell_boundary_lines`, and
  `_install_external_reference_calls`: existing explicit shell capabilities,
  boundary plans and external invocation paths.
- Normal Python lowering wraps its diagnostic callback with
  `CompileProgress.progress_callback` and passes it down to the AOT stage.

Observed gap: the standalone authored helper request in
`whole_program_compiler_bootstrap.py` supplies a source/entry/contract but no
record declaring that this particular capture is the shell's progress endpoint.
`compile_project_call` lowers it as an independent source root. The native
profile logger's signature is not `progress(str)` and its ring has no arbitrary
message payload. This is a binding/representation gap, not proof that native
shells lack information channels.

Trace the intended endpoint, exact capture identity, boundary metadata and text
representation end to end. Use the existing shell contracts and StringTable/
external-reference machinery where applicable. Confirm shell ownership for
executable coordinator code outside the DLL. Do not equate unrelated function
pointer signatures, infer a binding from the spelling `progress`, execute a
Python callback as a claimed native implementation, or bypass the unresolved
call gate. First prove a small real `_report` source lowering and a native
shell-delivered message; then integrate it into bootstrap request publication.

## Priority 2: repair compiler-object input lowering

Recompile the three other actual nested helpers individually. Start at their
source Input IDs, field/sequence operations and captures, then follow reduction,
planning, control/SSA, call-frame adaptation and final metadata. Identify the
first producer that loses the canonical input/capture/storage identity. Consult
existing program/record/sequence ABI mechanisms before changing them. Preserve
FusedProgram operand identities and state/alias semantics. Add exact small
native checks against the authored Python when the body actually becomes
representable; never treat a complete receipt alone as execution proof.

Two broader native shaped-closure tests also fail on unchanged main because
imported `binary_value` consumes undefined switch constants (including IDs
131..141). These are a separate native correctness frontier. Trace their
literal identities and the pass dropping/omitting them before repairing.

## Priority 3: move bounded jobs before the broad RAM peak

Current holistic `auto` schedules flat deployment-region jobs during SSA
expansion, after broad catalogue/deployment planning. More threads there do
not eliminate the earlier peak.

Existing mechanisms to extend:

- `compile_partitioned_source`: resolve/reduce once, publish scoped cuts and
  source concordance, release the planning graphs, then admit unit workers.
- `publish_resolved_process_graph_inputs`: refuses oversized source-node cuts
  before serialization. Native interface tiles retain one dependency SCC's
  bodies and import accepted declarations/objects; legacy transitive closures
  can repeat broad dependency bodies.
- `compile_process_graph_creep`, `publish_process_graph_subdivision_plan`,
  `compile_process_graph_subdivision_integral/plan`: deterministic recursive
  child plans, depth limits, fixed-point detection and function-shell routing
  when stateful storage requires it.
- Existing admission/elastic storage policies, rather than a new scheduler.

Bring a valid existing boundary earlier for bootstrap, with accurate ownership
of closures, mutations, calls and native interfaces. Measure per-unit time/RSS
and repeated dependency work before deciding how to resize cuts. Keep broad
optimization within a bounded SCC where possible; an unsupported interface is
an explicit frontier. Do not silently replace public holistic auto with legacy
partitioning or change its return/product contract without understanding it.

The default worker floor is 12 and can override memory estimates when ready
work exists; the measured diagnostic runs explicitly used floor 1 and max 2.
Review that existing policy against the user's resource-discipline requirement
before scaling concurrency. Never launch a pool based only on CPU count.

## Validation, environment and reproducible commands

Completed checks:

- Initial performance/storage focused groups: 229 passed, one existing
  optional-torch failure. Seven older progress UI failures and several
  aggregate/native failures were also reproduced on unchanged main.
- Accounting follow-up: 99 project/bootstrap/scoped-input tests passed.
  A numeric nested closure executed natively via LLVM against Python for three
  input/capture pairs. Stale capture IDs and omitted no-argument calls are gated.
- Two shaped native closure checks failed on both worktrees on the same
  undefined imported `binary_value` operands.
- Shell follow-up: four focused checks passed. A real C executable made three
  native launches into a two-entry ring, retained sequences 1/2, region 42,
  status 1, and reported one overwritten record without a Python logger.
- Mapping audits before/after: zero identity findings, unchanged 21 unsourced
  facts. The unsourced latch remains open; do not report a fully sealed audit.

Workspace dependencies are isolated under `/workspace/turing-deps`,
`/workspace/turing-compat` and `/workspace/turing-toolchain`. Joblib 1.5.2 is
used because 1.6 lacks the repository's `joblib.externals.cloudpickle` import.
GCC replaces an unavailable inherited clang setting. Zig's cache must be
writable. PyTorch and the unavailable nodus tensor core were not installed;
the same pure Python tensor backend was used across comparisons.

From `/workspace/Turing-perf`, focused continuation checks:

```sh
PYTHONPATH=/workspace/turing-compat:/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ ZIG_GLOBAL_CACHE_DIR=/workspace/zig-cache python -m pytest -q tests/test_project_compilation_product.py tests/test_whole_program_compiler_bootstrap.py tests/test_scoped_compilation_inputs.py
```

```sh
PYTHONPATH=/workspace/turing-compat:/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ ZIG_GLOBAL_CACHE_DIR=/workspace/zig-cache python -m pytest -q tests/test_live_tiles.py tests/test_memory_regulation.py tests/test_copy_on_read_scopes.py tests/test_deployment_progress_cost.py
```

```sh
PYTHONPATH=/workspace/turing-compat:/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ python -u tools/audit_identity_concordance.py mapping
```

Compile an actual helper through `compile_project_call`, which internally uses
the canonical public source entry (choose a fresh output directory):

```sh
PYTHONPATH=/workspace/turing-compat:/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ python -u -c 'from src.compiler.project_compilation_product import compile_project_call; r = compile_project_call("src/common/tensors/accelerator_backends/aot_compile.py", "_lower_process_graph_to_compilation.<locals>._report", "build/handoff-report-repro", progress=print, progress_bars=False); print(r["root_parameter_accounting"], r["extraction_boundary_accounting"])'
```

Once concrete helper/interface repairs justify another whole run, preserve
budget/admission comparisons and use a fresh product directory:

```sh
PYTHONPATH=/workspace/turing-compat:/workspace/turing-deps:/workspace/turing-toolchain:. CC=gcc CXX=g++ ZIG_GLOBAL_CACHE_DIR=/workspace/zig-cache python -u -m tools.bootstrap_compiler_whole_program --output build/bootstrap-handoff-next --jobs 2 --min-workers 1
```

The user has authorized actual bootstrap iterations. Still reproduce and repair
small failures first; do not spend another long run rediscovering a known partial
helper. Report completed stages, actual RSS/pressure behavior, receipts and
native checks. A stopped run is not a failure traceback or a completion proof.

External observation-only sampling used
`/workspace/turing-profiling/sitecustomize.py`, activated with
`TURING_SAMPLE_COMPILER=1` and that directory first on PYTHONPATH. It samples
the actual worker stack, not a substitute compiler. It is optional, outside
the repository, and not required to reproduce correctness failures. Component
benchmark scripts and native channel harness also remain in that directory.

## Completion criteria for the continuation

The intended result is a verified bootstrap with practical recorded wall time
and RAM behavior under retained adaptive regulation. Show that authored helper
behavior and accepted shell bindings actually execute, that dependency/native
interfaces preserve identities and state, and that the final builder seals only
complete and natively verified units. If a further blocker remains, provide its
exact source/SSA records, first wrong writer, reproducible small command and
next actionable repair. Keep the draft PR and evidence current; do not claim
that reporting a missing operation implements it.

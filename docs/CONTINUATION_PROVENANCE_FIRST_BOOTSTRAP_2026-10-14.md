# Provenance-first bootstrap continuation — 2026-10-14

Branch: `codex/dense-storage-bootstrap-depth-2026-10-14`.
Draft PR: https://github.com/sangderenard/Turing/pull/784 (base: `main`).
This supplements [the implementation receipt](CONTINUATION_NATIVE_BOOK_IMPLEMENTATION_2026-10-14.md) and supersedes its statement that first-stage scheduling has not been implemented. The actual native book binary still has NOT been built or installed.

## Start here on the local machine

Fetch and check out the PR branch, preserving any existing local work in a separate worktree. Do not start from the checkpoint branch: its requested changes are already merged here. Read repository `AGENTS.md`, `src/compiler/AGENTS.md`, and `TEST_BASELINE_AND_HAZARDS.md`. Use a local environment with the repository's compiler dependencies and a working native toolchain. `/workspace/turing-venv` and the diagnostic paths below belong only to the cloud workspace.

```bash
git fetch origin
git switch --track origin/codex/dense-storage-bootstrap-depth-2026-10-14
python -m tools.bootstrap_compiler_whole_program --plan-only --output build/provenance-handoff-plan
python -m pytest -q tests/test_native_provenance_bootstrap.py tests/test_provenance_compartment_operations.py tests/test_compiler_bootstrap_runtime.py tests/test_whole_program_compiler_bootstrap.py tests/test_bootstrap_expansion_policy.py tests/test_pursued_parameter_annotation_identity.py tests/test_copy_on_read_scopes.py -k 'not pursued_nested'
```

If the branch is already checked out locally, switch to it and fast-forward instead of creating it again. The final focused gate passed **55 tests, two deselected, in 2.72 seconds**. The excluded broad pursuit cases were not validated by this gate. The full test suite is not a useful handoff smoke test; consult the repository's baseline hazards.

For a fresh, expensive continuation from the repaired scheduling frontier:

```bash
python -m tools.bootstrap_compiler_concordance --lower-only --output build/provenance-core-reproduction
```

It writes `compartment.json`, `process-graph-units.json`, and the resolved-graph checkpoint when that phase succeeds. `--lower-only` still runs the full source-to-SSA lowering; it does not stop at graph discovery. The latest supported run exceeded 20 minutes after advancing into cold-tail adoption, so give this command a larger external time budget. Expect substantial RAM. Do not mistake a saved checkpoint for compiled or installed provenance.

## Merge

Merged `96edd51c14a1344f8319db5f745726f388285439` from `codex/local-state-2026-10-14`, including predecessors `57be687` and `1ff362d`, without textual conflicts. Merge commit: `d05401e`. This preserves both branches through their shared history. The incoming rank-preserving table-column receipt required updating one test expectation to include scalar rank zero.

## User's required boundary

The first critical compiled unit is the complete provenance/book subsystem. It must do enough work internally to amortize operator entry/exit:

- Posting, validation, concordance, revisions, timestamps and scope bookkeeping.
- Canonical coordinate interning, row/scope/history searches, forward and reverse provenance indexes, and traversal.
- Correlation and audit searches, finding construction, sorting, complete log formatting and compressed log writing.

No Python callback per record, field, edge, formatted line or hash comparison. No per-accessor shared libraries. Existing whole-book operations such as complete report rendering and log writing are coarse entrypoints; their loops and intermediate data stay native. Returning final results or batches to Python is permitted. Formatted streaming must remain one native operation instead of returning one line per Python dispatch.

Native records must be born native after restart. Do not retain a complete Python book as a shadow heap, then marshal it into a native table on every call. Opaque authored leaves may still require retained external references; account for those separately.

Two existing capabilities need care rather than blind reuse:

- `NativeObjectBoundary` leases containers and retains source Python objects for its boundary contract. That is not automatically a memory-saving, persistent book implementation.
- Its documented native dictionary lookup contract is **reference identity**, whereas book coordinate tuples require the incumbent value-equality/concordance semantics. Equal newly constructed coordinates must find the same cell. Typed canonical interning/equality must happen inside the native core; pointer identity is insufficient.

The coarse book tests exercise fresh equal coordinates and concord conflicts, which helps catch this distinction.

## Bootstrap and installation changes

The plan now names `provenance-core` as the blocking first unit, followed by the existing ProcessGraph outer partition. The latter cannot submit work unless the current process has selected the native provenance core.

The barrier performs an isolated, synchronous whole-core build in a fresh process. Its environment forces the Python provenance implementation and an empty compiler-product registry, so previous outer deployments cannot enter this first build. No outer jobs from this bootstrap run are admitted while it builds/verifies/installs.

Installation is a new whole-module capability inside the existing bootstrap runtime. It does not replace individual methods on live Python book instances. The selected link implements the standard CPython module initialization interface for the canonical module `src.compiler.identity_concordance`. A raw bind(C) library alone is insufficient.

Before atomically publishing a link, the installer checks source/dependency hashes, library and receipt hashes, interpreter compatibility, whole-module scope and zero fallback probes. A fresh isolated process then checks the actual module import, all source-declared classes/functions/methods, book/scoping semantics, complete formatting, and the designated short compiler program. Failed verification leaves the previous registry unchanged. Publishing ordinary compiler products preserves the provenance link.

Package startup selects the complete module once. Missing or incompatible links retain the incumbent Python implementation for ordinary compilation. The bootstrap itself refuses to proceed with that fallback. Following a new installation, the CLI restarts the interpreter before any outer submission, removing the source-only flag. The new process then imports the native module before any books exist. Future compiler processes use the same pinned link automatically.

This is a tested routing/barrier/installation contract, **not evidence of a working native provenance module**. The current builder still stops before producing the standard module binding. Its existing terminal native stage is `native-built-unverified`, which the barrier deliberately refuses. Completing the native object/operator binding and generating a verified module receipt remains implementation work.

## Designated short program

Use:

```bash
CC=gcc CXX=g++ python -m tools.compiler_probes.provenance_acceptance \
  --backend python --output build/provenance-acceptance-python

CC=gcc CXX=g++ python -m tools.compiler_probes.provenance_acceptance \
  --backend native --output build/provenance-acceptance-native
```

Use the configured C compiler on the target machine; the overrides above describe this workspace. `auto` uses normal startup selection. `native` refuses to run the program when the provenance core is unavailable; it never silently benchmarks the Python book as native.

The designated program is the existing `c-guarded_splice` case in `test_pythonic_control_torture.py`, not the entire torture suite. It exercises `while`/`break`, independent guards, mutable nested rows, starred tuple assembly, `enumerate`, destructuring and six repeated native calls covering guard combinations. It compiles the submitted program fully to C, verifies outputs and checks concordance. Its C case verifies cardinality/output behavior; it does not claim exhaustive nested payload or Python-language coverage.

The runner saves `program.log` and a machine-readable `receipt.json`, including actual provenance backend, zero expansion depths, elapsed wall time and maximum child-process RSS. This is maximum process RSS, not simultaneous aggregate worker RSS.

Measured on the final Python-book path:

| Result | Figure |
|---|---:|
| End-to-end compile and native program verification | passed |
| Wall time | 5.646322 seconds |
| Maximum child-process RSS | 147,619,840 bytes |
| Optional unroll / inline / custom-version depths | 0 / 0 / 0 |

An earlier fresh run took 5.783979 seconds and 146,657,280 bytes. The initial direct pytest run took 9.91 seconds while another large compile was active. These are individual observations, not a controlled speedup comparison. Native-provenance mode currently fails explicitly because no core is installed.

The final handoff rerun also passed: **4.809455 seconds**, **147,636,224 bytes** maximum child-process RSS, actual Python provenance and native C program execution. This is another source-backend observation, not a native-book speedup. Plan-only also completed and named `provenance-core` first; changed-source syntax compilation and `git diff --check` passed.

Book-level acceptance also includes copy-on-read scope tests and a 100-fact provenance chain. The complete rendered chain is 23,603 bytes with SHA-256 `912c1bf93a1c0e1ff52b5dfffd731e6114f3d2210f9e36e30044baa582288620`, independently confirmed identical on untouched main and this branch. Native installation must reproduce it and the compressed log's decoded content, along with exact edges, histories and concord refusals.

The storage benchmark now times complete book rendering as a single coarse operation. One Python run over 10,000 facts / 29,999 materialized cells rendered 2,072,333 bytes in a median 0.268243 seconds across three repetitions. Other compiler activity overlapped that run; it establishes workload scale, not a performance comparison. There is still no native-book memory or speed figure.

## Native compilation diagnostics and identity repair

The merged whole-module source reached 58,522 nodes, 63,437 edges and 493 function graphs. The checkpoint writer initially used plain pickle, which refused module bindings after 304.51 seconds. It now uses the compiler's existing resolved-graph serializer.

The next attempt saved a 171 MiB graph checkpoint and stopped after 320.24 seconds on an exact-source graph conflict for `Registry.__init__`. Both submitted and imported views have that canonical definition identity, but distinct declared function references (69 and 235) and separate value-ID spaces. Blindly merging them or accepting the first graph would be unsafe. Reconciliation now keys those graphs by canonical definition plus declared function instance. Lowered functions carry the same reference. All 493 saved graph instances index successfully; missing instance identity does not guess by local name or dotted suffix.

A diagnostic replay using the saved graph and its owning identity book then reached the existing shader-region failure after 180.57 seconds, at a measured maximum process RSS of 1,781,840 KiB. This replay is diagnostic only, not an accepted native compilation product.

The first concrete source call cycle was:

`IdentityBook.post -> IdentityBook._materialise_row -> IdentityBook.post`.

The catalogue also has the self-recursive `scope_rows` call component. **The function-call edges already exist. No missing-edge repair is indicated.** The catalogue projection presented inter-function `callee` edges plus function-body ownership edges to a shader reducer requiring an acyclic graph. `reduce_scheduled_shader_regions` copied the whole semantic graph and checked that copy for acyclicity.

The scheduling repair now treats the already-authored `recursive_backedge` call nodes as control boundaries alongside recursion-table control members. It removes only edges incident to those exact nodes from the shader-scheduling projection. The semantic graph, function-body ownership, call edges and termination guards remain untouched. Planner levels are computed against that acyclic projection when the semantic graph cannot provide a complete schedule; ASAP, ALAP, max-local-slack and the incumbent unknown-name fallback retain their existing meanings. A regression constructs the complete definition-body/call-callee cycle and proves ordinary deployment selection succeeds without deleting either authored operand.

A fresh supported public run used the command above with an external 1,200-second bound. The full 58,522-node / 63,437-edge catalogue again converged callsite specialization in three rounds. Root control partitioning took **76.931599 seconds**. Shader-region reduction then passed in **2.70499 seconds**, starting from 3,211 regions and reaching 2,993 after 164 merges. All **2,993/2,993** dispatch subgraphs extracted successfully in **739.856801 seconds** with the resident-region cap holding at eight. The external bound expired during cold-tail identity-page adoption (`roots=497`, `0/72` reported); there was no downstream compiler exception. This is proof that the original scheduling blocker is repaired on the supported whole-module path, but it is not a completed lower-only run and produced no native module.

A subsequent **diagnostic-only** replay selected all 493 function references through the existing `_lower_resolved_process_graph_deployment` selected-function path, with `activation_function_references` equal to those references and `runtime_closure_only=True`. That path uses the existing empty catalogue shell and schedules individual bodies. It passed the catalogue shader check, then failed after **78.768448 seconds**, maximum process RSS **1,541,644 KiB**, on:

```text
ConcordanceRefusal: post planner_specialization
(("lexical_reads:('src.compiler.ssa_object_operations', 'validate_object_call')", 0), '_helpers'):
REVISE without a changed source; every Derived cell is stamped at or before
the row's previous revision (485655) and the cell set is the previous revision's
```

Trace: child `strategize_shell_deployment` -> `_propagate_callsite_planner_specializations` -> `_post_planner_specialization` -> `_post_if_changed` -> `IdentityBook.post`. This is not an accepted compilation: the private selected-function interface normally consumes a recorded dependency-closed unit, and selecting every dependency as an externally activated root is only a diagnostic. It is not yet a public whole-module library election, and it has not proven module initialization correct. Do not suppress this refusal, manufacture a newer causal timestamp, or loosen concordance.

Inspect the proposed/previous specialization values and their source cells, including mutable `_helpers` state and repeated original/copied graph visits. Externally callable library roots cannot safely specialize their arguments solely from a subset of internal calls. This is an investigation direction, not an established explanation of this refusal. A fresh supported public compilation must reproduce any proposed repair.

The saved graph's `compilation_unit_identity_book` is an acceptance subset, not evidence that it contains the entire original compilation history. Restore the correct owning history when replaying provenance-sensitive phases. The 171 MiB checkpoint, replay scripts and full logs were scratch files under `/workspace`, **not committed artifacts available to a local checkout**. The failure details above are the portable handoff evidence; regenerate checkpoints locally with the public command.

The complete native book is therefore still blocked before SSA/native emission. A standard whole-module binding must also be produced after that lowering succeeds. No installed/native routing claim should be made until actual binary verification passes.

## Remaining work in order

1. Continue the supported whole-module deployment election past cold-tail adoption with a time budget longer than 20 minutes. Preserve authored global initialization (including `REGISTRY`, sentinels, context state, page/stage declarations and method wrapping); an empty catalogue alone does not implement these effects.
2. If the earlier diagnostic specialization refusal recurs on that supported path, diagnose it on the correct owning history. Keep full canonical module/definition plus declared-instance identities. Do not reintroduce indexes based on local names, dot suffixes or guessed owners. It has not recurred in the root catalogue through dispatch-subgraph extraction.
3. Complete native SSA/emission for the entire book operator surface, with contiguous native-owned records/indexes and internal searches/formatting. There is **no contiguous native book storage implementation in this branch yet**. The small slotted `Ref` change does not satisfy this requirement.
4. Produce the whole-module binding and verified receipt consumed by `ensure_native_provenance_core`; the current emitter produces only a raw library if it reaches emission. Prove import, globals, queries, equality/concordance, complete formatting and the designated compiler program with the actual native book. The barrier tests currently exercise controlled build outcomes, not a successful binary installation/restart.
5. Run the actual bootstrap CLI, verify provenance is installed before outer work, verify restart and automatic native selection in subsequent fresh compiler processes. Compare Python/native memory and wall time on identical workloads, including retained opaque Python leaves and crossing costs. Keep optional unroll/inline/custom-version expansion depths at zero for the initial bootstrap; this must never truncate execution, source discovery or dependency discovery.

Until these pass, keep this PR draft and do not merge it as a completed native bootstrap. Ordinary compilation still uses the Python book without an installed link, but the whole-program bootstrap now deliberately blocks rather than submitting outer work without the required core.

## Validation

- Bootstrap runtime, barrier, coarse operations, depth policy and canonical identity group: 32 passed, two broad nested-pursuit cases intentionally not selected.
- Source graph/literal provenance lookup group: 12 passed.
- Copy-on-read scopes and coarse query/formatting operations: 25 passed.
- The short designated program passes with the actual Python-book backend and fully native C program execution.
- Broader merged extraction tests reached 58 passes before the autograd `_tape` source-pursuit failure; the same case fails on untouched main in this environment. It is not suppressed.
- Python syntax compilation and diff checks pass. Tests of the barrier use controlled build results; they do not demonstrate successful native module generation.
- The scheduling/region gate passes 50 tests. The focused handoff gate still passes 55 tests with two broad nested-pursuit cases excluded. Three shader execution tests reach the known worker limitation that `glProgramParameteri` is unavailable; their pre-installation pipeline tests pass.

Keep the PR in draft. Remaining acceptance is actual core compilation, whole-module binding, native semantic verification, restart/use by the compiler, and fresh-process memory/time comparisons against these same workloads.

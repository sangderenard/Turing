# Work report: dual IR, native control and deployment

Date: 2026-11-01. This report closes the threading/control/deployment work and
hands off to compiler bootstrap and wider integration testing. It records
observed results, not a claim of general compiler convergence.

## Repository and execution context

Canonical repository: `C:\dev\Powershell\turing`. Work in this session was
performed in its cloud clone, `/workspace/Turing`, branch `work`, based on
`3eeab1bf`. The working tree contains cumulative modified and untracked files;
this is not a clean, committed release. Preserve that work. No commit, push,
reset or bootstrap installation is represented by this report.

Native execution evidence is Linux standalone AOT applications. Tests used
GCC and the repository's LLVM toolchain route. Matching Debian GCC14.2.0-19
Fortran compiler/development packages were extracted beneath
`/tmp/turing-fortran-toolchain`; `TURING_FC` selects that compiler. That directory
is temporary, not an installed repository dependency. The optional nodus tensor
core was absent; its compiler-side warning is separate from the native product
execution, which did not use a Python executor.

## Implemented behavior

| Area | Result | Contract preserved |
| --- | --- | --- |
| Source threads and dual IR | Source create/start/join, worker bodies/contexts, captured inputs and synchronization are retained as process/control/dependency facts. Structured regions copy actual graph topology. | Exact callable, capture and shared-storage identities; no spelling-based connection or invented comparative worker order. |
| Scheduling | ASAP/ALAP are configuration choices over eligible graph work. Source schedule and loop identity are retained in passive deployment discovery. | Required edges remain; scheduling does not synchronize unsynchronized side effects. |
| Barriers and fences | Separate control rendezvous and memory/compiler ordering projections, concorded to the same source synchronization where combined. | Explicit participants, generations, execution/memory scopes, semantics and storage; join remains a completion dependency. |
| Native C synchronization | 27 mapped primitives cover context, mutex/token state, waiter queues, latch generations, control/cyclic barriers and completion. | Existing native state engine, ownership, progress, retained outcomes and joined lifetime; unsupported scopes are refused. |
| Source Barrier | Native construction, wait, reset and abort, including captured handles; exact wait result retained separately from effect tokens. | Reusable phases, timeout/broken outcomes, arrival index, reset/abort wakeups. Unrelated same-spelled methods stay unrelated. |
| Exceptions and resource control | Explicit SSA records/failure arms and actual CFG/cleanup topology lower through the C error channel. | No C++ exception engine or Python implementation of fundamental operations. Worker failures retain their supported join/shell delivery policy. |
| Mixed AOT products | LLVM and Fortran emitters accept explicit `CShellControl` selection of native numerical sections and return a CModuleArtifact. | C supplies admitted orchestration; the requested backend supplies the selected compute. Existing function/call identities remain intact. |
| Passive deployment | General C emission under `deployment='auto'` discovers a narrowly proven independent source-SSA loop and uses the existing iteration outliner/pool. | Bounded unit-stride writes into one rank-one span; no shared reads/arbitrary calls; serial fallback retains values. |
| Execution budget | `deployment_cores` feeds the existing deployment strategy selector. Two slots means one parked worker plus the participating caller; one slot selects serial. | Physical workers are chosen in lowering, separately from authored logical contexts. |

The existing backend reference now includes the whole control surface, native
symbols, hardware requirements and wrapper/native capability distinctions. It
is documentation of the registry and contracts, not a parallel operator table.

## Concordance and integration repairs

All new section selection, deployment discovery, graph outlining and C policy
choices have declared concordance facts. Native section calls and compiled
piece files derive from those facts. Passive outlining records proposed
before/after topology before installing it, publishes current block geometry,
and validates both the accepted outline record and its actual edges at C use.

Observed failures prompted repairs to the responsible paths: captured-formal
dtype provenance; completion-handle lifetime roles; missing activity ownership
from inappropriate numerical loop unrolling; typed graph operations being
misread as opaque Python escapes; protected-construction binding/continuation
shape; and resource recipes being captured after source mutation. The linked
LLVM build also refused an unrouted PIECE_FILE; its provenance was routed,
rather than exempted. Detailed receipts are linked below.

These repairs are not evidence that every similar interaction is fixed.
Especially important: the original complex bootstrap and broad semantic corpus
were not rerun to completion as part of this focused milestone.

## Measured verification

These runs overlap and describe successive states. Do not add their counts
into a claimed total of unique tests or describe them as one full-suite run.

| Run | Observed result | Scope |
| --- | --- | --- |
| Native control/synchronization combined gate | 207 passed, 579.57s | C barriers, Conditions, threads, exceptions, source special cases/activity, dual IR/regions, SSA synchronization/lock box and pool. Before mixed-section/passive-deployment additions. |
| Mixed-section native regression gate | 87 passed, 351.51s | Barrier/thread/exception regressions, LLVM comparison spelling and Fortran control checks. |
| Deployment regression gate | 44 passed, 97.80s | Existing outlining/lowering, source threads, mixed sections and native counts at that revision. |
| Strengthened mixed/native count gate | 11 passed, 81.43s | LLVM/Fortran scalar sections, 1000 barrier phases, successful and caught-zero-divisor paths, ASAP/ALAP and OS counts. |
| Final count/budget suite | 8 passed, 9.98s | Authored workers, passive two-slot and one-slot execution, serial controls, shared-read exclusion, exact loop/schedule identity. |
| Mapping concordance audit | 28 rows / 2 functions; 0 findings, 0 unsourced; latch CLOSED | Repeated before/after changes. This mapping audit alone is not a whole-bootstrap audit. |
| Mixed source module audit | 237 rows / 11 functions per numerical backend; 0 findings, 0 unsourced | Post-emission actual source modules. |
| Passive source module audit | 26 rows / 4 functions; 0 findings, 0 unsourced | Actual discovered/outlining/pool example. |

`py_compile`, strict C11 syntax checks of the native runtime and
`git diff --check` also passed at the recorded milestones. Native tests retain
output-value checks in addition to scheduling and thread-count assertions.

Observed process thread counts, measured externally through Linux `/proc`:

| Compiled application | Peak native OS tasks, including main |
| --- | ---: |
| Two authored workers | 3 |
| One authored worker with LLVM or Fortran compute | 2 |
| Thread-free source, automatic two-slot deployment | 2 |
| Same source, serial or automatic one-slot policy | 1 |
| Shared-read loop excluded from automatic parallelization | 1 |

The observer watches only the standalone child and writes
`thread-observation.json` beside it. It neither injects an executor nor counts
threads in the Python test process.

## Explicit limits and next boundary

Native sections currently admit scalar numerical leaf functions; Fortran
sections require float64 scalars. Array/aggregate/alias transport, arbitrary
calls, finer splitting inside one function, and exceptional exits across a
native compute boundary remain unimplemented. An ordinary helper call inside
protected source try still needs its explicit failure-return ingestion contract;
the verified mixed test calls compute outside that protected call boundary.

Barrier action callbacks require a compiled callback ABI. General atomic and
deque coverage, cancellation/pause lifecycle, broader passive independence
proofs and native LLVM/Fortran synchronization/coarray/lane mappings remain
separate work. Linux execution does not prove Windows execution.

String-core coverage was investigated: reduced native regressions and source
identity tests do not establish full-core source compilation. Bootstrap
activation is early but does not automatically compile string core. Its external
source/layout setup was explicitly excluded by the user after clarification.

The next task is compiler bootstrap and wider testing. Expect the larger
programs to expose interactions that focused tests do not cover. That is a
working hypothesis; new failures must be measured before being diagnosed.

## References

- [Backend operator reference](backend_threading_operation_reference_2026-11-01.md)
- [Mixed AOT section contract](c_shell_control_native_sections_2026-11-01.md)
- [Native synchronization receipt](../hazards/20261101T165117Z_work.md)
- [Mixed-section receipt](../hazards/20261101T183000Z_c-shell-control-sections.md)
- [Passive deployment and thread-count receipt](../hazards/20261101T190000Z_native-thread-count-deployment.md)
- [Next-task intention report](compiler_bootstrap_next_task_intention_2026-11-01.md)

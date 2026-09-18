# Handoff: compiler bootstrap, recursive PE extraction, machine-state SSA, and representation unity

**Date:** 2026-08-12 (live state captured at approximately 20:27 America/Chicago)
**Repository:** `C:\dev\Powershell\turing`
**Branch:** `codex/recursive-reduction-bridge`
**HEAD at capture:** `70b977b` (`Add handoff: whole-object emission -> modular tensor-op SSA; loop engine is next`)
**Status:** active work in a very broad dirty worktree; do not reset, bulk-revert, clean, or stage indiscriminately.

This is the comprehensive continuation record for the work performed and
investigated on 2026-08-12. It supplements rather than replaces:

- `HANDOFF_tensor_op_ssa_modules.md`, which carries the longer source compiler,
  tensor, loop, and whole-object history;
- `BINARY_HEAD_IR_CONTINUATION.md`, which records the earlier binary-head pause
  boundary and the 308-form vocabulary state;
- `PE_SSA_REPEATED_OPERAND_ALGORITHMS_REPORT.md`, which proposes structural PE
  fixture families and exact reduction-analysis constraints.

The most important correction in this handoff is that the source compiler and
the machine-code subsystem are not competing entrypoints and must not be
collapsed into one falsely uniform IR. They are a tightly related family of
program representations connected by explicit, proof-carrying translations.
Machine-state SSA may reuse the repository's `IRModule`, `Function`,
`BasicBlock`, `Instr`, and `SSAValue` containers, but retained machine
instructions are not ordinary repository SSA until their complete architectural
state effects have been legalized.

---

## 1. Active objective

The active goal is:

> Iteratively expand PE/AMD64 decompilation coverage for cached host-code
> modules, eliminate `compile()` blocker occurrences by adding exact decoding
> and machine-state semantics, and determine and implement the sound boundary
> between a decompiler-specific machine SSA dialect and the existing repository
> SSA without numerical projection or runtime fallback.

The larger compiler objective remains one statically known program ecosystem:

1. ingest complete authored/source dependency closures;
2. retain full program structure, including loops, collections, object fields,
   tensors, calls, and process dependencies;
3. lower that structure to compilable repository SSA;
4. use machine-code ingestion when a dependency is available only as PE/AMD64;
5. recursively pursue the PE dependency surface rather than stopping at the
   first external import;
6. distinguish exact machine-state retention from successful translation to
   ordinary repository SSA;
7. ultimately compile deep compiler/planning code into a static code ecosystem.

### Non-negotiable constraints

- No numerical projection as a substitute for full-program extraction.
- No Python callback, runtime-language escape, opaque external handler, or
  tensor backend call standing in for compiled source.
- No deletion of difficult constructs to make completeness reports green.
- Loops may remain loops. They do not need to be unrolled.
- Control-only operations are still program operations and must not be dropped.
- Every blocker occurrence is retained, including duplicates at distinct
  callsites/functions.
- Source pursuit has no capricious depth, function-count, or fixed-point limit.
- Dynamic control destinations may remain typed residuals when their target
  genuinely depends on runtime machine state; they are not decode failures.
- The reference scalar AMD64 decoder is authoritative. Tensor decoding is an
  optional accelerator/verifier and may never narrow reference coverage.
- The active long extraction is a single process. Never launch a duplicate
  compile or profiling process against it.
- `FusedProgram` is a bounded numeric-region representation in design flux.
  It is not the compiler, not the full dual IR, and not the center of the
  representation-unity effort. The full graph-to-section renderer is the
  relevant non-numerical route.

---

## 2. Live process at handoff time

Exactly one recursive disk-cached host extraction was running when this file
was written.

| Field | Value |
|---|---|
| PID | `1176` |
| Start time | `2026-08-12 19:42:12 -05:00` |
| Command | `python -u scripts/extract_compile_host_library.py` |
| Root | native CPython `compile()` |
| stdout | `build/compile-host-library-v3.stdout.log` |
| stderr | `build/compile-host-library-v3.stderr.log` |
| Last observed elapsed | about 51 minutes |
| Last observed CPU | about 2,872 seconds |
| Last observed working set | about 17.38 GiB |
| Last observed private memory | about 25.21 GiB |
| Responsive | yes |
| stderr bytes | zero |
| stdout content | only the imported pygame greeting so far |

The process was CPU-active and responsive at every roughly 60-second sample.
Memory growth was gradual and consistent with the previous extraction's
serialization-heavy final phase. Do not kill it solely because memory is high.
The user explicitly requested that a healthy run be left alone and checked in
approximately 60-second rounds.

### Monitoring protocol

Use only read-only process/log inspection while it is alive:

```powershell
$process = Get-Process -Id 1176 -ErrorAction SilentlyContinue
if ($null -eq $process) {
    'STATUS=EXITED'
} else {
    $elapsed = (Get-Date) - $process.StartTime
    "ELAPSED_MIN=$([math]::Round($elapsed.TotalMinutes, 2))"
    "CPU_SEC=$([math]::Round($process.CPU, 2))"
    "WORKING_GB=$([math]::Round($process.WorkingSet64 / 1GB, 2))"
    "PRIVATE_GB=$([math]::Round($process.PrivateMemorySize64 / 1GB, 2))"
    "RESPONDING=$($process.Responding)"
}
Get-Content build/compile-host-library-v3.stdout.log -Tail 12
Get-Content build/compile-host-library-v3.stderr.log -Tail 12
```

Do not attach a second Python profiler and do not launch another recursive
extraction. Tests were deliberately stopped during the final high-memory phase.

### When PID 1176 exits

1. Capture complete stdout and stderr before editing the fingerprinted modules.
2. Inspect `build/compile-host-library-report.json` if it exists and is newer
   than the run start.
3. Preserve both the raw and effective blocker occurrence ledgers. Do not turn
   them into a set or summary-only count.
4. Record every dependency edge, including repeated occurrences.
5. Determine whether failure occurred during lifting, cache serialization,
   library materialization, report serialization, or process termination.
6. Do not start another long extraction until focused fixes and tests pass.

The report writer is `scripts/extract_compile_host_library.py`. Its report
includes root identity, unit/function counts, dependency occurrences, unresolved
dependency occurrences, raw/effective blockers, completeness flags, per-unit
cache metadata, and every occurrence record.

---

## 3. Worktree warning and ownership

The worktree contains tens of thousands of changed lines across source compiler,
tensor, loop, PE, SSA, backend, and test files, plus many untracked artifacts.
Some changes predate this continuation; some belong to the user or other work
streams. Preserve all of them.

Do **not**:

- use `git reset --hard`;
- use `git checkout --` or `git restore` broadly;
- run `git clean`;
- delete `.native_re_compile_*` directories;
- stage all changes;
- assume an untracked file was created in the current turn;
- reformat large files mechanically;
- weaken tests to accept incomplete lowering.

Today's directly relevant newly added or edited files include, but are not
limited to:

- `src/compiler/cpython_compile_ssa.py`
- `src/compiler/host_code_modules.py`
- `src/compiler/machine_dialect_ssa.py`
- `src/compiler/machine_symbolic_effects.py`
- `src/compiler/machine_ssa_execution.py`
- `src/compiler/native_code_retention.py`
- `src/compiler/freestanding_amd64.py`
- `src/compiler/machine_execution.py`
- `src/compiler/amd64_machine_semantics.py`
- `src/compiler/machine_program_graph.py`
- `src/compiler/machine_process_graph.py`
- `src/compiler/semantic_translation.py`
- `src/compiler/ssa_builder.py`
- `src/compiler/x86_tensor_read_head.py`
- `tests/test_host_code_modules.py`
- `tests/test_machine_dialect_ssa.py`
- `tests/test_machine_semantic_family_lifting.py`
- `tests/test_freestanding_amd64.py`
- `tests/test_reversible_machine_execution.py`
- `tests/test_machine_process_graph.py`
- `tests/test_semantic_translation.py`

The cache implementation digest in `host_code_modules.py` fingerprints these
six implementation modules:

1. `cpython_compile_ssa`
2. `machine_code_lifting`
3. `machine_reference_vocabulary`
4. `machine_dialect_ssa`
5. `machine_symbolic_effects`
6. `native_code_retention`

Those six were frozen after PID 1176 launched. Do not edit them while the
process is alive. Later edits during the run were confined to non-fingerprinted
executor, graph-adapter, semantic-contract, and test files.

---

## 4. Source compiler work completed earlier today

This section records the source-side work because the PE route exists to fill
gaps in the same full-program compiler, not to replace it.

### 4.1 Parameter memory contracts

`src/transmogrifier/function_table.py` gained typed, frozen parameter metadata:

- `ParameterTransfer`: `value`, `alias`, `copy`
- `ParameterAccess`: `in`, `out`, `inout`
- `ParameterStorage`: `scalar`, `span`, `record`, `table`
- `ParameterScope`: `local`, `caller`, `retained`
- `ParameterContract`: ordered name plus the flags above

`FunctionEntry` carries ordered `parameter_contracts`; declaration remains
backward compatible; redeclarations preserve contracts unless explicitly
replaced; checkpoint restoration defaults older entries to an empty tuple.

This is metadata about function ABI/memory transfer. It is not an OOP runtime
object, handler, Python fallback, or callable wrapper.

### 4.2 Sequence/table SSA foundation

The repository now has explicit sequence/table descriptors and CFG lowering:

- list semantics: duplicates allowed;
- set/dict/table semantics: uniqueness determined by key columns;
- fixed caller-provided data arena, length cell, capacity, and status cell;
- insert status: duplicate, inserted, or full;
- append/add share insertion mechanics but destination policy is authoritative;
- extend iterates its source and inherits destination uniqueness;
- optional live flags preserve tombstoned rows;
- dynamic growth remains an explicit typed unsupported boundary;
- no runtime collection handler or Python adapter is used.

The lowering emits ordinary comparisons, branches, phi nodes, GEP, load, and
store operations. Sequence helpers and tables survive module aggregation and
filtered Fortran emission.

Generator-backed `extend` is handled for a single retained generator loop by
connecting yielded values directly to the destination mutation and retaining
generator `if` predicates as compiled guards. Nested/multi-generator cases
remain explicit until nested loop ownership is represented.

### 4.3 Source pursuit and receiver ownership

`graph_express2` source pursuit was strengthened so that:

- resolved receiver/actual source identities propagate to callee formal names;
- class-field constructor provenance is recovered without constructing runtime
  objects;
- lexical list/set/dict/tuple constructors establish storage kinds;
- grounded receiver resolution wins over tensor spelling heuristics;
- attribute spelling alone never authoritatively classifies a tensor call;
- class admission installs `self`/`cls` owner bindings for methods;
- resolved external readable Python source may be pursued through the actual
  AOT include filter;
- `NetworkX DiGraph.remove_node` can be source-linked instead of left opaque;
- direct one-clause generator consumption is normalized into retained loop and
  filter structure without a generator runtime object.

The fixed point now uses a definition/call worklist with binding revisions,
instead of rescanning the complete module after every new binding. Source AST
templates are cached by exact code/source identity and deep-cloned on use.

### 4.4 Constant sentinel correction

The hierarchy planner no longer writes a Python `object()` sentinel into a
runtime-indexed constant table. Availability is represented by a separate
integer status array and value table, so literal `None` remains distinct from
“unavailable.”

### 4.5 Aggregate truth and loop predicates

`while pending:` previously aliased its condition to the materialized aggregate
value. The loop composer treated that value as a generic predicate, reserved a
latch value ID, and emitted no producer. It now emits an explicit
`sequence_nonempty` query, loads the sequence length cell, compares it with zero,
and produces both the initial and latch predicates.

This was independent of opaque mutations. Mutations remain semantically
important because append/pop must change the shared length used by termination.

### 4.6 Loop-carried store correction

Resident sequence/table mutations are not scalar values to be carried through
phi aliases. The sequence work removed synthetic `LoopResult` behavior for
in-place add/extend and stopped `IndexedStore` nodes from being exported as
value-carried aliases. Remaining true scalar/call/phi loop values require real
scheduled producers; they must not be deleted merely to silence producer checks.

### 4.7 Tensor provenance and first-class tensor SSA

Tensor provenance now propagates through shape-preserving arithmetic so a chain
such as `x.exp() + 1`, followed by `.log().sum()`, keeps every tensor operation
grounded. Tensor lowering uses first-class `SSATensorTable` and
`SSATensorDescriptor` records rather than backend/runtime handles.

The full source-side tensor architecture remains documented in
`HANDOFF_tensor_op_ssa_modules.md`.

### 4.8 Direct static function token cleanup

When a direct source-linked call had already captured a FunctionTable callee,
its disconnected `StaticReference(function_subgraph)` token could remain in the
render graph and be scheduled as an operator. Only dead, unconsumed, non-root
function-subgraph references are now removed. Returned/passed function values
remain intact.

---

## 5. `re._compile` source-side status

`re._compile` has a separate source-ingestion route from native PE extraction.
Existing evidence from this session:

- the source closure emitted 108 SSA functions;
- 68 exports were present;
- repository-SSA/Fortran source emission completed with zero SSA shortfalls;
- emitted Fortran was approximately 96 KiB;
- native Fortran compilation then reported 13 errors;
- those errors belong to the span/rank ABI family: invalid procedure arguments,
  scalar versus rank-1 mismatches, and invalid use of assumed-size arrays.

Therefore `re._compile` is structurally close on the source route but not done.
Success requires native link/run behavior to match, not merely emission.

Do not confuse this with native CPython `compile()` extraction. The current
long run begins at native `compile()` and recursively pursues its PE imports.

---

## 6. PE/AMD64 vocabulary and exact machine semantics

### 6.1 Authoritative vocabulary coverage

The x86 bidirectional catalogue currently covers 308/308 authoritative forms:

- reference decode;
- tensor verification;
- exact write-head re-encoding;
- repository-SSA legalization coverage in the bounded catalogue.

The latest added forms were:

- `ROR_RM64_CL` (token 306)
- `LOCK_INC_RM32` (token 307)

Their semantic details are exact:

- dynamic CL rotate count masking;
- zero-count preservation;
- defined CF/OF behavior;
- locked 32-bit increment as sequentially consistent atomic read-modify-write;
- CF preserved across INC;
- other arithmetic flags computed correctly.

An arithmetic-flags helper was also corrected so preserved-CF paths do not
compute a meaningless carry merely to overwrite it later.

### 6.2 Leaf functions without `.pdata`

Some exported UCRT leaf/thunk functions have no runtime-function entry.
`_code_owner_for_entry` now synthesizes conservative executable bounds using
the next export, runtime-function boundary, or executable-section end. This
made the UCRT `free` export at RVA `0xf020` liftable.

### 6.3 UCRT focused closure result

After the latest opcode and leaf-owner fixes, the focused UCRT closure produced:

- 97 lifted functions;
- 34 retained blocker occurrences;
- 33 `external-machine-module` occurrences;
- 1 `indirect-jump` occurrence;
- zero decode or missing-semantic blockers;
- zero machine-state blockers under the current classification;
- `machine_state_complete=True` for the local machine bodies;
- `dependency_context_complete=False`;
- `repository_ssa_complete=False`.

The indirect jump is an exact two-byte function at RVA `0x4af20`:

```text
ff e0    jmp rax
```

It is a genuine dynamic dispatch thunk reached through an internal slot/call,
not a missed jump table. Machine-state SSA retains it as `IndirectBr` with the
target and complete register/vector/MXCSR/flags/memory state. Its target cannot
be statically invented.

---

## 7. Completeness boundary

The code now distinguishes several claims that were previously too easy to
conflate:

### 7.1 Machine body completeness

Every decoded instruction in the locally owned body has a complete machine
semantic/state transition. This does not imply every external dependency is
present and does not imply ordinary repository SSA exists.

### 7.2 Dependency-context completeness

Every required linked module/export/callsite has a concrete recursively
extracted target or an explicitly retained dynamic boundary.

### 7.3 Repository-SSA completeness

All functions are ordinary repository SSA, every dependency is linked, and no
machine dialect remains. Direct repository backends must reject retained
machine dialect occurrences.

Current implementation fields include:

- `NativeCompileSSAResult.machine_state_blockers`
- `NativeCompileSSAResult.machine_state_complete`
- `CachedHostCodeLibrary.machine_bodies_complete`
- `CachedHostCodeLibrary.dependency_context_complete`
- `CachedHostCodeLibrary.machine_state_complete`
- `CachedHostCodeLibrary.repository_ssa_complete`

The machine-state implementation variant is advertised only when there is
actual machine-dialect content and its ecosystem is complete.

### Queued classification correction

`unresolved-call-target` is currently excluded from `machine_state_blockers` as
if it were purely contextual. That is probably too weak: a direct target with no
lifted body is missing machine behavior, not merely missing deployment context.
Revisit this after PID 1176 exits. Do not edit `cpython_compile_ssa.py` while the
active run is serializing results produced under its existing fingerprint.

---

## 8. Recursive cached host-code extraction

`src/compiler/host_code_modules.py` supplies the recursive closure:

1. resolve a Python/native value to `HostCodeIdentity`;
2. lift the root PE export/RVA;
3. inspect exact `pe-import` links in its machine-indirect table;
4. resolve API-set hosts and export forwarders;
5. extract each target through the disk cache;
6. enqueue newly discovered units without a depth or module-count cutoff;
7. retain missing module/export/forwarder edges exactly;
8. materialize the linked library as one collision-free module.

Cache records use atomic temporary-file replacement, highest pickle protocol,
and a custom pickler for immutable mapping proxies.

### Previous v2 failure and v3 correction

The previous long extraction reached UCRT and failed because a leaf export had
no `.pdata` owner. The synthesized leaf-owner correction described above is in
the v3 implementation digest. PID 1176 is the authoritative v3 run.

### Cache state observed during v3

At roughly 30 minutes, `.turing-cache/host-ssa` contained about 50 files and
3.685 GiB. Several new units were being written every minute. The process held
roughly twice the cache size or more in private memory, eventually reaching
about 19.7 GiB.

This led to a concrete performance diagnosis.

---

## 9. Principal extraction performance defect

The dominant cost is not the live reversible executor. The recursive extractor
does not run `compile()` instruction-by-instruction. It repeatedly parses,
decodes, lifts, stores, and finally deep-copies overlapping machine closures.

Two structural duplications were identified:

1. Cache identity is export-root-specific. Two exports in one DLL receive
   different cache keys and may independently store mostly overlapping reachable
   function closures.
2. `materialize_host_code_library` deep-copies each unit's complete module and
   namespaces its functions as if overlapping RVA-owned functions were distinct.

That causes repeated translation, huge cache files, high serialization cost,
large RAM usage, and loss of true shared function identity.

### Required redesign after the active run

Use module/function ownership rather than export-closure duplication:

- parse each immutable PE image once per content digest;
- own each function by `(PE content digest, entry/owner RVA)`;
- decode/lift each code owner once;
- cache function bodies independently or cache one module graph with an entry
  index;
- represent exports as entry references into that shared module graph;
- preserve callsite occurrence identity separately from function ownership;
- materialize one function per module/RVA, not one copy per export-root closure;
- retain raw blocker ledgers on owning functions and effective blocker ledgers
  on linked callsite occurrences.

Do not solve this by weakening recursion or truncating closure size. The work
must become shared and incremental, not incomplete.

---

## 10. Reversible executor efficiency pass

Although the executor is not the main v3 extraction cost, it matters for live
machine execution, stream interposition, and the later demo/runtime path.

### 10.1 Existing structure

`MachineExecutionOrchestrator` already:

- pre-binds translated operations;
- caches straight-line translated blocks by guest RIP;
- keeps an exact instruction journal through `ReversibleMachineExecutor`;
- versions executable pages when self-modifying writes occur;
- supports dynamic decode for executable guest memory;
- retains every per-instruction architectural transition.

### 10.2 Costs removed today

Before the change, a cached block still:

1. reread and compared instruction bytes before every instruction; and
2. called `_memory_changed_ranges` after every transition, comparing the full
   persistent memory-page map merely to detect executable-page writes.

The efficiency pass added exact copy-on-write provenance to `PagedByteMemory`:

- `_parent_pages_identity`
- `_changed_pages`

`map_bytes` and `unmap` record the immediately changed page IDs. If provenance
does not match (for example after serialization or reconstruction), the
executor falls back to the exact structural diff. Mapping a previously absent
all-zero page is correctly treated as a state change even though its bytes
match the zero template.

`MachineExecutionOrchestrator` now:

- reconciles executable writes by changed page IDs in O(number of writes);
- validates a translated straight-line block as a unit;
- avoids rereading each instruction on the normal path;
- stops the block immediately if executable-page reconciliation increments the
  translation generation.

Thus self-modifying-code correctness is preserved. A write to an executable
page clears the translated-block cache, increments generation, and prevents a
possibly modified successor from executing under stale translation.

### 10.3 Verification

Focused executor and AMD64 semantic suites passed 49 tests after this change,
including:

- exact rewind;
- translated block caching;
- guest self-modification;
- external executable writes;
- executable-page version restoration;
- multicore shared memory;
- mapped-zero memory behavior.

Later combined focused runs reached 55 passing tests with the semantic and
machine-ProcessGraph adapters included.

---

## 11. Freestanding portable AMD64 personality

`src/compiler/freestanding_amd64.py` defines a portable compatibility target:

- ABI: `turing-freestanding-amd64-v1`
- capability library: `turing-capability-v1.dll`
- capabilities: exit, input polling, output publication, monotonic clock, and
  memory commit
- environment context: `TURING_FREESTANDING_AMD64_LOADER`

Validation requires:

- PE image;
- AMD64 machine;
- OS personality `turing`;
- exact ABI;
- imports only from the fixed capability library;
- no unknown or delayed capability imports;
- authoritative instruction census;
- complete executable-byte coverage.

A Windows-linked PE does not become portable by changing labels. Windows DLL
imports, syscalls, TEB/PEB expectations, loader behavior, and API semantics are
part of the program. Portability requires translating those dependencies to
Turing capabilities, recursively including them, or retaining a Windows
environment. The validator prevents relabel-only conversion.

Five focused freestanding tests passed.

---

## 12. Bidirectional x86 head

The x86 head is a true bidirectional instruction table rather than unrelated
encode/decode tables. Recent work added:

- `X86ReadHeadProfile` namespacing/remapping;
- `X86ReadHeadCodeSetBank`;
- simultaneous code sets;
- procedural token/name renaming;
- collision prevention across code sets;
- exact reference/tensor/write-head catalogue coverage.

This is the correct substrate for code switching and future live stream
interposition. It is not an excuse to label retained machine IR as ordinary
repository SSA.

---

## 13. Representation-unity architecture

The user identified the central requirement: repository dual IR, ProcessGraph,
MachineProgramGraph, machine-state SSA, and repository SSA should form a tight
translation bundle.

The design adopted today is **shared semantic family plus preserved facets**.
It does not pretend that every representation has identical state.

For example:

- tensor/repository `Add` and machine `INTEGER_ADD` share family
  `arithmetic.add`;
- machine add retains width, flags, possible memory destination, ordering, and
  exact encoded instruction facets;
- tensor add retains shape, dtype, layout, device, and tensor storage facets;
- translation is exact only if every required source facet survives;
- otherwise it returns a typed residual naming missing facets.

This prevents two opposite failures:

1. five disconnected pipelines with ad hoc string matching; and
2. a falsely universal operation table that erases architecture/tensor/control
   semantics.

### 13.1 Shared semantic contract

New file: `src/compiler/semantic_translation.py`

It defines:

- `SemanticRepresentation`
  - dual IR
  - ProcessGraph
  - MachineProgramGraph
  - machine-state SSA
  - repository SSA
- `SemanticOperationIdentity`
  - family
  - representation
  - spelling
  - representation-specific facets
- `SemanticTranslationProof`
- `SemanticTranslationResidual`
- `semantic_family`
- `semantic_identity`
- `prove_exact_translation`

The initial family table covers common arithmetic, bitwise, comparison,
memory, control, selection, tensor shape/reduction, and transcendental
operations plus major machine semantic tokens.

This is an invariant/registry, not a new compiler pipeline.

### 13.2 ProcessGraph to repository-SSA view

`ssa_builder.process_graph_to_ssa_instrs` now preserves an existing semantic
family and facets when producing a repository-SSA view. It does not recompute a
machine family from the ProcessGraph operation spelling and accidentally turn
`INTEGER_ADD` into an unrelated generic family.

This helper is not the full compiler entrypoint. Retained control must still use
the graph-to-section renderer.

### 13.3 MachineProgramGraph semantic nodes

Machine instruction components now carry semantic identity attributes in
addition to their authoritative token IDs, exact bytes, instruction token,
prefixes, operands, and control relationships.

Do not remove the token graph. Semantic identity supplements the exact machine
schema; it does not replace it.

---

## 14. Direct MachineProgramGraph to ProcessGraph schema adapter

The user proposed that if machine graph/SSA operators can be described with a
schema, ProcessGraph should accept them directly. The existing per-instance
`ProcessGraph.role_schemas` mechanism is suitable.

New file: `src/compiler/machine_process_graph.py`

It provides:

- `machine_process_operation`
- `machine_process_role_schemas`
- `install_machine_role_schemas`
- `machine_program_to_process_graph`

### 14.1 What the adapter does

It imports an already decoded `MachineProgramGraph` directly. It does not:

- decode again;
- execute a concrete path;
- pass through FusedProgram;
- numerically project machine state;
- call a runtime/external handler.

Each decoded function becomes a separate complete-machine-state chain:

```text
machine.state.input
    -> machine.integer_add
    -> machine.return
```

Each instruction node carries:

- exact machine address;
- exact encoded bytes;
- instruction token;
- machine semantic token;
- ordered structured operand records;
- statically derived read resources;
- statically derived write resources;
- effect domains;
- may-trap flag;
- conditional flag;
- function RVA;
- instruction index;
- semantic family and facets;
- source representation identity.

Instruction input/output roles are both `machine_state`. This means ProcessGraph
consumers see the instruction as a complete state transformation rather than a
numerical expression with guessed scalar operands.

### 14.2 Control edges and loops

Relative branch/call targets are represented as `control-target` edges in
addition to the sequential machine-state chain. They are structural edges, not
numeric operands.

A `jne -2` self-loop remains a graph self-edge with role `control-target`. It
is not recursively unrolled, deleted, or fed into the arithmetic operand list.
External target addresses remain explicit in the node's control record.

### 14.3 Verification

Focused tests verify:

- direct schema installation on a ProcessGraph instance;
- exact `ADD` bytes and two ordered operands;
- state input/output roles;
- machine read/write resources;
- semantic family preservation;
- self-loop control retention;
- generic ProcessGraph-to-SSA view preservation of machine family/facets.

Combined semantic/adapter/executor testing reached 55 passing tests.

---

## 15. FusedProgram correction

An initial attempt briefly wired semantic identity into
`process_graph_fusion.fused_program_to_process_graph` and used a FusedProgram
fixture to test the unity layer. The user correctly objected: FusedProgram is in
design flux and is a sectional numeric reduction artifact, while the intended
path is full graph-to-section rendering without numerical fusion.

That integration and fixture were removed immediately.

Current state:

- no semantic-unity hook remains in `process_graph_fusion.py`;
- no FusedProgram-based test remains in `test_semantic_translation.py`;
- the semantic registry lists dual IR as a possible representation but does
  not make FusedProgram the authority or required route;
- direct machine ingestion targets full ProcessGraph;
- the next integration targets `lower_control_sections_to_ssa` and the
  hierarchy/section plan.

One broad `test_process_graph_fusion.py` run exposed an unrelated existing
dynamic scalar feed-ID assertion mismatch (`id(control)` versus internal IDs
`{15, 21}`). It was not caused by the semantic changes and was not modified or
waived.

---

## 16. Graph-to-section renderer boundary

The relevant full-program entry is:

```python
lower_control_sections_to_ssa(...)
```

in `src/compiler/precompile_to_ssa.py`.

Its contract is already correct in principle:

- whole-object emission;
- control plus zero or more planner regions;
- no FusedProgram construction;
- no required numerical projection;
- hierarchy-plan regions lower through `plan_region_to_ssa_instrs`;
- control lowers directly;
- sequence/table/tensor/reference metadata remains explicit.

### Next missing adapter

Machine-aware ProcessGraph nodes need a hierarchy/section-plan representation
that carries complete machine-state operations into this renderer. The adapter
must:

1. identify each machine function and its entry state;
2. preserve basic blocks and control-target/backedges;
3. express each instruction as a complete state transformation;
4. retain exact semantic facets and bytes;
5. emit machine-state SSA while machine-specific effects remain;
6. legalize to ordinary repository SSA only when every effect has an exact
   repository form;
7. return typed residuals otherwise;
8. never route the machine region through numeric fusion.

Do not use `process_graph_to_ssa_instrs` as the final control compiler. It is
only a useful straight-line semantic-view test. The section renderer owns loops,
branches, and whole-program control.

---

## 17. Machine-state SSA to repository SSA boundary

`src/compiler/machine_dialect_ssa.py` currently produces explicit
`turing.machine-state-ssa.amd64.v1` functions:

- one complete machine-state argument;
- state phi nodes per block;
- one `machine.<semantic>` instruction per decoded instruction;
- exact bytes/operands/read/write/effect attributes;
- explicit condition and branch nodes;
- retained external and indirect transfers;
- a complete machine-state result at return/termination.

Repository emitters reject these retained operations until legalized.

### Work deferred until PID 1176 exits

The active extraction fingerprints `machine_dialect_ssa.py` and
`machine_symbolic_effects.py`. After it exits:

1. attach the shared semantic identity/facets to machine-state SSA instructions;
2. add an exact `MachineProgramGraph -> machine ProcessGraph -> machine-state
   SSA` equivalence test;
3. compare direct decoded-function machine SSA with section-rendered machine
   ProcessGraph SSA;
4. require identical instruction addresses, token sequence, encoded bytes,
   operands, state-resource effects, and CFG targets;
5. add proof-gated machine-to-repository legalization records;
6. mark conversions with preserved facets or typed residuals;
7. refine `unresolved-call-target` completeness classification.

---

## 18. MachineProgramGraph versus machine-state SSA

The `MachineProgramGraph` is the owning decompiler representation. It contains:

- PE image and sections;
- runtime functions and discovered leaves;
- exact instruction/operand components;
- containment, sequence, operand, control-target, and internal-call edges;
- unreachable and unclassified byte regions;
- vocabulary failures;
- byte-coverage statistics;
- token atlas/evolution graph identity.

Machine-state SSA is an executable dataflow projection of that graph. It
versions architectural resources and supplies backend/compiler structure.

Repository SSA is a further exact legalization target, not an alias for either.

The intended relationship is:

```text
PE bytes
  -> MachineProgramGraph (owning decoded graph and byte coverage)
      -> machine-aware ProcessGraph view (shared graph ecosystem)
          -> machine-state SSA (complete architectural state flow)
              -> repository SSA, only when exact legalization is proved
```

The reverse/bidirectional head connects encoded instruction forms to the same
machine token/semantic table. It should eventually permit exact re-emission and
code-set switching without creating a second semantic registry.

---

## 19. Why the current long run consumes so much memory

The evidence points to retained duplicate closure material, not an infinite
loop or executor emulation:

- CPU increases continuously;
- cache files continue appearing;
- stderr remains empty;
- process stays responsive;
- cache units are export-root closure pickles;
- overlapping function bodies are repeated across roots;
- final materialization deep-copies every unit;
- prior runs also entered a large-memory serialization phase.

The correct response is module/RVA deduplication after this run, not imposing a
new recursion cap or dropping dependency pursuit.

---

## 20. Exact focused test evidence from today

The following bounded evidence was recorded during this continuation:

- parameter contracts: 10 focused tests passed (with one unrelated static
  reference test deselected at that earlier point);
- sequence/table/control integration: up to 79 focused tests passed in the
  relevant suite during iterative development;
- source parent ingestion: 24 tests passed after worklist, class binding,
  NetworkX, and SymPy `_find_opts` corrections;
- deterministic FunctionTable/static-reference suite: 11 passed;
- tensor provenance chain: targeted regression plus three-test tensor selection
  passed;
- UCRT probe: 97 functions, only external module dependencies and one genuine
  indirect jump remained;
- freestanding AMD64: 5 tests passed;
- executor forwarding plus AMD64 semantics: 49 passed;
- semantic unity, direct machine ProcessGraph adapter, executor, and AMD64
  semantics combined: 55 passed;
- `py_compile` and `git diff --check` passed for the new bounded files, aside
  from expected LF/CRLF warnings.

No full test suite was run. Do not extrapolate bounded evidence into a claim
that the entire compiler is green.

---

## 21. Known unresolved items

### 21.1 Awaiting the live extraction result

The authoritative recursive `compile()` blocker ledger does not exist until PID
1176 exits and the report is read. Do not substitute the earlier UCRT probe or
old ledgers for the real run.

### 21.2 Cache/module ownership duplication

Redesign cache identity and materialization around module content plus function
RVA. Preserve callsite occurrences separately.

### 21.3 `unresolved-call-target` classification

Likely belongs to missing machine bodies rather than contextual-only blockers.

### 21.4 ProcessGraph section rendering for machine nodes

Direct schema ingestion exists; retained-control section rendering does not yet
consume the machine schema end to end.

### 21.5 Machine-state SSA semantic identity

Shared semantic identity is attached to MachineProgramGraph and ProcessGraph,
but direct machine-state SSA stamping was deferred to preserve the live run's
fingerprint.

### 21.6 Exact translation matrix completeness

The registry is an initial foundation, not a complete matrix. It needs:

- systematic coverage reports for each representation pair;
- inverse/round-trip laws where inverse translation exists;
- facet requirements per family and edge;
- typed residual census;
- no silent spelling-derived fallback for operations claiming exact support;
- integration with tensor/operator catalog identities;
- integration with hierarchy plan and section-renderer operations;
- integration with the bidirectional x86 token table.

### 21.7 `re._compile` ABI errors

Thirteen native Fortran rank/span ABI errors remain after otherwise complete
source-side emission.

### 21.8 Source-side residuals

The longer source/loop/tensor residual ledger remains in
`HANDOFF_tensor_op_ssa_modules.md`. Do not treat PE work as having resolved those
unrelated representation/ABI issues.

---

## 22. Recommended continuation sequence

### Phase A: finish observing PID 1176

1. Wait in roughly 60-second rounds.
2. Report only meaningful health/phase changes.
3. On exit, capture logs and report atomically.
4. Preserve duplicate occurrences.
5. Identify the exact terminal phase and failure, if any.

### Phase B: focused blocker repair

1. Group only for diagnosis; keep occurrence records distinct.
2. For missing opcode/semantic forms, add exact reference decode, tensor
   verification, write-head encoding, symbolic effect, executor semantics, and
   repository legalization where sound.
3. Keep indirect/dynamic targets as explicit machine-state control when they
   are genuinely dynamic.
4. Re-run the 308+ vocabulary catalogue and focused host extraction tests.
5. Do not immediately launch another long compile.

### Phase C: module/RVA shared cache

1. Introduce module-content identity.
2. Introduce function-owner identity.
3. Separate export entry references from owned function bodies.
4. Make recursion enqueue references but cache bodies once.
5. Link exact import callsites to shared target entries.
6. Materialize without deep-copying overlapping functions.
7. Benchmark cold and warm extraction on a bounded multi-export DLL.
8. Verify raw/effective blocker equivalence against the old design.

### Phase D: representation matrix

1. Extend semantic family/facet definitions from authoritative existing tables.
2. Add machine ProcessGraph hierarchy/section-plan adapter.
3. Stamp direct machine-state SSA after the live run.
4. Add exact edge proofs:
   - MachineProgramGraph -> machine ProcessGraph
   - machine ProcessGraph -> machine-state SSA
   - machine-state SSA -> repository SSA
   - full ProcessGraph -> section-rendered repository SSA
5. Add round-trip/equivalence audits over instruction address, bytes, operands,
   resource effects, CFG, and function identity.
6. Treat FusedProgram only as an optional numeric section edge.

### Phase E: taxing later-stage compile

Only after first-stage recursive extraction is correct and the user still wants
it, compile a genuinely taxing hierarchy/planning entry. Compare compiled
ingestion-stage timing with Python only when behavior is correct and both paths
perform the same work.

---

## 23. Practical file map

### Native extraction and caching

- `scripts/extract_compile_host_library.py`
- `src/compiler/host_code_modules.py`
- `src/compiler/cpython_compile_ssa.py`
- `src/compiler/native_code_retention.py`

### PE parsing, machine graph, vocabulary, and lifting

- `src/compiler/binary_ingestion.py`
- `src/compiler/machine_program_graph.py`
- `src/compiler/machine_reference_vocabulary.py`
- `src/compiler/machine_code_lifting.py`
- `src/compiler/x86_tensor_read_head.py`

### Machine execution and state semantics

- `src/compiler/machine_execution.py`
- `src/compiler/amd64_machine_semantics.py`
- `src/compiler/machine_symbolic_effects.py`
- `src/compiler/machine_ssa_execution.py`
- `src/compiler/machine_instruction_control.py`

### Machine/repository SSA boundary

- `src/compiler/machine_dialect_ssa.py`
- `src/compiler/semantic_translation.py`
- `src/compiler/machine_process_graph.py`
- `src/compiler/ssa_builder.py`

### Full graph and section compiler

- `src/transmogrifier/graph/graph_express2.py`
- `src/compiler/hierarchical_plan.py`
- `src/compiler/control_source.py`
- `src/compiler/loop_composer.py`
- `src/compiler/precompile_to_ssa.py`
- `src/compiler/fortran_c_shell.py`

### Tensor and sequence repository SSA

- `src/transmogrifier/ssa.py`
- `src/transmogrifier/tensor_ssa_reference.py`
- `src/compiler/tensor_ssa_lowering.py`
- `src/compiler/ir_sequence_tables.py`
- `src/transmogrifier/function_table.py`

---

## 24. Restart checklist

Before changing code in a new session:

1. Read this file.
2. Read the opening/current-status sections of
   `HANDOFF_tensor_op_ssa_modules.md`.
3. Inspect whether PID 1176 still exists.
4. Inspect v3 stdout/stderr and report timestamps.
5. Inspect `git status --short`; assume unrelated work is user-owned.
6. Do not run recursive extraction as an orientation probe.
7. Do not edit the six cache-fingerprinted modules while PID 1176 is alive.
8. Run only focused tests for the files being changed.
9. Keep raw and effective blocker concepts separate.
10. Keep MachineProgramGraph, machine-state SSA, and repository SSA names
    explicit in code and user-facing reports.
11. Do not route full graph compilation through FusedProgram.
12. Never claim completeness from emission alone; verify link and execution at
    the scope required by the claim.

---

## 25. Bottom line

At this handoff, the native decoder/lifter has complete bounded vocabulary
coverage for all 308 authoritative forms, a recursively cached `compile()`
extraction is actively progressing through its dependency ecosystem, and the
machine/repository boundary is materially clearer than it was at the start of
the day.

The most important new architectural result is not another special-case
pipeline. It is the beginning of a shared semantic contract and a direct
machine-schema ProcessGraph view:

```text
MachineProgramGraph
    -> full machine-aware ProcessGraph
        -> graph-to-section machine-state SSA
            -> ordinary repository SSA only under exact facet-preservation proof
```

The most important performance result is equally structural: the long compile
is paying for export-root closure duplication and deep-copy materialization,
not merely slow instruction emulation. The next performance pass must share PE
module/function ownership by content and RVA while retaining unlimited
dependency pursuit.

The active run must finish before its exact blocker ledger, success state, or
remaining vocabulary obligations can be stated honestly.

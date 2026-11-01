# 2026-10-31 Source-derived thread-region dual IR (MEASURED, Linux cloud)

The configured workspace clone is `/workspace/Turing`, origin
`https://github.com/sangderenard/Turing.git`. After the user's push it was
fast-forwarded from `65755b6a` to `3eeab1bf`. This is work on Turing's own
repository history; the Windows canonical path is not mounted here. Read
AGENTS.md, its incoming amendment, and the then-newest hazard entry
`20261031T142023Z_main.md` before running the compiler. Historical results
were remeasured below, not used as a diagnosis.

## Baseline and environment

The default Python lacked pytest/tqdm. The installed dependency tree is
`/workspace/turing-deps`. Python's default CFFI compiler `clang` is absent.
Using `CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types`
allows the incumbent CTensor CFFI import to compile on Linux (its declared
long-long pointers differ from Linux's int64_t typedef). No compiler source
or mutation policy was changed to address this environment setup. All
reported successful source checks used this prefix plus
`PYTHONPATH=/workspace/turing-deps`.

Before editing, the two requested source modules measured **5 passed,
5 xfailed**. The mapping audit refused with the current-source error
`region value 7 (Add in function 'root') has no settled dtype ...`.
The first unconfigured attempts instead failed on missing Python packages
and clang; those do not establish any compiler-contract result.

## Implemented ingestion facts

Lexical reduction posts the enclosing environment while its exact binding
cell, definition occurrence and function address are known. Canonical
capture facts derive directly from both parent and worker binding cells.
This first slice admits single-binding environments; unknown cell/value
flow is a recorded region gap, never a guessed alias to an initializer.
The same proof identifies a captured `threading.Barrier.wait`. An alias of
the barrier retains identity; captured and ordinary `Other.wait()` do not
become barriers.

`ingest_thread_regions` runs immediately after lexical topology reduction,
before structural specialization. Its actual control and dependency
ProcessGraphs retain activation-qualified uses of the worker definition,
parent context and boundary edges. Every synthesized member, constraint,
synchronization identity and schedule has concordance provenance. Parent
control order is read from exact AST ownership/evaluation order, not line
ranges. Conditional/loop-dependent activation sets are not admitted by
this straight-line slice. Graph scheduling refuses unresolved captures.

The final dispatcher installer had already constructed start/join
DispatchBlocks, but omitted posting its replacement tree. That writer now
publishes the actual control rewrite and block placements.

For the two-worker SOURCE, the measured function refs remain root **0**,
worker **1**. Root allocation **4** reaches worker input **0** through a
concorded capture. Creates **8/9**, starts **10/11**, joins **12/13** and
post-join read **6** remain distinct. Worker read/add/store **3/5/6** become
region occurrences **17/19/20** for activation 8 and **24/26/27** for
activation 9. The one captured allocation is region node **4**; capture
edges reach boundary inputs **14** and **21**. Both region graphs have
**28 nodes**, with **67 dependency edges** and **36 control edges**.

The same region was scheduled with both existing registered ASAP/ALAP
providers. All edges increase levels, and neither topology changes nor
cross-worker reachability appear. For example, the parent post-join read
has dependency level **10** in both schedules; allocation 4 moves from
ASAP **1** to ALAP **3**. Worker A's read/add/store are **5/6/7** and B's
are **6/7/8** in both measured dependency arrangements. These levels are
chosen arrangements, not authored comparative guarantees.

For BARRIER_SOURCE, root constructor **4**, root wait **6**, worker wait
**2**, and activation create/start/join **10/11/12** remain exact source
nodes. The region has **17 nodes**, **24 dependency edges**, **13 control
edges**. Its parent wait at region **6** and worker wait at **15** each have
control/memory projections from one `SOURCE_SYNC_OPERATOR` cell. Both
arrivals constrain completion node **16**. The execution scope is the exact
barrier's participant set; the party-count operand is the authored 2.
Memory scope and semantics are explicitly unresolved. No memory fence
edges are invented, and scheduling that unresolved memory policy refuses.

## Verification and remaining boundary

The final focused/regression command is:

```sh
CC=gcc CXX=g++ CFLAGS=-Wno-error=incompatible-pointer-types PYTHONPATH=/workspace/turing-deps python -m pytest tests/test_thread_dual_ir_sync_contract.py tests/test_python_two_threads_shared_location.py tests/test_control_source.py tests/test_string_source_callee_identity.py tests/test_ssa_shared_memory_threads.py -q --tb=line --show-capture=no
```

It measured **41 passed, 1 skipped, 1 xfailed**. The two requested modules
alone measured **12 passed, 1 xfailed**. All four former strict dual-IR
xfails were removed after their assertions passed. The paired-projection
assertion now inspects the real newly retained region halves rather than
requiring the source function graph itself to become a memory barrier;
it additionally checks shared operator identity, exact scope, unresolved
memory operands and the control rendezvous. All new tests ingest authored
source through `lower_ast_source_to_ssa`; none constructs a substitute
graph or executes a worker.

`test_python_threading_special_cases.py` gave **7 passed, 3 failed** on both
this worktree and an untouched `3eeab1bf` worktree. The failures are
`test_condition_scope_keeps_numeric_regions_inside_and_after_cleanup`
(unsettled Add 9), `test_resource_wait_retains_loop_scope_and_unique_ssa_definitions`
(untyped root n), and `test_literal_seeded_counter_executes_native_iterations`
(untyped counter n). These were reproduced, not attributed from an old log.

The page-classification test still reports the same **13** incumbent
unclassified pages on untouched `3eeab1bf` and this worktree. All ten new
pages have lifecycle declarations. The source thread and barrier probes
each report **0 concordance findings, 0 unsourced facts** with the latch
closed; the audit now reads closure, region, projection and schedule
records. The post-change mapping audit repeats its **Add 7 dtype refusal**;
it is not a completed clean mapping audit. `py_compile` and
`git diff --check` pass. The full suite was not run.

The exact next implementation boundary is the existing structural-effect
and SSA planners consuming the retained region and its activation-qualified
worker effects. The ordinary SSA path still omits the worker and folds the
post-join read; that original strict xfail remains. The source region itself
retains the read and both stores. No mutation exemption, automatic lock,
Python executor, backend lane choice or native thread implementation was
added. General capture cell/value flow, conditional regions and reusable
barrier phases remain unresolved.

Running the remaining test with `--runxfail` reaches its intended assertion:
`worker function is absent from SSA`. A separate current-source inspection
measured only `root__root` in `module.functions`, with returned SSA value 3
defined by `Const`, `attributes={'value': 0}`. This is a consumer shortfall,
not a claim that retained graph effects already reach SSA.

The observed/proposed distinction and the remaining explicit fence/barrier
policy table are in
`docs/compiler_convergence/thread_dual_ir_ingestion_slice_2026-10-31.md`.
No barrier lowering or native execution is claimed.

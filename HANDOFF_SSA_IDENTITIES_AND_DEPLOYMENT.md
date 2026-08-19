# Handover: the namespace fix landed, and the optimizer that was never there

Continues `HANDOFF_NAMESPACE_ROOT_AND_INSTRUMENT_ARSENAL.md`. Written
2026-08-19. Two things happened this session: the one named defect that
document handed forward is **fixed and the flagship is green**, and chasing the
fluid demo's speed exposed something larger — the compiler applies no
arithmetic identities and mints no parallelism evidence for any loop with a
runtime extent.

Its corollary, `SSA_IDENTITY_CATALOGUE.md`, is the itemised analysis of which
identities exist, which are missing, and what each is worth. This document is
the state and the plan.

---

## 1. State of the tree

**Branch `nogodsnomasters`, base `af00599`. Nothing is committed.** Ten files
changed or added:

| file | change |
|---|---|
| `src/compiler/ir_indexing.py` | address temporaries allocated from one module-wide watermark instead of per-function |
| `src/compiler/fortran_c_shell.py` | structural-output recovery refuses a `GetElementPtr` result as a recovered output |
| `examples/run_symbolic_fluid_native.py` | runner learns the container/table ABI (dotted paths, `length`/`keys`/`values`, extent-sized buffers) |
| `examples/symbolic_fluid_live.py` | `--viscosity`, `--tracer-diffusivity`, `--max-iters` flags |
| `tools/bench_native_step.py` | **new** — per-cell cost of the compiled step, native separated from marshalling |
| `TEST_BASELINE_AND_HAZARDS.md` | **new** — known-failing manifest, hazards, the cheap regression gate |
| `SSA_IDENTITY_CATALOGUE.md` | **new** — the corollary to this document |
| `tests/conftest.py` | skips `KNOWN_FAILING_AT_AF00599` by default; `--run-known-failing` opts back in |
| `AGENTS.md` | points at the baseline document |
| `tools/TRANSLATION_DEBUGGING.md`, `HANDOFF_NAMESPACE_ROOT_AND_INSTRUMENT_ARSENAL.md` | corrected records |

### Verified green

| check | result |
|---|---|
| `tests/test_symbolic_fluid_native_runtime.py` | **passed** — `[(0.2, False), (0.1, True), (0.1, True)]`, the finish line the prior handoff named |
| `tools/translation_scorecard.py` | 10/10 |
| `tests/test_precompile_to_ssa.py` | 34/34 |
| `examples/symbolic_fluid_live.py` | 90 frames at 24², zero rejections, mass error ~1e-15 |
| `examples/run_symbolic_fluid_native.py` | 60 frames at 8², mass conserved to 1.4e-14 |

The ~13 failures in `test_ast_indexing_aot.py` and 2 in
`test_index_set_scatter.py` are **pre-existing at af00599** — confirmed
identical in a clean worktree — and are now skipped by default. Read
`TEST_BASELINE_AND_HAZARDS.md` before running anything.

---

## 2. The namespace defect: fixed, and the prior diagnosis corrected

The prior handoff named the boundary, the two colliding integers (80/89 =
`tracer_diffusivity`, `viscosity`), and the defect class correctly. **It named
the wrong minting site**, and one probe killed it: dumping every `PlanClosure`
reaching `plan_region_to_ssa_instrs` shows **no PlanLine anywhere in the
program outputs 80 or 89.** region_2's lines stop at 79, region_1's at 88.

The real chain:

1. `lower_indexing_to_ssa_addressing` (`ir_indexing.py`) lowers `Indexed` to
   `GetElementPtr`+`Load` and allocated addresses from **each function's own**
   maximum id. region_2's max is 79, so its first address is 80; region_1's is
   88, so its first is 89. A planner region shares its caller's value space, so
   "one past this function's max" is not free — it is whatever the caller holds.
2. The structural-output recovery in `fortran_c_shell.py` scans callees for
   **any** instruction whose `res.id` equals a `desired_id` the caller lacks,
   appends it to `output_ids`, and materializes an unpack. It matched on the
   bare integer, so it bound a caller scalar to a pointer into the height array.

Two fixes, each verified independently sufficient by disabling the other: a
module-wide watermark (the collision becomes unrepresentable) and the recovery
guard (an address is a location in the callee's storage, never a value the
caller asked for). Both are kept.

**Nothing needed changing at `region_signatures`.** An agent following the prior
handoff's instruction would have edited a correct function.

*The lesson, now doctrine in `TRANSLATION_DEBUGGING.md`:* record which parts of
a diagnosis were **measured** and which were **inferred**. Everything measured
held. The single inferred part was the single wrong part, and it is exactly
where the next probe should have been aimed.

---

## 3. The performance finding

`tools/bench_native_step.py` reproduces all of this in about a minute.

**Cost is 1683 ns/cell, flat from 24² to 256², with marshalling at zero.**
Linear with no fixed overhead means the kernel is the entire cost; the
Python-side array feeding across the ABI is measurement noise.

**The per-cell kernel is 290 instructions**: 140 Mul, 91 Add, **24 Pow**, 16
Max, 10 Abs, 9 Const. There is no divide and no sqrt instruction — SymPy
canonicalises `a/b` as `b**-1` and `sqrt(x)` as `x**0.5`, so both arrive as
`Pow`. **Every exponent is a compile-time constant**: 10× `2`, 6× `-1`,
6× `0.5`, 2× `-2`.

`ssa_llvm_backend.py:37` lowers `Pow` unconditionally to `@llvm.pow.f64` with no
constant case — 25 call sites in the emitted IR. At `-O2` LLVM folds
`pow(x,2)`→`x*x` itself; `pow(x,-1)`→`fdiv` and `pow(x,0.5)`→`sqrt` require
`afn`/fast-math, which is absent, so roughly 14 real libm calls survive per
cell. 14 × ~70–100 ns is the measured 1.68 µs.

**The compiled artifact tells LLVM nothing.** The complete flag set is
`zig cc -shared -O2`. The emitted module has **no `target triple` and no
`target datalayout`** (so a generic baseline x86-64: SSE2, no FMA, no AVX),
**zero `noalias`** (so the buffers may overlap and the cell loop cannot
vectorize), and **zero fast-math flags**.

`src/compiler/llvm_optimizing_pipeline.py` already exists and its docstring
names exactly these three gaps — a pass pipeline, a named host target, aliasing
facts. It is imported only by `llvm_jit_backend`, `llvm_simd_deployment` and the
torture runner. **The AOT path never calls it.**

---

## 4. The deployment finding

`deployment_regions` on the fluid advance is `()`. Empty. The classifier, the
Deploy/Join binding pass, the worker pools and the per-backend profiles never
engage.

I first blamed the `not carried_bindings` conjunct of the `parallel_candidate`
gate, reasoning that the loop's five accumulators disqualified it. **A probe
said otherwise**, and the truth is structurally worse:

```
[PROBE] loop node=94 carried=0 backpressure=False state_effects=0
        iter_outputs=1 lane_nodes=0 -> parallel=False
```

That fired twice, for one loop. The advance's `row`/`column` loops never reach
the gate. `ControlDeploymentRegion` has **exactly one construction site in the
repository** — `loop_composer.py:955`, inside `evaporate_unrolled_loops` — and
its lanes are built by enumerating `iteration_values`, the *unrolled*
iterations.

**So parallelism evidence is reachable only through static unrolling.** A loop
over a runtime extent cannot produce a lane, by construction. The fluid loop is
deliberately runtime-extent (`range(height_count)`, a record field, for reasons
documented at `symbolic_fluid_dt.py:38`), so it is invisible to the deployment
layer no matter what it carries.

---

## 5. The plan

### P0 — Commit what is here

Ten files, one coherent change plus its documentation. Not committed because it
was not asked for.

### P1 — `ir_identities.py`: one backend-neutral identity pass

**LANDED 2026-08-19 (first inhabitant):** `ir_identities.reduce_constant_exponent_pow`,
called at the `IRModule` finalization point in `fortran_c_shell` — it must run
after region carving and value pruning, or rewrites orphan exponent constants
that recovered output ledgers still name (journey 3 caught this when the pass
sat in `precompile_to_ssa`). Exact set default: 1683→~900 ns/cell, scorecard
10/10 at 0.0e+00. `TURING_POW_INEXACT=1`: ~480 ns/cell, fluid `mass_err <=
1e-15` held — the measured delta decision 1 below asked for. The FMA question
(catalogue 2.5) is audited: contraction permission alone forms zero FMAs; the
blocker is P2's `noalias`. `TURING_FMA_CONTRACT=1` switch is in
`ssa_llvm_backend`, off by default, ready for P2.

**Why here and not per backend:** all seven `ssa_*` backends consume the same
`IRModule`. Each has its own `Pow` handler, so a per-backend fix costs seven
edits and seven chances to diverge; an SSA-level pass costs one and every
backend inherits it — including SPIR-V, WASM and WebGPU, where there is no
`-O2` behind you to clean up afterwards.

**The seam already exists and is empty.** Six `ir_*.py` modules, every docstring
"backend-neutral", all running over shared SSA before any backend — and every
one of them is a *lowering* (subscripts→addresses, strings→tokens,
containers→tables), not a simplification.

The contents, the reduction ordering, and the value of each item are in
`SSA_IDENTITY_CATALOGUE.md`. The short form of the ordering, which is a design
decision and not an implementation detail:

1. **Constant folding** — so later stages see literals.
2. **Algebraic identities** — `x*1`, `x+0`, `x*0`, `x-x`, `x/1`.
3. **Strength reduction** — constant-exponent `Pow`. Must follow folding, or an
   exponent computed from constants is missed.
4. **CSE** — after folding and strength reduction, address and reciprocal
   subexpressions coincide; before that they do not.
5. **Dead value elimination** — last, so it collects what the others orphaned.

**Gate:** the scorecard plus the reference evaluator. An identity pass must be
provably result-preserving across all 10 journeys, and the evaluator is the
independent executor that proves it.

### P2 — Make the artifact declare what it knows

Emit `target triple` and `target datalayout`; name the host CPU; attach
`noalias` where it is **true**; route the AOT path through
`llvm_optimizing_pipeline` instead of a bare `zig cc -O2`.

**Care required on `noalias`.** The aliased-IO design in this tree is
*intentional* — one slot per carried value, read-then-written in place. Blanket
`noalias` would be a lie and would miscompile. The pass must derive aliasing
from the carried-slot metadata the storage-formal ABI already records, and mark
only what it can prove distinct.

### P3 — Deployment: absorb, do not package

The recommendation is to **absorb**. `evaporate_unrolled_loops` should stop
being the gatekeeper and become one client of a dependence test that runs over
the SSA loop body and does not care whether anything was unrolled. Two pieces,
which must land together:

1. **A dependence test independent of unrolling**, able to mint a
   `ControlDeploymentRegion` for a runtime-extent loop.
2. **Associative-reduction recognition** — `acc = acc + x`, `acc = max(acc, x)`.
   The fluid loop carries five such accumulators and the existing gate vetoes on
   any carried binding, so fixing (1) alone leaves the loop rejected for a new
   reason.

Packaging or flagging the existing modules is the wrong shape: there is nothing
to gate, because the analysis they wait on has never existed for the loops that
matter.

### P4 — Eigen / C++ lane — **blocked, do not start**

The intended lane ordering is SSA identities → AVX → Eigen, with Eigen opened
through a C++ backend using nodus' Eigen translation. **I have not read that
translation**; a search for it landed in a neighbouring `.venv` rather than
nodus. Designing a lane ordering around an unread component is exactly the
inference-instead-of-measurement failure this session already paid for once.
**First action: read nodus' Eigen translation and record what it actually
offers.** Only then propose the ordering.

P1 is a prerequisite regardless: an Eigen or AVX lane that receives
`pow(x,-1)` instead of a reciprocal inherits the same problem in a new backend.

### P5 — The backward arc

Unchanged from the prior handoff and still the destination: gradient computed
inside the compiled loop. The optimizer loop already compiles exactly (scorecard
level 7); what remains is capturing the backward pass into the same authored
program shape.

---

## 6. Decisions needed

1. **Numerics policy for identities.** `x**0.5`→`sqrt` and `x**-2` are not
   bit-identical to `pow`. This repository verifies backends against each other
   numerically and the fluid asserts `mass_err <= 1e-15`. Options: exact
   identities only (which leaves 6 of the ~14 surviving libm calls); allow the
   inexact pair globally; or allow them behind a per-module flag. **My
   recommendation: exact-only first, measure, then present the inexact pair with
   its measured delta in mass error** — that turns a policy argument into a
   number. See the catalogue for which identities fall on which side.
2. **`noalias` derivation.** Which metadata is authoritative for "these two
   buffers are distinct"? The storage-formal ABI records carried slots; is that
   sufficient, or does a new fact need recording at lowering time?
3. **Scope of absorption in P3.** Does `evaporate_unrolled_loops` keep minting
   regions for unrolled loops during the transition (two producers, briefly), or
   hand that responsibility over in one step?

---

## 7. Working rules for whoever picks this up

* **Read `TEST_BASELINE_AND_HAZARDS.md` first.** The full suite does not finish.
  There is no per-test timeout. Known failures are skipped automatically, so a
  red run means you broke something.
* **A measurement already taken is final.** Do not re-run an expensive file to
  confirm or to prettify a result. This session wasted two full runs and a stash
  learning that the indexing failures were pre-existing, then started a third to
  collect names that were recoverable from a progress mask.
* **Never baseline with `git stash` or `git checkout -- <path>`.** Use
  `git worktree add`, and remove the worktree when done.
* **Separate measured from inferred in anything you write down.** Both of this
  session's wrong turns — the PlanLine namespace theory and the carried-bindings
  gate theory — were confident inferences sitting inside otherwise-measured
  chains.

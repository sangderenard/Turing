# Handoff — precision pipeline, and the region-granularity frontier

2026-08-23. Companion to [PRECISION_PIPELINE.md](PRECISION_PIPELINE.md),
which holds the architecture; this holds the state, the evidence, and what
to do next.

Claims marked MEASURED were run. The distinction is kept because several
confident diagnoses during this work were wrong and only running them
caught it — those are listed under "dead hypotheses" so nobody pays for
them twice.

## Superseding update — completed later 2026-08-23

The open vertical integration and region-granularity work described below is
now implemented:

- `apply_precision_pipeline(IRModule)` is wired at the completed repository
  SSA seam and persists its exact-identity and section-contract receipt.
- Precision limbs now cross calls in source order. Locally derived low limbs
  are passed intact; an ordinary scalar is promoted as `[x, +0.0, ...]`, never
  by duplicating `x`. Canonical `SSACallRecord` ABI bindings are extended.
- LLVM accepts the precision contract; WASM reports both missing obligations;
  Fortran currently refuses precision because it cannot declare section
  isolation. Earlier successful Fortran precision measurements below remain
  historical evidence, not permission to bypass that contract.
- The SSA lowerer is proved only for width two. Widths above two now hard-fail
  instead of repeating a tail limb into wider storage. Eager `Precision`
  remains N-limb; a genuine N-limb SSA expansion is still required.
- Power-of-two scaling and delayed fixed-width renormalisation no longer fire
  from pattern shape alone. They require exponent-range and retained-width
  proofs respectively.
- The Fortran host view now splices a single-callsite, straight-line,
  outputless source region into its caller without mutating repository SSA.
  MEASURED on the same 100,000-element width-one sine pack: **1.142
  ns/element**, maximum error `5.55e-17` against NumPy; the old ~435 ns region
  call is absent. Shader-oriented backends retain the region boundary.

## Where it stands

Declaring a width in ordinary Python now produces double-double arithmetic,
compiled, correctly rounded, on real backends.

MEASURED through LLVM, 2048 accuracy points against exact rational
evaluation, 1e5 elements for timing:

| core | ns/element | correctly rounded | median ulp |
|---|---|---|---|
| sin w1 | 2.3 | 95.4% | 0.183 |
| sin w2 | 57.6 | **100.0%** | 0.007 |
| sinh w1 | 2.8 | 81.4% | 0.257 |
| sinh w2 | 57.7 | **100.0%** | 0.010 |

`sinh` at width 1 returns the wrong double for one argument in five. At
width 2 it is exact. That is the case the pipeline exists for.

Cross-checked on two backends: the standalone `sin` core measures 1.052e-16
worst relative error on BOTH Fortran and LLVM, agreeing to the bit.
`two_product` matches an exact rational oracle 0/2000 on C, LLVM and
Fortran.

## The frontier: region granularity

This is the one thing to pick up.

MEASURED, same path, 1e5 elements, Fortran:

| body | ns/element |
|---|---|
| trivial `y[i] = x[i]` | 411 |
| sin, 18 flops | 435 |
| sin at two limbs, 115 flops | 445 |

411 ns is per-element structural overhead. The arithmetic contributes 24
and 34 ns. So double-double costs ~1.4x ordinary on Fortran, and the 190x
gap against LLVM is a fixed tax on EVERY Fortran kernel, precision or not.

The decision chain, read with `tools/probe_comprehension_regions.py`'s
instrumentation:

1. **Loop strategy** — `LoopStrategy.NATIVE_SOURCE`, "the backend can
   retain this iteration in compiled source". CORRECT: `UNROLL` needs a
   static trip count and `range(n)` has none. The loop is properly kept.
2. **Body carving** — `reduce_scheduled_shader_regions` fuses the 20 body
   nodes through 19 `vertical-fusion` rewrites into one
   `FlatComputeDispatch(kind='shader_region')`.
3. That region becomes its own SSA function, invoked once per element.
4. LLVM inlines it away. gfortran cannot, and pays 411 ns.

The defect is step 2/3, not the loop strategy. A `shader_region` is a
GPU-shaped unit — a body a dispatch runs once per work-item — applied
unchanged to host lanes, where "one dispatch per work-item" means "one
subroutine call per element".
`deployment_classification.classify_region_executions` does pick host-linear
against shader-compute, but it runs AFTER the carving, so it can only
classify a region already shaped for a shader.

This also explains LLVM's remaining 24x at width 2. Neither backend gets a
loop body it can vectorise, because the body is behind a call boundary; one
of them merely inlines through it. So the region fix and the vectorisation
work are the same change.

**Two candidate approaches**, and this is the open decision:
* Do not carve a vertically-fused body when the region classifies
  host-linear — fixes it at the source, changes the planner.
* Have the emitters inline a single-callsite region — local to each
  backend, leaves the planner alone, must be done per backend.

## Dead hypotheses — do not re-run these

* **Memory-slot allocas are NOT the bottleneck.** The LLVM compile is
  hardcoded `-O2` (`ssa_llvm_backend.py:4357`), so `mem2reg` was always
  promoting them, and `-O3` on top changes nothing (56.0 against 57.6 ns).
* **Not the compile flags, on either lane.** Fortran already uses
  `-O3 -march=native -flto -funroll-loops`.
* **Not the C binding.** Emitting internal regions without `bind(C)` moved
  429.3 to 428.5 ns. Reverted rather than kept as an unmotivated change.
* **Optimizers did not delete the dual.** gfortran returned the exact
  `two_sum` residual at default, `-O2` and `-ffast-math` alike, on inputs
  that genuinely lose bits. The hazard is real in principle; treat a
  destination that inlines harder as untested, not safe.
* **Deferred renormalisation won by shortening the dependency chain**, not
  by relieving stack pressure — 62% of the time for 24% of the
  instructions.

## Traps that cost real time here

* **Four harness faults produced four false defects**: calling a loop-body
  region with `n` where it wanted `i`; passing `p` and `e` swapped;
  assuming formal order matches authored order (it does not —
  `(x, y, n, c0...)` arrives as `(n, x, y, c0...)`); and a reference model
  that only handled plain Horner. ALWAYS run the ordinary-arithmetic
  control first: a plain multiply cannot be blamed on rounding.
* **The general rule for the lowering**: anything it expands must leave the
  replaced id DEFINED. Metadata, ordinary consumers and declared outputs
  all name values, not instructions. This bit three times — precision
  operations, loads from precision arrays, and function outputs.
* **Heredocs collapse `\n`** when writing generated Python. Use the edit
  tool. This corrupted a committed file once today.

## Tools

* `tools/benchmark_precision_cores.py` — core x width x load size, with the
  full ulp distribution and a bit-exact fraction. Setup is hoisted out of
  the timed region and formals are bound by DERIVED role; both were faults
  that produced wrong numbers before they were fixed.
* `tools/demo_bit_exact.py` — is bit-exact reachable? Answers yes, 100% of
  4096 points, beating libm's 97.68%. Pure numpy plus mpmath, NOT our
  compiled path: it is the target, not a result.
* `tools/probe_comprehension_regions.py` — patches loop discovery and
  scheduled-region reduction and prints what each decided. This is what
  located the frontier above.
* `tools/diagnose_translation.py` — staged decision tree for a WRONG
  ANSWER, routing a defect to the layer that owns it.

## Smaller things left open

* `cos` at width 2 measures poorly in the benchmark while the kernel
  returns the exactly-correct value when called directly — a remaining
  defect in the benchmark's batch path, not the compiler.
* `sech` at 126 ns is forty times its neighbours, unexplained.
* `tools/demo_noise_floor_all.py` produced no output at all (exit 0).
* Four identities still unfired: `exact_identity_element` needs
  use-rewriting, `sterbenz_cancellation` needs a proven range (catalogue
  section 5), and two need the kernel bank.
* Precision now crosses call boundaries at widths two through four. Derived
  limbs are passed in source order, ordinary scalars are extended with exact
  positive-zero limbs, and canonical call records are refreshed with the new
  bindings. The former duplicated-high-limb defect is pinned by regression.
* Widths three and four now lower into distinct SSA limbs for arithmetic,
  calls, and channel-strided array loads/stores. Exact-rational regressions
  show width four improves on width three for both multiplication and long
  division. Wider than four is still refused explicitly; compiled performance
  measurement of the wider paths remains open.

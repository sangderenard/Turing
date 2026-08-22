# Signal math: the trigonometry pack

**Date:** 2026-08-21. **Commits:** `b3ce6ca` … `b5f58af`.
**Code:** `src/common/tensors/signal_math.py` (cores, bake, dispatcher),
`src/common/tensors/signal_kernels.py` (kernel source authoring),
`src/common/tensors/abstraction_methods/trigonometry.py` (the switch).
**Tools:** `tools/signal_math_survey.py`, `tools/demo_signal_field.py`.

**For:** whoever finishes this. Section 6 is the work list; section 5 is the
list of things that cost hours and will cost them again if unread.

---

## 1. What this replaced

`AbstractTensor.sin` dispatched `_apply_operator("sin")`, which lowered to the
repository kernel `unary_double` with opcode 29, whose body is
`call double @sin(double)` — a libm extern. Every packed trigonometry
artifact in the tree, forward **and** reverse, borrowed its digits that way.

`llvm_signal_math` already owned real alternatives (a baked sine table and a
reduced-range series), but as hand-written LLVM text reachable only from a
tape path that is dead to the compiler and from a torture-matrix oracle. No
deployment product had ever used it.

---

## 2. What exists now

**Cores** (`signal_math`). Fifteen declared cores, each with a stated
interval and reduction. Five families:

| family | what it is | when it wins |
|---|---|---|
| `exact` | parity-structured form, coefficients TAKEN | the default; reaches libm's own residual |
| `structured` | same form, coefficients FITTED | fewest constants for a loose target |
| `series` | plain Horner, exact coefficients | cores with no parity to carry |
| `polyspline` | segmented least-squares | last resort where the series diverges |
| `lut` | node table, a prerun of the series | uniform per-call cost |

Plus `AnglePalette`, which is not a family but a different thing: exact values
for a **declared** angle set.

**Every family sizes itself by MEASURED error**, never a set count, and stops
at the first admitted size rather than the largest tried — accuracy is not
monotone in size.

**Kernels** (`signal_kernels`). The cores are emitted as compiler SOURCE with
`n` as a real loop bound, the way `blas.py` does it, so `KernelBank` yields a
genuinely parametric variant plus a specialized matrix and `LaunchCoordinator`
routes per call. Authored: `sin`, `cos`, `tan`, `exp` forwards; `sin`, `cos`,
`tan` reverses. All parametric.

**The switch** (`trigonometry.py`). Both implementations live behind
`use_signal_math()` / `use_backend_operator()`. **Default is `operator`** and
should stay there until callers reach the pack through the bank — the
interpreted route is 99x slower, so flipping it now buys a slowdown.

---

## 3. The numbers, and how they were taken

Measured against **mpmath at 40–60 dps**, not against numpy. Where a figure
says "vs numpy" it measures *agreement with libm*, not accuracy; those are not
the same claim and this document uses only the former.

| core | family | consts | ulp |
|---|---|---|---|
| cos | series | 18 | 0.70 |
| sin | exact | 8 | 1.00 |
| exp | series | 14 | 1.00 |
| expm1 | series | 14 | 1.00 |
| cosh | exact | 9 | 1.30 |
| sinh | exact | 9 | 1.00 |
| log | series | 38 | 1.40 |
| atanh | exact | 23 | 1.80 |
| asin | exact | 21 | 3.80 |
| sqrt | polyspline | 256 | 9.70 |
| tanh | structured | 21 | 10.60 |
| sinc | polyspline | 256 | 11.30 |
| atan | structured | 19 | 24.00 |
| asinh | structured | 21 | 27.80 |
| log1p | structured | 17 | 2000.50 |

Nine of fifteen admit at a 1e-15 relative bar (~4.5 ulp).

**Compiled speed**, n = 65536: 1.3–1.7x numpy/libm, against 99x interpreted.
The interpreter was the whole gap. Precision is nearly free — 5 coefficients
to 7 bought ~36,000x accuracy for ~10% more time, because the reduction and
memory traffic dominate, not the polynomial.

**Palette**, N = 1024: 0.00 ulp, correctly rounded, against 11 ulp p95 for
`sin(2*pi*k/N)` computed the usual way. Stores one quadrant (2 KiB), and the
cardinal values are *placed* so `sin(0)`, `sin(1/2 turn)` are exactly +0,
quarter turns exactly ±1, and odd symmetry holds to the bit.

**Field render**: 12,582,912 samples at 2048² in 2.42 s, 5.2 M samples/s,
every sine and exponential from a compiled kernel.

---

## 4. Three findings that shaped the design

**Reduce in cycles, not radians.** `t - floor(t)` is exact at every
magnitude; dividing by a float TAU is not. Measured `sin(2*pi*c)`:

    cycles     radian-reduced     turn-reduced
     1e+03        6.140e-13        8.882e-16
     1e+12        5.256e-04        8.882e-16

`spectral-analyzer`'s `COMPLEX_OPTICAL_OPERATOR_CONTRACT` states the same rule
for carrier phase.

**Parity belongs in the FORM.** `sin(y) = y*P(y*y)` makes `sin(0)` exactly 0,
parity exactly 0, the range contained, and leaves no knots to jump across.
Against a plain 8-segment polyspline: parity 3.33e-16 → 0, `max|sin|-1`
+1.11e-15 (which hands `asin` an out-of-domain argument) → contained,
derivative jump 6.77e-13 → none, 80 coefficients → 9.

**Take coefficients, do not fit them.** A fit cannot go below its own
residual however much degree it is given:

    core                    coeffs      p50      p95
    structured (fitted)          7     3.00     7.00 ulp
    exact series order 15        8     0.00     1.00 ulp

I twice blamed the argument reduction for the ceiling, citing degrees 8/10/12
resting flat. That is what a fit's floor looks like from outside. Separately
measured, the reduction is already 1–2 ulp and splitting TAU into two terms
changes nothing.

---

## 5. Traps — read before authoring

**Authoring a kernel: a branch must WRITE ITS DESTINATION.** Setting a local
in each arm and using it after the merge emits a pointer-valued phi that does
not dominate its uses; LLVM rejects the module while `artifact.shortfalls`
stays **empty** — the false green `blas.py` records under §4.1b. Neither
splitting into two chains nor one assignment per branch avoids it. `tan` cost
three attempts and two wrong diagnoses to find this.

**`AbstractTensor.where` cannot appear in capturable code.** It bypasses the
backend's source-producing override and falls into the Python valuewise path,
which calls `tolist()` — fine eagerly, fatal under SSA capture.
`abstraction.nan_to_num` carries the same warning. Use
`type(condition).where(...)`, or better, branchless masked arithmetic:
`if_false + mask * (if_true - if_false)`. Its differential matches the
authored `where` rule exactly, and unlike a tensor `where` its reverse
compiles.

**Relative error near a function's zero is meaningless, and sometimes
inverted.** This produced a wrong headline SIX times in one session — a lone
sample on a zero moved a max from 2 ulp to 5.5e+15 while the median was 0.
For a bounded function use ulp of FULL SCALE. Report p50/p95, not max. The
palette scorer initially called its own exactly-placed zero wrong by 1e15 ulp
for differing from the reference's residue for pi.

**A propagated error bound is only valid on the interval its inputs were
measured on.** `sec` and `csc` predict to within 0.1 ulp from `e_cos + 0.5`
with no reference computed. `cot` came out 6x adrift because `cos`'s
core-interval figure was applied across `cos`'s zero. Error data must carry
its domain, exactly as `SSA_IDENTITY_CATALOGUE.md` §5 demands for facts.

**Use `tools/TRANSLATION_DEBUGGING.md`.** Its Q2 says any well-formedness
failure means the defect is at or above `precompile_to_ssa.py` and *do not
read backend code yet*. A whole investigation was spent inside
`ssa_llvm_backend.py` building and retracting two wrong accounts of a graph
that `ssa_self_check.run_all` diagnosed in one call.

---

## 6. What is left, in order

1. ~~**`sqrt` kernel.**~~ **DONE** (`13eeaa9`) — seeded Newton, 7 coefficients
   and 2 steps, admitted at p95 0.79 / max 1.00 ulp. Takes an argument already
   reduced to `[0.25, 1)`; the caller supplies the even binade. That boundary
   is forced: exponent extraction inside a kernel needs a `frexp` the authored
   vocabulary lacks, and a **data-dependent `while` loop HUNGS the compiler**
   — ten minutes, no module. Do not retry it.

   The lesson generalises past `sqrt`: a self-correcting iteration beats a
   better polynomial. 48 coefficients gave 328 ulp; 7 coefficients plus two
   Newton steps gave 0.79. Look for a fixed point before reaching for degree.

   Still to do behind it: the six inverse functions themselves, now that
   their `g / sqrt(...)` rules have a `sqrt` to call.

1b. **In-kernel range reduction — measured boundary** (`0b2a396` and after).
   The `frexp` gap is narrower than it looked. Three probes:

   | shape | result |
   |---|---|
   | `while v >= 1.0:` unbounded | timed out at 10 min — **UNPROVEN, see below** |
   | `for j in range(6):` with `if`/`else`, ONE carried scalar | **admits, correct** |
   | same, TWO carried scalars (`v` and `s`) | compiles, **fails verification** by 494 on a 500-magnitude input |

   **The `while` row is not a finding.** That probe ran while another agent
   was bootstrapping the compiler and using the machine, so a timeout is
   equally explained by contention. It is recorded as unproven rather than
   deleted, because the retest is cheap and worth doing on a quiet machine —
   but do not treat "`while` is unsupported" as established, and do not
   design around it. I stated it as a defect on one timed-out sample, which
   was wrong.

   The other two rows stand regardless of load: they are value results, not
   timings. A second loop-carried scalar mutated in a branch produces
   deterministically wrong numbers, which contention does not explain.
   Admission caught it — the oracle earned its keep, and it caught it
   precisely because the reference is the same source executed as Python
   rather than a hand-written twin that might have shared the mistake. That
   points at the frozen-carried defect
   `ssa_self_check.suspicious_loop_invariant_formals` exists to flag; run it
   on the module before believing any multi-accumulator kernel.

   A one-accumulator reduction is therefore available today, which is enough
   for a self-contained `sqrt` if the scale is recovered without a second
   carried variable.

1c. **Quadratic-quotient reverses — DONE** (`0b2a396`). The six inverse rules
   split: `atan` and `atanh` are `g / (1 +- x*x)`, pure arithmetic, needing no
   core at all — authored and admitted at 1.05 and 3.12 ulp. The other four
   are `root_quotient`, blocked not on the derivative but on the RADICAND:
   `1-x*x` on [0,1], `x*x+1` unbounded above, `x*x-1` from zero, none in the
   mantissa band the `sqrt` kernel takes. One gap, four methods.
2. **Hyperbolic kernels.** `sinh`/`cosh`/`tanh` need their cores emitted plus
   `exp` inlined for the out-of-band identity. `exp` is authored, so this is
   unblocked.
3. **The six cores that miss 1e-15.** Each has an error map pointing at where
   to cut — per-eighth max ulp:

       atan     8.7  9.2  8.9  9.9  9.8 24.0 20.2 18.6   (rises toward 1)
       tanh    10.6  7.2  6.2  6.1  6.1  6.2  7.2  9.9   (edge-heavy)
       asinh   12.8 12.6 27.7 27.5 27.5 27.7 12.6 12.8   (middle-heavy)
       sqrt     9.7  2.5  2.8  2.6  2.5  3.4  2.2  2.6   (one eighth only)

   These want piecewise resolution driven by the map, not more degree
   everywhere. `atan` additionally wants its reciprocal identity to switch at
   tan(pi/8) ≈ 0.414 rather than at 1.
4. **`log1p` at 2000 ulp** — diagnosed to the fit, not resolved.
5. **Special values.** `±inf`, `NaN`, `±0`, out-of-domain. Uncharacterised;
   blocking for production.
6. **Commit the baked coefficients.** Prebake is 6 s (`audio`) to 135 s
   (`double`) per process. It should be a build step with the numbers
   auditable and diffable, not a fit at first call.
7. **Tests.** Nothing is pinned — not the accuracy table, not the structural
   guarantees, not admission, not routing. Every number here is a
   hand-measurement taken once.
8. **A second backend.** "Backend agnostic" is an argument from where the
   source lives, not a measurement. Only LLVM/native is demonstrated.

Then the larger arcs already scoped: N-double via double-double (compensated
arithmetic was measured to survive both the `fast` and `develop` contracts),
and the identity library — whose real content is the *annotations*, since an
identity without a validity domain and a cost is not actionable, and the
amplification factor is exactly where the validity domain comes from.

---

## 7. Running it

    python -m tools.signal_math_survey          # accuracy/cost map, all cores
    python -m tools.demo_signal_field --size 2048   # the 5D render + accounting

`signal_math.signal_math(quality)` gives the eager surface;
`signal_kernels.signal_kernel_specs(quality)` gives the kernel specs for a
`KernelBank`. Qualities: `draft`, `audio`, `double` (the kernel default),
`reference`, `table`, `definitional`.

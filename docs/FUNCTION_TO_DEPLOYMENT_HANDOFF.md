# Reducing a Python function to a deployment, backed by profiling

**Date:** 2026-08-19.
**For:** whoever builds the *generic* tool. This document is the worked
single case (`eigh`) written up so the general one does not have to rediscover
it, plus the defects that will stop it and the instruments that find them.

**Continues:** `DIAGNOSTIC_DECISION_TREE_AND_MACHINE_READINESS.md` (the tree
every step below follows), `AUTO_PORT_ROUND_TRIP_AND_SPECTRAL_DIAGNOSIS.md`
(the auto-port half), `TEST_BASELINE_AND_HAZARDS.md` (read before running
anything).

## 0. The thesis, and the evidence for it

**We should not be hand-writing performant kernels. We should write the
algorithm plainly, profile it, and let the compiler produce the deployment.**

That is now demonstrated rather than asserted. `AbstractTensor.eigh` is a
readable pure-AT Jacobi implementation. It is also unusable at real sizes: on
the 115-node SSA graph a real program compiles to, it takes **521.5 s**. The
same algorithm, written as plain Python over a flat array and put through this
tree's own compiler, runs the same problem in **0.257 s** — about **2030x** —
to machine precision, with eigenvectors orthonormal and `A V = V diag(w)`
exact.

| n | `AT.eigh` | compiled | numpy (LAPACK) |
|---|---|---|---|
| 30 | 34.3 s | 0.053 s | 0.001 s |
| 115 | 521.5 s | 0.257 s | 0.009 s |
| 200 | — | 1.36 s | 0.013 s |

Nothing in the compiled kernel was hand-optimised. It is the algorithm,
written out plainly.

## 1. What profiling actually said (and how to profile correctly)

**Profile on the numpy backend.** The default nodus backend inflates `eigh`
**4.7x** (n=30: 161 s vs 34.3 s), so a nodus profile measures the wrong thing.

On numpy, at n=30, `cProfile` says the real numpy work is ~9.7 s of 89.8 s
cumulative: **~89% of the time is AbstractTensor dispatch, not arithmetic.**
Two specifics worth fixing on their own:

* `_pre_autograd` is the single largest cost and executes
  `from . import autograd` **inside the function body**
  (`abstraction.py:2410`) on each of ~916k calls — ~1.7M `importlib` calls in
  one `eigh`.
* `no_grad` buys nothing (x1.03). The cost is invoking the machinery, not
  recording, so "just turn off the tape" is not the fix.

This is the profile shape that says *compile it*: when the interpreter, not
the arithmetic, is the cost, a compiled kernel deletes essentially all of it.
A generic tool should treat "dispatch >> arithmetic" as its trigger.

## 2. The pipeline, and the exact API

```python
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_llvm_backend import (
    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
)

module, outputs, _exports = lower_ast_source_to_ssa(source, "fn", name="tag")
artifact = emit_ssa_function_to_llvm(module, "tag__fn")   # .shortfalls must be ()
native   = compile_artifact(artifact, directory=some_dir)
execution = prepare_artifact_execution(native, {value_id: numpy_array, ...})
execution.run()
result = np.asarray(execution.buffers[value_id])
```

Confirmed to compile with **zero shortfalls**: scalar arithmetic, `abs`,
`**0.5`, comparisons, bool→float arithmetic, single loops, nested loops,
dependent inner bounds (`range(p+1, n)`), computed flat indices
(`a[i*n+j]`), and in-place stores. A full two-sided Jacobi with eigenvector
accumulation — 25 basic blocks — emits clean.

### 2.1 The calling convention. Read this part twice.

This is where all the lost time went. Every one of these cost a wrong answer
that looked plausible.

1. **SSA value ids are not stable across lowering runs.** The same source
   lowered twice gives different ids. Never hardcode one, never carry one
   between processes. Derive everything inside the run that produced it.
2. **`artifact.buffer_order` is authoritative** for execution. It can
   **disagree with `parameter_names`** — observed `parameter_names` claiming
   `S→0` while the buffers had `S→1`. When they disagree, believe
   `buffer_order`.
3. **Extra formals are storage aliases, not junk.** A compiled function
   routinely has more formals than the author wrote. Some are *aliases of the
   same storage* as a named parameter, and you must pass **the same object**
   for both. Passing a separate copy of equal contents gives a *partially*
   correct answer — the rotated rows were right and the rest stale — which is
   far more dangerous than an obvious failure.
4. **Never scratch-fill an unbound formal and trust the result.**
   `bind_program_abi_arguments` (`ssa_reference_evaluator.py`) binds by
   declared identity, never by position, and returns `unbound`. Its docstring
   states the rule this tree keeps paying for: *an unbound formal is not the
   same claim as a zero*. It correctly refused to guess; accepting its zeros
   is what produced a silently wrong eigendecomposition.
5. Loop induction variables can appear as formals. Feeding them 0 is
   harmless *when they are genuinely loop-initialised* — verified — but see
   §4.2, because under specialisation this stops being cosmetic.

## 3. Diagnosing a wrong compiled result

Follow the tree, but these two moves resolved everything here:

* **Run the kernel in plain Python first.** It separates "my algorithm is
  wrong" from "the compiler is wrong" in one step. My Jacobi was correct in
  Python, which made the compiler the only suspect — and later, when the
  *evaluator* also disagreed, made my *calling convention* the only suspect.
* **`SSAReferenceEvaluator` runs the SSA without LLVM.** This is the
  instrument that pins the layer. Evaluator right + native wrong = codegen
  defect. Both wrong = lowering or driving. It moved this investigation twice,
  once away from a codegen theory that was wrong.

A third, cheap and decisive: **check whether an invariant survives.** The
compiled Jacobi preserved the matrix trace exactly while getting eigenvalues
wrong, which proved the rotations were orthogonal and pointed at the operand
feeding them rather than at the rotation math.

## 4. Defects the generic tool will hit

### 4.1 Fixed: a load re-read across an in-place store
`3406310`. A `Load` from a span address aliased its **result to the address**,
and array addresses are deliberately never register-cached, so every use
re-emitted its own load; an intervening store to the same element changed
later uses. Read-a-pair → combine → write-both — every plane rotation, every
in-place swap — silently computed with the overwritten operand while emitting
zero shortfalls. Fixed by pinning scalar loads into their own slot.
`tests/test_llvm_inplace_store_aliasing.py` pins both the behaviour and the
emitted IR shape.

### 4.2 OPEN, and it blocks specialisation: a write-only array parameter is dead-stored
`a4bc25d`. Under a **literal** loop bound, whichever array parameter the
returned value does not depend on is eliminated — gone from `args`,
`parameter_names` and `value_names` together.

```python
def f(A, B):
    for i in range(4):          # literal bound
        A[i] = A[i] * 2.0
        B[i] = B[i] + 1.0
    return A                    # -> emitted signature is (A,) alone
```

Not argument order: returning `B` instead drops `A`. Reading `B` into the
result keeps both. The same writes with no loop keep both. `range(n)` does not
trigger it. **The eliminated store lands in the caller's array — it is the
function's observable effect, not dead.** It only looks dead from inside.

Traced, not guessed: `value_name_histories` reaching `_ControlSSABuilder` is
`{'A': (0,)}` for the literal form against
`{'n':(0,), 'A':(1,8,13), 'B':(2,12,14)}` for the parameterised one. That
table is the ProcessGraph's `identity_table`
(`fortran_c_shell.py:3204`), built from `identity_bindings` in
`topological_reducer.py` — so the name is lost during graph construction, well
upstream of the SSA builder that reports the consequence.

**Why it matters to the generic tool:** specialisation is the biggest single
lever it has. A fully-specialised eigh (size and sweeps as literals) runs
**0.044 s vs 0.138 s** at n=115 — about **3x**, taking it from 23x off LAPACK
to **9.6x** — but it is currently wrong, because `V` is dropped exactly this
way. Fix this and specialisation becomes the default strategy.

Left unfixed on purpose: another session was actively committing to
`topological_reducer.py` during this investigation (`ecb2dab`), and a blind
edit to a dead-store rule in a file under concurrent change is the unscoped
move this tree's guidance warns against.

## 5. Is Jacobi still the right algorithm?

Yes for now, with one correction already made, and with no claim that it beats
LAPACK.

**A correction that mattered.** The kernel first used an **absolute** cutoff
(`|a_pq| < 1e-14`) to skip a rotation. On a graded SPD matrix with
`cond ≈ 1.2e18` that is catastrophic: the smallest eigenvalue came out with
**1.78e-02** relative error. Replacing it with the **relative** criterion the
high-relative-accuracy result actually requires
(`|a_pq| < eps*sqrt(|a_pp * a_qq|)`) fixed it completely — every eigenvalue,
including one at `8.4e-18`, now matches an `mpmath` (dps=60) oracle to
`3.66e-16`, **the same as LAPACK**. Ordinary matrices are unaffected in both
speed and accuracy.

That episode is the thesis in miniature: a real numerical bug was found and
fixed by editing readable Python and recompiling, not by touching a kernel.

**Honest standing:** on that test the corrected Jacobi *ties* LAPACK; it does
not beat it. I did not produce a case where it wins, so do not claim one.
What Jacobi does have, and what is worth keeping it for here:

* **It compiles.** Simple rotations, no deflation, no pivoting, no
  workspace-shape logic. LAPACK's algorithm would be a far harder thing to put
  through this pipeline, and the point is the pipeline.
* **It is embarrassingly parallel** — disjoint `(p,q)` pairs rotate
  independently under a tournament ordering. That is the natural fit for the
  deployment backends this tree already has (threads, GPU), and it is where
  the implementation gap below would actually be closed. *Unmeasured here.*
* **It warm-starts.** Feed the previous `V` and a slightly-changed matrix
  converges in a couple of sweeps. For spectral graph analysis over an
  evolving graph this is the property that matters most, and LAPACK cannot do
  it at all. *Unmeasured here — worth measuring early.*

## 6. Should we write a BLAS? Not now.

The gap to LAPACK at n=300 is **~52x**, and it splits cleanly:

* **~8x algorithmic** — Jacobi does about eight times the arithmetic of
  tridiagonal reduction + QR for the same answer.
* **~6.5x implementation** — we emit scalar code: 1.21 GF/s against LAPACK's
  7.85 GF/s. That is roughly **40% of scalar peak**, so it is honest scalar
  code, not a pathology. The missing factor is SIMD, cache blocking and
  threads.

Three reasons to leave BLAS alone right now:

1. **It would not help this kernel.** Classical Jacobi is rotation-based —
   BLAS-1/2 shaped. Only *block* Jacobi has GEMM-shaped inner work. Writing a
   GEMM and then not being able to call it from the thing we care about is the
   worst outcome.
2. **Correctness first.** §4.2 is live, and the ABI has surfaces that
   disagree with each other (§2.1). Optimising on top of a signature that can
   silently lose a parameter compounds the problem.
3. **The leveraged move is in the backend, not in a library.** Teaching the
   emitter to vectorise and block its loops is done **once** and speeds up
   *every* compiled kernel, which is the whole premise. A hand-written BLAS is
   exactly the "hand-written perfect kernel" this work exists to stop needing.

Revisit when §4.2 is fixed and a real workload is GEMM-shaped.

## 7. What the generic tool should be

A function goes in; a verified deployment comes out. Concretely:

1. **Trigger from a profile.** Measure on the numpy backend. If dispatch
   dominates arithmetic, the function is a candidate. Record the measured
   baseline — it is the only honest claim of improvement later.
2. **Take the definitional version as the spec.** The AbstractTensor
   implementation stays the readable authority. The compiled artifact is an
   accelerator, never the definition.
3. **Compile, then *prove equality against the definitional version*, not
   against a remembered number.** Include a degenerate and an extreme case:
   the absolute-threshold bug in §5 passes every well-conditioned test and
   fails only on a graded matrix.
4. **Bind by declared identity, and refuse on `unbound`.** Never scratch-fill
   and proceed. Surface aliased formals explicitly.
5. **Specialise when the shape is known** (§4.2 first) — worth ~3x — and keep
   the parameterised artifact as the fallback.
6. **Install behind the definitional API**, with the backward supplied
   explicitly rather than taped: `GradTape.backward_overrides`
   (`autograd.py:227`) takes `{op_name: callable}`. This is the legitimate
   case for deliberately coarsening the tape — differentiating *through*
   400k Jacobi operations is absurd when an analytic `eigh` backward exists.
   Note the general rule still holds elsewhere: composites must **not** wrap
   `_pre_autograd` (measured: identical gradients with and without, and it
   pollutes the tape with an op that has no `BACKWARD_RULES` entry).

**Not yet done, and the next concrete step:** the backward half. The forward
is working and verified; nothing installs it as `AbstractTensor.eigh` yet, and
no compiled backward exists. The analytic form is standard —
`dA = V (diag(gw) + F ∘ (Vᵀ gV)) Vᵀ` with `F_ij = 1/(w_j - w_i)` off-diagonal,
zero on it — and it is expressible in exactly the plain-Python-over-flat-array
style the forward used, so it should compile the same way.

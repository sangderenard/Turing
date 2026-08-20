# Auto-porting through the round trip, diagnosed with the spectral tool

**Date:** 2026-08-19.
**Continues:** `docs/DIAGNOSTIC_DECISION_TREE_AND_MACHINE_READINESS.md` (the
decision tree this investigation follows), `DTYPE_AND_SPECTRAL_DOMAIN_MANIFESTO.md`
(the complex/interp primitives this needed), `src/compiler/spectral_graph_analysis.py`
(the new diagnostic tool exercised here for the first time on real, not
hand-built, compiler output).

## 0. The goal, stated plainly

Real signal-processing math exists in a sibling repo (`spectral-analyzer/`'s
CQT/VQT filterbank and CWT wavelet transforms — surveyed separately, both
confirmed portable to `AbstractTensor` in principle: plain add/mul/matmul/fft,
no torch-specific autograd tricks). The question this report answers is
narrower and more load-bearing: **can material like that be *auto-ported* —
authored once in Python against `AbstractTensor`, then carried through this
tree's own round-trip compiler (source → SSA → re-materialized Python) into
what the compiler already calls "the universal translation world" — or does
today's pipeline only work for the toy scalar programs it has been tested
against?**

The honest answer, reached by actually trying it rather than reading the
pipeline and assuming: **not yet, and the reason is a real, silent defect,
not a missing feature.** Two real gaps got fixed along the way (both landed,
tested, committed). One larger gap got found, precisely bounded, and left
undone on purpose — it is deep, unfamiliar, and this report's job is to name
it exactly, not to guess a fix.

## 1. Following the decision tree

Branch 1 of the decision tree: *"Is the LOWERING's own claim suspect? …
Write the smallest authored program with the same shape … and score it."*
`tools/translation_scorecard.py` is exactly that instrument — it already
runs `LOWER → MATERIALIZE → EXECUTE → EQUIVALENT` on nineteen small authored
programs and reports the first stage each one fails. Run cold, unmodified,
before touching anything:

```
17/19 journeys equivalent end to end
```

Two known stops, both pre-existing and unrelated to anything below:
level 14 (a rebound parameter name shadowing three values) and level 18
(a generator predicate inside `any()`) both stop at `MATERIALIZE`. This is
the tool's own honest baseline, not something this session changed.

Every one of the nineteen journeys is a bare scalar `float` function —
`import math`, no tensors, no `AbstractTensor`. That is the first thing
worth being precise about: **the round-trip pipeline has never been
exercised on an AbstractTensor program before this investigation.**
Confirmed directly — `src/compiler/ssa_python_materializer.py` has zero
occurrences of `AbstractTensor`, `numpy`, or `torch` anywhere in it; it only
emits `import math` (`materialize_module`, near line 1001).

## 2. The interpolator gap, and a real one behind it

The CWT port's one genuinely missing primitive was 1-D linear interpolation.
`AbstractTensor.F.interpolate` already exists (`abstraction.py:2862-3000`)
but is an eager escape hatch: it hardcodes `import torch`, converts to a
concrete backend, calls `torch.nn.functional.interpolate` or
`PIL.Image.resize`/`scipy.ndimage.zoom` directly, and wraps the raw result
back — no `_pre_autograd` call, no backend dispatch, and critically **not
SSA-lowerable**: a compiler cannot lower a call to an external library
function it never sees as an operator. Using it here would have reproduced
exactly the problem this whole investigation is about, one level down.

`AbstractTensor.interp` (added, `abstraction.py`, next to `searchsorted`)
is base-operator only — `searchsorted` (already in the tree, itself
compare+sum, no backend hook) finds each query point's bracketing sample
indices, `index_select` gathers the four bracketing values, the blend is
ordinary arithmetic. No backend hook anywhere in it, so every backend
inherits it for free, matching the `unravel_index_`/`searchsorted` pattern
already established in this tree.

Building it surfaced a second, real, silent gap: `index_select` never
called `_pre_autograd` at all. Verified directly, isolated from `interp`
entirely — `x.index_select(0, idx)` followed by `.sum().backward()` returned
`x.grad is None`, silently, no error. Fixed by wiring the same
`_pre_autograd`/`finalize` pattern `gather` already uses, with a new
`backward_registry` entry reusing `index_adjoint` (the same repeated-index-
accumulating adjoint `__getitem__`'s own fancy-indexing backward already
uses) rather than `gather`'s own backward, which assigns rather than
accumulates on repeated indices and would have silently dropped duplicate
contributions.

Both fixed, both tested (14 new tests across two files), both committed
(`bd95481`). Verified against `numpy.interp` exactly, including default
boundary-clamp behaviour and degenerate zero-width segments.

## 3. The real blocker: unrecognized calls vanish silently at LOWER

With the interpolator real, the actual round-trip attempt:

```python
from src.common.tensors import AbstractTensor as AT

def helper(a):
    return a * 1.0

def train(x, y, q):
    return AT.interp(q, x, y)
```

```
LOWER:       ok
MATERIALIZE: ok, skipped={}
```

Emitted Python:

```python
def rt1__train(t1, t2, t0):
    pass
```

`AT.interp(q, x, y)` — the entire return statement — is gone. Not an
error, not a shortfall, not a warning. `outputs` for the function came back
as literally `{'rt1__train': ()}` and the whole entry block is one bare
`Ret []` with no other instruction. This is worse than a crash: every stage
the scorecard checks reports success on a program that computes nothing.

**Bounded precisely**, per the decision tree's instruction to test the
smallest shape that isolates the claim — four one-line probes, same
preamble, only the return expression changed:

| body | result |
|---|---|
| `x * 2.0` | lowers correctly — real `Call`/arithmetic instructions, non-empty `outputs` |
| `abs(x)` | lowers correctly |
| `x ** 0.5` | lowers correctly |
| `totally_made_up_name(x)` | **silently empties** — `outputs={'…': ()}`, one bare `Ret` |

`AT.interp` behaves identically to `totally_made_up_name`. This is not
specific to interpolation, or to `AbstractTensor`, or to anything this
session built — it is what `lower_ast_source_to_ssa` does with **any call
to a name outside its recognized operator vocabulary**: the statement
containing it is discarded rather than the lowering refusing.

This is the actual blocker named in the task that started this
investigation. It is not "we need to write more portable math" (the CQT/CWT
survey already confirmed the math itself is portable) and it is not "we
need an interpolator" (built, §2). It is: **the compiler that would carry
ported material into SSA has no failure mode for an operator it does not
recognize — it has a silent-success mode instead.** Every one of this
session's earlier findings about silent degradation
(`to_dtype_` → float32, `eigh` → the bare diagonal, `real()` dropping
gradients, `unravel_index_` keeping only element 0) was a defect *inside* a
function. This one is a defect in what the compiler is willing to say about
a function it does not understand, which makes it the more consequential
class: it silently authorizes shipping a program that does nothing, with
every diagnostic available today reporting success.

**Left undone on purpose.** The fix belongs in `lower_ast_source_to_ssa`
(`src/compiler/fortran_c_shell.py`) — the same enormous, unfamiliar file an
earlier investigation this session spent a long, careful trace inside
(`80ac938`) before finding a five-layer root cause. A blind attempt here,
in an already-very-long session, risks exactly the kind of unscoped change
this tree's own guidance warns against. What belongs on record instead is
the bounded reproduction above: a future session can start directly from
"why does an unrecognized `Call` never raise or get recorded as a
shortfall, when every other unhandled shape in this tree is required to
say so" — a Stage-0-shaped question in `docs/ADDING_AN_OPERATOR.md`'s own
terms, not a fresh investigation.

## 4. Exercising `spectral_graph_analysis.py` on real, not hand-built, output

Every prior use of the new spectral tool this session was on a graph built
by hand for a test or demo. Here it ran for the first time on the *actual*
SSA a real journey compiles to — level 7's adam-shaped loop
(`tools/translation_scorecard.py:131-148`), lowered through the same
`lower_ast_source_to_ssa` path, fed through `field_from_ssa` (one of the
four adapters `spectral_graph_analysis.py` was built to accept unmodified,
per `b74ec78`).

```
nodes: 115   edges: 118
```

`field.propagate()` completed. `analyze_graph_spectrum(field)` did not
finish within 70 seconds. Isolated directly rather than left as a vague
"it hung": a bare `AT.eigh` call on a 115×115 symmetric matrix (no graph
machinery at all, plain `numpy.random` input) still had not returned after
40 seconds. Compare against the number already on record from much earlier
in this same session: `eigh` on an 8×8 matrix took **3.3 s**. Going from
n=8 to n=115 (14.4×) and from 3.3s to over 40s (12×+) is consistent with the
pure-Python nested-pair Jacobi sweep's expected scaling — the algorithm
itself is correct (verified exactly against `numpy.linalg.eigvalsh` earlier
this session) and simply has no fast path for anything past toy sizes.

**This is a second, independent, honestly-reported limit** — not a bug in
`spectral_graph_analysis.py`'s design, which correctly delegates to
`AT.eigh` exactly as intended, but a performance ceiling in `AT.eigh` itself
that the tool inherits. A 115-node function is not an unusually large
program by this tree's own standards; it is what one small, real,
five-argument training loop compiles to. The tool's piecewise design (loop
regions dispatched to the FFT path when circulant, §`b74ec78`) partially
mitigates this — a genuinely circulant loop region, however large, is
`O(N log N)` regardless — but the *whole-graph* spectrum, and any
non-circulant region, still pays the full dense cost, and real loop bodies
(confirmed on the ring+tail demo in `87ff8a1`) are not simple circulant
rings, so the fast path does not reliably save the common case.

## 5. What auto-porting into the universal translation world actually needs

Three things, now precisely known rather than assumed, in the order that
unblocks the most:

1. **§3 is the gate.** Nothing new can be safely auto-ported through
   `lower_ast_source_to_ssa` until an unrecognized call either lowers
   correctly or refuses loudly. Until then, every "successful" round trip of
   ported material is unverified by construction — `MATERIALIZE: ok,
   skipped={}` is not evidence of anything.
2. **Once §3 is fixed, `AT.interp` and the portable half of the CQT/CWT
   survey are genuinely ready to try** — real, tested, base-operator-only
   primitives with nothing torch-specific in their math, which was the
   actual target of "auto-port until we can auto-port."
3. **§4 is a real, separate ceiling on the diagnostic tool itself**, worth
   fixing on its own schedule (a proper eigensolver, or accepting the
   dense-`eigh` path as toy-scale-only and leaning harder on the
   piecewise/circulant fast path for anything real) — but it does not block
   §1's translation goal, only how far the spectral diagnostic can currently
   see into a large real program.

## 6. Recorded conclusions

1. `AbstractTensor.interp` and the `index_select` autograd fix are real,
   tested, committed (`bd95481`) — the interpolator gap named at the start
   of this task is closed.
2. The round-trip pipeline (`lower_ast_source_to_ssa` +
   `materialize_ir_module`) silently discards any statement containing a
   call to an operator name it does not recognize, reporting success at
   every stage a scorecard-style check would look at. Bounded precisely
   with four one-line probes; not specific to `AT.interp` or to
   `AbstractTensor`. This is the actual, load-bearing blocker on auto-
   porting anything, named exactly rather than left as "it doesn't work
   yet" — and left unfixed on purpose, given its likely depth, rather than
   attempted blind.
3. `spectral_graph_analysis.py`, exercised on real round-tripped SSA for
   the first time, correctly extracted real topology (115 nodes, 118 edges)
   via an unmodified `field_from_ssa` adapter — confirming the "any IR
   representation" design claim from `b74ec78` on genuine compiler output,
   not a demo. `analyze_graph_spectrum` then hit a real, independently
   confirmed performance ceiling in `AT.eigh` (>40 s at n=115, against a
   known 3.3 s at n=8) — a second honest limit, reported rather than hidden,
   and orthogonal to the tool's own design.

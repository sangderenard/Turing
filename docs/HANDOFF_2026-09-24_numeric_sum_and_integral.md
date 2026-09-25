# Handoff 2026-09-24 — numeric Sum, and the Integral as a shim over it

## One-paragraph state

A SymPy `Sum` / `Product` / `Integral` that SymPy cannot close now lowers in
the SymPy ingestion (`src/compiler/symbolic_process_graph.py`) as one
**vectorized reduction over its declared Domain**, instead of being refused
(Sum) or unrolled as a fixed 5-point Gauss–Legendre expression (Integral).
The Integral is a shim: it is the same reduction with the quadrature rule's
nodes as the index axis and `weight * half` as a factor of the body. Eagerly
(stage source run through AbstractTensor) the sums are exact to rounding and
the integrals equal their quadrature rule to every printed digit. **The
integrals' remaining error is pure discretization: nothing yet chooses the
rule or the spline against an error target. That is the main work below.**

The change is in the working tree, uncommitted, in two files:
`src/compiler/symbolic_process_graph.py` and `src/compiler/bitops.py`
(`Limit.lower` / `Limit.upper` accessors — `Limit` kept its bounds only in
`integer_pieces[1:]`).

## How it is wired (read these first)

* `bitops.declare(expr)` (existing) gives a `Declaration(Sum|Integral, expr,
  Domain, fields)`. `Domain.axes` are the index/integration symbols,
  `Domain.limits` one `Limit` per axis (SymPy order: innermost first).
  `Integral` subclasses `Sum`.
* `bitops.PROGRAMMATIC_LOWERINGS` (was empty) now holds, plugged at import of
  `symbolic_process_graph`:
  * `Sum -> lower_sum_declaration`
  * `Integral -> lower_integral_declaration`
* In `ingest_sympy_expression`, a surviving Sum/Product goes:
  1. `value.doit()` — a closed form or finite expansion ingests normally;
  2. `_split_reduction(value)` — exact identities only: linearity,
     index-free factor out, index-free body `n*c` / `c**n`, product of
     factors. Each piece is re-ingested, so each is offered to SymPy again;
  3. `lower_declared(value)` → `declare` → the plugged lowering.
  The Integral branch keeps its SymPy-first path and bare-domain refusal, and
  its quadrature fallback is now `lower_declared(value)`.
* `lower_sum_declaration`: for each axis (outermost first) the index is
  `lo + arange(n)` with **n computed at run time**, `n = max(hi - lo) + 1`
  (the Domain states its own size; no declared maximum — user decision). The
  axis is unsqueezed so axis `p` of `rank` sits at position `p` of a
  `rank + 1`-dim tensor; the trailing dim is the law's batch, so every
  ordinary law value `(batch,)` broadcasts against it. Inner bounds are
  ingested with outer indices bound to their axes (triangular sums work).
  Each axis is masked `k <= hi - lo` to the reduction's neutral element with
  `AbstractTensor.where`, then `.sum(0)` / `.prod(0)` once per axis.
* `lower_integral_declaration`: per axis, `get_tensor(nodes)` and
  `get_tensor(weights)` of `_gauss_legendre_rule()` (5 points, 17-digit
  floats), mapped `mid + half * node`, body multiplied by `weight * half`,
  then the same per-axis `.sum(0)`. A `ParametricDomain` declaration
  re-ingests its pulled-back `fields["parametric"]` Integral.
* The ingestion hands a lowering an `ingest` context: `add_node`,
  `bound(expr, {symbol: node_id})` (binds symbols to existing nodes and drops
  the memo entries made under the binding afterwards), `op(name, parents)`
  (a node carrying `tensor_operation=name`, which `ssa_python_materializer`
  prints through the catalogued AbstractTensor call form), `literal(value)`.
* Everything printed is catalogued AbstractTensor vocabulary: `arange`
  (static), `unsqueeze`, `max`, `where` (static), `get_tensor` (static),
  `sum`, `prod`. `AbstractTensor.arange` accepts a 0-d tensor extent
  (checked).

## Measured (eager, stage source run through AbstractTensor, runtime n per batch element)

| law | worst relative error |
|---|---|
| Σ_{i=0}^{n} sin(ix)/(i+1) + cos(x i²)/(i+2)² (split into two reductions) | 1.5e-16 vs direct loop |
| Σ_{i=1}^{n} x·sin(ix) + 3 (factor + constant split) | 1.4e-16 |
| Σ_{i=0}^{n} Σ_{j=0}^{i} sin(ijx) (triangular) | 4.3e-16 |
| Π_{i=1}^{n} (1 + sin²(ix)) | 0 |
| ∫₀ᴸ e^(−at²)·cos(bt³) dt | 5.5e-2 vs mpmath — **equals GL-5 by hand to 10 digits** |
| ∫₀ᴸ ∫₀ᵗ e^(−atx)·cos(bx³t) dx dt (triangular) | 1.0e-3 vs mpmath — **equals GL-5 × GL-5 by hand** |

The integrands above are non-elementary: SymPy's `doit` fails on them, so
they genuinely are numeric integrals and they were routed correctly. The
numeric core is exact to its rule; the rule is the error.

## The work

1. **Discretization against an error target (the main item).** Replace the
   fixed `_gauss_legendre_rule()` (5 points, one panel) with a rule chosen to
   meet a target. The precision policy's tiers (`precision_policy.py`,
   `PRECISION_TIERS`, e.g. faithful = 1 ULP of the result) are the natural
   target. The adaptive polyspline (`src/common/tensors/youngman/piecewise.py`,
   `adaptive_polyspline`, `AdaptivePolyspline`; h- and p-indicators, declared
   breakpoints, Kuhn simplices) is the destreamed integrator meant for this.
   The question to settle with the user: when the integrand's inputs are
   runtime values, the subdivision is either chosen at compile time from
   declared samples (the precision planner's pattern) or realized at run time
   (e.g. vectorized composite panels with a mask, the same form as the Sum
   lowering). When every input is known at compile time the polyspline can
   produce the value directly.
2. **Exact nodes and weights.** `gauss_legendre(points, 17)` hands 17-digit
   floats into the program. Nodes/weights should enter as exact rationals or
   as limbs from exact values ("coefficients at the depth of their tier",
   as the proof cores do — `limb_decomposition`, `constant_limbs`). The
   rational tensor work in progress (`docs/RATIONAL_TENSOR_COMPOSITION_DESIGN.md`)
   is the other half of this.
3. **Accumulator status per reduction.** Every reduction is a plain
   `.sum(0)` today. The intended decision: body integer/rational in the index
   with rational constants → exact rational accumulation; floating body →
   the precision policy decides width, accumulation through the expansion
   `sum` (`Precision.sum`, pairwise wide adds) or the superaccumulator
   (`superaccumulator.py`). Note `precision_policy.WIDE_OPERATIONS` does not
   list `sum`/`where`/`arange`, so the planner will not currently widen a
   reduction.
4. **Native path (coordinate with the SSA work).** Not run. The law compiler
   (`symbolic_equation_compiler`) stamps every node `tensor={"shape": ()}`;
   these reductions carry an index axis of runtime extent next to the batch.
   The native side has to carry that axis before a law with a reduction can
   compile. Do not change SSA while another agent is in it.
5. **Smaller gaps.**
   * `declare` is called without `bindings`, so a `ParametricDomain`,
     `Table`/`Function` or operator binding cannot reach the ingestion yet.
   * Bodies that index arrays (`Σ a[i]`, `Indexed`) are untested.
   * `Sum.doit()` with large numeric bounds expands term by term; a large
     numeric count should probably go to the reduction instead.
   * `_split_reduction` handles single-axis sums only; multi-axis sums go
     straight to the vectorized reduction (correct, just unsplit).

## Constraints (from the user and AGENTS.md)

* Use the existing systems (`bitops.declare`/`plug`, Domain/Limit, the
  AbstractTensor vocabulary, the precision planner); do not add parallel
  machinery. Read before changing; ask one precise question when a choice is
  the user's.
* Integral stays a shim for Sum — one numeric reduction core.
* The vectorized form is the chosen realization; the Domain states its size
  at run time.
* Verify eagerly through `compile_sympy_equations` →
  `symbolic_abstract_tensor_source` → exec, against direct loops / mpmath,
  before any native compile.

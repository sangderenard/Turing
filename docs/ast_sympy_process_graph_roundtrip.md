# ProcessGraph ↔ SymPy: expressions, equations, and control flow

Turing now has two reverse projections in
`src/compiler/symbolic_process_graph.py`:

1. `process_graph_to_sympy_expressions()` rebuilds pure outputs as ordinary
   SymPy expressions. This is the compact form used for simplification and
   reconstruction.
2. `process_graph_to_sympy_relations()` keeps one symbol and equation per live
   graph value. This is the solver form: it preserves sharing and allows
   SymPy to infer inputs or intermediate states from constraints on outputs.

AST and SymPy remain source languages. Neither source object's field layout is
the ProcessGraph schema; both normalize through canonical value nodes and
operations.

## Coming back from SymPy

`SYMPY_PROCESS_GRAPH_TRANSLATIONS` is the inspectable reverse translation
table. Each `SympyProcessGraphRule` names the canonical ProcessGraph operation,
its fixed operand roles when applicable, and an optional node type. The
`ingest_sympy_expression()` visitor applies those rules while memoizing SymPy
subexpressions, so common-subexpression sharing survives reconstruction.

Some SymPy nodes need structural lowering in addition to a name:

| SymPy form | ProcessGraph form | Operand roles |
|---|---|---|
| `Add`, `Mul`, `Pow`, `Mod` | same canonical arithmetic operation | `arg:0`, ... |
| relations such as `StrictLessThan` | canonical comparison | `left`, `right` |
| `Piecewise` | nested three-input `Select` nodes | `condition`, `if_true`, `if_false` |
| `Indexed` and symbolic `getitem(...)` | `Indexed` | `base`, one or more `index` roles |
| `sin`, `cos`, `exp`, etc. | named canonical math operation | `arg:0`, ... |
| an undefined applied function `f(...)` | `Call` with `attributes["callee"] == "f"` | `arg:0`, ... |

The last row is explicit rather than guessed: the call is recoverable as an
uninterpreted SymPy function, but no purity or implementation semantics are
invented. Unknown non-function SymPy classes are listed in the graph's
`sympy_translation_fallbacks` metadata. Passing `strict=True` to
`ingest_sympy_expression()` rejects such classes instead. A successful
optimization round trip should require an empty fallback list and re-project
the rebuilt graph to the exact reduced SymPy expression.

## The control-flow trick

Control flow does not need a Python `if` once its predicate is represented by
a Boolean number. For a selector `c ∈ {0, 1}`, true value `t`, and false value
`f`, a branch merge is exactly

```text
c(c - 1) = 0
y = f + c(t - f)
```

The first polynomial restricts `c` to zero or one. The second is Turing's
`mu` selector: it returns `f` at zero and `t` at one. ProcessGraph `select`,
`Phi`, and `mu` nodes use this encoding in the relational projection. The
compact expression projection uses the equivalent SymPy `Piecewise` form.

This describes branch *meaning*, not just branch syntax. A solver can work
backwards from `y`, combine the merge with arithmetic on either arm, and infer
which predicate and inputs are possible.

## BitOps as polynomial algebra

The same construction covers Turing's functionally complete bit calculus.
For Boolean `a` and `b`:

```text
NOT(a)    = 1 - a
AND(a,b)  = ab
NAND(a,b) = 1 - ab
OR(a,b)   = a + b - ab
XOR(a,b)  = a + b - 2ab
```

Every free bit also receives `x(x - 1) = 0`. Consequently a one-bit
Turing-provenance graph made entirely of NAND and `mu` becomes a polynomial
system accepted by `solve`, `nonlinsolve`, or Gröbner-basis tools. Wider
integer values are reconstructed from little-endian bits with

```text
value = Σ 2**i * bit[i]
```

`boolean_polynomial()`, `boolean_domain_constraint()`,
`polynomial_select()`, and `unsigned_bit_expression()` expose these pieces.
The ProcessGraph relation projector currently applies the Boolean polynomial
directly when a primitive has one quantum. A vector primitive that still
hides several lanes is deliberately reported as uninterpreted; scalarize its
lanes (or splice the scalar NAND provenance) before claiming a complete
polynomial model.

## Loops are equations over time

A finite loop is not special syntax either. It is a repeated state transition:

```text
state[t + 1] = transition(state[t], inputs[t])
```

`unroll_symbolic_transition()` constructs a bounded set of simultaneous SymPy
equations. Branches inside the transition can use the same polynomial mux.
This matches SSA updates: every right-hand side at step `t` is substituted
before any step `t + 1` value is assigned.

The compiler should use the existing loop planner first:

- a statically bounded loop can be realized/unrolled and projected as a
  finite equation system;
- a retained dynamic loop is a recurrence plus its entry/exit conditions, not
  one acyclic expression;
- termination proofs and unbounded reachability are separate solver problems
  and must not be hidden behind `simplify()`.

## Effects and exactness boundary

Pure arithmetic, comparison, selection, and scalar Boolean primitives have
defined SymPy semantics. Calls with no registered mathematical meaning remain
uninterpreted functions and appear in `SymbolicProcessModel.uninterpreted`.
This is important for FIFO writes, entropy state, file output, exceptions,
opaque mutation, and external calls: their ordering remains explicit in the
ProcessGraph/control plan instead of being fabricated as commutative math.

A model is algebraically complete for a selected live slice when
`uninterpreted == ()` and all retained loops/effects have been realized or
given explicit transition relations.

“Uninterpreted” is not a dead-code or importance classification. The node and
its arguments remain in an explicit equation; SymPy simply treats its function
head as opaque because Turing has not supplied a valid mathematical rule. This
is safer than inventing commutative/pure semantics, although genuinely
stateful operations ultimately need state/event-transition equations rather
than a generic uninterpreted function.

The generated Mandelbrot homepage previously reported 647 uninterpreted
operations: 322 `minimum`, 322 `maximum`, and 3 `tanh`. Those were present in
the SymPy model and had not been reduced away. They were backend spelling gaps,
so `minimum`/`maximum` aliases and `tanh` are now registered and a regenerated
homepage model reports them as interpreted mathematics. The JFIF full-program
model has a different boundary: after registering lowercase constants and
indexed access, 65 operations remain unresolved (calls, static references,
attributes, unlowered control, tensor-shape operations, and raises).

## Example: solve a graph backwards

```python
import sympy

from src.compiler.symbolic_process_graph import (
    process_graph_to_sympy_relations,
)

model = process_graph_to_sympy_relations(process_graph)
output, = model.outputs
unknowns = tuple(dict.fromkeys(model.expressions.values()))
solutions = sympy.solve(
    (*model.relations, sympy.Eq(output, wanted)),
    unknowns,
    dict=True,
)
```

The expression round trip remains available for pure regions:

```text
ProcessGraph
  → process_graph_to_sympy_expressions()
  → simplify()/cse()/solve()
  → ingest_sympy_expression()/ProcessGraph.build_from_expression()
  → process_graph_to_sympy_expressions()  # validate exact reconstruction
```

Tensor-valued assignment is safe only when the selected SymPy operation has
the intended tensor semantics. Stateful calls and effect boundaries must stay
relational or explicit in ProcessGraph.

## AST/precompile round-trip measurements

The compact regression starts by AST-ingesting a deliberately redundant
polynomial function. Its direct precompile falls from 16 instructions to 4;
the ProcessGraph falls from 15 nodes to 4, and SymPy operation count falls
from 11 to 1 (`3*right`). The rebuilt graph re-projects exactly to that reduced
expression and uses no fallback translations.

The stress regression AST-ingests `encode_jfif_resident` and recursively
resolves its real pixel-to-JFIF source hierarchy. The corrected test applies
SymPy both to the compact output expression and to the complete per-node
equation model, then reconstructs the latter before invoking topology
reduction and precompile again.

The recorded complete run was:

| Stage | Measurement |
|---|---|
| recursively ingested AST module | 5,760 nodes |
| first reduced encoder | 108 nodes |
| first direct precompile | 109 instructions; graph becomes 109 nodes |
| compact SymPy before | 132 operations; `srepr` length 5,778 |
| compact SymPy after aggressive pass | 132 operations; `srepr` length 5,778; unchanged |
| complete SymPy program model | 107 equations; 65 uninterpreted operations |
| per-equation aggressive pass | 0 equations changed |
| reverse-table reconstruction | 143 nodes; zero fallbacks; all 109 modeled nodes mapped |
| second topology reduction | 143 nodes |
| final direct precompile | 145 instructions; graph becomes 145 nodes |

The larger reconstructed program is the honest result. Mathematical lowering
exposes comparison/control expressions as additional nodes, while the current
SymPy strategies find no reduction in this encoder. The earlier 109→48 result
was only the compact output expression and omitted process/effect equations;
it was not a valid whole-JPEG comparison. The full reconstruction also restores
all carried ordering edges, but executable JPEG equivalence remains a separate
test from structural preservation.

# Differential translation: phases

A translation is correct when every representation of the program computes
the same thing. That is a testable claim, and testing it directly is
cheaper than inferring it from structure. This is the design for making it
routine, for **any** translation scheme in this tree — not just the fluid
program, and not just LLVM.

The principle throughout: **compare representations against each other on
identical inputs, and report the earliest disagreement.** Structure checks
(`diagnose_translation.py`) prove a representation is well-formed. Only a
differential proves two representations *agree*.

---

## The lattice

Five representations, four useful comparisons:

```
authored Python / SymPy
        |                    (A) python <-> ssa      : did lowering change meaning?
   ProcessGraph
        |                    (B) ssa <-> backend     : did emission change meaning?
     dual IR
        |                    (C) python <-> backend  : end to end (built, Phase 3a)
  repository SSA
        |                    (D) backend <-> backend : which target is wrong?
   backend artifact
```

Each answers a different routing question, and the routing question is the
expensive one — this whole tree's worst time sinks were spent looking at
the wrong layer.

* **A disagrees, B agrees** → the defect is in lowering (ProcessGraph,
  planner, dual IR). The backend is faithfully rendering a broken plan.
* **A agrees, B disagrees** → the defect is in emission.
* **Both agree, C disagrees** → the defect is in the ABI/runtime seam:
  binding, extents, marshalling.
* **D disagrees** → at least one backend is wrong, and the one that
  matches A is right.

---

## Phase 3a — python ↔ backend (BUILT: `differential_translation.py`)

Runs the authored program through an **independent** oracle and through the
compiled artifact on bitwise identical inputs, comparing every state field
cell-by-cell, every `Metrics` field, every error channel.

The oracle is `sympy.lambdify` over
`symbolic_viscous_shallow_water_equations()` — the authored equations
evaluated by SymPy/NumPy, sharing no lowering, no SSA and no backend with
the artifact.

> **Independence is the whole point.** A reference that shares machinery
> with the thing under test inherits its bugs. This tree has already
> produced two wrong conclusions from a "ground truth" that had been
> computed from state the artifact had already corrupted.

Already earning its keep: on its first run it showed the compiled step
disagreeing with the SymPy equations in 11 of 16 cells, and the oracle
rejecting a step (`tracer_bounds`) that the artifact accepts — which is
precisely the behaviour the failing regression test expects, surfaced
without touching the test.

**Remaining work:** compare named intermediates, not just endpoints, so the
report names an authored local rather than a final field.

---

## Phase 3b — python ↔ ssa  ✅ BUILT AND CALIBRATED

`src/compiler/ssa_reference_evaluator.py`, calibrated in
`tests/test_ssa_reference_evaluator.py` on a pure function, on a synthetic
traversal with hand-computable truth, and on the real fluid traversal
against the authored oracle (exact, 0.0 difference).

**It has already produced the routing answer it was built for:** the fluid
traversal's SSA matches the authored mathematics exactly while the LLVM
artifact does not, so the outstanding defect is in **emission** — after a
long search of the planner and the AST/SymPy inlet on the assumption it
was upstream.

Two things that made it work, both worth copying into any future
evaluator:

* the vocabulary is **derived** from `ssa_llvm_backend`'s likeness table
  and audited at import, so it cannot grow a private opinion of what an
  opcode means;
* inputs are bound **by declared identity**, and a parameter with no
  accounting is found through the callee formal it feeds, by name — the
  rule the compiled runtime binder already uses. Binding by dtype-and-rank
  was tried, hit an ambiguity, fell through to a scratch default of 0.0,
  and made a working compiler look broken for a full session.

Original design notes retained below, since they still describe the shape:

* Scope it to the vocabulary the programs actually use; a shortfall on an
  unhandled op is fine and honest, and mirrors how the backends already
  report.
* It is a *reference*, so favour obvious over fast: no fixed points, no
  caching, no clever ordering.
* Feed it the same inputs the ABI binder feeds the artifact, so a
  disagreement is about semantics and not about marshalling.

This is the single highest-value remaining piece: it splits "lowering
changed the meaning" from "emission changed the meaning", which is the
question every hard defect in this tree has ultimately turned on.

---

## Phase 3c — ssa ↔ backend

Once 3b exists this is nearly free: run the SSA evaluator and the artifact
on identical inputs and compare every value that is observable in both.
The watch mechanism already makes internal SSA values observable in the
artifact without perturbing it (`watch=`, `history=`), so the comparison
can reach intermediates rather than only endpoints.

---

## Phase 4 — hooks at the inlet, not just at the ends

The comparisons above are end-to-end per stage. To localise *within* a
stage, the program itself has to be instrumented — and the honest place is
where a value still has an authored identity: the **AST / ProcessGraph
inlet**, where special-case and schema-replacement machinery conforms
nodes.

Design constraints, learned the hard way:

* **Doctor a copy, never the program under test.** Instrumenting the real
  AST shifts value ids and rebinds consumers — that has already produced
  two false readings here. Build an annotated *clone*, translate the clone,
  and compare it against the untouched original to prove the annotation
  changed nothing.
* **A hook records identity, not values.** Emit the cheapest true thing —
  an integer site id — and resolve it later through `trace_manifest`, which
  already joins `dependency` ↔ `dual_ir` ↔ `ssa` and carries authored
  `names`. The runtime must not carry strings.
* **Absent unless asked for.** Same contract as `trace_manifest` and
  `watch=`: with instrumentation off, the emitted artifact is unchanged.

This is where a tensor-intrinsic misrecognition would become visible: a
whole-array operation that arrives downstream already shaped as a scalar,
with every later stage faithfully preserving the mistake.

---

## Phase 5 — generalise past this program

Everything above is currently written against the fluid program. The
generalisation is a small protocol, not a framework:

```python
class TranslationSubject:
    def authored_reference(self, inputs): ...   # the oracle
    def lowered_module(self): ...               # repository SSA
    def artifact(self, *, watch=()): ...        # a compiled backend
    def bind(self, inputs): ...                 # ABI marshalling
    def observables(self, result): ...          # {name: value}
```

Any program that can answer those five questions gets all four comparisons
for free. The fluid program becomes the first implementation rather than
the only one.

---

## Phase 6 — telemetry as the transport

`shell_telemetry.TelemetryChannel` already carries `log`/`error`/`profile`/
`progress`/`trace` in one ordered stream, from Python at build time and
JavaScript at run time. Differential results belong on it as `trace`
records so a run's comparison and its compilation appear on one timeline.

`trace_manifest` is the join table and is currently built only on the
`aot_compile` path with `trace=True`. Binding it to the canonical
`lower_ast_source_to_ssa` path is what lets a watch or a divergence report
say `mass_error` instead of `value 141` at whichever level the reader
wants.

---

## Order of work, and why

1. ~~**3b** (python ↔ ssa)~~ — **done**, and it routed the outstanding
   defect to emission.
2. **3c** (ssa ↔ backend) — now nearly free: the evaluator and the watch
   mechanism can be compared per value, which turns "emission is wrong"
   into "this instruction is wrong".
3. **6** (manifest on the canonical path) — makes every report speak in
   authored names.
4. **4** (inlet hooks) — needed for defects that are *inside* one stage.
   Note that the fluid defect turned out NOT to be here, so this is no
   longer urgent.
5. **5** (protocol) — do this when a second program needs it, not before.

Phases 3a and 3b are built. **Route the layer before reading code**: two
of the longest hunts in this tree were spent in a layer that turned out to
be innocent, and both would have been redirected in minutes by this.

---

## Live defect (found by the matrix, not yet fixed)

`viscosity` and `tracer_diffusivity` reach the compiled step as a constant
**1.0**, whatever the state holds. Setting them to 0.5/0.25 changes
nothing — both still arrive as 1.0, against authored values of 0.0002 and
0.0001. The diffusion terms are amplified by 5000x and 10000x.

The signature is what identified it, and it is worth copying: `height_next`
matched the oracle EXACTLY in all 16 cells while both momenta and the
tracer differed in all 16. `height_next` is the one equation using neither
parameter. A defect that spares one output completely is naming the input
it does not touch.

Everything around it was verified faithful first, which is why the
remaining suspect list was short enough to see:

* the step's lowering — all 11 equations against SymPy at machine
  precision (0 or 2.2e-16);
* the traversal — `tracer_center` reproduces the initial tracer row-major,
  and all four neighbour gathers are the correct wrapped shifts;
* emission — ssa vs llvm is 0 across every next_* field;
* the oracle's own binding — the authored source passes its 28 arguments
  in `argument_names` order, position for position (checked with `ast`,
  after a naive comma-split "disproved" it by choking on `[row, column]`).

Four confident hypotheses died on the way: a permuted call site, six
unwired formals, a misbinding oracle, and misgathered height neighbours.
Each was killed by measurement, and three of them by a script after an
eyeball reading had already "confirmed" them.

**Mechanism (corrected).** It is NOT a folded constant. Neither value is
a `Const`: both are array GATHERS.

    viscosity          <- GetElementPtr [54, row, (col + 1) mod n]
    tracer_diffusivity <- GetElementPtr [45, row, col]

They read 1.0 because they gather HEIGHT, which starts at ~1.0 -- not
because a literal was baked in. Checking the defining instruction rather
than the observed value is what separated those two stories, and the
observed value alone had already suggested the wrong one.

The call's 28 arguments split into 8 scalar parameters and 20 neighbour
gathers. Six scalars -- coriolis, dt, dx, gravity, linear_drag,
minimum_height -- arrive correctly, as FORMALS of `advance`: hoisted,
unresolved parameters, exactly as they should be. The failing two are not
formals at all; their slots are fed from other regions' aggregate outputs
(region_2 output 6, region_1 output 22), both height gathers.

The discriminator is position. In the authored call, the six that work sit
at group BOUNDARIES, while `tracer_diffusivity` sits INSIDE the tracer
gather run (between `tracer[row, column]` and `tracer[row, east]`) and
`viscosity` sits immediately after that run ends. So a contiguous gather
run is claiming argument slots by position and swallowing a scalar
interleaved into it -- which is the same disease as every other bug in
this file: a positional assumption over a set whose order is incidental.
Alphabetical argument ordering is what puts a scalar inside a gather run.

**Next:** find the pass that builds the advance-to-step argument list. Both parameters are
state scalars that are NOT function formals of `advance` (the binder
reports them unbound and they have no defining instruction in the frame),
so the substitution is upstream of SSA — in the ProcessGraph/planner
capture of state scalars, which is where a constant would be folded in.
Note `minimum_height` and `dx` come through correctly, so it is not all
state scalars.

Correction to the commit that restricted the SSA column: it justified the
restriction by calling the `state.tracer = state.next_tracer + 0.0`
write-back the runtime wrapper's job. It is not — it is the last line of
the authored `symbolic_fluid_advance`. The restriction is still right, but
for a different reason: the evaluator mutates arrays through Store and
cannot REBIND a Python attribute, so that assignment is invisible to it.

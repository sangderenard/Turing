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

## Phase 3b — python ↔ ssa

The missing half of the routing decision. Needs a **reference evaluator for
repository SSA**: walk `Function.blocks`, honour `Phi`/`Br`/`CondBr`, and
execute the instruction vocabulary against NumPy values.

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

1. **3b** (python ↔ ssa) — unlocks the routing decision; everything else is
   easier once the layer is known.
2. **3c** (ssa ↔ backend) — nearly free after 3b, and closes the lattice.
3. **6** (manifest on the canonical path) — makes every report speak in
   authored names.
4. **4** (inlet hooks) — needed for defects that are *inside* one stage.
5. **5** (protocol) — do this when a second program needs it, not before.

Phase 3a is built. The next defect in this tree should be routed with it
before anyone reads code.

# Scope-qualified value names: audit

**Question.** Can an SSA value id carry its home scope in its name, so
`317` stops meaning one thing in `symbolic_fluid_step` and another in
`symbolic_fluid_advance`, without turning values into objects with
properties and wrecking the procedural guarantees SSA exists to provide?

**Verdict: yes, and it is not OOP — provided it is addressing syntax, not
a property on values.** The distinction is sharp and worth stating
precisely, because the wrong version of this would be genuinely corrosive.

---

## Why the problem is real

It is not hypothetical; it cost real time this session.

* `watch=` refuses region-local ids with "region-local ids are a different
  numbering space", so an internal value of `planned_region_0` is simply
  not addressable. Bisecting it required routing through the smallest
  caller that had a `Ret`.
* `317` is `tracer_violation` in the step and an unrelated GEP in the
  advance. Reading a trace without tracking which frame you are in
  silently pairs the wrong two things.
* `95` is `viscosity`'s slot in the advance AND a region-local value in
  `planned_region_1`. That collision produced a confident wrong reading
  during this very investigation, corrected only by re-deriving it.

An id is meaningful only relative to a function. That is fine as a
*representation* and bad as an *interface*.

---

## The line that must not be crossed

SSA's procedural guarantee rests on three properties:

1. every value is defined exactly once, textually, in one function;
2. uses are dominated by definitions — a static, structural fact;
3. no hidden mutable state and no runtime name resolution.

The OOP failure mode is not the dot. It is **search**. If `t317` were
resolved by walking enclosing scopes until something matched, you would
have dynamic scoping: property lookup with a fallback chain, defeating
(1) and making (2) unverifiable. Inheritance of names, overriding, or a
mutable scope bag attached to each value would all be the same mistake.

A dot that is **static, total, and never searched** is none of that. It
is what a linker symbol is: `module.symbol`, resolved at build time,
failing loudly when absent. Procedural languages have had this since
translation units existed, and it does not make C object-oriented.

So the three rules:

* **Static** — assigned at construction from the function being built,
  never computed, never reassigned.
* **Total** — every value has exactly one home scope. No defaults, no
  "current scope" ambient state, which would be the same silent-default
  disease as everything else in `TRANSLATION_DEBUGGING.md`.
* **No search** — `advance.t317` resolves in `advance` or it is an error.
  Never fall back to an enclosing frame. This single rule is what keeps
  the property procedural.

---

## What NOT to build

**Do not add a `scope` field to `SSAValue`.** It buys nothing
semantically: ids are already unambiguous *given* their function, and the
IR is correct as it stands. It would touch every construction site, break
every cached lowering, and put identity in two places at once — the field
and the owning function's table — which is a synchronization problem
waiting to disagree. The defect this would supposedly fix has never been
in the compiler; it has been in tools and in readers conflating frames.

**Do not put it in `accounting`.** That dict is mutable and free-form.
Identity that anything can overwrite is not identity.

---

## What to build instead

Addressing syntax at the tool boundary, resolved against tables that
already exist — `value_names`, `named_outputs`, `parameter_names`,
`argument_names` — plus the dotted conventions the tree already uses
(`storage_identity` is `record.field.part`; `authored_source_name` is
`parameter.field`; symbols are `class.method`). This invents no
convention; it extends one.

```
advance.t317            # value 317 in the advance's frame
advance.max_tracer_violation
planned_region_0.t83    # today: not addressable at all
step.tracer_next
```

Two pieces, both small:

* `resolve(module, "scope.name") -> (function_name, value_id)`, erroring
  on an unknown scope, an unknown name, or an id absent from that frame.
  No search, no fallback.
* `render(function, value) -> "scope.tname"` for every tool that prints an
  id, so a pasted identifier round-trips.

Then `watch=`, `history=`, `bisect_emission`, `correlate_compile` and
`symbol_provenance` all take qualified names, and the region-local
restriction becomes a resolution question rather than a refusal.

**Verification it stays honest:** a checker that asserts no tool pairs
ids across frames without qualifying them, and that `render` then
`resolve` is the identity. Neither requires the IR to change.

---

## Priority

Below finishing the native fluid executable. It is a debugging-ergonomics
change, and it is a real one — it removes an entire class of misreading
that has produced wrong conclusions three times in this tree — but the
compiler is not wrong for lacking it. Do it when the executable lands, or
when the next cross-frame confusion costs an hour, whichever comes first.

---

## Postscript: the same confusion, one layer up

While auditing this, the graph layer produced the mirror-image problem.
Building a ProcessGraph from the authored source directly — outside the
compiler's loop lowering — collapses the entire `for row / for column`
nest into ONE `opaque_python` node. Fifty nodes total, no `getattr` nodes
for the state scalars at all.

So a graph built that way is not the graph the compiler plans over, and a
predicate tested against it proves nothing about the real one. Same
lesson as the SSA ids: a representation is only meaningful together with
the context that produced it, and "I built the graph" is as incomplete a
statement as "value 317".

Worth recording because the standalone build LOOKS authoritative — it
parses the real source, runs the real builder, and returns a real graph.

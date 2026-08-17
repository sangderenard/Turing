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

---

## Blocker for the native executable (found, not fixed)

Two defects, in the order they must be fixed.

### 1. ProcessGraph node ids are Python object addresses

`ProcessGraph.ensure_node` (graph_express2.py:2614) does `nid = id(node)`.
That address becomes the graph node key, flows into `value_id`, and then
into SSA ids through `next_physical_id = 1 + max(node ids)`
(fortran_c_shell.py:4053). Measured on the fluid lowering: 153 of
`symbolic_fluid_frame`'s 167 value ids are above 1e9, consecutive from
1457923920625 -- squarely in the CPython id() range.

* **Non-deterministic.** Addresses vary per run, so the emitted Fortran
  varies per run. This is an independent reason not to trust a cached
  lowering.
* **Non-monotonic.** Ids carry no program order, and the frame's space is
  disjoint from every callee's -- the same seam the argument mismatch
  below sits on.
* **Address recycling is a correctness hazard.** CPython reuses an
  address after a free. If a node is collected and another allocated at
  that address, `if nid in self.G` is True for a DIFFERENT node and
  silently aliases them.

Fix without changing the dedup intent: keep an identity map keyed by
`id(node)` for deduplication, but map it to a MONOTONIC counter that
becomes the public nid, and retain a reference to the keyed object so its
address cannot be recycled while the map is alive. Deduplicate by
identity; do not let the address BE the identity.

Do this first. While ids are non-deterministic, any conclusion about the
call site below can shift between runs.

### 2. Fortran call sites pass literals in the wrong type

    tools/build_fluid_c_shell.py build/<lowering>/control_repository_ssa.pkl
    Error: Type mismatch in argument 't419' at (1);
           passed INTEGER(4) to REAL(8)

Read out of the generated shell: the callee declares
`real(c_double), intent(inout) :: t419`, and the call site passes integer
literals positionally -- `..., t10, 256, t1457923920650, ...` and
`..., t1457923920647, 0, ...`. gfortran is right to reject it. A constant
argument is emitted in its own natural type rather than in the callee
formal's declared type.

Verified pre-existing: the identical error appears when building from a
lowering made before this session's predicate/dtype work, so none of that
caused or masked it.

### Investigation notes: why monotonic ids collapse the program

Branch `id-identity-sweep` holds the attempt. What is established:

* the fix does what it is for -- two consecutive lowerings produce
  IDENTICAL value ids in every function, where address-based ids differed
  per run. Max SSA id falls from 1.45e12 to 27.
* `ensure_node` deduplication is UNCHANGED: both builds log exactly 355
  `already_defined=True` and 2488 `already_defined=False`. So the collapse
  is not extra merging at graph construction.
* the collapse is in "reducing source topology". Post-planning node counts
  fall `step_with_dt_control_used` 578 -> 87, `symbolic_fluid_advance`
  178 -> 57, `run_superstep` 70 -> 11, and `callee_ref` counts go to zero
  with them -- so shells stop being attributed and every region lands in
  one unnamed shell.
* `shell ?` is NORMAL, not a symptom: the good build shows it too, for the
  outermost shell. The real signal is that recursion into named callsite
  shells (`run_superstep`, `step_with_dt_control_used`, ...) stops.

**There are at least three numbering authorities**, which is the root of
the whole class of problem:

1. `ensure_node`/`new_identity` -- was `id(obj)`, now a monotonic mint;
2. the SymPy ingester, which numbers its own nodes `0..N`;
3. the reducer, which relabels every node to canonical `0..N` through
   `nx.relabel_nodes`, ordered by
   `(lineno, col_offset, type, int(node_id))` -- with the raw node id as
   the final tiebreaker.

Authority 1 colliding with authority 2 is already confirmed to cause real
damage: minting from 1 gave a synthesised Store node the id of an
expression node, the Store overwrote it, and a graph with no cycles
acquired a Store->Store cycle. Basing the mint at 2**40 fixed that
instance and is why the mint is based rather than starting at zero.

Authority 3 is the open lead. Because the reducer canonicalises ids
anyway, the ORIGINAL values should not matter downstream -- so the fact
that changing them changes what survives means the reduction is ordering
by raw node id, and the ordering decides the canonical mapping. Dense
sequential ids sort differently from sparse addresses, and equal keys
before the tiebreaker now resolve differently.

Next: instrument node counts immediately before and after the relabel in
`topological_reducer`, on both branches, and compare `ordered`. If the
reduction is dropping nodes rather than merely renaming them, the drop
site is between those two points.

Also worth fixing regardless: the build reported `"completed": true` for a
module that had fallen from 45 functions to 2. Nothing checks the
structure of its own output, so a catastrophic regression reports success.
That check would have caught this in one run instead of several.

### Design decision: one node-id authority

Multiple authorities is the defect class, not any one of its symptoms.
The rule for this system is one authority, and the invariant that makes an
authority correct is already demonstrated in the tree:

    # symbolic_process_graph.py -- the SymPy ingester
    next_id = max(
        (int(node_id) for node_id in graph.G if isinstance(node_id, int)),
        default=-1,
    ) + 1

It SEEDS ABOVE every id already in the graph. That single line is why it
has never collided with anything, and it is the property the other
producers lack:

* `ensure_node` used `id(obj)`, which never seeded. It avoided collision
  only because an address is astronomically large -- luck, not design;
* minting from 1 collided immediately and let a Store node overwrite an
  expression node, inventing a cycle in an acyclic graph;
* basing the mint at 2**40 avoided collision the same way `id()` did, by
  magnitude. Also luck, just more deliberate luck.

**The authority.** The GRAPH owns node identity, because the graph is the
thing ids have to be unique within:

    ProcessGraph.mint_node_id() -> int
        Monotonic. Seeded lazily to 1 + max(existing int node ids), so it
        is correct even when a graph arrives already populated by another
        producer or by a relabel.

    ProcessGraph.identity_for(obj) -> int
        mint_node_id(), memoised per object, retaining the object so its
        address cannot be recycled under the memo. Replaces id(obj) for
        deduplication: same object, same node, without the address being
        the identity.

**Producers to migrate**, all of which currently mint for themselves:

1. `graph_express2.ensure_node` -- `nid = id(node)`
2. `graph_express2` store nodes -- `id(f"Store[...]")`, the address of a
   temporary, which is the worst of them
3. `graph_express2` domain nodes -- `id(domain_node)`
4. `symbolic_process_graph` -- local `next_id`, already correct; change is
   to ask the graph rather than keep its own counter
5. `topological_reducer` canonical relabel -- assigns 0..N through
   `nx.relabel_nodes`. This one is a deliberate RE-AUTHORING of every id
   at once, which is legitimate, but afterwards the graph's counter must
   be reseeded above the new maximum or the next mint collides with a
   canonical id.

Point 5 is the open lead for the `id-identity-sweep` branch: node counts
collapse during reduction, and the reducer orders by
`(lineno, col_offset, type, int(node_id))` with the raw id as tiebreaker.
Because it canonicalises ids anyway the original values should not matter,
so the fact that changing them changes what survives says the ordering is
load-bearing.

**Order of work.** Land the authority and migrate 1-4 first, verifying at
each step that the module still lowers 45 functions. Then 5, which is the
one that actually changes what the reducer sees.

### Troubleshooting the built executable: what is established

The whole-program executable builds and runs. `--trace` now compiles the
launch digest in, and five frames report cleanly:

    {"trace":{"records":5,"lost":0,"launches":[
      {"seq":0,"shell_ns":740401,...,"status":1}, ...]}}

Ruled OUT by measurement, each of which looked like the answer:

* **"it failed"** -- `status:1` is SUCCESS. The shell's own accounting
  proves it: `if (!status) { stats->failures += 1; }`.
* **"inputs never arrived"** -- `initial-state.bin` decodes exactly right:
  `[0:2]` frame_duration 0.0333 and dt_initial 0.001, `[2:1026]` height
  min 1.0 max 1.12 (the wave bump), `[1026:2050]` momentum_x +/-0.0097,
  `[3074:4098]` tracer up to 0.78 (the dye blob).
* **"outputs read the wrong slots"** -- `state.height` loads into
  `slots[2]` and is read back from `slots[2]`, with the identical
  transposed index `((i/32)%32 + (i%32)*32)`. And `4100 = 4*1024 + 4`
  matches the output file exactly.
* **"the state arrays are intent(out), so the inputs are discarded"** --
  they are `intent(inout)`, 2-D. 5 arrays in, 35 inout, 343 scalars by
  value, 1 scalar out. The compute CAN write them back.

So the Fortran runs successfully, receives correct inputs through
correctly-bound slots that it is allowed to write, and every state output
comes back zero while `t14` alone is written.

Two leads, both concrete:

1. **Extents are guessed from generated names.** `build_fluid_c_shell`
   sizes workspace extents with a heuristic: group extent names by their
   `_N` suffix, and if a family has more than one dimension give each
   `grid`, otherwise `grid*grid`. That is a rule about NAME SHAPE, not a
   fact read from the api contract, and it decides how every array in the
   program is sized.

2. **36 of the 40 array dummies in the entry are named with
   address-derived ids** (`t2147983015473`). The node-identity defect is
   not cosmetic and not confined to diagnostics: it names most of the
   arrays in the program's ABI.

### Correction: the "366 vs 367" arity gap was a measurement bug, not a fact

The line above this originally read: *"The 366 actuals vs 367 formals gap
is in this same subroutine."* That is **false**, and the record is
corrected here rather than silently edited, because a wrong claim left
standing is worse than one admitted.

The 367 count came from splitting a Fortran declaration line on commas
with a naive `line[line.index('('):line.rindex(')')]`. `rindex(')')`
finds the **last** `)` on the line, which is inside the trailing
`bind(C, name="...")` clause, not the argument list's own closing paren
-- so the split silently captured `name="..."` fragments as if they were
two extra formal names. Read correctly (splitting on `) bind(C` instead),
`run_superstep__specialized_47985f25ff49` declares **366** formals, and
the call site at `symbolic_fluid_frame` passes **366** actuals. They
agree exactly. There is no arity gap in this subroutine.

Lesson for this file specifically: **never hand-parse Fortran text for a
structural fact the api contract already states.** Every measurement
below this point uses the `.api.yaml` contract or a real parser
(`tools/trace_fortran_alias.py`), not comma-splitting.

### Extent binding (lead 1) is fine for the arrays checked

Comparing the contract's declared extents for `t16..t19`
(`state.height`/`momentum_x`/`momentum_y`/`tracer`) against what the
frame's call into `run_superstep` actually passes: both sides agree,
`32, 32` for every one of the four arrays, matching the real grid. Lead 1
is not the cause, at least not for these four.

### The real finding: `state.height` is never written, anywhere reachable

Built `tools/trace_fortran_alias.py` (see the decision tree, Q5e) to
follow one named buffer's identity through the whole call graph
mechanically instead of by eye. Tracing `state.height` from
`symbolic_fluid_frame`:

    frame  t16           intent(inout)  no local write
      -> run_superstep                  t390   intent(inout)  no local write
        -> step_with_dt_control_used    t678   intent(inout)  no local write
          -> symbolic_fluid_advance     t54    intent(in)     no local write
            -> planned_region_1         t54    intent(in)     no local write
               (no further forwarding call found under this token)

Five hops deep, `state.height` is never assigned to anywhere in the
traced graph. Tracing `state.next_height` instead, from the same root,
finds a genuine write:

    frame -> run_superstep -> step_with_dt_control_used -> advance
      -> planned_region_3   t108   intent(inout)   WRITES here

So the compiled program **does** compute a new height field correctly
into its own buffer (`next_height`, C-shell slot 6) -- but nothing in the
traced graph ever copies that result into the persistent `height` buffer
(slot 2) the caller reads back. That much is consistent with the defect
this session has already named twice in other layers: `state.height =
state.next_height` in the authored Python REBINDS a Python attribute,
which is free in Python and has no native equivalent unless the compiler
specifically lowers it to an in-place array copy. See the standing memory
note: *"No final fused reduction for multi-function programs — fusing
kills state navigability; keep the SSA/control path."* This may be the
same disease in a new place.

**But that alone does not explain the observed value.** If `height`
(slot 2) is genuinely never written, it should retain whatever
`initial-state.bin` loaded (~1.0-1.12, confirmed present in the file and
confirmed loaded into slot 2 by direct reading of the C source's `fread`
block). The observed value is not "unchanged" -- it is **exactly 0.0** in
every cell, in the raw `final-outputs.bin` bytes, not merely in a
summary printf. Ruled out while chasing this:

* **slot-index collision** -- `state.height` is C-shell slot 2,
  `state.next_height` is slot 6, separate `calloc` calls, confirmed by
  direct grep of the allocation lines. Not the same memory.
* **a feedback-swap touching it** -- the only pointer swap in the whole
  generated `main()` is `slots[1] <-> slots[155]` (the scalar dt
  feedback). Slot 2 never appears in a swap.
* **a whole-array zero broadcast on this token** -- grepped every bare
  `tN = 0.0_c_double` statement in the file (9 of them); none target
  `t16` or any alias on the traced chain (`t390`, `t678`, `t54`).

### `state.height` arrives at `advance` under nine names -- checked, all correct

`trace_fortran_alias.py`'s first version only followed the FIRST
occurrence of a token within one call's argument list (`.index()`
instead of every match), which would have missed exactly this case:
`state.height` is passed into `symbolic_fluid_advance` NINE separate
times in one call (the five spatial neighbour views, `t54, t56, t58,
t60, t45, t52, t122, t23, t27`), all in the SAME `call` statement. Fixed
to report and follow every position a token occupies, then re-run.

Result: **all nine are correctly bound to the same real buffer**, traced
all the way from `t16` through `t390`, `t678`, and out to all nine
`advance` positions and their respective `planned_region` callees, with
the RIGHT declared intent at every hop (`intent(in)` for the five
read-only neighbour views, `intent(inout)` for the two that also feed a
write, `no local write` confirmed at every one of them for `height`
specifically). This was the most concrete open lead and it is now
**disproven**, not merely unverified -- there is no cross-view binding
error here.

So, cumulatively: `state.height` is proven never written anywhere in the
five-level call graph reachable from `symbolic_fluid_frame`, through
EVERY path that graph contains, with EVERY occurrence checked. And
`state.next_height` -- the buffer that legitimately IS computed correctly
-- lives in a separate, non-aliased, correctly-sized buffer that the
output writer never reads. Both of those are now facts, not hypotheses.

What is NOT yet established: why the observed value is exactly 0.0
rather than the ~1.0 the never-written buffer should retain from
`initial-state.bin`. Ruled out for this: slot-index collision with
`next_height` (separate `calloc`s, confirmed), the one pointer swap in
`main()` (targets slot 1/155 only, a scalar, confirmed by grep), a
second allocation of slot 2 anywhere in the file (grepped every
`slots[2]` occurrence in the generated C -- exactly one `calloc`, one
`fread` loop, one summary read, one final-output read; no other
assignment to the pointer), and a whole-array zero broadcast targeting
this token (grepped every bare `tN = 0.0_c_double`; none match the
traced chain).

Remaining, in order of cost to test:

1. **There is no prior WORKING whole-program build to diff against.**
   The executable could not be compiled at all before this session's
   Fortran fixes (predicate typing, discard-slot dtype, literal dtype
   threading) -- it failed at the `t419` type-mismatch error every time.
   So "did this session's edits cause it" cannot be answered by rebuilding
   an older lowering; there is nothing to compare it to. This defect may
   be exposed by this session's fixes rather than caused by them.
2. **Run under a bounds/memory checker** (gfortran `-fcheck=bounds` on a
   rebuild, or a C-side sanitizer on the shell) rather than continuing to
   read source by hand. At least two input slots decoded as
   `-1.24e+123` earlier this session (uninitialised bytes reaching
   `initial-state.bin` from an unresolved source), which is independent
   evidence that some buffer in this program is sized or initialised
   incorrectly -- a live, checkable hypothesis rather than another guess.
3. **Check the accept/reject/retry commit path specifically.** The
   authored `run_superstep` accepts a step and commits it, or rejects and
   restores a snapshot; the earlier LLVM-path investigation this session
   already established that EVERY attempt gets rejected (the
   `viscosity`/`tracer_diffusivity` defect drives the tracer out of
   bounds at every dt). If the compiled Fortran's rollback path writes
   from an unloaded/zero-initialised snapshot buffer rather than
   preserving the caller's original array, that is consistent with
   everything measured here and does not require a new binding error --
   only that "restore" and "leave untouched" were not the same operation
   in this lowering. `trace_fortran_alias.py --token` can follow whatever
   the "saved"/snapshot buffer is called once it is identified in the
   authored source.

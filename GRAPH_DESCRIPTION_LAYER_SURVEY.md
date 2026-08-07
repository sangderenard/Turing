# The graph-only description layer: survey and disambiguation

**Status: written mid-investigation, may be wrong in places.** This
document exists to separate three things that got tangled together in one
debugging session: throwaway prototypes, real-but-narrow internal special
cases, and the actual sanctioned compile path. Treat every claim here as
"true as of this reading of the code," not as a guarantee -- verify before
relying on any of it for a real change, and correct this document when you
find it wrong rather than leaving the wrong claim in place.

## The question that started this

Between `ProcessGraph` reduction (`topological_reducer.py`) and SSA/backend
emission, is there one coherent, language-neutral, graph-only description of
a program -- dependency structure, scope/ownership, control flow -- or does
each backend re-derive its own partial view ad hoc?

Short answer: **the coherent description already exists, in three separate
real structures, computed by the real compile path, but never assembled
into one object.** Nothing here needed inventing. It needed finding.

## The three real structures

All three are built directly from a reduced `ProcessGraph` (`graph.G`, a
`networkx` graph with `role=`-labeled edges) and are language-neutral at
this layer -- they know nothing about Python syntax, only about the graph
`ProcessGraph.build_from_ast` (or any other frontend) already produced.

### 1. Dependency structure -- `MapDependencyRegions` / `ShellReferenceTables`

`src/compiler/shell_reference_tables.py`

- `build_map_dependency_regions(graph, entrypoint)` -- a real transitive
  call-reachability closure. Starting from the entrypoint's function-table
  reference, it walks every node's `callee_ref` attribute across the whole
  function table to discover the complete set of functions actually
  reachable (`.runtime`), plus map-level retention (`.mapped`, `.retained`,
  `.map_only`, `.bindings`). This *is* the dependency DAG -- already
  computed, already whole-program, not something to build.
- `build_shell_reference_tables(graph)` -- returns a `ShellReferenceTables`
  with `functions`, `constants`, `memory` (`ShellMemoryReference`, each with
  a `base_node_id` correlating an `ast.Attribute` node back to whatever
  graph node computed its receiver, via the same `parents`/`role` edges
  used everywhere else in this codebase), and `correlations` tracing every
  table slot back to its `ProcessGraph` origin node.

**This is the piece a hand-rolled fix in this session (`_resolve_reference_node`
in `glsl_deployment_strategy.py`, added while chasing a missing-`self`
compile error) duplicated instead of using.** `ShellMemoryReference.base_node_id`
is already the sanctioned way to resolve an Attribute's receiver. The ad hoc
fix walked `identity_table` by hand to answer the same question a real table
already answers. Worth revisiting: does using `reference_tables.memory`
directly fix the missing-`self` case the ad hoc fix did not?

### 2. Scope / ownership -- `ClassNavigationTable`

`src/compiler/shell_reference_tables.py`, `build_class_navigation_table(graph)`

Binds class/object identities discovered in `graph.G.graph["map_ir"]["objects"]`
to the function table: which methods and attributes belong to which class,
with permissions and constructor references. This is the "scope/map graph"
-- real, already built, already keyed off the graph rather than off Python
AST directly.

A narrower, session-local prototype (`scope_graph_prototype.py`, in the
scratch directory, not committed) tried to build something in this spirit
from raw closure/`nonlocal`/`global` tracking. It is a reasonable
standalone exercise in seeing whether the *shape* generalizes across
languages (it does, partially -- see the "prototype" section below) but it
duplicates work `ClassNavigationTable` already does for the one language
(Python) that currently has a real frontend. It should not be mistaken for
the sanctioned mechanism.

### 3. Control flow -- `ControlProgram`

`src/compiler/control_source.py`

> "Target-neutral compiled-shell control structure... The planner owns
> control flow. Backends render this structure; they must not rediscover
> loops, reorder scheduled regions, or substitute a host-language
> coordinator after planning has selected a compiled target."

Loops, branches, and calls as `LoopBlock`/`ControlBlock`/`CallBlock`/etc,
referencing numeric regions by `__scheduled_region_N__` markers rather than
inlining them. Consumed for real by `emit_wasm_control_coordinator`
(`wasm_class_coordinator.py:545`), which is itself wired into the live
page-publishing path (`site_bundle.py:2070`), not a dormant alternate.

This is also the answer to a standing question from earlier in this
session (paraphrased: "does Wasm force a final numeric reduction of the
whole program, or can it hold real control flow?"). No -- `FusedProgram` is
deliberately scoped to the *numeric interior between control points*;
`ControlProgram` is the real structured control around it, and both are
live, connected mechanisms, not one being a stand-in for the other.

## Where all three are computed, and where they currently end up

All three are computed during a normal `compile_ast_aot` call:

- `dependency_regions` and `class_navigation`: `aot_compile.py:656` and
  `:666`, inside the frontend phase.
- `reference_tables`: `glsl_deployment_strategy.py:11814`, inside
  `strategize_shell_deployment` -- "the compilation choke point: every
  backend -- c, python, glsl, fortran, webgl, webgpu -- funnels its
  ProcessGraph through this one control-planning stage."
- `ControlProgram`: produced by the same planning stage, already returned
  as `AOTCompilation.shell_control_program` / `DualIRShell.shell_control_program`.

But `dependency_regions` and `class_navigation` do **not** survive as
typed objects on the compilation result. `class_navigation` is stuffed,
as the real dataclass instance, into the untyped `map_ir["class_navigation"]`
dict entry. `dependency_regions` is worse -- it gets *flattened* into plain
fields (`map_ir["dependency_regions"] = {"runtime": ..., "mapped": ...,
...}`) and the real `MapDependencyRegions` object is discarded. Only
`reference_tables` reaches something structured and persistent
(`self.reference_tables` on every compiled shell instance,
`glsl_deployment_strategy.py:12005`).

`DualIRShell` (`dual_ir_shell.py`) is the closest existing candidate for
"one object that owns all of these" -- it already pairs
`compiled_shell_program` (`FusedProgram`, numeric) with
`shell_control_program` (`ControlProgram`, control flow) and a raw `map_ir`
mapping. It does not yet carry `ClassNavigationTable`, `MapDependencyRegions`,
or `ShellReferenceTables` as first-class typed fields -- only their
flattened or buried remains inside `map_ir`.

## Prototype work from this session (not sanctioned, not committed to `src/`)

Two scratch scripts explored whether a from-scratch, language-neutral
closure/scope graph was feasible, independent of the real
`ClassNavigationTable`:

- `scope_graph_prototype.py` -- walks Python AST, builds an
  `nx.MultiDiGraph` of scopes with `role="parent"/"captures"/"mutates"`
  edges, using a monotonic node-id counter (not raw `id()`) matching the
  rationale behind `tensor_identity` in `abstraction.py`. Works for real
  capture/mutation resolution including explicit `global`/`nonlocal`
  (after a bug fix -- the first version silently dropped both).
- `lang_neutral_scope_test.py` -- a narrow, regex-based C++ closure
  extractor (lambda capture lists only, matching the same "controlled
  frontend" scope as the existing scalar-C `pycparser` route in
  `machine_code_lifting.py::c_function_token_multigraph`), tested against
  independently-written (not hand-matched) Python and C++ snippets.

**Explicit caution, restated because it matters:** the two extractors
produced structurally identical output (`role="mutates"` edges) for
analogous Python `global`/C++ lambda-`&capture` patterns, but the
*confidence* behind each edge differs sharply. Python's came from the real
AST (`ast.Global` is unambiguous). The C++ lambda capture list is also a
real, explicit, parsed signal (not a guess). But the same extractor's
handling of an *ordinary* C++ function reaching a global with no
declaration at all is a **regex heuristic** -- it would miss mutation
through a reference alias (`int& r = x; r += 1;`), miss anything behind
`this->`, and could false-positive on a shadowed local. Two graphs having
the same shape is not evidence the underlying signal is equally trustworthy
on both sides. This distinction is easy to lose once results are rendered
as "the same kind of edge" -- watch for it before trusting either
extractor for anything beyond this exploratory comparison.

Neither prototype is wired into anything real. They exist only in the
session scratch directory. If this direction is pursued further, the
question is not "build a scope graph" -- it's "does `ClassNavigationTable`
already cover what a real closure graph would need, and if not, what
exactly is missing" -- not a parallel implementation.

## No real C / C++ ingestion exists

`DREAM_LANGUAGE_TRANSLATIONS` (`dream_document.py:46`) has entries for
`python`, `sympy`, `javascript`, `glsl`, `wgsl`. Nothing for `c` or `cpp`.
A dream block declaring either falls to `_source_section_graph` --
structural bookkeeping only, `executable=False`. The one real C-parsing
code in the repo, `c_function_token_multigraph`
(`machine_code_lifting.py:1589`, pycparser-based), is explicitly scoped by
its own docstring to "parameters, binary expressions, constants, and one
return statement" -- no control flow, no pointers, no structs. It exists to
round-trip tiny scalar leaf functions against disassembled machine code,
not to ingest general C. `pycparser` cannot parse C++ at all (templates,
overloading are out of grammar, not just untested).

## `DreamDocument.lower_to_ssa()` does not lower dream block internals

Every dream block, regardless of language, becomes exactly one opaque
`Call` instruction to a per-language host stub (`__dream_python_host__`,
`__dream_cpp_host__`, ...) in the produced `IRModule`. `compile_sections()`
(same file) *does* do real per-block work for Python -- a genuine
`ProcessGraph` via `reduce_abstract_tensor_topology` -- but that result is
never connected forward into `lower_to_ssa()`. They are two disconnected
mechanisms today: one parses for real and discards the result structurally
(keeps only shortfalls/metadata), the other builds real SSA but only ever
emits a black-box call per block.

## Recommended path (as best understood right now)

For any block/program, regardless of source language:

```
source --(language frontend: ProcessGraph.build_from_ast for python,
           lower_glsl_source_to_ssa for glsl,
           _javascript_dependency_graph for javascript,
           ingest_sympy_expression for sympy,
           nothing real yet for c/cpp)-->
ProcessGraph (topological_reducer.reduce_abstract_tensor_topology)
  --> MapDependencyRegions      (dependency structure)
  --> ClassNavigationTable      (scope/ownership)
  --> ShellReferenceTables      (reference/memory correlation)
  --> ControlProgram            (control flow, planner-owned)
  --> FusedProgram              (numeric interior between control points)
--> SSA (precompile_to_ssa.lower_precompile_and_control_to_ssa)
--> backend emission (wasm / glsl / fortran / ...)
```

`DualIRShell` should be the one object carrying all of the graph-only
description forward (`FusedProgram` + `ControlProgram` it already has;
`MapDependencyRegions` + `ClassNavigationTable` + `ShellReferenceTables` it
does not yet carry as typed fields). Converging on that is the concrete
next step this document was written to set up, tracked separately from
this survey.

## Two convergence layers, not one, and where each is real

Later investigation (same session) found this codebase actually has *two*
distinct places programs converge, not one:

1. **`ProcessGraph` + `role_schemas`** (`operator_defs.py`) -- Python AST and
   SymPy expressions converge here today, string-dispatched on
   `type(node).__name__`. `role_schemas` is a flat dict (node-type-name →
   `{'up': {child_attr: arity}, 'down': {...}}`); `node_special_cases.py`
   (`transmogrifier/graph/node_special_cases.py`) is the switch block for
   collapsing recognized patterns before the generic schema walk. Real,
   proven for two languages, upstream of dependency/scope/control-flow
   analysis.
2. **SSA (`IRModule`)** -- many backends (C, Fortran, LLVM, GLSL, WGSL)
   already converge here. GLSL→WGSL, for instance, is not a dedicated
   translator; it's `lower_glsl_source_to_ssa` (GLSL → SSA) and
   `ssa_webgpu_backend.py` (SSA → WGSL) sharing the same `IRModule`.

**These two layers are not symmetric for reaching `DualIRShell`.**
`ssa_primitive_lowering.lower_ssa_to_fused_program` is a real, existing path
from generic SSA to `DualIRShell`'s numeric half (`FusedProgram`). Nothing
equivalent exists for the control half: every real construction site of
`ControlProgram` (`glsl_deployment_strategy.py`'s deployment planner,
`state_machine_ast.py`, `hierarchical_control.py`, `loop_composer.py`)
builds it from `ProcessGraph`, never from SSA directly. So "procedural
languages go through SSA, OOP languages go through ProcessGraph" is not
fully realizable as a clean split today -- a procedural language's numeric
work could reach `DualIRShell` via SSA, but its control flow still has
nowhere to go except through `ProcessGraph` and the deployment planner,
same as everything else.

**Deferred, not pursued now:** it is possible to go the other direction --
recognize SSA's own control-lowering vocabulary (the same special-command
shapes `ControlProgram`'s planner emits when it lowers `ProcessGraph`
control into SSA via `precompile_to_ssa.py`) and translate *that* back into
a synthetic `ProcessGraph`, effectively decompiling already-lowered SSA.
This is plausible and would likely be the fastest route if this project
ever needs to decompile procedural code or raw binaries from their SSA/IR
form rather than from source. It is a real additional parser, not a small
addition, and is explicitly out of scope right now -- we are not
decompiling procedurals or binaries at present. Noted here so it isn't
reinvented or forgotten later.

**Current direction:** OOP languages (C++, JavaScript, Java, ...) target
`ProcessGraph` via `role_schemas`, following the working SymPy precedent,
not SSA. See `src/transmogrifier/graph/oop_language_translations.py` for
the translation-table file this produced, and its own docstring for
per-language status (what's registered and verified vs. what's still a
todo).

## Open questions this document does not answer

- Does routing the missing-`self` receiver resolution through
  `ShellReferenceTables.memory` (instead of the ad hoc `identity_table`
  walk added this session) actually fix that compile? Not yet verified.
- Does `ClassNavigationTable` already capture everything a real
  closure/capture graph would need (nested function ownership, captured
  names, escape analysis), or is there a genuine gap the prototype found
  (container-mutation escapes, e.g. `_live_machines[key] = machine`) that
  isn't covered by any existing structure?
- What would a real (non-regex) C or C++ frontend actually require --
  `libclang`, `tree-sitter`, or a hand-written recursive-descent parser
  scoped the same way `c_function_token_multigraph` is scoped, just wider?

## Noted, not yet scoped: PRNG, time calls, struct lowering

Flagged during the C++ shell work, deliberately not chased down mid-task:
PRNG node acceptance, time-related calls, and `struct` all need to be
lowerable all the way to `DualIRShell`/SSA soon. Not investigated yet --
whether any of these already have partial support (e.g. `struct` clearly
exists at the `ProcessGraph`/C-shell level now, but whether it survives
lowering to `FusedProgram`/SSA is unchecked) is an open question for
whoever picks this up next.

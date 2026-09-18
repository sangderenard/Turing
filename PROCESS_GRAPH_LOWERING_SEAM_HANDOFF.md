# Joining language ingestion at ProcessGraph: handoff, not yet implemented

**Status: plan only. No code changes described here have been made.**
Written instead of doing the refactor live, deliberately -- this touches
`compile_ast_aot`, the ~860-line function that (as of this session) finally
compiles the reversible-machine-executor `tick()` entrypoint correctly after
a long investigation (see `GRAPH_DESCRIPTION_LAYER_SURVEY.md`). That fix is
hard-won and currently verified working; this refactor should be done
carefully, with a fresh read of the current code, not assumed from this
document's summary of it.

## The architectural rule this establishes

**There is to be exactly one point where a graph-only program description
(`ProcessGraph`) gets lowered into backend-facing form (SSA / `FusedProgram`
/ `ControlProgram` / `DualIRShell`).** Every language's ingestion path --
Python's `ast.parse` + `ProcessGraph.build_from_ast` today, the C++-like
shell's `desugar_cpp_shell` + `pycparser` + `role_schemas` built this
session, any future language -- must converge on *that one point* rather
than each carrying its own copy of the downstream pipeline
(`reduce_abstract_tensor_topology`, `build_map_dependency_regions`,
`build_class_navigation_table`, `strategize_shell_deployment`,
`capture_fused_programs`, ...).

This is a **higher → lower** rule specifically:

- **Lateral moves stay allowed wherever they already happen**, outside this
  one seam. Procedural-to-procedural transformation at the same level (SSA
  rewriting, `ssa_primitive_lowering.lower_ssa_to_fused_program`) and
  binary-to-binary transformation at the same level (relifting,
  disassembly-to-disassembly normalization) are not what this rule
  restricts.
- **Raising (lower → higher) is explicitly out of scope for this document.**
  `GRAPH_DESCRIPTION_LAYER_SURVEY.md` already notes the deferred
  SSA→`ProcessGraph` decompilation idea; that stays deferred, and is marked
  there as "a more philosophical topological question" than an engineering
  task to schedule alongside this one. Do not conflate the two -- this
  document is about the one-way, higher-to-lower convergence point only.

## Why this is a smaller change than it sounds

Reading `compile_ast_aot` end to end (this session, not assumed): only the
very first phase is Python-specific --

```
ast.parse(source) -> ProcessGraph.build_from_ast(module) -> entrypoint node id
```

Everything after that already operates on `graph` generically; nothing
downstream re-touches Python syntax. That means the extraction is a genuine
"split at the seam that already exists conceptually," not an invention of a
new abstraction.

## The concrete split (traced this session, not yet cut)

In `src/common/tensors/accelerator_backends/aot_compile.py`,
`compile_ast_aot` spans roughly lines 432-1292 (the last definition in the
file). The natural split point is right where `graph` (a built
`ProcessGraph`) and `entrypoint_node_id` (the entrypoint's own AST/graph
node identity -- see below) are both finalized, before
`reduce_abstract_tensor_topology(graph)` first runs (~line 677).

Proposed shape:

```python
def _lower_process_graph_to_compilation(
    graph, entrypoint_node_id, entrypoint, feeds, *,
    backend, remove_loops, unroll_limit, profiling, precompile_only,
    expanded_python_bindings, bake_mode, schedule_preference,
    mutable_parameters, progress, checkpoint_store, checkpoint_feeds,
    frontend_implementation, source_graph_implementation,
    planning_implementation, capture_implementation,
    deployment, frontend_ready, class_navigation, dependency_regions,
    map_ir, resume, source,  # source: checkpoint-identity/error-message use only, never re-parsed
) -> AOTCompilation:
    ...  # everything currently from "if not frontend_ready:" (~line 677)
    ...  # through the function's existing return, moved verbatim
```

`compile_ast_aot` becomes a thin Python-specific wrapper: keep the existing
checkpoint-setup and digest-computation prologue (lines ~468-563) and the
existing three-tier checkpoint-resume/fresh-build logic (lines ~564-676)
**unchanged** -- that logic already produces `graph` and (per this
session's fix) `entrypoint_node_id` correctly for Python -- then delegate to
the shared function instead of continuing inline.

A new `compile_cpp_shell_aot(source, entrypoint, feeds, ...)` (or similar
name) would do the analogous C++-specific ingestion --
`desugar_cpp_shell(source)` → `pycparser.CParser().parse(...)` →
`install_c_role_schemas(graph)` → `graph.build_graph(tree)` → find the
entrypoint's own `FuncDef` node (pycparser, not `ast.FunctionDef`) → same
`id()` capture pattern used for Python -- then call the *same* shared
`_lower_process_graph_to_compilation`.

## What must be passed by the caller, not re-derived (the actual contract)

The entrypoint fix this session added (`FunctionTable.reference_by_source_node`,
`src/transmogrifier/function_table.py`) is what makes this joinable at all:
resolving the entrypoint by **node identity**, not by name, is what lets a
`pycparser` `FuncDef` node and a Python `ast.FunctionDef` node share one
resolution mechanism without caring what language either came from. The
ingestion contract each language-specific wrapper must satisfy:

1. A built `ProcessGraph` (`graph`), already reduced through whatever
   frontend passes it needs (`reduce_abstract_tensor_topology` and its
   siblings currently happen *inside* the shared function, not before it --
   keep that; do not make each ingestion wrapper reduce its own graph).
2. `entrypoint_node_id`: `id()` of the entrypoint's own function-definition
   node in *that specific parse* -- found unambiguously by the caller (one
   `FunctionDef`/`FuncDef` matching the entrypoint name in that language's
   own freshly-parsed top-level source), never derived by name lookup
   against a function table that may already hold other same-named
   functions.

## Local variables that cross the split (catalogued this session, verify against current code before cutting)

`graph`, `entrypoint_node_id`, `entrypoint`, `feeds`, `source` (identity/
error-message use only), `backend`, `remove_loops`, `unroll_limit`,
`profiling`, `precompile_only`, `expanded_python_bindings`, `bake_mode`,
`schedule_preference`, `mutable_parameters`, `progress`/`_report`,
`checkpoint_store`, `checkpoint_feeds`, `frontend_implementation`,
`source_graph_implementation`, `planning_implementation`,
`capture_implementation`, `deployment`, `frontend_ready`, `class_navigation`,
`dependency_regions`, `map_ir`, `resume`. `retain` and `constant_map` are
used only in the ingestion phase (Python's `_apply_parameter_constant_map`,
`graph.build_from_ast(..., retain=retain, ...)`) and do **not** need to
cross into the shared function.

This list was built by reading the function's first ~830 lines this
session; the remaining ~460 lines (hierarchy recomposition, output
extraction, control-shortfall diagnostics, the final `AOTCompilation`
construction) were not re-read line by line for this document -- re-verify
that tail uses no additional Python-AST-specific state before assuming this
catalogue is complete.

## Verification plan (non-negotiable given what's at stake)

Before considering this refactor done:

1. `tests/test_abstract_tensor_topological_reducer.py`,
   `tests/test_dual_ir_shell.py`, `tests/test_class_navigation_slots.py`,
   `tests/test_shell_archive.py`, `tests/test_function_table.py` all still
   pass (the baseline this session established; one pre-existing unrelated
   failure in `test_function_table.py`,
   `test_process_graph_functions_assemble_static_definitions_with_kwargs`,
   is not this refactor's concern).
2. **The real regression test that actually matters**: re-run the full
   `binary_machine_tick.tick()` compile end to end
   (`compile_ast_aot(source, "tick", feeds, python_bindings=bindings,
   precompile_only=True)`, matching this session's verified working case)
   and confirm it still returns `control_shortfalls: ()` with no traceback.
   This takes ~200-230 seconds for real (whole-program discovery trace);
   budget for it, do not skip it, and do not assume the refactor is safe
   from unit tests alone -- this exact function is what the whole session's
   investigation was about.
3. Only after both pass: build the C++-shell wrapper
   (`compile_cpp_shell_aot` or similar) against the shared function and
   verify a real, simple C++ class (the `Counter` example already used in
   `tests/test_cpp_shell_desugar.py`/`tests/test_dream_cpp_pipeline.py`)
   reaches a real `DualIRShell`, the same way `tests/test_class_navigation_slots.py`
   verified this for Python.

## Where the code-comment version of the architectural rule should land

Once implemented, the shared function's own docstring is the right place
for the "exactly one higher-to-lower point" rule stated above -- not scattered
across each language wrapper. Cross-reference
`GRAPH_DESCRIPTION_LAYER_SURVEY.md`'s "two convergence layers" section and
its noted-but-deferred SSA→`ProcessGraph` raising question from there,
rather than restating either.

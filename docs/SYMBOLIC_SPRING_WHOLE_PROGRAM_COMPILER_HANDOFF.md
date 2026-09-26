# Symbolic spring whole-program compiler handoff

**Snapshot:** 2026-08-05 08:02 America/Chicago  
**Status:** active, not complete  
**Primary entrypoint:** `src/rendering/symbolic_spring_image.py::run_symbolic_spring_image`

## The required architecture

The application is one real Python function. The compiler must begin with that
function's AST and retain the complete program:

```text
expression_text (runtime string)
  -> sympy.sympify(..., evaluate=False)
  -> ProcessGraph.build_from_expression(...)
  -> symbolically_reduce_process_graph(...)
  -> load_fluxspring_graph_shaders()
  -> run_precompiled_graph(..., duration=math.inf, shader_sources=...)
```

The goal is not to extract a selected numeric subgraph and wrap it in a custom
driver. It is to improve the general compiler until this whole function, its
calls, classes, retained control, shader compilation, physics, and renderer are
represented in the normal ProcessGraph/hierarchy/card/SSA path. The
`expression_text` parameter must remain a public runtime input; compiling one
sample expression into constants is a failed result.

Do not add a second AST control interpreter. The existing path already owns
control semantics:

```text
Python AST -> ProcessGraph -> loop/branch planner -> PlanClosure/PlanCall
           -> ControlProgram + captured cards -> SSA -> backend
```

## Concrete source locations

- `src/rendering/symbolic_spring_image.py` — sole application entrypoint.
- `src/compiler/symbolic_process_graph.py` — symbolic ProcessGraph reduction.
- `src/compiler/glsl_deployment_strategy.py` — existing whole-program planning,
  capture, hierarchy composition, and shader-region deployment.
- `src/compiler/hierarchical_control.py` — composes per-closure
  `ControlProgram`s and inserts planned calls into their lexical loop scopes.
- `src/compiler/loop_composer.py` — existing loop analysis and control
  generation.
- `src/compiler/precompile_to_ssa.py` — retained control/card to SSA lowering.
- `src/compiler/ssa_fortran_backend.py` and `src/compiler/fortran_c_shell.py` —
  Fortran emission and host shell.
- `src/compiler/parametric_card_program.py` — concurrent multi-card work; it is
  not owned by this session and must be preserved.
- `src/compiler/shader_extractor.py` — source-level discovery of shader compile
  sites.
- `src/rendering/opengl_render/fluxspring_shader.py` — loads FluxSpring's live
  shader literals through the extractor.

## Completed work that is still relevant

### Mutable runtime parameter and checkpoint semantics

`compile_ast_aot` and planner specialization propagation now distinguish a
mutable public parameter from a discovery sample:

- mutable parameters are excluded from planner specialization;
- checkpoint identity uses ABI facts for mutable feeds instead of their sample
  values;
- a declared mutable parameter that disappears from the executable public ABI
  raises `RuntimeError` instead of being reported as a generic compilation;
- SymPy's `_global_parameters` singleton has an explicit checkpoint reducer;
- captured-program checkpoints avoid repeating long capture phases when source,
  compiler implementation, and ABI identity are unchanged.

The key regression tests are in `tests/test_aot_checkpoint.py`,
`tests/test_loop_composer.py`, and `tests/test_process_graph_shell.py`.

### Projected resident iterables

The normal `ControlProgram` ABI now carries projected iterable bindings for
`enumerate` and destructured resident rows. Those bindings propagate through
hierarchical composition, SSA, Fortran extent/index lowering, and GLSL row
access. This was added to the existing loop/control representation, not as a
new execution mechanism.

### Shader compile-site discovery, first tranche

The checked-in extractor already recognizes literal
`compileShader(source, GL_*_SHADER)` calls and finds the actual vertex and
fragment shaders used by `spring_async_toy.LiveVizGLPoints`. The whole-program
entrypoint calls `load_fluxspring_graph_shaders`; the source of truth remains
the live renderer rather than a copied shader string.

## Work that was deliberately undone

A separate `structural_ast_ssa.py` / `structural_program.py` path was briefly
started. That duplicated control-flow semantics already present in the
ProcessGraph compiler and violated the architecture above. It was fully
removed, including its tests and integrations. Do not restore it or build an
equivalent parallel interpreter.

## Checkpoint evidence

### Old baked sample — diagnostic only

```text
.turing-cache/accelerator_backends/aot-checkpoints/
800c06c5a54cd927c6577c69e95413d1c17799383baab9cfcb357f6bee769327/
```

This compiled a dynamic Fortran DLL for the discovery expression
`((x+x)+0)*1`, but the expression was baked into the captured trace. It is not
evidence for a general runtime expression compiler.

### Mutable/general checkpoint identity

```text
.turing-cache/accelerator_backends/aot-checkpoints/
4bf0715a2f7fe214ffebff6dd719b57accd9097e45e715a9f698e162a17a2c6a/
```

At the snapshot:

| Phase | Bytes | Time |
|---|---:|---|
| `frontend.pkl` | 24,698,061 | 2026-08-05 07:04 |
| `compiled_plan.pkl` | 640,547,324 | 2026-08-05 07:35 |
| `captured_program.pkl` | 233,525 | 2026-08-05 07:37 |

Two different discovery strings reused this identity because
`expression_text` is mutable and keyed by its ABI, not its value. That part is
correct. The saved captured artifact itself is not a generic success: its
public input map was empty.

## Exact false-success diagnosis

The generic run reported:

```text
aot: hierarchy recomposition skipped
(ValueError: planned calls reference enclosing loops absent from closure control: (161, 182))
```

Then AOT fell back to a seven-region numerical shell. The exported artifact
had:

```text
mutable_parameters = ('expression_text',)
function_parameters = ('expression_text',)
public_input_value_ids = {}
public_output_value_ids = {}
root region indices = (1, 2, 4, 5, 8, 9, 10)
```

Zero reported control shortfalls was therefore a false success. The current
mutable-parameter invariant should reject this artifact rather than allowing
it to masquerade as general.

The relevant insertion invariant is in
`hierarchical_control.compose_hierarchical_control`: a pending `PlanCall` whose
`enclosing_loop_ids` ends in `161` or `182` could not find a matching
`LoopBlock(induction='iteration_<id>')` in that closure control. Diagnose this
through the existing hierarchy and loop planner only.

## Concurrent loop-version work: do not overlap

Another live agent owns a related retained-loop correction in
`glsl_deployment_strategy.py` and `tests/test_process_graph_shell.py`. Its
cause and intended invariant are concrete:

- hierarchy identity reduction had treated a loop backedge version like an
  ordinary storage alias;
- `(updated, initial)` collapsed to `(initial, initial)`;
- the numerical body still produced `updated`, but SSA was asked to find a
  body producer for `initial`;
- the fix keeps updated and initial as distinct SSA versions while arena policy
  may still assign them the same resident storage over time.

At 07:53 the active diff added explicit protection for loop-carried update
endpoints during hierarchy identity resolution. A focused nested retained-loop
test verifies distinct backedge IDs and no `loop_carried` SSA shortfall. The
agent reported a full unreduced 4x4 viscosity/pressure capture in progress.

Wait for that run's result before editing the same alias/hierarchy seam. The
missing-loop-placement error may be separate, but should be reevaluated against
the corrected full capture rather than guessed at concurrently.

## Shader extractor state at final handoff

After the initial handoff, wrapper-flow extraction was connected to the public
`extract_shader_compile_calls()` function. It summarizes only recognized
shader API dataflow through helpers:

```text
compileShader(source, stage)
glCreateShader(stage) -> glShaderSource(handle, source)
                      -> glCompileShader(handle)
```

It models literal concatenation, tuple destructuring, fallback alternatives,
nested helpers, nested shader calls such as
`compileProgram(compileShader(...))`, and OpenGL stage constants that were
temporarily initialized to `None` before lazy import.

Verified behavior before the final documentation stop:

- actual FluxSpring `compileShader` vertex and fragment sites are extracted;
- raw `glCreateShader` / `glShaderSource` / `glCompileShader` helpers are
  followed through nested wrappers;
- unresolved runtime shader sources are not guessed;
- the canonical raw OpenGL renderer exports its six real `MESH_*`, `LINE_*`,
  and `POINT_*` shader literals;
- `tests/test_shader_extractor.py` passed 5 tests in 5.38 seconds.

A final, unverified edit then added `ExtractedShaderBundle`, manifest mapping,
and recursive `discover_shader_compile_calls(root)`. The user ordered feature
work to stop immediately after that patch. Therefore the bundle/discovery API
has **not** been tested and is the first continuation verification boundary.
Do not report it as green until the focused suite is rerun.

A read-only repository scan made immediately before the bundle patch found:

| Host source | Extracted shaders |
|---|---:|
| `spring_async_toy.py` | 2 |
| `opengl_render/renderer.py` | 6 |
| `inspiration/run_opengl_demo.py` | 7 |
| `inspiration/particles.py` | 4 |
| `common/double_buffer/base.py` | 0 (dynamic inputs only) |
| `quad_buffer.py` / `tribuffer.py` | 0 (generated/dynamic source) |
| `glsl_backend.py` | 0 (generated/dynamic source) |

Zero for a dynamic source host is expected under the fail-closed rule; it does
not mean the compiler call was absent.

After extraction, reuse the existing `turing.shader-component.v1` component
ABI documented in the experience report
`1785870479_DOC_Shader_Component_ABI_And_External_Link_SSA.md`. Avoid inventing
another shader registry or host harness.

## Verification snapshot

Most recent completed small verification before the final untested bundle edit:

```text
py -3.11 -m pytest -q tests/test_shader_extractor.py
5 passed in 5.38s
```

The repository `.venv` currently points to a removed Python 3.10 executable.
No environment repair or package installation was performed. Python 3.11 via
the Windows launcher had the existing test dependencies.

Earlier focused projected-iterable and mutable-parameter tests were green. No
full suite was run during this shared-worktree session.

## Safe continuation order

1. Receive and record the concurrent full 4x4 loop-version run result.
2. Re-run the symbolic spring compile from the newest valid checkpoint without
   changing the runtime expression between phases.
3. Require `expression_text` to appear in the whole executable public ABI.
4. If loop IDs 161/182 are still absent, compare each `PlanCall`'s closure and
   `enclosing_loop_ids` with that same closure's existing `ControlProgram`.
   Correct ownership/namespacing in the planner or hierarchy composer; do not
   interpret Python control again.
5. Make whole-program hierarchy recomposition fail closed when no valid prior
   hierarchical artifact exists. Do not silently select a numerical child
   shell after a required hierarchy fails.
6. Include hierarchy-building helpers in captured-checkpoint implementation
   fingerprints so a helper-only semantic change cannot resume a stale
   captured artifact.
7. Only after the runtime input and whole hierarchy survive, lower the normal
   control/card artifact to SSA and compile Fortran. Test a second expression
   with the same binary or ABI to prove it is not baked.
8. First run `py -3.11 -m pytest -q tests/test_shader_extractor.py` to verify the
   final bundle/discovery patch. Then connect extracted stages to the existing
   shader component ABI and select GLSL/WGSL/native raster deployment at the
   normal backend boundary.

## Shared-worktree cautions

- Two other agents are active in this repository.
- Do not reset, checkout, stage, or rewrite unrelated dirty files.
- Do not stop Python processes you did not start.
- `parametric_card_program.py`, `fortran_c_shell.py`, the retained-loop alias
  edits, and the large reversible-machine/system-port changes have concurrent
  owners.
- Check file modification times and focused diffs before every edit.

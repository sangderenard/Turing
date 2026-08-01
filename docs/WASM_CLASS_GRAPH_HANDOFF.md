# WASM class-module segmentation and process-graph shell handoff

Written 2026-08-01. State of the class-graph work, what is finished, what is
not, and the design corrections that got it here.

## The rule

**Runtime segmentation is not an OOP presentation of the source.** A
compiler reduces first; by the time a program is a flat, reduced
`FusedProgram`, its own AST-level function boundaries are gone (a call is
fully inlined). So the module boundaries this work creates are a graph cut
of the *reduced* program, named mechanically (`chunk0`, `chunk1`, ...), not
a rediscovery of `helper`/`kernel`. A meaningful, class-named presentation
of the same program is a separate, later concern for a shell to plan and
display -- it can draw on `hierarchical_plan.PlanClosure` for that -- but it
does not get to decide where the actual module boundaries fall. This was
the single biggest wrong turn during this work: an earlier version tried to
segment by `PlanClosure`/`PlanCall` (pre-reduction, AST-shaped) and had to
reverse-engineer ambiguous value provenance to make it work. It was
replaced outright.

## What is finished

**Segmentation.** `src/compiler/wasm_class_modules.py`'s
`partition_reduced_program(program, chunk_size, owner_name)` cuts a reduced
`FusedProgram`'s topological order into contiguous, roughly-equal-sized
chunks, reusing `process_graph_fusion.py`'s existing
`fused_program_to_process_graph`/`DispatchRegion`/
`dispatch_region_to_fused_program` for the boundary accounting -- a chunk's
`input_ids`/`outputs` *are* its calling contract, mechanically, no separate
provenance-reconstruction pass. The chunk holding the program's own declared
outputs is the root, named after `owner_name` (the real AST entrypoint, e.g.
`compile_ast_aot`'s own `entrypoint` string) since the whole graph is owned
by whatever was submitted as the entry point.

**Scheduling.** `build_module_process_graph`/`schedule_module_levels` build
a real `transmogrifier.graph.graph_express2.ProcessGraph` (the same class
`glsl_deployment_strategy.py` builds while compiling, with its own real
`ILPScheduler` attached) over the emitted class modules and read real
ASAP levels off it -- not a hand-rolled topological sort. Levels are shifted
so the root sits at 0 and every prerequisite sits behind it at a negative
level (`-1`, `-2`, ...), per the same "level is not a claim about being
first at absolute zero" reasoning.

**Emission and linking.** `wasm_binary.py` gained real WASM import-section
support (`WasmImport`, function + memory imports, `CodeBuilder.call`) --
proven end to end with an actual Node.js cross-module `call` instruction
crossing a real module boundary (`test_wasm_binary.py`). `emit_class_modules`
can wire a dependency either way: `link_calls=True` for real shared-memory
linkage, or `link_calls=False` for independently instantiable modules that a
host-side runner carries values between.

**Two runtimes for the same manifest shape.**
`wasm-gallery/shared/process_graph_runner.js` is a small FIFO/queue
scheduler for a page that fetches modules as separate files: it only ever
needs to know "what module is next" plus a table of edges -- no global
graph knowledge, no JS-side topological sort. `wasm_html_shell.py` gained
the same algorithm ported for a self-contained page (`ClassGraphRunner`,
instantiating from embedded base64 instead of `fetch`), wired in additively
behind a new `class_graph=` parameter that defaults to `None` and changes
nothing for every existing caller, `build_homepage.py` included.

**Whole-kernel API.** `describe_process_graph_api`/`build_embedded_class_graph`
build a real `CompiledProgramAPI` (the same format every other backend
already emits) for the *whole* segmented kernel -- its real external inputs,
resolved back to actual source-parameter names via
`program.extras["capture_feed_origins"]`, and the root chunk's own declared
outputs -- so `wasm_html_shell.emit_html_shell` renders it with zero new UI
code. Verified live in the browser: the existing, unmodified shell UI,
correct input row (`a`), correct output (`result_0`), a real click on Run
driving the segmented runtime, correct answer.

**A real, non-toy stress test surfaced one real bug**, now fixed:
`fused_program_to_process_graph` (`process_graph_fusion.py`) had no case for
a standalone `tensor_from_list` constant step (only the *inline*
`right_scalar`-folded case) -- `canonical_elementwise_op` doesn't know it
because `operator_catalog.py` classifies it as a `CREATION_OPERATOR`, not
elementwise. Fixed by reusing `fused_program_wasm_backend._constant_scalar`
(the exact same one-element-list extraction the WASM backend already uses)
to turn it into a `"const"` node directly, same as the inline case already
does. Covered by two new tests in `test_process_graph_fusion.py`.

## What is not finished

**A `link_calls=True` caller's body still cannot emit the call.** The
import/export declaration is real and proven, but `FusedProgram`/`OpStep`
has no opcode meaning "invoke an imported class-module function" -- every
op is a pure per-element scalar computation. An earlier attempt at this (a
`ClassCallStep` splicing whole-array calls into the per-element loop, with
its own fixed-offset scratch-memory region) was explicitly rejected and
reverted: it invented per-module memory allocation for a downstream
consumer's benefit, which is exactly backwards from "runtime segmentation,
not OOP" -- a module should not have to know or reserve anything for who
might call it. The process-graph-runner approach (a host-side FIFO carrying
values between independently-memoried modules) is the answer that was
settled on instead; closing the `link_calls=True` invocation gap, if it's
wanted later, needs its own design pass, not a bolt-on.

**The full real Mandelbrot kernel does not fully segment yet.** Run through
`partition_reduced_program` end to end (3578 steps, real `tanh`/`cos`/`sqrt`,
559 constant-materialization steps) it hits a *different*, pre-existing,
already-documented WASM backend limit: a `tensor_from_list` step whose
`values` is not reducible to one scalar (a genuine multi-element array
constant) has no WASM-backend representation at all --
`fused_program_wasm_backend.py` already reports this as a `WasmShortfall`
("only a one-element constant can become an immediate; a real array
constant would have to be placed in linear memory") for the single-module
path; `fused_program_to_process_graph` now raises the same, honestly, for
the same reason, rather than silently mishandling it. This is not a new bug
to chase -- it is the existing single-module WASM backend's own accepted
boundary, now surfaced identically through segmentation. Closing it (baking
multi-element constants into a data segment, the same way the LUT tables
already are) is real, separate, scoped work if the full Mandelbrot kernel
specifically is wanted as a class-graph page.

**`chunk_size` is a plain integer, chosen by hand.** No heuristic yet for
"pick a chunk size that balances module count against per-module overhead"
-- a caller states it directly (`two-class-demo`/`kernel-shell` use `1` on
purpose, to guarantee a real multi-module split on a tiny program).

## Where the working examples live

`wasm-gallery/` (new, at the `C:\dev\Powershell` root, **not** tracked by
any git repo yet -- the root `nogodsnomasters` repo ignores every top-level
child directory (`.gitignore`: `/*/`), and this folder has no `.git` of its
own):

- `pages/two-class-demo/` -- `process_graph_runner.js`-driven, real
  compiled two-function source, fetched `.wasm` files.
- `pages/kernel-shell/` -- the existing `wasm_html_shell.py` UI driven by
  the segmented runtime instead of one module.
- `shared/process_graph_runner.js`, `shared/class_graph_loader.js` (the
  real-linkage-capable loader, used by the `test_wasm_binary.py`-proven
  path, not currently wired into a page).

Each has its own `build.py`; none of them touch `turing/build_homepage.py`
or `turing/index.html`, verified after every change in this session by
rebuilding the real homepage and diffing (identical modulo Python's
nondeterministic-`id()` reprs in embedded SSA/Fortran text, a pre-existing
property of `build_homepage.py` unrelated to any of this).

## Rebuild

```
cd wasm-gallery/pages/two-class-demo && python build.py
cd wasm-gallery/pages/kernel-shell && python build.py
cd wasm-gallery/pages/mandelbrot-graph && python build.py   # currently fails, see above
```

`python -m pytest tests/test_wasm_binary.py tests/test_wasm_class_modules.py
tests/test_process_graph_shell.py tests/test_process_graph_fusion.py
tests/test_wasm_html_shell.py` from `turing/` runs everything this handoff
covers.

# Wasm static parameters and collective deployment handoff

Status date: 2026-08-03

## Scope

This handoff covers the managed columnar multifluid page and the general AOT
contracts changed while diagnosing its repeated-entity rendering.  It does
not cover unrelated dirty files in the shared Turing worktree.

## Root cause

The first Web Worker implementation divided the 45,056-element invocation
into element tiles and ran the complete numerical graph inside every tile.
That is correct only for pointwise graphs.  The fluid graph contains
whole-invocation `sum` reductions for ant steering, visibility, and food
neighborhoods.  Evaluating those reductions independently in each tile
created one state consensus per tile, which appeared as repeated or
stride-like virtual entities.

The earlier 173.7 ms threaded versus 409.8 ms serial timing compared
incorrect tiled semantics against the correct serial semantics and must not
be cited as a valid speedup.

## Implemented contracts

### Extent effects

`fused_program_extent_effect` classifies fused regions as `pointwise` or
`collective`.  Reduction-bearing methods and the whole deployment record this
metadata.  The HTML shell permits complete element-tile deployment only when
the compiler-authored effect is `pointwise`.  A collective graph executes the
existing whole-extent Wasm coordinator with the original count.  This is a
safety fallback, not a JavaScript implementation of numerical work.

The actual artifacts remain WebAssembly binaries.  A published coordinator
and region were directly checked for the `00 61 73 6d 01 00 00 00` Wasm
magic/version header.

### Static versus mutable parameters

`compile_ast_aot` accepts a literal/tensor `constant_map` and a separate
`mutable_parameters` contract.  It rejects any overlap before replacing AST
parameter reads.  `SourceContract.mutable_parameters` is derived from the
input keys of `TURING_PAGE["state_feedback"]`; site discovery independently
rejects a static declaration for those feedback inputs.  Both maps are
recorded in the bundle manifest.

The fluid page constructs its static dictionary as every supplied feed not
named by `state_feedback`.  Its static entries are:

- `column_x`: complete 256 x 176 tensor literal;
- `column_y`: complete 256 x 176 tensor literal;
- `rest_surface`: complete 256 x 176 tensor literal;
- `dt`: scalar literal `0.025`;
- `audio_low`, `audio_mid`, `audio_high`, `audio_level`: scalar zero literals.

The 27 displacement/entity/food/ink/time feedback inputs remain mutable.
Adding a new feedback input automatically excludes it from the page's static
dictionary, and either contract layer rejects an explicit overlap.

## Runtime boundary

The generated page contains no Python runtime.  Authored Python is consumed
at build time and lowered through ProcessGraph/SSA into Wasm region modules
and a Wasm coordinator.  The JavaScript shell loads artifacts, manages memory
and presentation, and does not reimplement the fluid equations.  The WebGL
fragment shader is generated from `columnar_multifluid_present`.

## Verification completed

- 72 Wasm class/deployment/HTML-shell tests passed after extent-effect work.
- 23 site/AOT tests passed after the first mutable-contract work.
- A combined focused run reached 96 passing tests; its sole failure was a new
  test that incorrectly expected mutable `entity_x` in the constant map.  The
  assertion was corrected to require its absence.
- Three direct contract tests then passed, covering page classification,
  site overlap rejection, and AOT overlap rejection.
- A small AOT probe proved list-valued static parameters lower to
  `tensor_from_list` and disappear from the runtime feed set.

## Work in progress at handoff

A full precompile of the fluid source with the three 45,056-element static
tensors was still running when this handoff was requested.  No immutable
bundle containing these latest contracts has been published yet.  The page
`v1-297bc12f93c86250` predates the fix and must be treated as invalid for
state-layout inspection.

## Required continuation

1. Finish or rerun the full configured AOT compile and inspect:
   - static names equal the eight names listed above;
   - mutable names equal all state-feedback inputs;
   - runtime feeds contain only mutable inputs;
   - the fused program contains three `tensor_from_list` constructors and
     retains its `sum` reductions.
2. Build a fresh immutable managed-world bundle.
3. Confirm `bundle.json` records `extent_effect: "collective"`, the collective
   region list, the full constant map, and `mutable_parameters`.
4. Run the real browser probe.  It must report a non-black frame,
   `threadProfile.eligible == false`, and zero spread (within floating-point
   tolerance) across `next_entity_x` and `next_entity_y`.
5. Do not report a performance improvement until a semantics-preserving
   parallel collective implementation exists and passes the same state-spread
   check.

## Focused commands

```powershell
python -m pytest tests/test_site_bundle.py tests/test_process_graph_shell.py `
  tests/test_wasm_class_modules.py tests/test_wasm_class_coordinator.py `
  tests/test_wasm_html_shell.py -q

python -m pytest `
  tests/dt_system/test_columnar_multifluid_web_demo.py::test_compiler_generated_webgl_presents_non_black_wasm_output `
  -vv
```

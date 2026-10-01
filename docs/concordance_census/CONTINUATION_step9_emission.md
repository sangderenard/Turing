# Continuation: step 9 part B, emission as the last layer (lane E9)

Spec: `docs/concordance_census/100_plan_step9_graph_input_and_emission_output.md`
part B (section 4, proof of section 7).  2026-10-01.  Function names only; no
compiler numberings; nothing committed.  HEAD moved during the session
(another lane merged; the measured numbers below did not change).

## What changed (files / functions)

`src/compiler/concordance_declarations.py`, sub-block "Step 9 Part B" at the
end of the Step 9 section: stages `EMISSION_C`, `EMISSION_LLVM`,
`EMISSION_FORTRAN`, `EMISSION_WASM`, `EMISSION_JAVASCRIPT`, `ARTIFACT_BUILD`;
reasons `VALUE_WITHOUT_IDENTITY_CELL`, `NO_BOOK_AT_EMISSION`,
`BLOCK_ORIGIN_UNROUTED`, `FUNCTION_TEXT_PENDING`, `PIECE_ARTIFACT_UNROUTED`,
`WASM_REGION_UNROUTED`, `UNIT_ELIDED` (`NO_FUNCTION_SCOPE` reused from step
8); transform `NATIVE_LOOP_WRAPPER_VALUE` (1); enums `Backend`, `UnitKind`,
`ArtifactPart`; facts `EmittedUnit`, `FunctionEmission`, `ArtifactFact`;
pages `emission_unit` `(function, backend, unit)`, `emission_function`
`(function, backend)`, `emission_artifact` `(artifact, backend, part)`.

`src/compiler/emission_concordance.py` (new; imports no reducer):
`emission_book(module)` reads `module.metadata["identity_book"]` directly
(never `identity_book(module)` / `current_identity_book()`); a missing or
detached book posts nothing and says so once per process on stderr.
`value_cell` (`ssa_value` under `function_scope_of`, then
`control_value_binding`).  `EmissionRecorder`: `header` (posts
`emission_function` pending `Unresolved(FUNCTION_TEXT_PENDING)`, then the
FUNCTION_HEADER unit), `unit`, `elided` (`Unresolved(UNIT_ELIDED,
read=(binding unit,))`), `span(list)` (a cursor over an emitter list:
`open(instruction)` / `take` / `take_pending` / `close`), `finish` (REVISE
with unit count and sha256, DERIVED from every unit cell).
`post_artifact_part`, `ArtifactEmission.build` (SOURCE_FILE <- MODULE_TEXT,
PIECE_FILE `Unsourced(PIECE_ARTIFACT_UNROUTED)`, COMPILE_COMMAND <- files,
LIBRARY <- command; `variant="standalone"` suffixes the part labels).

`src/compiler/ssa_record_return_state.py`: `ssa_value_identity_cell(...,
*, book=None)` (a backend passes the attached book; default unchanged).

`src/compiler/ssa_c_backend.py`: `emit_ssa_function_to_c` (header, LITERAL
for Const / Pi, TABLE, STATEMENT, OUTPUT_STORE, closing RETURN; MODULE_TEXT);
`CFunctionArtifact.emission` + `compile`; `emit_ssa_module_to_c` (per
function header / PROTOTYPE / piece shim; per block BLOCK_LABEL; per
instruction one unit closed at the next loop top; Br / CondBr carve
PHI_EDGE_ASSIGNMENT units with `phi_edge_values` reading the same
`incoming_blocks`; Ret carves OUTPUT_STORE; aggregate projections are elided
rows reading the call's unit; hoisted Phi DECLARATION with the Phi's cell;
the public wrapper under the root's row; MODULE_TEXT and BUFFER_ORDER);
`CModuleArtifact.emission` + `compile` + `compile_standalone` (host file as
`(SOURCE_FILE, "host", "standalone")` DERIVED from BUFFER_ORDER).

`src/compiler/ssa_llvm_backend.py`: `_emit_repository_call_module` (same
span scheme; fused runs post one unit with every step's values, the other
steps elided rows reading it; frame allocas DECLARATION; frees one unit;
wrapper under the root's row; MODULE_TEXT, BUFFER_ORDER);
`LLVMFunctionArtifact.emission`; `compile_artifact`.

`tools/compiler_probes/probe_emission_chain.py` (new).

## Decisions taken here (say so if wrong)

- Unit / function rows are keyed by the function SYMBOL: a planned region
  carries its root's control scope, so `function_scope_of` would put two
  functions on one key.  Values are still read under `function_scope_of`.
- `emission_function` derives from the lowering's `cell_set (scope, 0)` root,
  posted at emission through `precompile_to_ssa._function_root_cell` when the
  lowering never named it (stage `control_ssa_entry`).  `FUNCTION_SCOPE` is
  not declared on this tree.
- A unit re-emitted with different text is a REVISE (CONCORD otherwise).
- `NO_BOOK_AT_EMISSION` is declared but never posted: posting it needs the
  ambient book the task forbids; the stderr line is the "once".

## Verified

- `probe_emission_chain`: 14 cases, failures 0.  C and LLVM text
  byte-identical with a detached book vs the real one; every
  `emission_function.unit_count` equals its unit rows; every C unit line is a
  source line and every unclaimed line is preamble; LIBRARY -> COMPILE_COMMAND
  -> SOURCE_FILE -> MODULE_TEXT -> one `emission_function` per function, both
  backends.
- Rows (plain = annotated), C units/functions/artifacts (unsourced: labels,
  values): bump 22/2/5 (2,2), chain 24/2/5 (2,4), twice 23/2/5 (2,3), cond
  38/2/5 (5,4), loop 47/2/5 (6,0), scale 323/4/5 (51,243), shared 365/5/5
  (55,269).  LLVM units/functions (unsourced): bump 21/2 (3,2), chain 25/2
  (3,4), twice 23/2 (3,3), cond 36/2 (6,4), loop 48/2 (7,0), scale 30/2
  (3,6), shared 42/2 (3,11); 5 more artifact rows each.
- bump chain (7 hops): RETURN unit of the root -> `ssa_value` (control scope,
  returned value) -> `canonical_value` (forked read scope) ->
  `canonical_value` (read scope) -> `ingestion_value` -> `source_span` kind
  BinOp; and unit -> `emission_function` -> `cell_set` root.
- `py_compile`; the seven listed probes exit 0;
  `probe_scalar_native_correctness` failures 0.  Audit first lines unchanged
  (view 0, toplevel 1, energy 0, controller 1, controller_untyped 5, mapping
  0, oscillator 0) and unsourced counts identical before/after (the audit does
  not emit).  `measure_completeness.py mapping controller` identical.

## Unsourced units, and why

- `block_origin_unrouted`: every BLOCK_LABEL; `ssa_block` (plan 100, 2.6) is
  not declared or posted.
- `value_without_identity_cell`: (a) planned-region literals (`Const` whose id
  is the literal's graph id) and the region results computed from them --
  no `ssa_value` / `control_value_binding` row in the root's control scope;
  this is why the bump `k + 1` statement itself is unsourced and the plan's
  operand hop (`name_binding` / `scalar_parameter`) is not reached (printed
  OPEN); (b) the tensor reference kernels (`binary_value`,
  `binary_scalar_double`: `tensor_ssa_lowering` mints, no scope) -- the bulk
  of scale / shared.

## Not done (next)

- LLVM: `emit_ssa_function_to_llvm`'s single-block lane (only the module
  lane is routed), `with_native_sgd_loop` / `with_native_adam_loop`
  (`NATIVE_LOOP_WRAPPER_VALUE` declared, unposted), kernel definitions /
  declarations pulled in by symbol (module text, no unit), `_annotate_noalias`
  rewrites define lines after their units (header text is pre-annotation;
  the function hash too).
- C: `deployment_support` / trampolines text has no units; local tensor and
  frame declarations derive from the function cell only.
- Fortran (`_FunctionEmitter`, `emit_module`, `write`, `compile_module`),
  WASM (`emit_wasm_module` + variants, `_assemble`, `write`, `compile_wat`;
  needs plan 100 section 6 item 7), JavaScript: not started.
- Audit `emitted-function-count`, `spelled-values`, `--completeness`;
  viewer `token:` focus; E9.18 second log (held 10.1); the
  `TEST_BASELINE_AND_HAZARDS.md` line; R9.7 (`_HostSSACachePickler`) not read.

## Exact next edit

Give planned-region literals an identity: where the region function's
`Const` for a literal graph node is built, post its `ssa_value` row (DERIVED
from the literal's `canonical_value` cell, as `_value_cell` adopts graph
ids); then bump's region `Add` unit derives and the operand hop closes.
Then `ssa_block` (plan 100, 2.6) for the labels.

## Working tree

Uncommitted, this lane: the five files above plus the new module, probe and
this note.  Other lanes' uncommitted edits share the tree
(`control_source.py`, `glsl_deployment_strategy.py`, `loop_composer.py`,
`precompile_to_ssa.py`, more of `ssa_record_return_state.py`).  Line endings
as found (CRLF; `ssa_llvm_backend.py` and the two new files LF).  Scratch:
`e9_audit_{before,after,final}.txt`, `e9_measure_{before,after,final}.txt`.

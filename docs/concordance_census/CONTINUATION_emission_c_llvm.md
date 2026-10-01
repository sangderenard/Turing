# Continuation: complete C and LLVM emission (lane EB)

Spec: `100_plan_step9_graph_input_and_emission_output.md` part B; design
sections 2, 6, 7, 8.  Follows `CONTINUATION_step9_emission.md`.  2026-10-01.
Function names only; no compiler numberings; nothing committed.  Lane ID
(`ssa_block`, planned-region `ssa_value` rows) landed in the same tree during
this session; the "after" numbers include it.

## What changed

`concordance_declarations.py`, sub-block "Step 9 Part B" only: stage
`LLVM_NOALIAS_ANNOTATION`; transform `AUTHORED_KERNEL_TEXT` (0); `UnitKind.
KERNEL_TEXT`; `ArtifactPart.KERNEL_SOURCE`; enum `NativeLoop`; fact
`NativeLoopValue`; page `native_loop_value` `(wrapper, value)`.

`emission_concordance.py`:
- `EmissionRecorder(key=, scope_cells=, local_values=)`, `open_function`,
  `unit(block=, block_cell=)`, `revise_unit`, `kernel_texts`, `texts`.
- `ssa_block_cell` (lane ID's `ssa_block_identity_cell`, always `book=`);
  `kernel_source_cell`, `piece_source_cell`, `call_demands`,
  `calling_units`, `imported_kernel_scope`, `is_imported_kernel`,
  `kernel_callers`; `NativeLoopEmission`.
- `post_artifact_part(module=, what=)`: with no book a MODULE_TEXT is kept on
  `module.metadata["emission_gaps"]`; `replay_emission_gaps(module)` posts
  each as `(artifact, backend, (MODULE_TEXT, "no_book_at_emission"))`
  `Unsourced(NO_BOOK_AT_EMISSION)` once a book is attached.
- `ArtifactEmission(root=, function_cell=)`.

`ssa_llvm_backend.py`:
- `_emit_repository_call_module`: labels pass `block=`; the wrapper's
  `entry:` derives from the root's function cell; kernel / intrinsic / piece
  text is one KERNEL_TEXT unit per symbol under the root's row; every
  function is finished AFTER `_annotate_noalias`, and each rewritten define
  line REVISEs its FUNCTION_HEADER unit under `llvm_noalias_annotation`,
  derived from the old cell and every unit that calls the function.
- `emit_ssa_function_to_llvm` single-block lane: routed (`LLVM_SCALAR`).
- `with_native_sgd_loop` / `with_native_adam_loop`: `NativeLoopEmission`.

`ssa_c_backend.py`: labels pass `block=`; imported kernels get the kernel
scope; both MODULE_TEXT posts pass `module=`.

`probe_emission_chain.py`: single-block lane, both wrappers, noalias
revisions, KERNEL_TEXT sources, LLVM token chain, gap replay.

## Decisions taken here (say so if wrong)

- An imported kernel (`binary_value`, ...; declared by `llvm_argument_names`,
  which only `import_llvm_to_repository_ssa` writes) has no lowering.  Its
  C `emission_function` derives from its callers' rows, its call sites'
  operand cells and its KERNEL_SOURCE root.  Values with no cell inside it
  are the kernel's own: units derive from the kernel scope, and labels do
  too.  This covers 294 of 323 C units in `scale`, 323 of 365 in `shared`.
- Authored kernel text is a root: NOVEL `AUTHORED_KERNEL_TEXT`, no operands,
  as a source span is.
- Wrapper rows are keyed `(symbol, NativeLoop)`: the SGD wrapper reuses the
  wrapped entry's symbol.  Wrapper body units derive from the units that
  define the registers they read.
- `emission_book` keeps reading `module.metadata["identity_book"]`: with no
  book attached, `identity_book(module)` calls `current_identity_book()`.

## Verified

- `probe_emission_chain`: failures 0, byte-identical everywhere.  Unsourced
  units, before -> after, C / LLVM: bump 4/5 -> 0/0, chain 6/7 -> 0/0,
  twice 5/6 -> 0/0, cond 9/10 -> 0/0, loop 6/7 -> 0/0, scale 294/9 -> 0/0,
  shared 324/14 -> 0/0.  "Unit lines not in final IR" 2 -> 0.  Single-block,
  SGD, Adam: 0 unsourced; each minted id has a NOVEL row.  C and LLVM bump
  tokens `t2` walk 17 hops to `source_span` BinOp.
- The seven listed probes exit 0.  `probe_scalar_native_correctness`:
  failures 0 on three runs; one earlier run printed `failures: 1`, its line
  not captured.  Audit unchanged (0,1,0,1,5,0,0).

## Open

- Operand hop (`name_binding` / `scalar_parameter`) still not reached from
  either token.
- C `deployment_support` / trampolines; Fortran, WASM, JavaScript.
- One edit in `emission_concordance.py` was written with a Python
  replace, not the Edit tool (four `Backend.LLVM_MODULE` -> `self.backend`
  in `NativeLoopEmission`).

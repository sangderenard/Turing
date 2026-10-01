# Continuation: identity cells upstream of emission (lane ID)

Plan 100 sections 2.6 and 4.1 item 2; design sections 2, 6, 7, 8.
2026-10-01.  Function names only; no compiler numberings; nothing committed.
Concurrent lanes (backend: `ssa_c_backend.py` / `ssa_llvm_backend.py` /
`emission_concordance.py` and Step 9 Part B of the declarations; linker:
`fortran_c_shell.py`) were not touched.

## What changed (files / functions)

`src/compiler/concordance_declarations.py`, sub-block "Step 9 identities"
after Step 9 Part B: enum `SSABlockKind` (one member per stem
`_ControlSSABuilder.new_block` is called with); reasons
`SSA_BLOCK_OWNER_UNROUTED`, `REGION_VALUE_MINT_UNRECORDED`; stage
`TENSOR_SSA_LOWERING`; transform `TENSOR_LOWERING_VALUE` (1); page
`ssa_block` `(function_scope, function, label)` -> `SSABlockKind |
Unresolved(SSA_BLOCK_OWNER_UNROUTED)`, CONCORD.

`src/compiler/control_source.py`: the owner rule of `post_control_program`'s
nested `describe` moved verbatim to module level as
`_describe_control_block(block, *, cell, region_cell, callsite_cell)`;
`post_control_program` calls it with its graph-backed resolvers.  New
read-only `control_block_cell(book, scope, block)`: the same rule with
`canonical_value` / `call_binding` lookups, nothing posted; a block whose
owner is the program-cell stand-in returns None.

`src/compiler/precompile_to_ssa.py`:
- `_ControlSSABuilder.lower` pushes `control_block_cell` of each non-sequence
  block onto `_control_cells` while it is lowered.
- `new_block` posts `_post_ssa_block`: ENTRY / FUNCTION_EXIT DERIVED from the
  shell `control_program` cell and the function root; any other block
  DERIVED from the innermost control block's cell; with no own row, the fact
  is `Unresolved(SSA_BLOCK_OWNER_UNROUTED)` reading the nearest enclosing
  control cell (else the function cells).
- `_post_planned_region_identities` (module level), called in
  `lower_control_sections_to_ssa` right after `_post_region_signature`: every
  value the region body produces with no `ssa_value` row is posted
  ADOPTED_GRAPH_ID DERIVED from its `canonical_value` cell (`_graph_cell`, the
  post-relabel arm of `node_identity_cell`, read-only); no canonical cell ->
  `Unsourced(GRAPH_ID_WITHOUT_CANONICAL_CELL)`, or, for an id with the
  MINTED flag, origin MINTED and `Unsourced(REGION_VALUE_MINT_UNRECORDED)`.
  The region's block is DERIVED from the `region_signature` cell.

`src/compiler/tensor_ssa_lowering.py`, `lower_tensor_calls_to_repository_ssa`:
`fresh` mints through `_mint_ssa_id` (NOVEL `TENSOR_LOWERING_VALUE` under
`function_scope_of(function)`) from the cell of the instruction being lowered
(`lowering_at`, set at the top of the per-instruction loop: result cell, else
first argument with a cell, else the function root).  Same monotonic source,
so emitted text is byte-identical.

`src/compiler/ssa_record_return_state.py`: reader
`ssa_block_identity_cell(function, label, *, book=None)`.

## Decisions taken here (say so if wrong)

- `ssa_block` carries the function SYMBOL as well as the scope: a planned
  region shares its root's control scope, so `(scope, "entry")` would be one
  row for two blocks (Part B made the same call for emission rows).
- Region formals get no row of their own: the formal's id is the caller's
  feed id under the same scope, so the formal's cell IS the feed's
  (`ssa_value`, else `control_value_binding` through `value_cell`).
- Region values are posted when the region is assembled; the builder's
  `_value_cell` later finds the row and reuses it.

## Verified

- `py_compile`; CRLF kept in the CRLF files, `control_source.py` LF; ASCII.
- The seven probes exit 0 with no FAIL; `probe_scalar_native_correctness`
  failures 0; `probe_emission_chain` failures 0, byte-identical C and LLVM,
  every case `unsourced units=0` (with the backend lane's concurrent work).
- bump (both forms), SSA values / blocks without an identity cell: 1/8, 2/2
  -> 0/8, 0/2.  chain 3/10 -> 0, twice 2/9 -> 0, cond 3/12 + 5/5 blocks -> 0,
  loop 6/6 blocks -> 0.  The region `k + 1` unit is DERIVED in C and LLVM;
  the probe's token chain reaches `source_span` BinOp (17 hops).
- Audit first lines unchanged (0, 1, 0, 1, 5, 0, 0).  unsourced facts /
  identities before -> after: view 4105/48 -> 4105/0, toplevel 2829/10 ->
  2831/10, energy 3800/46 -> 3800/10, controller 4542/39 -> 4540/1,
  controller_untyped 4433/39 -> 4431/1, mapping 197/0, oscillator 2951/0
  unchanged.  Toplevel's +2 are two plan-fold Max ids now on the worklist
  as `region_value_mint_unrecorded` (before: no row at all).
- `measure_completeness.py mapping controller`: mapping DERIVED 51.9% ->
  52.3%, unsourced 30.4% -> 30.1%, minted-with-record 15/15; controller
  77.7% -> 77.7% (22061 -> 22142 cells), 24.1% -> 24.0%, minted-with-record
  104/143 -> 142/143.

## Not done (next)

- Tensor reference kernels (`binary_value`, `unary_double`, ...): no
  `ssa_value` / `ssa_block` rows (controller: 265 values, 114 blocks).  Site:
  the `reference.dependency_closure(...)` link loop at the end of
  `lower_tensor_calls_to_repository_ssa`.  Their root should agree with the
  backend lane's `AUTHORED_KERNEL_TEXT` root; not invented here.
- `hierarchical_plan` region expansion `fresh_like` (variadic min/max fold,
  clamp) mints with no record: post NOVEL there under the lowering's scope.
- Controller residue: four `NoneValue` graph ids in the specialized
  `pi_update`'s `if_merge` with no row (writer not identified); formals of
  the root wrapper (2) and `update_dt_max` specialization (1) with no row.
- The probe's operand hop (`name_binding` / `scalar_parameter`) is still
  OPEN: the formal `k`'s chain ends at `canonical_value` -> `ingestion_value`.
- Fused lowering (`numerical_region_N` / `planned_region_N` in the fused
  path): no read scope, no rows posted.
- `SequenceBlock`-free owners that resolve to the program-cell stand-in
  (statement blocks with an unrouted region) yield Unresolved block rows;
  none occur in the probes or the controller case.

## Reader signatures (backend lane)

    ssa_value_identity_cell(function, value_id, *, book=None) -> Ref | None
    ssa_block_identity_cell(function, label, *, book=None) -> Ref | None

Both in `src/compiler/ssa_record_return_state.py`; pass the module's attached
book after the compile closed.

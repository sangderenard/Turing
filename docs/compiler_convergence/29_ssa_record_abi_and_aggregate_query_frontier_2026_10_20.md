# SSA record ABI and aggregate-query frontier — 2026-10-20

This continues reports 23–28 in the canonical repository
`C:/dev/Powershell/turing`. No bootstrap, vehicle, or simulation source was
changed.

## Outcome

The isolated real `src/compiler/ssa_object_operations.py::_result_layout`
unit advanced from six unexplained structural inputs and an unresolved final
query to one explicit backend-neutral lowering frontier.

The compiler repairs are general:

1. Program-ABI row records now admit a nested `table` field as one `int64`
   child-table handle column. The compiler-object contract now describes
   `Function -> BasicBlock -> Instr -> SSAValue -> Extent` without an opaque
   Python object or invented opcode/dtype vocabulary.
2. Resident sequence queries now lower builtin `any(...)` and
   `next(iter(sequence))`. `any` is a real boolean reduction loop; scalar
   `first` loads row zero. `iter` is unwrapped only at this exact intrinsic
   boundary.
3. `ast.IfExp` now publishes the same immutable conditional aggregate layout
   as statement-form conditional assignment. Repeated reduction is idempotent.
4. Comprehension row publication and ordinary append/add mutation expansion
   consume the authoritative `SOURCE_AGGREGATE_LAYOUT` fact when disposable
   graph attributes are absent.
5. Conditional aggregate result aliases expand into their canonical member
   Phi identities. Historical ingestion member IDs remain provenance only and
   never become public SSA formals.
6. Scalar `first` now refuses a multi-column row instead of silently returning
   column zero and dropping the rest of the aggregate.

## Real-source receipts

After the Program-ABI and intrinsic-query repairs:

- all four source-region integrals were complete;
- root parameter accounting was complete for `function`, `layouts`, and
  `returns`;
- `any(...)` and `next(iter(...))` had no unresolved required values;
- the root return was published;
- only tuple value 41 remained unexplained.

Receipt:
`build/bootstrap_result_layout_intrinsic_queries_20261020/units/_result_layout/unit.json`.

After conditional aggregate layout and member-Phi repairs, tuple 41 and the
four historical ingestion-ID formals disappeared. The last structural value
was `values[0].dtype` (value 17), proving that the `Instr.args` handle has not
yet been correlated with its declared `SSAValue` child-pool columns. Receipt:
`build/bootstrap_result_layout_canonical_member_phis_20261020/units/_result_layout/unit.json`.

The final safety trial intentionally fails at stage 9/11 with exactly one
shortfall:

```text
first-aggregate-row: resident sequence first must publish every aggregate
row column before multi-column results are admitted
```

Receipt:
`build/bootstrap_result_layout_honest_aggregate_first_20261020/units/_result_layout/failure.json`.

This refusal is required: `layouts` is now correctly a two-column unique
resident table. Returning only its first column would claim compilation while
dropping the authored shape tuple.

## Tests

- `tests/test_precompile_to_ssa.py`: **134 passed**.
- Six focused nested-row, IfExp aggregate, comprehension publication,
  conditional member-Phi, `any`/scalar-`first`, and aggregate-`first` refusal
  regressions pass.
- `git diff --check` reports only the repository's existing LF/CRLF warnings.
- Broader pre-existing dirty-tree gates still stop at their previously exposed
  unrelated frontiers: `test_loop_composer.py` reaches the source-less
  `formal_shape` fixture refusal; `test_aggregate_call_identity.py` reaches an
  unregistered projected Input identity; `test_extraction_contract.py` has the
  incumbent native-extension default-policy expectation mismatch.

## Next repair

First, materialize the Program-ABI child-pool descriptor for a nested table in
a row record. In this unit that means correlating the `Instr.args` handle with
the declared `SSAValue.dtype` and `SSAValue.shape` columns and their per-handle
extent. The emitted `BasicBlock.instrs` record currently contains only the
outer `op` and `args` handle spans; no `SSAChildTablePoolDescriptor` is attached.

Second, extend `SequenceQueryBlock(operation="first")` to publish a complete
aggregate row and its exact aggregate return layout. Only after both columns
of `layouts` are returned should `_result_layout` be allowed to become
linkable. Then rerun this isolated unit before another full bootstrap census.

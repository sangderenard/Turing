# Authored typing container identity and physical ABI receipts

Base: clean Turing 554ac55e; isolated branch codex/annotation-container-abi-20261113.
The concurrently modified canonical compiler was not copied or edited.

Observed chain: the real signal graph API declares
`output_node_keys: Optional[List[str]]`. `_authored_annotation_python_types`
records builtins.NoneType/builtins.list, but `_authored_annotation_field_receipt`
accepted `list` and omitted `List` (also Tuple) from its physical container
interpretation. Optional faithfully propagated that missing receipt. The native
union consumer correctly refused its unbacked list alternative. This is not an
Optional layout failure and does not justify relaxing any union/concordance gate.

Fix: derive concrete physical container names from the existing semantic reader's
exact builtins identities, including nested sequence elements. No second alias
map, new layout, union fallback, or synthetic storage is introduced. Protocols
retain existing behavior; unknown element and heterogeneous tuple layouts remain
refused. Ordinary source-to-ABI registration retains parameter annotation cells.

Verification (Windows Python 3.11, CPU-only):
- Before fix: new 28-case helper regression: 22 failed, 6 passed, reproducing
  missing List/Tuple layouts including nested and Optional forms.
- After fix: 33 focused cases pass. This includes 2 whole-source lowerings through
  `lower_ast_source_to_ssa`, both Optional[List[str]] and Optional[list[str]],
  with exact None/list-span alternatives and source cells; concordance is clean.
- `python -u tools/audit_identity_concordance.py mapping` before and after:
  28 rows / 2 functions, 0 findings, 0 unsourced facts/identities; latch closed.
- Clean baseline `python -m pytest -q tests/test_value_union_abi.py --tb=short`:
  14 passed / 4 failed. The failures are closed_class_union_lowers_to_payload_and_tag,
  open_type_predicate... [Any] and [object] (unsettled entry dtype), and
  mapping_value_column_uses_writer_union_abi (four existing concordance findings).
- Patched full file before the last five test additions: 42 passed / the same
  4 failed, with identical refusal causes. Final full-file run: 47 passed /
  the same 4 failed (50.73s); all 33 new cases pass.
- `git diff --check`: clean.

Limits: this fixes one ingestion/ABI boundary, not AbstractNN compatibility or
full signal-graph lowering. No LLVM artifact, native execution, GPU run, service
change, merge, or deployment is claimed. Tensor-core discovery warns it is not
connected in these isolated worktrees; tests use the existing Python backend.

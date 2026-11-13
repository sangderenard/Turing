2026-11-13 Preserve linked optional type alternatives during structural folding (MEASURED, clean compiler)

Source: the radio abstract-NN network_materializer.compile_nodes lexical closure through source_divider._attempt_entry, LLVM target. With the annotation-container repair, lowering raised KeyError(44). A diagnostic trace located the wrong proof before that lookup: output_node_keys carried linked ABI python_types=(builtins.NoneType,builtins.list), storage=span, dtype=int64. The linked-ABI reader discarded the alternatives and inferred nonoptional builtins.int from element storage. Consequently `output_node_keys is not None` folded true, removed the fallback comprehension GetAttr44, and left LoopResult111 referring to its removed value source.

Repair: the existing linked-ABI reader retains type_expectations/python_types/singleton python_type with the same precedence as the formal parameter reader. None remains an optional arm; multiple alternatives do not become a fabricated singleton. The reference-handle reader preserves these alternatives too. No missing-node guard, weakened union ABI, fabricated dtype, or source-specific fallback was introduced.

Verification:
- Eight new source-built/reduced graph tests: before repair 6 failed, 2 passed; after 8 passed. Both `is` and `is not` remain runtime predicates for optional span/scalar/reference contracts; singleton int and None proofs still fold correctly.
- Full test_linked_reference_kind_folding.py: repaired 9 passed, 12 failed; clean base 554ac55e 1 passed, the same 12 failed. Eleven existing hand-built fixtures lack ingestion identities; the recursive collection test fails concordance on both versions. These are not claimed fixed.
- Mapping identity-concordance audit: 28 rows, 2 functions, 0 findings, 0 unsourced facts, latch CLOSED (unchanged from baseline).
- Actual compile_nodes rerun: optional=True and both alternatives survive; KeyError44 no longer occurs. Lowering now refuses region value48 (GetAttr in compile_nodes) with no settled dtype. No LLVM artifact or full GraphSolver execution is claimed.

Limits: no broad suite run, merge, deployment, or changes to the shared dirty compiler checkout. The next unsettled source-field/type evidence requires independent investigation; do not fill it in from storage guesses.

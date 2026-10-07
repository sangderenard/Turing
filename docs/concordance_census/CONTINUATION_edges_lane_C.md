# Concordance edges, lane C (reducer-synthesized ingestion rows)

Rule (user, absolute, 2026-10-03): "nothing is valid in any way that doesn't
go through the concordance leaving edges".  Lane C owns the
Unsourced(SYNTHESIZED_NO_SOURCE) `ingestion_value` rows that generic
reducer / graph-build code writes for every program (plan 70, step-3
worklist), found by lane B in the rule-graph book.

## 2026-10-03 measurement method

- Scratch probe (not in repo) wraps `IdentityBook.post`. It counts
  `ingestion_value` posts with `Unsourced(SYNTHESIZED_NO_SOURCE)` by
  `NodeFact.type` and call site.  It counts the rule graph
  (`_build_backward_rule_process_graph` in a fresh book) and each audit case.
  The per-case count equals the audit's `('ingestion_value', 'reduction')`
  unsourced group.
- Reverse-compile book: the probe runs
  `test_graph_reverse_is_a_compiled_parametric_vjp` and counts the same
  posts plus the new pages.

## 2026-10-03 baseline (tree a7f57c1e + other lanes' WIP)

Rule graph: 398 = StaticReference 220, `str` 104, Input 32, Phi 20,
Constant 8, Indexed 5, Name 3, Untranslated 3, int 2, NoneType 1.

| | view | toplevel | energy | controller | ctrl_untyped | mapping | oscillator |
|---|---|---|---|---|---|---|---|
| ingestion_value unsourced | 48 | 143 | 72 | 90 | 90 | 7 | 59 |
| audit unsourced facts | 4022 | 2831 | 3800 | 4546 | 4431 | 197 | 2952 |
| findings | 0 | 1 | 0 | 1 | 5 | 0 | 0 |

What the categories are:
- **StaticReference.**
  - `static_reference_node` (an external object: builtins, numpy,
    AbstractTensor methods, imported helpers, one class) and
    `first_class_function_node` (a function-table entry used as data).
  - Both are cached per object/address and shared by every use.
- **`str` / `int` / `NoneType`.** These are NOT made in `_set_operands`.
  - `graph_express2.build_graph` visits an AST node's scalar fields
    (`keyword.arg`, `Attribute.attr`, `FunctionDef.name`) as nodes.
  - `ensure_node` keys them by `id(obj)`, so an interned `'dim'` is one
    node shared by every keyword that spells it.
  - They have no AST, so `_post_ingestion_value` never ran.
  - The first `connect` hit `node_identity_cell`'s fallback.
- **Input.**
  - `bind_loop_target`: 25 loop/comprehension targets.
  - `input_value` without an `ast.arg`: 7 external free names
    (`transpose`, `tensor`; `ENERGY_J`, `np` in the audit cases).
- **Phi.** The If merge in `reduce_statement`.

## 2026-10-03 applied

Declarations (appended to `concordance_declarations.py`):
- `static_symbol_definition` row `(module, qualname, receiver)`.
  - Fact: `StaticSymbolFact(StaticSymbolKind, digest)`.
  - The row is the referenced object's own declared identity.  A bound
    method's receiver is part of the row: two receivers are two symbols.
  - Provenance is DERIVED from `backward_rule_definition` /
    `class_declaration` under the same (module, qualname) when the book has
    that row.  Otherwise it is a NOVEL(INGEST_SOURCE) root: an external
    declaration ingested by reference.
  - Mode: CONCORD.
  - Digest: sha256 of co_code + co_names for a Python function, else None.
- `static_reference_use` row `(ingestion scope, use id, reference node)`.
  - Fact: `StaticReferenceUseFact(path)`.
  - DERIVED(the occurrence's cell, the definition cell, the reference
    node's cell).
  - Mode: CONCORD.
- Reason `static_symbol_undeclared`: an object with no module or name.
  Not reached by any gate.

Writers:
- **`topological_reducer`**:
  - `new_node(source_cell=)` takes one Ref or a tuple of Refs, plus
    `unsourced_reason=`.
  - `_StaticPythonReference.occurrence` (compare=False): the AST use.
    - Set at the five creation sites in `resolve_expression`.
    - A read of a `static_environment` entry is re-stamped with the read
      (`dataclasses.replace`).  Sites in `bind_target` need no edit; they
      carry the occurrence on the reference.
  - StaticReference row DERIVED(definition cell):
    - the table's `function_address` cell (function-table entry or
      first-class function);
    - else the object's `static_symbol_definition` cell.
  - Every call (cached or new) posts the use row from the occurrence.
  - `first_class_function_node(occurrence=)` is passed by both callers.
  - Inputs:
    - loop target: DERIVED(target's ingestion cell, else its span).  Passed
      as `source_cell`, so no lexical `source_span` stamp is added.
    - external / closure / exception names: DERIVED(the occurrence's cell).
      The occurrence is the reading Name, or the ExceptHandler that binds
      the name.
    - A module global's binding never enters the book, so the read is the
      source.
  - Phi: DERIVED(test, body and orelse value cells, the `if` span).
  - Untranslated stand-in (`_untranslated_operand`, a small extra):
    DERIVED(consumer cell, the absent operand's surviving row).
- **`graph_express2.connect` -> `_post_field_value_use`**:
  - Each connect between a non-AST field value and its AST holder
    posts the value's `ingestion_value` row.  The first use writes the row;
    a later use re-posts the same fact (CONCORD no-op plus one more edge).
  - Provenance: DERIVED(holder's ingestion cell).  Every use leaves an edge
    on the shared node.

## 2026-10-03 results

Rule graph: 398 -> 16 (Constant 8, Indexed 5, Name 3).  Reverse-VJP book:
31 left (those 16 + Store 15).  The book has 316 `static_reference_use`
rows and `static_symbol_definition` 59 roots + 1 derived.  It has 110
field-value rows; 13 of them carry more than one use edge.

| | view | toplevel | energy | controller | ctrl_untyped | mapping | oscillator |
|---|---|---|---|---|---|---|---|
| ingestion_value unsourced | 30 | 104 | 58 | 53 | 53 | 3 | 47 |
| audit unsourced facts | 4004 | 2792 | 3786 | 4509 | 4394 | 193 | 2940 |
| findings | 0 | 1 | 0 | 1 | 5 | 0 | 0 |

Per-page group diff vs baseline: only `ingestion_value/reduction` moves; the
new pages add zero unsourced cells.  In the audit cases, all four lane-C
categories go to 0.

Remaining, not this lane:
- **Store**: 30/85/47/43/43/3/47.  Written by
  `_set_operands` <- `_dispatch_subgraph` / `finalize_graph_with_outputs`
  (glsl_deployment_strategy, owned by the dt-stall lane).
- **Input `*.__present`**: 18/10/10/10.  From `post_verdict` / `classify`
  and `_lower_optional_record_presence_graph`.
- **Constant/Indexed in `bind_target`**: 5+5.  Name-arm lane region.
- **`static_constant`**: 3.  Cached per name, same shape as
  StaticReference; held.
- **Name**: 3.  Unstamped AST nodes at build.

## 2026-10-03 gates

- native_scalar_loss_adjoint, graph-reverse VJP, linear motion,
  ssa_record_return_state: 26 passed (one process).
- record_return_site_identity: 14 passed, 1 xfailed (own process).
- precompile_to_ssa: 13 failed / 91 passed.  No failure touches lane-C
  code.
- orbital_transfer_compile: 12 passed (245 s).
- Audit findings 0,1,0,1,5,0,0, unchanged.

## Found, not fixed (lane B's mechanism)

Building the rule graph and then compiling a forward program in a NEW book,
in one process, refuses with `source cell does not exist:
Ref('backward_rule_definition', ('BACKWARD_RULES', 'clamp'), 0)`.  Two
triggers were seen: the scratch probe (rule then toplevel), before any
lane-C edit, and pytest running the VJP test before
record_return_site_identity (6 failed + 8 errors; each passes in its own
process).  `_turing_source_cells` stamps from `build_from_ast(source_cells=)`
apparently persist on AST objects that a later book reuses (a cached parse),
and `post_source_span` derives from a cell of the old book.

## 2026-10-07 cross-book stamps: resolved

The stamp is `graph_express2.SourceDefinition(page, row, fact)` (declared page
`backward_rule_definition`, row `(registry, name)`), not a Ref.
`resolve_source_definition` turns it into the CURRENT book's cell (latest cell of
that row, else posts the same NOVEL(INGEST_SOURCE) root); `post_source_span` is the
reader.  Writers: `build_from_ast(source_cells=)`, `_annotate_visual_source_owners`;
`process_graph_autograd._post_backward_rule_definitions` returns the identities.
Proof: `tests/test_source_definition_cross_book.py`; VJP test +
`test_record_return_site_identity.py` in one process: 16 passed, 1 xfailed.
The VJP test then exposed a second, independent fault: the root graph is also a
function-table entry, so `_propagate_callsite_tensor_specializations` visited its
callsites twice per round and posted `callsite_descriptor_reuse` (`computed` then
`reused`) for one row; the root is now skipped as a table entry (`entry.graph is not graph`).

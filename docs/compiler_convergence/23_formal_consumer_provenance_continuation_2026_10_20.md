# Formal-consumer provenance continuation — 2026-10-20

Canonical repository `C:/dev/Powershell/turing`, branch `main`. This report
continues report 22. The worktree changes described here are not committed.

## What moved

The SSA formal collector now reads the planner's `executable_node` row when it
creates a provisional value. It preserves the verdict and rule together with
the exact source node id, AST construct, unparsed source and authored span.
The receipt is stored both on the provisional value and in function metadata,
because sequence-ABI expansion can replace a source formal with minted frame
members before publication.

The compilation-product audit now joins unexplained public arguments to those
receipts. The native-publication refusal includes the joined details rather
than reporting only integer ids. This does not bless, lower or repair any
generated value; it makes the existing refusal name the concordance fact that
the consumer read.

The exact regression source is:

```python
def f(x: tuple[int, ...]) -> tuple[int, ...]:
    return (0, *x[1:])
```

It proves that the collector records the literal's dispatch-metadata verdict,
rule and source span. A synthetic unit-audit regression proves an unexplained
formal joins the real unit-034-shaped tuple receipt:
`(origin, *ref.row[1:])`, `Tuple`, `PYTHON_SYNTAX`.

## What did not move

The variadic tuple-display sequence lane is still not built. No classifier
rule was changed and no generated output was edited. In particular, `Starred`
was not relabelled executable: an experiment showed that doing so falsely
lowered `x[1:]` as one scalar and returned `(0, scalar)`.

Tuple-valued mapping storage is also still not built. Report 22's requirement
that the reducer, keyed ABI, lookup helper, store helper and minted result
leaves land as one general-N feature remains unchanged.

## Validation

- `tests/test_native_formal_provenance.py`: 12 passed.
- Focused cross-file gate over native provenance, project accounting and
  provisional-formal behavior: 10 passed, 230 deselected.
- `git diff --check`: clean apart from the repository's LF/CRLF warning.

A targeted public-entry compilation of
`IdentityBook._virtual_edges_into` was attempted. It stopped before unit 034's
formal/publication lane on an earlier missing identity row in
`IdentityPage._resolved`: node 78, `Input 'row[0]'`, was created without an
identity row. Therefore that run is not validation of the new receipt path.

## Full bootstrap measurement

At the user's request, the normal fail-fast bootstrap was then run with:

```text
python -u -m tools.bootstrap_compiler_whole_program
```

It produced fingerprint
`3c1b8a2900c4c0e207bd047aa912370655102cb259ef6afb37b6905708e9a172`
and stopped after 569.263 seconds with exit status 1. The 330-unit plan was
unchanged. The stage admitted 29 units: 26 compiled-unverified, 3 partial, 0
failed and 0 verified, leaving 301 pending. Fail-fast was enabled and the
three partial units were its blocking set. This is not a native-success
receipt and the product was not installed.

The previous report-22 run admitted 31 units (30 compiled-unverified and only
unit 034 partial). The lower admitted count in this run is not by itself a
two-unit regression: worker scheduling is concurrent, and units 081 and 115
finished partial before the fail-fast stage settled. They are newly catalogued
blockers under this fingerprint, not proven newly introduced defects.

The authoritative stage receipt is
`build/compiler-whole-program/products/provenance-core/3c1b8a2900c4c0e207bd047aa912370655102cb259ef6afb37b6905708e9a172/compartment.json`.
Each partial has `unit.json` and no `failure.json`, consistent with a named
accounting refusal rather than an exception.

## Blocking-unit catalogue

### Unit 034 — `IdentityBook._virtual_edges_into` (known)

The known unresolved required value remains tuple display 43, whose unresolved
leaf is `elts` value 42. Public values 13 and 19 remain unexplained. The new
consumer receipt path worked: value 19 is now a
`concordance-consumer-refusal` naming the exact dispatch-metadata `Tuple`,
source `(origin, *ref.row[1:])`, span 5966:21-43 and rule `python_syntax`.
Value 13 is also named as the executable `Subscript`
`__turing_assignment_value_47[1]`, span 5965:16-21. This improves the receipt;
it does not provide the missing variadic tuple lowering.

Receipt: `growth/product/units/unit_034/unit.json` under the fingerprint above.

### Unit 081 — `src.compiler.ssa_object_operations._result_layout` (newly catalogued)

This compiler-object introspection helper is incomplete at compiler
finalization:

- call value 58 is unresolved through callee 38 and argument 57, and its call
  result is unavailable;
- public values 5, 8, 9, 10, 41 and 49 are unexplained and all six are refused
  by the concordance consumer with dispatch-metadata receipts;
- the same six values have unresolved boundary types; and
- the authored result path `[0] -> 58` has no published output, producing
  `native-result-layout-mismatch`.

The six receipts name `i.args`, `function.blocks.values()`, `b.instrs`,
`i.op`, tuple layout construction
`(values[0].dtype, tuple(values[0].shape or ()))`, and the generator/`any`
cardinality check. The first four are compiler object-model traversal; the last
two are aggregate/coordinator work. The receipt establishes missing consumers
and boundary types; it does not establish one single implementation fix.

Receipt: `growth/product/units/unit_081/unit.json` under the fingerprint above.

### Unit 115 — `CorrelationTable._source_page_declaration_findings` (newly catalogued)

This audit routine publishes output 32 but remains incomplete:

- eleven call results are structurally unavailable: values 170, 172, 178,
  179, 187, 188, 191, 192, 194, 198 and 229;
- 27 public arguments are unexplained;
- 22 of those have exact dispatch-metadata receipts and are now named as
  `concordance-consumer-refusal`; the other five are source-less minted int64
  values and therefore have no source execution receipt to join; and
- 17 values still have unresolved boundary types.

The named receipts cluster around the actual authored operations: Identity
Book/Page/Ref traversal (`page.rows()`, `latest_ref`, `edges_into`, `.page`,
`.row`, `.row_fields`), string/tuple helpers, and nested generator-driven
`any`/`all` checks. This is a wider missing physical representation/consumer
surface than unit 034's single variadic tuple display. The receipt does not
justify treating metadata objects as numeric scalars or bypassing the book.

Receipt: `growth/product/units/unit_115/unit.json` under the fingerprint above.

## Next work

1. Make return slots, region feeds and display-leaf consumers perform the same
   concordance read before use.
2. Build tuple-display-as-sequence materialization as clear/append/append-slice
   through the existing resident sequence ABI.
3. Trace unit 081's unresolved call 58 to its existing compiler-object ABI and
   supply the missing result/boundary consumers at their generators; do not
   invent a parallel object representation.
4. Partition unit 115's named refusals by the existing Identity Book/Page/Ref,
   string/tuple and generator contracts before changing any classifier or
   lowering. Its five source-less minted values need an identity-origin audit.
5. Land tuple-valued mapping support only as the complete general-N feature
   specified in report 22.
6. Run the user's normal bootstrap only after those generator features and
   their focused regressions are complete.

## Prompt history

> I did not fully trust the agent who did the last section of work, but here we
> are, trying to move the needle on the bootstrap.

> can you do any of the rest? did i not send you a path to a continuation report

> can you run the bootstrap for me and catalog the new problems

Relevant governing instruction:

> Nothing may fix generated content after the fact.

> Nothing may operate without first checking the concordance on what they're
> going to use.

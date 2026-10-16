# Provenance, resident sort, glyph formatting, and bootstrap continuation

Date: 2026-10-15

This is the current continuation receipt for the compiler bootstrap work in
`C:/dev/Powershell/turing`.  It supersedes the failure frontier in the earlier
bootstrap handoff, but it does not claim that the whole bootstrap is green.

## Governing constraints

- The concordance is the persistent authority for source identity and derived
  facts.  Later passes consume or project those facts; they do not reconstruct
  them from names, positions, annotations, or neighboring graph shapes.
- Graph edges and their recorded roles are authoritative.  Source partitions,
  compilation units, copied scopes, and SSA plans must preserve that chain.
- A compiler repair belongs at the first wrong writer.  Do not patch emitted
  SSA, a partition receipt, or another intermediate product to hide a missing
  source fact.
- `src.compiler.fortran_c_shell.lower_ast_source_to_ssa` remains the sanctioned
  source entry.
- CPython is used only where the existing extraction policy explicitly admits
  it.  This work does not convert ordinary values into retained Python objects
  or broaden fallback policy.
- The user is running the long bootstrap.  Make a small exact reproduction,
  repair the compiler, run focused checks, clear the bootstrap caches, and then
  return the next long run to the user.

## Implemented work

### Source concordance and projection

The source book now carries explicit declarations for aggregate layouts,
glyph-sequence recipes, enum glyph tables, optional glyph arms, value classes,
and sort contracts.  Projection into accepted source components follows the
recorded predecessor cells.  Relabeling and copied call scopes retain those
facts rather than rebuilding them from graph attributes.

Source module identities are content-derived when there is no filename, so
independent anonymous source trees no longer share one declaration namespace.
Function-address identity and copied function scopes are recorded explicitly.
Class/field declarations, lexical read edges, call-result identities, and
carried aggregate layouts now participate in the same predecessor chain.

Primitive source class spellings are canonicalized (`builtins.str` and `str`,
and the corresponding `bool`, `float`, and `int` aliases).  Concording the same
fact retains every independently established predecessor edge.

### Call-formal binding regression

The newest bootstrap failure was:

```text
CorrelationTable._undefined_operand_findings value 8:
recorded=('str', 1), call actual CorrelationTable.findings:11 says
('CorrelationTable', 1)
```

The callee graph correctly recorded `method_binding="static"`.  A duplicate
binder in topology reduction nevertheless sorted every `operand` edge before
the positional arguments.  For
`self._undefined_operand_findings(str(name), function)`, it therefore assigned
the attribute receiver `self` to the static method's first formal `name`.

That local binder has been removed.  Source value-class propagation now uses
the existing shared call-argument mechanism in
`src/compiler/call_argument_binding.py`.  The shared mechanism binds an
attribute receiver only when the callee's exact metadata says `instance` or
`class`; a `static` method starts with `arg:0`.  This is a compiler-wide repair
for the binding class, not a special case for `CorrelationTable`.

### Resident sequence sort

`list.sort()` and `sorted()` are represented as source sort contracts and
lowered to ordinary repository SSA.  The implementation is a stable bottom-up
merge sort over resident sequence storage.  It has explicit scratch capacity
and status, copies complete fixed-width rows, and does not call a host sort or
hide an algorithm in a backend opcode.

Supported key routes are recorded source policies:

- the row's scalar column;
- a fixed row projection such as `lambda row: row[1]`;
- a resident mapping lookup such as `positions.__getitem__`.

`sorted()` materializes distinct storage before invoking the same stable sort,
so it does not mutate its input.  `tools/benchmark_ssa_sort.py` is the compile,
native-execute, and parity harness for unkeyed, mapping-keyed, row-projection,
and value-producing cases.

### Glyph sequences and formatted values

Strings remain resident sequences of reversible `StringTable` glyph tokens.
Literal text and resolvable formatted components publish exact glyph recipes
through the source concordance.

Authored Python helpers in `src/compiler/glyph_sequence_format.py` provide
decimal int64 and prefixed lower-hex int64 spelling.  Source pursuit compiles
their arithmetic, control flow, and sequence appends through the ordinary
compiler path.  There is no special formatting backend instruction.

For an unresolved runtime `!r`, the existing extraction contract admits one
repository-owned CPython call, `runtime_repr_utf8`.  Its result crosses the
boundary as a resident UTF-8 byte sequence.  An authored helper then segments
that byte sequence into reversible glyph identities.  This admission is
specific to the unresolved formatted-value operation; it does not change the
policy for arbitrary `repr` calls or retain a Python object in SSA.

### Aggregate, control, and resident-storage provenance

The current changes also preserve exact producers across conditional
constructors, loop-carried values, mapping/record projections, whole-object
lookups, comprehension ownership, field replacement, optional physical slots,
and copied function variants.  Sequence materialization consumes the accepted
source contracts for sort, glyphs, snapshots, queries, and record rows.

Bootstrap publication now keeps the staged source/provenance receipts required
to resume compilation units.  A missing composed repository module after a
partial partition is reported as such; the final detector no longer replaces
the real unit failure with a secondary `NoneType.functions` exception.

## Observed bootstrap frontier

The important runs, in order, were:

1. Compilation units reached 5/466.  Units 046-048 were partial in compiler
   finalization (`IdentityPage._reaches_origin`, `_fork_of_scope`, and the
   corresponding qualified methods).
2. After provenance repairs, unit 046 `src.compiler.id_space.compose` reached
   SSA lowering and failed because a formatted glyph component had no resident
   producer (`value=20`, `projection='scalar'`).
3. General scalar/glyph publication exposed a topology disagreement between
   `builtins.str` and `str`.  Primitive class identity is now canonical.
4. The next run stopped earlier in topology because the duplicate call binder
   assigned a `CorrelationTable` receiver to a static method's `name: str`
   formal.  The exact reproduction failed before the repair and passes after
   the repair.
5. The following run reached topology reduction of `IdentityBook.post` and
   refused a partially canonicalized glyph recipe.  The recipe named a
   reducer-generated formatter from a different retained function copy.  The
   first copy had published that private identity under the shared authored
   scope, and NetworkX's shallow subgraph copy had also shared the nested
   mutable `attributes` mapping with the root graph.  Function copies now
   detach that mapping, dynamic recipes publish in their existing copy-local
   ingestion scope, and canonical relabeling selects that exact local row.

The fifth repair has not yet been exercised by a complete bootstrap.  The
next truthful frontier is therefore the result of the next user-owned run, not
an inferred return to compilation unit 046.

## Verification receipts

Fresh after the static-method repair:

```text
python -m pytest \
  tests/test_abstract_tensor_topological_reducer.py::test_staticmethod_receiver_is_not_bound_to_first_formal_class \
  tests/test_abstract_tensor_topological_reducer.py::test_generic_formal_accepts_distinct_record_variants_without_class_stamp -q
2 passed in 1.05s

python -m py_compile src/common/tensors/topological_reducer.py
passed
```

Fresh focused sort/formatter coverage also passes:

```text
test_sequence_sort_is_stable_repository_ssa_over_complete_rows
test_sequence_sort_can_compare_resident_mapping_keys
test_authored_decimal_formatter_emits_reversible_glyph_tokens
test_authored_prefixed_hex_formatter_emits_reversible_glyph_tokens
test_utf8_repr_decoder_produces_glyph_identities_not_byte_values
5 passed in 1.33s
```

`python tools/audit_identity_concordance.py` still reports zero identity
findings for its successfully lowered fixtures.  Its `mapping` fixture reports
26 rows across two functions, zero findings, and zero unsourced facts.  The
audit's `toplevel` and `energy` fixtures still stop on three immutable physical
call-input dtype conflicts; those are not claimed fixed here.

The complete glyph provenance file now passes all 32 tests.  Its four former
failures were closed without changing glyph policy: enum projections retain
their exact authored table cell, optional arms consume the exact glyph row
recorded on the arm, and one copied test expectation now names the artifact it
actually requested.  The exact `id_space.compose` probe passed the former
missing-glyph-producer point and reached a later SSA block-geometry refusal.
Neither result is a whole-bootstrap success claim.

One neighboring provenance test,
`test_call_formal_class_projection_is_predecessor_closed`, currently fails
because its lookup for `CLASS_DECLARATION` at `("<string>", "Field")` returns
no cell after anonymous source scopes became content-derived.  Do not weaken
source scoping to satisfy that stale lookup; update the test to read the graph's
recorded declaration scope after confirming the exact source cell.

`git diff --check` is clean.  The only messages are Git's existing LF-to-CRLF
working-tree warnings.

## Addendum to checkpoint `c40eb1df`

The previous checkpoint's sort and glyph work must be continued through the
whole-program bootstrap, not treated as complete from focused tests alone.
`tools.bootstrap_compiler_whole_program` is the integration instrument: it
compiles the compiler's own authored source through the sanctioned public
entry, admits the staged provenance-core product, and exposes the first place
where retained source identity no longer agrees with a later compiler view.

Use the bootstrap as a measured failure sequence:

1. Start with cleared `build/compiler-partitions` and `build/compiler-logs` so
   every receipt belongs to the current source state.
2. Run `python -u -m tools.bootstrap_compiler_whole_program` without wrapping,
   filtering, or replacing its output.
3. Read the complete blocking-failure set, not only the first visible line.
   Record each unit, qualified symbol, compiler stage, exception, diagnostic
   log, unit receipt, and compartment receipt.
4. Group failures only after comparing their provenance predecessors.  Several
   failed units can be consequences of one wrong writer; one failed unit can
   also expose more than one independent disagreement.
5. Reproduce the earliest disagreement on a seconds-long slice of the real
   source.  Observe the source fact, every concordance derivation/revision, the
   graph resident expected by the consumer, and the exact consumer failure.
6. Repair the first writer or reader that abandoned the exact cell.  Never
   patch emitted SSA, a partition product, or a later receipt to compensate.
7. Run the exact regression, its nearest policy neighbors, and
   `tools/audit_identity_concordance.py`; then clear the two bootstrap cache
   directories before the next whole-program observation.

Provenance-core is not an auxiliary prepass.  It is the compiler product that
allows source identity, copied lexical scopes, generated residents, ABI
bindings, and SSA storage to remain one connected derivation across later
stages.  Its partial and failed unit receipts are therefore primary evidence,
not disposable bootstrap noise.

The concordance is the persistent authority that keeps that product coherent.
A later stage must consume an exact recorded cell or an explicit derivation of
it.  Reconstructing identity from a name, numeric node id, tuple position,
source spelling, inferred module, or neighboring graph shape breaks the chain
even when the reconstructed value looks equal.  The cross-copy glyph failure
after `c40eb1df` demonstrated the rule directly: an authored owner was shared,
but its generated formatter was resident only in one retained copy.  Publishing
that recipe in shared authored scope made a valid cell point outside the graph
that consumed it.  Copy-local ingestion scope plus exact-cell consumers restore
the single chain; weakening the resident-producer check would only hide the
violation.

## Continuation

The bootstrap caches were cleared after the latest repair:

- `build/compiler-partitions`
- `build/compiler-logs`

Run from `C:/dev/Powershell/turing`:

```powershell
python -u -m tools.bootstrap_compiler_whole_program
```

For the next failure:

1. Record the stage, qualified unit/function, exception, and receipt path.
2. Compare the stage with the chronology above.  Do not call an earlier failure
   progress merely because it is newly precise.
3. Build a seconds-long reproduction from the real source shape and capture the
   two disagreeing records plus the writer that posted the wrong one.
4. Repair that writer using the existing concordance/call-binding/storage
   mechanism.  Do not add an intermediate repair or change fallback policy.
5. Run the exact regression, the nearest existing policy regression, and the
   concordance audit.  Clear only the two bootstrap cache directories above.
6. Return the long bootstrap to the user.

If the run returns to compilation units, inspect every blocking receipt printed
by the stage summary before choosing a repair.  Several failures may be one
downstream class; do not patch units independently until their predecessor
chains prove they are independent.

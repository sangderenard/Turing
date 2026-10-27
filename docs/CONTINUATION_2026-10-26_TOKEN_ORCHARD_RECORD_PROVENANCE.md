# Token Orchard record-provenance continuation — 2026-10-26

## Scope

This continuation worked only on the sanctioned whole-core Token Orchard
lowering audit:

```powershell
cd C:\dev\Powershell\string_core
python tools\audit_turing_whole_core.py
```

No test suite was run. The audit intentionally stops at the first compiler
frontier.

## Repairs committed here

1. `SSASequenceTable` now publishes explicit member roles for every
   `SSAChildTablePoolDescriptor` cell: child columns, length, capacity, row
   stride, optional shape/rank/shape-stride, status, and live flags.
   `_linked_caller_member` resolves the same role in the caller descriptor.
   This removes the invalid positional comparison of the dict-backed
   `Vocabulary._prefix` field's full physical inventories.

2. Missing status/live-flags roles are explicitly callee-local operational
   storage. Missing payload columns, extents, pool columns, strides, or shape
   cells still refuse; no missing semantic storage is manufactured by the
   linker.

3. A declared `array(typecode)` field now establishes its one-column fixed
   sequence contract when its owner record is materialized, including when a
   constructor only writes the field. The column dtype comes from the declared
   array dtype; it is not inferred from append expressions.

## Audit progression

Before these changes the audit stopped while linking `Vocabulary._prefix`:
child-pool cells were part of the record field's physical inventory but had no
sequence-member role provenance.

After repair, the audit passed `_prefix`, then passed the callee-local
`Vocabulary._candidates` status boundary. It next refused that the entry
record exposed `Vocabulary.exact_atoms` only as a raw span while `intern_batch`
required its sequence column. The declared-array materialization repair cleared
that `exact_atoms` boundary.

## Current frontier

The current audit stops at:

```text
token_orchard_whole_core___intern_validated_span__specialized_891654e6c179
callsite 39: callee ...___cover_span__specialized_d77be5e27628
is sequence member ('column', 0) of 'Vocabulary.exact_begin';
the bound caller record 5 names no such member
```

`exact_begin` is another declared `array(typecode)` field. Its owner-side
contract is now available, but the intermediate receiver record view for
`_intern_validated_span` still carries only the raw span. The next repair must
be at record-view forwarding/materialization: retain the already-declared array
sequence contract on every `self: Vocabulary` receiver view before call-frame
linking. Do not add a field-specific linker fallback or map members by order.


# 2026-11-13: exact selected sequence lineage is distinct from owner transfer

## Work

Added a read-only selected-sequence proof that follows the existing address Phi
for data, logical length, capacity, status, and owner together. Every member
must have the same ordered predecessors, match the published block geometry,
and name one exact source descriptor with the same storage contract. Nested
selections recurse without manufacturing an owner for a selector.

Current direct sequence contract declarations identify the resident. Historical
append operations remain provenance, not competing identities of the resident.
Missing or ambiguous direct evidence refuses. Lifetime admission is a separate
join to validated sequence lifetime receipts; borrowed or missing births refuse.
This is not an escape or cleanup authorization.

## Verification

* New source selection plus existing nested native alias and LLVM bounded bytes:
  5 passed.
* Source constructor proof plus selection: 2 passed.
* Mapping audit: 28 rows, 2 functions, zero findings, zero unsourced, latch CLOSED.
* Actual GraphSolver source lowered to inspectable SSA, with output_node_keys100
  from constructor101. Exact outer selection names resident106 and resident104;
  source fallback97 aliases resident104 rather than introducing another owner.
  Both current terminal lifetime receipts are LOCAL_NEW with comprehension birth
  evidence. No native result execution was attempted by that inspection.
* Negative tests cover wrong owner predecessors, missing block geometry, wrong
  or ambiguous direct declaration, and missing terminal lifetime.

## Limits

New module absence is not counted as a fixed baseline failure. These readers
have not yet implemented public record-result storage, owner movement, release,
string decoding, multiple-field sharing, or native record-result lifetime. The
intentional record-result lifetime regression remains separate and red. The
existing direct bytes output path is not evidence that record fields work.

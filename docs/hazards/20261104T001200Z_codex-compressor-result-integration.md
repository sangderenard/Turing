2026-11-04 Owned bytes concat source and SSA return (MEASURED)

Checkout: C:/Users/alber/.codex/worktrees/compressor-results/turing.

The exact four-fragment bytes return from compress_octets was bounded as
`compress_tail(header, encoded_length, coded_commands, coded_literals)`.
Before this slice, SOURCE_SEQUENCE_CAPACITY_BOUND had no rows for the Add
chain. The source recognizer produced ordered bytes concatenations 4, 5, 6,
but the returned value 6 was excluded from dynamic ownership and born as a
FIXED descriptor with no capacity. Native C emission refused value 6 twice
as dynamic-growth-unavailable. The other two concat arenas were DYNAMIC and
already had exact heap owners.

SOURCE_OWNED_SEQUENCE_RETURN now cites the three PROCESS_GRAPH_OPERANDS
cells, four exact SOURCE_VALUE_CLASS bytes proofs, and the authored
RETURN_SITE_SLOT for value 6. This admits only that proven fresh returned
concat to the incumbent DYNAMIC heap path. PUBLIC_OWNED_SEQUENCE_RETURN
then cites the actual dynamic SEQUENCE_DESCRIPTOR and owner SSA_VALUE. The
read-only public_owned_sequence_returns view yields data 6, length
2305843010213693968, capacity 2305843010213693969, owner
2305843010213693971, and the completed public receipt Ref. Both pages are
audited; tampering with a class proof or owner Ref is refused. The existing
owned-heap allocator and explicit release contract are unchanged.

Fresh bounded source lower completes SSA. C emitter now reports complete
for that source, but public result marshalling/release is being integrated
separately; this is not native parity. The saved SSA is
`C:/Users/alber/AppData/Local/Temp/probe_compress_tail_capacity2_ssa.pkl`.
Focused test `tests/test_public_owned_sequence_return.py`: 1 passed in
5.89s. Mapping audit: 28 rows, zero findings, zero unsourced, CLOSED.

Full compressor operand 21 and 27 currently lack global SOURCE_VALUE_CLASS;
their class may be available through a closed call-result alternative, which
has not yet been measured on the new source. This receipt makes no full
compressor native-output claim.

Early source class follow-up (measured in the same isolated checkout):
`source_bytes_concat_class.post_bytes_concat_classes` now reads the exact
CANONICAL_VALUE Add and PROCESS_GRAPH_OPERANDS cells and the two bytes class
proofs for each Add before planning. The bounded four-fragment return posts
three source Add class cells and three canonical relabels. The second eager
evidence round posts no further cells. The new audit validates the authored
Add, operand identities, both class causes, and the relabel chain; a forged
result class is rejected. Bounded lowering reaches final concordance with
zero findings. Mapping audit remains 28 rows, zero findings/unsourced, CLOSED.
The source classifier reader returns the current class/operand/class refs and
the original producer refs only when the current and original Add and both
operand identities have the same unique ingestion origins. A post-flush
bounded source probe observed this reader accept all three Adds on both
eager rounds. A later specialized planner fork has not yet been measured.
The current combined native test passes its source and public-owned proof
assertions but reaches a concurrent runtime ABI v3 check refusing data/owner
ports as non-pointer cells. It has not yet established native parity after
this follow-up. No full compressor rerun was made for this class producer.

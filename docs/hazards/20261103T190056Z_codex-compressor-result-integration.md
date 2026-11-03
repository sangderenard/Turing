2026-11-03 Linked reference witnesses retain their provenance owner (MEASURED)

The canonical scan28 _split unit1 failed in
propagate_repository_ssa_call_metadata: CROSS_FUNCTION_REFERENCES for accepted
_read_header value5 had no source cell. A bounded official unit1 replay of the
frozen current plan reproduced it in 5.7 seconds. The trace showed two owners,
_read_header planned regions1 and4, both accepted from artifact digest
68d98c0991b27e40a56d2f360703f33edc2db74ff32cf67a367a04bf94e17cb8.
Both retained the same dependency book. Their SSA_VALUE cells and an identical
CROSS_FUNCTION_REFERENCES fact existed there; the composing book had neither
value nor block witnesses. The writer queried only the composing book.

The reference witness reader now selects each function's existing accepted
book and transports its exact value/block cells through IdentityBook.import_cell
using the verified artifact digest. The caller fact derives from these
existing import receipts. Foreign SSA rows are not copied or re-authored.
Two booked focused cases first reproduced the missing-cell failure. After
the repair, the positive case publishes exact region ownership with imported
witnesses, and a missing artifact digest refuses. Together with the existing
owner-book lease tests: 5 passed in 1.26 seconds. The frozen official plan has
not been replayed with the changed compiler; its fingerprint guard remains
intact. A fresh linked source integration check is still pending.

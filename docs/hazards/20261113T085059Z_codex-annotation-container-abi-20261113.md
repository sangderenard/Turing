2026-11-13 LLVM direct owned-sequence ABI (MEASURED, isolated compiler worktree)

Port the established C public-owned sequence protocol to LLVM, consuming only
existing validated source/public ownership receipts. Data, written length,
capacity and owner are distinct public ports. The host copies all written
results before a finally-release entry resets native cells. Windows exports the
release entry explicitly. Repeated release is safe for this exact direct owner
contract. No merged-owner escape or source-policy expansion is admitted.

API receipts use the compiler ptr layout's byte size for the host spelling
uintp. uintp is not introduced as a numeric compiler dtype. Compiler-owned
DYNAMIC returned descriptor handles without an escape receipt explicitly refuse;
borrowed authored public input aliases retain their existing storage.

Native tests: empty/binary/over4KB concatenations; repeated execution and retained
arrays; host-copy failure then release/reset and successful retry; repeated
release; tampered receipt/runtime ABI refusal; pointer byte-size receipt and
borrowed bytes input alias. Full CorrelationTable audit passes for owned output.

Candidate broad suite: 38 passed/2 failed. HEAD baseline: 36 passed/4 failed.
The two new tests fail baseline because the artifact ownership schema is absent;
this does not claim borrowed alias runtime behavior was previously broken.
The two incumbent failures are unchanged: C four-fragment concat private-root
storage refusal and synthetic owner sequence fixture raw column publication.
Logs llvm_owned_broad_verified.log and llvm_owned_baseline.log are local scratch.
One overlapping candidate run was invalidated, not counted. Baseline restored
byte-for-byte before subsequent tests or changes.

Selected GraphSolver list return still needs exact source owner escape and
original-arm lifetime transfer; exposing its pointer cell as an array is not a
solution. No service deployment, push or merge.

Combined-current mapping audit: 28 rows across 2 functions, 0 findings, 0 unsourced
facts/identities, latch CLOSED (owned_writer_mapping.log).

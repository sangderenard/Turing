2026-11-04 Frozen source evidence and owned-output boundary integration (MEASURED)

Validation checkout: C:/Users/alber/.codex/worktrees/compressor-validation/turing,
based on canonical 443e1fe5. Ready source-evidence, guarded intrinsic and owned
bytes output changes were copied here while tensor lowering continued in the
separate repair checkout. The experimental tensor_ssa_lowering changes are
absent. New tensor metadata declaration/audit hooks are inert without their
producer; no dynamic arange/topk implementation is claimed by this snapshot.

The bytes Add publisher and exact reached-return relay now run to a fixed
point before planning. `_sequence_concat_ops` reads the same class authority
instead of relying only on shallow aggregate attributes. Add remains in the
existing region instruction stream until precompile's incumbent concat
rewrite consumes it. An attempted metadata-only classifier was incorrect: it
removed the Add ticket and native output became empty. That classifier and
cache change were removed, and the departure is in AGENT_OBSTINANCE.md.

Public owned bytes concat results use the existing DYNAMIC descriptor and
native heap owner. C emits pointer-cell data/owner ports, native length and
capacity ports, and a release entry over the incumbent turing_heap_release.
Runtime ABI v3 and release text have explicit emission receipts derived from
the completed public ownership proof and descriptor cells. Host materialization
reads native lengths, copies while the owner is live, and releases in finally,
including failure. Public pointer-cell type follows the existing owned heap
value contract; its element dtype does not change it into a scalar int64.
The Python-free host writes a uint64 logical-length prefix and actual owned
payload, then releases native ownership.

Focused gate: 5 passed in 32.46s: guarded intrinsic integer ranges (2), owned
concat source/SSA/DLL/executable round trips (1), eager reached bytes class
relay (1), explicit narrow integer Cast width (1). Owned concat test covers
empty/high-byte/large fragments, repeated calls, surviving result copies,
idempotent native release and forced materialization-failure cleanup. It also
assembles real codec packets natively at the final concat stage and checks
exact Python compress_octets packet equality plus decoder round trips for
empty, single-byte, repeated and all-byte-alphabet inputs. Encoder stages in
this bounded check remain the Python reference, not a full native compressor.
Mapping audit: 28 rows, zero findings/unsourced, CLOSED. Whitespace check clean.

Previous exact full run33 on canonical 443e1fe5 stopped at 420.176s with the
reverse shape row for root value31. Its saved SSA at
C:/dev/Powershell/turing/build/compressor_native_33_failed_ssa_1.pkl is intact.
The append-slice helper formal reuse was effective, but planned region3 still
contained Add32 over storage31 and missing byte call21, while region4 Add33
used missing byte call27. The saved artifact predates this eager class relay.
Fresh whole-program lowering remains required; no complete compressor DLL or
whole compressor native round-trip claim is made here.

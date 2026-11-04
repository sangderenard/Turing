# Compiler integration checkpoint — 2026-11-04

User requested all agent-produced compiler work committed, merged, and pushed. This checkpoint includes the C/compressor work and the frozen direct Wasm patch; subsequent work is restricted to C and the compressor. The Meowdoku lane independently integrated its game files and the same Wasm patch in canonical commit 625b52d3; this merge preserves that commit and its files.

## Implemented work

- Runtime TopK shape/count selection uses source dimension evidence, checked integer conversions and products, zero-axis handling, and existing owned heap storage. The incumbent completion loop is shared by local and whole-program lowering.
- Native tensor wrapping refines an incumbent descriptor using its exact concordance receipt. Aggregate passthrough updates downstream Call/formal and metadata bindings. Frame-tail propagation publishes argument and Call-record views from the existing FRAME_TAIL record.
- Dynamic tensor returns cross ordinary source Calls with distinct semantic tensor and loaded pointer-carrier identities, forwarded owner cells, exact source/caller metadata projections, and caller-frame release. C uses the stored arena pointer rather than a pointer-cell address. Source-owned and existing lifetime audits validate those records.
- Generic Load type inference no longer copies an aggregate container dtype into a scalar field. Source-owned carrier projections retain their proved pointer contract. Accepted authored broadcast references are propagated between completion slices and source sections, with collision refusal.
- Frozen Wasm work includes callee CFG/Phi inlining, scoped pointer identities, typed primitive/arena work and four dedicated regressions. Full Meowdoku Wasm emission remains incomplete; older measured batch25 emission had 124 shortfalls and has not been remeasured here.

## Current verification

Focused gate: **27 passed, 2 failed**. All four Wasm regressions passed. Existing native TopK ensure cases, frame/Call binding tests, the generic pointer-field regression, runtime TopK selection/zero-axis/forged-proof tests, source metadata projection, reborrow audit, and authored helper reference test passed. The source-owned int64 DLL matches counts 0/1/7, twice per prepared execution, including values above 2**53 and forged-proof rejection.

The two enabled acceptance failures remain visible:
1. Float64 source-owned return: strict owner finalization rejects a consumer whose raw Add span output lacks its descriptor before C emission. Diagnostic: build/integration_checkpoint_20261104/failure_660485f2ca.pkl.
2. Actual _ascending_order: SSA block-geometry revision refuses REVISE without a changed source, before C emission. Diagnostic: build/integration_checkpoint_20261104/failure_ca8bc506e2.pkl (27 functions and attached book).

The bounded mapping audit is CLOSED: 28 rows, zero findings and zero unsourced facts/identities. git diff --check is clean apart from configured line-ending notices. Three older scalar-merge fixtures also fail on untouched canonical 349ca75f because they omit SSA identity roots; those pre-existing tests were not weakened.

No full compressor Run53, whole-compressor DLL, packet parity, or decoder round trip is claimed. The compressor goal remains active. Known unfinished work is preserved rather than disabled or described as complete.

# Continuation: dense compiler storage and bootstrap expansion policy

Date: 2026-10-14
Repository: sangderenard/Turing
Status: architectural audit and implementation requirements; no compiler changes.
Evidence: GitHub main inspected in this conversation; code-search snapshot c4023787de1008bc5a0d87dbf0026f31491d7f55. The receiving agent has newer changes unavailable to this audit and must reconcile observations with its tree.

Read alongside [the existing bootstrap handoff](compiler_convergence/12_bootstrap_handoff.md), applicable AGENTS.md instructions, and the current opening receipts in TEST_BASELINE_AND_HAZARDS.md. This report supplements that handoff; it does not supersede correctness blockers, source accounting, adaptive memory discipline or native verification.

## User intent

Audit the nature of provenance-book entries and compiler working objects that contribute to RSS. Assess whether densely packed ctypes structures, without per-entry Python keys or objects, could radically improve efficiency. Recommendations must be specific about architecture and feasibility without prescribing changes against an unavailable newer implementation.

Additional requirement: the bootstrap contract must explicitly communicate depth limits for loop unrolling and function inlining into custom versions. Both capabilities must remain available. The first Pythonic bootstrap should be able to avoid their optional expansion to reduce compiler complexity.

## Executive assessment

Dense typed storage is a strong architectural opportunity. The objective should be no persistent per-record Python objects in high-volume compiler storage, rather than no Python objects anywhere. Use contiguous records, integer handles, compact indexes and variable-length arenas, while retaining Python orchestration and explicit source-facing boundaries.

A mechanical replacement of dataclasses with individual ctypes instances will not achieve this. A Python list of separate ctypes structures retains wrappers, list pointers and allocation overhead; py_object fields preserve the referenced Python heap. Allocate records in chunked arrays or buffers and avoid retaining a wrapper for every record.

Packing and expansion control address different multipliers: packing reduces bytes per record; conservative bootstrap expansion reduces the number of records produced. Both are needed.

This was a static audit. No representative compilation or RSS benchmark ran. Several-fold savings in converted object-heavy structures are plausible, but no whole-process reduction or speedup is established.

## 1. What occupies memory

| Material | Contents | Dense-storage feasibility |
| --- | --- | --- |
| Book cells | Page, row identity, revision, fact, timestamp | Excellent |
| Provenance | Exact source/target cells, stages, mint and unresolved records | Excellent |
| Book indexes | Latest revision, histories, scope membership, reverse dependencies | Excellent with deliberate indexes |
| Structured facts | IDs, flags, shapes, spans, type/storage decisions | Mostly excellent |
| Graph/SSA | Nodes, edges, instructions, values, operands, blocks, descriptors | Excellent core; variable-length extensions |
| Source-facing leaves | ASTs, symbolic expressions, classes, callables, admitted arbitrary values | Mixed; explicit opaque boundary needed |

### Observed representation

IdentityPage maintains cells, stamps, row_columns, columns, column_positions and scopes through Python mappings and sequences. These indexes provide useful access, but add hash-table capacity, tuple objects, boxed values and pointer traversal around small payloads.

Provenance edges and reverse dependencies are themselves book rows. Their keys contain nested cell coordinates. Ref carries page, row and column; coordinate construction and hashing also cost CPU time. Sharing underlying objects means repeated coordinates must not be naively counted as fully duplicated payloads, but containers and indexes still cost memory.

SSAValue carries dtype, shape, device and an accounting mapping. Instr carries operand and role lists, an attribute mapping and optional source-span mapping. Blocks and functions add further containers.

The current system already has copy-on-read scope snapshots, compressed cold storage and a shared writable LiveTileArena. TiledObject preserves Python behavior by routing fields through writable tile maps. This controls residency, but decoded records remain object/container based. The next opportunity is compact active data, not another independent paging system.

Sources:
- [IdentityBook and IdentityPage](../src/compiler/identity_concordance.py)
- [Declared fact vocabulary](../src/compiler/concordance_declarations.py)
- [Live tile arena and object adapters](../src/compiler/live_tiles.py)
- [Book compression](../src/compiler/identity_compression.py)
- [SSA working objects](../src/transmogrifier/ssa.py)

## 2. Recommended book architecture

Treat the existing book as a typed versioned database with a provenance graph, preserving its API and semantics.

1. Assign compact vocabulary IDs to pages, stages, transforms, reasons and repeated symbols. Retain diagnostic names once.
2. Give rows and exact cells stable internal handles. Edges reference cell handles instead of repeatedly embedding page/row/revision coordinates.
3. Store common fact families as declared binary layouts. Variable-length fields reference separate arenas.
4. Store each canonical source-target-stage relationship once. Forward and reverse indexes carry edge IDs or offsets.
5. Preserve explicit latest-value, history and scope indexes. Compactness must not turn hot lookups into full scans.

Semantic guarantees remain mandatory: exact source revisions, construction order, concord conflicts, revision admission, minting, unresolved facts, unsourced latch behavior and snapshot visibility.

Stable handles alone do not solve versioning. The inspected writer can update an existing cell and its stamp, and scope snapshots depend on historical visibility. Separate logical cell identity, stored versions and timestamps, or provide an equivalent snapshot-preserving design. Do not assume every existing write is append-only.

“No keys” should mean no repeated general-purpose Python key objects in the high-volume representation. Identity lookup remains necessary. Use direct indexing for dense identities and typed sparse/composite indexes elsewhere. Some hashing is appropriate. External flagged or sparse semantic IDs must not determine array size: map them to compact internal handles without changing their meaning.

## 3. Fact layouts and variable material

Use three explicit categories:

- Fixed records: IDs, flags, enums, timestamps, positions and fixed-width numerics.
- Records plus arenas: strings, shapes, operand sequences, dependency lists and variable descriptor collections; records store offsets and lengths.
- Opaque references: genuinely live Python entities requiring identity or behavior, in a separately measured registry with explicit ownership/lifetime.

The declarations already identify many structured fact families, but some fact types admit arbitrary objects. Do not force arbitrary values into a universal maximum-size union, silently pickle every hot fact, or claim an opaque handle eliminates its referenced object's memory.

Use exact typed equality and canonicalization rules. Interning must preserve distinctions such as bool versus integer where semantics require them, identity-bearing objects, and floating-point bit patterns where relevant.

## 4. Working-object architecture

Use typed tables for SSA values, instruction headers, blocks, functions and descriptors. Instructions reference contiguous operand storage. Common attributes become declared fields or typed side tables; rare attributes use sparse extension storage.

Represent graph topology with integer node/edge handles and compact adjacency indexes. Mutable graphs need chunked storage or a compact base with mutable additions; avoid rebuilding every adjacency list after each edit. Instruction order must remain editable rather than assuming one immutable contiguous range per block.

Use phase-owned arrays, bitsets and queues for temporary analyses. Share immutable data across derived versions; release dead planning copies and recomputable analysis at valid ownership boundaries.

Adapt the existing tile framework to own typed record chunks where useful. Preserve its memory pressure, compression and disk tiers. Compact storage does not replace admission control or fix excessive overlapping lifetimes.

## 5. ctypes feasibility and constraints

ctypes is viable for prototyping layouts and exposing buffers. Specify the storage contract independently of ctypes so bulk operations can later use compiled code without redesigning persistence.

- Use arrays/chunks, not one allocated ctypes object per record.
- Prefer handles/offsets to raw pointers for relocation, sharing and disk storage.
- Keep wrappers transient; count any retained wrapper cache.
- Avoid universal _pack_=1. Field ordering and natural alignment often yield good density with simpler access.
- Validate widths and overflow. Narrow internal handles only where bounds are enforced; do not truncate semantic IDs.
- Python field access still produces Python-level values and runs interpreter loops. Packing alone does not guarantee faster traversal.
- Define schema versioning and portable encoding separately from host ABI layout if records are persisted or exchanged.

Memory gains can precede native execution. Large CPU gains additionally require removing repeated hashing/allocation and, where justified, batching work outside Python loops.

## 6. Expected gains and measurement

Several-fold reductions in particular object-heavy stores are plausible; an order of magnitude may be possible for records dominated by containers and nested coordinates. These are hypotheses.

For illustration only: if convertible structures account for 70% of peak memory and shrink fourfold, total memory becomes 30% + 70%/4 = 47.5% of baseline. A fourfold local improvement is approximately a twofold whole-process improvement in that example.

Remaining RSS includes AST/symbolic data, opaque leaves, caches, native libraries, temporary copies, allocator retention and concurrent workers. Serialized size is not decoded heap size, and released logical bytes need not immediately leave RSS.

Measure:
- Owned bytes per cell, edge, row, instruction and graph node, including indexes.
- Peak RSS and aggregate worker residency, wall time and allocation volume.
- Decoded versus compressed bytes, opaque bytes and wrapper/index overhead.
- Temporary migration, compression and compaction peaks.
- Small and large representative workloads with unchanged semantics and budgets.

The existing [memory probe](../tools/compiler_probes/profile_memory_n2.py) is a starting point. Its sampled peak and estimated histogram are not exhaustive ownership accounting.

## 7. Bootstrap contract: explicit expansion depth

### Observed starting point

[WorkContract](../src/compiler/work_contract.py) embeds LoopOptimizationContract. In the inspected tree, unroll_limit defaults to 8 and requires a positive value; it is described as a low-priority unroll threshold. That is a size threshold, not an explicit nesting-depth policy or a supported off switch.

The inspected work contract also distinguishes per-signature and per-callsite descriptor derivation. Descriptor reuse is not equivalent to controlling body inlining or creation of executable custom versions.

The existing bootstrap handoff discusses subdivision depth and fixed points. Those are different from optimization expansion depth. Do not reuse one ambiguous “depth” setting for all these faculties.

This audit did not establish every inlining/specialization consumer or prove that no equivalent depth control exists elsewhere. Trace the receiving tree before adding fields.

### Required architecture

The resolved bootstrap contract must independently state:

| Axis | Meaning |
| --- | --- |
| Loop unroll depth | Maximum nested optional unroll expansions along a source/control expansion lineage |
| Unroll size/factor | Permitted replication of an individual loop body |
| Function inline depth | Maximum nested optional body substitutions along a call-expansion lineage |
| Custom-version depth | Maximum successive specialization generations from a generic function |
| Growth budgets | Bounds on generated nodes/instructions and variant counts, in addition to depth |

Inlining and specialization are related but distinct. A custom function version may remain callable without being inlined; an inline substitution may not require a new persistent custom version. If the implementation exposes a combined control, it must document and enforce both effects.

Use an explicit disabled state or a documented zero depth for each optional expansion. Depth zero means preserve loops/calls and do not create optional custom versions. It does not mean drop a loop, omit a call, stop source discovery or claim unsupported code is complete.

Define depth relative to the original compilation root/lineage, not a helper invocation or worker. A copied graph, child compilation job, fixed-point round or cache lookup must not silently reset the remaining budget. Nested unrolls and nested inlining can multiply each other, so depth limits also need total growth limits.

Recursive calls require cycle-aware handling. Hitting a depth limit retains a supported callable boundary; it must not truncate recursion or its observable effects.

### Recommended first Pythonic bootstrap profile

Default optional loop unrolling, optional body inlining and optional custom-version proliferation to disabled. Preserve structured loops, ordinary calls and shared generic bodies with the existing exact ABI.

This is a bootstrap profile, not removal of optimization capabilities or a global change to deployment presets. Later bootstrap stages can opt into bounded expansion after the simpler route verifies and its cost is measured.

Necessary semantic adaptation is not optional optimization. If lowering needs type-specific representations or closure/storage adaptation, retain the minimum correct mechanism and account for it explicitly. If a backend cannot represent a retained loop/call, report a precise unsupported frontier or use an explicitly authorized lowering policy; never silently turn optimization back on.

Do not confuse optimization limits with source pursuit. The complete required source/call closure must still be discovered, represented and accounted for. Existing completeness gates remain authoritative.

### Contract communication and enforcement

Resolve policy once at the bootstrap entry and propagate its effective values through extraction/planning, loop realization, call/version planning, child requests, workers and backend handoff wherever those layers perform the transformations.

Publish requested and effective policy in the existing contract description, product manifest and book policy/decision receipts. Record applied expansions and refused/deferred optional expansions using existing accounting mechanisms, with original identities and reasons.

Include transformation-affecting effective values in relevant cache identities. A cached expanded body must not bypass a disabled-depth contract. Verify compatibility or regenerate. A preset name alone is insufficient if effective settings differ.

Separate compiler-owned expansion from downstream native optimizer behavior. A disabled source inliner does not automatically disable LLVM/C compiler inlining. State which layer each policy governs; if the bootstrap needs downstream restrictions too, communicate and verify them separately.

Unknown or unsupported requested settings must fail clearly. Do not accept a field that has no consumer. Preserve the contract's existing refusal doctrine.

## 8. Acceptance checks for the receiving agent

Storage:
- Compare decoded facts, histories, exact provenance, conflicts, snapshots and emitted/native behavior before and after.
- Exercise aliasing, mutation, opaque identity, sparse IDs and serialization.
- Measure indexes and transient peaks rather than reporting payload bytes alone.

Bootstrap expansion:
- Zero-depth profile retains real loops and call edges with no optional expansion.
- Depth one permits one lineage level; deeper nested candidates remain structured/callable.
- Separate unroll size from nesting depth; exercise nested loops and mixed call/loop expansion.
- Exercise recursive calls, mutual recursion, custom-version fanout and repeated fixed-point rounds.
- Confirm worker/child propagation and cache compatibility.
- Preserve closures, mutable aliases, effects, exceptions and completeness diagnostics.
- Compare actual bootstrap time/RSS under identical memory and admission budgets. A completed receipt alone is not native execution proof.

## 9. Suggested order of work

1. Reconcile this report with newer changes and inventory actual expansion consumers.
2. Establish and verify the conservative bootstrap contract, including effective receipts and cache behavior.
3. Measure ownership at the peak with that profile.
4. Compact book coordinates, cell references and provenance edges first.
5. Convert the dominant fact families, then active graph/SSA stores as measurements justify.
6. Preserve adaptive residency; move bulk hot operations into compiled code only where evidence warrants it.
7. Re-enable bounded optimization selectively for later stages and compare end-to-end outcomes.

Do not claim the bootstrap contract is fixed merely because this report is committed. This commit is the architectural handoff; wiring, correctness validation and performance measurements remain implementation work.

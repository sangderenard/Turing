# Privileged source-only compilation through Site and Nodus

This is an extension of the existing divider request adapter, not another
splitter, compiler, queue, or runtime transport. Ordinary numeric row requests
keep their existing AbstractTensor contract. Source-only attempts send the
complete authored Python source and the existing divider's qualified names to
the same Nodus compile host. That host calls the existing
`source_divider._attempt_main`, `partition_authored_source`, and
`fortran_c_shell.lower_ast_source_to_ssa` before the selected source emitter.

## Explicit permission and admission

All defaults are off. Existing accounts, including admins, acquire no new
capability from this implementation.

- The startup SiteCompiler configuration must explicitly contain the boolean
  `allow_unstable_source: true`.
- The signed-in account's current role must explicitly grant
  `compile_unstable`, as well as `compile_submit`. Tier or anonymous LAN access
  never grants the unstable capability.
- Each upload must opt in with `X-Compile-Unstable: source-only`. The page's
  source-only checkbox is unchecked initially.
- Unannotated admission is a separate unchecked choice. It requires the
  source-only choice and `X-Compile-Inference: allow` under the same current
  capability and operator gate. It lets the existing compiler try to resolve
  types; it does not supply or guarantee types.

The Site records the two booleans in its own upload receipt outside uploaded
source. The annotation plan remains truthful: an unannotated piece does not
become annotated just because the attempt is admitted. Current account identity
and both capabilities are rechecked during request preparation and immediately
before the native contract is submitted. A removed account, changed role, or
revoked policy blocks dispatch. Source metadata is intent, not authority.

No production policy, account, startup configuration, or service is modified by
these source changes. Only trusted compiler accounts should receive the grant.
Source-only mode is not an OS sandbox and does not relax the existing deployment
isolation requirements in `SITE_COMPILER_ARTIFACTS_2026-11-11.md`.

## Request and result boundary

The source-only request uses `unstable_source_only: true` and `source_names`
instead of `row_entry`, `columns`, or a synthetic `program_abi`. The selected
extraction contract is copied unchanged. Real source text, qualified nested
names (including `<locals>`), annotations, and callsites are preserved.
There are no invented root-call samples or numeric tensor substitutions.

Nodus handles this mode in its existing row-tool adapter. Its product schema
is `nodus-turing-source-only-product-v1`. Successful emission is
`emitted-not-built`, `loadable: false`, with `binary`, `buffer_abi`, and
`adoption_receipt` all null. The native host reports status 4 (`EmittedNotBuilt`)
and bypasses adoption even when its ordinary row configuration enables it.
Existing runnable receipts cannot be reused for a source-only request.

The Site accepts source-only success only when backend, source names, source
digest, and account/workspace context match the submitted request and each
entry says emitted-not-built. A missing or mismatched receipt cannot become a
successful executable result. Real compiler failures retain their stage, error
class, and diagnostic in the existing attempt row. All products and diagnostic
files remain in the account's charged external workspace.

## Runtime ABI is a separate boundary

Nodus already has GenericObject references and typed byte/pointer FIFOs.
Turing already has SCHEMA V1 interchange, real root-call evidence, external
reference codecs, and in-process object boundary mechanisms. This mode does
not replace any of them and does not claim they form a callable product binder.
No generic runtime binding, closure-cell native ABI, root-call serialization,
GPU link, or rendering follows merely from granting source admission.

## Measured original-source trial

The full annotated `spectral-analyzer/demo_pluck_gl.py` was uploaded unchanged
as a zip through an isolated signed-in Site instance. Its SHA-256 was
`903dba7c938a77e7881058b0c8db0732524955440407ea040fcfe57fed1e88b1`
(859,902 bytes). The original live upload and live services were untouched.

The existing queue selected `main.<locals>._doc_id(key: str) -> int`, piece 1.
The first attempt exposed a new adapter name-validation gap before lowering;
the adapter was corrected to accept the existing qualified-name convention.
The next attempt reached the real compiler through the native loopback
listener, native FIFO negotiator, and compile host. It failed at `ssa-lowering`:

> region value 8 (IndexedStore in function '_doc_id') has no settled dtype in
> the tensor, specialization, node, descriptor, planned scalar types, its
> operands, its call actuals or domain

The native host returned 17 (`CompileFailed`). The product was non-loadable,
all adoption counters were zero, and no source emission, native/GPU link, or
render occurred. The Site recorded the actual ValueError and lowering stage.
The failure receipt and complete source are retained under
`C:/Users/alber/AppData/Local/Nodus/UserFiles/original-source-trial-4d9a18260ad149a59fed8c54d60d7740`.
That trial is evidence of reaching the compiler, not evidence that all 377
pieces compile.

The existing nested partitioner was then found to discard the actual lexical
declaration `_doc_misc_ids: dict[str, int] = {}` (original line 15639). The narrow
repair retains dependencies in their defining lexical scope, without changing
source lines or adding dtype defaults. Tests cover captured declarations,
aliases, parameter shadowing, nested scopes, and class namespaces. The mapping
concordance audit reports no unsourced facts or identities.

One full-source retry after that repair reached the next real boundary:

> region value 0 (input '_doc_misc_ids' in function '_doc_id') has no settled
> dtype in the tensor, specialization, node, descriptor, planned scalar types,
> its operands, its call actuals or domain

That final receipt is retained under
`C:/Users/alber/AppData/Local/Nodus/UserFiles/original-source-trial-bb841e6f357342729b2d9e46722d8624`,
job `ec8dc8592961`, contract `7447874525838744298`. The authored mapping is now
recognized as a closure and writable two-column dictionary storage. No runtime
call evidence, homogeneous dtype, or native closure ABI was invented to pass
the refusal. This remains an unsuccessful source-only attempt; the native host
exited normally after reporting status 17, all adoption counters remained zero,
and the isolated Site instance stopped.

Final focused verification: 171 Turing tests passed (32.27 seconds), 8 selected
partition tests passed, and the mapping audit had 0 findings. The paired Nodus
branch at `f5bf718` had 85 Python tests pass, 4 existing compiler tests skipped,
and native host regression tests pass with aborting adoption/plugin stubs.
The CPU-only native trial harness is separate from source-product compilation;
no original-source native/GPU product was linked or rendered. No full-suite or
production deployment claim is made.

## Declared closure contracts and planner checkpoint

Further tracing found two missing handoffs in the existing compiler. The
declared mapping column facts existed on the parent binding but were not
transported to its captured input; the planner then skipped ordinary closure
inputs before its existing mapping-value-class rule. The repair transports a
whitelist of declared mapping facts along the actual closure capture edge,
posts existing `SEQUENCE_ROW_DTYPE` evidence with source provenance, and admits
only the matching declared capture to the existing mapping typing rule.
String keys retain their real string semantic class and string-table token
storage. No scalar width is selected by the admission gate.

Negative tests cover missing capture evidence, shadowed variables, mismatched
declarations and receipts for another capture. Partition review also corrected
global/nonlocal ownership and read-modify-write declaration chains using the
existing Python symbol-table authority. Exact source lines remain unchanged.

The next full original-source trial cleared the dtype failures and reached
final SSA concordance. It refused three generated mapping helpers (contains,
store and required lookup) with:

> elementwise span input has no matching physical span result

Retained root:
`C:/Users/alber/AppData/Local/Nodus/UserFiles/original-source-trial-011a8ab141cc48d188d31c5535060e48`,
job `791221a8c798`, contract `6229681601753546779`. The manifest records
`ConcordanceRefusal`, and all adoption counters remain zero. This is still a
failed source-only attempt, with no emission or runnable product. The detector
was not disabled; its helper classification/ABI finding is the next boundary
to investigate.

At this checkpoint 192 combined Turing tests passed (35.25 seconds), 33 focused
planner/mapping/capture/structural-table tests passed, and the mapping audit
still had zero findings or unsourced facts/identities. Early test invocations
created ignored pytest/bytecode caches in the owned worktree; these are test
caches, not compiler products, and were moved to external user space. Further
tests should select an external pytest root and cache prefix as well as the
external temporary directory.

## Verified string-key ABI and reference boundary

The retained original-source log confirms a real representation mismatch.
Authored `key: str` has rank-one int64 glyph-span storage. The dictionary key
column instead stores scalar int64 string-table tokens. Frame binding gives
the scalar helper query formal the caller's span facts, while the helper still
compares that query with a scalar stored key and expects one boolean. The store
helper also writes the query directly into its scalar key column.

These contracts come from `fortran_c_shell.py` (string ABI and frame binding),
`ir_sequence_tables.py` (scalar key helpers), and `identity_concordance.py`
(elementwise operand/result checking). `_table_query_values` in
`precompile_to_ssa.py` converts established opaque references to integer handles,
not string spans to tokens. `ir_string_interning.py` handles actual literals,
not this runtime parameter. Existing resident-string equality requires exact
descriptors and full-content comparison and explicitly forbids hash-only
substitution. No safe dynamic conversion covering lookup and insertion was
established. The concordance refusal must remain active.

The compiler's reference semantics are stronger than symbol names alone. The
retained log records `sequence_arena_by_reference`; contains/store/lookup share
the same column IDs and length, capacity and status-address identities. The
actual/formal storage receipts connect those exact values. This is shared
arena-storage intent inside attempted SSA, not a dictionary copy per helper.
See `ssa_call_input_adapters.py:1715-1746` and the retained log's sequence arena,
call-input-storage and closure-capture pages. Only declared metadata is copied
by the new propagation, not runtime dictionary contents.

That does not prove a runtime pointer to the original CPython dictionary. No
executable was produced or run. The Site sends source/configuration JSON; the
Nodus FIFO carries the compile contract ID. This route does not bind the live
dictionary into a Nodus callable. Existing GenericObject/pointer/byte FIFO
mechanisms were neither replaced nor claimed as newly implemented.

Recommended next boundary: first report the incompatible string-query/key
contract at the table-call edge, then design/test compatible stored-key
descriptors, full string equality and insertion using existing compiler
mechanisms. Do not grow Nodus to compensate for this compiler ABI mismatch.
Runtime closure publication remains a separate, unproved binding/lifetime task.

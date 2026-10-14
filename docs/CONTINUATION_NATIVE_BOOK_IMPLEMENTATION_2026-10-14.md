# Native book implementation continuation — 2026-10-14

Branch: `codex/dense-storage-bootstrap-depth-2026-10-14`.
Read with [the architecture audit](CONTINUATION_DENSE_STORAGE_BOOTSTRAP_DEPTH_2026-10-14.md). This receipt supersedes its original interpretation of depth: limits stop optional expansion, never source discovery, call execution, loop iterations, or dependency retention.

## Required architecture

Compile the existing complete book/concordance implementation into one native compartment. Book operators, accessors, history, scopes, concordance checks, provenance edges and indexes must execute internally on native data. Python crosses an operator boundary; it must not dispatch every field access or convert each internal record into a Python object. Per-accessor accelerators and Python ctypes record adapters were rejected and are not integrated.

The bootstrap must build and verify this compartment first, install its persistent-state operator ABI, then restart compiler workers so subsequent incremental products use native books. That scheduling/installation is NOT implemented on this branch. Do not treat a compile attempt, emitted artifact, or a reduced reference object as native book activation.

## Measured results and rejected approach

Small sourced-chain workload: 10,000 facts, 29,999 materialized cells, five repetitions; medians; timings measured separately from tracemalloc. These are Python allocation measurements, not process RSS and not whole-compiler speedups.

| Measurement | Baseline | Branch, slotted Ref | Rejected packed Python adapter |
|---|---:|---:|---:|
| Post wall seconds | 0.169120 | 0.150998 | 0.179256 |
| Latest-read wall seconds | 0.006867 | 0.007419 | 0.010158 |
| Traced retained bytes | 18,169,368 | 18,169,368 | 18,513,260 |
| Ref bytes including instance dictionary | 144 | 64 | 64 |
| 10,000 cached hashes, seconds | 0.001212 | 0.000917 | 0.000960 |

The adapter increased retained memory by 343,892 bytes and slowed reads about 48%; it was removed. An earlier struct-per-access prototype slowed timestamp scans about 16.6 times and was also rejected. The retained Ref change saves 80 bytes per separately retained reference; this workload retains coordinates instead and showed NO book allocation saving. Small timing differences are workload observations, not established end-to-end gains.

Reproduce the retained benchmark with `python tools/compiler_probes/benchmark_dense_book.py --root /path/to/checkout --cells 10000 --repeats 5`, selecting separate baseline and branch processes.

## Whole-module compilation attempt

Submitted all of `src.compiler.identity_concordance` through the public source-to-SSA compiler with optional expansion disabled and the full-native execution contract. This is not a hand-translated replacement implementation.

The first failure exposed annotation rows indexed by a lifted function's local AST name. Both the module `mint_scope` and `IdentityBook.mint_scope` became `mint_scope`; annotations `Any` and `Stage | None` then conflicted. A minimal two-definition test reproduced this on untouched main. Commit `12a14051d37e2483b945c4cdcc5333112b598a5c` introduced checked annotation posts; its predecessor used dictionary overwrite at this boundary. This explains why an older compiler need not raise the same exception; it does not establish that previous compiles actually encountered a collision.

After the ingestion writer correction, extraction reached 58,522 nodes, 63,437 edges and 493 function graphs. Deployment then failed with `shader-region reduction requires an acyclic graph`. Last sampled RSS was 2,610,340 KiB; this is a sample, not measured peak RSS. No native book library was produced or installed. Later canonical-reader changes were not loaded in that already-running process.

`tools/bootstrap_compiler_concordance.py` preserves the ordinary whole-module attempt, compiler/source fingerprint, effective depth policy, failure receipt and available compiler checkpoints. It has NOT successfully reached SSA or native emission. The existing whole-object library lowering already emits one module; investigate how to elect the complete operator surface without treating the declaration catalogue as an executable shader region. Preserve real module initialization and all control/calls. Do not break cycles by deleting work or substitute tiny operator builds.

## Identity rules and changes

A name following `.` is an attribute binding, not a global identity. Never use a bare method/function name, reconstructed `owner.name`, a unique-looking suffix, or “only matching name” as a provenance index. Carry the canonical source identity or the book's declared reference. Missing identity must remain missing or fail explicitly; guessing makes correctness depend on what other source happens to have been compiled.

Implemented on this branch:

- Annotation rows and their derived catalogue use `(module, authored definition identity)` obtained from source-span provenance. Readers require that exact identity.
- Extracted function graphs carry `authored_function_identity`; lowered functions preserve it. Lambdas retain their full `(module, enclosing definition, AST path)` source-span identity, rather than inheriting their enclosing function's key. Canonical identities supersede local FunctionTable spelling aliases.
- Source-graph reconciliation no longer indexes bare function names or retries dotted suffixes. Conflicting exact graph keys raise instead of first-writer selection.
- Numeric concordance scopes use the full authored identity; synthetic graph scopes are book-minted, never derived from diagnostic names. Specialization scopes include the complete base identity and specialization digest. Downstream numeric/precision readers carry the same keys.
- Reachability's callee field is a scope, not a NAME string. This is a persisted row-layout change; older local-name numeric receipts must be rebuilt, not aliased back in.

This is NOT a certification that every compiler subsystem is free of suffix matching. The remaining class/record ABI resolvers in `fortran_c_shell`, and class recognition in the reducer, still contain suffix comparisons and require provenance-led review. Splitting a dotted name to execute Python attribute navigation is distinct from using that suffix as a global index. Do not delete all string splitting indiscriminately.

## Expansion policy

The work/bootstrap contracts communicate independent `unroll_depth`, `inline_depth`, and `custom_version_depth`. Bootstrap defaults are zero; ordinary compiler defaults remain unbounded (`None`). Policies travel through worker environments, receipts, version identities and the bootstrap CLI. Zero retains complete loops and ordinary calls. At a custom-version bound the generic callable shell remains available; required ABI/descriptor propagation continues.

Positive unroll depth currently measures lexical containment. The existing host-linear inliner has one expansion level. These are not a claim that every compiler expansion mechanism has a unified recursive lineage counter. Extend that contract before promising arbitrary recursive expansion-depth semantics.

## Validation and remaining acceptance

Before the final numeric-scope extension, focused tests recorded 105 passes and one host-linear metadata failure also reproduced on untouched main. Mapping audit recorded zero findings and the same 21 unsourced facts as baseline. These results must not be represented as verification of later edits.

The final branch receipt below records the latest focused runs. Whole native book equivalence, memory improvement and speed have NOT been demonstrated. Required acceptance remains: post/concord/revise/history, exact edge provenance, scopes and snapshots, copy/spill/compression/resume, deterministic diagnostics, and a restarted real compiler using the verified native book. Measure fresh-process RSS and wall/CPU time against untouched main with the same source/policy. Do not recommend merging as a completed native storage replacement before those pass.

Latest focused verification after numeric-scope changes:

- Seven canonical identity regressions pass, including separate modules/classes, missing suffix fallback, conflicting identity rejection, book-minted synthetic scopes, and distinct lambda AST paths.
- Source-graph reconciliation, copy-on-read scopes and loop composer group: 99 passed in 18.13 seconds.
- Annotation/depth/numeric group reached 26 passes; a larger repeated-division compiler test lowered to SSA but failed on an absent precision-pipeline receipt. The same test on untouched main failed at the identical absent-receipt assertion (13.54 seconds). The branch run took 38.01 seconds with other test activity overlapping; these runs are correctness comparisons, not controlled timing evidence.
- Mapping audit: zero findings, 26 rows, two functions, 21 unsourced facts and zero unsourced identities, matching the previous baseline. The latch remains open; this is not a fully sourced compile claim.

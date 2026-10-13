# Bootstrap performance investigation, 2026-10-13

The outer compiler bootstrap is not yet verified. Repeated actual runs through
`tools.bootstrap_compiler_whole_program` exposed substantial avoidable work,
then reached live-storage pressure during deployment selection. The four
authored nested helpers already publish partial repository-SSA receipts, so
finishing the outer lowering alone cannot seal this bootstrap.

## Measured costs and changes

The comparison uses unchanged main `61a4818ae63c89cc160597f4550eec70d96ec0c4`
and the performance worktree, on Python 3.12 with four CPU quota cores,
16-GiB cgroup RAM, no swap, and the pure Python tensor backend. Runs use
`--jobs 2 --min-workers 1`, the default 10-GiB worker/aggregate budgets,
2-GiB reserve and existing 75-percent decoded-front elasticity. No memory
regulation was disabled or budget raised to obtain these results.

| Observation | Baseline | Changed | Scope |
| --- | ---: | ---: | --- |
| Source closure | 133.78 s | 105.91 s | Actual bootstrap pass |
| Tensor callsite specialization | 581.12 s | 185.07 s | Actual bootstrap pass, about 3.14x |
| Arrival at catalogue control partition | 1058.66 s | 591.62 s | Partial bootstrap, before empty-partition fix |
| 100 repeated class source-availability probes | 4.690 s | 0.051 s | Repeated exact-definition lookup microbenchmark |
| 10,000 discarded graph progress messages | 10,002 RSS reads | 2 RSS reads | Fixed-clock reporting microbenchmark |
| 100,000 scalar adoption calls | 0.0568 s | 0.0040 s | Exact scalar dispatch microbenchmark |
| Ten repeated 1-MiB LZMA tile relocations | 0.3268 s | 0.0045 s | Validated-copy microbenchmark; equal disk bytes |

These are component measurements, not a claimed whole-bootstrap speedup.
Compiler source edits themselves slightly enlarge the resolved source graph.

The changes preserve compiler facts and storage decisions:

- Discarded graph-detail events no longer read process memory. Retained
  snapshots and regulator/admission reads remain live.
- Source availability is cached by exact definition identity within one
  extraction contract, bounded to 256 entries. Each occurrence still goes
  through contract classification; a derived contract owns a fresh cache.
- Lexical graph extraction finds terminal-merge candidates once. Recursive
  membership uses one immutable address set. Copy-on-read projection counts
  compute the same union cardinality without constructing its complement.
- Planner progress samples whole-graph counts at most once per second during
  progress, and records their measurement timestamp. Phase boundaries retain
  exact counts. Identity/concordance accounting remains unchanged.
- An explicitly empty executable-node tuple returns an empty control
  partition immediately. The top-level source catalogue deliberately supplies
  this tuple; its functions are planned separately. It previously built broad
  branch/descendant maps only to return no partition keys.
- Exact scalar atoms bypass tiled-container dispatch and conversion-memo
  retention. Memo overrides remain authoritative; subclasses take the original
  path. Mutable containers, aliases, cycles and object adoption are unchanged.
- Disk compaction validates and relocates existing LZMA payloads without
  encoding them again. Fresh, modified, zlib and raw tiles still receive the
  existing bounded deeper compression pass. Checksums, atomic index commit,
  disk reclamation and pressure tiers remain in place. Disk headers and codecs
  retain their existing format.

## The remaining RAM and subdivision issue

The run with the empty-partition fix crossed the 10-GiB worker threshold while
planning roughly 160 of 621 function shells. Initial live-storage adoption and
successive full disk compactions then dominated samples. It was deliberately
stopped after about 31 minutes to test the storage changes consistently; it
did not produce a complete outer unit. The later storage run has a separate
output directory so it cannot reuse an earlier compiler's product. That run
reconfirmed all four partial helper receipts and was stopped at 337 seconds
during lexical extraction once the existing readiness failure was established.
The new storage optimizations are validated by forced-pressure/native tests
and focused measurements; their full-bootstrap benefit remains unmeasured.
No background bootstrap worker was left running.

Current automatic compilation forms its region jobs during SSA expansion,
after the broad catalogue/deployment plan is built. More region threads do not
remove this earlier peak. The existing earlier boundary is
`compile_partitioned_source`: resolve/reduce once, publish scoped dependency
units, release planning graphs, then admit bounded unit workers. With native
interface tiles, workers retain one strongly connected component's bodies and
import accepted declarations/objects for dependencies. The older dependency
closure mode can repeat broad bodies across workers.

`publish_resolved_process_graph_inputs` already refuses oversized cuts before
serializing them. `compile_process_graph_creep` already consumes deterministic
child subdivision plans with a depth limit and repeated-plan fixed-point
detection. These mechanisms should be extended rather than replaced with a
second graph or heuristic call boundary.

The next methodology change should put this existing scoped boundary before
deployment selection for the bootstrap, then use measured unit RSS/time to
adjust admission and cut sizes. Keep broad optimization within a bounded
strongly connected component; permit larger units when headroom allows. A cut
requires an accepted interface for captures, records, mutation effects and
calls. Stateful regions may need their complete authored function shell;
unsupported interfaces must remain explicit frontiers. Source-node count is a
useful guard, not a promise of an RSS cap: one large fact is indivisible.

This investigation does not silently replace holistic `auto` with that legacy
route or claim arbitrary recursive regions have a valid linking ABI. Existing
scoped-input tests validate exact nodes/edges/scopes/addresses, release before
workers, refusal of oversized closures, and actual native calls across units.

## Existing bootstrap correctness frontier

The `_lower_process_graph_to_compilation._report` helper's actual `unit.json`
reports missing used parameter `message`, missing lexical capture `progress`,
and an unexplained root argument used by a control-expression comparison.
`repository_ssa_complete` is false. The same receipt occurs on unchanged main.
All four authored helpers remain partial. The publisher correctly prevents
complete planner-owned integrals from upgrading an incomplete enclosing
method, and the whole-program builder requires every source record and its
native verification to be complete. This is evidence of a bootstrap failure,
not proof of which source/ABI writer caused it.

## Validation and reproduction

Use the canonical public source entry. For the actual bootstrap:

```sh
python -u -m tools.bootstrap_compiler_whole_program \
  --output build/bootstrap-performance --jobs 2 --min-workers 1
```

The workspace used isolated dependencies, GCC for CFFI, a writable Zig cache,
and joblib 1.5.2 (1.6 lacks the existing `joblib.externals.cloudpickle` import).
Neither PyTorch nor the unavailable nodus core was installed to mask failures.
An external sampling thread observed the actual compiler stack; it did not
modify compiler planning or semantics.

The final focused checks passed 90 storage/memory/scope/planner tests (including
the retried forced-disk native execution), ten scoped-input tests with real
native interface linkage, and 129 project/contract/compression/bootstrap tests.
That last group also encountered the existing optional-torch failure described
below. The mapping concordance audit retained zero identity
findings and the same 21 visible unsourced facts. No full test suite was run.

Broader diagnostics also encountered existing failures, reproduced on unchanged
main: missing optional torch backend in the autograd source test, seven older
progress UI expectations, aggregate parameter/call expectations, and undefined
LLVM switch operands in imported `binary_value`. These are recorded as
frontiers, not suppressed or counted as new regressions. A first forced-disk
native check used an unwritable default Zig cache; rerunning with the workspace
cache passed. The compact machine-readable observations and helper frontiers
are in [the evidence receipt](evidence/11_bootstrap_performance.json).

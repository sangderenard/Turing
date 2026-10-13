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

Before the follow-up repair below, the `_lower_process_graph_to_compilation._report` helper's actual `unit.json`
reports missing used parameter `message`, missing lexical capture `progress`,
and an unexplained root argument used by a control-expression comparison.
`repository_ssa_complete` is false. The same receipt occurs on unchanged main.
All four authored helpers remain partial. The publisher correctly prevents
complete planner-owned integrals from upgrading an incomplete enclosing
method, and the whole-program builder requires every source record and its
native verification to be complete. This is evidence of a bootstrap failure,
not proof of which source/ABI writer caused it.

## Follow-up: exact closure accounting repair

A real three-line nested numeric function (`inner(x): return x + captured`)
reproduced a false completeness failure in about one second. Its reduced graph
marks `captured` as a closure Input. `_record_module_closure_formals` publishes
the same surviving argument ID under `closure_formals`, with
`priority=exact_source_capture_identity`. The existing SSA signature checker
already accepts that receipt. The project publisher ignored it in both the
represented-parameter check and the unexplained-root-argument check.

The publisher now consumes that existing receipt for the exact surviving
argument ID. A stale receipt for a different ID does not explain another
argument. The numeric nested function now has complete repository SSA; emitted
LLVM native execution matched Python for three pairs of explicit parameter
and capture inputs. No fabricated parameter, guessed name correlation or
relaxed missing-parameter gate was added.

Fresh compilation of all four actual bootstrap helpers gives:

| Helper | Still missing parameters | Still missing captures | Unexplained root IDs |
| --- | --- | --- | --- |
| `_report` | `message` | none | none |
| `close_region_feed_set` | `program` | none | 9, 8, 6, 5, 7 |
| `rooted_attribute_path` | `expression` | `root_parameters` | 4, 40 |
| `collect_hierarchy_values` | `closure` | `collect_hierarchy_values`, `executable_value_ids` | none |

All four remain partial. `_report`'s source graph contains Input 0 (`message`),
Input 1 (`progress`), and Call 4 with callee/argument edges to those exact
inputs. Its conditional control block retains `body_callsite_ids=(4,)`.
An observation hook at `_schedule_loop_callsites` sees an empty hierarchy plan;
there is no `PlanCall` for that call to schedule. The final SSA true arm contains
only a branch, and `message` has no formal. Correcting capture accounting does
not provide a native invocation plan for this unbound callback/string call.
It remains refused by the used-parameter gate.

A no-argument callback variant exposed why that gate alone is insufficient:
after recognizing its real capture, the omitted call could otherwise make the
unit look complete. The compiler now records a surviving Call whose exact
callee edge names an unbound parameter/closure Input and has no planned or
lowered invocation in the existing unresolved `SSACallRecord` table. Existing
resolved references and already represented calls are preserved. The record
derives from the source node cells, carries the callee input ID in the existing
diagnostic channel, and is consumed by existing extraction/native link gates.
Records are published once per function rather than copying the growing tuple
for every omitted call. `_report` now explicitly reports unresolved call 4 to
`progress`; the no-argument variant also stays incomplete. No callback ABI or
new opcode is invented.

The other helpers manipulate `FusedProgram` fields, AST objects, and recursive
hierarchy objects/captured mutable sets. Their observed missing input/capture
representation and anonymous formals are still real frontiers. This pass does
not identify every upstream writer responsible for those losses, or claim
that a record ABI annotation alone would fix them. Their exact fresh receipts
remain in `build/bootstrap-helper-accounting-final/<helper>/unit.json` in the
workspace. Full bootstrap was not rerun for this narrowly reproduced repair.

The final focused project/bootstrap/scoped-input gate passed 99 tests. Two
broader existing shaped-closure native checks failed on undefined operands in
imported `binary_value`, with the same two failures reproduced on unchanged
main. The new scalar closure's three LLVM native pairs passed. The mapping
audit before and after retains zero identity findings and 21 unsourced
facts. The earlier component measurements above remain historical measurements
of the performance change, not measurements of this accounting repair.

## Validation and reproduction

### Shell channel investigation

The existing architecture provides shell information channels. The shared
`shell_telemetry.TelemetryChannel` carries log/error/profile/progress/trace
records. `CompileProgress.progress_callback` supplies the current compiler's
Python diagnostic route, and `record_launch_profile` forwards native launch
profiles into that same stream. `_lower_process_graph_to_compilation` receives
this callback from its caller.

The existing C native shell provides `turing_launch_logger(user, profile)` and
an optional bounded `TuringTraceRing`. Standalone C main supplies the native
logger and region context to `turing_profiled_launch_ex`; compute can be a C,
LLVM, GLSL adapter, or Fortran closure. Native ring entries contain sequence,
shell/device durations, region and status. That ABI is not the same signature
as `_report`'s `progress(str)`, and its records do not contain arbitrary message
strings. Shell file/external-reference boundaries have their own explicit API
metadata. The isolated authored bootstrap request supplies no declaration
binding the `progress` capture to a shell endpoint.

Therefore the observed unresolved call is a missing connection to an existing
shell facility, not evidence that shells lack diagnostics. Routing must retain
the existing endpoint identity and invocation/text representation. This pass
does not invent another channel, equate the different callback signatures, or
mark the standalone helper complete without that binding.

The native shell also had a reproducible LP64 Linux build error: its callback
declaration accepts `unsigned long long *device_ns`, but two local variables
used `uint64_t`, which is `unsigned long` on this platform. The locals now
match the existing ABI. Both shell tests failed on unchanged main and passed
after the fix; two shared-telemetry checks also passed. An executable compiled
from the existing native shell sources with pointer-type errors enabled made
three native launches into a two-record ring, retained sequences 1/2 for
region 42/status 1, and correctly reported one overwritten record. No Python
callback entered that native launch/logger path.

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

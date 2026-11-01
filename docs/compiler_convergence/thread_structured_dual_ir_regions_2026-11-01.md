> Native C update: counted latch arrival/reset, control/combined rendezvous,
> completion waits and Python Barrier wait/reset/abort now have native mappings.
> See the [operator reference](backend_threading_operation_reference_2026-11-01.md)
> and [native verification receipt](../hazards/20261101T165117Z_work.md).
> The earlier native-boundary statements below describe the preceding slice.

# Structured thread regions in the dual IR

This replaces the earlier straight-line-only region admission described in
[the ingestion slice](thread_dual_ir_ingestion_slice_2026-10-31.md).
It describes graph ingestion and scheduling, not native barrier execution.

## Representation implemented

`ingest_thread_regions` reads the existing thread table, dependency operand
snapshots, source control IR and source activity graph. It does not inspect
AST objects, source positions or cached function bodies. External callable
identity is recorded by the ingestion writer as `SOURCE_EXTRACTION_IDENTITY`;
captured methods retain their existing `SOURCE_CLOSURE_CALLABLE` proof.
Barrier classification reads those cells rather than mutable attributes.
Each worker use is
qualified by the exact constructor identity. The original function definition
and shared storage stay shared. A constructor inside a loop is a factory site;
its runtime instances are not collapsed into one native worker.

Both graph halves retain source process nodes and the activity ports. Every
activity port copies its exact `SOURCE_ACTIVITY_NODE` identity, phase, control
owner and arm. Every source activity edge is projected through the resident
operation at its completion port, with the source edge as its concordance
cause. Entry/completion bindings retain their `SOURCE_CONTROL_IR` and
`SOURCE_ACTIVITY_BINDING` through `SOURCE_THREAD_REGION_CONTROL`. The source
ControlProgram's existing block and placement identities remain authoritative;
there is no second parser or new execution engine.

Loop repeat edges remain actual graph edges. Branch arms keep their distinct
owner/arm tuples; activity is not flattened into a chosen branch or iteration.
Resource cleanup and exception-control conversions retain their ingested
control/effect nodes in the same way. Exact capture and argument-table receipts
connect actual/formal nodes, without name matching or new shared allocations.
Repeated joins each retain their own completion constraints. Local induction
Inputs are distinguished from activation arguments by the exact formal and
capture tables, rather than by their common Input node type.

`worker_nodes(token)` selects source process operations;
`worker_activity_nodes(token)` selects the associated control ports.
`worker_boundary_edges` includes both when determining the worker's extent.
Hardware worker assignment remains a later lowering choice.

## Scheduling contract

The existing source activity machinery schedules one static traversal of a
structured region: a declared `repeat` edge means the next iteration. It is
not a strict same-iteration inequality. Thread-region scheduling uses that
same cut and the existing registered ASAP/ALAP dependency-level provider.

`scheduling_graph(half)` retains all nodes and all non-repeat constraints.
Its NetworkX edges **and** dependency parents/children express the same
iteration cut; ILPScheduler consumes those adjacency lists. The authoritative
IR keeps its repeat edges. Neither scheduling preference changes the source
edges, chooses branch outcomes or manufactures a cross-worker order.
Concordance records the schedule and all source constraints that caused it.
The audit checks ownership, completion bindings, paired synchronization,
source provenance of every repeat cut and the resulting level inequalities.

An `unadmitted` source control edge or missing capture/participant proof stays
an explicit gap. Scheduling refuses it; it does not discard it to obtain a DAG.

## Host Python barrier policy adopted

Observed ingestion facts identify `threading.Barrier.wait`, its exact barrier
object, captured receiver identity and constructor participant operand. Those
facts alone do not supply a shader/device memory scope.

The following is an explicit compiler policy for the host threading source
contract, recorded as `SOURCE_THREAD_BARRIER_POLICY` with policy-declaration
provenance. It is not a measured property of every target backend:

| Operand/fact | Policy |
| --- | --- |
| Execution scope | `('barrier_participants', exact_barrier_cell)` |
| Memory-effect scope | `('barrier_participant_effects', exact_barrier_cell)` |
| Ordering semantics | Acquire/release (`MemorySemantics.ACQUIRE_RELEASE`, 8) |
| Control projection | Rendezvous for this object's participant cohort/phase |
| Memory projection | Order participant effects across that rendezvous |
| Concordance identity | One `SOURCE_SYNC_OPERATOR` per authored wait occurrence and activation, shared by both projections |
| Backend obligation | Map the participant/effect selectors to supported target scopes; refuse unsupported mappings instead of widening to global/device scope |

An unrelated worker is not enrolled because it shares the parent function or
storage. An unrelated same-spelled `wait` stays unrelated. Joins remain
single-worker completion dependencies.

For statically identified complete cohorts, waits in each path are ordered by
actual graph reachability, forming separate rendezvous records per phase.
Loop recurrence preserves repeated dynamic epochs. Participant counts, uneven
static phase sets or mutually exclusive wait alternatives that cannot prove a
complete cohort remain explicit gaps; this does not prove arbitrary barrier
programs deadlock-free. Runtime-selected cohorts, timeout, abort and broken
barrier behavior require their own runtime/lowering contracts.

## Additional writer repair

The conditional-lifecycle test exposed a late field-effect ordering pass that
moved `if flag > 0` before the numerical region computing `flag > 0`. Its
private source-coordinate order ignored the predicate dependency already
present in control IR. This pass now reads configured source activity ports:
conditional selection follows predicate evaluation, and a numerical region is
placed at its members' completion. The rewrite derives from the exact activity
schedule receipt. No SSA dominance patch or backend special case was added.

## Verification and backend boundary

Tests start from authored Python through the public compiler entry. They
exercise nested while/if/try/except/finally in two worker activations, a loop
thread factory, conditional lifecycle ownership, repeated joins, separate
barrier phases, participant isolation, and operation with diagnostic AST
removed from the region reader. Native checks compile standalone C executables
and compare declared output buffers with the authored Python result.

The C tests establish execution of the loop/exception/thread combinations.
They do not establish native `threading.Barrier` lowering. General barrier
arrival/reset, dynamic cohorts and non-C backend mappings remain the separate
operations listed in the [backend reference](backend_threading_operation_reference_2026-11-01.md).

Detailed measured graph identities, failures traced to their writers, and test
receipts are in the [hazard record](../hazards/20261101T161149Z_work.md).

Measured verification: the combined source/C regression gate passed132tests;
after the final ingestion/admission/audit refinements, the focused structured
suite passed16tests and the dual-IR/For/factory check passed14tests. The separate
synchronization/lock-box/shared-location gate passed31tests. Mapping audits before
and after report0findings and0unsourced facts. These are overlapping gates,
not an additive unique-test count.

# Independent thread paths and synchronization operators

This report extends
`thread_simultaneity_region_demonstration_2026-10-31.md`. Its measured graph
identities and the missing closure-capture connection remain the starting
evidence. The contract below is the user's clarified design direction. It is
not an implemented compiler feature or a measured native result.

## The promise made by a thread

An authored thread gives up knowledge of its **comparative schedule** with
another thread. The compiler retains each path and its internal dependencies,
but the source does not acquire an ordering between otherwise independent
paths merely because one `start` occurs earlier in the parent. Hardware lane
assignment, pooling, or a serial implementation are lowering choices. No
specific comparative schedule is promised by the SSA representation.

The compiler must still preserve the operations and their exact identities.
It may not erase a worker's store, replace a post-join read with the original
literal, clone shared storage, or pretend a worker has no effects. The author
is responsible for the effects of unsynchronized accesses; the compiler is
responsible for faithfully representing that those accesses may occur.

The authored current test requires two activations of the same worker
definition to reach the same captured storage identity; the current graph
does not yet prove that identity. Within each activation, its indexed read
precedes its add, which precedes its indexed store. Both starts precede their
respective worker paths, and each join follows completion of its matching
path. There is no required order between the two workers' reads or stores.
That is a **partial order**, not a hidden choice of one interleaving.

```text
                 worker A: read -> add -> store
start A --------<                              >-------- join A

                 worker B: read -> add -> store
start B --------<                              >-------- join B

        A and B both reach one captured storage identity.
        No A-to-B or B-to-A dependency is added by that fact alone.
```

The ordered parent path still contains the two starts and the two joins.
Those parent dependencies do not turn the worker bodies into parent calls.

## What the ingestion graph scheduler does

The run-to-join simultaneity region contains both activation paths, their
closure boundary, the parent path, and shared storage relationships. The
ingestion scheduler may inspect and rearrange its dependency graph into an
SSA-ready form. It must distinguish **graph presentation/order chosen for
lowering** from **a dependency guaranteed by the authored program**. A
topological listing of nodes is not by itself an added cross-thread edge.

The earlier report's “double operator” detection is an analysis point, not
an automatic serialization point. Once two operators have exact identities,
the compiler can record that they belong to independent paths and possibly
reach the same storage. Their overlap alone does not authorize a lock,
barrier, total order, or result claim. A policy table should decide what
graph transformation an *explicit synchronization operator* requests. Where
the authored program requests no synchronization, the policy leaves the
paths independent. Every added constraint, and its source, belongs in the
concordance.

## Shader terminology for the operator set

The relevant established terms are **barrier** and **dependency**. In
SPIR-V, `OpControlBarrier` waits for invocations in an execution scope;
`OpMemoryBarrier` constrains observation order of memory accesses within a
memory scope. They are separate concepts, and a control barrier may also
carry memory semantics. Scope and memory semantics are operands; memory
semantics includes a bit mask of flags. They are not names for the
synchronizing operation. “Gate” is
not the term used for these SPIR-V operators. See the
[SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html).

For Turing's ingestion dependency graph, the proposed vocabulary is:

| Concept | Graph meaning | Origin required |
| --- | --- | --- |
| `Spawn` | Establish a separately schedulable activation path and its closure environment. | Exact authored thread target and start handle. |
| `Join` | Add completion dependency from one exact activation to the waiting parent continuation. | Exact authored join receiver. |
| `ControlBarrier` | Constrain participating paths to meet before passing the barrier. | An explicit authored synchronization operation and scope. |
| `MemoryBarrier` | Constrain visibility/order for selected storage effects. | An explicit authored memory-ordering operation and memory scope. |

### Placement in Turing's dual IR

Turing already separates `ControlProgram` blocks, whose placement is posted
as `CONTROL_BLOCK` and `CONTROL_BLOCK_PLACEMENT`, from ProcessGraph value and
dependency edges. This is a close representational match for the distinction
between control and memory synchronization. The two barriers need different
graph effects, even when one authored operation requests both:

* A **control barrier node** belongs to the control graph. It identifies the
  participating activation paths and a meeting point. Its incoming and
  outgoing control edges establish phase boundaries for a simultaneity region.
* A **memory barrier node** belongs to the dependency graph. It identifies
  the storage effects and memory scope to which an ordering rule applies. Its
  edges constrain those effects. Its mere presence does not make all paths
  meet at one control point.

If one synchronization operation requests both, the two projections must
share one concordance identity, derived either from an exact authored
operation or from a compiler policy decision with its own recorded causes.
A scheduler reads their facts and
places them in the same run-to-join region before solving its sections. It
does not independently infer a memory barrier from a control barrier, or a
control rendezvous from a memory barrier. The region is bounded by the exact
start/join relationships; an interior control barrier divides it into
sections, and an interior memory barrier contributes dependency constraints
to those sections.

This is the proposed graph shape, not a request to insert a barrier merely
because two ordinary operators touch the same storage:

```text
       control graph                     dependency graph
  activation paths -> ControlBarrier    memory effects -> MemoryBarrier
                         |                                  |
                         +----- one operator identity -----+
                                  concordance receipt
```

The current compiler has a `ControlBlockKind.DISPATCH` and durable
`ControlDeploymentRegion` lane records, but no `ControlBarrier` or
`MemoryBarrier` source node type established for this thread program. Adding
the nodes requires an exact source or policy identity and an ingestion-stage
concordance fact. A late SSA annotation cannot give the
region builder or the structural fold those edges.

These are proposed *graph meanings*, not claims that Turing already has
these four operator classes. `thread_start` and `thread_join` currently
survive as dispatch operations; the report is naming the dependency
relationships they would need to carry. A backend may translate a graph
operator to a different native primitive according to its target and
capabilities. Vulkan likewise distinguishes execution and memory
dependencies in its synchronization model.
[Vulkan synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)

For a graph scheduler to apply one of these operators, the policy table needs
an exact operation identity, participant activation identities, its execution
scope, its memory scope and storage identities where applicable, the ordering
semantics, and the source/concordance cells that authorize it. The graph
rewrite then adds only the dependency edges that those operands justify and
records each edge's cause. A missing scope or unsupported combination must
remain unresolved or be refused at its source; it cannot quietly become a
global barrier. The corresponding shader terms *scope* and *memory semantics*
are the useful “flags” the question was reaching for.

An atomic read-modify-write, if later requested, would be a typed memory
operation with its own ordering contract. It should not be inferred from an
ordinary `shared[0] = shared[0] + 1` expression. This report does not add an
atomic or specify a conflict-resolution policy for ordinary writes.

## Consequence for the first compiler change

Before graph scheduling or structural folding, the dual IR and concordance
need the exact relation from the worker's closure input to the parent's
binding/storage identity. That relation says **which location both paths
reach**. It does **not** say which path reaches it first. The current source
graph has the worker closure input and parent list allocation separately,
and the full SSA still omits the worker body. The first change therefore
connects the capture and carries the worker path into the region, without
inventing a comparative schedule.

The acceptance condition is structural: SSA contains one worker definition,
two distinct activations, one shared captured storage identity, both worker
effects, exact joins, and no accidental cross-worker ordering edge. Native
placement and the observed outcome of unsynchronized writes are outside this
ingestion demonstration.

## Source-driven dual-IR tests added before implementation

`tests/test_thread_dual_ir_sync_contract.py` compiles ordinary authored
Python through `lower_ast_source_to_ssa` and inspects the actual ProcessGraph
and concordance. Three assertions currently pass: both thread handles have
their exact create/start/join edges and one worker target; the authored
`threading.Barrier(2)` constructor and its `wait()` are distinct nodes; and
an unrelated class's same-spelled `wait()` is not assigned the barrier
callable identity.

Four strict expected failures are the next construction boundaries:

1. The worker's captured binding has no concorded relation to the exact
   parent binding.
2. The two starts and two joins have no placed `CONTROL_BLOCK` facts in the
   returned identity book.
3. The worker's `gate.wait()` loses the exact
   `threading.Barrier.wait` callable identity through its closure input.
4. The real `wait()` has no paired control-barrier and memory-barrier
   projections in the dual IR.

The authored barrier constructor already survives as a source graph node
with `extraction_identity='threading.Barrier'`; it is not itself the wait
operation to schedule. The parent wait has
`extraction_identity='threading.Barrier.wait'`; the worker wait currently has
only `source_type='Call'`. These facts make source-node identification the
first step of a future replacement, without selecting nodes by a `wait`
spelling. The focused new suite measured 3 passed and 4 xfailed. No barrier
node replacement or simultaneity-region solver was implemented by these
tests.

> Historical slice: the AST-derived region builder described below was
> withdrawn. The current graph-only structured implementation and explicit
> host barrier policy are documented in
> [structured dual-IR regions](thread_structured_dual_ir_regions_2026-11-01.md).

# Source thread dual IR: implemented ingestion boundary

Current correction (2026-10-31): the AST-driven ThreadRegion projection
described below has been withdrawn. The current source thread table retains
worker bodies, captures and activation boundaries, but it does not supply the
missing run-to-join control/dependency region. `SOURCE_THREAD_REGION` records
`concorded_control_topology_required`. Paired barrier projections and ASAP/ALAP
region acceptance are unfinished; the corresponding assertions remain failing.

## Observed implementation

`lower_ast_source_to_ssa` now retains source-derived `ThreadRegion` products
immediately after lexical topology reduction and before structural
specialization. Each has a control ProcessGraph and a dependency ProcessGraph,
both using the existing memory-backed ProcessGraph representation. Their
membership, constraints and scheduling results are concorded. These are
compiler ingestion products, not test graphs or backend deployment lanes.

The region's closure is its graph extent. Python lexical captures are one
kind of edge crossing that extent. The graphs retain parent setup and
continuation as boundary context, rather than cutting away their incident
edges. Worker source operations are qualified by their exact thread-create
activation; their source function definition and captured allocation stay
shared. A start activates its own path and its join depends on that path's
completion. Parent control order comes from exact AST ownership/evaluation
order, not source-line ranges. The first slice admits straight-line
lifecycles and worker bodies.

Both graph halves expose each body node's `worker_token`: the existing
concorded thread-create cell, including its source scope. Region
`worker_tokens` and `worker_nodes(token)` select a worker directly; its
start/join/create nodes carry `dispatch_worker_token` on the parent path.
`worker_boundary_edges(token, half)` retains crossing captures, lifecycle and
barrier constraints when separating the worker for later native binding.
Parent context, shared storage and rendezvous completion have no exclusive
worker owner. These are views of existing concordance identities, not new
worker numbers or hardware assignments.

Lexical reduction records `SOURCE_CLOSURE_ENVIRONMENT` while the enclosing
environment, its exact binding cell, the nested definition occurrence and
function-address cell are available. Canonical relabeling derives the
canonical environment and `SOURCE_CLOSURE_CAPTURE` from both exact binding
cells. The admitted cases have one binding of that enclosing name. Other
cell/value flows remain unresolved; region scheduling refuses a missing
capture rather than guessing by a spelling or permanently aliasing an
initializer. This limit concerns Python lexical binding, not the definition
of the simultaneous graph extent.

For a captured barrier, callable identity is derived from that capture and
the exact parent's `threading.Barrier` constructor result. A
`SOURCE_CLOSURE_CALLABLE` receipt authorizes `threading.Barrier.wait` on the
worker call. An ordinary `Other.wait()` remains unrelated. A captured alias
of the same barrier allocation also resolves through the exact value.

The final dispatcher control installer already constructed `DispatchBlock`s
for starts and joins. It did not post its replacement tree. It now calls
`post_control_rewrite` at that writer, preserving placed `CONTROL_BLOCK`
facts rather than relying on later SSA annotations.

## Existing scheduling route and its limit

`TURING_PAGE['schedule_preference']` is validated in `site_bundle.py`, carried
to shell planning, and retained in `ControlDeploymentRegion`. In
`strategize_shell_deployment`, deployment lane membership maps preferences to
dispatch regions. `_dispatch_subgraph` passes the preference to
`ProcessGraph.compute_levels(method=..., order='dependency')`; deferred
region scheduling uses `dependency_snapshot` / `schedule_dependency_snapshot`
and the registered `DEPENDENCY_LEVELS` provider. Its Python oracle is the
existing `ILPScheduler`. The provider uses dependency parents/children and
ASAP/ALAP equations; it does not need hardware lane selection.

The deployment permission is narrower: `ControlDeploymentLane` means
**proven independent**, and a deployment region retains a serial lexical
fallback. Shared-storage thread activations do not establish that permission.
This slice does not relabel them as independent deployment lanes or change
that contract. `ThreadRegion.schedule(preference)` configures the same
registered dependency-level provider directly on the retained graph halves.
Control constraints also appear, with their control provenance, in the
dependency scheduling graph. Schedule receipts record levels; they add no
authored comparative edge. No alternate DFS or scheduling algorithm was added.

The source-driven two-worker test measures both preferences on the same
region. Every required edge increases its assigned level. Neither preference
changes the topology or creates reachability between worker paths. The
common captured allocation alone supplies no synchronization edge.

## Barrier node facts implemented

Each recognized wait occurrence and activation has one
`SOURCE_SYNC_OPERATOR` cell. Both its `ControlBarrier` projection in the
control graph and its `MemoryBarrier` projection in the dependency graph
derive from that cell. The original constructor remains a different source
node. Facts include the original wait cell, activation cell, exact barrier
object, execution scope, memory scope, semantics and resolution status.

The execution-scope operand is `('barrier_participants', exact_object_cell)`.
For one straight-line phase, the rendezvous is admitted only when the
distinct participant paths equal the authored literal party count. A
`SOURCE_CONTROL_RENDEZVOUS` fact derives from every wait identity and the
constructor's exact party-count operand. Arrivals precede one completion
node; continuation operations depend on that completion. Missing
participants or phase ownership are recorded as region gaps. A join remains
its own activation's completion dependency.

**Unresolved:** the Python constructor supplies no shader memory-scope operand
or memory-semantics mask. The memory projection therefore carries explicit
`None` operands and `unresolved_memory_policy`. No memory fence edges are
invented. Region scheduling refuses to call this a solved memory barrier.
Control-origin rendezvous edges in the dependency graph remain control
constraints; their presence is not a memory-policy decision.

## Proposed remaining policy table

These rows specify the next design boundary, not implemented lowering.

| Exact operation | Required operands/provenance | Control projection | Memory projection |
| --- | --- | --- | --- |
| Thread start | Exact handle, target and activation environment | Activation edge to that path | Preserve captured storage/effects; no automatic fence |
| Thread join | Exact handle and matching activation completion | Completion to waiting continuation | Any target memory guarantee must come from an explicit language policy; no all-participant rendezvous |
| Python `threading.Barrier.wait` | Exact instance, participant/phase set and authored party count | Participant-local rendezvous | Resolve a named Python memory policy before assigning scope/order flags |
| Explicit control-only barrier | Exact source or declared policy, participants, execution scope | Rendezvous | None unless separately requested |
| Explicit memory fence/barrier | Exact source or declared policy, memory scope, storage/effect set, ordering flags | None | Selected before-effects → fence → after-effects |
| Combined control/memory operation | One operator identity plus both sets of operands | Rendezvous projection | Ordering projection from the same identity |

Each policy row must declare admissible scopes, storage classes, ordering
flags, phase rules, legal graph rewrites and source-located refusal reasons.
Unknown operands stay unresolved. A local scope must never become global.
Ordinary overlapping loads/stores remain ordinary effects; this table gives
no permission to insert a lock, atomic operation or comparative worker order.

## Exact next boundary

Make the existing structural-effect and SSA planners consume this retained
region, its captures and activation-qualified worker effects. The source
region already retains the post-join indexed read and both worker stores;
the ordinary SSA path still omits the worker body and folds the return to
the initializer, so its strict expected failure remains. Do not bypass that
consumer with a mutation exemption. General closure cell/value flow,
control-dependent activation sets, reusable barrier phases and the explicit
memory-policy rows follow at their respective ownership boundaries.

No native thread execution, barrier lowering or backend emission is claimed.

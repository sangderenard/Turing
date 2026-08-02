# Computational Narnia: headless world-engine contract

## Status

This document records the architectural decision for the optional
"computational Narnia" development world. It is a headless Minecraft-like
engine written in Python so its transition logic can follow Turing's existing
source-to-compiled-kernel path. It is not a renderer, a camera, a menu, or a
second compiler IR.

The following decisions are normative:

1. `src.common.dt_system` is the only authority over advancement. The world
   has no game clock, frame clock, observer clock, replay clock, timer loop, or
   independently selected timestep.
2. The shell decides whether the world is admitted to a managed-time round.
   A "tick lease" means admission to an existing `dt_system` window; it does
   not grant the shell or world authority to invent the window's time
   semantics.
3. With no lease, the world is inert. It does not step, poll, scan source,
   consume provenance, or harvest live material.
4. The authoritative game state is a small collection of sparse
   `AbstractTensor` tables and graph tensors. Python object graphs and
   NetworkX graphs are not authoritative world state.
5. The engine is headless. Graphics may later project the state, but no
   graphics dependency belongs in the world transition contract.
6. State-machine recognition happens during AST ingestion and lowers through
   the existing `StateMachineTick`, ProcessGraph, fused-program, and SSA
   machinery. This design adds no new SSA operators.

## The sole time contract

The shell may admit the world through either of the modes already defined by
`dt_system`; it may not create another mode.

`DtCompatibleEngine.world_time` and `DtCompatibleEngine.observer_time` are
existing dt-system accounting fields advanced by `step_with_state`. They do
not authorize an engine-local clock or another time manager. The world may
report them exactly as maintained by that boundary but may not advance,
rescale, or schedule work from them independently.

### Scientific managed time

An external coordinator submits a validated `TimeWindowRequest` to
`ManagedTimeRuntime`. The world advances only as a registered subordinate of
that request. Consequently:

- `t_start` must equal the current committed time;
- `t_end` must be greater than `t_start`;
- request IDs must increase and generations must match;
- authored event times are finite, unique, ordered, and strictly inside the
  requested window;
- the accepted microsteps land on every authored event boundary and exactly
  cover the requested window;
- attempted `dt` does not increase within the window unless the existing plan
  explicitly permits it;
- terminal failure or commit-gate rejection restores the entire managed
  state and controller to the start of the window.

The world state therefore implements `copy_shallow()` and
`restore(snapshot)`. Those operations must cover every mutable tensor, sparse
index, allocation cursor, player value, voxel value, BoundSpring value,
pending input, consumed-provenance cursor, and engine-owned counter. A retry
must be observationally equivalent to an attempt that never occurred.

### Realtime managed time

If the shell selects the existing realtime path, compute budgets come from
`RealtimeConfig`, `RealtimeState`, and `compile_allocations`. Realtime clipping
and liveness behavior retain the meanings already specified by `dt_system`,
including `time_slip` and `advanced_dt`. A realtime result must not be
presented as scientific completion.

"Slow Narnia" is therefore not a clock. The shell can lease fewer managed
windows, request smaller valid windows, or admit fewer queued provenance
events at authored event boundaries. The world experiences only the admitted
managed time. Buffered events retain sequence and provenance identity, not a
private timebase.

### Hierarchy and concurrency

Time delegation is strictly top-down. A parent regulator proposes a window;
subordinates may satisfy it with their own controller and managed microsteps.
There is no global timestep negotiation.

The world may be a nested `RoundNode` containing voxel, player, provenance,
and spring children. A worker thread may compute an admitted step through
`ThreadedSystemEngine`, but the parent dt graph still owns the slice and waits
for the result. Thread ownership never implies time ownership.

## State-machine marker and class contract

The canonical source marker will be the base-class name
`AbstractTensorStateMachine`:

```python
class ComputationalWorld(AbstractTensorStateMachine):
    ...
```

The fully qualified base identity, not a coincidental method name, tells AST
ingestion that this class is a state machine. A decorator may later be a
convenience spelling, but it must resolve to the same marker contract before
lowering.

`AbstractTensorStateMachine` is a class-bound state contract, not a new time
runtime. It borrows its owning class's storage and requires:

- a declared tensor-state schema;
- initialization of that state into an explicit `StateTable`;
- one transition entry point accepting externally admitted `dt`, state, and
  `StateTable`;
- `Metrics` for acceptance, stability limits, named error channels, and
  actual advancement;
- complete snapshot/restore coverage;
- explicit input and output fields;
- no mutation outside the managed transaction.

At the engine boundary the compiled or Python implementation conforms to
`DtCompatibleEngine`:

```text
step(dt, state, state_table) -> (ok, Metrics, state)
get_state(state=None) -> state
snapshot()/restore(snapshot) for engine-owned mutable state
```

The existing `StateMachineTick` remains the compiled control primitive: one
admitted transition, not a host polling loop. AST ingestion identifies the
marked class, records its state fields and transition cases, and feeds the
ordinary pipeline:

```text
Python class AST
  -> existing ProcessGraph operators plus ControlProgram/StateMachineTick
  -> existing fused numeric regions
  -> existing SSA
  -> selected compiled backend
```

Known expressions must be reduced to existing operators. Unsupported source
remains available to the existing opaque-node diagnostics; ingestion does not
invent operators or preserve representable syntax as opaque payloads.

## BoundSpring is the world state machine

The world machine owns the BoundSpring network state. BoundSpring is not a
detached animation and not a renderer-side approximation. Its sparse topology,
positions, velocities, masses, rest lengths, group masks, activation state,
growth state, containment state, and force parameters are fields of the
`ComputationalWorld` state contract.

The legacy `transmogrifier/bound_spring.py` remains the behavioral source for
the fast tensor spring system, including node activation, graph growth,
containment, and smooth physics. Its current `step()` cannot be used unchanged
because it contains an independent per-vertex sub-dt planner and advances
discrete counters per Python call. The managed port must:

- express state with `AbstractTensor`, not require Torch as the semantic
  definition;
- accept only the `dt` admitted by its parent `dt_system` round;
- expose stability or causality ceilings with `Metrics.dt_limit` and named
  error channels;
- let the existing controller retry or subdivide rejected scientific steps;
- report `Metrics.advanced_dt`;
- roll back positions, velocities, rest lengths, growth, queued forces,
  active group, glow/activation values, and all counters on rejection;
- advance discrete cycles from accepted managed transitions, never host render
  calls;
- use a nested managed rate domain if spring physics needs local microsteps.

This is a port of time ownership, storage, and operator expression. It is not
authorization to redesign the spring forces or replace FluxSpring/
BoundSpring behavior.

## Sparse authoritative world state

The first state schema is structure-of-arrays and COO-oriented. Concrete
field widths remain backend choices, but semantic fields are stable.

```text
entity_id[N]                 stable world entity identity
entity_kind[N]               voxel object, player, terminal, code embodiment
entity_flags[N]
position[N, 3]
velocity[N, 3]

edge_index[2, E]             source/destination COO convention
edge_kind[E]
edge_state[E, Fe]

occupied_block_coord[K, 3]   only occupied or changed voxel cells
occupied_block_kind[K]
occupied_block_state[K, Fb]

component_entity[C]          sparse component membership
component_kind[C]
component_state[C, Fc]

artifact_entity[A]
artifact_reference[A]        stable external artifact/provenance IDs
provenance_cursor[1]         last admitted event sequence
```

The contract follows `AbstractGraphCore`'s stable conventions:
`edge_index` is `(2, E)`, edge values align with `E`, node features align with
`N`, and indices are zero-based. The current `AbstractGraphCore`
NetworkX-synchronization code is not part of the world-state contract. It may
be used for inspection, but compiled execution must be able to operate on the
sparse tensors without materializing a Python graph.

`StateTable` is the overarching transaction table. World tables may use
scopes such as:

```text
world / topology   / edge_index, edge_kind, edge_state
world / entities   / id, kind, flags, position, velocity
world / voxels     / coordinate, kind, state
world / player     / entity, intent, inventory, contact
world / provenance / cursor, pending_sequence, embodiment
engine / spring    / mass, rest_length, masks, activation
dt_tape / ...      / attempted, accepted, metrics, advanced
```

All values mutated by a candidate step participate in the same rollback.
There is no parallel "game state" dictionary that can drift from these
tables.

## Voxel, object, and player semantics

The engine supplies mechanics, not graphics:

- sparse occupied-cell storage and chunk/index lookup;
- object identity and sparse component records;
- player position, velocity, contact, inventory, and authored intent;
- collision and interaction transitions;
- terminals binding world entities to stable code/provenance references;
- portals as ordinary world relationships, without imposing software-stack
  or repository hierarchy on geography;
- deterministic queries and state snapshots for a future renderer, text
  client, test, or compiled host.

Rooms and terrain have mnemonic and historical meaning only. The engine does
not force directories, imports, compiler stages, or ownership layers into a
geographic hierarchy.

## Shell lease and live-material harvesting

The shell owns mode selection and admission; it does not own simulation time.
The boundary is:

```text
shell mode inactive
  -> no world RoundNode admission
  -> no source/provenance harvesting
  -> no world mutation

shell mode active
  -> shell captures one immutable status/input batch
  -> batch events receive authored boundaries in a TimeWindowRequest
  -> dt_system admits and transactionally advances the world
  -> accepted transition updates sparse world/provenance state
```

A shell status batch may contain player intent, connection status, available
artifact identities, and newly captured `EvolutionMetaGraph` event sequences.
It must not contain arbitrary live Python objects. Material is harvested by
stable reference and only when its event is admitted. Repository scanning,
source reads, and provenance subscription while inactive are prohibited.

`EvolutionEvent.captured_ns` is capture telemetry and ordering evidence; it is
not simulation time. Event sequence establishes ingestion order. Authored
managed event boundaries establish when the world may consume those events.

The compiler-side `EvolutionMetaGraph` stays the exact, append-only provenance
record beside compiler IR. The world stores only the admitted cursor,
embodiment references, and sparse consequences. It does not replace or mutate
the compiler's IR or provenance ledger.

## Compilation boundary

The Python engine is the readable definition. Automatic compilation must use
the established source path and operator tables:

- marked class and method AST are inspected without executing source merely
  to discover structure;
- transition expressions lower to existing ProcessGraph operators;
- state-machine control lowers to existing `StateMachineTick` and then SSA;
- AbstractTensor backend selection chooses Python, C, GLSL, WebGL, Fortran,
  or another supported target;
- the Python and compiled forms share one state schema and transaction ABI;
- backend-specific storage restrictions receive explicit translation/storage
  tables rather than changing state-machine semantics.

The world engine must remain useful in Python before every method is
compilable. Unsupported methods are reported at the normal lowering boundary;
the runtime must not silently fall back from a requested compiled scientific
transaction to an unaccounted Python loop.

## Initial implementation sequence

1. Add the `AbstractTensorStateMachine` marker contract and AST recognition,
   lowering its transition dispatch to the existing `StateMachineTick`.
2. Define a rollback-complete `ComputationalWorldState` containing sparse
   entity, edge, voxel, player, provenance-cursor, and spring tensors.
3. Implement a headless `ComputationalWorld` as a `DtCompatibleEngine` and
   register it in an explicit `StateTable`.
4. Port BoundSpring storage and equations to AbstractTensor operations while
   moving all substep authority into a nested dt-system rate domain.
5. Add a shell lease adapter that includes the world in managed rounds only
   while game mode is active and creates immutable status batches only then.
6. Admit buffered `EvolutionMetaGraph` sequences at authored event boundaries
   and embody them as sparse world records.
7. Qualify Python and one compiled backend against identical accepted-step,
   rollback, exact-landing, and state-parity tests.

No renderer is required by any of these steps.

## Acceptance invariants

The engine is not conformant until tests demonstrate:

- no mutation or harvesting without a shell lease;
- no time advancement outside `dt_system`;
- exact scientific window and event-boundary landing;
- identical state after rejected-window rollback and deterministic retry;
- stale generation, replayed request, reordered request, and discontinuous
  time rejection;
- complete `StateTable`, controller, world, spring, player, voxel, and input
  rollback;
- accepted/rejected timestep and named-error telemetry;
- no use of wall-clock capture timestamps as simulation time;
- sparse state remains authoritative with no required NetworkX materialization;
- AST recognition is based on the canonical state-machine marker;
- lowering uses existing ProcessGraph/SSA operators and reports unsupported
  source normally;
- Python and compiled kernels produce contract-equivalent accepted state;
- the engine imports and runs without a graphics context.

## Existing source anchors

- `src/common/dt_system/dt.py`: superstep plans/results and exact windows.
- `src/common/dt_system/dt_api.py`: strict top-down `DtStepper` hierarchy.
- `src/common/dt_system/time_runtime.py`: generation, request ordering,
  transaction, exact landing, and commit gates.
- `src/common/dt_system/dt_graph.py`: nested managed rate domains,
  transaction coverage, metrics, and `StateTable` publication.
- `src/common/dt_system/engine_api.py`: engine registration, causal ceilings,
  metrics, and explicit state stepping.
- `src/common/dt_system/threaded_system.py`: concurrency subordinate to parent
  dt slices.
- `src/common/tensors/abstract_graph_core.py`: sparse graph tensor conventions.
- `src/compiler/ast_process_graph.py`: semantic AST lowering through existing
  operators and existing diagnostics.
- `src/compiler/control_source.py`: compiled `StateMachineTick` semantics.
- `src/compiler/evolution_metagraph.py`: optional append-only provenance.
- `../transmogrifier/bound_spring.py`: legacy BoundSpring behavior to port,
  including the internal timing that must not survive as an independent time
  authority.

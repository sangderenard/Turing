# Managed time runtime

## Architectural position

Turing owns universal computational accounting:

- abstract tensor values;
- state identity and checkpoints;
- adaptive timestep control;
- accepted and rejected attempt records;
- exact event-boundary landing;
- scientific error accounting.

Nodus may author and schedule the process graph, queues, resources, and
backpressure around these operations. It must not bypass Turing's managed-time
round or reinterpret a requested physical interval as one scheduler tick.

Domain engines retain their physical authority. A camera authors an
electromechanical slice budget. An optical engine defines field residual,
power, phase, arrival, and conservation metrics. Turing decides whether a
candidate advance satisfies the declared targets and accounts for rollback,
rerun, and exact landing.

## Public process boundary

`src.common` exports:

- `TimeWindowRequest`
- `TimeAdvanceReport`
- `ManagedTimeRuntime`

A request supplies an absolute start/end time, generation, initial microstep,
and ordered event times. The runtime:

1. rejects stale, replayed, reordered, or discontinuous requests;
2. converts absolute event times to relative superstep boundaries;
3. lets the Turing controller choose and rerun microsteps;
4. lands exactly on every admitted event and the final window;
5. restores the whole window on terminal failure;
6. optionally asks a domain-neutral commit gate whether surrounding process
   queues, readers, and output reductions reached their authored boundary;
7. commits the new runtime time only after exact completion and gate approval.

`ManagedTimeRuntime.advance(..., commit_gate=...)` invokes the gate after the
candidate window lands exactly but before committed time or request identity
advances. A false result or gate exception restores the managed state and
controller to the start of the entire window. Nodus may use this boundary for
quiescence and reader-release accounting; domain adapters may use it for
reception or detector completion. Turing does not interpret those conditions.
The gate should be an observation-only predicate. Turing cannot roll back
external side effects performed by the gate itself.

The extracted Nodus runtime adapter now supplies a quiescent FIFO transaction checkpoint
covering storage, publication tags, writer binding, and every reader cursor.
Spectral Analyzer's `ManagedProcessState` composes that participant with solver
state, allowing commit-gate rejection to restore both before rerun. Turing also
restores controller state in a `finally` path if a native participant reports
rollback failure, then surfaces the rollback failure explicitly.

The managed state must implement `copy_shallow()` and `restore(snapshot)`.
Those methods must cover every value the callback can mutate, including native
handles or persistent solver generations represented outside a Python object.

Spectral Analyzer also has an idle-only native T4 wave participant. It
checkpoints arena field blocks, active spectral lanes, progress, link
generations, and boundary telemetry, and rejects topology/configuration drift
on restore. This widens real native coverage but is intentionally not a claim
about the complete ray pipeline: T1-T5 queues, sensor/BDPT accumulators,
ray-tracer UV/illumination state, and GPU-resident buffers remain separate
transaction participants.

An additional idle CPU-only queue participant now checkpoints every T1-T5
`PipelineQueue`, including output, wave-exit, BDPT side-data, and T5-ready
channels. Qualification proves a rejected T4 traversal's queued output is
removed before deterministic retry. Pending/stash vectors, accumulators,
counters, and GPU-resident state are still not covered by that queue layer.

## Scientific metrics

`Metrics.error_channels` and `Targets.error_limits` provide named domain
errors without disguising them as fluid mass or divergence. Conservative graph
aggregation retains the maximum value for each named error.

Examples:

- `field_residual`
- `relative_power_error`
- `phase_error`
- `boundary_leakage`
- `unresolved_arrival_power`
- `symplectic_residual`

`Metrics.hard_failure` rejects a candidate unconditionally.
`Metrics.dt_limit` supplies a causal or stability ceiling for the retry.
`Metrics.advanced_dt` records actual engine advancement.

`SuperstepResult` reports:

- attempted timesteps;
- accepted timesteps;
- rejected-attempt count;
- exact event boundaries reached;
- next controller proposal.

## Scientific and realtime modes

Scientific execution is transactional and exact. A causal-ceiling violation
rejects the candidate without advancing state; the controller retries at an
admissible timestep.

Realtime execution is explicitly approximate. It may clip an advance to a
causal ceiling for liveness, but records `time_slip` and `advanced_dt`.
Realtime results cannot be presented as scientific completion.

## Graph execution

Scientific `MetaLoopRunner` execution preserves nested `RoundNode` hierarchy
and calls adaptive `run_superstep` at every managed rate domain. The flattened
ILP schedule remains useful for explicit realtime execution and future process
deployment, but it is not a replacement for managed-time semantics.

When Nodus integration is added, each accepted Turing microstep may activate a
scheduled process frontier. Rejected candidates must roll back every process
state in that transaction before a smaller timestep is attempted.
Terminal failure restores the complete requested graph window, including
microsteps that had already been accepted in that window.

## Bisection solver

The bisection solver now requires:

- an explicit `StateTable`;
- engine or conventional inner-state snapshot/restore;
- a monotonic scalar objective;
- finite iteration and committed-step budgets.

Every candidate evaluation restores both engine and shared state-table state.
Only the selected timestep is committed. Bisection remains opt-in because many
scientific errors are not monotonic in timestep.

## Current limitations

- `StateTable.snapshot()` currently uses correctness-first deep copies.
  Production native/GPU integration needs generation checkpoints or
  copy-on-write state handles rather than bulk hot-path copies.
- `RoundNode.schedule="interleave"` retains historical operator-splitting
  semantics: each child receives a fraction of the parent timestep. It must not
  be used for coupled engines that each need to reach the same physical time
  unless that splitting is explicitly authored.
- One runtime-only Nodus typed FIFO/reception-token adapter and complete edge
  transaction checkpoint are qualified, including byte parity with the legacy
  table ABI; general process-graph and multi-edge deployment remain.
- Native solver transactions must prove that all persistent state is covered
  by the managed state's checkpoint contract.

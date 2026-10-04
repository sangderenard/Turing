# Orbital game integration status — 2026-10-03

## Current independent-planning work

Root commit `6c22bdf` extends the existing `SolutionService` with captured
requests and completed result/error publication. Its four lifecycle tests
pass in 1.27 s. It preserves the firing API, suppresses superseded/stopped
results, and prevents a second worker while the first is still stopping.
It cannot interrupt an already-running solver call.

Game, tracker, and collocation ownership changes remain uncommitted WIP.
The game submits immutable captures; the worker never reads live craft or
station columns. The main thread accepts/announces results, validates the
request generation and simulation epoch, and checks captured assumptions
against the existing ON PLAN threshold. The craft and station timestamps
are captured separately because their updates occur at different points
inside a frame. Initial future plans are checked against the captured
circular coast. Replan freshness transports the captured discrepancy on
the old plan's reference; this is an assumption check, not a replacement
physical integrator. Existing delayed-burn recovery remains authoritative.
Accepted replans are applied only at tracker round boundaries; the old plan
continues while a request is pending. Solver failure is announced once.

Each transcription gets its own `ReverseRows` wrapper and thus its own
native execution arena via the existing allocator. Warm starts copy the
arrays they read and use an independent transcription. A lifecycle-only
selection passes 21 tests in 7.71 s, with mocked native preparation. It covers
nonblocking frames under a held solver, replacement, failure, stale input,
declared fuel availability, shutdown, headless termination, and ownership.
It does not prove actual game flight. A subsequent native arena gate now
passes: two executions share one artifact but have disjoint scalar arenas,
with 200 concurrent distinct-feed/seed Jacobian and forward evaluations
(12.15 s, sampled peak max(working set, private bytes) 2,583,293,952 bytes).
The emitted audit has 451 rows / 20 functions, zero structural findings,
5,939 unsourced facts and zero unsourced ids; its latch remains OPEN.

The planner slice defect now calls the actual library `RK4Integrator.step`
on its seven-state SymPy matrix, using the existing continuous laws at each
stage. No RK coefficients or staging adapter were authored. Source-only
measurement gives 165 unique DAG nodes / 33 input symbols for c0/t1 and
331 / 77 for c1/t6, versus the old kick/drift/kick's 94 / 33 and 200 / 77.
This written connection is not yet numerically verified. Its small c0/t1
row/Jacobian artifact compiled and emitted an audit (6,637 rows / 332
functions, zero structural findings, 72,487 unsourced facts and 10 unsourced
ids, OPEN). The first test stopped at its 240 s wall bound before numerical
assertions, at a 3,853,746,176-byte memory peak; this is not a numerical or
compiler failure. A rerun reuses the exact artifact through the existing
expression-key/compiler-record/library validation and labels its prior
audit explicitly. The old process's full SSA module was not persisted by
the production row cache; no reconstructed provenance is claimed. The
numerical gate includes both zero propellant and finite propellant with
changing library-stage mass and retains its four-ULP bound.

The delayed actual-Hohmann plan/reference gate passes (17.38 s, sampled
peak 2,573,942,784 bytes): unchanged captured assumptions permit a delayed
immediate-start candidate, while a disturbance rejects it. This does not
test a physical game flight. No production collocation row bank has been
compiled for this work; that build waits for the dt-controller correction
and a frozen compiler stamp. The older adapter notes below are historical.

## Earlier adapter integration

The live game now calls the collocation planner through its declared API:
`CollocationProblem`, `plan_transfer`, and `collocation_replanner`. It derives
the phasing sweep from the probe plan. Hohmann's sweep comes directly from
the first and last `HohmannPlan.legs()` orientations, whose declared geometry
spans half an orbit; no compiled reference sampling is needed.

Collocation orders plan from a future circular start after the phasing wait.
While waiting, the game coasts the craft and stations in lockstep and clears
throttle, gimbal, and wheel-torque commands before advancing. The order path
is blocked while an earlier transfer is in progress, so this does not add
mid-transfer retargeting.

The phasing calculation is not yet acceptance evidence for collocation. It
uses a probe plan's sweep and duration, then solves again from the rotated
start state. The existing six-axis `pointing_proxy` prices forces by its
declared axes; rotating the start can change the optimized trajectory, sweep,
and duration. Rendezvous behavior must remain unclaimed until the full native
acceptance run measures the final plan against the moving station.

Pure game adapter/order tests passed (3 passed); they stub `plan_transfer`
and do not exercise the collocation solve. The cache-provenance change makes
unknown or stale reverse-row artifacts rebuild. The long collocation
acceptance command is:

```powershell
Set-Location C:\dev\Powershell\engine_toy
python -m pytest tests/test_orbital_collocation.py -q
```

On a reverse-row cache miss, this invokes the native slice-row lowering,
measured at about 15 minutes, followed by the arrival and cost rows. The
game-level regression command is:

```powershell
Set-Location C:\dev\Powershell\engine_toy
python -m pytest tests/test_orbital_game.py -q
```

Neither acceptance command was run for this integration update. The current
priority from the user's latest steering is to recover the missing steering
energy and `dt` metrics; no physics equations or their defaults were changed
here.

## Additional prerequisite: independent planning and native row ownership (2026-10-03)

The user's requirement that planning run independently and announce completed
plans remains open. This note records a read-only ownership review; it does not
implement an asynchronous planner or claim that results are announced.

The existing `SolutionService` is the nearby worker precedent:
`solution_service.py:101-164` snapshots its immutable `Conditions` reference,
solves on a daemon thread, swaps one completed `plan`, and offers nonblocking
`latest()`. It does not expose a ready-event callback; callers can observe the
plan/solve count. It also does not compare the captured conditions with the
current conditions before publishing, or validate the plan start against the
simulation clock after solve completion. Any independent planner integration
still needs a visible completion announcement and a current-state/stale-start
check before its plan is accepted for flight. Current `OrbitalGame.select()`
invokes the collocation planner synchronously (`orbital_game.py:482-517`; the
adapter calls `plan_transfer` at `orbital_game.py:221-230`).

There is a concrete native-buffer ownership issue to resolve before moving
collocation planning off-thread while the game reads a published plan. The
module-level `_slice_rows`, `_arrival_rows`, `_fuel_rows`, and `_cost_rows`
factories are `lru_cache`d (`orbital_collocation.py:457-479`), and each
transcription's cached properties return those same row objects
(`orbital_collocation.py:594-608`). `ReverseRows._execution` is a cached
property, not a dataclass field: it calls `prepare_artifact_execution` once and
retains one `LLVMExecution` plus its float64 scalar arena
(`orbital_collocation.py:231-249`). Calls write inputs/seeds into that arena
and invoke the same execution without a lock (`orbital_collocation.py:251-275`).

The published plan's `reference()` can use that same shared slice-row execution:
coast/reference branches call `transcription.advance` at
`orbital_collocation.py:1280-1303`; `advance` runs `step` repeatedly, and
`step` calls `slice_rows.values` (`orbital_collocation.py:727-740`). Concurrent
planning's `evaluate`/`flow` uses `slice_rows.jacobian` and sometimes
`slice_rows.values`, plus shared arrival, fuel, and cost rows
(`orbital_collocation.py:659-716`, `724-757`). Equal row keys therefore let a
planner worker and the game/reference reader mutate the same native feed, seed,
and output buffers. This is a source-level race path; no concurrency test was
run.

A fresh `dataclasses.replace(cached_rows)` wrapper would preserve the same
compiled artifact and row-ID maps while leaving the cached-property execution
behind, because `_execution` is not a dataclass field. Its first call would
therefore create a fresh `LLVMExecution` via the existing
`prepare_artifact_execution` factory. This is an existing per-execution ABI
allocation mechanism, not a new integrator or simulation. Each transcription
would need its own wrappers for the row families it uses: reference reads the
slice rows, while planner evaluation uses slice, arrival, fuel, and cost rows.
This is a source-based ownership conclusion; no runtime clone or race proof
was performed.

The shared artifact/DLL side was also inspected. The retained current
`training_motion__colloc_*.ll` files under `%TEMP%\\colloc_colloc_*` for
arrival, fuel, slice, and trip-cost contain only math intrinsics/libm calls;
there are no `turing_validation_error`, `external_slots`, `external_fault`,
optimizer/watch, or `@... global` declarations. The current row-generation
source constructs a strict ProcessGraph from symbolic expressions, lowers the
reverse/fused motion, then calls ordinary LLVM emission
(`orbital_collocation.py:282-360`). The artifact's `external_slots` is populated
from emitted-module slot references (`turing/src/compiler/ssa_llvm_backend.py:4489`);
the current inspected IR had none. The persisted row contract omits external
slot metadata, and `_load_cached_rows` reconstructs an artifact with
`llvm_ir=""` and default empty `external_slots`
(`orbital_collocation.py:398-408`, `437-448`), so this cache route relies on the
rows being free of external-slot calls.

The validation helper itself is thread-local:
`turing/src/common/tensors/accelerator_backends/c_backend/turing_validation_runtime.c:3-17`
uses `_Thread_local` state for set/reset/take. `LLVMExecution.run` resets and
reads it only when the artifact LLVM text contains a validation call, and
checks external-slot faults only when `external_slots` is nonempty
(`turing/src/compiler/ssa_llvm_backend.py:4610-4618`). No validation call or
external-slot table was found in the inspected collocation IR. The ordinary
row compilation path also does not wrap an optimizer loop or request watches;
optimizer state defaults to `None` and watch fields to empty on
`LLVMFunctionArtifact` (`turing/src/compiler/ssa_llvm_backend.py:4528-4544`).
Thus the current generated row kernels show no shared mutable artifact/DLL
state beyond the cached `LLVMFunctionArtifact` handle/function pointer; their
working ABI buffers are per `LLVMExecution`. This inspection covers retained
current IR and source, not every historical or future collocation artifact.

The previous-plan warm start does not invoke those native rows. At
`orbital_collocation.py:1335-1360`, it calls `previous._thrusting()` and then
reads the previous plan's times and throttles to form duration/throttle/burn
pieces. `_thrusting()` is NumPy-only: `slice_delta_v()` uses stored times,
throttles, propellant, and transcription design/dry mass/attitude through
`actuation_matrix` and `clamp_throttles` (`orbital_collocation.py:1213-1236`;
`orbital_actuation.py:352-400`). It does not call `reference`, `advance`,
`evaluate`, or a `ReverseRows` execution. The plan's own `reference()` does
mutate `_thrusting_cache` and `_coast` (`orbital_collocation.py:1285-1299`), so
multiple simultaneous reference readers would also share those plan-local
caches.

# Managed-dt mode conformance

The dt runtime has two execution modes. Scientific execution is transactional:
an inadmissible candidate changes no committed state, and the controller retries
with a smaller interval. Realtime execution may accept less than the requested
interval, records the accepted interval in `Metrics.advanced_dt`, and records
the remainder in the `time_slip` channel.

`BIND`, `DILATE`, `SUBCYCLE`, and `HOLD` are participant contracts used by the
controller. They are not execution modes. A simulator should publish only the
contract it can establish from its own physics.

## Inputs consumed by each mode

| Reasoning path | Engine-supplied facts |
|---|---|
| Scientific acceptance and retry | `snapshot()`, `restore()`, error values and presence, `hard_failure`, `dt_limit`, participant publications |
| Realtime causal clipping | callable or scalar `causal_ceiling_dt`, exact accepted state, `advanced_dt` |
| Realtime budget allocation | measured `proc_ms`, normalized scientific penalty |
| Physical `BIND` | a measured or derived physical tau whose fraction bounds the external step |
| Closed-form `DILATE` | equilibrium, positive relaxation tau, and an exact elapsed-time evaluation law |
| Internal `SUBCYCLE` | the real private cadence and an implementation that consumes the whole accepted external interval internally |
| `HOLD` | explicit absence of evidence that a larger step is safe |

Processing cost is measured at the common `MetaLoopRunner` boundary. Engines do
not estimate it. The runner stores cost and penalty under the same scheduled
node identity used by `compile_allocations`; this makes the next allocation
depend on the preceding measured run.

## Current dewar participants

| Top-level simulation | Scientific transaction | Realtime ceiling | Tau contract |
|---|---|---|---|
| Atmosphere chamber | complete package snapshot and rollback | previous chamber-law `dt_limit` | `BIND` when energy and power are present; otherwise `HOLD` |
| Mechanical machine | complete `MachineSim` snapshot and rollback | unbounded until a mechanical limit is derived | `HOLD` |
| Reciprocating engine | exact engine state span | 50 ms fixed-step accumulator capacity | 1 ms `SUBCYCLE` |
| Fluid circuits | complete shared circuit-ledger snapshot and rollback | unbounded until a circuit-wide external limit is derived | `HOLD` |
| Complex electrical network | device, phasor, loss, and safety snapshot | unbounded algebraic solve | `HOLD` |
| Thermal system | every temperature column and clocks | geometry/material stable step | energy/power `BIND`, otherwise `HOLD` |

The explicit `HOLD` rows are deliberate. Fluid and electrical state can still
advance in both execution modes, but neither currently has a general law that
justifies a fabricated tau. A future fluid residence-time law or electrical
device dynamic may replace its `HOLD` row when that law exists.

## Test battery

- `tests/dt_system/test_dt_mode_conformance.py` tests rejection, clipping,
  accepted-time accounting, rollback, and budget-ledger feedback.
- `tests/dt_system/test_time_contracts.py` tests every tau contract independently
  of execution mode.
- `tests/test_dewar_dt_mode_contracts.py` runs the real assembled dewar graph and
  proves independent top-level clocks by forcing only thermal to a 5 ms ceiling
  during a 20 ms realtime request.
- The engine-toy and spectral-analyzer focused tests assert the declarations
  emitted by engine, machine, fluid, thermal, and electrical participants.

This conformance work does not add catch-up dispatch. `time_slip` is now an
honest observable from every top-level participant, but accumulated clock debt
is not yet an allocation weight and every scheduled node still fires once per
realtime round.

# dt system: last-substep ratchet (2026-10-02)

Status: fixed, verified eagerly. Nothing committed.

## Defect

`Targets.energy_exchange_fraction=0.2` with pieces that publish no energy
channel: every participant is `HOLD`, and `exchange_time_bound` caps the
next proposal at `dt_current`, the step just taken. The final substep of a
round is clipped to land on the window, so its proposal equals the clipped
remainder; `run_superstep` returned it as `dt_next`, `advance_round`
carried it as the next round's first attempt, and dt ratcheted down.

Repro (scratchpad `repro_ratchet.py`, 2 s): orbital jumper circular orbit,
window = period/20 = 291.4 s, CFL dt 3.31 s, read-only hook on
`step_with_dt_control_used`.

| round | before: start dt | after: start dt |
|---|---|---|
| 0 | 3.313 | 3.313 |
| 1 | 3.275 | 3.311 |
| 2 | 3.185 | 3.3095 |
| 3 | 1.608 | 3.3095 |
| 4 | 0.4546 | 3.3095 |
| 5 | 0.0243 (substep cap hit) | 3.3095 (through round 7) |

Writer, observed by hooking `participants.exchange_time_bound`: final
substep `dt_proposed=3.2915`, `dt_current=3.2755` (clipped), bound
3.2755, contracts all HOLD; `run_superstep` `last_dt_next = dt_next`.

## Fix

- `src/common/dt_system/dt_controller.py` `run_superstep`: a substep whose
  `dt_try` was clipped below `dt_cap` (window remainder or event boundary)
  does not author `last_dt_next`. The continuation is the last unclipped
  substep's proposal, or the round's opening `dt_cap` when every substep
  was clipped.
- `examples/llvm_dt_system.py` `advance_round`: no longer pre-clips the
  carried dt with `min(window, carried)`. That made a short window's length
  the unclipped controller step, so under HOLD it became the continuation
  and later longer windows never recovered (repro: a 1.0 s window then a
  full window returns to 3.3095 s). `run_superstep` already clips. No new
  fields; `dt_system_over` (the lowered entry) is unchanged.

## Absent energy channel reads HOLD: intended

`participants.py` documents silence as `HOLD`; `Targets` and
`exchange_time_bound` document HOLD as "do not grow, do not shrink".
Not a second defect and not the same writer; unchanged.

## Not changed (same pattern, different writer)

Mid-round event-boundary clips still feed `dt_cap = min(dt_cap, dt_next)`
for the rest of that round under HOLD. Not hit by any current caller here.

## Verification

- `engine_toy/orbital_jumper.py`: `energy_exchange_fraction=0.2` restored.
  `tests/test_orbital_jumper.py`: 3 passed.
- `tests/dt_system/test_llvm_dt_system.py`: 2 passed.
- Woodshop eager, five `world_rules.step(1/240)`: every round one unclipped
  substep with carried == window, so inputs are identical by construction;
  `momentum_z[0:3]` = -0.15207499555165677, -0.3748003796594457,
  -0.500138540948139 (reference -0.152075, -0.37480038, -0.50013854).

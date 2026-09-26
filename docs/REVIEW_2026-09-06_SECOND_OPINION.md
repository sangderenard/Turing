# Second-opinion review: full-native DT compiler frontier (2026-09-06)

Scope: read-only review of the saved artifacts against the authored source.
No compiler edits, no builds, no driver runs.

Evidence used:

- `build/full_formal_diagnostic/formals.json` (18:33, 23 findings)
- `build/full_formal_diagnostic/repository-ssa.pkl` (18:33)
- `build/step_field_state.pkl` (18:43, bounded step with the field ledger)
- `src/common/dt_system/dt_controller.py`, `dt_scaler.py`
- `src/compiler/vehicle_python_compilation.py` (call site, state class, advance)
- `src/common/tensors/topological_reducer.py` lines 1100-1115, 2055-2165,
  2830-2860, 2965-2990, 3360-3610, 4649-4654 (the field ledger)

Ids below are the ids in the saved SSA pickle. They are one renumbering off
the ids quoted in the continuation: the returned Metrics record is 1313 with
fields 1449..1462, the `coerce_metrics` record is 438 with fields 1345..1358,
`error_channels` is the triplet 1354/1355/1356, `hard_failure` is 1357.
`producer_locations.txt` and `missing_consumers.txt` are from the 13:48 run
(33 findings, old ids) and should not be read against `formals.json`.

Call-site constants that the saved SSA has already folded (from
`balloon_tire_managed_window`): `allow_increase_mid_round=True`,
`allow_unresolved=False`, `max_retries=None`, `rollback=True`,
`attempt_log=None`, `substep="steered"`, `event_boundaries=()`,
`distribution=None`, `targets.energy_exchange_fraction=None`.

## Headline

1. The 23 findings are real, but the formal gate is not a completeness
   gate. It only fires when something consumes a fabricated formal. The same
   SSA contains three defect classes that produce no formal at all: values
   placed before the effects they depend on, dropped writes, and ABI
   stand-ins. Ten such defects are listed under S1-S10 with block and
   instruction evidence. Driving 23 to 0 will not by itself produce a correct
   native controller, and two of the ten (S1, S2) make the compiled rejection
   logic wrong on every attempt.
2. Seven of the 23 are consumed only in code that is already unreachable
   under the folded constants. They need reachability pruning, not producers.
3. For 516/517, record-field SSA versioning is the right model for the scalar
   field and the wrong model for the keyed field. Both fail today for one
   reason: the return merge and the record layout never consume a field
   version. The ledger repro proves the graph half works and the physical half
   does not.
4. The receiver-identity ledger is unsafe in five situations, four of which
   occur in this source.
5. The remaining producer problems group into seven general mechanisms, none
   of which needs a DT name, id, source rewrite, callback, or fabricated input.

## Q1. Does `formals.json` account for every non-authored formal?

Yes for formals; no for defects. The gate is complete over `function.args`
minus named/storage/member/ABI formals, and every one of the 23 has zero
definitions and only real consumers. But the accounting exemptions are where
equivalent fabrications hide:

- `program_abi_field_written` formals are exempt even when the function's own
  writes to that field are lost (S3, `ctrl.clamp_events`).
- `linked_call_frame_storage` formals are exempt even when they stand in for
  a call result (S5, `%1418`).
- Writes that never reach a consumer produce no formal at all (S4, S7).
- Values scheduled before their effects have real definitions and pass (S1,
  S2, S10).

Classification of the 23 by source role:

| function | id | source role | class |
|---|---|---|---|
| run_superstep | 284 | `boundary_values` arena: `tuple(sorted({...}))` over `event_boundaries` | M3 comprehension producer; constant-foldable to an empty arena under `event_boundaries=()` |
| run_superstep | 174 | `callable(hint)` | M0 fold: `BalloonTireManagedState.dt_limit_hint` is a method, so this is statically `True` |
| run_superstep | 57 | `hint()` | M2/M4: a method call on the state record; `declared_dt_s` is already in the state ABI |
| run_superstep | 240 | `(metrics.error_channels or {}).get("dt_unresolved", 0.0)` on the step call's returned record | M2 projection link: the triplet 693/694/695 is loaded from the call aggregate but never linked; `lookup_or_default` helper already exists |
| run_superstep | 124 | `first.unresolved_report` | M0 fold: undeclared attribute read through `getattr(..., ())`; only assigned in the `allow_unresolved` arm, which is dead |
| run_superstep | 126 | `[1:]` of 124 | dead with 124 |
| step | 559 | `metrics.osc_flag or metrics.stiff_flag` as a call argument | M5 value-position `LOr` of resident 1349/1350 |
| step | 516 | `metrics.error_channels = channels` (floor arm) | M4 keyed-field version |
| step | 517 | `metrics.hard_failure = False` (floor arm) | M4 scalar-field version |
| step | 422 | `len(floor_reasons)` | M3 length of the copied arena |
| step | 421 | `floor_reasons = tuple(reasons)` | M3 arena copy |
| step | 444, 452, 470, 478 | four `f"mass_err ... > ..."` / `f"div_inf ..."` reasons | M6 formatted token |
| step | 248 | `f"timestep controller failed after {len(failures)} attempts:"` | M6 formatted token |
| step | 207 | a formatted line in the `unresolved` report | M6, but consumed only in `if_true.19`, reachable only from `while_exit`, which `while True` never reaches |
| step | 488, 518 | the two `attempt_log.append({...})` rows | dead: `attempt_log=None`; the consuming block `unreachable_return_control` has no predecessors |
| step | 408 | `rejected` at the while exit | dead: `%715 = Phi(True, True)`; `while_exit` is unreachable |
| _no_exchange_observed | 11 | `channels["power_w"]` under `"power_w" in channels and ...` | M2/M5 guarded keyed load; the whole `LAnd` starts with `Const False` (`fraction is not None` folded), so the value is dead here |
| _propose_dt_pen | 49 | `max(..., *(ratio for name, limit in error_limits.items()), 1.0)` | M3 reduction over the generator loop; also S10 placement |
| coerce_metrics | 0 | the dict-comprehension row `channel` | M3; see S9, the whole comprehension is absent, not one row |

Dead under the folded constants: 488, 518, 408, 207, 124, 126, 11 (seven).

## Q2. Are the 23 truly missing producers, or existing definitions disconnected?

Disconnected, not missing: 240 (the returned triplet is loaded three
instructions earlier), 174 and 57 (statically resolvable method), 559 (both
operands resident), 11 (helper exists). Missing: the comprehension family, the
formatted strings, and the record-field versions.

More important are the defects with no formal at all. Each is verified in the
saved SSA by control-flow order (reverse postorder from `entry`), not by dump
order.

S1. `rejected = bool(reasons)` is evaluated before four of the five reason
appends. `%310 = Gt(len(seq 76), 0)` sits at `if_merge.2` index 18. The
appends for `not ok` (`if_true.12`), `hard_failure` (`if_true.8`), mass
rollback (`if_true.9`), div rollback (`if_true.10`) and the floor arm's
`reasons.clear()` (`if_true.5`) all execute after it. Only the channel
rollback append (`loop_body.1`) precedes it. `%406 = Phi(%405, %310)` is the
`rejected` that gates the `continue`. Native rejection ignores four rules.

S2. The accept path is hoisted above the rejection test. `_propose_dt_pen`
(`%556`), `pi_update` (`%560`), `_apply_energy_sidechain` (`%570`) and
`update_dt_max` are in `loop_exit.1` and `if_merge.2`, executed on every
attempt. In Python a rejected attempt `continue`s before any of them.
`pi_update` and `update_dt_max` mutate `ctrl`.

S3. `ctrl.clamp_events += 1` is lost at all three sites. `planned_region_1`
computes `%1357 = Add %144 1` and returns `%146`, which is its own input.
`value_aliases` maps 414, 146, 415 to the parameter 144. The formal 144 is
exempt because `program_abi_field_written` is true.

S4. Terminal-arm field writes never reach the return. The failure arm writes
`metrics.hard_failure = True` and `metrics.error_channels = channels` before
`return metrics, dt*0.5, 0`. `function_exit` has `%1461 = Phi(1357, 1357,
1357)` and `%1458..%1460 = Phi(1354.., 1354.., 1354..)` on all three return
edges. The floor arm's writes are the 516/517 formals on both Phi edges.
`%536` and `%542`, the field Phis, have no consumers at all.

S5. `max_vel` and `max_flux` of the advance result are replaced by formal
`%1418`, accounted as `material.last_maximum_velocity_m_s` with
`split_from_unproven_alias: 1344`. Aggregate indices 1 and 2 of the advance
call are never loaded. This is right only because
`balloon_tire_managed_advance` assigns that field immediately before building
`Metrics`. As a mechanism it is unsound, and a by-value scalar formal cannot
carry a per-retry value unless the link makes it a storage slot. Verify.

S6. `run_superstep`'s while condition is stale. `%71 = round_max - total`
is computed once in `entry` with the initial total. Both `while_header`
(`%364 = Gt %71 eps`) and `while_latch` (`%365 = Gt %71 %456`) reuse it. The
loop exits only through the `max_iters` and `dt_used <= 0` breaks, so the
native window performs one extra attempt at `dt_try = 0` before stopping.

S7. `last_metrics = metrics` is dropped. `record_return_layouts` for
`last_metrics` (291) is the 11 entry constants 61..65 and 500..505 from the
`Metrics(0,0,0,0, hard_failure=True)` literal. The returned record carries no
`error_channels` triplet, and the `remaining > eps` block that annotates the
channels has no `CondBr` anywhere. The caller's `metrics.hard_failure` test
in `balloon_tire_managed_window` would always see `True`.

S8. The mid-round `ctrl.dt_min` / `ctrl.dt_max` clamps on `dt_cap` are
absent: `%368 = Phi(215, 250)` takes the raw `dt_next` loaded from the step
call. The `dt_used <= 0` break port `%290` selects the current `dt_next`
where the source keeps the previous `last_dt_next`.

S9. `coerce_metrics` lowers to `Ret` of its 14 input formals plus one dead
`_scalar` call on the comprehension formal. The `isinstance` arm, the
`setattr` loop and the whole comprehension are gone. Numerically harmless in
native (fields are already float64), but it is why the record's
`error_channels` projection 455 is `unresolved`, and it shows the
comprehension mechanism is absent, not one row of it.

S10. In `_propose_dt_pen`, `planned_region_2` (the `max`) consumes `%49` at
`if_merge` index 1, before the generator loop that appends each ratio to
sequence 52. The loop's arena never reaches the reduction. Same root cause as
S1 and S2.

## Q3. Step values 516 and 517

Trace in the full SSA. Source 517 is `metrics.hard_failure = False` in the
floor arm (`if_true.5`). Its RHS is a constant. The reducer's old effect-token
path produced `%1882 = Phi(517, 517)` at `if_merge.5` and `%536 = Phi(517,
517)` at `if_merge.4`, the formal on both edges. Source 516 is
`metrics.error_channels = channels` in the same arm. `channels` is resident
as keyed sequence 539 (storage 663) built by `dict(...)` copy (sequence 513)
and two keyed stores. `%1879 = Phi(539, 513)` shows the handle Phi the
reducer emits for the dict itself. `%542 = Phi(516, 516)` is the field.
Nothing downstream consumes 536 or 542. `function_exit` returns 1357 and
1354..1356 on all edges, so neither write reaches `Ret`.

Trace in `build/step_field_state.pkl`. The ledger produced the correct graph
shape for the scalar: `%529 = Phi(%153 False, %1346)` at `if_merge.8` and
`%530 = Phi(%529, %1346)` at `if_merge.7`, named `metrics.hard_failure`,
with `initial_value_id` the callee's returned field. But
`record_return_layouts` is `(324 -> 1334..1347)`, the `coerce_metrics`
result's own layout, and `Ret %1296 = Phi(%1455, %1455, %1455)` returns the
callee record wholesale. 530 has no consumer. The keyed field became
`%576 = Phi(%575, %353)` with `%575 = Phi(%392, %353)`, dtype `None`, where
353 is a fabricated formal for the initial projection. The ledger also
emitted 395, 554, 550 as fabricated formals (a second Phi `%1529 =
Phi(%529, %395)` for the same field under a different key, and the `channels`
RHS). Unaccounted formals in the bounded repro: 27. So the ledger currently
violates the rule that a field with no proven projection must stay loud.

Recommendation, no deletion and no aliasing:

- Scalar fields: keep versioning. The `function_exit` return-merge Phis
  already carry `record_phi`, `record_field` and `initial_value_id`; their
  incoming values must be the ledger's version at each return edge, including
  terminal arms. That single change fixes 517, S3 and S4 for scalars.
- Keyed fields: not a Phi and not a new operator. The record already owns the
  physical arena (the triplet). Lower `record.keyed = local_dict` to an arena
  copy-assign into the field's own storage, versioned as an ordinary store on
  one storage (the store-chain identity rule: the region publishes the root),
  and from that statement on bind the local name as an alias of the field
  arena so later `local[key] = v` writes land in the field. `dict(record.keyed
  or {})` is the reverse copy into a local arena. A conditional assignment is
  then a conditional store on a single storage, which existing machinery
  already handles. This also repairs S7's `last_metrics.error_channels`.
- Do not remove 516/517 until the return layout and both branch outcomes are
  proven in the native/eager regression the continuation asks for.

## Q4. Is the ledger safe under receiver identity alone?

The ledger is keyed by `(resolved receiver node id, attribute)`; `SetAttr`
records its RHS node, `GetAttr` records the projection node, and only the
`if` handler snapshots and merges it (reducer 3372-3404, 3527-3609). Unsafe
cases, each with a witness in this source:

1. Loops. No loop header snapshot or merge exists. A field written in
   iteration N and read at the top of iteration N+1 reads the pre-loop value.
   Witness: `ctrl.clamp_events += 1` in the retry loop, and every
   `initial_value_id` inside the loop pointing at the pre-loop projection.
2. Terminal arms. `if body_terminal and not else_terminal` keeps the
   else-state, which is right for fall-through but drops the body's versions
   without carrying them to the return edge. Witness: S4.
3. Aliasing through a call. `metrics = coerce_metrics(metrics)` returns the
   same object in Python; the graph gives node 438 versus 437, so two ledgers
   exist for one object and writes through one are invisible through the
   other. `m` from `failures` rows and `first = unresolved[0]` are the same
   shape. `last_metrics = metrics` only works because both names resolve to
   one node.
4. Mutable RHS after assignment. The ledger stores the RHS node. In the
   `allow_unresolved` arm the source does `metrics.error_channels = channels`
   and later `channels["dt_unresolved_report"] = 0.0`; Python's field sees
   the key, the ledger's version does not. Receiver identity plus RHS node is
   not enough for a mutable field; storage identity is required.
5. First observation inside an arm. A field not in `before_attribute_values`
   gets no Phi and its arm write is silently dropped from the ledger rather
   than diagnosed. Witness: `last_metrics.error_channels = channels` inside
   `if remaining > eps` (S7). The continuation's "loud diagnostic" intent is
   not implemented; the drop is silent.

Nested conditionals are handled correctly as long as the field was projected
before the outer `if`: the inner merge writes the Phi into
`attribute_value_nodes`, and the outer merge sees it as the arm value.
Relabeling remaps the `record_field_state` receiver (4649), but the
`binding_name` `field:<id>.<attr>` bakes the pre-relabel id into a name;
cosmetic, but a sign that the key is not a stable identity.

## Q5. General mechanisms

M0. Reachability pruning after specialization. `CondBr` on a constant,
`LAnd`/`LOr` with a constant absorbing operand, `while_exit` of a `while
True`, blocks with no predecessors, `callable` of a resolved method,
`getattr` with a default on an attribute outside the live field set. Removes
seven findings and the dead blocks that pad every later diff.

M1. Effect-ordered scheduling. Every read of a sequence (truth, length, keyed
lookup, reduction) is ordered after the last reachable effect on that
storage, exactly as `attribute_effect_nodes` already orders attribute reads
after attribute writes; effectful calls that mutate a record parameter are
anchored to their authored control position. Fixes S1, S2, S10. Add a
stale-read audit to `ssa_self_check` beside the formal gate so this class is
counted, not just the formal class.

M2. Record projection linking for call-returned records and keyed fields.
Link `.get`, `[]`, `or {}` and `in` on a returned record's keyed field to its
physical triplet through the existing `lookup_or_default` and `contains`
helpers. Covers 240, 11, the `unresolved` receiver 455, and the method call
57 on an ABI field.

M3. One comprehension lowering. A loop region that appends into a resident
arena (list, tuple, dict, set comprehensions, `tuple(seq)`, `dict(seq)`,
`sorted` with a small sort helper) and a reduction-accumulator variant for
`max`, `min`, `sum`, `any`, `all` over a generator. Covers 284, 421, 422,
49, 0 and S9.

M4. Record field versioning as described under Q3: scalar versions flow into
per-edge return merges; keyed assignment is an arena copy-assign with alias
rebinding; ledger keyed by storage identity, snapshotted at loop headers, and
loud on first observation inside an arm. Covers 516, 517, S3, S4, S7.

M5. Value-position boolean operators. `LOr`/`LAnd` already exist as
control expressions; they need to be emitted as values (559). Guarded keyed
loads (11) need no control once the load is total through
`lookup_or_default`.

M6. Formatted string tokens. `format_token(template_token, operands...)` as a
hash-combine over the template token and each operand's formatted token.
Preserves string equality semantics that `len({tuple(why) ...})` relies on,
without Python. Covers 444, 452, 470, 478, 248, 207, 488, 518 (the last three
only if M0 does not remove them first).

M7. Loop-current expression recovery for `.item()` while conditions and for
statements after the loop's call (S6, S8). Possibly the already accepted
"loop-current scalar expression recovery" not applied to `Cast item`
conditions.

## Smallest general order of repairs

1. M0 pruning. 23 becomes 16. No producer added, no risk.
2. M1 ordering plus the stale-read audit. The count does not move, but S1,
   S2 and S10 are corrected, and every later producer would otherwise be
   consumed before it exists.
3. M4 record fields with the native/eager regression the continuation
   specifies (child returns a record, caller conditionally changes one scalar
   field and one keyed field, caller returns the record, both outcomes).
   16 becomes 14; S3, S4, S7 close.
4. M2 projection linking. 14 becomes 12 (240, 57; 11 is already pruned).
5. M3 comprehension producers. 12 becomes 7.
6. M5 and M6. 7 becomes 0.
7. M7 loop-current conditions, then the full driver, then one detached
   native build, then frame parity.

Do not remove 516/517 first, and do not call the ledger accepted while
`step_field_state.pkl` shows 27 unaccounted formals and a return layout that
ignores the field versions.

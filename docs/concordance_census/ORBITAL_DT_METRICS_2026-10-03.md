# Orbital craft dt metric audit — 2026-10-03

Current work uses the actual library RK4 algorithm and the existing dt
state, window, rejection, and rollback mechanisms. Native rotor and
configured pure-torque checks pass, as do isolated production actuator,
derivative, and energy-publication checks. The complete 27-piece craft bank
now builds below the memory ceiling, and its command-change gate passes on
the final finite-headroom controller (`6a879db3`).
Long-duration angular momentum, desaturation, near-empty cutoff and the
ordinary 20-second burn plus cutoff now also pass on that controller. The
full-supply exact-one readout fails; its diagnostic quotient and an initial
mass summation mismatch are traced below. Frame-partition and game acceptance
remain open.
The completed-piece memory accumulation now has a verified archive/runtime split. The later
sections record the ownership trace, verified compiler commits, failures,
and measured gates. The first section preserves the audit that motivated
the correction; its missing-publication findings describe the old source.

## Compiler audit archives and runtime retention

The user elected disk storage for the full compiler books instead of raising
the existing approximately 6 GB process ceiling. Turing `c73c2a26` adds
`LLVMPiece.for_runtime()` and explicit `retain_compilation` save/load options.
Copies retain exact column ABI, native LLVM text (including validation), entry
identity and `PieceCompilerRecord`; they release SSA/source/output metadata and
the artifact's emission book without mutating the full original. Runtime-only
pieces explicitly refuse source linking, which requires the full archive.
Retention disposition is recorded on the existing piece-build book row.

Root commit `c4b4fd2` adds `equation_piece(..., retain_compilation=False)`:
compile with concordance as before, persist the complete `.piece` archive,
publish a separate `.runtime.piece`, and retain only the latter in memory.
Normal runtime loads use that smaller file. A current full cache can supply
the first runtime index; existing archives are not overwritten by compact
pieces. The two orbital dt factories opt in in the still-unverified physics
working tree, with an explicit retention option for compiler callers.

Verification: four native compiler-payload tests pass in 2.84 s (peak process
memory 1,626,808,320 bytes); two native full/runtime cache tests pass in 9.03 s
(peak 1,630,035,968 bytes). These cover repeated execution into registered
columns, constants, source stamps, original archive preservation, both cache
creation orders, legacy loading and source-link refusal. The original tiny
law's concordance report is unchanged: zero structural findings and 1,128
unsourced facts, OPEN. The complete craft command-change gate is recorded below;
these small compiler tests alone do not establish flight acceptance.

A read-only comparison loaded the same 19 historical production archives
(714,433,717 bytes) one at a time with `retain_compilation=False`, retained all
19 runtime pieces and collected unreachable cycles. No stale DLL was executed
and no archive was modified. Peak sampled RSS was 766,705,664 bytes; ending
RSS was 435,572,736 bytes. Compact serialization totaled 7,879,555 bytes and
preserved 6,883,464 characters of LLVM text plus the compiler records. All 19
passed absence checks for module/source/outputs/emission. The older full-graph
estimate below is a summed reachable-object size, not measured process RSS.
This comparison establishes retained-load behavior, not peak memory during
a complete fresh bank compilation.

That read-only measurement used `turing/.venv/Scripts/python.exe` (Python
3.11.7), sampling Windows working set every 50 ms; native regression builds
used system Python 3.11. The separate streaming compression check on the
largest original (`stage13`) reduced 61,223,411 bytes to 10,470,417 bytes with
zlib level 1 in 0.229 s. This was a no-write archive-size measurement; no
compressed cache format or loaded-memory improvement is claimed from it.

Memory-mapping a pickle would still reconstruct its Python objects when
unpickled. Compression can reduce archive bytes and I/O, but does not eliminate
those live objects; no alternate file-backed book implementation was added.

A separate archive-read defect was exposed while tracing the planner's
finite-fuel Jacobian: `ArgumentBindingFact`, a two-field tuple subclass,
inherited tuple's one-pair pickle reconstruction arguments but required two
constructor arguments. The existing declaration now supplies both arguments
for new archives and accepts exactly the old two-item tuple for old archives.
It preserves source-object aliases and explicit `None`; no identity or schema
is inferred. Nine focused tests pass for protocols 0 through 5, deepcopy,
shared aliases and an actual pre-fix byte fixture. The previously recorded
native child-record return failure now passes (1 test, 4.94 s; sampled root/
child peak 1,574,211,584 bytes), exercising all four initial/guard combinations.

The actual 92 MB orbital RK4 SSA archive can now be read normally. Its audit
matches the saved receipt exactly; writing and reloading it again preserves
that audit, the root argument IDs and the named output IDs. This check takes
24.16 s at a 3,398,623,232-byte sampled peak. The audit remains 6,637 rows /
332 functions, zero structural findings, 72,487 unsourced facts and 10
unsourced identities, OPEN. No stale native artifact was executed. This
repair establishes archive reconstruction, not planner Jacobian acceptance.

The previously pending postcommit properties and inverse checks now pass in
the runtime-payload configuration (two tests, 113.95 s, peak 1,441,718,272
bytes). COM matches within `1e-12`; inertia retains `rel=1e-11, abs=1e-10`;
inverse times spin-free inertia matches identity within `1e-12` with zero
relative tolerance. The copper-updated coupled rotor gate also passes
(48.94 s, peak 1,006,317,568 bytes): 140 attempts, world angular-momentum
drift `2.30317515257e-9 Nms`, copper ledger `0.013 J`, and electrical minus
copper minus mechanical work below `1e-12 J`. Raw concordance debt remains
OPEN (properties/inverse each 19 findings; rotor 21). These isolated and
small-system results do not establish the full 27-piece craft flight.

## Adaptive growth correction

The user rejected a blanket no-growth policy. The proposed change that would
have retained a smaller accepted retry as the round's cap was stopped before
any production edit, and the new tests imposing that requirement were removed.
Earlier attempt sequences remain observations, not a specification to enforce.

The false `SuperstepPlan.allow_increase_mid_round` default originates in
`aeb6b7c8` (2025-08-16), whose message is "intigrating dynamic dt controller,
bugs remain". More directly relevant, `llvm_dt_system.dt_system_over` and
`RoundPiece` did not forward the graph's growth choice at all, and the
PieceState ABI did not carry it. OrbitalJumper also inherited the false plan
default. Existing direct graph and managed-time paths already support growth;
the managed tire source explicitly enables it.

The repair now enables existing adaptive defaults and forwards that choice
through the LLVM-piece state/ABI. OrbitalJumper explicitly requests it. No
new step-size algorithm or no-growth cap is being added. Declared physical
limits, rollback, fixed outer windows and explicitly pinned operation remain
the acceptance contract. Turing commit `78c4f502` passes 31 focused tests,
including native growth, rejection/rollback, fixed-window landing and bool
field transport (20.31 s; peak max(working set, private bytes) 1,980,862,464
bytes). Full coordinator compilation remains unverified. The 40-window
pure-torque gate also passes on this configuration (113.96 s including its
rebuild), with unchanged 7,003 attempts and physical tolerances.

## Error feedback inside the existing controller (`cf78f24f`)

The user explicitly permitted nonlinear rejection sizing. The default
proposal now uses the actual attempted dt and the largest declared error
ratio R: `min(CFL/energy bound, attempted_dt / sqrt(R))` for positive R.
Zero error supplies no finite error bound. Ratios below one retain measured
headroom. This is a controller response, not an assertion that every RK4
metric has the same convergence order. The existing three-argument custom
distribution contract remains unchanged.

Both accepted and rejected evidence now feed the existing log PI controller.
Rejected publication spans are consumed before the existing state restore;
no new snapshot or history mechanism was added. A failure with no useful
error estimate, or a PI proposal that fails to shrink, uses the existing
halving fallback. Declared energy/participant limits, dt floors, acceptance
thresholds, fixed outer windows, and rollback retain their existing owners.
There is no no-growth cap.

The revised focused selection passes 21 tests in 26.56 s (sampled peak
1,938,812,928 bytes). Three stricter native checks subsequently pass in 14.39 s
(sampled max(working set, private bytes) 1,756,524,544 bytes). These execute
the authored helper under the existing full-native dt contract, for zero
and two participants and an appended conservation channel. Representable
results retain exact comparisons; the square-root result matches the
authored helper within one ULP. The larger interrupted run is not counted
green. Two older persistent-failure tests remain unchanged and excluded:
their expected raise conflicts with existing floor-retention/default
unresolved behavior. Bounded floor retention and explicit strict rollback
are exercised separately.

Compiled sixth- and eighth-power error laws verify growth, an abrupt command
change, rejection restoration into the original registered span, and exact
0.5-second window completion. The sixth-power case takes 17/80/2 attempts
at gain 8/32/0; the eighth-power case takes 10/42/2. Those are measured toy
law results, not orbital performance acceptance. Production banks require
normal current-stamp regeneration before measuring the same flight gates.

The generic Python-host extraction contract is not a whole-native execution
contract. A helper test initially used it and directly executed an extracted
artifact; its missing square-root contribution was not evidence of a bug in
the full-native numerical path. The source occurrence is classified as a
host/native-extension boundary, but its absence is not fully represented in
the aggregate extraction-boundary accounting. That accounting remains OPEN.
The corrected tests use `dt_system_contract` and select the unique returned
export carrying the requested source-root identity.

An adjacent source-order test remains red: the original
`test_item_capture_depends_on_its_operand_producer` reproduces on untouched
commit `0d101e3c` in `C:/dev/Powershell/.dt-feedback-baseline` (1 failed,
9.48 s). Its two `use-not-dominated` findings are in `_energy_time_limit`:
values 35 and 46 are consumed by `function_exit` isfinite instructions but
defined in `if_merge` blocks 2 and 11. This matches the working-tree failure
and establishes that it predates the feedback correction. No compiler fix
or concordance-debt closure is claimed.

Focused command, from `C:/dev/Powershell/turing`, using system Python 3.11
and `PYTHONPATH=C:/dev/Powershell/turing`:

```powershell
python -u -m pytest -p no:faulthandler -q --tb=short --show-capture=no tests/dt_system/test_dt_error_feedback.py tests/dt_system/test_dt_superstep.py::test_error_beyond_soft_band_restores_then_retries tests/dt_system/test_dt_adaptive_growth.py tests/dt_system/test_llvm_dt_channel_layout.py tests/dt_system/test_participant_energy_sidechain_native.py tests/dt_system/test_tensorized_dt.py::test_real_proposal_consumes_channel_spans_natively tests/dt_system/test_tensorized_dt.py::test_unpublished_nan_and_zero_limit_do_not_become_measurements
```

The subsequent stricter run selected
`test_real_proposal_consumes_channel_spans_natively` (two parameter cases)
and `test_declared_conservation_error_reaches_native_proposal` from that
same selection. No source changes followed it before the orbital rebuild.

The first post-feedback physical result is green: the unchanged 40-window
torque gate passes in 115.73 s including its nine-piece rebuild, at a sampled
peak of 2,186,260,480 bytes. Ten simulated seconds now take 6,828 attempts,
5,165 above-limit attempts and 1,663 within-limit attempts. Compared with
7,003 / 5,594 / 1,409 before feedback, total attempts fall only 2.50%; no
material runtime speedup is claimed. Maximum Gram defect is
`1.77635683940025e-15`, determinant `0.9999999999999982`, phase
`2.39999999999983` versus `2.4`, and angular speed `0.480000000000000`.
The original integrated-force and all other physical bounds pass. A bounded
cached observation is examining the remaining rejection cost before any
further controller decision. The full 27-piece rebuild is temporarily held
while the separate planner Jacobian failure is traced.

### Verified finite headroom correction

A cache-only observation of that same torque case completed in 15.64 s
(peak 1,597,399,040 bytes) and reproduced 6,828 attempts / 5,165 rejections.
No compilation fallback was allowed. In window 20, a trial of
`0.00603750082364 s` published exactly zero for every judged error. The loose
CFL proposal was about `5e35 s`; the existing PI accumulator jumped from
`0.1934164113` to its `1.5` bound, returning about `3.9769e13 s` before the
existing outer-window/sidechain limits. Later rejected trials could still
receive a growing PI proposal because of positive accumulated feedback, then
fall back to halving. Error granularity was observed; its cause is unproven.

Bounded experiments kept the real controller, rollback and window loop and
changed only its proposal transfer. `2h/(1+sqrt(R))` removed the singularity
but produced 230 consecutive above-limit retries for an eighth-power error,
approaching `R=1` from above. `2h/(1+R)` also approached that boundary slowly
in a small-start sixth-power case. Neither was applied to production.

The selected proposal is `min(CFL/energy bound, 1.8h/(1+sqrt(R)))`. Its
zero-error proposal is finite and permits repeated growth; its proposal
equilibrium is `R=.64`, while acceptance remains the original `R<=1`.
It does not claim a universal integration order or add acceptance slack,
new controller state, a no-growth cap, or a rollback mechanism. Long coasts
can still eventually saturate the existing accumulator; the correction
removes the single-zero-measurement jump, not all integral saturation.

All 23 targeted nodes pass across an initial 18-pass / five-failure result
and a corrected five-node rerun (4.43 s, peak 1,625,178,112 bytes). Corrections addressed a
test reading telemetry as Metrics and a growth sequence clipped by the
remaining window. Native power-six/eight laws now cover initial dt `.05`
and `1e-6`, command changes, four zero-error windows and a restart. Small-start
ignition takes 27/2 and 28/3 attempts/rejections; subsequent restart takes
33/11 and 30/10. Every variant recovers single full `.5 s` zero-error steps.
Registered span identity, exact windows and rollback pass. Full-native
proposal helpers keep exact representable comparisons and one-ULP parity.
The unchanged physical torque gate then passed in 98.07 s including its
nine-piece rebuild (peak 2,187,173,888 bytes). A guarded cache-only invocation
of the identical test emitted its endpoint receipt in 4.235 s (peak
1,593,548,800 bytes), with both compiler entrances forbidden and normal
source-stamp validation. It used 1,081 attempts, one metric rejection and
1,080 within-limit attempts for the same 40 windows / 10 simulated seconds.
This is an 84.17% attempt reduction against `cf78f24f`'s 6,828 and an 84.56%
reduction against the original 7,003. Differently instrumented/cold timings
are not used to claim a wall-time speedup.

Maximum Gram defect is `6.66133814775094e-15`, below the unchanged `1e-14`
limit; determinant is `0.9999999999999933`, phase `2.39999999999929` versus
`2.4`, and angular speed `0.480000000000000`. The original integrated-force,
rotation and phase assertions all pass. No source changes followed this
result. The full 27-piece craft bank remains held while the independent
planner Jacobian trace determines whether another compiler edit is needed.

## Complete craft command-change gate with finite headroom

The current-stamp 27-piece c0 bank and unchanged command-change test pass
against `6a879db3`: 1 passed in 1,478.85 s, including a 1,442.193 s constructor.
The sampled peak max(working set, private bytes) is 2,969,436,160 bytes,
below the unchanged 6 GB ceiling. The normal source/ABI cache validation was
used throughout. No stale artifact or memory-cap bypass was used.

| Two-second command phase | Attempts | Error-limit rejections |
| --- | ---: | ---: |
| Wheel ignition | 273 | 81 |
| Wheel reversal | 271 | 90 |
| Main ignition | 1,360 | 356 |
| Main cutoff | 1,086 | 188 |
| Cold coast | 901 | 205 |
| Total | 3,891 | 920 |

All 2,971 accepted attempts complete the ten-second command sequence. The
pre-feedback baseline was 27,556 attempts / 23,969 rejections / 3,587 accepted.
The same test's five-phase runtime is 30.123 s versus the prior 186.260 s;
these phase timings exclude compilation and construction.

Every original angular-momentum, electrical/copper/mechanical ledger,
tank/mass/impulse, integrated-delta-v, exact no-burn coast, and registered-span
identity assertion passes. Final Gram defect is `1.82051943793e-15`, mass
`979.353748255 kg`, burned propellant `2.52625174452 kg`, electrical energy
`903.408316553 J`, and copper loss `900 J`. Adaptive growth remains enabled.
The c1 frame/game gates remain separate work.

The following guarded cache-only c0 batch finishes with four passed and one
failed in 1,778.19 s, at a 2,507,046,912-byte peak. Compiler entrances were
forbidden and normal cache-stamp validation remained active. These physical
tests expose attempt counts, not rejection counts:

- Closed-system angular momentum over 150 s passes with 52,733 attempts;
  worst error is `6.33707055476e-13 Nms`, relative `6.027e-14`.
- Desaturation over 60 s passes with 1,045 attempts; wheel speed falls from
  596.9 to 447.566365379 rad/s, hydrazine use is 0.0135 kg, and worst body
  angular speed is `7.21642797521e-4 rad/s`, below the original `2e-3` bound.
- Near-empty exhaustion passes. At one second MMH is zero, NTO is
  `0.0007000000000000004 kg`, impulse is `16.11232595 Ns` and attempts are 70.
  Later windows preserve the exhausted-tank cutoff and impulse.
- The 20-second untrimmed main burn and both subsequent cutoff/coast windows
  pass every original fuel-share, impulse, mass, COM and inertia assertion.
  Attempts at the 20-second endpoint are 152,315; the final total was not
  printed and is not inferred.

The sole batch failure is the full-tank supply readout:
`0.9999999999999999` against exact `1.0`. A separate guarded rerun reproduces
it in 26.76 s at a 1.644 GB peak. Every observed full-tank stage propellant
rate is the exact sign inverse of its demand rate. The discrepancy is in the
new diagnostic expression `-6*Max(-P,(dt/6)*sum(supplied))/(dt*sum(demand))`,
not a measured stage shortage. Its algebraic correction remains unverified.

The separate empty fixture delivers exactly zero impulse, changes no tank
charge and retains zero velocity. Its exact initial-mass comparison exposes
another mismatch: constructor mass `741.8800000000001` versus first native
canonical mass `741.8799999999999`, unchanged thereafter. The constructor
takes the machine-package charged-node reduction, while native mass uses
the separately reduced dry mass plus registered tank charges. Initial mass,
momentum and energy must use the same canonical law. This is a traced
initialization gap, not a verified fix or a relaxed assertion.

## Complete craft command-change gate (before error feedback)

The 27-piece production c0 bank compiled under the existing 6,000,000,000-byte
ceiling. Sampled peak max(working set, private bytes) was 3,041,787,904 bytes;
sampled working-set peak was 1,969,360,896 bytes. The initial runtime was
stopped after compilation to add test-only phase/retry progress output; the
completed archives were preserved and no physics/controller source changed.

The cached rerun of
`tests/test_orbital_library_integration.py::test_production_craft_declares_energy_and_refines_command_changes`
passed in 211.83 s, with 19.445 s construction and 186.260 s phase execution.
Its sampled peak max(working set, private bytes) was 1,644,638,208 bytes.
The five 2-second windows produced these observations:

| Phase | Attempts | Attempts exceeding a declared error limit |
| --- | ---: | ---: |
| Wheel ignition | 1,241 | 1,065 |
| Wheel reversal | 1,394 | 1,201 |
| Main ignition | 8,753 | 7,666 |
| Main cutoff | 8,621 | 7,454 |
| Cold coast | 7,547 | 6,583 |

Total: 27,556 attempts, 23,969 error-limit failures and 3,587 within-limit
attempts over 10 simulated seconds. Inertial angular-momentum drift remains
below `1e-5 Nms` through the wheel phases; electrical minus copper loss minus
mechanical work remains below `1e-10 J`. Tank mass/impulse accounting, exact
no-burn coast, integrated applied delta-v and registered-span identity pass
their unchanged assertions. Final mass is `979.353820972 kg`, propellant
burned `2.52617902765 kg`, electrical energy `903.408316553 J`, copper loss
`900 J`, and maximum global Gram defect `3.33066907388e-15`.

This is physical acceptance of that command sequence, not performance or
whole-game acceptance. Incremental Gram error dominates the retry trace.
The existing default accepted-step proposal floors all error ratios at one
and aims back toward the loose CFL timescale, repeatedly attempting much
larger steps. Adaptive growth remains enabled; no blanket cap or new retry
formula has been added. Long-duration, cutoff, and frame-partition gates are
still pending, as are actual planned game flights.

The existing growth-after-stop gate also passes unchanged (320.81 s including
the four-thruster bank build; sampled memory peak 2,147,876,864 bytes).
Attempts per 10-second spin-up, despin, and subsequent coast window were
`[9772, 25721, 42926, 42933, 25734, 9758, 2, 1, 1]`. Residual angular speed
was `6.30571983518e-16 rad/s`, below its `1e-12` bound. The controller grows
back to full-window steps after the spin stops; high retry cost during spin
is a separate unresolved proposal issue.

Further baseline evidence: the complete 150-second wheel reversal/coast
angular-momentum test passed its unchanged assertions after 192,625 attempts,
with worst drift `3.6959222772e-13 Nms` (relative `3.515e-14`, versus its
`1e-3` bound). Desaturation passed over 60 seconds / 1,216 attempts: wheel
speed `596.9 -> 447.566 rad/s`, hydrazine consumption `0.0135 kg`, and worst
body rate `7.22858432220e-4 rad/s < 2e-3`. The enclosing four-test batch was
subsequently stopped during the ordinary untrimmed burn, whose last printed
point was 5 seconds / 26,702 attempts. That batch has no final pytest summary;
the ordinary burn and queued empty-initial cutoff are not counted as passes.
Its sampled memory peak was 2,502,717,440 bytes. Those unfinished assertions
will run after the controller correction rather than duplicating the costly
old proposal behavior.

A separate near-empty physical cutoff gate passes in 28.80 s on the same
cached c0 bank. It begins with `0.002 kg` MMH and `0.004 kg` NTO. At one
second MMH is exactly zero, NTO is `0.0007000000000000003 kg` and main impulse
is `16.11232595 Ns`. Subsequent windows retain exactly the same tanks,
impulse and mass while the main remains commanded ON; supply is zero. The
existing `1e-12` stoichiometric/impulse and `1e-13` mass tolerances remain.
The first fixture used 396 kg NTO; subtracting nearly equal endpoint charges
lost `2.43e-14 kg`, obscuring a relative ratio of a milligram-scale draw.
Using the still-nonlimiting smaller NTO charge resolves that test-observation
cancellation without changing a law, native layout, or tolerance.

## Remaining full-native coordinator boundary

These flight gates use the existing Python dt coordinator with compiled
physical pieces. They do not prove that the entire coordinator compiles.
The authoritative native entry remains `examples/llvm_dt_system.py`'s
`dt_system_over`, lowered by `lowered_system` through
`lower_ast_source_to_ssa` with `dt_system_contract` and the full-native
execution overlay. No alternate runner or extraction contract is substituted.

The historical conditional-guard/Phi fixes have source-lowering evidence on
the recorded small b/d/e variants, but the last recorded full run still
stopped on the `Metrics.unresolved_report` call-result arena. The base YAML
already declares this mutable token table; the unresolved issue is carrying
its physical ABI through the returned record, not a missing base field.
The old `fv/repro_field_version_loop.py` and `fv/repro_d.py` scratch files are
absent from this checkout. Tracked `tools/repro_step_with_dt_control_used.py`
and `tools/repro_run_superstep.py` lower real helper source with a fake advance,
but do not compile/execute the authoritative full-native entry. Nearby native
scalar-return and record-return tests do not cover this nested table-field
arena end to end. See `CONTINUATION_dt_full_native.md` and
`CONTINUATION_dt_compile_stall.md` for the historical receipts.

A later read-only inventory found the nearest structural coverage at
`tests/test_process_graph_function_linking.py::test_late_returned_record_with_sequence_publishes_scalar_field_to_caller`.
It verifies a returned descriptor's `unresolved_report.sequence_id` and the
caller's scalar `hard_failure` read, but does not execute native table arena
columns or writeback. The current base contract still declares the mutable
rank-one int64-token table. No new full-native lowering was run for this
inventory, so the historical arena failure remains unverified on this head.

## Initial source audit (before the integration correction)

This initial source audit covers `engine_toy/orbital_jumper.py`,
`engine_toy/orbital_craft_machine.py`, and the existing dt contract in
`src/common/dt_system`. It records declared outputs; it does not claim a
compiled runtime measurement.

| Piece | Contract | Declared metric output | dt role |
| --- | --- | --- | --- |
| Craft slew | `BIND` | `dt_limit` from actuator sampling bound | Absolute bound on the next substep |
| Wheels (if installed) | `BIND` | `dt_limit` from next stored wheel momentum and the spin-free inverse-inertia norm; `wheel_energy` is a state accumulator | Gyroscopic stability bound; wheel energy is not a dt exchange metric |
| Momentum | `BIND` | Propellant `dt_limit`; with gravity centers, `energy_j` and `power_w` | Propellant bound; gravity exchange-time sidechain only when centers exist |
| Position / attitude | `BIND` | `max_vel` and attitude `dt_limit` | CFL input and attitude rotation bound |
| Other craft pieces | `BIND` | No `energy_j` / `power_w` pair | Missing exchange-time publication does not introduce `HOLD` |

`declare_binding()` sets each orbital piece's contract to `BIND`. In the
dt-system contract, a `BIND` participant pins only when it publishes an
exchange time; `HOLD` is what caps growth when the participant cannot justify
a larger step. There are two existing publication routes. In generated LLVM
piece publications, a piece with both `power_w` and an energy output schedules
its own exchange time; it prefers `exchangeable_energy_j` and falls back to
`energy_j`. That gives the per-piece bound
`dt <= Targets.energy_exchange_fraction * E_exchangeable / power_w` when
`power_w > 0`. Separately, the legacy/global error-channel sidechain requires
both `energy_j` and `power_w` present, finite, and positive to form
`fraction * energy / power`; absent information does not make a per-piece
exchange publication. A present global nonpositive power holds growth.
`Targets.energy_exchange_fraction` is set to `0.2` for the orbital machine
path, but that setting only acts on published channels. Consequently the
wheel `wheel_energy` accumulator does not feed either route: it has a
different column identity and no `exchangeable_energy_j` / `energy_j` plus
`power_w` publication.

The supported `exchangeable_energy_j` precedent is
`turing/examples/symbolic_chamber_solvers.py::exchangeable_energy`: derive the
energy movable by the law's exchange from its symbolic state derivatives,
then publish it with the exchange power. This is not equivalent to choosing
an arbitrary energy floor or publishing `|tau * omega|` against total
rotational kinetic energy. The chamber helper describes relaxation to an
equilibrium; a pure source with no restoring slope has no finite exchange
time and remains governed by its law's own `dt_limit`.

The machine momentum piece reuses `exchange_publication()`. With gravity
centers it publishes translational kinetic energy and the magnitude of
gravitational power, `|F_gravity . v|`; applied thrust is subtracted from the
gravity-force column for this measure. With no centers it publishes neither
channel. Rotational kinetic energy, applied torque power, and wheel motor
electrical/mechanical power are not published as dt exchange metrics. The
wheel piece does publish its own absolute `dt_limit`, based on next stored
wheel momentum and `||I'^{-1}||_F`; the position piece separately bounds
attitude rotation. These are source-declared bounds, not measured evidence
that the exchange metric captures wheel energy transfer.

The relative-speed saturation candidate is also unmeasured. The wheel torque
law computes its speed-band clamp from current `omega` and `h`, then the
momentum piece updates `omega` in the same substep. Since the rating is
relative speed (`h / I_w - a . omega`), a runtime check must compare the
post-step value against the declared band during a case with large angular
acceleration. This is not established by the declaration-only test below.

To measure it without changing the wheel law, wrap only the instantiated
state's `program["advance_pieces"]` in a test-local observer, retaining the
original callable. On every call, copy the `wheel{w}_momentum` and
`angular_velocity_{x,y,z}` columns, call the original with the same state and
`dt`, then record those columns again, the wheel's
`wheel{w}_torque_command` before the call, the call's `dt`, and
`speed = h / rotor_inertia - axis . omega` before and after. Also record the
wheel participant's published `pub_dt_limit` row. This observes each
substep (including attempts) while preserving its inputs, outputs and order.
Infer delivered motor torque from `(h_after - h_before) / dt`; the wheel law
does not publish a per-wheel delivered-torque column.
Use the existing saturation scenario with the z wheel initialized at 95% of
its momentum limit, then add a real craft torque that drives `omega_z`
strongly during one substep; report `max(abs(speed_after)) / max_speed` and
the exact `dt`, old/new `h`, old/new `omega`, delivered wheel torque and
published bound. A value over 1 is direct evidence of the mismatch. The
wrapper is diagnostic scaffolding only and must not be retained in the law.

Native focused verification used:

```powershell
$env:TURING_GRAPH_BUILD_VERBOSE='0'
$env:PYTHONPATH='C:/dev/Powershell/turing'
python -u -m pytest -s -q tests/test_orbital_craft_machine.py::test_a_slew_on_the_wheels_burns_no_propellant tests/test_orbital_craft_machine.py::test_total_angular_momentum_is_conserved_without_external_torque tests/test_orbital_craft_machine.py::test_saturation_hands_off_to_rcs_desaturation tests/test_orbital_craft_machine.py::test_substeps_follow_the_fast_states
```

Result: 3 failed, 1 passed, 1 existing cffi `imp` deprecation warning in
61.53 s. The saturation/desaturation test passed: z-wheel speed fell
596.9 -> 447.6 rad/s against a 439.8 rad/s dump band, hydrazine use was
0.0135 kg, and worst craft rate was `7.26e-4 rad/s`. The allocator bang-bang
test failed its `1e-6 N m` tolerance: for a requested `+0.8 N m`, the
reported result was `+0.7999984000031999 N m` (error `1.5999968e-6 N m`).
The angular-momentum conservation test measured worst inertial `|H-H0| =
56.23 N m s` (`|H0| = 10.5137 N m s`) over 150 s / 75 substeps; final
`|omega| = 0.2127 rad/s`, wheel momenta `[-18, 18, 24] N m s`. The fast-state
test measured 46 substeps per 2 s round during the final quiet interval,
where it expected one. These are observed test outcomes; no dynamics or
tolerances were changed. The largest observed Python working set was about
2.32 GB, below the 6 GB ceiling.

The declaration-only test run for this audit was
`python -m pytest tests/test_orbital_craft_machine.py::test_wheels_are_declared_rotors_on_their_bodies -q`
from `engine_toy/`. It checks machine declarations, rotor properties, axes,
rigid load paths, and dry mass. It does not construct dt pieces or instantiate
a machine state. The four native wheel tests above instantiate machine states
and compile/load pieces.

Read-only first-substep conservation trace (same runtime columns passed to
the compiled pieces and to their authored SymPy equations): every compared
wheel output (`wheel0..2_momentum_next`, aggregate wheel torque and momentum
on x/y/z, `wheel_energy_next`, `dt_limit`) matched exactly, with maximum
absolute difference 0.0. The compiled momentum-rate outputs x/y/z likewise
matched `euler_rate_tensor_rhs(3)` exactly. Yet the first actual substep was
2 s, changed rotor momentum `[0,0,0]` to `[1.2,-0.8,1.8]` N m s, and changed
craft angular velocity `[.01,-.02,.015]` to
`[.00420584,-.01776181,.01084314]` rad/s. Inertial total angular momentum
error after that one substep was `0.080483321` N m s, equal to the cumulative
error at round end because this round accepted only one substep. This
establishes authored/compiled expression agreement at the sampled point; it
does not attribute the conservation defect to a particular dt integration
semantics or validate that the current authored update is physically
conservative.

## Integration boundary established for the correction

The failing implementation uses the real dt outer coordinator, but its
`leapfrog_momentum`, explicit angular-rate update, and Cayley attitude update
are authored orbital discretizations. It does not invoke the library
integration algorithms. Calling that implementation simply "the dt integrator"
would conflate two different responsibilities.

The existing connection to retain is:

1. Catalogue continuous laws are spelled on craft columns. The actual
   `src.common.dt_system.integrator.integrator.RK4Integrator.step` constructs
   the coupled update over a symbolic state vector. Its stage algebra is
   generic and has no persistent state or independent time loop.
2. `honorary_engine_equation_catalogue.equation_piece` accepts composed and
   discretized catalogue equations. `piece_from_law` compiles those equations
   through SymPy, AbstractTensor source, and the public
   `lower_ast_source_to_ssa` entry with the symbolic identity book into an
   `LLVMPiece`. No orbital copy of the RK stages is needed.
3. `examples.llvm_dt_system.instantiate_system` registers the pieces and
   creates their persistent column spans. `instantiate_state` owns the
   per-state program, participant registry, and instantiated pieces.
   The generated `PieceState.copy_shallow` / `restore` checkpoint and restore
   its physical columns in place.
4. `advance_round(state, requested_window)` calls the bound `dt_system_over`,
   which calls the existing `run_superstep`. This coordinator owns internal
   attempts, rejection, subdivision, and landing the requested outer window.
   The tracker must not reconstruct those substeps from `dt_next`.

This is an implementation boundary, not a claim that the correction has
passed. The coupled native physics and energy gates remain pending until
their measured results are appended below.

History/usage evidence: `engine_toy/dt_benchmark.py` already constructs the
real `Integrator` with compiled force providers, registers identities with
the existing `StateTable`, and runs `GraphBuilder` / `MetaLoopRunner`.
That is a separate supported engine lane; its identity registration does
not automatically bind arbitrary orbital columns to `Integrator._state`.
The orbital design recorded on 2026-10-02 chose the existing LLVM-piece lane.
The current user instruction replaces its bespoke integration while retaining
dt ownership, rather than introducing another runner or state table.

Two configuration facts matter for the energy correction:

- The generic dt channel contract allows an explicitly declared layout that
  appends channels after the shared `DT_CHANNEL_NAMES` prefix. The LLVM-piece
  state, publication, limits, and extraction contract currently fix that
  layout at 15 channels and filter metrics through `METRIC_FIELDS`. A
  conservation discrepancy therefore needs the existing extension contract
  carried consistently through this bridge; stored energy is not an error.
- `step_with_dt_control_used` retains a violating attempt at a configured
  `dt_min` and reports the reasons the floor overruled. The current craft
  configures that floor from its window. The requested scientific behavior
  must use the existing rejection semantics without silently treating a
  floor-retained violation as a passing flight.

### First coupled native slice (not full-craft acceptance)

The actual library RK4 call now constructs four derivative-piece callbacks
and one final physical-state commit. A one-wheel, 13-column construction
completed in 0.886 s; the fully inlined alternative had been stopped after
more than 90 CPU seconds. Both use the library algorithm, but the piece
graph retains stage dependencies without expanding them into one expression.

`engine_toy/tests/test_orbital_library_integration.py` compiled the five
pieces and passed in 44.25 s. A 2-second candidate published energy error
`2.2238e-6 J`, angular-momentum error `1.9367e-4 N m s`, Gram error
`6.9134e-4`, and no overspeed. Dt rejected and restored the registered
columns. Two 2-second windows, including a motor command reversal, landed;
final inertial angular-momentum difference was `4.8713e-10 N m s`, and
state span identities were retained.

This first test used explicit slice limits of `1e-8 J`, `1e-8 N m s`,
`1e-9` for the Gram error, and `1e-8 rad/s` overspeed. They are not adopted
as production tolerances by this report. The test exposed a metric problem:
the Gram publication measured accumulated orthogonality error instead of
the defect of this attempt, causing 3,047 attempts. With the exact factored
incremental Gram defect, the same native gate passed in 56.92 s with 140
attempts over four simulated seconds. Its first 2-second candidate still
fails, both windows land, and final inertial angular-momentum drift is
`2.30317515257e-9 N m s`. Full-craft integration remains unverified.

The first compilation also printed 24 raw-row concordance findings, including
`consumer_operand` (184 rows with no inbound/mint edge) and
`cross_function_references` (124 such rows). These findings have not yet
been baselined. The incremental-metric rerun printed 21 raw-primitive
findings. Successful native execution does not establish edge closure.

### Shared publication bridge

Turing commit `3202f8ba` carries one declared channel layout through the
existing piece state, publication, limits, and native extraction contract.
The native appended-channel proposal checks pass for both aggregate and
participant routes. The compiled-law retry check rejects `dt=0.25` and
`0.125`, restores the same physical span before retry, and completes its
requested `0.25` window with the correct state. The shared-layout native
publication-extent failure reproduced identically on untouched `e068ce59`
at `.dt-layout-baseline`; it was not patched around. The appended and shared
proposal audits both reported zero structural findings but 7,189 unsourced
facts over six identities.

The energy consumer has a separately measured authority conflict. A real
compiled law published stored energy `1 J`, exchangeable energy `64 J`,
power `128 W`, and an explicit participant contract. With exchange fraction
`0.25`, its exchange time is `0.5 s`; nevertheless, the native controller
returned `0.001953125 s` for all four contracts by applying the legacy
aggregate stored-energy bound after the participant publication. With zero
power it likewise imposed the legacy hold on explicitly nonbinding rows.
The contract history promises that DILATE and SUBCYCLE do not bound the
other participants. The correction uses that existing participant authority
and retains the legacy path for metrics without participant rows; it does not
change physical energy or introduce a floor. Turing commit `8ab64810`
contains the correction. Seven focused native/legacy checks passed in
26.90 s. For the same published law, the initial proposal is now `0.5 s`;
the next-step results are BIND `0.125 s`, HOLD `0.25 s`, DILATE `0.5 s`,
and SUBCYCLE `0.5 s`. With zero power, BIND and the nonbinding contracts
permit `0.5 s`, while HOLD retains `0.25 s`. Metrics without participant
rows retain the legacy behavior.

The source-compiled sidechain audit reports zero structural findings both
before and after this correction. Unsourced facts fell from 11,883 over
38 identities to 9,749 over 34 identities. Those remaining facts are open;
this result does not claim complete concordance closure.

### Production integration in progress

The craft and station source assembly now calls the actual RK4 algorithm.
The machine stages share momentum, position, attitude, angular rate, wheel
momentum, propellant, impulse and work state. Forces and mass properties are
evaluated at each stage; physical commit and metric publications follow the
stages. This source change is unverified until full native craft gates pass.
The game currently runs native law pieces under the existing Python dt
coordinator. A small source-compiled native controller gate proves the shared
publication correction; it is not a claim that the whole game coordinator
has been compiled.

The tracker no longer reconstructs substeps from `dt_next`. Its command
pricing uses a native equation for the exact area of the authored bounded
throttle ramp, while actual progress uses integrated craft impulse. The
nine-lane deadband/ramp native test passed, but recompiling the same law for
one lane in the same identity book exposed a batch-shape concordance and
output-extent defect. After the targeted compiler correction, all three
native actuator tests pass in 11.70 s, including arbitrary interval splits
and command reversal. The compiler's same-book 9-to-1-to-9 law checks pass
with exact shape provenance and strict disagreement rejection at the same
call occurrence. Turing commit `31285882` contains the correction. The final
broader gate reports 34 passed and five failures matching the untouched
baseline; no new law name or identity-book reset bypassed the defect.
Root-workspace commit `aadda11` contains the separately verified native ramp
integral and its three tests. The tracker itself remains uncommitted pending
the full flight gates.

Output-only `<column>_next` publications exposed another concrete ownership
gap: `column_names_of` previously collected only piece arguments. Turing
commit `302dcbc3` registers declared write-only columns through that same
existing function. Both focused tests passed in 7.60 s. An observer confirms
the initial readout is restored before the smaller retry, and the accepted
readout persists in the same span. No alternate state or snapshot mechanism
was introduced.

The tracker also previously approximated delivered delta-v from scalar
thruster impulses, midpoint world directions, and mean mass at each frame
cut. The coupled builders now integrate `applied_force / mass` at their
actual library stages into registered delta-v columns, and the tracker reads
their endpoint difference. This removes that host quadrature, but the full
frame-partition flight gate still needs to run.

The first complete production-craft compile was stopped at the documented
roughly 6 GB process ceiling. Its first 19 pieces (properties, inverse,
actuator bound, and all 16 callback groups) finished and cached. The combined
endpoint law reached 4,516 graph nodes and 8,015 edges at
`fold-callsite-structural-values-after-tensors`; the last recorded working
set was 7,164,207,104 bytes. No runtime assertion ran. The ceiling was missed
at the first over-limit observation, then enforced; this is a stopped gate,
not a physics failure or a completed build. Endpoint partitioning is being
reviewed against the existing sequential equation-piece contract before
another native compile. No new whole craft bank is authorized by this result.

Read-only accounting of the 19 completed `.piece` files found 714,433,717
bytes on disk. Loading them individually and traversing their reachable
Python objects gave a summed 6,006,139,090 bytes, including 5,911,052,738
bytes in identity books. These are approximate sums, not an exact allocation
profile of the stopped process. Each law owns a distinct symbolic-program
book; the ordinary in-memory piece cache retains them all. Both the piece's
module and its artifact emission keep provenance alive. Thus partitioning the
endpoint alone does not remove the memory retained by the completed bank.
No books or concordance edges have been discarded to evade this constraint.

The endpoint source now has separate precommit metrics and diagnostics,
one canonical physical commit, then the original mass-property and inverse
laws at distinct postcommit participant occurrences. A source dependency
check found no precommit write intersecting the canonical commit's inputs.
The 27-piece declaration builds in 12.625 seconds; its largest endpoint
group has 1,038 distinct SymPy nodes. This is source evidence only.

The first isolated native check failed: the actual stage-zero law evaluates
to delivered throttle one for command one, old state zero, and infinite
slew, but the native piece returned zero. The finite-slew initial-state
case passed. This check took 40.16 seconds and its sampled peak working set
was 1,487,339,520 bytes. The source-to-native infinity comparison is being
traced before any correction; no physics tolerance was changed.

Turing commit `147dd2be` fixes the traced literal ownership loss. The
symbolic SSA Const retained SymPy `oo`, which Python source materialization
spelled as an unresolved name. In the production artifact this became
unproduced formal 10; its nonfeed buffer was initialized to zero. The
existing numeric normalization and AST literal printer now preserve positive
and negative infinity as `1e309` and `-1e309`. All four native equality and
publication cases pass. The adjacent gate reports 55 passed and six
failures, all reproduced identically on pristine `e068ce59`. The two tiny
equality audits retain zero structural findings, with unsourced facts
1,256 -> 1,254 and 1,263 -> 1,252; their status remains OPEN. Actual
production-piece reruns follow this fix, with normal compiler-stamp
invalidation of older artifacts.

The isolated production fixture also omitted the base `thruster_columns`
initialization for throttle bounds. Its zero-filled maximum correctly
clamped the command to zero, so the production failure alone did not isolate
the infinity defect. The rebuilt production source already contains the
correct `1e309` literal. That fixture is being corrected through the existing
initializer before interpreting the production result. The separate tiny
native reproductions and literal-ownership trace above remain valid.

With both existing initializers supplied, the actual production stage-zero
native check passes in 21.45 seconds: finite-rate old-state evaluation,
instantaneous ignition and cutoff, and instantaneous gimbal commands. The
isolated fixture now zeros only declared scratch and asserts that every
physical input has an explicit value. Full-bank verification is still open.

All three actual production energy-publication pieces pass independently
on the corrected compiler (about 21.3 seconds and 597 MB peak per process):
stored energy, exchange power, and the positive-infinity source bound match.
The actual derivative piece also passes (74.75 seconds, 1,329,561,600 bytes
sampled peak). Its analytical mass-drain case publishes mechanical work
`-1.45 W` and absolute exchange `1.45 W`; inertia flux remains in its exchange
publication. Opposing wheel motors publish net electrical input `2.104 W`,
copper loss `0.104 W`, and absolute electrical exchange `10.04 W`.

The base craft's 10-second pure-torque run exposed an insufficient new
default for the incremental Gram-error channel. At `1e-9`, 86 attempts gave
`max|R^T R-I| = 1.17688078171696e-8`, failing the preserved `1e-14` physical
requirement. With only the existing caller target changed to `1e-17`, the
same native bank and forty 0.25-second outer windows produce
`2.3314683517128287e-15` Gram drift, determinant `0.9999999999999976`, and
angle `2.399999999999658` against the continuum `2.4`. The run took 13.672
seconds including loading, with 7,003 attempts and 5,594 Gram rejections;
sampled peak was 1,395,056,640 bytes. The real controller refined the first
0.25-second proposal through 0.125 to 0.0625 seconds. This measured target
is being adopted as the production default; the physical tolerance remains
unchanged. No attitude projection, replacement integrator, or precision
workaround was used. Rejection cost remains a performance concern.

The complete updated pure-torque pytest subsequently passes at that new
default in 19.82 seconds (1,399,357,440 bytes sampled peak), including all
retained physical bounds and the rotating-thrust integral identity. Only
old exact substep-count/Cayley-formula expectations were removed. The actual
production endpoint-metrics piece also passes its controlled variable-mass
case in 78.82 seconds, with 1,287,823,360 bytes sampled peak. Its raw-row
concordance findings remain OPEN; these numerical passes do not close them.

## Adaptive growth correction

The user explicitly rejected a blanket no-growth policy. The inherited false
default dates to `aeb6b7c8` (2025-08-16, "intigrating dynamic dt controller,
bugs remain"). An additional bridge omission made the orbital plan's choice
ineffective: `instantiate_system`, generated PieceState, its native ABI, and
`dt_system_over` did not carry `allow_increase_mid_round`. The proposed retry
cap correction was stopped before any production edit. No such cap was added.

The existing adaptive policy is now the default in the plan, controller,
graph builder, time request, and bath/cell entry points. The LLVM-piece bridge
carries the declared choice through instantiated, direct, nested, and
subcycle paths. Explicit legacy opt-out and pinned operation remain supported;
physical HOLD, energy bounds, rejection, rollback, and exact window landing
retain their existing meaning. No new proposal or rejection formula is used.

The bounded regression gate passes 31 tests in 20.31 seconds, with sampled
peak max(working set, private bytes) 1,980,862,464 bytes. It includes all nine
cases in `tests/dt_system/test_dt_adaptive_growth.py`, selected controller and
superstep cases, the 50-seed scheduler Monte Carlo gate, graph tests, the
managed event request, LLVM piece/channel tests, and native participant
BIND/HOLD/DILATE/SUBCYCLE checks. The native law demonstrates growth from
0.03125 to 0.21875 within a 0.25 window; the rejection case attempts 0.25,
0.125, 0.0625, 0.1875, restoring registered x to zero for each rejected
attempt and finishing at x = 0.25. The typed bool also crosses the sanctioned
source compiler into actual C execution and determines a written state span.
That ABI proof does not compile the entire coordinator.

The two new audits have zero structural findings and zero unsourced ids,
with 1,472 and 1,680 unsourced facts respectively; concordance debt remains
OPEN. An earlier broader test batch was interrupted without a summary and
is not counted as a completed gate. Full orbital gates are being rerun with
these corrected defaults. Other optional graph-policy transport gaps
(distribution, event boundaries, epsilon, and schedule lattice) remain OPEN.

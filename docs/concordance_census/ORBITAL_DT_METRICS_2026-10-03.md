# Orbital craft dt metric audit — 2026-10-03

This is a source audit of `engine_toy/orbital_jumper.py`,
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
call occurrence. The broader final compiler gate remains pending; no new
law name or identity-book reset bypassed the defect.

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

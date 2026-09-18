"""Per-participant time contracts: tau, and how it negotiates.

These pin the behaviour that lets several coupled simulations step at
different cadences without the most constrained one imposing itself on
everyone, and without a dilating participant accruing catch-up debt.
"""

import math

import pytest

from src.common.dt_system.time_contracts import (
    BIND,
    DILATE,
    HOLD,
    SUBCYCLE,
    ParticipantRegistry,
    StepContracts,
    binding_mask,
    bound_dt,
    crossing_time,
    dilated_state,
    holds_dt,
)


def _registry(*names):
    registry = ParticipantRegistry()
    for name in names:
        registry.declare(name)
    return registry


@pytest.mark.dt
@pytest.mark.fast
def test_ids_are_monotonic_and_stable():
    registry = _registry("air", "pool")
    assert registry.declare("air") == 0
    assert registry.declare("pool") == 1
    assert registry.declare("air") == 0  # redeclaring never renumbers
    assert registry.name_of(1) == "pool"


@pytest.mark.dt
@pytest.mark.fast
def test_silence_is_hold_with_no_tau_not_a_zero_tau():
    """The distinction the dense-dict representation could not carry.

    A participant that published nothing is not claiming tau = 0; a zero
    tau would pin dt to zero and stop the simulation."""

    registry = _registry("air", "pool")
    contracts = StepContracts.of(registry, {"air": (0.01, BIND)})

    assert [bool(x) for x in contracts.tau_present.tolist()] == [True, False]
    assert [int(x) for x in contracts.contract.tolist()] == [BIND, HOLD]
    assert [bool(x) for x in binding_mask(contracts).tolist()] == [True, False]


@pytest.mark.dt
@pytest.mark.fast
def test_binding_tau_pins_the_step():
    """dt <= fraction * tau -- the same law as the energy/power pin, but
    per participant."""

    registry = _registry("air")
    contracts = StepContracts.of(registry, {"air": (0.01, BIND)})
    # fraction 0.1, tau 0.01 -> 1 ms, below the 5 ms proposed.
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(1.0e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_the_stiffest_binding_participant_wins_not_a_blend():
    """The failure the summed fold produced: a small stiff simulation must
    not be averaged away inside a large slow one."""

    registry = _registry("slow_big", "fast_small")
    contracts = StepContracts.of(
        registry, {"slow_big": (1.0, BIND), "fast_small": (1.0e-4, BIND)}
    )
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(1.0e-5)


@pytest.mark.dt
@pytest.mark.fast
def test_dilating_participant_does_not_pin_anyone():
    """The whole point: an inconsequential participant keeps its own
    accurate cadence without dragging the system to it."""

    registry = _registry("solver", "waiting_reactor")
    contracts = StepContracts.of(
        registry,
        {
            "solver": (0.01, BIND),
            # A far stiffer tau -- but dilating, so it must not bound.
            "waiting_reactor": (1.0e-9, DILATE),
        },
    )
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(1.0e-3)
    assert [bool(x) for x in binding_mask(contracts).tolist()] == [True, False]


@pytest.mark.dt
@pytest.mark.fast
def test_subcycling_participant_does_not_pin_either():
    registry = _registry("solver", "substepper")
    contracts = StepContracts.of(
        registry, {"solver": (0.01, BIND), "substepper": (1.0e-9, SUBCYCLE)}
    )
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(1.0e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_hold_prevents_growth_without_shrinking():
    """``power_w == 0`` used to mean this by arithmetic accident; it is now
    stated. Nobody claimed a larger step is unsafe -- they claimed not to
    know, so dt is held at what it already was rather than reduced."""

    registry = _registry("quiet")
    contracts = StepContracts.of(registry, {"quiet": (None, HOLD)})
    assert holds_dt(contracts) is True
    # Proposed growth to 5 ms is held back to the current 2 ms...
    assert bound_dt(contracts, 0.1, 5.0e-3, dt_current=2.0e-3) == pytest.approx(2.0e-3)
    # ...and with no current step to hold to, nothing is invented.
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(5.0e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_no_binding_participant_leaves_the_step_alone():
    registry = _registry("a", "b")
    contracts = StepContracts.of(
        registry, {"a": (1.0e-9, DILATE), "b": (1.0e-9, SUBCYCLE)}
    )
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(5.0e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_dilated_state_is_well_relaxation():
    """tau is the well's relaxation time -- the same quantity BIND uses to
    bound, used here to evaluate."""

    # One tau of relaxation closes 1 - 1/e of the gap to equilibrium.
    assert dilated_state(x0=1.0, x_eq=0.0, tau_s=2.0, elapsed_s=2.0) == pytest.approx(
        math.exp(-1.0)
    )
    # Arbitrarily long elapsed intervals are evaluated, never replayed:
    # this is why DILATE accrues no catch-up debt.
    assert dilated_state(1.0, 0.0, 2.0, 1.0e6) == pytest.approx(0.0, abs=1e-12)
    assert dilated_state(5.0, 5.0, 2.0, 7.0) == pytest.approx(5.0)


@pytest.mark.dt
@pytest.mark.fast
def test_crossing_time_answers_when_a_condition_is_met():
    """A dilating reactor can be left alone safely because the coordinator
    can ask WHEN its barrier is reached instead of stepping finely enough
    not to miss it."""

    # Relaxing 1 -> 0 with tau = 2; half way is at t = 2 ln 2.
    t = crossing_time(x0=1.0, x_eq=0.0, tau_s=2.0, barrier=0.5)
    assert t == pytest.approx(2.0 * math.log(2.0))
    # And the closed form agrees with itself at that instant.
    assert dilated_state(1.0, 0.0, 2.0, t) == pytest.approx(0.5)


@pytest.mark.dt
@pytest.mark.fast
def test_unreachable_barrier_is_none_not_a_bogus_time():
    """A barrier past the well minimum is never reached by relaxation, so
    the coordinator is free to step over the whole span rather than
    rendezvous with a crossing that cannot happen."""

    # Relaxing 1 -> 0 never reaches -1, and never returns to 2.
    assert crossing_time(1.0, 0.0, 2.0, -1.0) is None
    assert crossing_time(1.0, 0.0, 2.0, 2.0) is None


@pytest.mark.dt
@pytest.mark.fast
def test_reproduces_the_existing_energy_power_law():
    """Equivalence with ``test_energy_power_channels_pin_the_next_step``.

    That test publishes energy 10 J and power 1000 W with
    ``energy_exchange_fraction = 0.1`` and requires
    ``dt_next <= 0.1 * 10 / 1000``.  tau IS energy/power, so the same
    numbers through the contract give the same bound -- the law is
    unchanged, only where tau comes from and who it applies to."""

    energy_j, power_w, fraction = 10.0, 1000.0, 0.1
    tau = energy_j / power_w

    registry = _registry("core")
    contracts = StepContracts.of(registry, {"core": (tau, BIND)})
    assert bound_dt(contracts, fraction, 5.0e-3) <= fraction * energy_j / power_w + 1e-15


@pytest.mark.dt
@pytest.mark.fast
def test_concerns_and_contracts_compose_into_the_coordinator_decision():
    """The whole first step, end to end.

    A binding solver sets the pace.  A dilating reactor watches inputs it
    cares about; while those inputs are still it is neither invalidated nor
    allowed to pin, so it keeps its own accurate cadence for free.  The
    step its inputs finally move, it comes back as invalidated and must
    re-bind."""

    from src.common.dt_system.state_concerns import (
        ConcernUnion,
        changed_mask,
        invalidated,
    )

    union = ConcernUnion.of([("velocity", "pressure"), ("heat", "reagent")])
    masks = {
        "solver": union.mask(("velocity", "pressure")),
        "reactor": union.mask(("heat", "reagent")),
    }
    registry = _registry("solver", "reactor")
    contracts = StepContracts.of(
        registry, {"solver": (0.01, BIND), "reactor": (1.0e-9, DILATE)}
    )

    # The reactor's tau is a thousand times stiffer, and pins nothing.
    assert bound_dt(contracts, 0.1, 5.0e-3) == pytest.approx(1.0e-3)

    quiet = {n: [0.0] for n in union.names}
    solver_moved = dict(quiet, velocity=[1.0])
    assert invalidated(changed_mask(union, quiet, solver_moved), masks) == ("solver",)

    fed = dict(solver_moved, heat=[1.0])
    assert invalidated(changed_mask(union, quiet, fed), masks) == ("solver", "reactor")


@pytest.mark.dt
@pytest.mark.fast
def test_dilation_requires_a_positive_tau():
    """Closed-form evaluation is meaningless without a relaxation time, and
    silently returning the starting state would hide a broken participant."""

    with pytest.raises(ValueError):
        dilated_state(1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        crossing_time(1.0, 0.0, 0.0, 0.5)

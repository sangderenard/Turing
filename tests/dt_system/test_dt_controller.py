import copy
import pytest

from src.common.dt_system.dt_scaler import Metrics
from src.cells.bath.dt_controller import (
    STController,
    Targets,
    step_with_dt_control,
    step_with_dt_control_used,
)


class DummyState:
    def __init__(self):
        self.mass = 1.0
        self.vel = 0.1

    def copy_shallow(self):
        return copy.deepcopy(self)

    def restore(self, saved):
        self.__dict__.update(copy.deepcopy(saved.__dict__))

    def step(self, dt):
        # constant velocity, mass stays the same
        pass

    def total_mass(self):
        return self.mass

    def compute_metrics(self, prev_mass):
        return Metrics(
            max_vel=self.vel,
            max_flux=self.vel,
            div_inf=0.0,
            mass_err=abs(self.mass - prev_mass) / prev_mass,
        )


@pytest.mark.dt
@pytest.mark.fast
def test_dt_controller_step():
    state = DummyState()
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=1e-6, dt_max=1e-2)
    dx = 0.1

    def advance(state, dt):
        prev_mass = getattr(state, "_last_mass", state.total_mass())
        state.step(dt)
        metrics = state.compute_metrics(prev_mass)
        state._last_mass = state.total_mass()
        return True, metrics

    metrics, dt_next = step_with_dt_control(state, 1e-4, dx, targets, ctrl, advance)
    assert dt_next > 0
    assert isinstance(metrics.max_vel, float)


@pytest.mark.dt
@pytest.mark.fast
def test_dt_controller_no_clamps():
    ctrl = STController(dt_min=None, dt_max=None)
    dt_small = ctrl.pi_update(dt_prev=1e-8, dt_pen=1e-9, osc=False)
    assert dt_small < 1e-6
    ctrl2 = STController(dt_min=1e-6, dt_max=None)
    dt_large = ctrl2.pi_update(dt_prev=1.0, dt_pen=10.0, osc=False)
    assert dt_large > 1.0


@pytest.mark.dt
@pytest.mark.fast
def test_no_dt_min_no_max_retries_still_terminates_deterministically():
    """max_retries=None with no dt_min floor used to have no bound at all.

    A caller that always rejects has no numeric retry budget (max_retries is
    explicitly None, exactly ``balloon_tire_managed_window``'s own call
    shape) and no physical floor (dt_min is None).  Before the recursion-to-
    loop conversion and the numeric-underflow exhaustion check, this
    recursed until Python's own call-stack limit turned it into an
    uncontrolled RecursionError -- never a clean, named result, and never
    even reaching the true float64 floor (~1074 halvings) that a caller
    asking for unlimited patience is entitled to.  It must now return a
    deterministic hard failure once halving a float64 value genuinely stops
    changing it -- the true mathematical floor, not an earlier
    approximation of it -- reached in well under 2000 ``advance`` calls.
    """

    state = DummyState()
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=None, dt_max=None)
    dx = 0.1
    calls = {"count": 0}

    def always_rejects(state, dt):
        calls["count"] += 1
        return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)

    metrics, dt_next, dt_used = step_with_dt_control_used(
        state, 1e-3, dx, targets, ctrl, always_rejects,
        max_retries=None, allow_unresolved=False,
    )

    assert calls["count"] < 2000
    assert bool(metrics.hard_failure) is True
    assert float(dt_used) == 0.0


@pytest.mark.dt
@pytest.mark.fast
def test_no_dt_min_no_max_retries_honours_allow_unresolved():
    """The same pathological config, but the caller opted into acceptance.

    Before the fix, ``allow_unresolved=True`` combined with
    ``max_retries=None`` was a dead branch: ``retries_exhausted`` could
    never become True, so this flag had no effect and the search still ran
    away.  The numeric-underflow exhaustion condition now makes the
    caller's explicit "accept an unresolved step eventually" reachable.
    """

    state = DummyState()
    targets = Targets(cfl=0.5, div_max=1e-3, mass_max=1e-6)
    ctrl = STController(dt_min=None, dt_max=None)
    dx = 0.1
    calls = {"count": 0}

    def always_rejects(state, dt):
        calls["count"] += 1
        return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)

    metrics, dt_next, dt_used = step_with_dt_control_used(
        state, 1e-3, dx, targets, ctrl, always_rejects,
        max_retries=None, allow_unresolved=True,
    )

    assert calls["count"] < 2000
    assert bool(metrics.hard_failure) is True
    # dt itself has genuinely underflowed to 0.0 by the time this fires --
    # that IS the true float64 floor, not a leftover nonzero remainder --
    # so the attempt count, not dt's own value, is the meaningful signal
    # that the exhaustion path actually ran.
    assert float(metrics.error_channels.get("dt_unresolved_attempts", 0.0)) > 0.0


def _metrics_with(energy, power):
    return Metrics(
        max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0,
        error_channels={"energy_j": energy, "power_w": power},
    )


@pytest.mark.dt
@pytest.mark.fast
def test_energy_power_channels_pin_the_next_step():
    """dt_next <= fraction * energy / power: a time measured from the core."""

    from src.cells.bath.dt_controller import step_with_dt_control_used

    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0, energy_exchange_fraction=0.1)
    ctrl = STController(dt_min=None, dt_max=None)

    def advance(state, dt):
        return True, _metrics_with(energy=10.0, power=1000.0)   # tau = 0.01 s

    for rollback in (True, False):
        metrics, dt_next, dt_used = step_with_dt_control_used(
            DummyState(), 5.0e-3, 0.1, targets, ctrl, advance, rollback=rollback)
        assert float(dt_used) == 5.0e-3
        assert float(dt_next) <= 0.1 * 10.0 / 1000.0 + 1.0e-15, (rollback, float(dt_next))


@pytest.mark.dt
@pytest.mark.fast
def test_no_energy_exchange_holds_dt_instead_of_growing():
    """A step that moved no energy is no evidence a larger step is safe."""

    from src.cells.bath.dt_controller import step_with_dt_control_used

    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0, energy_exchange_fraction=0.1)
    ctrl = STController(dt_min=None, dt_max=None)

    def advance(state, dt):
        return True, _metrics_with(energy=10.0, power=0.0)

    metrics, dt_next, dt_used = step_with_dt_control_used(
        DummyState(), 2.0e-3, 0.1, targets, ctrl, advance)
    assert float(dt_next) <= 2.0e-3 + 1.0e-15

    # Without the energy contract the same zero-velocity step doubles.
    plain = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    metrics, dt_grown, dt_used = step_with_dt_control_used(
        DummyState(), 2.0e-3, 0.1, plain, STController(dt_min=None, dt_max=None), advance)
    assert float(dt_grown) > 2.0e-3


@pytest.mark.dt
@pytest.mark.fast
def test_dt_limit_hint_bounds_the_first_attempt_of_a_round():
    """The opener of a round never exceeds the core's declared safe step."""

    from src.cells.bath.dt_controller import run_superstep

    class DeclaredState(DummyState):
        def dt_limit_hint(self):
            return 1.0e-4

    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    ctrl = STController(dt_min=None, dt_max=None)
    log: list[dict] = []

    def advance(state, dt):
        return True, Metrics(max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0)

    total, dt_next, metrics = run_superstep(
        DeclaredState(), 1.0e-3, 1.0e-2, 0.1, targets, ctrl, advance,
        attempt_log=log, max_iters=200)
    assert log and log[0]["dt"] <= 1.0e-4 + 1.0e-18
    assert float(total) >= 1.0e-3 - 1.0e-15

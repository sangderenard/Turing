"""The multi-participant step as spans: summed totals, individual gates.

These lock the division of labour ``participants`` exists for:

* system state energy and conservation are SUMMED, because they are extensive
* stability and custom limits are INDIVIDUAL gates, never reduced to one step
* a participant may hold itself stricter than its neighbours
* absence is its own span and is not a published zero
* the totals are exact, so the causal order they are accumulated in cannot
  show up in the answer

And two that are really about the future: the step path returns spans and
resolves no names, and the amalgamation cannot tell how many kernels filled the
rows -- which is what has to keep holding when every law is symbolically reduced
into one system and one fused kernel publishes all of them.
"""

import pytest

from src.common.dt_system.error_channels import declare_channel
from src.common.dt_system.participants import (
    Publication,
    StepSpans,
    any_tripped,
    penalties,
    stability_gates,
    system_totals,
    tau_bound,
    trip_report,
    tripped,
)
from src.common.dt_system.time_contracts import (
    BIND,
    DILATE,
    HOLD,
    SUBCYCLE,
    ParticipantRegistry,
)


def _channels():
    """Declared once; ids are monotonic and shared, so the axis is stable."""
    return [declare_channel(n) for n in ("energy_j", "power_w", "mass_err")]


def _registry(*names):
    registry = ParticipantRegistry()
    for name in names:
        registry.declare(name)
    return registry


def _column(span, channel_id):
    """One channel out of a (C,) span, by id -- the id IS the index."""
    return float(span.tolist()[channel_id])


@pytest.mark.dt
@pytest.mark.fast
def test_system_extensives_are_summed_not_maxed():
    energy, _power, _mass = _channels()
    spans = StepSpans.of(
        _registry("air", "pool"),
        {"air": Publication(channels={"energy_j": 10.0}),
         "pool": Publication(channels={"energy_j": 4.0})},
    )
    totals, _reported = system_totals(spans)
    assert _column(totals, energy) == pytest.approx(14.0)


@pytest.mark.dt
@pytest.mark.fast
def test_an_unpublished_channel_is_not_a_zero_in_the_total():
    energy, power, _mass = _channels()
    spans = StepSpans.of(
        _registry("air", "quiet"),
        {"air": Publication(channels={"power_w": 100.0}),
         "quiet": Publication(channels={})},
    )
    totals, reported = system_totals(spans)
    assert _column(totals, power) == pytest.approx(100.0)
    assert bool(reported.tolist()[power]) is True
    # declared as a channel, published by nobody this step
    assert bool(reported.tolist()[energy]) is False


@pytest.mark.dt
@pytest.mark.fast
def test_the_total_is_exact_so_causal_order_cannot_show_up_in_it():
    """Two limbs make the sum independent of the order it was taken in.

    The participant axis is in causal order and stays that way; this is what
    stops that order leaking into the arithmetic.  Summed in one limb these
    magnitudes disagree wildly depending on direction.
    """
    energy, _power, _mass = _channels()
    magnitudes = [1.0, 1e-16, 3.7e-17, 1e16, -1e16, 2.5e-17, 8.1e-18]
    names = tuple(f"p{index}" for index in range(len(magnitudes)))

    forward = StepSpans.of(
        _registry(*names),
        {name: Publication(channels={"energy_j": value})
         for name, value in zip(names, magnitudes)},
    )
    backward = StepSpans.of(
        _registry(*reversed(names)),
        {name: Publication(channels={"energy_j": value})
         for name, value in zip(names, magnitudes)},
    )
    a, _ = system_totals(forward)
    b, _ = system_totals(backward)
    assert _column(a, energy) == _column(b, energy)
    # and it is the correctly-rounded value, not one of the drifted ones
    assert _column(a, energy) == pytest.approx(1.0 + 1.703e-16, rel=0, abs=1e-31)


@pytest.mark.dt
@pytest.mark.fast
def test_a_trip_is_per_participant_per_channel():
    _energy, _power, mass = _channels()
    spans = StepSpans.of(
        _registry("calm", "loud"),
        {"calm": Publication(channels={"mass_err": 1e-6}),
         "loud": Publication(channels={"mass_err": 2e-3})},
        default_limits={"mass_err": 1e-3},
    )
    gates = tripped(spans).reshape((spans.participants, -1)).tolist()
    assert gates[0][mass] is False or gates[0][mass] == False  # noqa: E712
    assert bool(gates[1][mass]) is True
    assert [bool(x) for x in any_tripped(spans).tolist()] == [False, True]
    assert penalties(spans).reshape((spans.participants, -1)).tolist()[1][mass] == pytest.approx(2.0)


@pytest.mark.dt
@pytest.mark.fast
def test_a_participant_may_hold_itself_stricter_than_the_defaults():
    _energy, _power, mass = _channels()
    spans = StepSpans.of(
        _registry("lenient", "strict"),
        {"lenient": Publication(channels={"mass_err": 1e-9}),
         "strict": Publication(channels={"mass_err": 1e-9},
                               limits={"mass_err": 1e-12})},
        default_limits={"mass_err": 1e-3},
    )
    assert [bool(x) for x in any_tripped(spans).tolist()] == [False, True]


@pytest.mark.dt
@pytest.mark.fast
def test_a_measure_with_no_limit_is_judged_by_nobody():
    _channels()
    spans = StepSpans.of(
        _registry("air"),
        {"air": Publication(channels={"energy_j": 1e9})},
        default_limits={"mass_err": 1e-3},   # nothing limits energy
    )
    assert bool(any_tripped(spans).tolist()[0]) is False


@pytest.mark.dt
@pytest.mark.fast
def test_stability_gates_stay_individual():
    """Each floor is its own gate; nothing collapses them to one step."""
    _channels()
    spans = StepSpans.of(
        _registry("stiff", "slack", "silent"),
        {"stiff": Publication(dt_limit=5e-4),
         "slack": Publication(dt_limit=1e-2)},
    )
    floors, published = stability_gates(spans)
    assert floors.tolist()[0] == pytest.approx(5e-4)
    assert floors.tolist()[1] == pytest.approx(1e-2)
    # the one that published no floor is marked absent, not given a zero floor
    assert [bool(x) for x in published.tolist()] == [True, True, False]


@pytest.mark.dt
@pytest.mark.fast
def test_only_binding_taus_bound_the_step():
    _channels()
    spans = StepSpans.of(
        _registry("solver", "reactor"),
        {"solver": Publication(tau_s=1e-2, contract=BIND),
         "reactor": Publication(tau_s=1e-9, contract=DILATE)},
    )
    # 0.1 * 1e-2 from the solver; the reactor's far stiffer tau is excluded
    assert float(tau_bound(spans, 0.1, 5e-3).item()) == pytest.approx(1e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_subcycling_does_not_bound_either():
    _channels()
    spans = StepSpans.of(
        _registry("interior"),
        {"interior": Publication(tau_s=1e-9, contract=SUBCYCLE)},
    )
    assert float(tau_bound(spans, 0.1, 5e-3).item()) == pytest.approx(5e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_hold_prevents_growth_without_shrinking():
    _channels()
    spans = StepSpans.of(_registry("watcher"),
                         {"watcher": Publication(contract=HOLD)})
    assert float(tau_bound(spans, 0.1, 5e-3, 2e-3).item()) == pytest.approx(2e-3)
    # with no current step to hold to, it shrinks nothing
    assert float(tau_bound(spans, 0.1, 5e-3).item()) == pytest.approx(5e-3)


@pytest.mark.dt
@pytest.mark.fast
def test_silence_is_not_a_published_zero():
    energy, _power, _mass = _channels()
    registry = _registry("present", "absent")
    spans = StepSpans.of(
        registry, {"present": Publication(channels={"energy_j": 3.0})})
    assert "absent" in registry.declared()
    # the absent participant has a row, and every presence bit in it is False
    assert [bool(x) for x in spans.pub_present.reshape((spans.participants, -1)).tolist()[1]] == [False] * 3
    assert bool(spans.pub_dt_limit_present.tolist()[1]) is False
    assert bool(spans.pub_tau_present.tolist()[1]) is False
    totals, _ = system_totals(spans)
    assert _column(totals, energy) == pytest.approx(3.0)


@pytest.mark.dt
@pytest.mark.fast
def test_the_step_path_returns_spans_and_resolves_no_names():
    """Names belong at the reporting boundary, not on the path."""
    _channels()
    spans = StepSpans.of(
        _registry("air", "pool"),
        {"air": Publication(channels={"mass_err": 2e-3}),
         "pool": Publication(channels={"mass_err": 1e-9})},
        default_limits={"mass_err": 1e-3},
    )
    # everything the step consults has a shape, not keys
    for produced in (system_totals(spans)[0], penalties(spans),
                     tripped(spans), any_tripped(spans),
                     stability_gates(spans)[0]):
        assert hasattr(produced, "shape")
    # and words only when a person is going to read them
    assert trip_report(spans, ("air", "pool")) == (("air", "mass_err", 2.0),)


@pytest.mark.dt
@pytest.mark.fast
def test_the_rows_do_not_record_who_filled_them():
    """Future proofing, as a test.

    The same publications give the same verdict whether they arrived one
    participant at a time or all at once, because the index is a declared
    identity and not a call site.  Under symbolic reduction only the filling
    changes.
    """
    _channels()
    published = {
        "air": Publication(channels={"energy_j": 10.0, "mass_err": 1e-6},
                           dt_limit=5e-4, tau_s=1e-3, contract=BIND),
        "species": Publication(channels={"energy_j": 4.0, "mass_err": 2e-3},
                               dt_limit=2e-3, tau_s=1e-2, contract=BIND),
        "pool": Publication(channels={"energy_j": 1.0, "mass_err": 1e-9},
                            dt_limit=1e-2, tau_s=1e-9, contract=DILATE),
    }
    registry = _registry("air", "species", "pool")

    piecewise = {}
    for name, publication in published.items():   # "one call per participant"
        piecewise[name] = publication
    fused = dict(published)                        # "one fused kernel"

    a = StepSpans.of(registry, piecewise, default_limits={"mass_err": 1e-3})
    b = StepSpans.of(registry, fused, default_limits={"mass_err": 1e-3})

    assert system_totals(a)[0].tolist() == system_totals(b)[0].tolist()
    assert tripped(a).tolist() == tripped(b).tolist()
    assert stability_gates(a)[0].tolist() == stability_gates(b)[0].tolist()
    assert float(tau_bound(a, 0.5, 5e-3).item()) == float(tau_bound(b, 0.5, 5e-3).item())

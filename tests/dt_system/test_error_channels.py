"""Error channels as id-indexed spans.

These pin the behaviour the controller's penalty/soft/rollback bands depend
on, in the representation that replaces the string-keyed dict: dense ids,
no hashing on the step path, and absence carried as a mask rather than as a
zero.
"""

import pytest

from src.common.dt_system import error_channels as ec


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test gets its own registry; ids are process-global otherwise."""
    monkeypatch.setattr(ec, "_CHANNEL_NAMES", [])
    monkeypatch.setattr(ec, "_CHANNEL_BY_NAME", {})


@pytest.mark.dt
@pytest.mark.fast
def test_ids_are_dense_monotonic_and_stable():
    assert ec.declare_channel("height_positivity") == 0
    assert ec.declare_channel("tracer_bounds") == 1
    assert ec.declare_channel("height_positivity") == 0  # never renumbered
    assert ec.declared_channels() == ("height_positivity", "tracer_bounds")
    assert ec.channel_name(1) == "tracer_bounds"


@pytest.mark.dt
@pytest.mark.fast
def test_ids_stay_exact_in_a_float_column():
    """The reason for dense ids over string tokens.

    An fnv1a-64 token does not survive float64's 53-bit mantissa, so it
    stops matching its own lookup.  A dense id is exact in any numeric
    column, which is what removes the int64-vs-float64 question entirely
    rather than resolving it."""

    for name in (f"channel_{index}" for index in range(500)):
        identity = ec.declare_channel(name)
        assert int(float(identity)) == identity
        assert identity < 2 ** 53


@pytest.mark.dt
@pytest.mark.fast
def test_absence_is_a_mask_not_a_zero():
    """``.get(name, 0.0)`` silently turned "not published" into a measure of
    zero.  These are different claims and are now representable as such."""

    ec.declare_channel("published")
    ec.declare_channel("silent")
    spans = ec.ChannelSpans.of({"published": 0.0})

    assert spans.has(0) is True
    assert spans.value(0) == 0.0     # published, and genuinely zero
    assert spans.has(1) is False
    assert spans.value(1) is None    # not published at all


@pytest.mark.dt
@pytest.mark.fast
def test_only_channels_both_published_and_limited_are_judged():
    measured_only = ec.declare_channel("measured_only")
    limited_only = ec.declare_channel("limited_only")
    both = ec.declare_channel("both")

    spans = ec.ChannelSpans.of({"measured_only": 5.0, "both": 2.0})
    limits = ec.ChannelLimits.of({"limited_only": 1.0, "both": 4.0})

    judged = [bool(x) for x in ec.judged_mask(spans, limits).tolist()]
    assert judged[measured_only] is False  # nobody asked for it to be judged
    assert judged[limited_only] is False   # the step never reported on it
    assert judged[both] is True


@pytest.mark.dt
@pytest.mark.fast
def test_penalty_is_measure_over_limit():
    ec.declare_channel("residual")
    spans = ec.ChannelSpans.of({"residual": 3.0})
    limits = ec.ChannelLimits.of({"residual": 1.5})
    assert ec.penalties(spans, limits).tolist()[0] == pytest.approx(2.0)


@pytest.mark.dt
@pytest.mark.fast
def test_exceeded_reports_names_at_the_boundary():
    """Ids on the path, words where a person reads them."""

    ec.declare_channel("ok_channel")
    ec.declare_channel("blown_channel")
    spans = ec.ChannelSpans.of({"ok_channel": 0.5, "blown_channel": 9.0})
    limits = ec.ChannelLimits.of({"ok_channel": 1.0, "blown_channel": 1.0})
    assert ec.exceeded(spans, limits) == ("blown_channel",)


@pytest.mark.dt
@pytest.mark.fast
def test_a_new_channel_does_not_disturb_existing_consumers():
    """Plug and play: a simulator declares its own channel and publishes it;
    consumers that never heard of it are unaffected, and the ids already
    handed out never move."""

    first = ec.declare_channel("original")
    limits = ec.ChannelLimits.of({"original": 1.0})
    before = ec.penalties(ec.ChannelSpans.of({"original": 0.5}), limits).tolist()[first]

    newcomer = ec.declare_channel("late_arrival")
    assert newcomer != first and ec.declare_channel("original") == first

    after_limits = ec.ChannelLimits.of({"original": 1.0})
    after = ec.penalties(
        ec.ChannelSpans.of({"original": 0.5, "late_arrival": 99.0}),
        after_limits,
    ).tolist()[first]
    assert after == pytest.approx(before)
    # The newcomer is unjudged: it published a measure but nobody limits it.
    assert ec.exceeded(
        ec.ChannelSpans.of({"original": 0.5, "late_arrival": 99.0}),
        after_limits,
    ) == ()


@pytest.mark.dt
@pytest.mark.fast
def test_a_zero_limit_does_not_divide_by_zero():
    """``max(float(limit), 1e-30)`` guarded this in the old generator; the
    vectorized form must not reintroduce an infinity."""

    ec.declare_channel("degenerate")
    spans = ec.ChannelSpans.of({"degenerate": 1.0})
    limits = ec.ChannelLimits.of({"degenerate": 0.0})
    ratio = ec.penalties(spans, limits).tolist()[0]
    assert ratio == ratio  # not NaN
    assert abs(float(ratio)) != float("inf")

"""The controller's own record, separated from engine measurements."""

import pytest

from src.common.dt_system.control_diagnostics import ControlDiagnostics


@pytest.mark.dt
@pytest.mark.fast
def test_absent_is_distinct_from_zero():
    """``dt_unresolved = 0.0`` means the search resolved; ``None`` means the
    question never arose.  ``.get(name, 0.0)`` could not tell them apart."""

    resolved = ControlDiagnostics(dt_unresolved=0.0)
    never_asked = ControlDiagnostics()

    assert resolved.recorded() == {"dt_unresolved": 0.0}
    assert never_asked.recorded() == {}


@pytest.mark.dt
@pytest.mark.fast
def test_updates_replace_rather_than_mutate_shared_state():
    """The defect this removes: the controller used to copy an engine's dict,
    mutate it, reassign it, and then write one more key afterwards -- which
    landed only because both names aliased the same dict."""

    before = ControlDiagnostics(dt_unresolved=1.0)
    after = before.with_(dt_unresolved_attempts=3)

    assert before.dt_unresolved_attempts is None  # untouched
    assert after.dt_unresolved_attempts == 3
    assert after.dt_unresolved == 1.0             # carried forward


@pytest.mark.dt
@pytest.mark.fast
def test_unknown_diagnostic_is_refused():
    """A typo'd key silently created a new channel in the dict, changing
    ``error_channels.length`` -- a quantity the compiled ABI measures."""

    with pytest.raises(TypeError):
        ControlDiagnostics().with_(dt_unresvoled=1.0)


@pytest.mark.dt
@pytest.mark.fast
def test_reasons_accumulate_as_stable_identities():
    diagnostics = (
        ControlDiagnostics()
        .note_soft("mass_err")
        .note_soft("div_inf")
        .note_rollback("div_inf")
    )
    assert diagnostics.soft_reasons == ("mass_err", "div_inf")
    assert diagnostics.rollback_reasons == ("div_inf",)
    assert "soft_reasons" in diagnostics.recorded()


@pytest.mark.dt
@pytest.mark.fast
def test_empty_reason_lists_are_not_reported():
    assert "soft_reasons" not in ControlDiagnostics().recorded()


@pytest.mark.dt
@pytest.mark.fast
def test_superstep_window_fields_round_trip():
    diagnostics = ControlDiagnostics().with_(
        superstep_window_requested_s=1.0,
        superstep_window_advanced_s=0.25,
        superstep_window_remaining_s=0.75,
        superstep_iteration_count=4,
        superstep_iteration_cap_hit=False,
    )
    recorded = diagnostics.recorded()
    assert recorded["superstep_window_remaining_s"] == 0.75
    # False is a recorded answer, not an absence.
    assert recorded["superstep_iteration_cap_hit"] is False

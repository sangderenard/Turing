"""Contracts for separating device capture from interaction focus."""

from src.compiler.abstract_ui_control_focus import ControlFocusPolicy


def test_focus_policy_routes_one_capture_without_conflating_authority():
    model = ControlFocusPolicy("focus", "actor").to_data()
    assert model["routes"] == ["game", "projected-pointer", "dialogue"]
    assert model["switch_action"] == "secondary-action"
    assert model["dialogue"] == {
        "priority": 100, "return_rule": "resume-previous", "response_required": True,
    }
    assert model["projected_pointer"]["source"] == "captured-pointer-motion"
    assert model["projected_pointer"]["destination"] == "document-coordinate-space"

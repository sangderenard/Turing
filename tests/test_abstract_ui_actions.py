"""System timer and action-edge recency contract tests."""

import pytest

from src.compiler.abstract_ui_actions import (
    ActionEdgeTable,
    IssuedAction,
    SystemTimer,
    system_action_mezzanine_model,
)


def _wired():
    table = ActionEdgeTable("system/entities/action-edges", "system/timer")
    table = table.register(
        identity="edge:inspect-room",
        source="room:packet",
        type="inspect",
        destination="room:packet",
    )
    timer = SystemTimer("system/timer").connect(table.identity)
    return timer, table


def test_system_timer_connects_to_action_edges_with_update_actions():
    model = system_action_mezzanine_model("world")
    timer = model["timer"]
    table = model["action_edges"]
    assert timer["connections"] == [table["identity"]]
    assert model["edges"] == [{
        "source": "world/timer",
        "target": "world/entities/action-edges",
        "relationship": "update(actions)",
    }]
    assert table["operation"] == "update(actions)"


def test_issued_action_lights_registered_row_and_increments_count():
    timer, table = _wired()
    action = IssuedAction(
        "action:1", "pointer.primary", "inspect", "room:packet",
        "edge:inspect-room", 10.0,
    )
    timer, table = timer.tick(10.1, actions=(action,), action_edges=table)
    assert timer.sequence == 1
    assert table.rows[0].issue_count == 1
    assert table.rows[0].last_issued_at == 10.0
    assert table.rows[0].recent


def test_timer_update_extinguishes_row_after_recent_window():
    timer, table = _wired()
    action = IssuedAction(
        "action:1", "pointer.primary", "inspect", "room:packet",
        "edge:inspect-room", 10.0,
    )
    timer, table = timer.tick(10.1, actions=(action,), action_edges=table)
    timer, table = timer.tick(11.0, actions=(), action_edges=table)
    assert not table.rows[0].recent
    assert table.rows[0].issue_count == 1


def test_actions_must_name_a_registered_edge_and_connected_table():
    timer, table = _wired()
    unknown = IssuedAction(
        "action:bad", "pointer.primary", "inspect", "room:missing",
        "edge:missing", 1.0,
    )
    with pytest.raises(KeyError, match="unregistered edges"):
        timer.tick(1.0, actions=(unknown,), action_edges=table)
    with pytest.raises(ValueError, match="not connected"):
        SystemTimer("system/timer").tick(1.0, actions=(), action_edges=table)

from __future__ import annotations

import math

import pytest

from src.common.parametric_l_system import ParametricLSystem


def test_hilbert_preset_exposes_every_generation_and_square_geometry():
    system = ParametricLSystem.preset("hilbert")

    generations = tuple(system.iter_derivation(2))
    trace = system.interpret(generations[-1])

    assert generations[0] == "A"
    assert generations[1] == "+BF-AFA-FB+"
    assert generations[-1].count("F") == 15
    assert len(trace.segments) == 15
    assert trace.bounds == pytest.approx((0.0, 0.0, 3.0, 3.0), abs=1e-12)


def test_context_sensitive_rule_has_priority_and_can_ignore_turtle_symbols():
    system = ParametricLSystem(
        "A+[B]-C",
        {
            "B": "ordinary",
            ("A", "B", "C"): "context",
        },
    )

    assert system.derive(1) == "A+[context]-C"


def test_callable_and_weighted_rules_are_seeded_and_parameterized():
    def grow(context):
        if context.generation >= context.parameters["branch_after"]:
            return {"F[+F]": 2.0, "F[-F]": 1.0}
        return "FF"

    first = ParametricLSystem(
        "F", {"F": grow}, parameters={"branch_after": 1}, seed=8128
    )
    second = ParametricLSystem(
        "F", {"F": grow}, parameters={"branch_after": 1}, seed=8128
    )

    assert first.derive(5) == second.derive(5)
    assert "[" in first.derive(5)


def test_branches_restore_all_mutable_turtle_parameters():
    system = ParametricLSystem(
        "F[>W+F]F",
        {},
        actions={">": "step_up", "W": "width_up"},
        angle_degrees=90,
        step_scale=2,
        width_scale=3,
    )

    trace = system.trace(0)

    assert len(trace.segments) == 3
    trunk, branch, resumed = trace.segments
    assert trunk.end == pytest.approx((1.0, 0.0))
    assert branch.end == pytest.approx((1.0, 2.0))
    assert branch.width == 3.0
    assert resumed.start == pytest.approx((1.0, 0.0))
    assert resumed.end == pytest.approx((2.0, 0.0))
    assert resumed.width == 1.0
    assert trace.maximum_branch_depth == 1


def test_callable_action_can_compose_commands_from_parameters():
    system = ParametricLSystem(
        "AAA",
        {},
        parameters={"draw": True},
        actions={
            "A": lambda context: (
                ("draw", "left") if context.parameters["draw"] else "move"
            )
        },
        angle_degrees=120,
    )

    trace = system.trace(0)

    assert len(trace.segments) == 3
    assert math.dist(trace.final_state.position, (0.0, 0.0)) < 1e-12


def test_expansion_and_geometry_budgets_fail_before_runaway_growth():
    system = ParametricLSystem("F", {"F": "FFFF"}, max_symbols=16)
    assert len(system.derive(2)) == 16
    with pytest.raises(OverflowError, match="max_symbols=16"):
        system.derive(3)

    geometry_limited = ParametricLSystem(
        "FFF", {}, max_segments=2
    )
    with pytest.raises(OverflowError, match="max_segments=2"):
        geometry_limited.trace(0)


def test_strict_branch_validation_reports_both_failure_modes():
    system = ParametricLSystem("F", {})

    with pytest.raises(ValueError, match="unmatched branch pop"):
        system.interpret("F]")
    with pytest.raises(ValueError, match="unclosed branch"):
        system.interpret("F[F")


def test_jitter_and_svg_output_are_reproducible():
    system = ParametricLSystem(
        "F+F+F", {}, angle_jitter_degrees=3, length_jitter=0.1, seed="demo"
    )

    first = system.trace(0)
    second = system.trace(0)

    assert first == second
    assert first.svg_path(precision=3) == second.svg_path(precision=3)
    assert first.svg_path().count("M ") == 3


@pytest.mark.parametrize(
    ("name", "segments"),
    [
        ("dragon", 2),
        ("koch", 12),
        ("sierpinski", 9),
        ("plant", 3),
    ],
)
def test_classic_presets_produce_expected_first_generation_edges(name, segments):
    assert len(ParametricLSystem.preset(name).trace(1).segments) == segments

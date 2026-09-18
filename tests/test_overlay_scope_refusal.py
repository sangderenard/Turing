"""``overlay_scheduled_control`` must refuse a cross-scope split up front,
naming it, instead of composing a duplicated tree that a downstream
duplicate-region-count guard fails on with no context.

Frontier task D (see ``frontier_tasks/TASK_D_overlay_embed_scope_refusal.md``):
re's ``_compile`` schedules its if/elif cascade's markers across several
sequence scopes because its owning loop is blocked from composing. This
reproduces the shape structurally, without running the real compiler.
"""
import pytest
from types import SimpleNamespace

from src.compiler.control_source import (
    ConditionalBlock,
    ControlOverlayScopeError,
    ControlProgram,
    LoopBlock,
    SequenceBlock,
    StatementBlock,
    overlay_scheduled_control,
)
from src.compiler.glsl_deployment_strategy import (
    CompilationSubdivisionRequired,
    _overlay_control_or_require_subdivision,
)


def _marker(index: int) -> StatementBlock:
    return StatementBlock((f"__scheduled_region_{index}__",))


def _count_markers(block, counts: dict[int, int]) -> None:
    """Mirror fortran_c_shell's own marker_counts walk, for the healthy case."""

    if isinstance(block, StatementBlock):
        for line in block.lines:
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                index = int(line[len("__scheduled_region_"):-2])
                counts[index] = counts.get(index, 0) + 1
        return
    if isinstance(block, SequenceBlock):
        for child in block.blocks:
            _count_markers(child, counts)
        return
    if isinstance(block, LoopBlock):
        _count_markers(block.body, counts)
        return
    if isinstance(block, ConditionalBlock):
        _count_markers(block.body, counts)
        if block.orelse is not None:
            _count_markers(block.orelse, counts)
        return


def test_conditional_wholly_inside_one_loop_scope_composes_cleanly():
    """Healthy case: a conditional's markers live entirely inside one loop
    body. Composition succeeds and every marker appears exactly once."""

    outer = ControlProgram(
        LoopBlock(
            "k", "0", "n", "1",
            SequenceBlock((_marker(0), _marker(1))),
        ),
        region_indices=(0, 1, 2),
    )
    conditional = ControlProgram(
        SequenceBlock((
            ConditionalBlock(
                predicate_value_id=7,
                body=_marker(1),
                orelse=_marker(2),
                source_node_id=3,
            ),
        )),
        region_indices=(1, 2),
    )

    overlaid = overlay_scheduled_control((0, 1, 2), (outer, conditional))

    counts: dict[int, int] = {}
    _count_markers(overlaid.root, counts)
    assert counts == {0: 1, 1: 1, 2: 1}


def test_conditional_split_across_top_level_and_loop_body_refuses_named():
    """Defect case: one conditional's markers straddle the flat top level
    and a loop body -- the exact shape a blocked main loop produces in
    re's _compile. Must refuse before composing, naming both scopes.

    ``outer`` owns region 2 in addition to the conditional's 0/1 so the
    containment is a STRICT subset (equal region sets hit the unrelated,
    pre-existing "maximal control blocks overlap" guard instead -- that
    guard fires before nesting is even considered and is not this defect)."""

    outer = ControlProgram(
        SequenceBlock((
            _marker(0),
            LoopBlock("k", "0", "n", "1", _marker(1)),
            _marker(2),
        )),
        region_indices=(0, 1, 2),
    )
    conditional = ControlProgram(
        SequenceBlock((
            ConditionalBlock(
                predicate_value_id=7,
                body=_marker(0),
                orelse=_marker(1),
                source_node_id=3,
            ),
        )),
        region_indices=(0, 1),
    )

    with pytest.raises(ValueError, match="sequence scopes"):
        overlay_scheduled_control((0, 1, 2), (outer, conditional))


def test_conditional_split_names_the_loop_and_top_level_scope_paths():
    """The refusal message must be actionable: name each distinct scope
    and which region indices live there."""

    outer = ControlProgram(
        SequenceBlock((
            _marker(0),
            LoopBlock("k", "0", "n", "1", _marker(1)),
            _marker(2),
        )),
        region_indices=(0, 1, 2),
    )
    conditional = ControlProgram(
        SequenceBlock((
            ConditionalBlock(
                predicate_value_id=7,
                body=_marker(0),
                orelse=_marker(1),
                source_node_id=3,
            ),
        )),
        region_indices=(0, 1),
    )

    with pytest.raises(ControlOverlayScopeError) as excinfo:
        overlay_scheduled_control((0, 1, 2), (outer, conditional))

    message = str(excinfo.value)
    assert "top" in message
    assert "loop(k)" in message
    assert "regions [0]" in message
    assert "regions [1]" in message
    assert excinfo.value.control_index == 1
    assert excinfo.value.region_indices == (0, 1)


def test_cross_scope_refusal_becomes_an_exact_loop_subdivision_boundary():
    outer = ControlProgram(
        SequenceBlock((
            _marker(0),
            LoopBlock("k", "0", "n", "1", _marker(1)),
            _marker(2),
        )),
        region_indices=(0, 1, 2),
    )
    conditional = ControlProgram(
        ConditionalBlock(
            predicate_value_id=7,
            body=_marker(0),
            orelse=_marker(1),
            source_node_id=3,
        ),
        region_indices=(0, 1),
    )
    graph = SimpleNamespace(G=SimpleNamespace(graph={}))
    reduction = SimpleNamespace(loop_node_id=41)

    with pytest.raises(CompilationSubdivisionRequired) as excinfo:
        _overlay_control_or_require_subdivision(
            graph,
            (0, 1, 2),
            (reduction,),
            (outer,),
            (conditional,),
            {},
        )

    boundary, = excinfo.value.to_failure_mapping()[
        "subdivision_boundaries"
    ]
    assert boundary["kind"] == "loop-control-owner"
    assert boundary["loop_node_id"] == 41
    assert boundary["region_indices"] == [0, 1, 2]
    assert boundary["blockers"][0] == "control-overlay-sequence-scope"


def test_known_nesting_still_composes_two_conditionals_in_one_scope():
    """Nesting hints must keep working: two conditionals whose region sets
    are equal (one wholly nested in the other) are not a scope split."""

    inner = ControlProgram(
        LoopBlock("iteration_2", "0", "4", "1", _marker(32)),
        region_indices=(32,),
    )
    outer = ControlProgram(
        LoopBlock("iteration_1", "0", "4", "1", _marker(32)),
        region_indices=(32,),
    )

    overlaid = overlay_scheduled_control(
        (32,), (outer, inner), known_nesting={0: (1,)},
    )

    assert isinstance(overlaid.root, SequenceBlock)
    outer_root = overlaid.root.blocks[0]
    assert isinstance(outer_root, LoopBlock)
    assert isinstance(outer_root.body, LoopBlock)
    assert outer_root.body.induction == "iteration_2"

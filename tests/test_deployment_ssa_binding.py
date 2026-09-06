"""Deploy/Join dataflow binding over emitted-shape SSA streams.

The synthetic functions here mirror the exact stream
``test_callblock_evaporates_and_parallel_lanes_linearize_without_fake_call``
proves the emitter produces: ``[Deploy, Call(lane0), Call(lane1), Join,
Ret]`` in one block, lanes distinguished only by
``deployment_memberships``.
"""

from __future__ import annotations

from src.compiler.deployment_lowering import legalize_deployments_serial
from src.compiler.deployment_ssa_binding import (
    DEPLOY_TOKEN_DTYPE,
    analyze_deployment_dataflow,
    bind_deployment_dataflow,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def _marker(op: str, region_id: int) -> Instr:
    return Instr(op, [], None, attributes={
        "deployment_frame": True,
        "region_id": region_id,
        "scale": 1,
        "join_mode": "barrier",
        "reduction_operator": None,
        "allow_reassociation": False,
        "schedule_preference": "alap",
    })


def _lane_call(result_id: int, region_id: int, lane: int, args=()) -> Instr:
    return Instr(
        "Call",
        list(args),
        SSAValue(result_id, dtype="f64"),
        attributes={
            "callee": f"numerical_region_{lane}",
            "deployment_memberships": ((region_id, lane),),
        },
    )


def _parallel_function(*, cross_lane_use: bool = False) -> Function:
    lane0 = _lane_call(10, 7, 0)
    lane1 = _lane_call(
        11, 7, 1, args=[lane0.res] if cross_lane_use else [],
    )
    ret = Instr("Ret", [lane0.res, lane1.res], None)
    entry = BasicBlock("entry", [
        _marker("Deploy", 7), lane0, lane1, _marker("Join", 7), ret,
    ])
    return Function("root", [], {"entry": entry})


def test_analysis_reports_lanes_and_live_outs():
    region, = analyze_deployment_dataflow(_parallel_function())
    assert region.region_id == 7
    assert region.has_markers
    assert region.independent
    assert region.deploy_site == ("entry", 0)
    assert region.join_site == ("entry", 3)
    assert [lane.lane_index for lane in region.lanes] == [0, 1]
    assert region.lanes[0].live_out_value_ids == (10,)
    assert region.lanes[1].live_out_value_ids == (11,)


def test_cross_lane_use_is_a_reported_violation_and_blocks_binding():
    function = _parallel_function(cross_lane_use=True)
    region, = analyze_deployment_dataflow(function)
    assert not region.independent
    assert any("not independent" in item for item in region.violations)

    report = bind_deployment_dataflow(function)
    assert report.bound_region_ids == ()
    assert not report.complete
    ((region_id, reason),) = report.skipped
    assert region_id == 7
    assert reason.startswith("independence-violation")
    deploy = function.blocks["entry"].instrs[0]
    assert deploy.res is None  # untouched


def test_binding_gives_deploy_a_token_and_join_its_operands():
    function = _parallel_function()
    report = bind_deployment_dataflow(function)
    assert report.bound_region_ids == (7,)
    assert report.complete

    deploy = function.blocks["entry"].instrs[0]
    join = function.blocks["entry"].instrs[3]
    assert deploy.res is not None
    assert deploy.res.dtype == DEPLOY_TOKEN_DTYPE
    assert join.args[0] is deploy.res
    assert [value.id for value in join.args[1:]] == [10, 11]
    assert join.arg_roles == ["frame", "lane0.out", "lane1.out"]
    assert join.attributes["lane_live_outs"] == ((0, 10), (1, 11))
    # The token id is fresh, colliding with nothing in the stream.
    assert deploy.res.id not in {10, 11}


def test_binding_is_idempotent():
    function = _parallel_function()
    bind_deployment_dataflow(function)
    second = bind_deployment_dataflow(function)
    assert second.bound_region_ids == ()
    assert second.skipped == ((7, "already-bound"),)


def test_marker_less_regions_are_reported_not_bound():
    # dream_document's shape: memberships without any Deploy/Join pair.
    call = _lane_call(20, 3, 0)
    function = Function("dream_main", [], {
        "entry": BasicBlock("entry", [call, Instr("Ret", [call.res], None)]),
    })
    report = bind_deployment_dataflow(function)
    ((region_id, reason),) = report.skipped
    assert region_id == 3
    assert reason.startswith("missing-markers")


def test_serial_legalization_removes_markers_and_keeps_lanes():
    function = _parallel_function()
    bind_deployment_dataflow(function)
    report = legalize_deployments_serial(function)
    assert report.removed_markers == 2
    ops = [instr.op for instr in function.blocks["entry"].instrs]
    assert ops == ["Call", "Call", "Ret"]
    # Provenance survives legalization.
    assert function.blocks["entry"].instrs[0].attributes[
        "deployment_memberships"
    ] == ((7, 0),)

"""Repository deployment records plan real native deployment frames."""

from __future__ import annotations

from src.compiler.repository_ssa_dispatch import (
    plan_repository_ssa_dispatch,
)
from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSADeploymentLane,
    SSADeploymentRegion,
)


def _module() -> IRModule:
    def marker(op: str) -> Instr:
        return Instr(op, [], None, attributes={
            "deployment_frame": True,
            "region_id": 4,
        })

    calls = [
        Instr("Call", [], None, attributes={
            "callee": name,
            "deployment_memberships": ((4, lane),),
        })
        for lane, name in enumerate(("lane_a", "lane_b"))
    ]
    root = Function("root", [], {
        "entry": BasicBlock("entry", [
            marker("Deploy"), *calls, marker("Join"), Instr("Ret", [], None),
        ]),
    })
    # lane_a has a trivial internal closure. Its helper must remain part of
    # the deployable lane instead of causing a serial fallback.
    lane_a = Function("lane_a", [], {
        "entry": BasicBlock("entry", [
            Instr("Call", [], None, attributes={"callee": "helper"}),
            Instr("Ret", [], None),
        ]),
    })
    lane_b = Function("lane_b", [], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    helper = Function("helper", [], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    region = SSADeploymentRegion(
        region_id=4,
        function="root",
        kind="parallel",
        schedule="parallel",
        lanes=(
            SSADeploymentLane(0, (("entry", 1),), ("lane_a",)),
            SSADeploymentLane(1, (("entry", 2),), ("lane_b",)),
        ),
    )
    return IRModule(
        {"root": root, "lane_a": lane_a, "lane_b": lane_b, "helper": helper},
        deployment_table={"root": (region,)},
    )


def test_planner_consumes_frames_and_follows_internal_closures():
    plan = plan_repository_ssa_dispatch(_module(), backend="llvm", cores=4)
    frame, = plan.frames
    assert frame.parallel
    assert frame.choice.strategy == "pool"
    assert frame.lanes[0].closure == ("lane_a", "helper")
    assert frame.lanes[1].closure == ("lane_b",)


def _iteration_module(*, outlined: bool) -> IRModule:
    lane_callees = ("lane_template",) if outlined else ()
    root = Function("loop_root", [], {
        "entry": BasicBlock("entry", [
            Instr("Deploy", [], None, attributes={
                "deployment_frame": True, "region_id": 0,
            }),
            Instr("Call", [], None, attributes={
                "callee": "lane_template",
                "deployment_memberships": ((0, 0),),
            }),
            Instr("Join", [], None, attributes={
                "deployment_frame": True, "region_id": 0,
            }),
            Instr("Ret", [], None),
        ]),
    })
    template = Function("lane_template", [], {
        "entry": BasicBlock("entry", [Instr("Ret", [], None)]),
    })
    region = SSADeploymentRegion(
        region_id=0,
        function="loop_root",
        kind="parallel_candidate",
        schedule="independent_iterations",
        lanes=(SSADeploymentLane(0, (("entry", 1),), lane_callees),),
        iteration_space=("0", "8", "1"),
        origin="retained_loop",
    )
    module = IRModule(
        {"loop_root": root, "lane_template": template},
        deployment_table={"loop_root": (region,)},
    )
    if outlined:
        module.metadata["deployment_outlines"] = {
            ("loop_root", 0): object(),
        }
    return module


def test_outlined_iteration_region_is_launchable_with_one_lane():
    plan = plan_repository_ssa_dispatch(
        _iteration_module(outlined=True), backend="c", cores=8,
    )
    frame, = plan.frames
    assert frame.launchable
    assert frame.parallel
    assert frame.lanes[0].roots == ("lane_template",)


def test_unoutlined_iteration_region_names_the_outlining_pass():
    plan = plan_repository_ssa_dispatch(
        _iteration_module(outlined=False), backend="c", cores=8,
    )
    frame, = plan.frames
    assert not frame.launchable
    assert any(
        "outline_independent_iteration_lanes" in item
        for item in frame.shortfalls
    )


"""Two SSA deployment lanes access the same memory location.

This is a deliberately unsynchronized lowering probe.  It does not assert a
particular result for the two concurrent read/modify/write operations.
"""

from src.transmogrifier.ssa import (
    BasicBlock,
    Function,
    IRModule,
    Instr,
    SSADeploymentLane,
    SSADeploymentRegion,
    SSAValue,
)


def shared_location_module() -> IRModule:
    shared = SSAValue(0, "int64", shape=(1,))

    def lane(name: str) -> Function:
        buffer = SSAValue(0, "int64", shape=(1,))
        index = SSAValue(1, "int64")
        address = SSAValue(2, "ptr")
        old = SSAValue(3, "int64")
        one = SSAValue(4, "int64")
        updated = SSAValue(5, "int64")
        return Function(name, [buffer], {
            "entry": BasicBlock("entry", [
                Instr("Const", [], index, attributes={"value": 0}),
                Instr("GetElementPtr", [buffer, index], address),
                Instr("Load", [address], old),
                Instr("Const", [], one, attributes={"value": 1}),
                Instr("Add", [old, one], updated),
                Instr("Store", [updated, address], None),
                Instr("Ret", [], None),
            ]),
        })

    def marker(op: str) -> Instr:
        return Instr(op, [], None, attributes={
            "deployment_frame": True,
            "region_id": 0,
        })

    root = Function("shared_location_root", [shared], {
        "entry": BasicBlock("entry", [
            marker("Deploy"),
            Instr("Call", [shared], None, attributes={
                "callee": "lane_a",
                "deployment_memberships": ((0, 0),),
            }),
            Instr("Call", [shared], None, attributes={
                "callee": "lane_b",
                "deployment_memberships": ((0, 1),),
            }),
            marker("Join"),
            Instr("Ret", [], None),
        ]),
    })
    region = SSADeploymentRegion(
        region_id=0,
        function=root.name,
        kind="parallel",
        schedule="parallel",
        lanes=(
            SSADeploymentLane(0, (("entry", 1),), ("lane_a",)),
            SSADeploymentLane(1, (("entry", 2),), ("lane_b",)),
        ),
    )
    return IRModule(
        {root.name: root, "lane_a": lane("lane_a"), "lane_b": lane("lane_b")},
        deployment_table={root.name: (region,)},
    )


def test_both_lanes_access_the_same_root_buffer_element():
    module = shared_location_module()
    calls = module.functions["shared_location_root"].blocks["entry"].instrs[1:3]
    assert calls[0].args[0] is calls[1].args[0]
    assert [instruction.op for instruction in module.functions["lane_a"].blocks["entry"].instrs] == [
        "Const", "GetElementPtr", "Load", "Const", "Add", "Store", "Ret",
    ]
    assert [instruction.op for instruction in module.functions["lane_b"].blocks["entry"].instrs] == [
        "Const", "GetElementPtr", "Load", "Const", "Add", "Store", "Ret",
    ]

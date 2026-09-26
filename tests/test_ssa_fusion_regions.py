from src.compiler.ssa_fusion_regions import discover_contiguous_ssa_regions
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def test_fusion_candidates_exist_only_between_repository_ssa_barriers():
    a = SSAValue(0, "float64")
    b = SSAValue(1, "float64")
    x = SSAValue(2, "float64")
    y = SSAValue(3, "float64")
    z = SSAValue(4, "float64")
    function = Function(
        "controlled",
        [a, b],
        {
            "entry": BasicBlock("entry", [
                Instr("Add", [a, b], x),
                Instr("Mul", [x, b], y),
                Instr("Call", [y], z, attributes={"callee": "opaque"}),
                Instr("Sub", [y, a], z),
                Instr("Ret", [z], None),
            ])
        },
    )

    regions = discover_contiguous_ssa_regions(
        function, {"Add", "Sub", "Mul"},
    )

    assert [region.operations for region in regions] == [
        ("Add", "Mul"),
        ("Sub",),
    ]

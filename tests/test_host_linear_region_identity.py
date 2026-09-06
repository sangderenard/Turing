from src.compiler.ir_identities import inline_host_linear_source_regions
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def test_single_callsite_outputless_source_region_inlines_for_host_view_only():
    formal, produced = SSAValue(1, "float64"), SSAValue(20, "float64")
    actual = SSAValue(2, "float64")
    region = Function(
        "owner__planned_region_0",
        [formal],
        {"entry": BasicBlock("entry", [Instr("Neg", [formal], produced)])},
        metadata={"source_region_integral": {
            "owner": "owner",
            "output_value_ids": (),
            "identity_token_chain": ("source-region", "owner", "region_0"),
        }},
    )
    caller = Function(
        "owner",
        [actual],
        {"loop_body": BasicBlock("loop_body", [Instr(
            "Call", [actual], None,
            attributes={"callee": region.name, "result_convention": "ssa.aggregate"},
        )])},
    )
    original = {caller.name: caller, region.name: region}

    view, receipts = inline_host_linear_source_regions(original)

    assert region.name not in view
    inlined = view[caller.name].blocks["loop_body"].instrs
    assert [(item.op, [arg.id for arg in item.args], item.res.id)
            for item in inlined] == [("Neg", [actual.id], produced.id)]
    assert inlined[0].attributes["inlined_source_region"] == region.name
    assert receipts[0]["instruction_count"] == 1
    assert original[caller.name].blocks["loop_body"].instrs[0].op == "Call"
    assert region.name in original


def test_region_with_multiple_callsites_stays_a_call():
    formal = SSAValue(1, "float64")
    region = Function(
        "region",
        [formal],
        {"entry": BasicBlock("entry", [Instr(
            "Neg", [formal], SSAValue(10, "float64")
        )])},
        metadata={"source_region_integral": {
            "output_value_ids": (),
            "identity_token_chain": ("source-region", "region"),
        }},
    )
    caller = Function("owner", [formal], {
        "entry": BasicBlock("entry", [
            Instr("Call", [formal], None, attributes={"callee": "region"}),
            Instr("Call", [formal], None, attributes={"callee": "region"}),
        ])
    })

    view, receipts = inline_host_linear_source_regions({
        "owner": caller, "region": region,
    })

    assert receipts == ()
    assert "region" in view
    assert [item.op for item in view["owner"].blocks["entry"].instrs] == [
        "Call", "Call",
    ]


def test_fortran_host_view_splices_a_real_planned_loop_body():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_fortran_backend import emit_module

    module, outputs, _exports = lower_ast_source_to_ssa(
        "def scale(x, y, n):\n"
        "    for i in range(n):\n"
        "        y[i] = x[i] * 2.0\n"
        "    return y\n",
        "scale",
    )
    artifact = emit_module(
        module, name="host_inline_test", outputs=outputs,
        progress=lambda _message: None,
    )

    receipts = artifact.api.metadata["host_linear_region_inlining"]
    assert receipts[0]["instruction_count"] >= 4
    assert "__planned_region_" not in artifact.source
    assert artifact.complete

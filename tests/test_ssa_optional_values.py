import inspect
from pathlib import Path

from src.common.tensors import AbstractTensor
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
from src.common.dt_system import dt_controller
from src.common.dt_system.dt_scaler import _scalar
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_to_c
from src.compiler.ssa_optional_values import lower_optional_scalar_returns
from src.compiler.ssa_self_check import check_formal_parity, check_optional_merges
from src.compiler.vehicle_python_compilation import (
    balloon_tire_managed_extraction_contract,
    balloon_tire_managed_python_compilation_inputs,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


def test_scalar_optional_return_and_linked_none_test_split_into_payload_presence():
    absent = SSAValue(1, "none")
    value = SSAValue(2, "float64")
    merged = SSAValue(3)
    callee = Function("maybe", [], {
        "entry": BasicBlock("entry", [Instr("CondBr", [SSAValue(0, "bool")], None)]),
        "present": BasicBlock("present", [Instr("Br", [], None)]),
        "absent": BasicBlock("absent", [Instr("NoneValue", [], absent), Instr("Br", [], None)]),
        "exit": BasicBlock("exit", [
            Instr("Phi", [value, absent], merged, attributes={
                "incoming_blocks": ("present", "absent"),
                "binding": "return_merge",
            }),
            Instr("Ret", [merged], None),
        ]),
    }, metadata={"semantic_output_ids": (3,)})
    caller_result = SSAValue(10)
    none = SSAValue(11, "none")
    compared = SSAValue(12, "bool")
    caller = Function("root", [], {"entry": BasicBlock("entry", [
        Instr("Call", [], caller_result, attributes={"callee": "maybe", "source_linked": True}),
        Instr("NoneValue", [], none),
        Instr("Ne", [caller_result, none], compared),
        Instr("Ret", [compared], None),
    ])})
    module = IRModule({"maybe": callee, "root": caller})

    receipts = lower_optional_scalar_returns(module)

    assert receipts
    assert check_optional_merges(module) == []
    returned = callee.blocks["exit"].instrs[-1].args
    assert [value.dtype for value in returned] == ["float64", "bool"]
    call = caller.blocks["entry"].instrs[0]
    assert call.res.dtype == "ssa.aggregate"
    assert call.attributes["result_convention"] == "ssa.aggregate"
    assert len(call.attributes["native_result_contract"]) == 2
    comparison = next(i for i in caller.blocks["entry"].instrs if i.res is compared)
    assert comparison.op == "Cast"
    assert comparison.args[0].accounting["ssa_optional_presence"] is True
    assert module.metadata["optional_scalar_return_receipts"][-1][
        "tie_policy"
    ] == "incumbent"
    assert lower_optional_scalar_returns(module) == ()


def test_nested_optional_phi_presence_flows_into_return_presence():
    value = SSAValue(1, "float64")
    first_none = SSAValue(2, "none")
    nested = SSAValue(3)
    second_none = SSAValue(4, "none")
    returned = SSAValue(5)
    function = Function("nested_maybe", [], {
        "inner_present": BasicBlock("inner_present", [Instr("Br", [], None)]),
        "inner_absent": BasicBlock("inner_absent", [
            Instr("NoneValue", [], first_none), Instr("Br", [], None),
        ]),
        "inner_merge": BasicBlock("inner_merge", [
            Instr("Phi", [value, first_none], nested, attributes={
                "incoming_blocks": ("inner_present", "inner_absent"),
                "binding": "conditional_result",
            }),
            Instr("Br", [], None),
        ]),
        "outer_absent": BasicBlock("outer_absent", [
            Instr("NoneValue", [], second_none), Instr("Br", [], None),
        ]),
        "exit": BasicBlock("exit", [
            Instr("Phi", [nested, second_none], returned, attributes={
                "incoming_blocks": ("inner_merge", "outer_absent"),
                "binding": "return_merge",
            }),
            Instr("Ret", [returned], None),
        ]),
    })
    module = IRModule({function.name: function})

    lower_optional_scalar_returns(module)

    assert check_optional_merges(module) == []
    inner_presence = function.blocks["inner_merge"].instrs[1].res
    outer_presence = function.blocks["exit"].instrs[1]
    assert outer_presence.args[0] is inner_presence
    assert [value.dtype for value in function.blocks["exit"].instrs[-1].args] == [
        "float64", "bool",
    ]


def test_energy_limit_exact_helper_has_closed_optional_and_structural_abi():
    prepared = balloon_tire_managed_python_compilation_inputs(1)
    strict = balloon_tire_managed_extraction_contract(
        prepared.feeds["material"]
    )
    contract = ExtractionContract(
        Path("extraction_contracts/program_extraction.yaml")
    ).with_program_abi(strict.program_abi.receipt())
    source = "\n\n".join((
        "import math",
        "from src.common.tensors import AbstractTensor",
        "from src.common.dt_system.error_channels import ENERGY_J, POWER_W",
        inspect.getsource(_scalar),
        inspect.getsource(dt_controller._energy_time_limit),
        "def root(metrics, targets):\n"
        "    limit, present = _energy_time_limit(metrics, targets)\n"
        "    return limit if present else -1.0\n",
    ))
    resolved = []

    module, _outputs, _exports = lower_ast_source_to_ssa(
        source,
        "root",
        name="test_energy_limit_optional",
        extraction_contract=contract,
        resolved_process_graph_sink=resolved.append,
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )

    assert check_formal_parity(module) == []
    assert check_optional_merges(module) == []
    helper = next(
        function for name, function in module.functions.items()
        if name.endswith("___energy_time_limit")
    )
    assert any(
        (argument.accounting or {}).get("program_abi_optional_payload")
        for argument in helper.args
    )
    assert not any(
        instruction.attributes.get("ssa_sequence_operation") in ("contains", "table_load")
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    energy_graph = next(
        entry.graph.G for entry in resolved[0].function_table
        if entry.qualified_name == "_energy_time_limit"
    )
    receipt = next(
        receipt for receipt in energy_graph.graph[
            "optional_record_presence_receipts"
        ]
        if receipt["field"] == "energy_exchange_fraction"
    )
    assert receipt["test_count"] == 1
    root_name = next(
        name for name in module.functions if name.endswith("__root")
    )
    emitted = emit_ssa_to_c(module, root_name)
    assert emitted.complete, emitted.shortfalls

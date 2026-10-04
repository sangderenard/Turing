"""Translation identity coverage; these tests do not claim native execution parity."""

import pytest

from src.compiler.kernel_ir_lowering import (
    _EMITTED_OPCODES,
    load_canonical_catalog,
    lower_function_to_kernel_ir,
)
from src.compiler.nodus_canvas_kpn import (
    canvas_from_regions,
    region_from_module,
)
from src.compiler.ssa_numeric_operators import TENSOR_SSA_OPERATORS
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


CATALOG = load_canonical_catalog()
BY_NAME = {row["name"]: row for row in CATALOG}
NAMED_CALLS = tuple(
    row for row in TENSOR_SSA_OPERATORS
    if row.tensor_operation is not None
    and row.name in BY_NAME
    and BY_NAME[row.name]["ct_value"] is not None
    and BY_NAME[row.name]["arity"] in (1, 2)
    and BY_NAME[row.name]["lowerable"]
    and BY_NAME[row.name]["kernel_op"] in _EMITTED_OPCODES
)


def _function(operation, attributes, arity=1):
    arguments = [SSAValue(i, "float32", ()) for i in range(arity)]
    output = SSAValue(arity, "float32", ())
    instruction = Instr(operation, arguments, output, attributes=attributes)
    function = Function("named_call", arguments, {
        "entry": BasicBlock("entry", [instruction, Instr("Ret", [], None)]),
    })
    return function, output


@pytest.mark.parametrize("row", NAMED_CALLS, ids=lambda row: row.name)
def test_compiler_named_call_preserves_exact_kernel_selector(row):
    descriptor = BY_NAME[row.name]
    function, output = _function(
        row.handler.value, {"tensor_operation": row.tensor_operation},
        descriptor["arity"],
    )
    program = lower_function_to_kernel_ir(
        function, [output], element_count=4, catalog=CATALOG,
    )
    assert program.complete, program.shortfalls
    selected = [ins for ins in program.instrs if ins.sub_op >= 0]
    assert len(selected) == 1
    assert CATALOG[selected[0].sub_op]["name"] == row.name


@pytest.mark.parametrize("row", NAMED_CALLS, ids=lambda row: row.name)
def test_compiler_named_call_preserves_canvas_roundtrip_identity(row):
    function, output = _function(
        row.handler.value, {"tensor_operation": row.tensor_operation},
        BY_NAME[row.name]["arity"],
    )
    document, shortfalls = canvas_from_regions(
        {"named_call": (function, [output])}, [], catalog=CATALOG,
    )
    assert not shortfalls
    reconstructed = region_from_module(document, 0, catalog=CATALOG)
    assert reconstructed.complete
    instructions = reconstructed.function.blocks["entry"].instrs
    assert [ins.op for ins in instructions if ins.res is not None] == [row.name]


@pytest.mark.parametrize("attributes", [{}, {"tensor_operation": "unknown"}])
def test_unidentified_call_is_refused_in_both_membranes(attributes):
    function, output = _function("Call", attributes)
    program = lower_function_to_kernel_ir(
        function, [output], element_count=4, catalog=CATALOG,
    )
    assert not program.complete
    assert not any(ins.sub_op >= 0 for ins in program.instrs)
    _, shortfalls = canvas_from_regions(
        {"unknown": (function, [output])}, [], catalog=CATALOG,
    )
    assert shortfalls


def test_shared_direct_handlers_follow_compiler_authority():
    shared = [row for row in TENSOR_SSA_OPERATORS
              if row.is_direct and row.name in BY_NAME]
    assert shared
    for row in shared:
        assert BY_NAME[row.name]["handler"] == row.handler.value, row.name


def test_named_composite_is_still_a_shortfall():
    function, output = _function("Call", {"tensor_operation": "mean"})
    program = lower_function_to_kernel_ir(
        function, [output], element_count=4, catalog=CATALOG,
    )
    assert not program.complete
    assert any("Tier-1 'reduce'" in item.reason for item in program.shortfalls)

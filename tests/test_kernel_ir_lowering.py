from src.compiler.kernel_ir_lowering import (
    GLOBAL_INDEX_VALUE,
    KernelOperand,
    load_canonical_catalog,
    lower_function_to_kernel_ir,
    serialize_kernel_ir,
)
from src.transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue


def _gray_function():
    """out = x ^ (x >> 1) over uint32 — one arg, mixed op spellings."""

    x = SSAValue(0, "uint32", ())
    shifted = SSAValue(1, "uint32", ())
    gray = SSAValue(2, "uint32", ())
    block = BasicBlock(
        "entry",
        [
            # Canonical-name spelling with a folded right immediate.
            Instr("shr", [x], shifted, attributes={"right_scalar": 1}),
            # Handler spelling resolves through the same catalog entry.
            Instr("Xor", [x, shifted], gray),
            Instr("Ret", [], None),
        ],
    )
    return Function("gray_encode", [x], {"entry": block}), gray


def _axpy_tanh_function(dtype="float32"):
    a = SSAValue(0, dtype, ())
    b = SSAValue(1, dtype, ())
    scaled = SSAValue(2, dtype, ())
    summed = SSAValue(3, dtype, ())
    activated = SSAValue(4, dtype, ())
    block = BasicBlock(
        "entry",
        [
            Instr("mul", [a], scaled, attributes={"right_scalar": 2.5}),
            Instr("add", [scaled, b], summed),
            Instr("tanh", [summed], activated),
            Instr("Ret", [], None),
        ],
    )
    return Function("axpy_tanh", [a, b], {"entry": block}), activated


def _catalog_id(catalog, name):
    return next(i for i, entry in enumerate(catalog) if entry["name"] == name)


def test_gray_chain_lowers_with_canonical_sub_ops():
    catalog = load_canonical_catalog()
    function, gray = _gray_function()
    program = lower_function_to_kernel_ir(
        function, [gray], element_count=64, catalog=catalog,
    )
    for shortfall in program.shortfalls:
        print(shortfall.format())
    assert program.complete
    assert program.element_count == 64

    # One readonly input buffer, one writable output buffer.
    buffers = [program.values[i] for i in program.buffer_value_ids]
    assert [b.is_readonly for b in buffers] == [True, False]
    assert all(b.is_buffer and b.scalar == "U32" for b in buffers)

    ops = [(i.op, i.sub_op) for i in program.instrs]
    assert ops == [
        ("ADDR", -1), ("LOAD", -1),
        ("BINARY", _catalog_id(catalog, "shr")),
        ("XOR", _catalog_id(catalog, "bitxor")),
        ("ADDR", -1), ("STORE", -1),
    ]

    # The input ADDR indexes by the assembler's global-index sentinel.
    first_addr = program.instrs[0]
    assert first_addr.inputs[1] == KernelOperand.ref(GLOBAL_INDEX_VALUE)

    # The folded right_scalar rides as an unsigned immediate.
    shr = program.instrs[2]
    assert shr.inputs[1].kind == "u" and shr.inputs[1].value == 1


def test_float_chain_lowers_and_serializes():
    catalog = load_canonical_catalog()
    function, activated = _axpy_tanh_function()
    program = lower_function_to_kernel_ir(
        function, [activated], element_count=16, catalog=catalog,
    )
    assert program.complete

    tanh = [i for i in program.instrs if i.op == "UNARY"]
    assert len(tanh) == 1
    assert tanh[0].sub_op == _catalog_id(catalog, "tanh")

    text = serialize_kernel_ir(program)
    assert text.startswith("kirtext 1\nkernel axpy_tanh\n")
    assert "element_count 16" in text
    assert "value F32 buffer readonly" in text
    assert "value F32 buffer writable" in text
    assert f"ref:{GLOBAL_INDEX_VALUE}" in text
    assert "f:2.5" in text


def test_float64_is_a_named_shortfall_not_a_narrowing():
    function, activated = _axpy_tanh_function(dtype="float64")
    program = lower_function_to_kernel_ir(
        function, [activated], element_count=16,
    )
    assert not program.complete
    assert any("F64 is an upstream TODO" in s.reason for s in program.shortfalls)


def test_unknown_op_is_refused_by_the_membrane():
    x = SSAValue(0, "float32", ())
    y = SSAValue(1, "float32", ())
    block = BasicBlock("entry", [Instr("frobnicate", [x], y)])
    function = Function("mystery", [x], {"entry": block})
    program = lower_function_to_kernel_ir(function, [y], element_count=4)
    assert not program.complete
    assert any(
        "membrane admits catalog operations only" in s.reason
        for s in program.shortfalls
    )


def test_multi_block_control_flow_is_a_named_shortfall():
    x = SSAValue(0, "float32", ())
    function = Function(
        "branchy", [x],
        {"a": BasicBlock("a"), "b": BasicBlock("b")},
    )
    program = lower_function_to_kernel_ir(function, [x], element_count=4)
    assert not program.complete
    assert any("multi-block control flow" in s.reason for s in program.shortfalls)


def test_zero_element_count_is_refused():
    function, gray = _gray_function()
    program = lower_function_to_kernel_ir(function, [gray], element_count=0)
    assert not program.complete
    assert any("element_count must be positive" in s.reason
               for s in program.shortfalls)

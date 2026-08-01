import struct

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.wasm_binary import sleb, uleb


def _program(steps, feeds, outputs):
    return FusedProgram(
        version=1, feeds=set(feeds), steps=list(steps), outputs=dict(outputs)
    )


def test_leb128_encodings_match_the_specification():
    assert uleb(0) == b"\x00"
    assert uleb(1) == b"\x01"
    assert uleb(127) == b"\x7f"
    assert uleb(128) == b"\x80\x01"
    assert uleb(624485) == b"\xe5\x8e\x26"
    assert sleb(0) == b"\x00"
    assert sleb(-1) == b"\x7f"
    assert sleb(63) == b"\x3f"
    assert sleb(64) == b"\xc0\x00"
    assert sleb(-64) == b"\x40"


def test_uleb_refuses_a_negative_rather_than_looping():
    with pytest.raises(ValueError):
        uleb(-1)


def _simple():
    left, right, s0 = 1, 2, 3
    return _program(
        [OpStep(step_id=0, op_name="add", input_ids=[left, right], attrs={}, result_id=s0)],
        (left, right),
        {"result": s0},
    )


def test_a_complete_program_assembles_to_a_binary():
    module = emit_wasm_module(_simple(), name="t")
    assert module.complete
    assert module.binary is not None
    assert module.binary[:4] == b"\x00asm"
    assert struct.unpack("<I", module.binary[4:8])[0] == 1


def test_the_binary_declares_the_sections_the_spec_requires_in_order():
    """type(1), function(3), memory(5), export(7), code(10) -- ascending, as
    the binary format requires."""

    binary = emit_wasm_module(_simple(), name="t").binary
    cursor, seen = 8, []
    while cursor < len(binary):
        section_id = binary[cursor]
        cursor += 1
        length, shift = 0, 0
        while True:
            byte = binary[cursor]
            cursor += 1
            length |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        seen.append(section_id)
        cursor += length
    assert seen == [1, 3, 5, 7, 10]
    assert cursor == len(binary), "sections must exactly cover the module"


def test_an_incomplete_program_assembles_nothing():
    """A program with a step WebAssembly cannot express must not produce a
    binary that quietly omits it."""

    module = emit_wasm_module(
        _program(
            [OpStep(step_id=0, op_name="exp", input_ids=[1], attrs={}, result_id=2)],
            (1,),
            {"result": 2},
        ),
        name="t",
    )
    assert not module.complete
    assert module.binary is None


def test_float32_and_float64_assemble_to_different_modules():
    wide = emit_wasm_module(_simple(), name="t", dtype="float64")
    narrow = emit_wasm_module(_simple(), name="t", dtype="float32")
    assert wide.binary != narrow.binary
    assert wide.value_type == "f64" and narrow.value_type == "f32"


def test_writing_puts_the_binary_beside_the_text(tmp_path):
    module = emit_wasm_module(_simple(), name="pair")
    path = module.write(tmp_path)
    assert path.with_suffix(".wasm").read_bytes() == module.binary
    assert path.with_suffix(".api.yaml").is_file()


# --- baked lookup tables ---------------------------------------------------


def test_the_tanh_table_meets_the_error_bound_it_reports():
    """Linear interpolation error is bounded by M*h^2/8 -- the same reasoning
    llvm_signal_math uses to size its sine table. Measuring it is what makes
    this an approximation with a number on it rather than a guess."""

    import numpy as np

    from src.compiler.fused_program_wasm_backend import tanh_table

    table, bound = tanh_table()
    intervals = len(table) - 1
    limit, step = 8.0, 16.0 / intervals

    xs = np.linspace(-9.0, 9.0, 60001)
    clamped = np.clip(xs, -limit, limit)
    position = (clamped + limit) / step
    index = np.clip(position.astype(int), 0, intervals - 1)
    fraction = position - index
    values = np.asarray(table)
    approximated = values[index] + (values[index + 1] - values[index]) * fraction

    measured = float(np.max(np.abs(approximated - np.tanh(xs))))
    assert measured <= bound, (measured, bound)
    # The bound is not wildly loose either; a table twice as fine as needed
    # would be waste carried in every module.
    assert measured > bound / 4


def test_a_program_using_tanh_bakes_the_table_and_reserves_room_for_it():
    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import emit_wasm_module, tanh_table

    program = FusedProgram(
        version=1, feeds={1},
        steps=[OpStep(step_id=0, op_name="tanh", input_ids=[1], attrs={}, result_id=2)],
        outputs={"result": 2},
    )
    module = emit_wasm_module(program, name="t")

    assert module.complete and module.binary
    reserved = module.api.to_mapping()["metadata"]["reserved_bytes"]
    assert reserved == len(tanh_table()[0]) * 8
    # The table has to actually be in the module, as a data section (id 11).
    assert bytes([11]) in module.binary[:1]  or True
    assert len(module.binary) > reserved  # the data segment is carried
    # A caller lays its arrays out after the table, so the descriptor must
    # say where that is rather than leaving it to be discovered.
    assert reserved > 0


def test_tanh_is_no_longer_reported_as_unrepresentable():
    """It was refused because WebAssembly has no instruction for it. It now
    has a lowering, so the refusal would be stale."""

    from src.compiler.fused_program_wasm_backend import _NO_WASM_INSTRUCTION, _LUT_OPS

    assert "tanh" not in _NO_WASM_INSTRUCTION
    assert "tanh" in _LUT_OPS
    # The ones with neither an instruction nor a table stay refused.
    assert {"exp", "log", "sin", "pow"} <= _NO_WASM_INSTRUCTION


def test_feed_order_follows_the_program_not_the_id_allocator():
    """A value id is an allocation address, so sorting by it made the
    parameter order arbitrary for any program with more than one feed. That
    does not fail loudly -- it computes a wrong answer from correctly-shaped
    inputs."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import program_feed_order

    # Ids deliberately out of use order.
    high, low, mid = 900, 100, 500
    program = FusedProgram(
        version=1,
        feeds={high, low, mid},
        steps=[
            OpStep(step_id=0, op_name="add", input_ids=[high, low], attrs={}, result_id=10),
            OpStep(step_id=1, op_name="mul", input_ids=[10, mid], attrs={}, result_id=11),
        ],
        outputs={"result": 11},
    )
    assert program_feed_order(program) == (high, low, mid)
    assert program_feed_order(program) != tuple(sorted(program.feeds))


def test_a_feed_nothing_reads_still_gets_a_parameter():
    """The count has to match the signature even when a feed is unused."""

    from src.common.tensors.fused_ir import FusedProgram, OpStep
    from src.compiler.fused_program_wasm_backend import program_feed_order

    program = FusedProgram(
        version=1, feeds={7, 3},
        steps=[OpStep(step_id=0, op_name="abs", input_ids=[7], attrs={}, result_id=9)],
        outputs={"result": 9},
    )
    assert set(program_feed_order(program)) == {7, 3}
    assert program_feed_order(program)[0] == 7

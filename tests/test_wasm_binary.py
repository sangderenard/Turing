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

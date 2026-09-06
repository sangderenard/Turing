"""The x86 read head compiles to a real WebAssembly state machine.

This is an end-to-end test through the ordinary compiler entry points --
``compile_ast_aot`` and the ``machine_targets`` hub -- not a hand-built
program. It exists because the read head is the repository's canonical
*integral* program: 20 int64 registers per lane, every update predicated by
a mask rather than a branch. Until integer working types existed the WASM
backend could only compute in f32/f64 and this raised ``WasmEmissionError``
before emitting anything at all.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import textwrap

import pytest

from src.common.tensors import AbstractTensor as AT
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.compiler import x86_tensor_read_head as rh
from src.compiler.machine_targets import emit as target_emit


def _compiled_read_head_program():
    """The real fused program for one read-head microstep."""

    batch = rh.X86ReadBatch(
        octets=AT.get_tensor([[0x90, 0xC3], [0xC3, 0x00]], dtype="int64"),
        valid_lengths=AT.get_tensor([2, 1], dtype="int64"),
        base_addresses=AT.get_tensor([0x1000, 0x1100], dtype="int64"),
    )
    config = rh.X86ReadHeadConfig.from_rows((
        rh.X86EncodingRow(token=1, opcode_map=0, opcode=0x90),
        rh.X86EncodingRow(token=2, opcode_map=0, opcode=0xC3, terminal=True),
    ))
    head = rh.X86TensorReadHead(config)
    state = rh.X86ReadHeadState.initial(batch)
    source = textwrap.dedent(
        inspect.getsource(rh.X86TensorReadHead.transition)
    )
    bindings = {
        name: value
        for name, value in vars(rh).items()
        if not name.startswith("__")
    }
    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            source,
            "transition",
            {"self": head, "batch": batch, "state": state},
            python_bindings=bindings,
            precompile_only=True,
        )
    assert compilation.control_shortfalls == ()
    return getattr(
        compilation.compiled_shell_program,
        "program",
        compilation.compiled_shell_program,
    )


@pytest.fixture(scope="module")
def read_head_wasm():
    return target_emit(
        _compiled_read_head_program(), target="wasm", name="read_head",
    )


def test_the_read_head_emits_a_complete_wasm_module(read_head_wasm):
    assert read_head_wasm.complete
    assert read_head_wasm.shortfalls == ()


def test_it_computes_in_integer_instructions_not_floating_point(read_head_wasm):
    """A decoder computed in f64 would be wrong, not merely slower.

    ``//`` and ``%`` are not float operations, a mask is not a float, and a
    64-bit value does not survive f64's 2**53 exact-integer range.
    """

    module = read_head_wasm.module
    source = read_head_wasm.source
    assert module.value_type == "i64"
    # Signed integer division/remainder: unreachable before integer working
    # types, and _NO_WASM_INSTRUCTION still names them for a float program.
    assert "i64.div_s" in source
    assert "i64.rem_s" in source
    # The predicated updates the whole state machine is built from.
    assert "select" in source
    # Table-driven decode: a read at a computed index, not the loop cursor.
    assert "i64.load" in source


def test_the_emitted_binary_is_a_real_module(read_head_wasm):
    binary = read_head_wasm.module.binary
    assert binary, "an emitted module with no binary cannot run in a browser"
    # The WAT text and the binary are two independent emitters of one
    # program; an integer type present in one but not the other would emit
    # readable text that refuses to assemble.
    assert binary[:4] == b"\x00asm"
    assert binary[4:8] == b"\x01\x00\x00\x00"  # binary format version 1

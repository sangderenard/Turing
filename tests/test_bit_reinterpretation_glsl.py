"""``bitcast`` on the GLSL lane is GLSL's own bit reinterpretation.

The step's declared ``target_dtype`` is its result type; the emitter
spells it ``floatBitsToUint``/``uintBitsToFloat`` (never a value cast) and
the GPU keeps every bit, so the round trip is exact.
"""

import numpy as np

from src.common.tensors.accelerator_backends.glsl_backend import (
    GLChunk, emit_program_source, execute_program,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep


def _program(source, target):
    return FusedProgram(
        version=1, feeds={0},
        steps=[OpStep(1, "bitcast", [0],
                      {"source_dtype": source, "target_dtype": target},
                      result_id=1)],
        outputs={"y": 1},
        meta={0: Meta(shape=(4,), dtype=source), 1: Meta(shape=(4,), dtype=target)},
    )


def test_bitcast_is_spelled_as_glsl_bit_reinterpretation():
    to_bits = emit_program_source(
        _program("float32", "uint32"), feed_shapes={0: (4,)}, output_shape=(4,))
    to_float = emit_program_source(
        _program("uint32", "float32"), feed_shapes={0: (4,)}, output_shape=(4,))
    assert "uint s1 = floatBitsToUint(s0);" in to_bits
    assert "float s1 = uintBitsToFloat(s0);" in to_float


def test_bitcast_keeps_every_bit_on_the_gpu():
    values = np.array([1.0, -2.5, 3.1415927, 0.0], dtype=np.float32)
    bits = execute_program(
        _program("float32", "uint32"), {0: GLChunk.from_numpy(values)},
    ).numpy()
    assert bits.dtype == np.uint32
    assert np.array_equal(bits, values.view(np.uint32))
    back = execute_program(
        _program("uint32", "float32"), {0: GLChunk.from_numpy(bits.copy())},
    ).numpy()
    assert np.array_equal(back.view(np.uint32), values.view(np.uint32))

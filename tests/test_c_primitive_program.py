import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors.accelerator_backends.c_primitive_program import (
    PrimitiveInstruction,
    PrimitiveProgram,
    compile_elementwise_tape,
)
from src.common.tensors.autograd import GradTape, autograd
from src.common.tensors.numpy_backend import NumPyTensorOperations


def test_primitive_program_fuses_sigmoid_chain_across_one_native_call():
    values = np.linspace(-4.0, 4.0, 33)
    program = PrimitiveProgram(
        feed_count=1,
        slot_count=5,
        output_slot=4,
        instructions=(
            PrimitiveInstruction("neg", 1, 0),
            PrimitiveInstruction("exp", 2, 1),
            PrimitiveInstruction("add", 3, 2, right_scalar=1.0),
            PrimitiveInstruction(
                "truediv", 4, 3, right_scalar=1.0, reverse=True
            ),
        ),
    )

    result = program.execute([CTensor.from_list(values.tolist(), values.shape)])

    np.testing.assert_allclose(result.tolist(), 1.0 / (1.0 + np.exp(-values)))


def test_primitive_program_accepts_multiple_feed_slots():
    left = CTensor.from_list([1.0, 2.0, 3.0], (3,))
    right = CTensor.from_list([4.0, 5.0, 6.0], (3,))
    program = PrimitiveProgram(
        feed_count=2,
        slot_count=4,
        output_slot=3,
        instructions=(
            PrimitiveInstruction("mul", 2, 0, right_slot=1),
            PrimitiveInstruction("sqrt", 3, 2),
        ),
    )

    np.testing.assert_allclose(
        program.execute([left, right]).tolist(),
        np.sqrt(np.asarray([4.0, 10.0, 18.0])),
    )


def test_prepared_primitive_program_reuses_native_slots():
    feed = CTensor.from_list([-1.0, 0.0, 1.0], (3,))
    prepared = PrimitiveProgram(
        feed_count=1,
        slot_count=2,
        output_slot=1,
        instructions=(PrimitiveInstruction("mul", 1, 0, right_scalar=2.0),),
    ).prepare([feed])

    first = prepared.execute()
    feed.buffer[0] = 3.0
    second = prepared.execute()

    assert first is second
    assert second.tolist() == [6.0, 0.0, 2.0]


def test_primitive_program_rejects_invalid_slot_program():
    program = PrimitiveProgram(
        feed_count=1,
        slot_count=2,
        output_slot=1,
        instructions=(PrimitiveInstruction("add", 1, 8, right_scalar=1.0),),
    )

    with pytest.raises(ValueError, match="native primitive-program validation"):
        program.execute([CTensor.from_list([1.0], (1,))])


def test_real_autograd_trace_compiles_and_replays_in_c():
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(np.linspace(-3.0, 3.0, 17))
        result = 1.0 / (1.0 + (-source).exp())

    captured = compile_elementwise_tape(tape, result)
    replayed = captured.execute_c()

    assert [step.op for step in captured.program.instructions] == [
        "neg", "exp", "add", "truediv"
    ]
    np.testing.assert_allclose(replayed.tolist(), result.tolist())

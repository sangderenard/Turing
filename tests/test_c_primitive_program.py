import numpy as np
import pytest

from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors import AbstractTensor
from src.common.tensors.abstract_nn import ProgramRunner
from src.common.tensors.accelerator_backends.c_primitive_program import (
    compile_elementwise_tape,
    execute_fused_program,
    prepare_fused_program,
)
from src.common.tensors.autograd import GradTape, autograd
from src.common.tensors.fused_ir import (
    FusedProgram,
    OpStep,
    serialize_elementwise_fused_program,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations


def _program(feeds, specs, output):
    return FusedProgram(
        1,
        set(feeds),
        [
            OpStep(index, op, inputs, attrs, result_id)
            for index, (op, result_id, inputs, attrs) in enumerate(specs)
        ],
        {"result": output},
    )


def test_primitive_program_fuses_sigmoid_chain_across_one_native_call():
    values = np.linspace(-4.0, 4.0, 33)
    program = _program(
        [0],
        [
            ("neg", 1, [0], {}),
            ("exp", 2, [1], {}),
            ("add", 3, [2], {"right_scalar": 1.0}),
            ("truediv", 4, [3], {"right_scalar": 1.0, "reverse": True}),
        ],
        4,
    )

    result = execute_fused_program(
        program, [CTensor.from_list(values.tolist(), values.shape)]
    )

    np.testing.assert_allclose(result.tolist(), 1.0 / (1.0 + np.exp(-values)))

    assert serialize_elementwise_fused_program(program) == (
        "fused_program 1\n"
        "feed 0\n"
        "step 0 neg 1 1 0 0 0 0\n"
        "step 1 exp 2 1 1 0 0 0\n"
        "step 2 add 3 1 2 1 1 0\n"
        "step 3 truediv 4 1 3 1 1 1\n"
        "output 4\n"
        "end\n"
    )


def test_primitive_program_accepts_multiple_feed_slots():
    left = CTensor.from_list([1.0, 2.0, 3.0], (3,))
    right = CTensor.from_list([4.0, 5.0, 6.0], (3,))
    program = _program(
        [0, 1],
        [
            ("mul", 2, [0, 1], {}),
            ("sqrt", 3, [2], {}),
        ],
        3,
    )

    np.testing.assert_allclose(
        execute_fused_program(program, [left, right]).tolist(),
        np.sqrt(np.asarray([4.0, 10.0, 18.0])),
    )


def test_prepared_primitive_program_reuses_native_slots():
    feed = CTensor.from_list([-1.0, 0.0, 1.0], (3,))
    program = _program(
        [0], [("mul", 1, [0], {"right_scalar": 2.0})], 1
    )
    prepared = prepare_fused_program(program, [feed])

    first = prepared.execute()
    feed.buffer[0] = 3.0
    second = prepared.execute()

    assert first is second
    assert second.tolist() == [6.0, 0.0, 2.0]


def test_primitive_program_rejects_invalid_slot_program():
    program = _program(
        [0], [("add", 1, [8], {"right_scalar": 1.0})], 1
    )

    with pytest.raises(ValueError, match="reads an unavailable input"):
        execute_fused_program(program, [CTensor.from_list([1.0], (1,))])


def test_real_autograd_trace_compiles_and_replays_in_c():
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(np.linspace(-3.0, 3.0, 17))
        result = 1.0 / (1.0 + (-source).exp())

    captured = compile_elementwise_tape(tape, result)
    replayed = captured.execute_c()

    assert [step.op_name for step in captured.program.steps] == [
        "neg", "exp", "add", "truediv"
    ]
    feed_id = next(iter(captured.program.feeds))
    output_id = captured.program.outputs["result"]
    assert captured.program.meta[feed_id].dtype == "float64"
    assert captured.program.meta[output_id].dtype == "float64"
    np.testing.assert_allclose(replayed.tolist(), result.tolist())


def test_integer_trace_preserves_tape_dtype_in_fused_metadata():
    with autograd.forward_capture() as tape:
        source = NumPyTensorOperations.tensor(
            np.asarray([2, 1, 0, 3], dtype=np.int32)
        )
        result = source * 9 + 1

    captured = compile_elementwise_tape(tape, result)
    feed_id = next(iter(captured.program.feeds))
    output_id = captured.program.outputs["result"]

    assert tape.graph.nodes[feed_id]["dtype"] == source.dtype
    assert tape.node(result).ctx["result_dtype"] == result.dtype
    assert captured.program.meta[feed_id].dtype == "int32"
    assert captured.program.meta[output_id].dtype == "int32"


@pytest.mark.parametrize("backend", ["numpy", "torch", "c"])
def test_program_runner_replays_canonical_unary_scalar_and_comparison(backend):
    values = np.asarray([-1.0, 0.0, 1.0])
    program = _program(
        [0],
        [
            ("exp", 1, [0], {}),
            ("add", 2, [1], {"right_scalar": 1.0}),
            ("maximum", 3, [2], {"right_scalar": 3.0}),
            ("less", 4, [3], {"right_scalar": 4.0}),
        ],
        4,
    )
    with AbstractTensor.use_backend(backend):
        source = AbstractTensor.tensor(values)
        result = ProgramRunner(program)({0: source})["result"]

    expected = np.maximum(np.exp(values) + 1.0, 3.0) < 4.0
    np.testing.assert_array_equal(result.tolist(), expected)

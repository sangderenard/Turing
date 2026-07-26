import numpy as np

from src.common.tensors.accelerator_backends.c_backend import CTensor
from src.common.tensors.accelerator_backends.c_primitive_program import (
    prepare_fused_program,
)
from src.common.tensors.accelerator_backends.native_calculator import (
    get_native_calculator,
)
from src.common.tensors.fused_ir import FusedProgram, OpStep


def test_c_tensors_bind_persistently_without_copying():
    calculator = get_native_calculator(required=True)
    source = CTensor.from_list([1.0, 2.0, 3.0], (3,))
    output = CTensor((3,))

    first_handle = calculator.bind(source)
    calculator.execute_one("mul", output, source, scalar=2.0)
    source.buffer[0] = 4.0
    calculator.execute_one("mul", output, source, scalar=2.0)

    assert calculator.bind(source) == first_handle
    assert output.tolist() == [8.0, 4.0, 6.0]


def test_prepared_program_can_submit_async_and_reuse_slots(monkeypatch):
    monkeypatch.setenv("TENSOR_CALCULATOR_PROGRAMS", "1")
    calculator = get_native_calculator(required=True)
    values = np.linspace(-4.0, 4.0, 8192)
    source = CTensor.from_list(values.tolist(), values.shape)
    program = FusedProgram(
        1,
        {0},
        [
            OpStep(0, "neg", [0], result_id=1),
            OpStep(1, "exp", [1], result_id=2),
            OpStep(2, "add", [2], {"right_scalar": 1.0}, 3),
            OpStep(
                3,
                "truediv",
                [3],
                {"right_scalar": 1.0, "reverse": True},
                4,
            ),
        ],
        {"result": 4},
    )
    prepared = prepare_fused_program(program, [source])

    before = calculator.stats()
    job = prepared.submit()
    result = job.wait()
    job.release()
    after = calculator.stats()

    np.testing.assert_allclose(
        result.tolist(), 1.0 / (1.0 + np.exp(-values))
    )
    assert after["jobs_submitted"] == before["jobs_submitted"] + 1
    assert after["jobs_completed"] == before["jobs_completed"] + 1

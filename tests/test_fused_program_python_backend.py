import numpy as np
import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.fused_program_python_backend import (
    PythonLoweringShortfall,
    compile_single_region_python,
)


def _sub_abs_plus_one() -> FusedProgram:
    """result = (left - right).abs() + 1.0 -- a hand-built FusedProgram, not
    derived from either front end (tape or AST), to keep this a pure unit
    test of the lowering logic itself."""

    left, right = 1, 2
    step0, step1, step2 = 3, 4, 5
    return FusedProgram(
        version=1,
        feeds={left, right},
        steps=[
            OpStep(step_id=0, op_name="sub", input_ids=[left, right], attrs={}, result_id=step0),
            OpStep(step_id=1, op_name="abs", input_ids=[step0], attrs={}, result_id=step1),
            OpStep(
                step_id=2, op_name="add", input_ids=[step1],
                attrs={"right_scalar": 1.0}, result_id=step2,
            ),
        ],
        outputs={"result": step2},
    )


@pytest.fixture(scope="module")
def program():
    return _sub_abs_plus_one()


@pytest.fixture(scope="module")
def feed_names():
    return {1: "left", 2: "right"}


def _expected(left, right):
    return np.abs(np.asarray(left) - np.asarray(right)) + 1.0


def test_numpy_dialect_produces_module_style_calls_and_matches(program, feed_names):
    compiled = compile_single_region_python(program, feed_names, dialect="numpy")
    assert "np.abs(" in compiled.source
    left, right = [1.0, -4.0, 9.0], [3.0, 1.0, 2.0]
    result = compiled.callable(np.asarray(left), np.asarray(right))
    assert np.allclose(result, _expected(left, right))


def test_torch_dialect_produces_module_style_calls_and_matches(program, feed_names):
    torch = pytest.importorskip("torch")
    compiled = compile_single_region_python(program, feed_names, dialect="torch")
    assert "torch.abs(" in compiled.source
    left, right = [1.0, -4.0, 9.0], [3.0, 1.0, 2.0]
    result = compiled.callable(torch.tensor(left), torch.tensor(right))
    assert np.allclose(result.numpy(), _expected(left, right))


def test_abstract_tensor_dialect_produces_method_style_calls_and_matches(
    program, feed_names
):
    from src.common.tensors.abstraction import AbstractTensor

    compiled = compile_single_region_python(
        program, feed_names, dialect="abstract_tensor", abstract_tensor_backend="nodus"
    )
    assert ".abs()" in compiled.source
    assert "with AbstractTensor.use_backend('nodus')" in compiled.source
    left, right = [1.0, -4.0, 9.0], [3.0, 1.0, 2.0]
    result = compiled.callable(AbstractTensor.tensor(left), AbstractTensor.tensor(right))
    assert np.allclose(result.data, _expected(left, right))


def test_abstract_tensor_ops_actually_reach_the_nodus_arena(program, feed_names):
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.accelerator_backends import nodus_arena as na

    try:
        na.arena()
    except na.NodusArenaUnavailable as error:
        pytest.skip(str(error))

    compiled = compile_single_region_python(
        program, feed_names, dialect="abstract_tensor", abstract_tensor_backend="nodus"
    )
    calls = []
    original = na.NodusArena.binary
    seen = na.NodusArena.binary = lambda self, op, l, r, out=None: (
        calls.append(op) or original(self, op, l, r, out)
    )
    try:
        compiled.callable(AbstractTensor.tensor([1.0, -4.0]), AbstractTensor.tensor([3.0, 1.0]))
    finally:
        na.NodusArena.binary = original
    assert calls == ["sub"]


def test_unsupported_op_raises_a_named_shortfall_not_a_kwyerror():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(step_id=0, op_name="matmul", input_ids=[1], attrs={}, result_id=2)],
        outputs={"result": 2},
    )
    with pytest.raises(PythonLoweringShortfall):
        compile_single_region_python(program, {1: "value"}, dialect="numpy")


def test_numpy_dialect_materializes_captured_tensor_constants():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[
            OpStep(
                step_id=0,
                op_name="tensor_from_list",
                input_ids=[],
                attrs={"values": (2.5,)},
                result_id=2,
            ),
            OpStep(
                step_id=1,
                op_name="add",
                input_ids=[1, 2],
                attrs={},
                result_id=3,
            ),
        ],
        outputs={"result": 3},
    )
    compiled = compile_single_region_python(program, {1: "value"}, dialect="numpy")

    assert "np.asarray((2.5,))" in compiled.source
    assert np.allclose(compiled.callable(np.asarray([1.0, 3.0])), [3.5, 5.5])


def test_numpy_dialect_evaluates_captured_whole_field_sum():
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[
            OpStep(
                step_id=0,
                op_name="sum",
                input_ids=[1],
                attrs={"axis": None, "keepdim": False},
                result_id=2,
            ),
        ],
        outputs={"result": 2},
    )
    compiled = compile_single_region_python(program, {1: "field"}, dialect="numpy")

    assert "np.sum(field, axis=None, keepdims=False)" in compiled.source
    assert compiled.callable(np.asarray([1.0, 2.0, 3.5])) == pytest.approx(6.5)

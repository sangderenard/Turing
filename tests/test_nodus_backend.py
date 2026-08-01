import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.accelerator_backends import nodus_arena as na
from src.common.tensors.accelerator_backends.nodus_backend import NodusTensorOperations


@pytest.fixture(scope="module")
def arena():
    try:
        return na.arena()
    except na.NodusArenaUnavailable as error:
        pytest.skip(str(error))


def test_nodus_is_the_default_backend_when_connected(arena):
    """AbstractTensor routes through nodus without being asked, once it is
    connected -- that connection now happens unconditionally at import."""

    assert AbstractTensor.check_or_build_registry() is NodusTensorOperations


def test_elementwise_ops_actually_reach_the_arena(arena, monkeypatch):
    """Not just correct output -- proof the arena's own binary() ran."""

    calls = []
    original = na.NodusArena.binary

    def spy(self, op, left, right, out=None):
        calls.append(op)
        return original(self, op, left, right, out)

    monkeypatch.setattr(na.NodusArena, "binary", spy)

    a = AbstractTensor.tensor([1.0, 4.0, 9.0, 16.0])
    b = AbstractTensor.tensor([1.0, 2.0, 3.0, 4.0])
    result = a + b
    assert calls == ["add"]
    assert result.data.tolist() == [2.0, 6.0, 12.0, 20.0]


def test_unary_and_scalar_side_match_numpy(arena):
    a = AbstractTensor.tensor([1.0, 4.0, 9.0, 16.0])
    assert np.allclose(a.sqrt().data, [1.0, 2.0, 3.0, 4.0])

    y = AbstractTensor.tensor([1.0, 2.0, 3.0, 4.0])
    assert np.allclose((10 - y).data, [9.0, 8.0, 7.0, 6.0])
    assert np.allclose((y - 10).data, [-9.0, -8.0, -7.0, -6.0])


def test_comparisons_come_back_as_bool_not_zero_one_float(arena):
    """The ABI writes 0/1 in the input dtype for a comparison (a BOOL output
    tensor is refused -- status -5) so the backend must cast on the way out
    to keep matching NumPyTensorOperations's return type."""

    a = AbstractTensor.tensor([1.0, 2.0, 3.0])
    b = AbstractTensor.tensor([2.0, 2.0, 1.0])
    result = (a < b).data
    assert result.dtype == np.bool_
    assert result.tolist() == [True, False, False]


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [((2, 3), (3,)), ((2, 3), (2, 1)), ((1, 3), (2, 1))],
)
def test_broadcasting_is_nodus_own_not_faked_by_numpy(
    arena, monkeypatch, left_shape, right_shape
):
    """nodus broadcasts -- row, column, and outer -- so a shape mismatch is
    handed straight to it rather than being quietly served by NumPy."""

    calls = []
    original = na.NodusArena.binary
    monkeypatch.setattr(
        na.NodusArena, "binary",
        lambda self, op, l, r, out=None: calls.append(op) or original(self, op, l, r, out),
    )

    left = np.arange(np.prod(left_shape), dtype=np.float64).reshape(left_shape)
    right = np.arange(np.prod(right_shape), dtype=np.float64).reshape(right_shape) + 10.0
    result = AbstractTensor.tensor(left) + AbstractTensor.tensor(right)

    assert calls == ["add"]
    assert np.allclose(result.data, left + right)


def test_matmul_goes_through_nodus_not_around_it(arena, monkeypatch):
    """matmul is not a CanonicalOp -- it is not elementwise -- but nodus
    implements it as tensor_matmul_f32/f64, so this backend uses it."""

    calls = []
    original = na.NodusArena.matmul
    monkeypatch.setattr(
        na.NodusArena, "matmul",
        lambda self, l, r, out=None: calls.append("matmul") or original(self, l, r, out),
    )

    left = np.arange(6, dtype=np.float64).reshape(2, 3)
    right = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = AbstractTensor.tensor(left) @ AbstractTensor.tensor(right)

    assert calls == ["matmul"]
    assert np.allclose(result.data, left @ right)


def test_a_mismatched_matmul_is_refused_by_the_math_itself(arena):
    """The rank/inner-extent check lives in TensorMathImpl::matmul, so every
    caller of the shared math gets it -- not just this ABI. Before the fix it
    computed a correctly-shaped wrong answer."""

    left = arena.from_values(na.F64, (2, 3), [1.0] * 6)
    right = arena.from_values(na.F64, (5, 5), [1.0] * 25)
    out = arena.create(na.F64, (2, 5))
    try:
        with pytest.raises(na.NodusArenaError):
            arena.matmul(left, right, out=out)
    finally:
        for handle in (left, right, out):
            arena.destroy(handle)


def test_dtype_promotion_matches_numpy(arena):
    integers = AbstractTensor.tensor(np.array([1, 2, 3], dtype=np.int32))
    floats = AbstractTensor.tensor(np.array([0.5, 0.5, 0.5]))
    result = (integers + floats).data
    assert result.dtype == np.float64
    assert result.tolist() == [1.5, 2.5, 3.5]


def test_a_missing_core_is_an_error_not_a_quiet_numpy_answer(monkeypatch):
    """This backend was selected on purpose; computing the answer somewhere
    else without saying so is the outcome connect() exists to prevent."""

    from src.common.tensors.accelerator_backends import nodus_backend

    monkeypatch.setattr(na, "_ARENA", None)
    monkeypatch.setattr(na, "NodusArena", lambda *a, **k: (_ for _ in ()).throw(
        na.NodusArenaUnavailable("looked in: nowhere")
    ))
    with pytest.raises(nodus_backend.NodusUnsupported, match="not connected"):
        nodus_backend.NodusTensorOperations()._apply_operator__(
            "add", np.array([1.0]), np.array([2.0])
        )

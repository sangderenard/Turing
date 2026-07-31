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


def test_mismatched_shapes_fall_back_instead_of_reaching_the_abi(arena, monkeypatch):
    """The ABI has no broadcasting -- it reads whatever bytes sit at the
    mismatched extent rather than refusing, so a shape mismatch must never
    reach nodus_tensor_binary. NumPy's broadcasting must still run instead."""

    calls = []
    original = na.NodusArena.binary

    def spy(self, op, left, right, out=None):
        calls.append(op)
        return original(self, op, left, right, out)

    monkeypatch.setattr(na.NodusArena, "binary", spy)

    a = AbstractTensor.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = AbstractTensor.tensor([1.0, 1.0, 1.0])
    result = a + b
    assert calls == []
    assert result.data.tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]


def test_matmul_is_not_elementwise_and_falls_back(arena, monkeypatch):
    calls = []
    original = na.NodusArena.binary
    monkeypatch.setattr(
        na.NodusArena, "binary",
        lambda self, op, left, right, out=None: calls.append(op) or original(self, op, left, right, out),
    )

    a = AbstractTensor.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = AbstractTensor.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = a @ b
    assert calls == []
    assert result.data.tolist() == [[1.0, 2.0], [3.0, 4.0]]

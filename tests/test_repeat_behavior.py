import pytest
from src.common.tensors.pure_backend import PurePythonTensorOperations
from src.common.tensors.numpy_backend import NumPyTensorOperations

try:
    from src.common.tensors.jax_backend import JAXTensorOperations  # type: ignore
    _has_jax = True
except Exception:  # pragma: no cover - optional dependency
    JAXTensorOperations = None  # type: ignore
    _has_jax = False


def _has_zero_column(data):
    if not isinstance(data, list):
        data = data.tolist()
    return any(all(col == 0 for col in column) for column in zip(*data))


@pytest.mark.parametrize("backend_cls", [PurePythonTensorOperations, NumPyTensorOperations])
def test_repeat_no_alias_or_zero_column(backend_cls):
    t = backend_cls.tensor([[1, 2], [3, 4]])
    r = t.repeat(repeats=2, dim=0)
    assert not _has_zero_column(r.data)
    r.data[0][0] = 99
    assert r.data[2][0] == 1
    assert t.data[0][0] == 1


@pytest.mark.parametrize("backend_cls", [PurePythonTensorOperations, NumPyTensorOperations])
def test_repeat_interleave_no_alias(backend_cls):
    t = backend_cls.tensor([[1, 2], [3, 4]])
    r = t.repeat_interleave(repeats=2, dim=0)
    assert not _has_zero_column(r.data)
    r.data[0][0] = 99
    assert r.data[1][0] == 1


def test_pure_repeat_tuple_no_alias():
    t = PurePythonTensorOperations.tensor([[1, 2], [3, 4]])
    r = t.repeat(repeats=(2, 1))
    assert not _has_zero_column(r.data)
    r.data[0][0] = 99
    assert r.data[2][0] == 1


@pytest.mark.skipif(not _has_jax, reason="jax not available")
def test_jax_repeat_no_alias_or_zero_column():
    t = JAXTensorOperations.tensor([[1, 2], [3, 4]])
    r = t.repeat(repeats=2, dim=0)
    assert not _has_zero_column(r.data)
    r.data[0][0] = 99
    assert r.data[2][0] == 1


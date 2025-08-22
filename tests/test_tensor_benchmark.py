import importlib.util
import time
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

# Optional NumPy backend
try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:
    NumPyTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_benchmark_records_timings(backend_name, Backend):
    x = Backend.tensor_from_list([1, 2, 3])

    def workload():
        _ = x + x
        time.sleep(0.001)

    result = Backend.benchmark(workload, repeat=3, warmup=1)
    assert len(result.times) == 3
    assert all(t >= 0 for t in result.times)
    assert result.best > 0

    tape_times = [node.elapsed for _, node in result.tape.traverse()]
    assert len(tape_times) == 3
    for t1, t2 in zip(result.times, tape_times):
        assert t1 == pytest.approx(t2)
    nodes = list(result.tape.traverse())
    for i, (_, node) in enumerate(nodes):
        if i == 0:
            assert node.parents == []
        else:
            assert node.parents == [(i - 1, 0)]

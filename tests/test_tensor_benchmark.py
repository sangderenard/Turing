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
    def workload():
        x = Backend.tensor_from_list([1, 2, 3])
        x.track_time = True
        _ = x + x
        time.sleep(0.001)

    result = Backend.benchmark(workload, repeat=3, warmup=1)
    stats = result.per_op()
    assert stats["add"]["count"] == 3
    assert stats["add"]["mean"] > 0

    op_times = [data["elapsed"] for _, data in result.tape.graph.nodes(data=True) if data.get("kind") == "op"]
    assert len(op_times) == 3
    assert all(t >= 0 for t in op_times)

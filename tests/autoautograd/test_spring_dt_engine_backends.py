from pytest import approx

from src.common.tensors.abstraction import AbstractTensor, BACKEND_REGISTRY
from src.common.tensors.numpy_backend import NumPyTensorOperations  # noqa: F401
from src.common.tensors.pure_backend import PurePythonTensorOperations  # noqa: F401
from src.common.tensors.autoautograd.spring_async_toy import (
    Node,
    Edge,
    SpringRepulsorSystem,
    SpringDtEngine,
)


def _run_system():
    n0 = Node(id=0, p=AbstractTensor.get_tensor([0.0, 0.0, 0.0]))
    n1 = Node(id=1, p=AbstractTensor.get_tensor([1.0, 0.0, 0.0]))
    n0.hist_p.append(n0.p.clone())
    n1.hist_p.append(n1.p.clone())
    e = Edge(key=(0, 1, "spring"), i=0, j=1, op_id="spring",
             l0=AbstractTensor.tensor(1.0), k=AbstractTensor.tensor(1.0))
    sys = SpringRepulsorSystem([n0, n1], [e], eta=0.0, gamma=1.0, dt=0.1)
    engine = SpringDtEngine(sys)
    engine.step(0.1)
    return [sys.nodes[0].p.tolist(), sys.nodes[1].p.tolist()]


def _run_backend(name, backend_cls):
    orig = BACKEND_REGISTRY.copy()
    try:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY[name] = backend_cls
        return _run_system()
    finally:
        BACKEND_REGISTRY.clear()
        BACKEND_REGISTRY.update(orig)


def test_spring_dt_engine_numpy_vs_pure_python():
    numpy_pos = _run_backend("numpy", NumPyTensorOperations)
    pure_pos = _run_backend("pure_python", PurePythonTensorOperations)
    for np_p, py_p in zip(numpy_pos, pure_pos):
        assert np_p == approx(py_p)

import os
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from src.common.tensors.autoautograd.spring_async_toy import (
        SpringRepulsorSystem,
        Node,
        build_dirichlet_neumann_pipeline,
    )
except Exception:
    pytest.skip("spring_async_toy optional deps missing", allow_module_level=True)

from src.common.tensors import AbstractTensor


def _make_sys(n_nodes: int) -> SpringRepulsorSystem:
    AT = AbstractTensor
    nodes = [
        Node(id=i, theta=0.0, p=AT.zeros(2, dtype=float), v=AT.zeros(2, dtype=float))
        for i in range(n_nodes)
    ]
    return SpringRepulsorSystem(nodes, [])


def test_feedback_topologies():
    AT = AbstractTensor
    X = AT.zeros((1, 2), dtype=float)
    y = AT.zeros((1, 2), dtype=float)

    sys_paired = _make_sys(2)
    build_dirichlet_neumann_pipeline(sys_paired, X, y, [0, 1], layers=0, feedback="paired")
    assert len(sys_paired.feedback_edges) == 2

    sys_full = _make_sys(2)
    build_dirichlet_neumann_pipeline(sys_full, X, y, [0, 1], layers=0, feedback="full")
    assert len(sys_full.feedback_edges) == 4

import os
import pytest

pytestmark = pytest.mark.xfail(
    reason="spring_async_toy transitioning to FluxSpring wrappers",
    strict=False,
)

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
    nodes = []
    for i in range(n_nodes):
        p = AT.zeros(2, dtype=float)
        param = AT.zeros(1, dtype=float)
        nodes.append(
            Node(
                id=i,
                param=param,
                p=p,
                v=AT.zeros(2, dtype=float),
                sphere=AbstractTensor.concat([p, param], dim=0),
            )
        )
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

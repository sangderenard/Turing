import time
import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.coo_matrix import COOMatrix
from src.common.tensors.abstract_convolution.laplace_nd import BuildGraphLaplace


def _random_graph(n: int, p: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < p
    weights = rng.random((n, n)) * mask
    weights = np.triu(weights, 1)
    return weights + weights.T


@pytest.mark.parametrize("n,p", [(50, 0.1), (100, 0.05)])
def test_build_graph_laplacian_random_dense_and_sparse(n: int, p: float) -> None:
    adj_np = _random_graph(n, p)
    adjacency_dense = AbstractTensor.get_tensor(adj_np)
    builder_dense = BuildGraphLaplace(adjacency_dense)
    start = time.time()
    L_dense, _, _ = builder_dense.build()
    elapsed_dense = time.time() - start

    idx = np.nonzero(adj_np)
    edge_index_np = np.vstack(idx).astype(np.int64)
    edge_weight_np = adj_np[idx]
    edge_index = AbstractTensor.get_tensor(edge_index_np)
    edge_weight = AbstractTensor.get_tensor(edge_weight_np)
    adjacency_sparse = COOMatrix(edge_index, edge_weight, (n, n))
    builder_sparse = BuildGraphLaplace(adjacency_sparse)
    start = time.time()
    L_sparse, _, _ = builder_sparse.build()
    elapsed_sparse = time.time() - start

    deg = adj_np.sum(axis=1)
    expected = np.diag(deg) - adj_np
    expected_t = AbstractTensor.get_tensor(expected, like=adjacency_dense)
    assert AbstractTensor.allclose(L_dense, expected_t)
    assert AbstractTensor.allclose(L_sparse, expected_t)
    assert elapsed_dense < 1.0
    assert elapsed_sparse < 1.0


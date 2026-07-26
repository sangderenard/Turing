import numpy as np

from src.common.tensors import AbstractTensor as AT


def test_abstract_linalg_solve_accepts_unbatched_matrix_rhs():
    matrix = AT.tensor([[2.0, 0.0], [0.0, 3.0]])
    rhs = AT.tensor([[4.0], [9.0]])
    solved = AT.linalg.solve(matrix, rhs)
    assert np.allclose(solved.tolist(), [[2.0], [3.0]])

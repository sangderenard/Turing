import numpy as np

from src.common.tensors import AbstractTensor as AT


def test_abstract_linalg_solve_accepts_unbatched_matrix_rhs():
    matrix = AT.tensor([[2.0, 0.0], [0.0, 3.0]])
    rhs = AT.tensor([[4.0], [9.0]])
    solved = AT.linalg.solve(matrix, rhs)
    assert np.allclose(solved.tolist(), [[2.0], [3.0]])


def test_tied_pivot_magnitudes_select_one_row_not_their_sum():
    """``col == col.max()`` marks every tie, which is not an argmax.

    With two equal magnitudes the mask held two ones and
    ``_masked_pivot_rows`` summed those rows instead of swapping one.  Equal
    magnitudes of the same sign survived that by accident -- the sum is still
    an invertible row operation, and ``solve`` applies it to the right-hand
    side too -- but ``|a| == |-a|`` cancels in the pivot column and the next
    division produced NaN.
    """

    matrix = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 4.0]])
    rhs = np.array([1.0, 2.0, 3.0])
    solved = np.asarray(
        AT.linalg.solve(AT.tensor(matrix.tolist()), AT.tensor(rhs.tolist())).tolist()
    )
    assert np.all(np.isfinite(solved))
    assert np.allclose(solved.reshape(-1), np.linalg.solve(matrix, rhs))


def test_solve_matches_numpy_over_tie_prone_integer_systems():
    """Small integer entries make exact magnitude ties common."""

    rng = np.random.default_rng(7)
    checked = 0
    for _trial in range(120):
        order = int(rng.integers(2, 6))
        matrix = rng.integers(-2, 3, size=(order, order)).astype(np.float64)
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        rhs = rng.integers(-3, 4, size=order).astype(np.float64)
        expected = np.linalg.solve(matrix, rhs)
        produced = np.asarray(
            AT.linalg.solve(
                AT.tensor(matrix.tolist()), AT.tensor(rhs.tolist())
            ).tolist()
        ).reshape(expected.shape)
        scale = max(1.0, float(np.max(np.abs(expected))))
        assert np.all(np.isfinite(produced)), (matrix.tolist(), rhs.tolist())
        assert np.max(np.abs(produced - expected)) / scale < 1e-8, (
            matrix.tolist(), rhs.tolist()
        )
        checked += 1
    assert checked > 80


def test_determinant_sign_survives_tied_pivots():
    rng = np.random.default_rng(11)
    checked = 0
    for _trial in range(60):
        order = int(rng.integers(2, 6))
        matrix = rng.integers(-2, 3, size=(order, order)).astype(np.float64)
        expected = np.linalg.det(matrix)
        if abs(expected) < 1e-8:
            continue
        produced = float(
            np.asarray(AT.linalg.det(AT.tensor(matrix.tolist())).tolist())
        )
        assert abs(produced - expected) / max(1.0, abs(expected)) < 1e-8, (
            matrix.tolist()
        )
        checked += 1
    assert checked > 30

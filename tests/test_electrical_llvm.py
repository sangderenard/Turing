from __future__ import annotations

import numpy as np

from src.compiler.electrical_llvm import compile_complex_nodal_solver


def test_batched_complex_nodal_solve_compiles_and_matches_numpy(tmp_path):
    solver = compile_complex_nodal_solver(
        tmp_path / "electrical", lanes=2, nodes=2,
    )
    admittance = np.asarray([
        [[4.0 + 1.0j, -1.0 - 0.5j],
         [-1.0 - 0.5j, 3.0 + 0.25j]],
        [[2.0 + 0.1j, -0.5 + 0.2j],
         [-0.5 + 0.2j, 1.5 - 0.3j]],
    ], dtype=np.complex128)
    current = np.asarray([
        [2.0 + 1.0j, 1.0 - 0.5j],
        [0.25 - 1.0j, 2.0 + 0.75j],
    ], dtype=np.complex128)

    produced = solver.solve(admittance, current)
    expected = np.linalg.solve(admittance, current[..., None]).squeeze(-1)

    assert solver.artifact.shortfalls == ()
    np.testing.assert_allclose(produced, expected, rtol=1.0e-12, atol=1.0e-12)

    next_current = current * (0.5 - 0.25j)
    next_produced = solver.solve(admittance, next_current)
    next_expected = np.linalg.solve(
        admittance, next_current[..., None],
    ).squeeze(-1)
    np.testing.assert_allclose(
        next_produced, next_expected, rtol=1.0e-12, atol=1.0e-12,
    )

    # Exercise an actual row interchange.  Diagonally dominant fixtures can
    # pass while the pivot search or the two-row publication is broken.
    pivoting_admittance = np.asarray([
        [[0.1 + 0.0j, 2.0 + 0.0j],
         [2.0 + 0.0j, 0.1 + 0.0j]],
        [[0.2 + 0.1j, 3.0 - 0.2j],
         [2.5 + 0.3j, 0.15 - 0.1j]],
    ], dtype=np.complex128)
    pivoting_current = np.asarray([
        [1.0 + 0.0j, 3.0 + 0.0j],
        [0.5 - 0.25j, 2.0 + 1.0j],
    ], dtype=np.complex128)
    pivoting_produced = solver.solve(pivoting_admittance, pivoting_current)
    pivoting_expected = np.linalg.solve(
        pivoting_admittance, pivoting_current[..., None],
    ).squeeze(-1)
    np.testing.assert_allclose(
        pivoting_produced, pivoting_expected, rtol=1.0e-12, atol=1.0e-12,
    )

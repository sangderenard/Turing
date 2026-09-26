"""Retained-loop compiler source for the canonical linear solve.

``linalg.solve`` is the semantic implementation.  This flat-buffer source is
its parametric native realization, following the same source/oracle contract
used by ``tensors.blas``. It uses partial-pivot Gauss-Jordan elimination so
matrix extent changes do not expand the solve into one SSA arithmetic cell per
scalar operation.
"""
from __future__ import annotations

import numpy as np


SOLVE_SOURCE = """
def solve(matrix, rhs, current_row, pivot_row_values, current_rhs, pivot_rhs, pivot_rows, pivot_magnitudes):
    for lane in range(__BATCH__):
        matrix_base = lane * __N__ * __N__
        vector_base = lane * __N__
        for pivot_column in range(__N__):
            pivot_state_base = matrix_base + pivot_column * __N__
            pivot_rows[pivot_state_base + pivot_column] = pivot_column * 1.0
            pivot_magnitudes[pivot_state_base + pivot_column] = abs(matrix[matrix_base + pivot_column * __N__ + pivot_column])
            for candidate_row in range(pivot_column + 1, __N__):
                candidate_magnitude = abs(matrix[matrix_base + candidate_row * __N__ + pivot_column])
                prior_state = pivot_state_base + candidate_row - 1
                candidate_state = pivot_state_base + candidate_row
                take_candidate = candidate_magnitude > pivot_magnitudes[prior_state]
                candidate_row_float = candidate_row * 1.0
                pivot_rows[candidate_state] = candidate_row_float if take_candidate else pivot_rows[prior_state]
                pivot_magnitudes[candidate_state] = candidate_magnitude if take_candidate else pivot_magnitudes[prior_state]
            pivot_row = pivot_rows[pivot_state_base + __N__ - 1]
            for swap_column in range(__N__):
                pivot_index = matrix_base + pivot_column * __N__ + swap_column
                swap_index = matrix_base + pivot_row * __N__ + swap_column
                current_row[vector_base + swap_column] = matrix[pivot_index]
                pivot_row_values[vector_base + swap_column] = matrix[swap_index]
            for swap_column in range(__N__):
                pivot_index = matrix_base + pivot_column * __N__ + swap_column
                swap_index = matrix_base + pivot_row * __N__ + swap_column
                matrix[pivot_index] = pivot_row_values[vector_base + swap_column]
                matrix[swap_index] = current_row[vector_base + swap_column]
            rhs_pivot_index = vector_base + pivot_column
            rhs_swap_index = vector_base + pivot_row
            current_rhs[rhs_pivot_index] = rhs[rhs_pivot_index]
            pivot_rhs[rhs_pivot_index] = rhs[rhs_swap_index]
            rhs[rhs_pivot_index] = pivot_rhs[rhs_pivot_index]
            rhs[rhs_swap_index] = current_rhs[rhs_pivot_index]
            pivot_value = matrix[matrix_base + pivot_column * __N__ + pivot_column]
            for normalization_column in range(pivot_column, __N__):
                normalization_index = matrix_base + pivot_column * __N__ + normalization_column
                matrix[normalization_index] = matrix[normalization_index] / pivot_value
            rhs[vector_base + pivot_column] = rhs[vector_base + pivot_column] / pivot_value
            for elimination_row in range(pivot_column + 1, __N__):
                factor = matrix[matrix_base + elimination_row * __N__ + pivot_column]
                for elimination_column in range(pivot_column, __N__):
                    target_index = matrix_base + elimination_row * __N__ + elimination_column
                    source_index = matrix_base + pivot_column * __N__ + elimination_column
                    matrix[target_index] = matrix[target_index] - factor * matrix[source_index]
                rhs[vector_base + elimination_row] = rhs[vector_base + elimination_row] - factor * rhs[vector_base + pivot_column]
            for elimination_row in range(__N__):
                eliminate_above = elimination_row < pivot_column
                factor = matrix[matrix_base + elimination_row * __N__ + pivot_column] if eliminate_above else 0.0
                for elimination_column in range(pivot_column, __N__):
                    target_index = matrix_base + elimination_row * __N__ + elimination_column
                    source_index = matrix_base + pivot_column * __N__ + elimination_column
                    matrix[target_index] = matrix[target_index] - factor * matrix[source_index]
                rhs[vector_base + elimination_row] = rhs[vector_base + elimination_row] - factor * rhs[vector_base + pivot_column]
    return rhs
"""


def solve_reference(matrix, rhs, *, batch, n):
    """Independent numerical oracle for the retained-loop publication."""
    matrices = np.asarray(matrix, dtype=float).reshape(int(batch), int(n), int(n))
    vectors = np.asarray(rhs, dtype=float).reshape(int(batch), int(n))
    return np.linalg.solve(matrices, vectors[..., None]).squeeze(-1)


__all__ = ["SOLVE_SOURCE", "solve_reference"]

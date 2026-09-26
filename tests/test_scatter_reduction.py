"""``AbstractTensor.scatter``: repeated destinations, and the reduce policy.

``scatter``'s docstring said "Scatter-add src into x", its entry in
``backward_registry`` declares ``y_i = x_i + src_j`` with the adjoint
``gsrc = g[index]``, and the GLSL backend's ``scatter_snippet`` accumulates
into ``accumulated`` on the GPU.  The CPU implementation did
``result[index] = old + src`` -- plain fancy-index assignment, which keeps
only the LAST write when an index repeats -- so it disagreed with its own
docstring, its own backward rule, and the other backend.

Measured before the fix: scattering ``[1, 1, 1, 5]`` into ``[0, 0, 0, 1]``
returned ``[1, 5, 0]`` where scatter-add gives ``[3, 5, 0]``.

``reduce`` uses the ``SegmentReduce`` vocabulary already declared in
``abstract_graph_core`` for ``coalesce_edges``, rather than a second
spelling of the same four policies.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


def _index(values):
    return AT.get_tensor(values).astype(AT.long_dtype_)


# --------------------------------------------------------------------------
# the defect
# --------------------------------------------------------------------------

def test_repeated_destinations_accumulate() -> None:
    """Three contributions to one slot sum; they do not overwrite."""
    out = AT.scatter(AT.zeros((3,)), _index([0, 0, 0, 1]),
                     AT.get_tensor([1., 1., 1., 5.]), dim=0)

    assert _np(out).tolist() == [3.0, 5.0, 0.0]


def test_accumulation_matches_numpy_add_at() -> None:
    """The reference semantics, spelled by the operation everyone means."""
    destinations = [2, 0, 2, 2, 1, 0]
    values = [1.5, -2.0, 0.25, 4.0, 7.0, 0.5]

    out = AT.scatter(AT.zeros((3,)), _index(destinations),
                     AT.get_tensor(values), dim=0)

    reference = np.zeros(3)
    np.add.at(reference, destinations, values)
    assert np.allclose(_np(out), reference)


def test_rows_accumulate_in_two_dimensions() -> None:
    """A row scatter sums whole rows, which is the mesh and graph case."""
    out = AT.scatter(AT.zeros((3, 2)), _index([0, 0, 2]),
                     AT.get_tensor([[1., 2.], [3., 4.], [5., 6.]]), dim=0)

    assert _np(out).tolist() == [[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]]


def test_the_existing_value_is_one_of_the_values_combined() -> None:
    """``y_i = x_i + sum(src)``: the registry's latex, as arithmetic."""
    out = AT.scatter(AT.get_tensor([10., 20., 30.]), _index([0, 0, 2]),
                     AT.get_tensor([1., 2., 3.]), dim=0)

    assert _np(out).tolist() == [13.0, 20.0, 33.0]


# --------------------------------------------------------------------------
# what must not have changed
# --------------------------------------------------------------------------

def test_unique_destinations_are_untouched_by_the_fix() -> None:
    """No repeat means one pass and the previous answer, entry for entry.

    Every caller in the tree that places values at distinct positions --
    the compression and JPEG paths -- is on this path.
    """
    out = AT.scatter(AT.get_tensor([10., 20., 30.]), _index([2, 0]),
                     AT.get_tensor([1., 2.]), dim=0)

    assert _np(out).tolist() == [12.0, 20.0, 31.0]


def test_a_slice_index_still_takes_the_single_pass() -> None:
    """Basic indexing is not a list of destinations and cannot repeat one."""
    out = AT.scatter(AT.zeros((4,)), slice(1, 3), AT.get_tensor([5., 6.]), dim=0)

    assert _np(out).tolist() == [0.0, 5.0, 6.0, 0.0]


def test_an_empty_index_changes_nothing() -> None:
    out = AT.scatter(AT.get_tensor([1., 2.]), _index([]), AT.get_tensor([]), dim=0)

    assert _np(out).tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------
# the reduce policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("policy,expected", [
    ("sum", [18.0, 5.0, 0.0]),     # 10+2+6      ; 0+5
    ("mean", [6.0, 2.5, 0.0]),     # (10+2+6)/3  ; (0+5)/2
    ("max", [10.0, 5.0, 0.0]),     # max(10,2,6) ; max(0,5)
    ("min", [2.0, 0.0, 0.0]),      # min(10,2,6) ; min(0,5)
])
def test_every_declared_reduction_policy(policy, expected) -> None:
    """All four ``SegmentReduce`` policies, each including ``x``'s own value.

    Including it is what makes the four consistent with one another: each
    one combines the standing value with every contribution under the same
    operator, so ``"sum"`` adds to it and ``"mean"`` averages with it.
    """
    out = AT.scatter(AT.get_tensor([10., 0., 0.]), _index([0, 0, 1]),
                     AT.get_tensor([2., 6., 5.]), dim=0, reduce=policy)

    assert _np(out).tolist() == expected


def test_an_undeclared_policy_is_refused() -> None:
    """A policy outside the vocabulary is an error, not a silent default."""
    with pytest.raises(ValueError, match="is not one of"):
        AT.scatter(AT.zeros((2,)), _index([0]), AT.get_tensor([1.]),
                   dim=0, reduce="nonsense")


def test_a_non_integer_index_cannot_carry_a_policy() -> None:
    """A slice has no destinations to reduce over, so asking is an error."""
    with pytest.raises(ValueError, match="needs an integer index"):
        AT.scatter(AT.zeros((4,)), slice(1, 3), AT.get_tensor([5., 6.]),
                   dim=0, reduce="max")

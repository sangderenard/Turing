"""AbstractTensor.interp and the index_select autograd fix it needed.

interp() is a real, base-operator-only 1-D linear interpolation primitive
(searchsorted + index_select + arithmetic, no backend hook) built for
porting a CWT inverse transform's upsampling step -- the existing
AbstractTensor.F.interpolate is an eager torch/PIL/scipy escape hatch
(no autograd, no backend dispatch, not SSA-lowerable) and the wrong shape
besides (2-D image resize, not 1-D signal interpolation).

Building it surfaced a real, silent gap: index_select() never called
_pre_autograd, so any gradient reaching one anywhere in a forward graph
stopped there with no error. That's fixed here too, reusing index_adjoint
(the same repeated-index-accumulating adjoint __getitem__'s own fancy-
indexing backward already uses), verified independently of interp().
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors import AbstractTensor as AT


def _np(t) -> np.ndarray:
    return np.asarray(t.data if hasattr(t, "data") else t)


# --------------------------------------------------------------------------
# index_select: the autograd gap, fixed
# --------------------------------------------------------------------------

def test_index_select_records_an_autograd_node():
    x = AT.get_tensor(np.array([1.0, 2.0, 3.0]))
    x.requires_grad_(True)
    idx = AT.get_tensor(np.array([0, 2]))
    out = x.index_select(0, idx)
    out.sum().backward()
    assert x.grad is not None


def test_index_select_gradient_matches_hand_derivative():
    x = AT.get_tensor(np.array([10.0, 20.0, 30.0, 40.0]))
    x.requires_grad_(True)
    idx = AT.get_tensor(np.array([1, 3]))
    out = x.index_select(0, idx)
    (out * out).sum().backward()  # d/dx_i (x_i^2) = 2*x_i, only at selected i
    expected = np.array([0.0, 40.0, 0.0, 80.0])
    assert np.allclose(_np(x.grad), expected)


def test_index_select_repeated_indices_accumulate_not_overwrite():
    """The bug this would have hit if built on gather's backward instead of
    index_adjoint: gather's backward assigns (gx[idx] = g), which drops all
    but the last contribution for a repeated index. index_adjoint sorts and
    prefix-sums, so repeated indices accumulate correctly."""

    x = AT.get_tensor(np.array([0.0, 0.0, 0.0]))
    x.requires_grad_(True)
    idx = AT.get_tensor(np.array([1, 1, 1]))  # index 1 used three times
    out = x.index_select(0, idx)
    out.sum().backward()
    assert np.allclose(_np(x.grad), [0.0, 3.0, 0.0])


# --------------------------------------------------------------------------
# interp: values match numpy exactly, including boundary extrapolation
# --------------------------------------------------------------------------

def test_interp_matches_numpy_including_boundaries():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 10.0, 5.0, 20.0, 15.0])
    xq = np.array([-1.0, 0.0, 0.5, 1.0, 2.5, 4.0, 5.0])

    got = AT.interp(AT.get_tensor(xq), AT.get_tensor(x), AT.get_tensor(y))
    assert np.allclose(_np(got), np.interp(xq, x, y))


def test_interp_at_exact_sample_points_returns_the_sample_value():
    x = np.array([0.0, 2.0, 4.0, 6.0])
    y = np.array([1.0, -3.0, 8.0, 0.5])
    got = AT.interp(AT.get_tensor(x.copy()), AT.get_tensor(x), AT.get_tensor(y))
    assert np.allclose(_np(got), y)


def test_interp_repeated_sample_point_does_not_produce_nan():
    """A degenerate zero-width segment (repeated x) must not divide by zero."""

    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 5.0, 5.0, 10.0])
    xq = np.array([0.5, 1.0, 1.5])
    got = _np(AT.interp(AT.get_tensor(xq), AT.get_tensor(x), AT.get_tensor(y)))
    assert not np.any(np.isnan(got))
    assert np.allclose(got, np.interp(xq, x, y))


# --------------------------------------------------------------------------
# interp: differentiable in y (the point of fixing index_select)
# --------------------------------------------------------------------------

def test_interp_is_differentiable_in_y_at_a_query_between_two_samples():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = AT.get_tensor(np.array([0.0, 10.0, 5.0, 20.0, 15.0]))
    y.requires_grad_(True)

    out = AT.interp(AT.get_tensor(np.array([1.5])), AT.get_tensor(x), y)
    out.sum().backward()

    grad = _np(y.grad)
    # 1.5 is exactly midway between samples 1 and 2: dy/dy1 = dy/dy2 = 0.5,
    # zero everywhere else, matching the interpolation weights exactly.
    assert np.allclose(grad, [0.0, 0.5, 0.5, 0.0, 0.0])


def test_interp_gradient_sums_to_one_across_probe_points():
    """sum(dinterp/dy_i) over the two bracketing samples must be exactly 1 --
    a linear interpolant's output is an affine (weight-1) combination of its
    two neighbours at any interior query point."""

    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = AT.get_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    y.requires_grad_(True)

    out = AT.interp(AT.get_tensor(np.array([0.3, 1.7, 2.9])), AT.get_tensor(x), y)
    out.sum().backward()

    # Each of the three probes contributes weight 1 total, split over its
    # two neighbours -- three probes, so the total gradient sums to 3.
    assert _np(y.grad).sum() == pytest.approx(3.0)

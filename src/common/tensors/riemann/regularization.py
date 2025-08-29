from __future__ import annotations

"""Regularization helpers for Riemann grid modules."""

from typing import Optional, Dict

from ..abstraction import AbstractTensor


def smooth_bins(bins: AbstractTensor, lam: float) -> AbstractTensor:
    """Penalise non‑smooth bin assignments via finite differences.

    Parameters
    ----------
    bins : AbstractTensor
        Tensor of shape ``(Cin, D, H, W)`` representing per‑voxel bin weights.
    lam : float
        Scaling coefficient. ``0`` disables the penalty.

    Returns
    -------
    AbstractTensor
        Scalar tensor containing the Laplacian penalty.
    """
    if lam == 0.0:
        return AbstractTensor.get_tensor([0.0]).sum()

    diffs = []
    if bins.shape[1] > 1:
        diffs.append(bins[:, 1:, :, :] - bins[:, :-1, :, :])
    if bins.shape[2] > 1:
        diffs.append(bins[:, :, 1:, :] - bins[:, :, :-1, :])
    if bins.shape[3] > 1:
        diffs.append(bins[:, :, :, 1:] - bins[:, :, :, :-1])

    penalty = AbstractTensor.get_tensor([0.0]).sum()
    for d in diffs:
        penalty = penalty + (d * d).sum()
    return penalty * lam


def weight_decay(
    casting: Optional[object],
    conv: Optional[object],
    post: Optional[object],
    coeffs: Dict[str, float],
) -> AbstractTensor:
    """Apply L2 penalties to parameter groups with separate coefficients.

    Parameters
    ----------
    casting, conv, post : modules or ``None``
        Components whose parameters will be regularised.
    coeffs : dict
        Mapping with optional ``"pre"``, ``"conv"`` and ``"post"`` entries
        specifying the decay strength for each group.
    """
    penalty = AbstractTensor.get_tensor([0.0]).sum()

    pre_coef = coeffs.get("pre", 0.0)
    if casting is not None and pre_coef != 0.0 and getattr(casting, "pre_linear", None) is not None:
        for p in casting.pre_linear.parameters():
            penalty = penalty + pre_coef * (p * p).sum()

    conv_coef = coeffs.get("conv", 0.0)
    if conv is not None and conv_coef != 0.0:
        for p in conv.parameters():
            penalty = penalty + conv_coef * (p * p).sum()

    post_coef = coeffs.get("post", 0.0)
    if post is not None and post_coef != 0.0:
        for p in post.parameters():
            penalty = penalty + post_coef * (p * p).sum()

    return penalty

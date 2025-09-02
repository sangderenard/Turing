"""
pca_layer.py
-------------

Light‑weight layer that projects data onto its principal components.  The
layer performs an on‑the‑fly PCA fit on the first forward pass if no basis has
been supplied.  It is intentionally stateless with respect to optimisation and
therefore exposes no trainable parameters.

This resides in the abstract_nn namespace so it can be chained with other
layers in demo pipelines.
"""

from __future__ import annotations

from ..abstraction import AbstractTensor as AT


class PCATransformLayer:
    """Reduce the trailing feature dimension via PCA.

    Parameters
    ----------
    n_components : int, optional
        Number of principal components to retain.  Defaults to ``3``.
    like : AT, optional
        Tensor like object for intermediate tensor creation.
    """

    def __init__(self, n_components: int = 3, like: AT | None = None):
        self.n_components = n_components
        self.like = like or AT.get_tensor
        self._mean = None
        self._components = None

    def _fit(self, x):
        x2d = x.reshape(-1, x.shape[-1])
        self._mean = x2d.mean(dim=0)
        x_centered = x2d - self._mean
        cov = x_centered.swapaxes(-1, -2) @ x_centered / AT.get_tensor(x2d.shape[0])
        evals, evecs = AT.linalg.eigh(cov)
        self._components = evecs[:, -self.n_components :]

    def forward(self, x):
        if self._components is None or self._mean is None:
            self._fit(x)
        x2d = x.reshape(-1, x.shape[-1])
        transformed = (x2d - self._mean) @ self._components
        return transformed.reshape(*x.shape[:-1], self.n_components)

    def parameters(self):  # pragma: no cover - no trainable params
        return []


from __future__ import annotations

"""
ManifoldPackage: canonical geometry bundle for Riemannian ops.

Responsibilities (future work):
- Accept a Transform and GridDomain
- Build Laplace package via BuildLaplace3D (metric, operators)
- Optionally compute a small set of LB eigenpairs (k modes)
- Cache helpers (e.g., mass/stiffness, neighbor stencils)
"""

from typing import Any, Optional
from ..abstraction import AbstractTensor


class ManifoldPackage:
    def __init__(
        self,
        transform: Any,
        grid_domain: Any,
        *,
        laplace_kwargs: Optional[dict] = None,
        num_eigenpairs: int = 0,
    ) -> None:
        self.transform = transform
        self.grid_domain = grid_domain
        self.laplace_kwargs = laplace_kwargs or {}
        self.num_eigenpairs = int(num_eigenpairs)
        self.package = None  # to be built lazily
        self.evals = None
        self.evecs = None

    def build(self) -> None:
        """Build the Laplace package (and optionally eigenpairs)."""
        from ..abstract_convolution.laplace_nd import BuildLaplace3D

        builder = BuildLaplace3D(
            grid_domain=self.grid_domain,
            metric_tensor_func=self.transform.metric_tensor_func,
            **self.laplace_kwargs,
        )
        # Build dense Laplacian for small grids to enable AbstractTensor eigh
        laplace_dense, _, package = builder.build_general_laplace(
            self.grid_domain.U,
            self.grid_domain.V,
            self.grid_domain.W,
            dense=True,
            return_package=True,
        )
        self.package = package
        # Eigenpairs (optional)
        if self.num_eigenpairs and isinstance(laplace_dense, AbstractTensor):
            # Compute full symmetric eigendecomposition (ascending eigenvalues)
            w, V = AbstractTensor.linalg.eigh(laplace_dense)
            # Keep the first num_eigenpairs (smallest magnitude modes)
            k = max(1, int(self.num_eigenpairs))
            self.evals = w[:k]
            self.evecs = V[:, :k]

    def laplace_package(self) -> dict:
        if self.package is None:
            self.build()
        return self.package

    def eigenpairs(self) -> tuple[AbstractTensor, AbstractTensor] | None:
        if self.package is None:
            self.build()
        if self.evals is None or self.evecs is None:
            return None
        return self.evals, self.evecs

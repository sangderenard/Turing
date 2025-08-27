"""
riemann_convolutional.py
-----------------------

Defines a RiemannConvolutional3D layer: a metric-aware 3D convolutional layer that builds its geometry and Laplacian package from a user-supplied PCANDTransform (or compatible transform), using the canonical GridDomain pipeline. This is the intended, correct pattern for geometry-driven convolution.
"""

from .laplace_nd import BuildLaplace3D, GridDomain
from .ndpca3conv import NDPCA3Conv3d
from ..abstraction import AbstractTensor

class RiemannConvolutional3D:

    def parameters(self):
        params = []
        if hasattr(self.conv, 'parameters') and callable(self.conv.parameters):
            params.extend(self.conv.parameters())
        # laplace_package may be a dict of modules
        if isinstance(self.laplace_package, dict):
            for v in self.laplace_package.values():
                if hasattr(v, 'parameters') and callable(v.parameters):
                    params.extend(v.parameters())
        elif hasattr(self.laplace_package, 'parameters') and callable(self.laplace_package.parameters):
            params.extend(self.laplace_package.parameters())
        return params

    def zero_grad(self):
        if hasattr(self.conv, 'zero_grad') and callable(self.conv.zero_grad):
            self.conv.zero_grad()
        if isinstance(self.laplace_package, dict):
            for v in self.laplace_package.values():
                if hasattr(v, 'zero_grad') and callable(v.zero_grad):
                    v.zero_grad()
        elif hasattr(self.laplace_package, 'zero_grad') and callable(self.laplace_package.zero_grad):
            self.laplace_package.zero_grad()
    """
    Metric-aware 3D convolutional layer using a Riemannian geometry pipeline.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    grid_shape : (Nu, Nv, Nw)
    transform : PCANDTransform or compatible
    boundary_conditions : tuple[str, str, str, str, str, str]
    k : int (number of principal directions)
    eig_from : 'g' or 'inv_g'
    pointwise : bool
    laplace_kwargs : dict (optional, for BuildLaplace3D)
    """
    def __init__(self, in_channels, out_channels, grid_shape, transform, boundary_conditions=("dirichlet",)*6, k=3, eig_from="g", pointwise=True, laplace_kwargs=None):
        Nu, Nv, Nw = grid_shape
        self.transform = transform
        self.grid_domain = GridDomain(
            AbstractTensor.linspace(-1.0, 1.0, Nu).reshape(Nu, 1, 1) * AbstractTensor.ones((1, Nv, Nw)),
            AbstractTensor.linspace(-1.0, 1.0, Nv).reshape(1, Nv, 1) * AbstractTensor.ones((Nu, 1, Nw)),
            AbstractTensor.linspace(-1.0, 1.0, Nw).reshape(1, 1, Nw) * AbstractTensor.ones((Nu, Nv, 1)),
            grid_boundaries=(True,)*6,
            transform=transform,
            coordinate_system="rectangular"
        )
        self.laplace_kwargs = laplace_kwargs or {}
        self.laplace_package = self._build_laplace_package(boundary_conditions)
        self.conv = NDPCA3Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            like=AbstractTensor.zeros((1, in_channels, Nu, Nv, Nw)),
            grid_shape=(Nu, Nv, Nw),
            boundary_conditions=boundary_conditions,
            k=k,
            eig_from=eig_from,
            pointwise=pointwise,
        )

    def _build_laplace_package(self, boundary_conditions):
        builder = BuildLaplace3D(
            grid_domain=self.grid_domain,
            wave_speed=343,
            precision=getattr(AbstractTensor, "float_dtype_", None) or self.grid_domain.U.dtype,
            resolution=self.grid_domain.U.shape[0],
            metric_tensor_func=self.transform.metric_tensor_func,
            boundary_conditions=boundary_conditions,
            artificial_stability=1e-10,
            device=getattr(self.grid_domain.U, "device", None),
            **self.laplace_kwargs
        )
        _, _, package = builder.build_general_laplace(self.grid_domain.U, self.grid_domain.V, self.grid_domain.W, return_package=True)
        return package

    def forward(self, x):
        """
        x: (B, C, Nu, Nv, Nw)
        Returns: (B, out_channels, Nu, Nv, Nw)
        """
        return self.conv.forward(x, package=self.laplace_package)

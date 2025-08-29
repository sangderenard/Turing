"""
Riemannian suite (scaffold)
---------------------------

Future modules for true Riemannian convolution and operators:
- ManifoldPackage: wraps Transform → GridDomain → BuildLaplace3D + eigenpairs
- SpectralConv3D: LB‑spectral convolution with learned multipliers
- GeodesicConv3D: local geodesic kernel aggregation
- ParallelTransport: feature/frame transport utilities
- HeatKernel3D: diffusion operators (e.g., e^(−tL))

This package is scaffolded; implementations will be added incrementally.
"""

from .manifold import ManifoldPackage
from .spectral import SpectralConv3D
from .geodesic import GeodesicConv3D
from .transport import ParallelTransport
from .heat import HeatKernel3D
from .grid_block import RiemannGridBlock
from .regularization import smooth_bins, weight_decay


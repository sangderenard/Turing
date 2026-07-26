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
from .adaptive_triangulation import (
    AdaptiveSurfaceTriangulator,
    RefinementCertificate,
    TriangulationGeneration,
    TriangulationTolerance,
)
from .mesh_laplace import (
    CotangentTopology,
    CotangentMeshGeometry,
    MeshLaplaceResult,
    build_cotangent_topology,
    build_cotangent_geometry,
    mesh_laplace_beltrami,
)
from .mesh_transform import TriangulatedSurfaceTransform
from .abstract_mesh_laplace import abstract_mesh_laplace

__all__ = [
    "AdaptiveSurfaceTriangulator",
    "CotangentMeshGeometry",
    "CotangentTopology",
    "GeodesicConv3D",
    "HeatKernel3D",
    "ManifoldPackage",
    "MeshLaplaceResult",
    "ParallelTransport",
    "RefinementCertificate",
    "RiemannGridBlock",
    "SpectralConv3D",
    "TriangulationGeneration",
    "TriangulationTolerance",
    "TriangulatedSurfaceTransform",
    "build_cotangent_geometry",
    "build_cotangent_topology",
    "abstract_mesh_laplace",
    "mesh_laplace_beltrami",
    "smooth_bins",
    "weight_decay",
]


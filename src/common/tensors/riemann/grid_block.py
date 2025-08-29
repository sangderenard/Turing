from __future__ import annotations

"""Riemannian grid processing block.

This small module bridges geometry packages from :mod:`geometry_factory`
with the metric-steered :class:`~src.common.tensors.abstract_convolution.ndpca3conv.NDPCA3Conv3d`
convolution.  The block optionally applies a per-voxel "casting" stage that
includes a pre-channel projection, simple FiLM modulation by grid coordinates
and an optional post-convolution linear projection.

All tensor manipulations remain within :class:`AbstractTensor` so the block can
operate on any registered backend.
"""

from typing import Any, Dict, List, Optional

from ..abstraction import AbstractTensor
from ..abstract_nn.core import Linear
from ..abstract_convolution.ndpca3conv import NDPCA3Conv3d
from .geometry_factory import build_geometry


class _FiLM:
    """Minimal feature-wise linear modulation using grid coordinates."""

    def __init__(self, in_dim: int, out_dim: int, like: AbstractTensor) -> None:
        self.linear = Linear(in_dim, 2 * out_dim, like=like)
        self.out_dim = out_dim

    def parameters(self) -> List[AbstractTensor]:
        return self.linear.parameters()

    def forward(self, coords: AbstractTensor, x: AbstractTensor) -> AbstractTensor:
        """Apply modulation ``x * gamma + beta`` where ``gamma`` and ``beta``
        are derived from ``coords`` via a learned linear map."""
        D, H, W, C = coords.shape
        flat = coords.reshape(D * H * W, C)
        gamma_beta = self.linear.forward(flat)
        gamma_beta = gamma_beta.reshape(D, H, W, 2 * self.out_dim)
        gamma = gamma_beta[..., : self.out_dim]
        beta = gamma_beta[..., self.out_dim :]
        gamma = gamma.reshape(1, self.out_dim, D, H, W)
        beta = beta.reshape(1, self.out_dim, D, H, W)
        return x * gamma + beta


class RiemannGridBlock:
    """Composite block combining casting and metric‑steered convolution."""

    def __init__(
        self,
        *,
        conv: NDPCA3Conv3d,
        package: Dict[str, Any],
        pre_linear: Optional[Linear] = None,
        film: Optional[_FiLM] = None,
        coords: Optional[AbstractTensor] = None,
        bin_map: Optional[Any] = None,
        post_linear: Optional[Linear] = None,
    ) -> None:
        self.conv = conv
        self.package = package
        self.pre_linear = pre_linear
        self.film = film
        self.coords = coords
        self.bin_map = bin_map
        self.post_linear = post_linear

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "RiemannGridBlock":
        """Build a :class:`RiemannGridBlock` from ``config``.

        The configuration must contain a ``"geometry"`` entry which is passed to
        :func:`geometry_factory.build_geometry`.  Convolution options live under
        ``"conv"`` and follow :class:`NDPCA3Conv3d`'s constructor.  Optional
        ``pre_linear``, ``film`` and ``post_linear`` dictionaries control the
        casting modules.
        """

        geom_cfg = config.get("geometry", {})
        transform, grid, package = build_geometry(geom_cfg)

        # Grid coordinates (D,H,W,3) for FiLM modulation
        AT = AbstractTensor
        coords = AT.stack([grid.U, grid.V, grid.W], dim=-1)

        like = AT.get_tensor([0.0])

        pre_cfg = config.get("pre_linear")
        pre = None
        if pre_cfg is not None:
            pre = Linear(pre_cfg["in_dim"], pre_cfg["out_dim"], like=like)

        film_cfg = config.get("film")
        film = None
        if film_cfg is not None:
            film = _FiLM(film_cfg.get("in_dim", 3), film_cfg.get("out_dim", pre_cfg["out_dim"] if pre_cfg else config["conv"]["in_channels"]), like=like)

        bin_map = package.get("bin_map") if isinstance(package, dict) else None

        conv_cfg = config.get("conv", {})
        grid_shape = grid.U.shape
        conv = NDPCA3Conv3d(
            conv_cfg["in_channels"],
            conv_cfg["out_channels"],
            like=like,
            grid_shape=grid_shape,
            boundary_conditions=conv_cfg.get("boundary_conditions", ("dirichlet",) * 6),
            k=conv_cfg.get("k", 3),
            eig_from=conv_cfg.get("metric_source", "g"),
            pointwise=conv_cfg.get("pointwise", True),
        )

        post_cfg = config.get("post_linear")
        post = None
        if post_cfg is not None:
            post = Linear(post_cfg["in_dim"], post_cfg["out_dim"], like=like)

        return cls(
            conv=conv,
            package=package,
            pre_linear=pre,
            film=film,
            coords=coords,
            bin_map=bin_map,
            post_linear=post,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def parameters(self) -> List[AbstractTensor]:
        params: List[AbstractTensor] = []
        for mod in (self.pre_linear, self.film, self.conv, self.post_linear):
            if mod is None:
                continue
            if hasattr(mod, "parameters"):
                params.extend(mod.parameters())
        return params

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        """Apply casting modules and metric‑steered convolution."""
        B, C, D, H, W = x.shape

        if self.pre_linear is not None:
            z = x.reshape(B * D * H * W, C)
            z = self.pre_linear.forward(z)
            x = z.reshape(B, -1, D, H, W)
            C = x.shape[1]

        if self.film is not None and self.coords is not None:
            x = self.film.forward(self.coords, x)

        y = self.conv.forward(x, package=self.package)

        if self.post_linear is not None:
            B, Cout, D, H, W = y.shape
            z = y.reshape(B * D * H * W, Cout)
            z = self.post_linear.forward(z)
            y = z.reshape(B, -1, D, H, W)

        return y

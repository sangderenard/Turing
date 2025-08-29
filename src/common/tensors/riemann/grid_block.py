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


def soft_assign(*args: Any, **kwargs: Any) -> AbstractTensor:
    """Placeholder for a differentiable assignment routine.

    The eventual implementation will scatter values to an irregular grid while
    preserving gradients.  It is not yet implemented and currently raises
    :class:`NotImplementedError`.
    """

    raise NotImplementedError("soft_assign is not implemented")


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


class _Casting:
    """Optional casting stage applied before convolution."""

    def __init__(
        self,
        *,
        mode: str,
        like: AbstractTensor,
        in_channels: int,
        grid,
        film: bool = False,
        coords_mode: Optional[str] = None,
        inject_coords: bool = False,
    ) -> None:
        self.mode = mode
        self.inject_coords = inject_coords

        D, H, W = grid.U.shape
        self.pre_linear: Optional[Linear] = None

        if mode == "pre_linear":
            cin = in_channels
            size = cin * D * H * W
            self.pre_linear = Linear(size, size, like=like)
        elif mode == "fixed":
            pass
        elif mode == "soft_assign":
            # Actual implementation delegated to ``soft_assign`` when used.
            self.pre_linear = None
        else:
            raise ValueError(f"Unknown casting mode: {mode}")

        # Coordinate preparation -------------------------------------------------
        self.coords: Optional[AbstractTensor] = None
        self.coords_as_channels: Optional[AbstractTensor] = None
        coord_dim = 0
        if coords_mode is not None or film or inject_coords:
            base_ch = AbstractTensor.stack([grid.U, grid.V, grid.W], dim=0)  # (3,D,H,W)
            if coords_mode == "fourier":
                sin = base_ch.sin()
                cos = base_ch.cos()
                base_ch = AbstractTensor.cat([sin, cos], dim=0)  # (6,D,H,W)
            self.coords_as_channels = base_ch.unsqueeze(0)  # (1,C,D,H,W)
            coords = base_ch
            coords = coords.swapaxes(0, 1)  # (D,C,H,W)
            coords = coords.swapaxes(1, 2)  # (D,H,C,W)
            coords = coords.swapaxes(2, 3)  # (D,H,W,C)
            self.coords = coords
            coord_dim = coords.shape[-1]

        # FiLM modulation --------------------------------------------------------
        self.film: Optional[_FiLM] = None
        if film:
            if self.coords is None:
                raise ValueError("FiLM requires coordinates")
            self.film = _FiLM(coord_dim, in_channels, like=like)

    # -- API --------------------------------------------------------------------
    def parameters(self) -> List[AbstractTensor]:
        params: List[AbstractTensor] = []
        if self.pre_linear is not None:
            params.extend(self.pre_linear.parameters())
        if self.film is not None:
            params.extend(self.film.parameters())
        return params

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        if self.mode == "soft_assign":
            return soft_assign(x)

        if self.inject_coords and self.coords_as_channels is not None:
            x = AbstractTensor.cat([x, self.coords_as_channels], dim=1)

        if self.pre_linear is not None:
            B, C, D, H, W = x.shape
            z = x.reshape(B, C * D * H * W)
            z = self.pre_linear.forward(z)
            x = z.reshape(B, C, D, H, W)

        if self.film is not None and self.coords is not None:
            x = self.film.forward(self.coords, x)

        return x


class RiemannGridBlock:
    """Composite block combining casting and metric‑steered convolution."""

    def __init__(
        self,
        *,
        conv: NDPCA3Conv3d,
        package: Dict[str, Any],
        casting: Optional[_Casting] = None,
        bin_map: Optional[Any] = None,
        post_linear: Optional[Linear] = None,
    ) -> None:
        self.conv = conv
        self.package = package
        self.casting = casting
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
        casting behaviour is controlled by the ``"casting"`` dictionary.
        """

        geom_cfg = config.get("geometry", {})
        transform, grid, package = build_geometry(geom_cfg)

        AT = AbstractTensor
        like = AT.get_tensor([0.0])

        casting_cfg = config.get("casting")
        casting = None
        conv_cfg = config.get("conv", {})
        if casting_cfg is not None:
            casting = _Casting(
                mode=casting_cfg.get("mode", "fixed"),
                like=like,
                in_channels=conv_cfg.get("in_channels", 1),
                grid=grid,
                film=casting_cfg.get("film", False),
                coords_mode=casting_cfg.get("coords"),
                inject_coords=casting_cfg.get("inject_coords", False),
            )

        bin_map = package.get("bin_map") if isinstance(package, dict) else None

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
            casting=casting,
            bin_map=bin_map,
            post_linear=post,
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def parameters(self) -> List[AbstractTensor]:
        params: List[AbstractTensor] = []
        for mod in (self.casting, self.conv, self.post_linear):
            if mod is None:
                continue
            if hasattr(mod, "parameters"):
                params.extend(mod.parameters())
        return params

    def forward(self, x: AbstractTensor) -> AbstractTensor:
        """Apply casting modules and metric‑steered convolution."""
        if self.casting is not None:
            x = self.casting.forward(x)

        y = self.conv.forward(x, package=self.package)

        if self.post_linear is not None:
            B, Cout, D, H, W = y.shape
            z = y.reshape(B * D * H * W, Cout)
            z = self.post_linear.forward(z)
            y = z.reshape(B, -1, D, H, W)

        return y

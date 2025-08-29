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
from .regularization import smooth_bins, weight_decay


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration for :class:`RiemannGridBlock`.

    Parameters
    ----------
    config:
        Configuration dictionary to validate.

    Raises
    ------
    TypeError
        If ``config`` or any sub-entries have incorrect types.
    ValueError
        If required keys are missing or values are out of the accepted range.
    """

    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    # Geometry -----------------------------------------------------------
    geom = config.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError("'geometry' section is required")
    if not isinstance(geom.get("key"), str):
        raise ValueError("geometry.key must be a string")
    if "grid_shape" in geom:
        gs = geom["grid_shape"]
        if (
            not isinstance(gs, (tuple, list))
            or len(gs) != 3
            or not all(isinstance(n, int) for n in gs)
        ):
            raise TypeError("geometry.grid_shape must be a 3-tuple of ints")

    # Convolution --------------------------------------------------------
    conv = config.get("conv")
    if not isinstance(conv, dict):
        raise ValueError("'conv' section is required")
    for k in ("in_channels", "out_channels"):
        if not isinstance(conv.get(k), int):
            raise ValueError(f"conv.{k} must be an int")
    if "boundary_conditions" in conv:
        bc = conv["boundary_conditions"]
        if not isinstance(bc, (tuple, list)) or len(bc) != 6:
            raise TypeError("conv.boundary_conditions must be a 6-element sequence")
    if "pointwise" in conv and not isinstance(conv["pointwise"], bool):
        raise TypeError("conv.pointwise must be a bool")
    if "k" in conv and not isinstance(conv["k"], int):
        raise TypeError("conv.k must be an int")
    if "metric_source" in conv and not isinstance(conv["metric_source"], str):
        raise TypeError("conv.metric_source must be a string")
    if "stencil" in conv:
        st = conv["stencil"]
        if not isinstance(st, dict):
            raise TypeError("conv.stencil must be a dict")
        if "offsets" in st and not isinstance(st["offsets"], (tuple, list)):
            raise TypeError("stencil.offsets must be a sequence")
        if "length" in st and not isinstance(st["length"], int):
            raise TypeError("stencil.length must be an int")
        if "normalize" in st and not isinstance(st["normalize"], bool):
            raise TypeError("stencil.normalize must be a bool")

    # Casting ------------------------------------------------------------
    casting = config.get("casting")
    if casting is not None:
        if not isinstance(casting, dict):
            raise TypeError("casting must be a dict")
        mode = casting.get("mode", "fixed")
        if mode not in {"pre_linear", "fixed", "soft_assign"}:
            raise ValueError("casting.mode must be 'pre_linear', 'fixed' or 'soft_assign'")
        if "film" in casting and not isinstance(casting["film"], bool):
            raise TypeError("casting.film must be a bool")
        if "coords" in casting and casting["coords"] is not None and not isinstance(
            casting["coords"], str
        ):
            raise TypeError("casting.coords must be a string or None")
        if "inject_coords" in casting and not isinstance(casting["inject_coords"], bool):
            raise TypeError("casting.inject_coords must be a bool")
        if "map" in casting and casting["map"] not in {"1to1", "row_major", "normalized_span"}:
            raise ValueError(
                "casting.map must be '1to1', 'row_major' or 'normalized_span'"
            )

    # Post-linear --------------------------------------------------------
    post = config.get("post_linear")
    if post is not None:
        if not isinstance(post, dict):
            raise TypeError("post_linear must be a dict")
        for k in ("in_dim", "out_dim"):
            if not isinstance(post.get(k), int):
                raise ValueError(f"post_linear.{k} must be an int")

    # Regularization -----------------------------------------------------
    reg = config.get("regularization")
    if reg is not None:
        if not isinstance(reg, dict):
            raise TypeError("regularization must be a dict")
        if "smooth_bins" in reg and not isinstance(reg["smooth_bins"], (int, float)):
            raise TypeError("regularization.smooth_bins must be numeric")
        if "weight_decay" in reg:
            wd = reg["weight_decay"]
            if not isinstance(wd, dict):
                raise TypeError("weight_decay must be a dict")
            for v in wd.values():
                if not isinstance(v, (int, float)):
                    raise TypeError("weight_decay coefficients must be numeric")



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
        map_strategy: str = "row_major",
    ) -> None:
        """Create a casting module.

        Args:
            mode: Casting mode. ``"pre_linear"`` enables a learned linear
                projection before convolution.
            like: Backend tensor to mirror for parameter creation.
            in_channels: Number of input channels (``C_in``).
            grid: Geometry grid supplying ``(D,H,W)`` dimensions.
            film: Whether to apply FiLM modulation.
            coords_mode: Optional coordinate encoding strategy.
            inject_coords: If ``True``, append coordinates as channels.
            map_strategy: How to reshape the flattened ``pre_linear`` output
                back into ``(C_in,D,H,W)``. Available strategies are:

                - ``"1to1"`` – Treats each voxel as a contiguous block of
                  ``C_in`` channels (voxel-major order).
                - ``"row_major"`` – Standard C-major/row-major layout where all
                  positions of channel 0 appear before channel 1. This is the
                  default to preserve existing behaviour.
                - ``"normalized_span"`` – Currently mirrors ``"row_major"`` but
                  is reserved for future schemes based on normalized index
                  spans.
        """
        self.mode = mode
        self.inject_coords = inject_coords
        self.map_strategy = map_strategy

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
            x = self._reshape_pre_linear(z, B, C, D, H, W)

        if self.film is not None and self.coords is not None:
            x = self.film.forward(self.coords, x)

        return x

    # ------------------------------------------------------------------
    def _reshape_pre_linear(
        self, z: AbstractTensor, B: int, C: int, D: int, H: int, W: int
    ) -> AbstractTensor:
        """Map flattened ``pre_linear`` output back to ``(B,C,D,H,W)``.

        The mapping is controlled by ``self.map_strategy``.
        """

        if self.map_strategy == "row_major" or self.map_strategy == "normalized_span":
            return z.reshape(B, C, D, H, W)
        if self.map_strategy == "1to1":
            z = z.reshape(B, D, H, W, C)
            z = z.swapaxes(4, 3)
            z = z.swapaxes(3, 2)
            z = z.swapaxes(2, 1)
            return z
        raise ValueError(f"Unknown mapping strategy: {self.map_strategy}")


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
        regularization: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conv = conv
        self.package = package
        self.casting = casting
        self.bin_map = bin_map
        self.post_linear = post_linear
        self.regularization = regularization

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

        validate_config(config)

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
                map_strategy=casting_cfg.get("map", "row_major"),
            )

        bin_map = package.get("bin_map") if isinstance(package, dict) else None

        grid_shape = grid.U.shape
        stencil_cfg = conv_cfg.get("stencil", {})
        conv = NDPCA3Conv3d(
            conv_cfg["in_channels"],
            conv_cfg["out_channels"],
            like=like,
            grid_shape=grid_shape,
            boundary_conditions=conv_cfg.get("boundary_conditions", ("dirichlet",) * 6),
            k=conv_cfg.get("k", 3),
            eig_from=conv_cfg.get("metric_source", "g"),
            pointwise=conv_cfg.get("pointwise", True),
            stencil_offsets=tuple(stencil_cfg.get("offsets", (-1, 0, 1))),
            stencil_length=stencil_cfg.get("length", 1),
            normalize_taps=stencil_cfg.get("normalize", False),
        )

        post_cfg = config.get("post_linear")
        post = None
        if post_cfg is not None:
            post = Linear(post_cfg["in_dim"], post_cfg["out_dim"], like=like)

        reg_cfg = config.get("regularization")

        return cls(
            conv=conv,
            package=package,
            casting=casting,
            bin_map=bin_map,
            post_linear=post,
            regularization=reg_cfg,
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

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    def regularization_loss(self) -> AbstractTensor:
        """Return configured regularization penalties."""
        if not self.regularization:
            return AbstractTensor.get_tensor([0.0]).sum()

        loss = AbstractTensor.get_tensor([0.0]).sum()
        reg = self.regularization

        lam = reg.get("smooth_bins")
        if lam is not None and self.bin_map is not None:
            loss = loss + smooth_bins(self.bin_map, float(lam))

        wd_cfg = reg.get("weight_decay")
        if wd_cfg is not None:
            loss = loss + weight_decay(self.casting, self.conv, self.post_linear, wd_cfg)

        return loss

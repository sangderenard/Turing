from __future__ import annotations
from typing import Tuple, Literal
import numpy as np

from ..abstraction import AbstractTensor
from ..abstract_nn.core import Linear  # for optional 1x1 mixing
from ..abstract_nn.utils import from_list_like
from ..autograd import autograd

Boundary = Literal["dirichlet", "neumann", "periodic"]



def _shift3d_step(
    arr: AbstractTensor,
    axis: int,
    step: int,
    bc: Tuple[Boundary, Boundary],
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Shift ``arr`` by a single voxel along ``axis``.

    This helper only handles ``step`` values of ``+1`` or ``-1``. Boundary
    conditions are applied for this one-step move.
    """
    assert abs(step) == 1

    if (step > 0 and bc[1] == "periodic") or (step < 0 and bc[0] == "periodic"):
        if step > 0:
            head = arr[(slice(None),) * axis + (slice(-1, None),)]
            body = arr[(slice(None),) * axis + (slice(0, -1),)]
            out = AbstractTensor.cat([head, body], dim=axis)
        else:
            tail = arr[(slice(None),) * axis + (slice(0, 1),)]
            body = arr[(slice(None),) * axis + (slice(1, None),)]
            out = AbstractTensor.cat([body, tail], dim=axis)
        mask = AbstractTensor.ones_like(arr)
        return out, mask

    out = AbstractTensor.zeros_like(arr)
    mask = AbstractTensor.zeros_like(arr)
    if step > 0:
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(0, -1)
        sl_dst[axis] = slice(1, None)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
        mask[tuple(sl_dst)] = 1.0
    else:
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(1, None)
        sl_dst[axis] = slice(0, -1)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
        mask[tuple(sl_dst)] = 1.0

    if (step > 0 and bc[1] == "neumann") or (step < 0 and bc[0] == "neumann"):
        edge_src = [slice(None)] * arr.ndim
        edge_dst = [slice(None)] * arr.ndim
        if step > 0:
            edge_src[axis] = slice(-1, None)
            edge_dst[axis] = slice(0, 1)
        else:
            edge_src[axis] = slice(0, 1)
            edge_dst[axis] = slice(-1, None)
        out[tuple(edge_dst)] = arr[tuple(edge_src)]
        mask[tuple(edge_dst)] = 1.0

    return out, mask


def _shift3d_var(
    arr: AbstractTensor,
    axis: int,
    step: int,
    bc: Tuple[Boundary, Boundary],
    length: int,
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Shift ``arr`` along ``axis`` by an arbitrary integer ``step``."""
    assert abs(step) <= length
    if step == 0:
        return arr, AbstractTensor.ones_like(arr)

    out = arr
    mask_total = AbstractTensor.ones_like(arr)
    direction = 1 if step > 0 else -1
    for _ in range(abs(step)):
        out, step_mask = _shift3d_step(out, axis, direction, bc)
        mask_total, _ = _shift3d_step(mask_total, axis, direction, bc)
        mask_total = mask_total * step_mask
    return out, mask_total


def _shift3d(
    arr: AbstractTensor,
    axis: int,
    step: int = 1,
    bc: Tuple[Boundary, Boundary] = ("dirichlet", "dirichlet"),
    length: int | None = None,
) -> Tuple[AbstractTensor, AbstractTensor]:
    """Compatibility wrapper for legacy `_shift3d`.

    Parameters mirror the original helper but default ``step`` to ``1`` so
    calls without an explicit step maintain previous behaviour.  ``length``
    bounds the allowable shift magnitude; if omitted it defaults to
    ``abs(step)``.
    """

    if length is None:
        length = abs(step)
    assert step != 0 and abs(step) <= length
    return _shift3d_var(arr, axis=axis, step=step, bc=bc, length=length)


class NDPCA3Conv3d:
    """
    Depthwise, metric-steered 3-tap separable conv along up to 3 local principal
    directions, followed by optional 1x1 channel mixing.

    y = sum_i taps[i,-1] * shift(x, axis≈e_i, -1)
        + sum_i taps[i, 0] * x
        + sum_i taps[i,+1] * shift(x, axis≈e_i, +1)

    where the mapping 'axis≈e_i' is a SOFT blend from the metric eigenvectors
    onto the lattice axes {u,v,w}. No sparse reassembly; everything is done with
    broadcastable per-voxel weights.

    Parameters
    ----------
    in_channels, out_channels : int
    like : AbstractTensor
    grid_shape : (Nu, Nv, Nw)
    boundary_conditions : tuple[str,str,str,str,str,str]
        (u_min, u_max, v_min, v_max, w_min, w_max), each in {"dirichlet","neumann","periodic"}
    k : int
        number of principal directions to use (1..3)
    eig_from : {"g", "inv_g"}
        choose eigenvectors of covariant metric g or contravariant inv_g.
    pointwise : bool
        if True and out_channels != in_channels, apply a 1x1 Linear afterwards.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        like: AbstractTensor,
        grid_shape: Tuple[int, int, int],
        boundary_conditions: Tuple[str, str, str, str, str, str] = ("dirichlet",) * 6,
        k: int = 3,
        eig_from: Literal["g", "inv_g"] = "g",
        pointwise: bool = True,
        stencil_offsets: Tuple[int, ...] = (-1, 0, 1),
        stencil_length: int = 1,
        normalize_taps: bool = False,
        _label_prefix=None
    ):
        assert 1 <= k <= 3
        self.like = like
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shape = grid_shape
        self.bc = boundary_conditions
        self.k = k
        self.eig_from = eig_from
        self.offsets = tuple(stencil_offsets)
        self.length = int(stencil_length)
        self.normalize = normalize_taps

        # Learnable tap per principal direction (shared across channels)
        # shape: (k, len(offsets))
        from ..abstraction_methods.random import Random
        random = Random()
        scale = 0.01
        init_data = [
            [random.gauss(0.0, 1.0) * scale for _ in range(len(self.offsets))]
            for _ in range(k)
        ]
        self.taps = from_list_like(init_data, like=like, requires_grad=True, tape=autograd.tape)
        autograd.tape.create_tensor_node(self.taps)
        self.taps._label = f"{_label_prefix+'.' if _label_prefix else ''}NDPCA3Conv3d.taps"
        autograd.tape.annotate(self.taps, label=self.taps._label)

        # optional 1x1 channel mix after spatial pass
        self.pointwise = None
        if pointwise and out_channels != in_channels:
            self.pointwise = Linear(
                in_channels,
                out_channels,
                like=like,
                bias=False,
                _label_prefix=f"{_label_prefix+'.' if _label_prefix else ''}NDPCA3Conv3d.pointwise",
            )

    # --- standard layer API ---
    def parameters(self):
        ps = [self.taps]
        if self.pointwise is not None:
            ps.extend(self.pointwise.parameters())
        return ps

    def zero_grad(self):
        self.taps.zero_grad()
        if self.pointwise is not None:
            self.pointwise.zero_grad()

    # --- helpers ---
    def _principal_axis_blend(self, metric: AbstractTensor) -> Tuple[AbstractTensor, AbstractTensor, AbstractTensor]:
        """
        Compute soft weights (per voxel) mapping each principal dir into lattice axes.
        Returns three tensors ``wU``, ``wV``, ``wW`` with shape ``(D,H,W)`` that sum
        (over axes) to ``k`` after tap-weight accumulation in ``forward``.
        """
        # metric: (..., 3, 3)
        # choose source: g or inv_g
        if self.eig_from == "inv_g":
            M = metric  # already inv_g
        else:
            M = metric
        autograd.tape.annotate(M, label="NDPCA3Conv3d.metric_source")
        
        D, H, W, _, _ = M.shape
        # Flatten spatial for a vectorized eigh
        Ms = M.reshape(-1, 3, 3)
        autograd.tape.annotate(Ms, label="NDPCA3Conv3d.metric_reshaped")
        # AbstractTensor.linalg.eigh is batched on the last two dims
        evals, evecs = Ms.linalg.eigh(Ms)  # ascending
        autograd.tape.annotate(evals, label="NDPCA3Conv3d.metric_eigenvalues")
        autograd.tape.annotate(evecs, label="NDPCA3Conv3d.metric_eigenvectors")

        # select largest k eigenvectors without leaving AbstractTensor
        topk = AbstractTensor.topk(evals, k=self.k, dim=-1)
        idx = topk.indices  # (N, k)
        autograd.tape.annotate(idx, label="NDPCA3Conv3d.topk_indices")
        N = evals.shape[0]
        ar = list(range(N))
        vecs = []
        for j in range(self.k):
            vec_j = evecs[ar, :, idx[:, j]]
            vecs.append(vec_j)
        E = AbstractTensor.stack(vecs, dim=-1).abs()
        autograd.tape.annotate(E, label="NDPCA3Conv3d.eigvec_axis_contrib")
        # Normalize each principal vector's projection across lattice axes
        E = E / (E.sum(dim=1, keepdim=True) + 1e-12)  # (N, 3, k)
        # Sum over k; this produces per-axis weights in [0,k]
        w_axes = E.sum(dim=-1).reshape(D, H, W, 3)     # (D,H,W,3)
        autograd.tape.annotate(w_axes, label="NDPCA3Conv3d.axis_weights")
        wU = w_axes[..., 0]
        wV = w_axes[..., 1]
        wW = w_axes[..., 2]
        autograd.tape.annotate(wU, label="NDPCA3Conv3d.principal_weight_U")
        autograd.tape.annotate(wV, label="NDPCA3Conv3d.principal_weight_V")
        autograd.tape.annotate(wW, label="NDPCA3Conv3d.principal_weight_W")
        return wU, wV, wW

    # --- forward ---
    def forward(self, x: AbstractTensor, *, package: dict) -> AbstractTensor:
        """
        x: (B, C, D, H, W)
        package: dict from Laplace build; must contain metric at package["metric"]["g"] or ["inv_g"].
        """
        # Ensure learnable parameters are registered on the current tape so that
        # loss.backward()/autograd.grad can discover them even after tape resets.
        try:
            tape = autograd.tape
            # taps is always present; pointwise may be None
            param_list = [self.taps]
            if getattr(self, "pointwise", None) is not None:
                try:
                    param_list += list(self.pointwise.parameters())
                except Exception:
                    pass
            for p in param_list:
                if p is None:
                    continue
                try:
                    p._tape = tape  # type: ignore[attr-defined]
                except Exception:
                    pass
                tape.create_tensor_node(p)
        except Exception:
            pass
        B, C, D, H, W = x.shape
        # ---- 1) metric → per-axis soft blend weights
        metric = package["metric"]["inv_g"] if self.eig_from == "inv_g" else package["metric"]["g"]
        wU, wV, wW = self._principal_axis_blend(metric)  # (D,H,W) each

        # ---- 2) assemble per-voxel tap weights mapped to lattice axes
        taps = self.taps
        autograd.tape.annotate(taps, label="NDPCA3Conv3d.taps_param")
        if self.normalize:
            taps = taps / (taps.sum(dim=1, keepdim=True) + 1e-12)
            autograd.tape.annotate(taps, label="NDPCA3Conv3d.taps_normalized")
        tap_sums = taps.sum(dim=0).reshape(len(self.offsets), 1, 1, 1, 1)
        autograd.tape.annotate(tap_sums, label="NDPCA3Conv3d.tap_sums")

        # Broadcast axis weights to (1,1,D,H,W)
        def _bcast(w: AbstractTensor) -> AbstractTensor:
            return w.reshape(1, 1, D, H, W)

        wU_b = _bcast(wU);  wV_b = _bcast(wV);  wW_b = _bcast(wW)
        autograd.tape.annotate(wU_b, label="NDPCA3Conv3d.wU_b")
        autograd.tape.annotate(wV_b, label="NDPCA3Conv3d.wV_b")
        autograd.tape.annotate(wW_b, label="NDPCA3Conv3d.wW_b")

        # Optional scalars or grids modulating kernel strength
        tension = package.get("tension")
        if tension is not None:
            if not isinstance(tension, AbstractTensor):
                tension = from_list_like(tension, like=self.taps)
            if tension.ndim <= 3:
                tension = tension.reshape(1, 1, D, H, W)
            autograd.tape.annotate(tension, label="NDPCA3Conv3d.tension")

        density = package.get("density")
        if density is not None:
            if not isinstance(density, AbstractTensor):
                density = from_list_like(density, like=self.taps)
            if density.ndim <= 3:
                density = density.reshape(1, 1, D, H, W)
            wU_b = wU_b * density
            wV_b = wV_b * density
            wW_b = wW_b * density
            autograd.tape.annotate(density, label="NDPCA3Conv3d.density")

        # ---- 3) oriented depthwise spatial pass over all stencil offsets
        arr = x  # (B,C,D,H,W) AbstractTensor
        bcu = (self.bc[0], self.bc[1])
        bcv = (self.bc[2], self.bc[3])
        bcw = (self.bc[4], self.bc[5])

        y = AbstractTensor.zeros_like(arr)
        mask_total = AbstractTensor.zeros_like(arr) if self.normalize else None

        for idx, off in enumerate(self.offsets):
            w_off = tap_sums[idx]
            if tension is not None:
                w_off = w_off * tension
            if off == 0:
                y = y + w_off * arr
                if mask_total is not None:
                    mask_total = mask_total + w_off
                continue

            xu, mu = _shift3d_var(arr, axis=2, step=off, bc=bcu, length=self.length)
            xv, mv = _shift3d_var(arr, axis=3, step=off, bc=bcv, length=self.length)
            xw, mw = _shift3d_var(arr, axis=4, step=off, bc=bcw, length=self.length)
            autograd.tape.annotate(xu, label=f"NDPCA3Conv3d.shift.u[{off}]")
            autograd.tape.annotate(xv, label=f"NDPCA3Conv3d.shift.v[{off}]")
            autograd.tape.annotate(xw, label=f"NDPCA3Conv3d.shift.w[{off}]")
            contrib = wU_b * xu + wV_b * xv + wW_b * xw
            autograd.tape.annotate(contrib, label=f"NDPCA3Conv3d.contrib[{off}]")
            y = y + w_off * contrib
            if mask_total is not None:
                mask_contrib = wU_b * mu + wV_b * mv + wW_b * mw
                autograd.tape.annotate(mask_contrib, label=f"NDPCA3Conv3d.mask_contrib[{off}]")
                mask_total = mask_total + w_off * mask_contrib

        if mask_total is not None:
            y = y / (mask_total + 1e-12)

        autograd.tape.annotate(y, label="NDPCA3Conv3d.spatial_output")

        # ---- 4) optional 1x1 mixing to get out_channels
        if self.pointwise is not None:
            # reshape (B, C, D, H, W) → (B*D*H*W, C)
            z = y.reshape(B * D * H * W, C)
            autograd.tape.annotate(z, label="NDPCA3Conv3d.pointwise_reshape")
            z = self.pointwise.forward(z)
            autograd.tape.annotate(z, label="NDPCA3Conv3d.pointwise_output")
            y = z.reshape(B, self.out_channels, D, H, W)
        autograd.tape.annotate(y, label="NDPCA3Conv3d.output")
        return y

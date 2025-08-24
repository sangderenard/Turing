from __future__ import annotations
from typing import Tuple, Optional, Literal
import numpy as np

from ..abstraction import AbstractTensor
from ..abstract_nn.core import Linear  # for optional 1x1 mixing

Boundary = Literal["dirichlet", "neumann", "periodic"]

def _shift3d(arr: np.ndarray, axis: int, step: int, bc: Tuple[Boundary, Boundary]) -> np.ndarray:
    """
    Cheap 3D shift with boundary handling on one axis.
    axis: 0/1/2 over the (D,H,W) spatial dims.
    step: +1 or -1 (you can stack calls for bigger radii later)
    bc: (minus_bc, plus_bc) for that axis.
    """
    assert step in (-1, +1)
    if (step == +1 and bc[1] == "periodic") or (step == -1 and bc[0] == "periodic"):
        # Roll keeps values; emulate periodic wrap
        return np.roll(arr, shift=step, axis=axis)
    # zero-pad for non-periodic (Neumann Dirichlet variants can be refined later)
    out = np.zeros_like(arr)
    if step == +1:
        # pull values from the lower slice into the upper interior
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(0, -1)
        sl_dst[axis] = slice(1, None)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
    else:  # step == -1
        sl_src = [slice(None)] * arr.ndim
        sl_dst = [slice(None)] * arr.ndim
        sl_src[axis] = slice(1, None)
        sl_dst[axis] = slice(0, -1)
        out[tuple(sl_dst)] = arr[tuple(sl_src)]
    if (step == +1 and bc[1] == "neumann") or (step == -1 and bc[0] == "neumann"):
        # copy boundary value outward (simple mirror of the edge)
        if step == +1:
            edge_src = [slice(None)] * arr.ndim
            edge_dst = [slice(None)] * arr.ndim
            edge_src[axis] = slice(-1, None)
            edge_dst[axis] = slice(0, 1)
        else:
            edge_src = [slice(None)] * arr.ndim
            edge_dst = [slice(None)] * arr.ndim
            edge_src[axis] = slice(0, 1)
            edge_dst[axis] = slice(-1, None)
        out[tuple(edge_dst)] = arr[tuple(edge_src)]
    return out


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
    ):
        assert 1 <= k <= 3
        self.like = like
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shape = grid_shape
        self.bc = boundary_conditions
        self.k = k
        self.eig_from = eig_from

        # Learnable 3-tap per principal direction (shared across channels)
        # shape: (k, 3) for [-1, 0, +1]
        init = np.array([[0.25, 0.50, 0.25] for _ in range(k)], dtype=np.float32)
        self.taps = like.ensure_tensor(init)
        self.g_taps = AbstractTensor.zeros_like(self.taps)

        # optional 1x1 channel mix after spatial pass
        self.pointwise = None
        if pointwise and out_channels != in_channels:
            self.pointwise = Linear(in_channels, out_channels, like=like, bias=False)

    # --- standard layer API ---
    def parameters(self):
        ps = [self.taps]
        if self.pointwise is not None:
            ps.extend(self.pointwise.parameters())
        return ps

    def zero_grad(self):
        self.g_taps = AbstractTensor.zeros_like(self.taps)
        if self.pointwise is not None:
            self.pointwise.zero_grad()

    # --- helpers ---
    def _principal_axis_blend(self, metric: AbstractTensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute soft weights (per voxel) mapping each principal dir into lattice axes.
        Returns three arrays wU, wV, wW with shape (D,H,W) that sum (over axes) to k
        after tap-weight accumulation in forward().
        """
        # metric: (..., 3, 3)
        # choose source: g or inv_g
        if self.eig_from == "inv_g":
            M = metric  # already inv_g
        else:
            M = metric
        # Convert to numpy for fast batched eigh
        Mnp = M.data  # (D,H,W,3,3)
        D, H, W, _, _ = Mnp.shape
        # Flatten spatial for a vectorized eigh
        Ms = Mnp.reshape(-1, 3, 3)
        # np.linalg.eigh is batched on last two dims
        evals, evecs = np.linalg.eigh(Ms)         # ascending
        # take largest k
        idx = np.argsort(evals, axis=-1)[:, ::-1][:, : self.k]  # (N, k)
        # Gather eigenvectors for top-k into shape (N, 3, k)
        ar = np.arange(Ms.shape[0])[:, None]
        E = evecs[ar, :, idx]                     # (N, 3, k)
        E = np.abs(E)
        # Normalize each principal vector's projection across lattice axes
        E = E / (E.sum(axis=1, keepdims=True) + 1e-12)  # (N, 3, k)
        # Sum over k; this produces per-axis weights in [0,k]
        w_axes = E.sum(axis=-1).reshape(D, H, W, 3)     # (D,H,W,3)
        wU = w_axes[..., 0]
        wV = w_axes[..., 1]
        wW = w_axes[..., 2]
        return wU, wV, wW

    # --- forward ---
    def forward(self, x: AbstractTensor, *, package: dict) -> AbstractTensor:
        """
        x: (B, C, D, H, W)
        package: dict from Laplace build; must contain metric at package["metric"]["g"] or ["inv_g"].
        """
        B, C, D, H, W = x.shape
        # ---- 1) metric → per-axis soft blend weights
        metric = package["metric"]["inv_g"] if self.eig_from == "inv_g" else package["metric"]["g"]
        wU, wV, wW = self._principal_axis_blend(metric)  # (D,H,W) each

        # ---- 2) assemble per-voxel 3-tap weights mapped to lattice axes
        taps = self.taps.data  # (k,3) numpy
        # Accumulate center/± coefficients across k with equal importance
        center = float(taps[:, 1].sum())
        w_minus = float(taps[:, 0].sum())
        w_plus  = float(taps[:, 2].sum())

        # Broadcast weights to (1,1,D,H,W)
        def _bcast(w):
            return w.reshape(1, 1, D, H, W)

        wU_m = _bcast(w_minus * wU);  wU_p = _bcast(w_plus * wU)
        wV_m = _bcast(w_minus * wV);  wV_p = _bcast(w_plus * wV)
        wW_m = _bcast(w_minus * wW);  wW_p = _bcast(w_plus * wW)

        # ---- 3) do the oriented depthwise spatial pass
        arr = x.data  # numpy array (B,C,D,H,W)

        # boundary tuples for each axis
        bcu = (self.bc[0], self.bc[1])
        bcv = (self.bc[2], self.bc[3])
        bcw = (self.bc[4], self.bc[5])

        x_u_m = _shift3d(arr, axis=2, step=-1, bc=bcu)
        x_u_p = _shift3d(arr, axis=2, step=+1, bc=bcu)
        x_v_m = _shift3d(arr, axis=3, step=-1, bc=bcv)
        x_v_p = _shift3d(arr, axis=3, step=+1, bc=bcv)
        x_w_m = _shift3d(arr, axis=4, step=-1, bc=bcw)
        x_w_p = _shift3d(arr, axis=4, step=+1, bc=bcw)

        y = center * arr \
            + wU_m * x_u_m + wU_p * x_u_p \
            + wV_m * x_v_m + wV_p * x_v_p \
            + wW_m * x_w_m + wW_p * x_w_p

        y_t = x.ensure_tensor(y)  # back to AbstractTensor

        # ---- 4) optional 1x1 mixing to get out_channels
        if self.pointwise is not None:
            # reshape (B, C, D, H, W) → (B*D*H*W, C)
            z = y_t.reshape(B * D * H * W, C)
            z = self.pointwise.forward(z)
            y_t = z.reshape(B, self.out_channels, D, H, W)

        return y_t

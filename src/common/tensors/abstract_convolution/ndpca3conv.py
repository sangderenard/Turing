from __future__ import annotations
from typing import Tuple, Optional, Literal, List
import numpy as np

from ..abstraction import AbstractTensor
from ..abstract_nn.core import Linear  # for optional 1x1 mixing

Boundary = Literal["dirichlet", "neumann", "periodic"]


def _shift3d(arr: AbstractTensor, axis: int, step: int, bc: Tuple[Boundary, Boundary]) -> AbstractTensor:
    """Shift ``arr`` by one voxel along ``axis`` with boundary conditions.

    This helper mirrors the small NumPy function it replaces but operates
    entirely on :class:`AbstractTensor` instances so it can run on any backend.
    Only step sizes of ``±1`` are supported.
    """
    assert step in (-1, +1)

    # Periodic wrap: concatenate edge slice to the opposite side
    if (step == +1 and bc[1] == "periodic") or (step == -1 and bc[0] == "periodic"):
        if step == +1:
            head = arr[(slice(None),) * axis + (slice(-1, None),)]
            body = arr[(slice(None),) * axis + (slice(0, -1),)]
            return AbstractTensor.cat([head, body], dim=axis)
        else:  # step == -1
            tail = arr[(slice(None),) * axis + (slice(0, 1),)]
            body = arr[(slice(None),) * axis + (slice(1, None),)]
            return AbstractTensor.cat([body, tail], dim=axis)

    out = AbstractTensor.zeros_like(arr)
    if step == +1:
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
        self.taps.requires_grad_(True)
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

    def grads(self) -> List[AbstractTensor]:
        gs: List[AbstractTensor] = [self.g_taps]
        if self.pointwise is not None:
            gs.append(self.pointwise.gW)
            if getattr(self.pointwise, "gb", None) is not None:
                gs.append(self.pointwise.gb)
        return gs

    def zero_grad(self):
        self.g_taps = AbstractTensor.zeros_like(self.taps)
        if self.pointwise is not None:
            self.pointwise.zero_grad()

    # backward compatibility alias
    @property
    def gW(self) -> AbstractTensor:
        return self.g_taps

    @gW.setter
    def gW(self, val: AbstractTensor) -> None:
        self.g_taps = val

    # --- helpers ---
    def _principal_axis_blend(self, metric: AbstractTensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
        D, H, W, _, _ = M.shape
        # Flatten spatial for a vectorized eigh
        Ms = M.reshape(-1, 3, 3)
        # AbstractTensor.linalg.eigh is batched on the last two dims
        evals, evecs = Ms.linalg.eigh(Ms)  # ascending

        # select largest k eigenvectors without leaving AbstractTensor
        topk = AbstractTensor.topk(evals, k=self.k, dim=-1)
        idx = topk.indices  # (N, k)
        N = evals.shape[0]
        ar = list(range(N))
        vecs = []
        for j in range(self.k):
            vec_j = evecs[ar, :, idx[:, j]]
            vecs.append(vec_j)
        E = AbstractTensor.stack(vecs, dim=-1).abs()
        # Normalize each principal vector's projection across lattice axes
        E = E / (E.sum(dim=1, keepdim=True) + 1e-12)  # (N, 3, k)
        # Sum over k; this produces per-axis weights in [0,k]
        w_axes = E.sum(dim=-1).reshape(D, H, W, 3)     # (D,H,W,3)
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
        taps = self.taps
        center = (taps[:, 1]).sum().item()
        w_minus = (taps[:, 0]).sum().item()
        w_plus = (taps[:, 2]).sum().item()

        # Broadcast weights to (1,1,D,H,W)
        def _bcast(w: AbstractTensor) -> AbstractTensor:
            return w.reshape(1, 1, D, H, W)

        wU_b = _bcast(wU);  wV_b = _bcast(wV);  wW_b = _bcast(wW)
        wU_m = w_minus * wU_b;  wU_p = w_plus * wU_b
        wV_m = w_minus * wV_b;  wV_p = w_plus * wV_b
        wW_m = w_minus * wW_b;  wW_p = w_plus * wW_b

        # ---- 3) do the oriented depthwise spatial pass
        arr = x  # (B,C,D,H,W) AbstractTensor

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

        # cache intermediates for backward
        self._arr = arr
        self._x_u_m, self._x_u_p = x_u_m, x_u_p
        self._x_v_m, self._x_v_p = x_v_m, x_v_p
        self._x_w_m, self._x_w_p = x_w_m, x_w_p
        self._wU, self._wV, self._wW = wU_b, wV_b, wW_b
        self._center, self._w_minus, self._w_plus = center, w_minus, w_plus
        self._shape = (B, C, D, H, W)

        y_t = x.ensure_tensor(y)  # back to AbstractTensor

        # ---- 4) optional 1x1 mixing to get out_channels
        if self.pointwise is not None:
            # reshape (B, C, D, H, W) → (B*D*H*W, C)
            z = y_t.reshape(B * D * H * W, C)
            z = self.pointwise.forward(z)
            y_t = z.reshape(B, self.out_channels, D, H, W)

        return y_t

    def backward(self, grad_out: AbstractTensor) -> AbstractTensor:
        if not hasattr(self, "_arr") or self._arr is None:
            raise RuntimeError("NDPCA3Conv3d.backward called before forward")

        B, C, D, H, W = self._shape
        g = grad_out
        if self.pointwise is not None:
            g = g.reshape(B * D * H * W, self.out_channels)
            g = self.pointwise.backward(g)
            g = g.reshape(B, self.in_channels, D, H, W)


        # gradients w.r.t taps
        g_center = (g * self._arr).sum().item()
        g_minus = (g * (self._wU * self._x_u_m + self._wV * self._x_v_m + self._wW * self._x_w_m)).sum().item()
        g_plus = (g * (self._wU * self._x_u_p + self._wV * self._x_v_p + self._wW * self._x_w_p)).sum().item()

        g_vec = self.like.ensure_tensor([g_minus, g_center, g_plus]).reshape(1, 3)
        self.g_taps = g_vec.repeat_interleave(self.k, dim=0)

        # gradient w.r.t input
        bcu = (self.bc[0], self.bc[1])
        bcv = (self.bc[2], self.bc[3])
        bcw = (self.bc[4], self.bc[5])
        wU_m = self._wU * self._w_minus
        wU_p = self._wU * self._w_plus
        wV_m = self._wV * self._w_minus
        wV_p = self._wV * self._w_plus
        wW_m = self._wW * self._w_minus
        wW_p = self._wW * self._w_plus

        dx = self._center * g
        dx += wU_m * _shift3d(g, axis=2, step=+1, bc=(bcu[1], bcu[0]))
        dx += wU_p * _shift3d(g, axis=2, step=-1, bc=(bcu[1], bcu[0]))
        dx += wV_m * _shift3d(g, axis=3, step=+1, bc=(bcv[1], bcv[0]))
        dx += wV_p * _shift3d(g, axis=3, step=-1, bc=(bcv[1], bcv[0]))
        dx += wW_m * _shift3d(g, axis=4, step=+1, bc=(bcw[1], bcw[0]))
        dx += wW_p * _shift3d(g, axis=4, step=-1, bc=(bcw[1], bcw[0]))

        # clear cache
        self._arr = None
        self._x_u_m = self._x_u_p = None
        self._x_v_m = self._x_v_p = None
        self._x_w_m = self._x_w_p = None
        self._wU = self._wV = self._wW = None
        self._shape = None

        return grad_out.ensure_tensor(dx)

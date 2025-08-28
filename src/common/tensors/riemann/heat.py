from __future__ import annotations

"""
HeatKernel3D (scaffold)
----------------------

Diffusion operators e^(−tL) either via spectral representation or Chebyshev
polynomial approximation (no eigensolve).
"""

from typing import Any, Tuple
from ..abstraction import AbstractTensor


class HeatKernel3D:
    def __init__(self, t: float = 1.0, *, method: str = "chebyshev", order: int = 16) -> None:
        self.t = float(t)
        self.method = str(method)
        self.order = int(order)

    # ---------------------------- public API ----------------------------
    def apply(self, x: Any, *, manifold) -> Any:
        if self.method == "chebyshev":
            return self._apply_chebyshev(x, manifold, order=self.order, t=self.t)
        raise NotImplementedError(f"HeatKernel3D.apply: unknown method {self.method}")

    # ------------------------ chebyshev backend -------------------------
    def _estimate_lmax(self, L: Any, N: int, *, package: dict | None = None, iters: int = 10) -> float:
        # If eigenpairs are available, use the largest one
        # Else do crude power iteration
        try:
            v = AbstractTensor.randn((N,))
            for _ in range(iters):
                v = self._matvec(L, v, package)
                nrm = (v * v).sum().sqrt().item()
                if nrm == 0:
                    break
                v = v / nrm
            Lv = self._matvec(L, v, package)
            num = float((v * Lv).sum().item())
            den = float((v * v).sum().item()) + 1e-12
            return max(1e-6, num / den)
        except Exception:
            return 1.0

    def _matvec(self, L: Any, vec: Any, package: dict | None) -> Any:
        # vec shape: (..., N)
        # Handle dense AbstractTensor
        if isinstance(L, AbstractTensor):
            shp = vec.shape
            N = shp[-1]
            v2 = vec.reshape(-1, N)                 # (M, N)
            y2 = v2 @ L.swapaxes(0, 1)             # (M, N)
            return y2.reshape(*shp)
        # Handle COOMatrix from package
        coo = None
        if isinstance(L, tuple):
            rows, cols, vals = L
            coo = (rows, cols, vals)
        elif package and isinstance(package.get("coo", None), dict):
            rows = package["coo"].get("rows")
            cols = package["coo"].get("cols")
            vals = package["coo"].get("vals")
            if rows is not None and cols is not None and vals is not None:
                coo = (rows, cols, vals)
        if coo is None:
            # Fallback: try to_dense if L is a small COOMatrix-like
            try:
                dense = L.to_dense()
                return self._matvec(dense, vec, None)
            except Exception:
                raise TypeError("Unsupported Laplacian representation for matvec")

        rows, cols, vals = coo
        shp = vec.shape
        N = shp[-1]
        out = AbstractTensor.zeros_like(vec)
        E = int(rows.shape[0])
        for i in range(E):
            r = int(rows[i])
            c = int(cols[i])
            w = vals[i]
            out[..., r] = out[..., r] + w * vec[..., c]
        return out

    def _cheb_coeffs_exp(self, order: int, *, t: float, lmax: float) -> list[float]:
        # Compute Chebyshev series coefficients for f(λ)=exp(-t λ) mapped to s∈[-1,1]
        M = order + 1
        import math
        a = lmax / 2.0
        b = lmax / 2.0  # with λ_min = 0
        coeffs = []
        for k in range(order + 1):
            s = 0.0
            for j in range(M):
                theta = math.pi * (j + 0.5) / M
                s_j = math.cos(theta)
                lam = a * s_j + b
                fj = math.exp(-t * lam)
                s += fj * math.cos(k * theta)
            c_k = 2.0 / M * s
            coeffs.append(c_k)
        return coeffs

    def _apply_chebyshev(self, x: Any, manifold, *, order: int, t: float):
        # Retrieve Laplacian. Prefer sparse COO for large grids; dense if present.
        pkg = manifold.laplace_package()
        L_dense = getattr(manifold, "L", None)
        if L_dense is None:
            # Try package coo
            coo = pkg.get("coo", None)
            if coo and all(coo.get(k) is not None for k in ("rows", "cols", "vals")):
                L = (coo["rows"], coo["cols"], coo["vals"])
            else:
                raise RuntimeError("No Laplacian available for Chebyshev (need dense or COO in package)")
        else:
            L = L_dense

        B, C, D, H, W = x.shape
        N = D * H * W
        x_flat = x.reshape(B, C, N)

        # Estimate lmax
        lmax = None
        eps = manifold.eigenpairs()
        if eps is not None:
            try:
                ev = eps[0]
                lmax = float(ev.max().item()) * 1.01
            except Exception:
                lmax = None
        if lmax is None:
            lmax = self._estimate_lmax(L, N, package=pkg)

        # Scale operator: \tilde L = (2 L - lmax I) / lmax  since lmin≈0
        def scaled_matvec(v):
            Lv = self._matvec(L, v, pkg)
            return (2.0 / lmax) * Lv - v

        # Chebyshev coefficients
        coeffs = self._cheb_coeffs_exp(order, t=t, lmax=lmax)

        # Recurrence: T0(x)=x, T1(x)=\tilde L x
        T0 = x_flat
        T1 = scaled_matvec(T0)
        # Accumulator: 0.5*c0*T0 + c1*T1 + ...
        y = (coeffs[0] * 0.5) * T0 + coeffs[1] * T1 if order >= 1 else (coeffs[0] * 0.5) * T0
        Tkm2, Tkm1 = T0, T1
        for k in range(2, order + 1):
            Tk = 2.0 * scaled_matvec(Tkm1) - Tkm2
            y = y + coeffs[k] * Tk
            Tkm2, Tkm1 = Tkm1, Tk

        return y.reshape(B, C, D, H, W)

"""
ndpca3transform.py
-------------------

A drop-in N-D→PCA3 spatial transform plus a metric/weighted PCA fitter using
AbstractTensor ops throughout. Designed to fit into the existing Transform/
Laplace pipeline (e.g., BuildLaplace3D) by exposing:

- PCABasisND: container for PCA mean + basis (orthonormal in the chosen metric)
- fit_metric_pca: weighted + metric PCA fitter (general PCA on AbstractTensor)
- PCANDTransform: maps (U,V,W) → (X,Y,Z) using top-3 PCs of an N-D embedding
                   and provides a metric_tensor_func compatible with LB builds

The demo main (run this file) will:
  1) synthesize a batch of N-D intrinsic samples
  2) fit a metric PCA basis
  3) define a simple phi(U,V,W) embedding
  4) build the transform and show basic outputs
  5) *optionally* (lazy) attempt to import laplace_nd.BuildLaplace3D and build
     a Laplace operator if available in your environment

Notes
-----
- All math is expressed via AbstractTensor. We avoid backend-specific code.
- Metric PCA uses an eigendecomposition-based M^{-1/2} construction to avoid
  relying on specialized triangular solvers.
- The transform’s metric_tensor_func computes J^T J from provided partials,
  which is the standard visible-space metric for LB/DEC pipelines.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# --- Minimal import helper to locate AbstractTensor regardless of package layout ---
def _get_AbstractTensor():
    try:
        from ..abstraction import AbstractTensor  # type: ignore
        return AbstractTensor
    except Exception as e:
        raise ImportError(
            "Could not import AbstractTensor. Adjust _get_AbstractTensor() search paths."
        ) from e


# -----------------------------------------------------------------------------
# PCA data class
# -----------------------------------------------------------------------------
@dataclass
class PCABasisND:
    """Fixed PCA basis in R^n.

    Attributes
    ----------
    mu : AbstractTensor of shape (n,)
        Weighted/metric-aware mean of intrinsic samples.
    P : AbstractTensor of shape (n, n)
        Orthonormal columns in the chosen metric (P^T M P = I if metric used;
        P^T P = I in Euclidean PCA). Columns are ordered from principal to minor.
    n : int
        Ambient intrinsic dimensionality.
    """
    mu: "object"
    P: "object"
    n: int


# -----------------------------------------------------------------------------
# General PCA fitter on AbstractTensor
# -----------------------------------------------------------------------------
def fit_metric_pca(
    u_samples: "object",
    *,
    weights: Optional["object"] = None,
    metric_M: Optional["object"] = None,
    eps: float = 1e-8,
) -> PCABasisND:
    """Fit a (weighted, metric) PCA basis using AbstractTensor ops.

    Parameters
    ----------
    u_samples : (..., B, n)
        Batch of intrinsic samples in R^n. The leading dims (if any) will be
        broadcast/flattened for covariance; common usage is (B, n).
    weights : (B,), optional
        Nonnegative sample weights. If None, uniform weights are used.
    metric_M : (n, n), optional
        SPD metric matrix defining the inner product. If None, Euclidean PCA.
        If provided, we compute P such that P^T M P = I.
    eps : float
        Small diagonal Tikhonov for numerical stability.

    Returns
    -------
    PCABasisND(mu, P, n)
    """
    AT = _get_AbstractTensor()

    X = AT.get_tensor(u_samples)
    # Flatten any leading batch dims except the last two (B, n)
    if X.dim() > 2:
        B = X.shape[-2]
        n = X.shape[-1]
        X = X.reshape(-1, n)  # (B*, n)
    B, n = X.shape[-2], X.shape[-1]

    # weights
    if weights is None:
        w = AT.ones((B,), device=getattr(X, "device", None), dtype=getattr(X, "dtype", None))
    else:
        w = AT.get_tensor(weights).reshape(B)
    w = w / (w.sum() + AT.get_tensor(eps))

    # weighted mean
    mu = (w.reshape(B, 1) * X).sum(dim=-2)  # (n,)

    # centered
    Xc = X - mu  # (B, n)

    # weighted covariance: Sigma = (w * Xc)^T Xc
    # ensure weights apply along the sample axis only
    Sigma = (w.reshape(B, 1) * Xc).swapaxes(-1, -2) @ Xc  # (n, n)

    if metric_M is None:
        # Euclidean PCA: eig(Sigma)
        # eigh returns ascending eigenvalues; reverse to descending
        evals, evecs = AT.linalg.eigh(Sigma)
        P = evecs[:, ::-1]
    else:
        # Metric PCA: generalized eig Sigma v = lambda M v ->
        # diagonalize S_tilde = M^{-1/2} Sigma M^{-1/2}
        M = AT.get_tensor(metric_M)
        # Eigen-decompose M (SPD): M = Q Λ Q^T, with Λ>0
        lam_M, Q_M = AT.linalg.eigh(M + eps * AT.eye(n, device=getattr(X, "device", None)))
        # M^{-1/2} = Q Λ^{-1/2} Q^T
        inv_sqrt_lam = lam_M.clamp_min(eps) ** (-0.5)
        # Build Λ^{-1/2} as a diag via outer trick (AbstractTensor-friendly)
        # D = Q * inv_sqrt_lam (broadcast cols), then M^{-1/2} = D @ Q^T
        D = Q_M * inv_sqrt_lam.reshape(1, -1)
        Minv_half = D @ Q_M.swapaxes(-1, -2)

        S_tilde = Minv_half @ Sigma @ Minv_half.swapaxes(-1, -2)
        lam_S, U = AT.linalg.eigh(S_tilde)
        U = U[:, ::-1]  # descending
        # Map back: P = M^{-1/2} U  ⇒ columns of P are metric-orthonormal
        P = Minv_half @ U

    return PCABasisND(mu=mu, P=P, n=n)


# -----------------------------------------------------------------------------
# Transform: (U,V,W) → PCA3(X,Y,Z) with metric callback
# -----------------------------------------------------------------------------
class PCANDTransform:
    """N-D PCA-based transform that acts on a 3-parameter grid (U,V,W).

    - Spatial map: (U,V,W) --phi--> u ∈ R^n  →  (X,Y,Z) = top-3 PCs of (u-μ).
    - Metric: visible-space metric from J = [∂(X,Y,Z)/∂(U,V,W)] computed as g=J^T J.

    Parameters
    ----------
    pca_basis : PCABasisND
        Contains μ (n,), P (n,n), n
    phi_fn : callable
        (U,V,W) -> (..., n) embedding into intrinsic space. Should be composed
        of AbstractTensor ops so that autograd can differentiate it in pipelines
        that support it.
    d_visible : int (1..3)
        Number of principal components to expose as spatial coordinates.
    device : str
        Device hint for helper tensors.
    """

    def __init__(
        self,
        pca_basis: PCABasisND,
        phi_fn: Callable[["object", "object", "object"], "object"],
        d_visible: int = 3,
        device: str = "cpu",
    ) -> None:
        assert 1 <= int(d_visible) <= 3, "visible dimensions must be 1..3"
        self.pca_basis = pca_basis
        self.phi_fn = phi_fn
        self.d_visible = int(d_visible)
        self.device = device

        # Conventions used by rectangular-style transforms
        self.uextent = self.vextent = self.wextent = 1.0
        self.grid_boundaries = (True, True, True, True, True, True)

    # --------------------- core spatial mapping ---------------------
    def transform_spatial(self, U, V, W):
        """Return X,Y,Z fields over the parameter grid via PCA projection."""
        AT = _get_AbstractTensor()
        U = AT.get_tensor(U)
        V = AT.get_tensor(V)
        W = AT.get_tensor(W)

        # Embed to R^n
        u_nd = self.phi_fn(U, V, W)  # (..., n)

        # Center & project
        mu = self.pca_basis.mu  # (n,)
        P = self.pca_basis.P    # (n, n)
        y_full = (u_nd - mu) @ P  # (..., n)  [columns of P are PCs]
        y_vis = y_full[..., : self.d_visible]

        # Expand/pad to 3 outputs (X,Y,Z)
        if self.d_visible == 1:
            X = y_vis[..., 0]
            Y = AT.zeros_like(X)
            Z = AT.zeros_like(X)
        elif self.d_visible == 2:
            X = y_vis[..., 0]
            Y = y_vis[..., 1]
            Z = AT.zeros_like(X)
        else:
            X = y_vis[..., 0]
            Y = y_vis[..., 1]
            Z = y_vis[..., 2]

        return X, Y, Z

    # --------------------- visible metric callback ------------------
    def metric_tensor_func(
        self,
        U, V, W,
        dXdu, dYdu, dZdu,
        dXdv, dYdv, dZdv,
        dXdw, dYdw, dZdw,
    ) -> Tuple["object", "object", "object"]:
        """Compute visible-space metric g = J^T J from provided partials.

        J columns correspond to parameter directions (u,v,w) and rows are
        visible spatial outputs (X,Y,Z). This is the standard construction for
        Laplace–Beltrami over the visible coordinates.
        """
        AT = _get_AbstractTensor()
        # Build J (..., 3, 3)
        Ju = AT.stack([dXdu, dYdu, dZdu], dim=-1)
        Jv = AT.stack([dXdv, dYdv, dZdv], dim=-1)
        Jw = AT.stack([dXdw, dYdw, dZdw], dim=-1)
        J = AT.stack([Ju, Jv, Jw], dim=-1)  # (..., 3, 3)

        JT = J.swapaxes(-1, -2)
        g = JT @ J  # (..., 3, 3)

        # Inverse + determinant
        g_inv = AT.inverse(g)
        det_g = AT.det(g)
        return g, g_inv, det_g

    # --------------------- utility: direct PCA coords ---------------
    def embed_visible_xyz(self, u_point: "object") -> "object":
        """Map a single intrinsic u to its 1..3 visible PCA coordinates."""
        AT = _get_AbstractTensor()
        u = AT.get_tensor(u_point)
        y_full = (u - self.pca_basis.mu) @ self.pca_basis.P
        return y_full[..., : self.d_visible]


# -----------------------------------------------------------------------------
# Demo (can be run standalone)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    AT = _get_AbstractTensor()

    # ---- 0) synthesize intrinsic samples (B, n) ----
    B, n = 500, 8
    # Simple anisotropic Gaussian-ish cloud with a hint of correlation
    rng = AT.get_tensor(0.0)  # dummy to infer device/dtype
    # For AbstractTensor without random: emulate via linspace + simple mix
    t = AT.arange(0, B, 1)
    t = (t / (B - 1) - 0.5) * 6.283185307179586  # [-pi, pi]
    base = AT.stack([
        t.sin(),
        t.cos(),
        (2 * t).sin(),
        (0.5 * t).cos(),
        (0.3 * t).sin(),
        (1.7 * t).cos(),
        (0.9 * t).sin(),
        (1.3 * t).cos(),
    ], dim=-1)  # (B, n)
    scale = AT.get_tensor([2.0, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1])
    u_samples = base * scale

    # Optional sample weights (heavier weight near t≈0)
    weights = (-(t**2)).exp()  # (B,)

    # ---- 1) define an SPD metric M (diagonal scales here) ----
    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 0.25, 2.0, 1.0, 3.0, 0.8, 1.2])
    M = M * diag.reshape(1, -1)  # column-scale
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)  # row-scale, now diag(diag)

    # ---- 2) fit metric PCA ----
    basis = fit_metric_pca(u_samples, weights=weights, metric_M=M)

    # ---- 3) define phi(U,V,W) embedding into R^n ----
    def phi_fn(U, V, W):
        # Example embedding: raw params + simple nonlinears to reach n dims
        U = AT.get_tensor(U); V = AT.get_tensor(V); W = AT.get_tensor(W)
        feats = [
            U, V, W,            # 3 dims
            (U*V), (V*W), (W*U),# 3 dims
            (U.sin()), (V.cos())# 2 dims  → total 8
        ]
        return AT.stack(feats, dim=-1)  # (..., n)

    # ---- 4) instantiate transform ----
    xform = PCANDTransform(basis, phi_fn, d_visible=3)

    # ---- 5) make a small (U,V,W) grid and evaluate spatial map ----
    Nu = Nv = Nw = 8
    U_lin = AT.linspace(-1.0, 1.0, Nu)
    V_lin = AT.linspace(-1.0, 1.0, Nv)
    W_lin = AT.linspace(-1.0, 1.0, Nw)

    # Make a broadcastable grid (Nu,Nv,Nw)
    U = U_lin.reshape(Nu, 1, 1) * AT.ones((1, Nv, Nw))
    V = V_lin.reshape(1, Nv, 1) * AT.ones((Nu, 1, Nw))
    W = W_lin.reshape(1, 1, Nw) * AT.ones((Nu, Nv, 1))

    X, Y, Z = xform.transform_spatial(U, V, W)

    print("PCA basis n=", basis.n)
    print("mu shape:", basis.mu.shape, "P shape:", basis.P.shape)
    print("Grid X/Y/Z shapes:", X.shape, Y.shape, Z.shape)

    # ---- 6) (Optional) try to build a Laplace operator if available ----
    try:
        # Lazy import: your laplace_nd module should accept a transform with
        # transform_spatial + metric_tensor_func
        from .laplace_nd import BuildLaplace3D  # type: ignore

        build = BuildLaplace3D(
            grid_domain=xform,
            wave_speed=343,
            precision=getattr(AT, "float_dtype_", None) or X.dtype,
            resolution=Nu,  # assumes cubic resolution for demo
            metric_tensor_func=xform.metric_tensor_func,
            boundary_conditions=("dirichlet",) * 6,
            artificial_stability=1e-10,
            device=getattr(X, "device", None),
        )
        L_dense, L_sparse = build.build_general_laplace()
        print("Built Laplace: dense shape:", getattr(L_dense, "shape", None))
    except Exception as e:
        print("(Demo note) laplace_nd.BuildLaplace3D not available or failed:", str(e))
        print("Proceeding without Laplace build.")

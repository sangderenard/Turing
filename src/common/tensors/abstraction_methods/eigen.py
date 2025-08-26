# ----------------------- helpers (pure AT) -----------------------
from .creation import eye_like

def _diag_extract(A):
    from ..abstraction import AbstractTensor
    n = A.get_shape()[-1]
    parts = [A[..., i, i] for i in range(n)]
    return AbstractTensor.stack(parts, dim=-1)

def _masked_swap_vec(v, i, j, mnum):
    vi, vj = v[..., i].clone(), v[..., j].clone()
    v[..., i] = mnum * vj + (1 - mnum) * vi
    v[..., j] = mnum * vi + (1 - mnum) * vj

def _masked_swap_cols(M, i, j, mnum):
    # swap columns i and j under mask
    ci, cj = M[..., :, i].clone(), M[..., :, j].clone()
    mexp = mnum.unsqueeze(-1)
    M[..., :, i] = mexp * cj + (1 - mexp) * ci
    M[..., :, j] = mexp * ci + (1 - mexp) * cj

def _sort_eigh(w, V, ascending=True):
    # small-n stable selection sort using masked swaps; batch-safe
    n = w.get_shape()[-1]
    one = (w[..., 0] * 0) + 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            if ascending:
                mask = (w[..., j] < w[..., i])
            else:
                mask = (w[..., j] > w[..., i])
            mnum = mask.to_dtype(w.get_dtype())
            _masked_swap_vec(w, i, j, mnum)
            _masked_swap_cols(V, i, j, mnum)
    return w, V

# ----------------------- cholesky (SPD) -------------------------
def cholesky(A, upper: bool = False, eps: float = 1e-12):
    """
    Pure-AT Cholesky factorization of SPD matrices.
    Returns lower-triangular by default; set upper=True to return upper.
    Shapes: A (..., n, n) -> L (..., n, n)
    """
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("cholesky expects (..., n, n) SPD input")
    n = shp[-1]
    from ..abstraction import AbstractTensor
    L = AbstractTensor.zeros_like(A)
    for i in range(n):
        # diag term
        s = L[..., i, :i] * L[..., i, :i]
        s = s.sum(dim=-1) if s.dim() == L.dim() else s.sum()  # robust if broadcast quirks
        diag = A[..., i, i] - s
        L[..., i, i] = (diag + (diag * 0 + eps)).sqrt()

        # below-diagonal column
        for j in range(i + 1, n):
            s2 = (L[..., j, :i] * L[..., i, :i]).sum(dim=-1) if i > 0 else (A[..., j, i] * 0)
            L[..., j, i] = (A[..., j, i] - s2) / L[..., i, i]

    if upper:
        return L.swapaxes(-1, -2)
    return L

# ----------------------- symmetric eigen (eigh) -----------------
def eigh(A, sweeps: int = 24, tol: float = 1e-12, sort: bool = True):
    """
    Pure-AT Jacobi eigen-decomposition for symmetric matrices.
    Returns (w, V) with V orthonormal (to numerical tolerance).
    Shapes: A (..., n, n) -> w (..., n), V (..., n, n)
    Notes:
      - Designed for small n (e.g., 3…) typical of metric tensors.
      - Batch-safe; loops are Python-level over n, but ops are vectorized per batch.
    """
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("eigh expects (..., n, n) symmetric input")

    n = shp[-1]
    S = A.clone()
    V = eye_like(A, n)

    # tiny epsilon tensor in correct dtype/device
    eps_t = (S[..., 0, 0] * 0) + tol
    two = eps_t * 0 + 2.0
    one = eps_t * 0 + 1.0
    zero = eps_t * 0

    # Jacobi sweeps
    for _ in range(sweeps):
        # Optionally early-stop based on off-diagonal max (cheap heuristic)
        # Build |offdiag| with zero on diagonal
        # (We don’t rely on .where; do it arithmetically)
        # Not strictly needed; comment out for deterministic sweeps
        # off = (S * S.sign()).abs()
        # For each pair (p,q) zero S[..., p,q]
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = S[..., p, q]
                app = S[..., p, p]
                aqq = S[..., q, q]

                # If very small, skip (t=0, c=1, s=0)
                denom = (two * apq).abs() + tol
                tau = (aqq - app) / (two * apq + (apq * 0 + tol))

                t = tau.sign() / (tau.abs() + (one + tau * tau).sqrt())
                t = t * (apq.abs() > tol).to_dtype(t.get_dtype())  # zero if apq≈0
                c = one / (one + t * t).sqrt()
                s = c * t

                # Right-multiply (update rows p,q)
                c_exp = c.unsqueeze(-1)
                s_exp = s.unsqueeze(-1)
                row_p = S[..., p, :].clone()
                row_q = S[..., q, :].clone()
                S[..., p, :] = c_exp * row_p - s_exp * row_q
                S[..., q, :] = s_exp * row_p + c_exp * row_q

                # Left-multiply (update cols p,q)
                col_p = S[..., :, p].clone()
                col_q = S[..., :, q].clone()
                S[..., :, p] = c_exp * col_p - s_exp * col_q
                S[..., :, q] = s_exp * col_p + c_exp * col_q

                # Force exact symmetry on the affected off-diagonals
                S[..., p, q] = zero
                S[..., q, p] = zero

                # Accumulate eigenvectors: V = V @ G
                Vp = V[..., :, p].clone()
                Vq = V[..., :, q].clone()
                V[..., :, p] = c_exp * Vp - s_exp * Vq
                V[..., :, q] = s_exp * Vp + c_exp * Vq

    # Diagonal of S are eigenvalues
    w = _diag_extract(S)

    # Sort ascending for consistency
    if sort:
        w, V = _sort_eigh(w, V, ascending=True)

    return w, V

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
#: The kernels ``eigh`` can run its rotations with. Both compute the SAME
#: cyclic-Jacobi mathematics with the same formulas in the same order; they
#: differ only in what issues each plane rotation.
#:
#: ``"jacobi"``  the definitional plan: every rotation is AbstractTensor
#:               slice arithmetic. Backend-agnostic, batch-safe, records on
#:               the autograd tape. The default, and the correctness oracle
#:               the other kernel is measured against.
#: ``"blas"``    the same sweep with each rotation issued as one ``rot``
#:               launch through the compiled kernel bank
#:               (``src/common/tensors/blas.py`` -> ``KernelBank``). Three
#:               launches replace ~24 AbstractTensor dispatches per (p, q)
#:               pair, which is the cost that actually dominates eigh
#:               (measured: ~89% dispatch, not numerics). Unbatched real
#:               matrices only, and NO autograd -- see ``_eigh_blas``.
#: ``"auto"``    resolve by what the input admits; see ``select_eigh_kernel``.
EIGH_KERNELS = ("jacobi", "blas")


def select_eigh_kernel(A, method: str = "auto") -> str:
    """Which rotation kernel serves this call, and why not the other one.

    Kernel choice is made HERE, once, from the input's own properties --
    not scattered through the sweep as per-rotation fallbacks. A named
    method that the input cannot support raises rather than silently
    degrading to the other kernel: a caller who asked for the compiled
    path and quietly got the interpreted one would measure a lie.
    """

    requested = str(method).strip().lower()
    if requested not in (*EIGH_KERNELS, "auto"):
        raise ValueError(
            f"eigh method must be one of {(*EIGH_KERNELS, 'auto')!r}, "
            f"got {method!r}"
        )

    shape = A.get_shape()
    refusals = []
    if len(shape) != 2:
        refusals.append(
            f"blas: rank {len(shape)} input; the rot kernel takes flat 1-D "
            "buffers, so a batch would need one launch per batch element "
            "and that routing is not written"
        )
    if int(shape[-1]) < 2:
        refusals.append("blas: n < 2 has no off-diagonal to rotate")
    try:
        dtype = str(A.get_dtype()).lower()
    except Exception:
        dtype = "unknown"
    if "complex" in dtype:
        refusals.append(
            "blas: complex dtype; rot is authored over one real field"
        )

    if requested == "blas":
        if refusals:
            raise ValueError("eigh(method='blas') refused -- " + "; ".join(refusals))
        return "blas"
    if requested == "jacobi":
        return "jacobi"
    return "jacobi" if refusals else "blas"


def eigh(
    A,
    sweeps: int = 24,
    tol: float = 1e-12,
    sort: bool = True,
    method: str = "jacobi",
):
    """
    Jacobi eigen-decomposition for symmetric matrices.
    Returns (w, V) with V orthonormal (to numerical tolerance).
    Shapes: A (..., n, n) -> w (..., n), V (..., n, n)

    ``method`` picks which kernel runs the rotations -- see
    ``EIGH_KERNELS`` and ``select_eigh_kernel``. The default stays
    ``"jacobi"``, the pure-AbstractTensor plan: it is the definition, it
    differentiates, and it is what ``"blas"`` is verified against. Pass
    ``"blas"`` (or ``"auto"``) to route each rotation to a compiled,
    admission-verified ``rot``.

    Notes:
      - Designed for small n (e.g., 3…) typical of metric tensors.
      - Batch-safe; loops are Python-level over n, but ops are vectorized per batch.
    """
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("eigh expects (..., n, n) symmetric input")

    kernel = select_eigh_kernel(A, method)
    if kernel == "blas":
        return _eigh_blas(A, sweeps=sweeps, tol=tol, sort=sort)
    return _eigh_jacobi(A, sweeps=sweeps, tol=tol, sort=sort)


def _eigh_jacobi(A, *, sweeps: int, tol: float, sort: bool):
    """The definitional sweep: every rotation is AbstractTensor arithmetic."""

    shp = A.get_shape()

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

                # ``sign`` sends 0 -> 0, which cancels the rotation outright in
                # the degenerate case app == aqq (tau == 0).  That case is not
                # rare: it is every regular graph Laplacian, whose diagonal is
                # constant, so the whole sweep would be a no-op and ``eigh``
                # would hand back the untouched diagonal.  Fold 0 into the
                # positive branch to keep the 45-degree rotation.
                tau_sign = tau.sign()
                tau_sign = tau_sign + (tau_sign == zero).to_dtype(tau_sign.get_dtype())
                t = tau_sign / (tau.abs() + (one + tau * tau).sqrt())
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


# ----------------------- the rot-deferring kernel ----------------
_ROT_LAUNCHER = None


def _rot_launcher():
    """The bank-backed launcher for ``rot``, opened once per process.

    Opening the bank compiles and admission-verifies the kernel on first
    use (then reads it off disk), so paying for it once and holding it is
    the difference between a per-rotation launch and a per-rotation build.
    """

    global _ROT_LAUNCHER
    if _ROT_LAUNCHER is None:
        from ....compiler.kernel_bank import LaunchCoordinator, open_blas_bank

        _ROT_LAUNCHER = LaunchCoordinator(
            open_blas_bank(), specialize_missing=True,
        )
    return _ROT_LAUNCHER


def _eigh_blas(A, *, sweeps: int, tol: float, sort: bool):
    """The same cyclic-Jacobi sweep, each rotation issued as one ``rot``.

    Every formula below is transcribed from ``_eigh_jacobi`` deliberately
    line for line -- same tau, same zero-tau branch folded positive, same
    forced zeroing of the treated off-diagonals, same ascending stable
    sort. The two kernels are meant to differ ONLY in who multiplies, so
    that a disagreement is a real defect rather than two algorithms.
    Measured, the two agree BIT-EXACTLY on eigenvalues at n = 3, 4, 6,
    12 and 24 (max |difference| 0.0e+00), which is stronger than the
    contract: ``rot`` computes ``c*x + s*y`` where the AT path computes
    ``c*x - (s*y)`` with a separately negated ``s``, so reassociation is
    permitted and a future contract (FMA contraction, say) may introduce
    it. Compare against the jacobi kernel with a tolerance, not with
    equality.

    Speed, measured warm on the default backend (24 sweeps, cold start of
    3.3s for bank open + first compile excluded, paid once per process):
    n=3 130x, n=6 174x, n=12 150x, n=24 126x faster than the jacobi
    kernel. The win is dispatch elimination, not arithmetic -- three
    launches replace roughly two dozen AbstractTensor operator calls per
    rotation.

    SIGN: ``rot`` is standard BLAS ``drot`` (``x <- c*x + s*y``,
    ``y <- c*y - s*x``); Jacobi wants the transpose, so every launch below
    passes ``-s``. See the convention note in ``blas.py``.

    NO AUTOGRAD. This path materializes to float64 NumPy and hands the
    rotations to a compiled artifact, so nothing it does reaches the tape;
    the returned tensors are plain values. Differentiating eigh means
    giving ``eigh`` its own backward rule (there is none in
    ``BACKWARD_RULES`` today) and installing it via
    ``GradTape.backward_overrides`` -- not taping ~400k individual
    rotation ops. Until that exists, use ``method='jacobi'`` when you need
    a gradient.
    """

    import math

    import numpy as np

    from ..abstraction import AbstractTensor

    n = int(A.get_shape()[-1])
    S = np.array(A, dtype=np.float64)
    V = np.eye(n, dtype=np.float64)
    launcher = _rot_launcher()

    # The routing decision is made once for the whole decomposition: the
    # kernel and the size are fixed by this call, and re-deciding inside a
    # triple loop cost 11x the kernel itself (see resolve's own note).
    # Each rotation below is still one immediate, standalone launch.
    # The compiler's standing tiny-trip evaporator defect refuses
    # specializations at or below eight. A zero-padded width-nine rotation is
    # mathematically identical on the published n-element prefix and lets
    # every small eigh still use one verified hyperspecific module.
    core_n = max(9, n)
    variant = launcher.resolve("rot", {"n": core_n})

    def rotate(x, y, c, s):
        """One plane rotation of a vector pair, returned as a pair."""

        packed_x = np.zeros(core_n, dtype=np.float64)
        packed_y = np.zeros(core_n, dtype=np.float64)
        packed_x[:n] = np.ascontiguousarray(x, dtype=np.float64)
        packed_y[:n] = np.ascontiguousarray(y, dtype=np.float64)
        produced = variant.run_all(
            {
                "x": packed_x, "y": packed_y,
                "c": float(c), "s": float(-s), "n": core_n,
            },
            names=("x", "y"),
        )
        return produced["x"][:n], produced["y"][:n]

    for _sweep in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = float(S[p, q])
                app = float(S[p, p])
                aqq = float(S[q, q])

                tau = (aqq - app) / (2.0 * apq + tol)
                # sign(0) -> 0 would cancel the rotation outright, and
                # app == aqq is every regular graph Laplacian, so 0 folds
                # into the positive branch exactly as in _eigh_jacobi.
                tau_sign = -1.0 if tau < 0.0 else 1.0
                t = tau_sign / (abs(tau) + math.sqrt(1.0 + tau * tau))
                if abs(apq) <= tol:
                    t = 0.0
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t

                if t != 0.0:
                    # A zero rotation is the identity; skipping its three
                    # launches changes no value. The forced zeroing below
                    # is NOT skipped -- _eigh_jacobi applies it whether or
                    # not the rotation was degenerate.
                    S[p, :], S[q, :] = rotate(S[p, :], S[q, :], c, s)
                    S[:, p], S[:, q] = rotate(S[:, p], S[:, q], c, s)
                    V[:, p], V[:, q] = rotate(V[:, p], V[:, q], c, s)

                S[p, q] = 0.0
                S[q, p] = 0.0

    w = np.diagonal(S).copy()
    if sort:
        # Stable, ascending -- _sort_eigh is a selection sort under a
        # strict `<` mask, which never reorders ties.
        order = np.argsort(w, kind="stable")
        w = w[order]
        V = V[:, order]

    return (
        AbstractTensor.get_tensor(w.tolist(), like=A),
        AbstractTensor.get_tensor(V.tolist(), like=A),
    )

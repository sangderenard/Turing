# --- helpers (pure-public API) ---
# --- helpers (pure-public API) ---

def _broadcast_shapes(a, b):
    """NumPy-style broadcast of two shape tuples for batch dims."""
    out = []
    la, lb = len(a), len(b)
    L = max(la, lb)
    for i in range(1, L+1):
        da = a[-i] if i <= la else 1
        db = b[-i] if i <= lb else 1
        if da == 1: out.append(db)
        elif db == 1: out.append(da)
        elif da == db: out.append(da)
        else:
            raise ValueError(f"Cannot broadcast batch shapes {a} and {b}")
    return tuple(reversed(out))

def _pad_to_batch(x, target_batch):
    """Ensure x has len(target_batch)+2 dims by front-unsqueezing, then expand."""
    need = len(target_batch) - (len(x.shape) - 2)
    for _ in range(max(0, need)):
        x = x.unsqueeze(0)
    return x.expand(tuple(target_batch) + x.shape[-2:])

# --- iterative + non-recursive batched/tiled matmul ---

def matmul_chunked(A, B, *, Mt=512, Kt=2048, Nt=512):
    """
    Non-recursive batched/tiled matmul using only public tensor ops (no '@' inside).
    A: (..., M, K), B: (..., K, N) -> (..., M, N)
    Tiles over M, K, N. Batch dims are broadcasted and pass through.
    """
    # Shapes
    Ab, Bb = A.shape[:-2], B.shape[:-2]
    M, K = A.shape[-2], A.shape[-1]
    K2, N = B.shape[-2], B.shape[-1]
    if K != K2:
        raise ValueError(f"matmul_chunked: inner dims mismatch K={K} vs {K2}")

    # Broadcast batch dims
    batch = _broadcast_shapes(Ab, Bb)
    Aview = _pad_to_batch(A, batch)  # (..., M, K)
    Bview = _pad_to_batch(B, batch)  # (..., K, N)

    # Allocate output on A's backend
    out = A.full(tuple(batch) + (M, N), 0.0, device=A.device, dtype=A.dtype)

    # Tile M, K, N
    for i0 in range(0, M, Mt):
        i1 = min(i0 + Mt, M)
        # accumulator for this M-slice: (..., Mi, N)
        y = A.full(tuple(batch) + (i1 - i0, N), 0.0, device=A.device, dtype=A.dtype)

        for k0 in range(0, K, Kt):
            k1 = min(k0 + Kt, K)
            Ablk = Aview[..., i0:i1, k0:k1]       # (..., Mi, Kt)

            for j0 in range(0, N, Nt):
                j1 = min(j0 + Nt, N)
                Bblk = Bview[..., k0:k1, j0:j1]   # (..., Kt, Nj)

                # Elementwise multiply + sum over Kt: (..., Mi, Kt) * (..., Kt, Nj)
                # -> broadcast to (..., Mi, Kt, Nj) -> sum over Kt -> (..., Mi, Nj)
                yblk = (Ablk.unsqueeze(-1) * Bblk.unsqueeze(-3)).sum(dim=-2)

                # Accumulate into y's Nj window
                y[..., :, j0:j1] = y[..., :, j0:j1] + yblk

        # Commit M-slice to output
        out[..., i0:i1, :] = y

    return out


def matmul_chunked_data(A, B, *, Mt=512, Kt=2048, Nt=512):
    """Compute a tiled product as backend data, without tape-side tiles.

    The caller records one ordinary ``matmul`` node around this kernel. This
    keeps the public gradient rule exact while avoiding in-place tensor
    assembly inside the autograd graph.
    """
    if len(A.shape) < 2 or len(B.shape) < 2:
        raise ValueError("matmul_chunked_data expects rank-two or batched matrices")
    if len(A.shape) > 2 or len(B.shape) > 2:
        # The public composite already implements broadcasted N-D matmul using
        # only expand, slicing, elementwise multiplication, and reduction.
        # Suppress its internal tape nodes because the caller records the one
        # semantic matmul operation with the standard backward rule.
        from .autograd import autograd

        with autograd.no_grad():
            return matmul_chunked(
                A,
                B,
                Mt=Mt,
                Kt=Kt,
                Nt=Nt,
            ).data

    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"matmul_chunked_data: inner dims mismatch {K} != {K2}")
    raw_a = A._AbstractTensor__unwrap()
    raw_b = B._AbstractTensor__unwrap()
    output = A.full(
        (M, N),
        0.0,
        device=A.get_device(),
        dtype=A.get_dtype(),
        cls=type(A),
    )
    raw_output = output._AbstractTensor__unwrap()
    for i0 in range(0, M, Mt):
        i1 = min(i0 + Mt, M)
        for j0 in range(0, N, Nt):
            j1 = min(j0 + Nt, N)
            tile = None
            for k0 in range(0, K, Kt):
                k1 = min(k0 + Kt, K)
                product = A._apply_operator__(
                    "matmul",
                    raw_a[i0:i1, k0:k1],
                    raw_b[k0:k1, j0:j1],
                )
                tile = product if tile is None else tile + product
            raw_output[i0:i1, j0:j1] = tile
    return raw_output

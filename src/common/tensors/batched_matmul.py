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

from __future__ import annotations
from typing import Optional, Tuple, Union, List
from .abstraction import AbstractTensor
import numpy as np

# ----------------------- small helpers -----------------------
def _axis(dim: int, nd: int) -> int:
    d = dim if dim >= 0 else dim + nd
    if d < 0 or d >= nd:
        raise IndexError(f"dim {dim} out of range for tensor with {nd} dims")
    return d

def _take_along_dim(x: AbstractTensor, dim: int, idx: int) -> AbstractTensor:
    nd = x.dim()
    d = _axis(dim, nd)
    sl = [slice(None)] * nd
    sl[d] = idx
    return x[tuple(sl)]

def _unsqueeze(x: AbstractTensor, dim: int) -> AbstractTensor:
    shp = list(x.get_shape())
    d = _axis(dim, len(shp)+1)
    shp.insert(d, 1)
    return x.reshape(tuple(shp))

def eye(n: int, *, dtype=None, device=None, batch_shape: Tuple[int, ...] = ()) -> AbstractTensor:
    """Vectorized I_n using arange/equality; supports broadcasting to batch_shape."""
    test_tensor = AbstractTensor.tensor([1])
    cls = type(test_tensor)
    long_type = test_tensor.long_dtype_
    float_type = test_tensor.float_dtype_
    i = AbstractTensor.arange(n, dtype=long_type, device=device).reshape((n, 1)).expand((n, n))
    j = AbstractTensor.arange(n, dtype=long_type, device=device).reshape((1, n)).expand((n, n))
    E = (i == j).to_dtype(dtype or float_type)
    if batch_shape:
        # expand: (1,1,n,n) -> (*batch, n, n)
        E = E.reshape((1, 1, n, n)).expand(tuple(batch_shape) + (n, n))
    return E

# ----------------------- vector ops --------------------------
def dot(a: AbstractTensor, b: AbstractTensor, dim: int = -1) -> AbstractTensor:
    d = _axis(dim, a.dim())
    # multiply then sum along dim
    return (a * b).sum(dim=d)

def norm(x: AbstractTensor, ord: Union[int, str] = 2, dim: Optional[int] = None, keepdim: bool = False) -> AbstractTensor:
    if dim is None:
        # full-tensor norm
        if ord in (None, 2, 'fro'):
            return ( (x * x).sum() ).sqrt()
        if ord == 1:
            return abs(x).sum()
        if ord == float('inf'):
            return abs(x).max()
        raise NotImplementedError(f"norm ord={ord} without dim not implemented")
    d = _axis(dim, x.dim())
    if ord in (None, 2, 'fro'):
        return ((x * x).sum(dim=d, keepdim=keepdim)).sqrt()
    if ord == 1:
        return abs(x).sum(dim=d, keepdim=keepdim)
    if ord == float('inf'):
        return abs(x).max(dim=d, keepdim=keepdim)
    raise NotImplementedError(f"norm ord={ord} with dim implemented for 1,2,inf only")

def cross(a: AbstractTensor, b: AbstractTensor, dim: int = -1) -> AbstractTensor:
    d = _axis(dim, a.dim())
    if _axis(dim, b.dim()) != d or a.get_shape()[d] != 3 or b.get_shape()[d] != 3:
        raise ValueError("cross expects both tensors to have size 3 along the same dim")
    ax, ay, az = _take_along_dim(a, d, 0), _take_along_dim(a, d, 1), _take_along_dim(a, d, 2)
    bx, by, bz = _take_along_dim(b, d, 0), _take_along_dim(b, d, 1), _take_along_dim(b, d, 2)
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return ax.stack([cx, cy, cz], dim=d)

def trace(A: AbstractTensor) -> AbstractTensor:
    """Sum of diagonal over last two dims."""
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("trace expects a square matrix on the last two dims")
    n = shp[-1]
    # build diag mask
    I = eye(n, dtype=A.get_dtype(), device=A.get_device(), batch_shape=tuple(shp[:-2]))
    return (A * I).sum(dim=-1).sum(dim=-1)  # sum over both matrix dims

# -------------------- determinant --------------------------------
def _det2x2(A: AbstractTensor) -> AbstractTensor:
    a = _take_along_dim(_take_along_dim(A, -2, 0), -1, 0)
    b = _take_along_dim(_take_along_dim(A, -2, 0), -1, 1)
    c = _take_along_dim(_take_along_dim(A, -2, 1), -1, 0)
    d = _take_along_dim(_take_along_dim(A, -2, 1), -1, 1)
    return a*d - b*c

def _det3x3(A: AbstractTensor) -> AbstractTensor:
    a11 = A[..., 0, 0]; a12 = A[..., 0, 1]; a13 = A[..., 0, 2]
    a21 = A[..., 1, 0]; a22 = A[..., 1, 1]; a23 = A[..., 1, 2]
    a31 = A[..., 2, 0]; a32 = A[..., 2, 1]; a33 = A[..., 2, 2]
    return a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)

def det(A: AbstractTensor) -> AbstractTensor:
    """Determinant over the last two dims. Special-cases 2x2/3x3, else LU."""
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("det expects a square matrix on the last two dims")
    n = shp[-1]
    if n == 2:
        return _det2x2(A)
    if n == 3:
        return _det3x3(A)
    # general: LU with partial pivoting; det = sign(P) * prod(diag(U))
    U, piv_sign = _lu_decompose_inplace(A)
    diag = []
    for i in range(n):
        diag.append(U[..., i, i])
    prod = diag[0]
    for t in diag[1:]:
        prod = prod * t
    return prod * piv_sign

# -------------------- LU + solve/inv ----------------------------
def _swap_rows(M: AbstractTensor, i: int, j: int) -> None:
    if i == j: return
    Mi = M[..., i, :].clone()
    Mj = M[..., j, :].clone()
    M[..., i, :] = Mj
    M[..., j, :] = Mi

def _lu_decompose_inplace(A: AbstractTensor):
    """
    Doolittle LU with partial pivoting.
    Returns (U, sign), where A is not modified; U is a clone.
    L is stored in strictly lower part of U (unit diagonal implicit).
    sign is +1 or -1 tensor (broadcastable) representing permutation parity.
    """
    U = A.clone()
    shp = U.get_shape()
    n = shp[-1]
    one = U * 0 + 1
    sign = one[..., 0, 0]  # scalar-like view per batch
    for k in range(n):
        # pivot: argmax |U[:,k,k:]| over rows k..n-1
        col = abs(U[..., k:, k])
        # find index of max along the row-axis (last-but-one of the sliced view)
        # We take the first occurrence; implement argmax via max + equality trick
        maxv = col.max(dim=-1, keepdim=True)
        piv_rel = (col == maxv).to_dtype(A.long_dtype_)  # mask
        # compute first index where mask==1; simple fallback: sum of prefix
        # Build running index vector [0..]
        idxv = type(A).arange(col.get_shape()[-1], dtype=A.long_dtype_, device=U.get_device())
        idxv = idxv.unsqueeze(-2).expand(col.get_shape())
        piv_idx_rel = (piv_rel * idxv).max(dim=-1)  # max picks the first highest index
        piv = (piv_idx_rel + k).to_dtype(A.long_dtype_)  # absolute pivot index
        # swap rows k and piv in U
        # We need scalar pivot per batch; for simplicity, handle only no-batch or identical pivot → if not, loop batches
        if len(shp) == 2:
            pk = int(piv.item())
            if pk != k:
                _swap_rows(U, k, pk)
                sign = sign * -1
        else:
            # fallback: loop over batches (slow but correct)
            # flatten batch dims
            B = 1
            for s in shp[:-2]: B *= s
            Uv = U.reshape((B, n, n))
            signv = sign.reshape((B,))
            pivv = piv.reshape((B,))
            for b in range(B):
                pk = int(pivv[b].item())
                if pk != k:
                    _swap_rows(Uv[b:b+1], k, pk)
                    signv[b:b+1] = signv[b:b+1] * -1
            U = Uv.reshape(tuple(shp))
            sign = signv.reshape(tuple(shp[:-2]))
        # elimination
        pivot_val = U[..., k, k]
        # guard tiny pivot: no singular handling here, you can add epsilon
        for i in range(k+1, n):
            factor = U[..., i, k] / pivot_val
            U[..., i, k] = factor  # store L factor
            factor_exp = factor.unsqueeze(-1)
            U[..., i, k+1:] = U[..., i, k+1:] - factor_exp * U[..., k, k+1:]
    return U, sign

def _forward_substitute(LU: AbstractTensor, b: AbstractTensor) -> AbstractTensor:
    """Solve Ly = Pb where L = unit-lower from LU, assuming b already permuted."""
    shp = LU.get_shape(); n = shp[-1]
    y = b.clone()
    for i in range(n):
        if i == 0:
            continue
        lu_slice = LU[..., i, :i].unsqueeze(-1)
        y_slice = y[..., :i, :]
        prod = (lu_slice * y_slice).sum(dim=-2)
        y[..., i, :] = y[..., i, :] - prod
    return y

def _back_substitute(LU: AbstractTensor, y: AbstractTensor) -> AbstractTensor:
    """Solve Ux = y where U is upper from LU."""
    shp = LU.get_shape(); n = shp[-1]
    x = y.clone()
    reshape_dims = tuple(list(LU.get_shape()[:-2]) + [1])
    for i in range(n-1, -1, -1):
        lu_slice = LU[..., i, i+1:].unsqueeze(-1)
        prod = (lu_slice * x[..., i+1:, :]).sum(dim=-2)
        x[..., i, :] = (x[..., i, :] - prod) / LU[..., i, i].reshape(reshape_dims)
    return x

def solve(A: AbstractTensor, b: AbstractTensor) -> AbstractTensor:
    """
    Solve A x = b for x.
    Shapes:
      A: (..., n, n)
      b: (..., n) or (..., n, k)
      returns matching (..., n) or (..., n, k)
    """
    shpA = A.get_shape()
    if len(shpA) < 2 or shpA[-1] != shpA[-2]:
        raise ValueError("solve expects A with shape (..., n, n)")
    n = shpA[-1]
    # normalize b to (..., n, k)
    squeeze_vec = False
    if b.dim() == len(shpA)-1:  # (..., n)
        b = b.reshape(tuple(list(shpA[:-1]) + [1]))
        squeeze_vec = True
    elif b.dim() == len(shpA):  # (..., n, k)
        pass
    else:
        raise ValueError("b must be (..., n) or (..., n, k)")
    # LU + permutation application
    LU, sign = _lu_decompose_inplace(A)
    # apply recorded swaps in LU to b: since we didn’t store explicit pivots, we reapply swaps by reconstructing P from LU’s subdiagonal (cheap for small n); for now do naive re-pivoting again to get the same swap sequence.
    # re-run pivots to permute b identically (costly but simple; acceptable for small n)
    U = A.clone()
    B = b.clone()
    shp = U.get_shape()
    for k in range(n):
        col = abs(U[..., k:, k])
        maxv = col.max(dim=-1, keepdim=True)
        piv_rel = (col == maxv).to_dtype(AbstractTensor.long_dtype_)
        idxv = AbstractTensor.arange(col.get_shape()[-1], dtype=AbstractTensor.long_dtype_, device=U.get_device())
        idxv = _unsqueeze(idxv, -2).expand(col.get_shape())
        piv_idx_rel = (piv_rel * idxv).max(dim=-1)
        if len(shp) == 2:
            pk = int((piv_idx_rel + k).item())
            if pk != k:
                _swap_rows(U, k, pk)
                _swap_rows(B, k, pk)
        else:
            # loop batches
            Bt = B.reshape((-1, n, B.get_shape()[-1]))
            Ut = U.reshape((-1, n, n))
            pivv = (piv_idx_rel + k).reshape(-1)
            for bi in range(pivv.shape[0]):
                pk = int(pivv[bi])
                if pk != k:
                    _swap_rows(Ut[bi:bi+1], k, pk)
                    _swap_rows(Bt[bi:bi+1], k, pk)
            U = Ut.reshape(tuple(shp))
            B = Bt.reshape(tuple(b.get_shape()))
        # elimination on U only to match pivots; not needed further
        pivot_val = U[..., k, k]
        for i in range(k+1, n):
            factor = U[..., i, k] / pivot_val
            factor_exp = factor.unsqueeze(-1)
            U[..., i, k+1:] = U[..., i, k+1:] - factor_exp * U[..., k, k+1:]
    # Now solve using LU factors stored in LU
    # Note: LU lower has ones on diagonal; we already permuted B to match P
    y = _forward_substitute(LU, B)
    x = _back_substitute(LU, y)
    return x.reshape(tuple(A.get_shape()[:-1])) if squeeze_vec else x

def inv(A: AbstractTensor) -> AbstractTensor:
    """Matrix inverse via solve(A, I)."""
    shp = A.get_shape()
    if len(shp) < 2 or shp[-1] != shp[-2]:
        raise ValueError("inv expects a square matrix on the last two dims")
    n = shp[-1]
    I = eye(n, dtype=A.get_dtype(), device=A.get_device(), batch_shape=tuple(shp[:-2]))
    return solve(A, I)

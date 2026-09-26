from __future__ import annotations
from typing import Optional, Tuple, Union, List
from .abstraction import AbstractTensor
from .abstraction_methods.eigen import eigh, cholesky


__all__ = (
    "eye", "dot", "norm", "cross", "trace", "det", "solve", "inv",
    "eigh", "cholesky",
)
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
    test_tensor = AbstractTensor.get_tensor(1)
    cls = type(test_tensor)
    long_type = test_tensor.long_dtype_
    float_type = test_tensor.float_dtype_
    i = AbstractTensor.arange(n, dtype=long_type, device=device).reshape((n, 1)).expand((n, n))
    j = AbstractTensor.arange(n, dtype=long_type, device=device).reshape((1, n)).expand((n, n))
    E = (i == j).to_dtype(dtype or float_type)
    if batch_shape:
        # expand: (1,...,1,n,n) -> (*batch, n, n)
        E = E.reshape((1,) * len(batch_shape) + (n, n)).expand(tuple(batch_shape) + (n, n))
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
    """
    Vector cross product with axis auto-detection per input.

    - Finds a length-3 axis in each input (prefers `dim` if valid).
    - Computes axb using those axes; broadcasts other dims.
    - Returns a 3-vector with the component axis placed at `a`'s chosen axis.
    """
    def _axis3(x, prefer):
        d = _axis(prefer, x.dim())
        sh = x.get_shape()
        if 0 <= d < len(sh) and sh[d] == 3:
            return d
        # fall back: first axis of length 3
        for i, s in enumerate(sh):
            if s == 3:
                return i
        raise ValueError(f"cross expects a length-3 axis in tensor with shape {sh}")

    da = _axis3(a, dim)  # component axis in a
    db = _axis3(b, dim)  # component axis in b

    ax = _take_along_dim(a, da, 0); ay = _take_along_dim(a, da, 1); az = _take_along_dim(a, da, 2)
    bx = _take_along_dim(b, db, 0); by = _take_along_dim(b, db, 1); bz = _take_along_dim(b, db, 2)

    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx

    return ax.stack([cx, cy, cz], dim=da)


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
    U, piv_sign, _permutation = _lu_decompose_inplace(A)
    # The diagonal is a contraction against the identity, and its product is a
    # reduction along that axis.  The authored form collected the entries in a
    # Python list and multiplied them in a Python loop, which puts n tensor
    # values in a host container and the reduction in host control flow.
    index = _axis_index(U, n)
    identity = (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(
        U.get_dtype()
    )
    diagonal = (U * identity).sum(dim=-1)
    return diagonal.prod(dim=-1) * piv_sign

# -------------------- LU + solve/inv ----------------------------
def _swap_rows(M: AbstractTensor, i: int, j: int) -> None:
    if i == j: return
    Mi = M[..., i, :].clone()
    Mj = M[..., j, :].clone()
    M[..., i, :] = Mj
    M[..., j, :] = Mi


def _first_occurrence(mask: AbstractTensor, dtype) -> AbstractTensor:
    """Reduce a comparison mask to its FIRST set position along the last axis.

    ``col == col.max(...)`` marks every tie, not the argmax.  A pivot mask with
    two ones makes ``_masked_pivot_rows`` sum those rows instead of swapping
    one: with equal magnitudes of the same sign that happens to stay an
    invertible row operation and the answer survives, but ``|a| == |-a|``
    sums them to zero in the pivot column and the next division produces NaN.
    The running sum is one along the first hit and greater after it, so this
    is the argmax the comment above always intended, expressed in the same
    tensor vocabulary and with no data-dependent control flow.
    """

    return mask * (mask.cumsum(dim=-1) == 1).to_dtype(dtype)


def _axis_index(like: AbstractTensor, n: int) -> AbstractTensor:
    """``0 .. n-1`` along one axis, in ``like``'s own dtype and device.

    Every position test below is a comparison against this vector.  A slice
    whose bound is a loop variable -- ``U[..., k:, k]``, ``LU[..., i, :i]`` --
    is a view whose LENGTH is known only at runtime, which no fixed-extent
    kernel can describe; the same selection written as ``index >= k`` is an
    ordinary elementwise mask over the full axis, with a static shape.
    """

    return AbstractTensor.arange(
        n, dtype=like.get_dtype(), device=like.get_device()
    )


def _one_hot_axis(like: AbstractTensor, n: int, position) -> AbstractTensor:
    """A one at ``position`` along an axis of length ``n``."""

    return (_axis_index(like, n) == position).to_dtype(like.get_dtype())


def _row(M: AbstractTensor, selector: AbstractTensor) -> AbstractTensor:
    """The rows of ``M`` weighted by ``selector`` and summed.

    With a one-hot selector this reads one row; with the pivot mask it reads
    the selected row.  Either way it is a contraction over the row axis, not
    an indexed view, so the result's shape never depends on an index value.
    """

    return (selector.unsqueeze(-1) * M).sum(dim=-2)


def _column(M: AbstractTensor, selector: AbstractTensor) -> AbstractTensor:
    """The columns of ``M`` weighted by ``selector`` and summed."""

    return (M * selector.unsqueeze(-2)).sum(dim=-1)


def _first_occurrence(mask: AbstractTensor, dtype) -> AbstractTensor:
    """Reduce a comparison mask to its FIRST set position along the last axis.

    ``col == col.max(...)`` marks every tie, which is not an argmax.  A pivot
    mask with two ones makes the exchange below sum those rows instead of
    swapping one: equal magnitudes of the same sign survive that by accident,
    since the sum is still an invertible row operation applied to both sides,
    but ``|a| == |-a|`` cancels in the pivot column and the next division
    produces NaN.  The running sum is one at the first hit and greater after
    it, so this is the argmax the source always intended, in the same tensor
    vocabulary and with no data-dependent control flow.
    """

    return mask * (mask.cumsum(dim=-1) == 1).to_dtype(dtype)


def _pivot_mask(M: AbstractTensor, k) -> AbstractTensor:
    """One-hot over ALL rows: the largest ``|M[..., k]|`` at or below ``k``.

    The authored form sliced the column (``M[..., k:, k]``) and produced a
    mask of the sliced length, so both the view and the mask carried runtime
    extents.  Masking the full column with ``row >= k`` states the same
    selection at full width.
    """

    dtype = M.get_dtype()
    rows = M.get_shape()[-2]
    columns = M.get_shape()[-1]
    eligible = (_axis_index(M, rows) >= k).to_dtype(dtype)
    column = abs(_column(M, _one_hot_axis(M, columns, k)))
    # Rows above k are pushed below every magnitude, and magnitudes are >= 0.
    candidates = column * eligible - (1 - eligible)
    largest = candidates.max(dim=-1, keepdim=True)
    return _first_occurrence(
        (candidates == largest).to_dtype(dtype) * eligible, dtype
    )


def _masked_pivot_rows(M: AbstractTensor, k, pivot_mask: AbstractTensor):
    """Exchange row ``k`` with the one-hot selected row, as whole-matrix
    arithmetic.

    Each row of the result is one of three things -- the selected row (at
    ``k``), the old row ``k`` (wherever the mask is set), or itself -- and
    those three selectors partition the row axis, so the exchange is a sum of
    three masked terms rather than a loop of indexed writes.  When the mask
    selects ``k`` itself the first two terms both fire and ``keep`` is ``-1``
    there, cancelling the duplicate: no swap, which is what a self-selecting
    pivot means.

    ``M`` is not mutated; the caller rebinds.  That is deliberate.  The
    authored version wrote rows back through ``__setitem__`` inside a loop,
    which made one storage identity carry n+1 versions per pivot step.
    """

    dtype = M.get_dtype()
    n = M.get_shape()[-2]
    mask = pivot_mask.to_dtype(dtype)
    hot_k = _one_hot_axis(M, n, k)
    selected = _row(M, mask)
    original = _row(M, hot_k)
    keep = 1 - hot_k - mask
    exchanged = (
        hot_k.unsqueeze(-1) * selected.unsqueeze(-2)
        + mask.unsqueeze(-1) * original.unsqueeze(-2)
        + keep.unsqueeze(-1) * M
    )
    parity = (mask * hot_k).sum(dim=-1) * 2 - 1
    return exchanged, parity


def _lu_decompose_inplace(A: AbstractTensor):
    """
    Doolittle LU with partial pivoting.
    Returns ``(U, sign, P)``.  ``A`` is not modified; ``U`` is a fresh value.
    L is stored in the strictly lower part of U (unit diagonal implicit).
    ``sign`` is +1 or -1 representing permutation parity, and ``P`` is the
    permutation actually applied, as a matrix.  Returning ``P`` is what lets
    ``solve`` permute its right-hand side with one contraction instead of
    replaying the entire elimination a second time to rediscover the pivots.
    """
    U = A.clone()
    dtype = U.get_dtype()
    n = U.get_shape()[-1]
    index = _axis_index(U, n)
    # The identity, stated as a comparison rather than built by ``eye``, so it
    # needs no batch shape: the masked exchange broadcasts it against whatever
    # batch the pivot mask carries.
    permutation = (index.unsqueeze(-1) == index.unsqueeze(-2)).to_dtype(dtype)
    sign = (U * 0 + 1).sum(dim=-1).sum(dim=-1) * 0 + 1
    for k in range(n):
        pivot = _pivot_mask(U, k)
        U, parity = _masked_pivot_rows(U, k, pivot)
        permutation, _ = _masked_pivot_rows(permutation, k, pivot)
        sign = sign * parity
        hot_k = _one_hot_axis(U, n, k)
        after_k = (index > k).to_dtype(dtype)
        pivot_row = _row(U, hot_k)
        pivot_value = (pivot_row * hot_k).sum(dim=-1)
        # One rank-1 update replaces the authored ``for i in range(k+1, n)``
        # over ``U[..., i, k+1:]``: the row mask and the column mask carry the
        # bounds those slices used to.
        factor = (_column(U, hot_k) / pivot_value.unsqueeze(-1)) * after_k
        U = U - (
            factor.unsqueeze(-1)
            * pivot_row.unsqueeze(-2)
            * after_k.unsqueeze(-2)
        )
        # The stored L factor REPLACES column k below the diagonal.
        stored = after_k.unsqueeze(-1) * hot_k.unsqueeze(-2)
        U = U * (1 - stored) + factor.unsqueeze(-1) * stored
    return U, sign, permutation


def _forward_substitute(LU: AbstractTensor, b: AbstractTensor) -> AbstractTensor:
    """Solve Ly = Pb where L is the unit-lower part of LU and b is permuted.

    The loop over rows is genuinely sequential -- row i reads the rows solved
    before it -- so it stays.  Its inner ``LU[..., i, :i]`` slice does not:
    ``index < i`` selects the same elements at full width.
    """
    dtype = LU.get_dtype()
    n = LU.get_shape()[-1]
    index = _axis_index(LU, n)
    y = b.clone()
    for i in range(n):
        if i == 0:
            continue
        hot_i = _one_hot_axis(LU, n, i)
        prefix = (index < i).to_dtype(dtype)
        coefficients = _row(LU, hot_i) * prefix
        product = (coefficients.unsqueeze(-1) * y).sum(dim=-2)
        solved = _row(y, hot_i) - product
        y = (
            y * (1 - hot_i.unsqueeze(-1))
            + hot_i.unsqueeze(-1) * solved.unsqueeze(-2)
        )
    return y


def _back_substitute(LU: AbstractTensor, y: AbstractTensor) -> AbstractTensor:
    """Solve Ux = y where U is the upper part of LU."""
    dtype = LU.get_dtype()
    n = LU.get_shape()[-1]
    index = _axis_index(LU, n)
    x = y.clone()
    for i in range(n - 1, -1, -1):
        hot_i = _one_hot_axis(LU, n, i)
        suffix = (index > i).to_dtype(dtype)
        row_i = _row(LU, hot_i)
        product = ((row_i * suffix).unsqueeze(-1) * x).sum(dim=-2)
        diagonal = (row_i * hot_i).sum(dim=-1)
        solved = (_row(x, hot_i) - product) / diagonal.unsqueeze(-1)
        x = (
            x * (1 - hot_i.unsqueeze(-1))
            + hot_i.unsqueeze(-1) * solved.unsqueeze(-2)
        )
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
    # normalize b to (..., n, k)
    squeeze_vec = False
    if b.dim() == len(shpA) - 1:  # (..., n)
        b = b.reshape(tuple(list(shpA[:-1]) + [1]))
        squeeze_vec = True
    elif b.dim() == len(shpA):  # (..., n, k)
        pass
    else:
        raise ValueError("b must be (..., n) or (..., n, k)")
    # LU with partial pivoting, and the permutation it applied.  The authored
    # version discarded P and re-ran the whole elimination on a second copy
    # purely to rediscover the same pivot sequence for b; one contraction with
    # P states it directly, and halves the program a compiler has to carry.
    LU, _sign, permutation = _lu_decompose_inplace(A)
    y = _forward_substitute(LU, permutation.matmul(b))
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


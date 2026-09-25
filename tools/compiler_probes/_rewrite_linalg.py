"""Replace the LU/solve block of linalg.py with a fully tensorized one."""

from pathlib import Path

REPLACEMENT = '''def _axis_index(like: AbstractTensor, n: int) -> AbstractTensor:
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


'''

path = Path(__file__).resolve().parents[2] / "src/common/tensors/linalg.py"
text = path.read_text(encoding="utf-8")
start = text.index("def _masked_pivot_rows(")
end = text.index("def inv(A: AbstractTensor)")
path.write_text(text[:start] + REPLACEMENT + text[end:], encoding="utf-8")
print("rewrote the LU path")

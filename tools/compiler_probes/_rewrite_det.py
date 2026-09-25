"""Tensorize det's diagonal product and take the new three-value LU return."""

from pathlib import Path

OLD = '''    # general: LU with partial pivoting; det = sign(P) * prod(diag(U))
    U, piv_sign = _lu_decompose_inplace(A)
    diag = []
    for i in range(n):
        diag.append(U[..., i, i])
    prod = diag[0]
    for t in diag[1:]:
        prod = prod * t
    return prod * piv_sign
'''

NEW = '''    # general: LU with partial pivoting; det = sign(P) * prod(diag(U))
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
'''

path = Path(__file__).resolve().parents[2] / "src/common/tensors/linalg.py"
text = path.read_text(encoding="utf-8")
assert text.count(OLD) == 1, text.count(OLD)
path.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("tensorized det")

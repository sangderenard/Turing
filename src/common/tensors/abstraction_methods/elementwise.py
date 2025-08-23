# ---- Imports ----
from typing import Dict, Any
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..abstraction import AbstractTensor

# ---- scalar kernels for elementwise predicates/logicals/where ----
@staticmethod
def _scalar_kernel(op: str):
    tbl = {
        "equal":         lambda a, b, **k: bool(a == b),
        "not_equal":     lambda a, b, **k: bool(a != b),
        "less":          lambda a, b, **k: bool(a <  b),
        "less_equal":    lambda a, b, **k: bool(a <= b),
        "greater":       lambda a, b, **k: bool(a >  b),
        "greater_equal": lambda a, b, **k: bool(a >= b),
        "logical_and":   lambda a, b, **k: bool(bool(a) and bool(b)),
        "logical_or":    lambda a, b, **k: bool(bool(a) or  bool(b)),
        "logical_xor":   lambda a, b, **k: bool(bool(a) ^    bool(b)),
        "invert":        lambda a,     **k: bool(not bool(a)),
        "where":         lambda c, a, b, **k: (a if bool(c) else b),
    }
    if op not in tbl: raise NotImplementedError(op)
    return tbl[op]

@staticmethod
def _as_scalar(x):
    try: return x.item()  # 0-d tensor/np scalar
    except Exception: return x

# ---------------- v1: unary (NO implicit broadcast) ----------------
def _v1_valuewise(self, op: str, *, annotate: Dict[str, Any] | None = None):
    from ..abstraction import AbstractTensor
    finalize = AbstractTensor._pre_autograd(op, [self])
    flat = self.reshape(-1).tolist()
    K = self._scalar_kernel(op)
    out = [K(self._as_scalar(a)) for a in flat]
    out = self.ensure_tensor(out).reshape(*self.get_shape())
    out = finalize(out)
    tape = getattr(out, "_tape", None)
    if tape and annotate:
        tape.annotate(out, **({"eval_mode":"valuewise","v":"v1","length":len(flat)} | annotate))
    return out

# --------------- v2: binary (NO implicit broadcast) ----------------
def _v2_valuewise(
    self,
    op: str,
    other: "AbstractTensor | Any",
    *,
    allow_scalar: bool = True,
    annotate: Dict[str, Any] | None = None,
):
    from ..abstraction import AbstractTensor
    other_t = other if isinstance(other, AbstractTensor) else self.ensure_tensor(other)
    finalize = AbstractTensor._pre_autograd(op, [self, other_t])

    a = self.reshape(-1).tolist()
    b = other_t.reshape(-1).tolist()
    na, nb = len(a), len(b)
    lifted = {"left": False, "right": False}
    if na != nb:
        if not allow_scalar: raise ValueError(f"{op}: size mismatch {na} vs {nb}")
        if na == 1 and nb > 1: a, na, lifted["left"]  = [a[0]] * nb, nb, True
        elif nb == 1 and na > 1: b, nb, lifted["right"] = [b[0]] * na, na, True
        else: raise ValueError(f"{op}: incompatible lengths {na} vs {nb}")

    K = self._scalar_kernel(op)
    out = [K(self._as_scalar(a[i]), self._as_scalar(b[i])) for i in range(na)]
    out = self.ensure_tensor(out).reshape(*self.get_shape())
    out = finalize(out)
    tape = getattr(out, "_tape", None)
    if tape and annotate:
        tape.annotate(out, **({"eval_mode":"valuewise","v":"v2","length":na,"scalar_lift":lifted} | annotate))
    return out

# --------------- v3: ternary (where) (NO implicit broadcast) ---------------
def _v3_valuewise(
    self,
    op: str,  # "where"
    a: "AbstractTensor | Any",
    b: "AbstractTensor | Any",
    *,
    allow_scalar: bool = True,
    annotate: Dict[str, Any] | None = None,
):
    from ..abstraction import AbstractTensor
    a_t = a if isinstance(a, AbstractTensor) else self.ensure_tensor(a)
    b_t = b if isinstance(b, AbstractTensor) else self.ensure_tensor(b)
    finalize = AbstractTensor._pre_autograd(op, [self, a_t, b_t])

    c = self.reshape(-1).tolist()
    A = a_t.reshape(-1).tolist()
    B = b_t.reshape(-1).tolist()
    n = len(c)

    def lift(lst, name):
        if len(lst) == n: return lst, False
        if allow_scalar and len(lst) == 1: return [lst[0]] * n, True
        raise ValueError(f"{op}: {name} length {len(lst)} != {n}")

    A, liftA = lift(A, "a"); B, liftB = lift(B, "b")

    K = self._scalar_kernel("where")
    out = [K(self._as_scalar(c[i]), self._as_scalar(A[i]), self._as_scalar(B[i])) for i in range(n)]
    out = self.ensure_tensor(out).reshape(*self.get_shape())
    out = finalize(out)
    tape = getattr(out, "_tape", None)
    if tape and annotate:
        tape.annotate(out, **({"eval_mode":"valuewise","v":"v3","length":n,"scalar_lift":{"a":liftA,"b":liftB}} | annotate))
    return out

# ----------------- Tiny user-facing shims (preserve real op names) ----------
def __eq__(self, other):         return self._v2_valuewise("equal", other, annotate={"op":"equal"})
def __ne__(self, other):         return self._v2_valuewise("not_equal", other, annotate={"op":"not_equal"})
def __lt__(self, other):         return self._v2_valuewise("less", other, annotate={"op":"less"})
def __le__(self, other):         return self._v2_valuewise("less_equal", other, annotate={"op":"less_equal"})
def __gt__(self, other):         return self._v2_valuewise("greater", other, annotate={"op":"greater"})
def __ge__(self, other):         return self._v2_valuewise("greater_equal", other, annotate={"op":"greater_equal"})

def __and__(self, other):        return self._v2_valuewise("logical_and", other, annotate={"op":"logical_and"})
def __or__(self, other):         return self._v2_valuewise("logical_or",  other, annotate={"op":"logical_or"})
def __xor__(self, other):        return self._v2_valuewise("logical_xor", other, annotate={"op":"logical_xor"})
def __invert__(self):            return self._v1_valuewise("invert", annotate={"op":"invert"})

@staticmethod
def where(cond, a, b, *, allow_scalar: bool = True):
    if not isinstance(cond, AbstractTensor):
        raise TypeError("AbstractTensor.where expects first arg to be an AbstractTensor condition")
    return cond._v3_valuewise("where", a, b, allow_scalar=allow_scalar, annotate={"op":"where"})

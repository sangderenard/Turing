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

    # Explicitly handle zero-length operands to avoid div-by-zero
    if na == 0 or nb == 0:
        if na == nb == 0:
            shape = self.get_shape()
        elif allow_scalar and ((na == 0 and nb == 1) or (nb == 0 and na == 1)):
            shape = self.get_shape() if na == 0 else other_t.get_shape()
        else:
            raise ValueError(f"{op}: incompatible lengths {na} vs {nb}")
        out = self.ensure_tensor([]).reshape(*shape)
        out = finalize(out)
        tape = getattr(out, "_tape", None)
        if tape and annotate:
            tape.annotate(out, **({"eval_mode":"valuewise","v":"v2","length":0,"scalar_lift":{"left":False,"right":False}} | annotate))
        return out

    target = max(na, nb)

    def lift(lst, name):
        if len(lst) == target:
            return lst, False
        if allow_scalar and len(lst) == 1:
            return [lst[0]] * target, True
        if target % len(lst) == 0:
            k = target // len(lst)
            return [lst[i // k] for i in range(target)], True
        raise ValueError(f"{op}: incompatible lengths {na} vs {nb}")

    a, left_lift = lift(a, "left")
    b, right_lift = lift(b, "right")
    lifted = {"left": left_lift, "right": right_lift}

    K = self._scalar_kernel(op)
    out = [K(self._as_scalar(a[i]), self._as_scalar(b[i])) for i in range(target)]
    shape = self.get_shape() if na == target else other_t.get_shape()
    out = self.ensure_tensor(out).reshape(shape)
    out = finalize(out)
    tape = getattr(out, "_tape", None)
    if tape and annotate:
        tape.annotate(out, **({"eval_mode":"valuewise","v":"v2","length":target,"scalar_lift":lifted} | annotate))
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

    target = max(len(c), len(A), len(B))

    def lift(lst, name, *, allow_div: bool = False):
        if len(lst) == target:
            return lst, False
        if allow_div and len(lst) > 0 and target % len(lst) == 0:
            k = target // len(lst)
            return [lst[i // k] for i in range(target)], True
        if allow_scalar and len(lst) == 1:
            return [lst[0]] * target, True
        raise ValueError(f"{op}: {name} length {len(lst)} != {target}")

    A, liftA = lift(A, "a")
    B, liftB = lift(B, "b")
    c, liftC = lift(c, "cond", allow_div=True)

    K = self._scalar_kernel("where")
    out = [K(self._as_scalar(c[i]), self._as_scalar(A[i]), self._as_scalar(B[i])) for i in range(target)]
    # Result shape follows whichever operand carried the target length
    shape = a_t.get_shape() if len(A) == target else b_t.get_shape()
    out = self.ensure_tensor(out).reshape(*shape)
    out = finalize(out)
    tape = getattr(out, "_tape", None)
    if tape and annotate:
        tape.annotate(out, **({"eval_mode":"valuewise","v":"v3","length":target,"scalar_lift":{"a":liftA,"b":liftB,"cond":liftC}} | annotate))
    return out

# ---------------------- elementwise max/min helpers -------------------------
def maximum(self, other):
    """Elementwise maximum with automatic promotion."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    other_arg = other.data if isinstance(other, AbstractTensor) else other
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.maximum_(other_arg)
    return result


def minimum(self, other):
    """Elementwise minimum with automatic promotion."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    other_arg = other.data if isinstance(other, AbstractTensor) else other
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.minimum_(other_arg)
    return result

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
    from ..abstraction import AbstractTensor
    if not isinstance(cond, AbstractTensor):
        raise TypeError("AbstractTensor.where expects first arg to be an AbstractTensor condition")
    return cond._v3_valuewise("where", a, b, allow_scalar=allow_scalar, annotate={"op":"where"})

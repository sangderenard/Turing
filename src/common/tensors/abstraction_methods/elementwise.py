# ---- Imports ----
from typing import Dict, Any
try:
    from ..branch_oracle import BRANCH_ORACLE as _BRANCH_ORACLE
except Exception:  # fallback if module not available
    _BRANCH_ORACLE = None  # type: ignore
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
        # Bitwise integer ops. These are what Python's ``&``/``|``/``^``/
        # ``<<``/``>>``/``~`` mean by language definition -- distinct from the
        # ``logical_*`` connectives above, which coerce to bool first. Operands
        # arrive as plain Python scalars from ``tolist()``, so Python's own
        # arbitrary-precision integer operators are the reference semantics
        # every backend must agree with.
        "bitand":        lambda a, b, **k: int(a) & int(b),
        "bitor":         lambda a, b, **k: int(a) | int(b),
        "bitxor":        lambda a, b, **k: int(a) ^ int(b),
        "shl":           lambda a, b, **k: int(a) << int(b),
        "shr":           lambda a, b, **k: int(a) >> int(b),
        "invert":        lambda a,     **k: ~int(a),
        "logical_not":   lambda a,     **k: bool(not bool(a)),
        "where":         lambda c, a, b, **k: (a if bool(c) else b),
        "sign":          lambda a,     **k: (a if (a != a) else (1 if a > 0 else (-1 if a < 0 else 0)))
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
    # Branch override: if oracle forces this predicate, return full-mask tensor
    if _BRANCH_ORACLE is not None:
        forced = _BRANCH_ORACLE.maybe_mask(op)
        if forced is not None:
            mask_val = 1 if forced else 0
            out = self.ensure_tensor([mask_val] * max(1, self.numel())).reshape(*self.get_shape())
            out = finalize(out)
            tape = getattr(out, "_tape", None)
            if tape and annotate:
                tape.annotate(out, **({"forced": True} | annotate))
            return out
    flat = self.reshape(-1).tolist()
    K = self._scalar_kernel(op)
    # tolist() already guarantees plain Python scalars (that is its whole
    # contract, on every backend) -- routing each one through _as_scalar's
    # try/except .item() is a no-op that only pays for the failed attempt.
    # Across a real-sized tensor that failed-exception cost, not the
    # comparison itself, was the actual bottleneck.
    out = [K(a) for a in flat]
    if not out and op in {
        # Predicate/logical unaries produce a boolean empty result; ``invert``
        # is bitwise (``~int``) and produces an integer, so it keeps the
        # input's own dtype rather than being forced to bool.
        "logical_not", "isfinite", "isinf", "isnan",
    }:
        out = type(self).tensor(
            [], dtype="bool", tape=getattr(self, "_tape", None)
        ).reshape(*self.get_shape())
    else:
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
    # Branch override: if oracle forces this predicate, return full-mask tensor
    if _BRANCH_ORACLE is not None:
        forced = _BRANCH_ORACLE.maybe_mask(op)
        if forced is not None:
            # Result shape follows broadcast rules; compute as below but with constants
            a = self.reshape(-1).tolist(); b = other_t.reshape(-1).tolist()
            target = max(len(a), len(b)) or 1
            shape = self.get_shape() if len(a) == target else other_t.get_shape()
            mask_val = 1 if forced else 0
            out = self.ensure_tensor([mask_val] * target).reshape(*shape)
            out = finalize(out)
            tape = getattr(out, "_tape", None)
            if tape and annotate:
                tape.annotate(out, **({"forced": True} | annotate))
            return out

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
        boolean_result = op in {
            "equal", "not_equal", "less", "less_equal", "greater",
            "greater_equal", "logical_and", "logical_or", "logical_xor",
        }
        out = type(self).tensor(
            [],
            dtype="bool" if boolean_result else getattr(self, "dtype", None),
            tape=getattr(self, "_tape", None),
        ).reshape(*shape)
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
    # Same reasoning as _v1_valuewise: a/b came from tolist(), so they are
    # already plain scalars -- the per-element _as_scalar try/except was
    # pure waste, not part of the comparison's actual cost.
    out = [K(x, y) for x, y in zip(a, b)]
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
    # Branch override: if oracle forces 'where', we still evaluate both a/b but gate with constant cond
    if _BRANCH_ORACLE is not None:
        forced = _BRANCH_ORACLE.maybe_mask(op)
        if forced is not None:
            # shape follows standard lift below
            c = self.reshape(-1).tolist(); A = a_t.reshape(-1).tolist(); B = b_t.reshape(-1).tolist()
            target = max(len(c), len(A), len(B)) or 1
            shape = a_t.get_shape() if len(A) == target else b_t.get_shape()
            # Evaluate branches normally
            outA = a_t
            outB = b_t
            out = outA if forced else outB
            out = finalize(out)
            tape = getattr(out, "_tape", None)
            if tape and annotate:
                tape.annotate(out, **({"forced": True} | annotate))
            return out

    c = self.reshape(-1).tolist()
    A = a_t.reshape(-1).tolist()
    B = b_t.reshape(-1).tolist()

    orig_len_c = len(c)
    orig_len_a = len(A)
    orig_len_b = len(B)

    target = max(orig_len_c, orig_len_a, orig_len_b)

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
    # Same reasoning as _v2_valuewise: c/A/B came from tolist(), already
    # plain scalars -- skip the no-op try/except unwrap per element.
    out = [K(*triple) for triple in zip(c, A, B)]

    # Determine result shape using the operand that originally carried the target length
    if orig_len_a == target and orig_len_a != 1:
        shape = a_t.get_shape()
    elif orig_len_b == target and orig_len_b != 1:
        shape = b_t.get_shape()
    elif orig_len_c == target:
        shape = self.get_shape()
    else:
        shape = (target,)

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
    if not isinstance(other, AbstractTensor):
        other = AbstractTensor.tensor(other)
    finalize = AbstractTensor._pre_autograd("maximum", [self, other])
    other_arg = other.data
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.maximum_(other_arg)
    return finalize(result)


def minimum(self, other):
    """Elementwise minimum with automatic promotion."""
    from ..abstraction import AbstractTensor
    if not isinstance(self, AbstractTensor):
        self = AbstractTensor.tensor(self)
    if not isinstance(other, AbstractTensor):
        other = AbstractTensor.tensor(other)
    finalize = AbstractTensor._pre_autograd("minimum", [self, other])
    other_arg = other.data
    result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
    result.data = self.minimum_(other_arg)
    return finalize(result)

# ----------------- Tiny user-facing shims (preserve real op names) ----------
def __eq__(self, other):         return self._v2_valuewise("equal", other, annotate={"op":"equal"})
def __ne__(self, other):         return self._v2_valuewise("not_equal", other, annotate={"op":"not_equal"})
def __lt__(self, other):         return self._v2_valuewise("less", other, annotate={"op":"less"})
def __le__(self, other):         return self._v2_valuewise("less_equal", other, annotate={"op":"less_equal"})
def __gt__(self, other):         return self._v2_valuewise("greater", other, annotate={"op":"greater"})
def __ge__(self, other):         return self._v2_valuewise("greater_equal", other, annotate={"op":"greater_equal"})

# ``&``/``|``/``^``/``~`` carry two meanings in a tensor system, and which one
# applies is decided by dtype -- exactly as NumPy/PyTorch decide it:
#
#   * On boolean operands they are the *logical* connectives. This is the
#     mask algebra existing tensor code relies on (``mask_a & mask_b``), and
#     it must keep both its logical semantics and its boolean result dtype.
#   * On integer operands they are *bitwise* -- real bit manipulation, which
#     is what a byte/bitfield decoder (PE parsing, Huffman packing) needs.
#
# For ``&``/``|``/``^`` the two readings coincide on booleans (``1 & 1 == 1``),
# so the only thing dtype-dispatch protects there is the result dtype (bool in,
# bool out). ``~`` genuinely differs -- logical ``~True`` is ``False`` while
# bitwise ``~1`` is ``-2`` -- so its dispatch is load-bearing, not cosmetic.
# ``<<``/``>>`` have no logical reading at all and are always bitwise.
def _both_bool(self, other) -> bool:
    from ..abstraction import AbstractTensor
    if not _is_bool_like(self):
        return False
    if isinstance(other, bool):
        return True
    if isinstance(other, AbstractTensor):
        return _is_bool_like(other)
    # A raw Python int/float operand is not boolean -> integer (bitwise) result.
    return False


def _is_bool_like(tensor) -> bool:
    # ``get_dtype()`` is normalized to a torch-style dtype (``torch.bool``)
    # even on the NumPy backend, while ``bool_dtype`` is that backend's own
    # sentinel (``numpy.bool``) -- so an identity/equality check misses. A
    # string match is what the rest of the abstraction already uses to
    # classify dtype families (see ``_apply_operator``'s ``kind``).
    try:
        dt = tensor.get_dtype()
    except Exception:
        return False
    if dt is None:
        return False
    try:
        if dt == tensor.bool_dtype:
            return True
    except Exception:
        pass
    return "bool" in str(dt).lower()


def __and__(self, other):
    if _both_bool(self, other):
        return self._v2_valuewise("logical_and", other, annotate={"op":"logical_and"})
    return self._v2_valuewise("bitand", other, annotate={"op":"bitand"})
def __or__(self, other):
    if _both_bool(self, other):
        return self._v2_valuewise("logical_or", other, annotate={"op":"logical_or"})
    return self._v2_valuewise("bitor", other, annotate={"op":"bitor"})
def __xor__(self, other):
    if _both_bool(self, other):
        return self._v2_valuewise("logical_xor", other, annotate={"op":"logical_xor"})
    return self._v2_valuewise("bitxor", other, annotate={"op":"bitxor"})
def __lshift__(self, other):     return self._v2_valuewise("shl", other, annotate={"op":"shl"})
def __rshift__(self, other):     return self._v2_valuewise("shr", other, annotate={"op":"shr"})
def __invert__(self):
    if _is_bool_like(self):
        return self._v1_valuewise("logical_not", annotate={"op":"logical_not"})
    return self._v1_valuewise("invert", annotate={"op":"invert"})

@staticmethod
def where(cond, a, b, *, allow_scalar: bool = True):
    from ..abstraction import AbstractTensor
    if not isinstance(cond, AbstractTensor):
        raise TypeError("AbstractTensor.where expects first arg to be an AbstractTensor condition")
    return cond._v3_valuewise("where", a, b, allow_scalar=allow_scalar, annotate={"op":"where"})

def sign(self):
    return self._v1_valuewise("sign", annotate={"op":"sign"})

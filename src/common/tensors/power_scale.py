"""A power-of-two scale on every AbstractTensor handle, free unless used.

Every handle carries ``scale_exponent`` (class default 0) and ``track_scale``
(class default False).  The value a handle denotes is::

    logical = data * 2**scale_exponent

A power of two is the one rescaling that is always exact -- it moves the
exponent and touches no mantissa bit -- so a scale only ever changes RANGE,
never precision.

WHY THIS COSTS NOTHING UNLESS USED.  A handle only acquires the machinery
when someone gives it a nonzero scale or asks it to track one: its class is
then swapped for a cached subclass of its own backend class
(``Scaled<Backend>``).  Unscaled handles keep their ordinary class, so the
one seam every operator passes through (``AbstractTensor._apply_operator``)
bypasses on a type check, and nothing else in the tensor surface is touched.

HOW A SCALED HANDLE STAYS CORRECT EVERYWHERE.

* Arithmetic (``_apply_operator``) carries the scale exactly: ``*`` adds
  exponents, ``/`` subtracts them, ``+``/``-``/max/min/comparisons align to
  the larger exponent, negation/abs keep it, an integer power multiplies it.
  Anything else is materialized first.
* Structural and linear methods (reshape, transpose, index, sum, cumsum,
  mean, clone, ...) run on the raw data and the result keeps the scale.
* Every other read of a scaled handle's ``data`` returns the MATERIALIZED
  value (``data * 2**scale`` folded in), so any method that knows nothing
  about scales computes on the correct logical values.  Materializing a value
  that would overflow or underflow the element is refused by name -- never
  an inf, never a silent flush to zero.

OVERFLOW PROTECTION is for handles that opt in (``track_scale=True``): their
operands are normalized by an exact power of two before products and
quotients (so the product of the raw data cannot overflow), and every result
is rebalanced so its largest magnitude sits in [1, 2).

The scale is purely numeric.  A different reading of the bits (factorial
radix, gray code, ...) is an ENCODING, a separate field, not a scale value.

Not carried: autograd through a scaled operation (the tape records the raw
operations), and backend-compiled code (compiled tensors have scale 0, the
class default, which is the bypass).
"""

from __future__ import annotations

import math
from typing import Any

from .abstraction import AbstractTensor


#: Largest and smallest normal magnitudes of binary64; materialization
#: refuses to leave this range.
_FLOAT_MAX_EXPONENT = 1023
_FLOAT_MIN_NORMAL_EXPONENT = -1022

# --------------------------------------------------------------------------
# THE SCALE RULE OF EVERY CANONICAL OPERATION
#
# ``operator_catalog.CANONICAL_ABSTRACT_TENSOR_OPERATORS`` owns WHICH
# operations exist; this table owns what each does to a power-of-two scale.
# The audit at the bottom of this module refuses to import if a canonical
# operation has no rule, so the vocabulary cannot drift past the table.
#
#   preserve     structural, linear or order-based: run on the raw data, the
#                result keeps the operand's exponent
#   exponent     the result's exponent is a function of the operands'
#                (``_EXPONENT_RULES``)
#   align        several scaled operands: bring them to the larger exponent,
#                then operate (the result keeps it; a boolean result has none)
#   free         the result does not depend on a positive scale (signs,
#                indices, predicates, accessors): run on the raw data, scale 0
#   settle       in-place writes: the written value is aligned to the
#                destination's exponent, so no write lands on a temporary
#   materialize  everything nonlinear (transcendentals, rounding, integer or
#                boolean casts, the host boundary): read the logical value,
#                refusing overflow/underflow by name
#   create       a new tensor: exponent 0

SCALE_RULES: dict[str, str] = {}


def _declare(rule: str, *names: str) -> None:
    for name in names:
        if name in SCALE_RULES and SCALE_RULES[name] != rule:
            raise ValueError(f"{name!r} declared twice: {SCALE_RULES[name]} and {rule}")
        SCALE_RULES[name] = rule


_declare("preserve",
         # structure
         "reshape", "view", "view_flat", "flatten", "transpose", "permute", "unsqueeze",
         "squeeze", "swapaxes", "repeat", "repeat_interleave", "expand", "broadcast_to",
         "broadcast_rows", "diag", "pad", "pad2d", "fold2d", "fold3d", "unfold2d", "unfold3d",
         "split", "unstack", "gather", "gather_and", "index_select", "select_by_indices",
         "boolean_mask_select", "interpolate",
         # linear / order-based reductions and transforms
         "sum", "cumsum", "mean", "max", "min", "trace", "norm", "percentile",
         "fft", "ifft", "rfft", "irfft", "deg2rad", "rad2deg",
         "neg", "abs", "real", "imag",
         # value lifecycle, float-to-float type and device moves
         "clone", "copy", "detach", "ensure_tensor", "astype", "to_dtype", "float", "double",
         "cast_like", "to", "to_backend", "to_device", "cpu", "cuda")
_declare("exponent",
         "mul", "truediv", "pow", "sqrt", "cbrt", "inv", "inverse", "prod", "det",
         "matmul", "dot", "outer", "cross", "einsum", "solve", "cholesky")
_declare("align",
         "add", "sub", "maximum", "minimum", "clamp", "clamp_min", "clamp_max", "clip",
         "where", "cat", "concat", "concatenate", "stack", "pad_cat",
         "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
         "allclose", "searchsorted")
_declare("free",
         "sign", "argmax", "argmin", "argwhere", "nonzero", "topk", "unravel_index",
         "isfinite", "isinf", "isnan", "isinfinite", "logical_not", "logical_and",
         "logical_or", "all", "any",
         "datastring", "device", "dim", "dtype", "get_device", "get_dtype", "get_ndims",
         "get_shape", "nbytes", "ndim", "ndims", "numel", "shape", "tensor_type")
_declare("settle",
         "assign_at_indices", "copyto", "increment_at_indices", "scatter", "scatter_and",
         "scatter_row")
_declare("materialize",
         # transcendental and nonlinear elementwise
         "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
         "asinh", "acosh", "atanh", "sigmoid", "erf", "sinc", "softmax", "log_softmax",
         "coth", "cot", "csch", "csc", "sec", "sech",
         # rounding, integer arithmetic, bit patterns, integer and boolean casts
         "floor", "ceil", "round", "trunc", "int_trunc", "floordiv", "mod",
         "bitand", "bitor", "bitxor", "shl", "shr", "invert",
         "fptosi", "fptoui", "sitofp", "uitofp", "sext", "zext",
         "int", "long", "long_cast", "bool", "nan_to_num",
         # results a scale cannot describe: (values, indices) and eigen pairs
         "eigh",
         # the host boundary and scalar readout
         "item", "numpy", "tolist", "tobytes", "save", "load", "jpg", "avi", "mjpeg_frames")
_declare("create",
         "arange", "empty", "eye", "eye_like", "from_nested", "full", "full_like", "get_tensor",
         "hanning", "linspace", "meshgrid", "ones", "ones_like", "rand_like", "randint",
         "randint_like", "randn", "random_source", "random_tensor", "randoms", "range",
         "sparse_coo_tensor", "tensor", "tensor_from_list", "tensor_like", "zeros",
         "zeros_like", "pi", "long_pi", "fftfreq", "rfftfreq")

#: Method-level ``exponent`` rules: (self exponent, the call, its result) -> result exponent.
def _count_reduced(receiver, result) -> int:
    before = 1
    for extent in tuple(raw_view(receiver).shape):
        before *= int(extent)
    after = 1
    for extent in tuple(getattr(result, "shape", ()) or ()):
        after *= int(extent)
    return before // max(after, 1)


def _matrix_order(receiver) -> int:
    return int(tuple(raw_view(receiver).shape)[-1])

#: Arithmetic whose result exponent follows exactly from the operands'.
_ALIGNED = frozenset({"add", "sub", "maximum", "minimum"})
_COMPARE = frozenset({"less", "less_equal", "greater", "greater_equal",
                      "equal", "not_equal", "lt", "le", "gt", "ge", "eq", "ne"})

_SCALED_CLASSES: dict[type, type] = {}


# --------------------------------------------------------------------------
# class swapping


def _raw_data(tensor: AbstractTensor) -> Any:
    return object.__getattribute__(tensor, "data")


def _exp(value) -> AbstractTensor:
    """An exponent as a 0-d AbstractTensor (integral values held in float64)."""

    if isinstance(value, AbstractTensor):
        return value
    return AbstractTensor.get_tensor(float(value))


def _scale(tensor: Any) -> AbstractTensor:
    """A handle's exponent as a 0-d tensor (0 for anything unscaled)."""

    if not isinstance(tensor, AbstractTensor):
        return _exp(0)
    held = object.__getattribute__(tensor, "__dict__").get("scale_exponent")
    return _exp(0) if held is None else _exp(held)


def _tmax(exponents) -> AbstractTensor:
    exponents = [_exp(e) for e in exponents]
    target = exponents[0]
    for other in exponents[1:]:
        target = target.maximum(other)
    return target


def _tracked(tensor: Any) -> bool:
    if not isinstance(tensor, AbstractTensor):
        return False
    return bool(object.__getattribute__(tensor, "__dict__").get("track_scale", False))


def _is_scaled_class(cls: type) -> bool:
    return bool(cls.__dict__.get("_is_scaled_class", False))


def unscaled_class(cls: type) -> type:
    return cls.__dict__.get("_unscaled_class", cls) if _is_scaled_class(cls) else cls


def _keep(result, exponent: int, track: bool):
    if isinstance(result, AbstractTensor):
        return set_scale(result, exponent, track=track)
    if isinstance(result, (list, tuple)):
        return type(result)(_keep(item, exponent, track) for item in result)
    if _is_number(result):
        # A bare scalar has nowhere to hold an exponent; dropping it would be
        # the silent loss this module exists to prevent.  On the scaled path
        # it becomes a 0-d handle that carries it.
        return set_scale(AbstractTensor.get_tensor(result), exponent, track=track)
    return result


def _is_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return type(value).__module__ == "numpy" and hasattr(value, "dtype") and getattr(value, "shape", None) == ()


def _method_exponent(name: str, exponent: int, receiver, args, result) -> int | None:
    """The result exponent of a scaled receiver's ``exponent``-rule method, or
    None when the rule needs the logical value (then the call materializes)."""

    tensors = [a for a in args if isinstance(a, AbstractTensor)]
    if name in {"mul", "matmul", "dot", "outer", "cross", "einsum"}:
        for t in tensors:
            exponent = exponent + _scale(t)
        return exponent
    if name == "truediv":
        for t in tensors:
            exponent = exponent - _scale(t)
        return exponent
    if name in {"inv", "inverse"}:
        return -exponent
    if name in {"prod"}:
        return exponent * _count_reduced(receiver, result)
    if name == "det":
        return exponent * _matrix_order(receiver)
    if name == "pow" and args and isinstance(args[0], int) and not isinstance(args[0], bool):
        return exponent * int(args[0])
    return None


def _scaled_getattribute(self, name):
    value = object.__getattribute__(self, name)
    if name == "data":
        exponent = _scale(self)
        return value if exponent == 0 else _fold(self, value, exponent)
    if name.startswith("_") or not callable(value):
        return value
    if (object.__getattribute__(self, "__dict__").get("encoding", "binary") != "binary"
            and name in SCALE_RULES):
        # Only the canonical surface is gated; backend hooks (``shape_``, ...)
        # are the implementation the canonical operations call.
        return _encoding_gate(self, name, value)
    rule = SCALE_RULES.get(name)
    if rule is None or rule in {"materialize", "create"}:
        return value
    exponent = _scale(self)
    track = _tracked(self)
    raw_method = getattr(raw_view(self), name)
    if rule == "preserve":
        return lambda *a, **k: _keep(raw_method(*a, **k), exponent, track)
    if rule == "free":
        return raw_method
    if rule == "exponent":
        if name in {"sqrt", "cbrt"}:
            root = 2 if name == "sqrt" else 3

            def rooted(*a, **k):
                shift = exponent - (exponent / root).floor() * root   # exponent mod root, exactly
                shifted = _shift_raw(raw_view(self), shift)
                return _keep(getattr(shifted, name)(*a, **k), (exponent - shift) / root, track)

            return rooted

        def mapped(*a, **k):
            raw_args = [raw_view(x) if isinstance(x, AbstractTensor) else x for x in a]
            result = raw_method(*raw_args, **k)
            out = _method_exponent(name, exponent, self, a, result)
            if out is None:
                return value(*a, **k)
            return _keep(result, out, track)

        return mapped
    if rule == "align":
        def aligned(*a, **k):
            target = _tmax([exponent, *(_scale(x) for x in a if isinstance(x, AbstractTensor))])
            raw_args = [_align(raw_view(x), _scale(x), target) if isinstance(x, AbstractTensor)
                        else (_align(x, 0, target) if _is_number(x) else x)
                        for x in a]
            result = getattr(_align(raw_view(self), exponent, target), name)(*raw_args, **k)
            dtype = str(getattr(result, "dtype", "") or "")
            return result if "bool" in dtype else _keep(result, target, track)

        return aligned
    if rule == "settle":
        def settled(*a, **k):
            written = [_align(raw_view(x), _scale(x), exponent) if isinstance(x, AbstractTensor) else x
                       for x in a]
            return raw_method(*written, **k)

        return settled
    return value


def _scaled_setitem(self, key, value):
    """Write into the raw data: the value is aligned to this handle's exponent."""

    exponent = _scale(self)
    view = raw_view(self)
    if isinstance(value, AbstractTensor):
        value = _align(raw_view(value), _scale(value), exponent)
    elif _is_number(value):
        value = _pow2(-exponent) * value
    view[key] = value
    object.__setattr__(self, "data", _raw_data(view))


# --------------------------------------------------------------------------
# encoding: a separate field, default "binary" (the bypass)

#: encoding name -> {operation: handler(tensor, method, *args, **kwargs)}.  An
#: encoding other than binary supports exactly what it registers here; every
#: other operation is refused by name ("declared, never inferred").
ENCODING_RULES: dict[str, dict[str, Any]] = {}


def register_encoding(encoding: str, operations: dict) -> None:
    ENCODING_RULES.setdefault(str(encoding), {}).update(operations)


def _encoding_gate(self, name, value):
    encoding = object.__getattribute__(self, "__dict__").get("encoding")
    if SCALE_RULES.get(name) == "free" and name not in {"sign", "argmax", "argmin", "topk"}:
        return value                                        # accessors: shape, dtype, ...
    handler = ENCODING_RULES.get(encoding, {}).get(name)
    if handler is None:
        raise TypeError(
            f"{name!r} is not declared for encoding {encoding!r}; register it with "
            "power_scale.register_encoding or decode to binary first")
    return lambda *a, **k: handler(self, value, *a, **k)


def set_encoding(tensor: AbstractTensor, encoding: str) -> AbstractTensor:
    state = object.__getattribute__(tensor, "__dict__")
    if encoding == "binary":
        state.pop("encoding", None)
        return set_scale(tensor, _scale(tensor))
    state["encoding"] = str(encoding)
    tensor.__class__ = scaled_class(type(tensor))
    return tensor


def _scaled_getitem(self, key):
    result = raw_view(self)[key]
    if isinstance(result, AbstractTensor):
        set_scale(result, _scale(self), track=_tracked(self))
    return result


def _aligned_compare(name: str):
    """A comparison dunder for scaled handles: align to the larger exponent,
    compare the raw data (comparisons do not pass through _apply_operator)."""

    def compare(self, other):
        lexp, rexp = _scale(self), _scale(other)
        target = _tmax([lexp, rexp])
        left = _align(raw_view(self), lexp, target)
        right = _align(raw_view(other) if isinstance(other, AbstractTensor) else other, rexp, target)
        return getattr(left, name)(right)

    compare.__name__ = name
    return compare


_COMPARISON_DUNDERS = ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__")


def scaled_class(cls: type) -> type:
    """The cached ``Scaled<Backend>`` subclass of an unscaled backend class."""

    base = unscaled_class(cls)
    held = _SCALED_CLASSES.get(base)
    if held is None:
        held = type(f"Scaled{base.__name__}", (base,), {
            "_is_scaled_class": True,
            "_unscaled_class": base,
            "__getattribute__": _scaled_getattribute,
            "__getitem__": _scaled_getitem,
            "__setitem__": _scaled_setitem,
            "__hash__": base.__hash__,
            **{name: _aligned_compare(name) for name in _COMPARISON_DUNDERS},
        })
        _SCALED_CLASSES[base] = held
    return held


# --------------------------------------------------------------------------
# the handle-level surface


def set_scale(tensor: AbstractTensor, exponent, *, track: bool | None = None) -> AbstractTensor:
    """Set a handle's power-of-two exponent (and optionally its tracking), in place.

    The exponent is stored as a 0-d AbstractTensor.  Only a literal ``0`` with
    no tracking and no encoding returns the handle to its ordinary class; an
    exponent computed at run time keeps the scaled class, whatever its value.
    """

    state = object.__getattribute__(tensor, "__dict__")
    if track is not None:
        state["track_scale"] = bool(track)
    tracking = bool(state.get("track_scale", False))
    literal_zero = (not isinstance(exponent, AbstractTensor)) and int(exponent) == 0
    if literal_zero and not tracking and "encoding" not in state:
        state.pop("scale_exponent", None)
        tensor.__class__ = unscaled_class(type(tensor))
    else:
        state["scale_exponent"] = _exp(exponent)
        tensor.__class__ = scaled_class(type(tensor))
    return tensor


def raw_view(tensor: AbstractTensor) -> AbstractTensor:
    """An unscaled handle over the same raw data (the mantissas, scale ignored)."""

    base = unscaled_class(type(tensor))
    if base is type(tensor):
        return tensor
    view = base(track_time=getattr(tensor, "track_time", False), tape=getattr(tensor, "_tape", None))
    view.data = _raw_data(tensor)
    return view


#: A shift is applied in at most this many clamped steps of 2**1000, so an
#: exponent gap up to 4000 is exact (float64's whole range is about 2100).
_SHIFT_STEPS = 4
_LN2 = math.log(2.0)


def _pow2(exponent) -> AbstractTensor:
    return AbstractTensor.get_tensor(2.0) ** _exp(exponent)


def _shift_raw(view, exponent) -> AbstractTensor:
    """``view * 2**exponent`` in exact power-of-two steps (each exact while normal)."""

    remaining = _exp(exponent)
    result = view
    for _ in range(_SHIFT_STEPS):
        step = remaining.clamp(-1000.0, 1000.0)
        result = result * _pow2(step)
        remaining = remaining - step
    return result


def _log2_floor(magnitude: AbstractTensor) -> AbstractTensor:
    """floor(log2(m)) for m > 0 (0 where m == 0).  A log may land one off near
    a power of two; every shift built from it is still an exact power of two."""

    positive = magnitude > 0.0
    safe = AbstractTensor.where(positive, magnitude, magnitude * 0.0 + 1.0)
    return AbstractTensor.where(positive, (safe.log() / _LN2).floor(), magnitude * 0.0)


def max_abs(tensor: AbstractTensor) -> AbstractTensor:
    """Largest raw magnitude (scale ignored), as a 0-d tensor."""

    return abs(raw_view(tensor)).max()


def _smallest_nonzero(view: AbstractTensor) -> AbstractTensor:
    magnitude = abs(view)
    return AbstractTensor.where(magnitude > 0.0, magnitude, magnitude * 0.0 + math.inf).min()


def _fold(tensor: AbstractTensor, raw: Any, exponent) -> Any:
    """The raw data with ``2**exponent`` folded in, refusing to leave the element's range."""

    view = raw_view(tensor)
    exponent = _exp(exponent)
    top = _log2_floor(max_abs(tensor)) + exponent
    if bool(top >= float(_FLOAT_MAX_EXPONENT)):
        raise OverflowError(
            "materializing a scaled tensor overflows float64 (its largest value reaches "
            "about 2**" + str(top.tolist()) + "); keep it scaled or collapse a smaller piece")
    smallest = _smallest_nonzero(view)
    if bool(smallest < math.inf) and bool(
            _log2_floor(smallest) + exponent <= float(_FLOAT_MIN_NORMAL_EXPONENT)):
        raise ArithmeticError(
            "materializing a scaled tensor would flush values below float64's normal "
            "range; keep it scaled")
    return _raw_data(_shift_raw(view, exponent))


def materialize(tensor: AbstractTensor) -> AbstractTensor:
    """An unscaled handle holding the logical value (refuses overflow/underflow)."""

    base = unscaled_class(type(tensor))
    result = base(track_time=getattr(tensor, "track_time", False), tape=getattr(tensor, "_tape", None))
    result.data = (_raw_data(tensor) if base is type(tensor)
                   else _fold(tensor, _raw_data(tensor), _scale(tensor)))
    return result


def rebalance(tensor: AbstractTensor) -> AbstractTensor:
    """Move an exact power of two from the data into the scale, in place, so the
    largest raw magnitude sits near [1, 2)."""

    shift = _log2_floor(max_abs(tensor))
    shifted = _shift_raw(raw_view(tensor), -shift)
    object.__setattr__(tensor, "data", _raw_data(shifted))
    return set_scale(tensor, _scale(tensor) + shift)


# --------------------------------------------------------------------------
# the operator seam


def _operand(value: Any, *, normalize: bool):
    """(raw operand, exponent) for one side of an operator."""

    if not isinstance(value, AbstractTensor):
        return value, 0
    if normalize:
        value = rebalance(_copy(value))
    return raw_view(value), _scale(value)


def _copy(tensor: AbstractTensor) -> AbstractTensor:
    clone = raw_view(tensor)
    fresh = type(clone)(track_time=getattr(clone, "track_time", False), tape=getattr(clone, "_tape", None))
    fresh.data = _raw_data(clone)
    return set_scale(fresh, _scale(tensor), track=_tracked(tensor))


def _align(value, exponent, target):
    """``value`` re-expressed at ``target``: an exact shift by ``exponent - target``."""

    if isinstance(value, AbstractTensor):
        return _shift_raw(value, _exp(exponent) - _exp(target))
    return _pow2(_exp(exponent) - _exp(target)) * value


def scaled_apply_operator(receiver, op: str, left: Any, right: Any, **kwargs):
    """``_apply_operator`` for operands at least one of which is a scaled handle."""

    tracking = _tracked(left) or _tracked(right)
    base = op[1:] if op in {"radd", "rsub", "rmul", "rtruediv", "rpow", "rfloordiv", "rmod"} else (
        op[1:] if op.startswith("i") and op != "invert" else op)
    reversed_op = op.startswith("r") and base != op
    normalize = tracking and base in {"mul", "truediv"}
    lraw, lexp = _operand(left, normalize=normalize)
    rraw, rexp = _operand(right, normalize=normalize)
    raw_receiver = lraw if isinstance(lraw, AbstractTensor) else rraw
    apply = raw_receiver._apply_operator

    if base == "mul":
        result, exponent = apply(op, lraw, rraw, **kwargs), lexp + rexp
    elif base == "truediv":
        numerator_exp, denominator_exp = (rexp, lexp) if reversed_op else (lexp, rexp)
        result, exponent = apply(op, lraw, rraw, **kwargs), numerator_exp - denominator_exp
    elif base in _ALIGNED:
        target = _tmax([lexp, rexp])
        result = apply(op, _align(lraw, lexp, target), _align(rraw, rexp, target), **kwargs)
        exponent = target
    elif base in _COMPARE:
        target = _tmax([lexp, rexp])
        return apply(op, _align(lraw, lexp, target), _align(rraw, rexp, target), **kwargs)
    elif base in {"neg", "abs"}:
        result, exponent = apply(op, lraw, rraw, **kwargs), lexp
    elif base == "sign":
        return apply(op, lraw, rraw, **kwargs)
    elif base == "pow" and not reversed_op and isinstance(right, int) and not isinstance(right, bool):
        result, exponent = apply(op, lraw, rraw, **kwargs), lexp * int(right)
    else:
        left_value = materialize(left) if isinstance(left, AbstractTensor) else left
        right_value = materialize(right) if isinstance(right, AbstractTensor) else right
        receiver_value = left_value if isinstance(left_value, AbstractTensor) else right_value
        return receiver_value._apply_operator(op, left_value, right_value, **kwargs)

    if not isinstance(result, AbstractTensor):
        return result
    set_scale(result, exponent, track=tracking)
    if tracking:
        rebalance(result)
    return result


# --------------------------------------------------------------------------
# the method surface on every handle


def _with_scale(self, exponent: int, *, track: bool | None = None) -> AbstractTensor:
    """A new handle over the same raw data with this power-of-two exponent."""

    return set_scale(_copy(self), exponent, track=track)


AbstractTensor.with_scale = _with_scale
AbstractTensor.set_scale = set_scale
AbstractTensor.set_encoding = set_encoding
AbstractTensor.rebalance = rebalance
AbstractTensor.materialize = materialize
AbstractTensor.raw_view = raw_view

# --------------------------------------------------------------------------
# the static seam: operations called on the class (AbstractTensor.det(A), ...)


def _tensor_leaves(values):
    for value in values:
        if isinstance(value, AbstractTensor):
            yield value
        elif isinstance(value, (list, tuple)):
            yield from _tensor_leaves(value)


def _is_boolean(tensor) -> bool:
    return "bool" in str(getattr(raw_view(tensor), "dtype", "") or "").lower()


def _map_tensors(values, convert):
    out = []
    for value in values:
        if isinstance(value, AbstractTensor):
            out.append(convert(value))
        elif isinstance(value, (list, tuple)):
            out.append(type(value)(_map_tensors(value, convert)))
        else:
            out.append(value)
    return out


def _static_exponent(name, tensors, result):
    scales = [_scale(t) for t in tensors]
    if name in {"einsum", "outer", "cross", "dot", "matmul"}:
        total = scales[0]
        for other in scales[1:]:
            total = total + other
        return total
    if name in {"inv", "inverse"}:
        return -scales[0]
    if name == "det":
        return scales[0] * _matrix_order(tensors[0])
    if name == "solve" and len(scales) >= 2:
        return scales[1] - scales[0]
    return None


def _static_seam(name: str, rule: str, original):
    def seam(*args, **kwargs):
        tensors = list(_tensor_leaves(list(args) + list(kwargs.values())))
        if not any(type(t).__dict__.get("_is_scaled_class", False) for t in tensors):
            return original(*args, **kwargs)
        track = any(_tracked(t) for t in tensors)
        numeric = [t for t in tensors if not _is_boolean(t)]
        if rule == "align" or rule == "preserve":
            target = _tmax([_scale(t) for t in numeric] or [0])
            convert = lambda t: raw_view(t) if _is_boolean(t) else _align(raw_view(t), _scale(t), target)  # noqa: E731
            result = original(*_map_tensors(args, convert),
                              **dict(zip(kwargs, _map_tensors(list(kwargs.values()), convert))))
            dtype = str(getattr(result, "dtype", "") or "")
            return result if "bool" in dtype else _keep(result, target, track)
        if rule == "exponent":
            if name == "cbrt":
                (tensor,) = numeric[:1]
                exponent = _scale(tensor)
                shift = exponent - (exponent / 3).floor() * 3
                shifted = _shift_raw(raw_view(tensor), shift)
                return _keep(original(shifted), (exponent - shift) / 3, track)
            result = original(*_map_tensors(args, raw_view),
                              **dict(zip(kwargs, _map_tensors(list(kwargs.values()), raw_view))))
            out = _static_exponent(name, numeric, result)
            if out is None:
                return original(*_map_tensors(args, materialize))
            return _keep(result, out, track)
        if rule == "settle":
            destination = tensors[0]
            exponent = _scale(destination)
            converted = [raw_view(destination)] + [
                _align(raw_view(t), _scale(t), exponent) for t in tensors[1:]]
            it = iter(converted)
            return original(*_map_tensors(args, lambda _t: next(it)))
        return original(*args, **kwargs)

    seam.__wrapped__ = original
    seam.__name__ = name
    return seam


def _install_static_seams() -> tuple[str, ...]:
    import inspect

    installed = []
    for name, rule in SCALE_RULES.items():
        if rule not in {"preserve", "exponent", "align", "settle"}:
            continue
        try:
            held = inspect.getattr_static(AbstractTensor, name)
        except AttributeError:
            continue
        if not isinstance(held, staticmethod):
            continue
        original = held.__func__
        if getattr(original, "__name__", "") == name and hasattr(original, "__wrapped__") and                 getattr(original, "_power_scale_seam", False):
            continue
        seam = _static_seam(name, rule, original)
        seam._power_scale_seam = True
        setattr(AbstractTensor, name, staticmethod(seam))
        installed.append(name)
    return tuple(installed)


STATIC_SEAMS = _install_static_seams()


def _audit_rules() -> None:
    from .operator_catalog import CANONICAL_ABSTRACT_TENSOR_OPERATORS

    missing = sorted(CANONICAL_ABSTRACT_TENSOR_OPERATORS - SCALE_RULES.keys())
    invented = sorted(SCALE_RULES.keys() - CANONICAL_ABSTRACT_TENSOR_OPERATORS)
    if missing:
        raise ImportError(
            "power_scale: canonical operations with no scale rule: " + ", ".join(missing))
    if invented:
        raise ImportError(
            "power_scale: scale rules for operations the catalogue does not list: "
            + ", ".join(invented))


_audit_rules()

__all__ = (
    "ENCODING_RULES",
    "SCALE_RULES",
    "register_encoding",
    "set_encoding",
    "materialize",
    "max_abs",
    "raw_view",
    "rebalance",
    "scaled_apply_operator",
    "scaled_class",
    "set_scale",
    "unscaled_class",
)

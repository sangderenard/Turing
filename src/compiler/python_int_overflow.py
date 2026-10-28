"""What a Python ``int`` result does when it leaves int64.

Program: ``def root(n): return n * n`` with ``n`` an int64 no interval
bounds.  Python's product is exact; int64 arithmetic wraps.  Stage 1 made the
native op report the overflow (``native_faults.INT_OVERFLOW``).  This module
holds the two decisions that turn that report into Python's exact result:

* the POLICY (``python_int_overflow_policy``): Python semantics, ``exact``,
  is the default; ``fault`` (stage 1) and ``wrap`` (C semantics) are the
  configurable alternatives a later per-source / per-region int policy will
  choose.  It is the one place that decision is read.
* the CARRIER (``PRECISION_INT_CARRIER``): the exact value continues in the
  existing ``Precision`` limb type (n binary64 limbs whose exact sum is the
  integer), as the second arm of an ordinary native value union
  ``int: int64 | Precision[n]`` built on the existing union ABI (tag plus one
  payload per arm).  Swapping the carrier (for example to integer limbs)
  changes only this module.

The exact result of one op on two int64 operands fits int128, so the slow arm
is built from that one wide value: the region computes it at int128 (the
width an int64 x int64 operand interval proves), and the union frame is the
int64 view, the carrier view, and a tag saying which one is the value.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Iterable


#: ``exact``: Python semantics, the result continues in the carrier arm.
#: ``fault``: the result raises ``NativeIntOverflow`` (stage 1).
#: ``wrap``: two's complement int64 (C semantics; opt-in only).
INT_OVERFLOW_POLICIES = ("exact", "fault", "wrap")


def python_int_overflow_policy(scope: Any = None) -> str:
    """The int overflow policy in force for ``scope`` (a source/region).

    The one decision point.  Python semantics is the default for every
    scope; a configurable policy (like ``precision_policy.PrecisionPolicy``)
    is a change to this function alone.
    """

    del scope
    return "exact"


@dataclass(frozen=True)
class IntOverflowCarrier:
    """Physical spelling of the exact arm of an ``int`` union."""

    #: Union arm identity (``ValueTypeABIFact.alternatives``).
    identity: str
    #: Limbs of the carrier; enough that an int128 value is exact.
    limbs: int
    dtype: str
    #: The width the exact result is computed at before it is split.
    wide_dtype: str


#: Three binary64 limbs hold any int128 exactly: the top limb rounds the
#: value to 53 bits, the residual (at most 2**74) rounds to the next 53, and
#: what remains (at most 2**22) is exact in the third.
PRECISION_INT_CARRIER = IntOverflowCarrier(
    identity="Precision", limbs=3, dtype="float64", wide_dtype="int128",
)


def carrier_arm(int_arm: Any) -> Any:
    """The carrier's ``ValueTypeABIAlternative`` beside ``int_arm``."""

    carrier = PRECISION_INT_CARRIER
    return replace(
        int_arm,
        identity=carrier.identity,
        storage="span",
        dtype=carrier.dtype,
        shape=(int(carrier.limbs),),
        rank=1,
        storage_identity=(("precision_limbs", int(carrier.limbs)),),
    )


def union_alternatives() -> tuple[str, str]:
    """The union's arm identities, in tag order."""

    return tuple(sorted(("builtins.int", PRECISION_INT_CARRIER.identity)))


def is_int_overflow_union(alternatives: Iterable[str]) -> bool:
    return tuple(alternatives) == union_alternatives()


#: Python operators whose exact result on two int64 operands fits int128.
_WIDENING_OPERATORS = (ast.Add, ast.Sub, ast.Mult)


def widening_operand_roles(expression: Any) -> tuple[str, ...] | None:
    """The operand roles of an op whose int64 result may continue exactly."""

    if isinstance(expression, ast.BinOp) and isinstance(
        expression.op, _WIDENING_OPERATORS
    ):
        return ("lhs", "rhs")
    if isinstance(expression, ast.UnaryOp) and isinstance(
        expression.op, ast.USub
    ):
        return ("operand",)
    return None


def emit_carrier_frame(builder: Any, value_id: int, fact: Any) -> None:
    """Emit the union frame of a wide (int128) result in ``builder``.

    ``int64 = low word``; ``fits = (widen(int64) == wide)``; the carrier
    limbs are the error-free split of ``wide`` into binary64 (each limb is the
    rounding of what the previous ones left, the residual computed exactly in
    int128).  The tag selects the int arm exactly when ``fits``.
    """

    from ..transmogrifier.ssa_registry import Handler
    from ..common.tensors.topological_reducer import PRECISION_PACK_NAME
    from .precompile_to_ssa import (
        CONTROL_CAST, CONTROL_EXPRESSION, CONTROL_PREDICATE,
    )

    carrier = PRECISION_INT_CARRIER
    wide = builder.ssa_value_objects.get(int(value_id))
    if wide is None or str(wide.dtype) != carrier.wide_dtype:
        raise ValueError(
            "int overflow union has no exact wide result: "
            f"value={int(value_id)}, wide={wide!r}"
        )
    tags = dict(getattr(fact, "tags", ()) or ())

    def cast(source, dtype):
        result = builder.fresh_value(
            dtype=dtype, transform=CONTROL_CAST, operands=(source,),
        )
        builder.emit(
            Handler.Cast, [source], result, attributes={"target_dtype": dtype},
        )
        return result

    def binary(op, left, right, dtype):
        result = builder.fresh_value(
            dtype=dtype, transform=CONTROL_EXPRESSION, operands=(left, right),
        )
        builder.emit(op, [left, right], result)
        return result

    low = cast(wide, "int64")
    fits = builder.fresh_value(
        dtype="bool", transform=CONTROL_PREDICATE, operands=(wide, low),
    )
    builder.emit(Handler.Eq, [cast(low, carrier.wide_dtype), wide], fits)
    limbs = []
    residual = wide
    for index in range(int(carrier.limbs)):
        limb = cast(residual, carrier.dtype)
        limbs.append(limb)
        if index + 1 < int(carrier.limbs):
            residual = binary(
                Handler.Sub, residual, cast(limb, carrier.wide_dtype),
                carrier.wide_dtype,
            )
    packed = builder.fresh_value(
        dtype=carrier.dtype, shape=(int(carrier.limbs),),
        transform=CONTROL_EXPRESSION, operands=tuple(limbs),
    )
    packed.accounting.update({"precision_limbs": int(carrier.limbs)})
    # ``precision_pack`` is the Precision pipeline's own limb constructor
    # (``ir_identities.lower_precision_operations``); it has no Handler row.
    builder.emit(
        SimpleNamespace(value=PRECISION_PACK_NAME), list(limbs), packed,
        attributes={
            "precision_limbs": int(carrier.limbs),
            "binding": "int_overflow_carrier",
        },
    )
    tag = builder.fresh_value(
        dtype="int64", transform=CONTROL_EXPRESSION, operands=(fits,),
    )
    builder.emit(
        Handler.Select,
        [
            fits,
            builder.constant_value(
                int(tags["builtins.int"]), fits, dtype="int64",
            ),
            builder.constant_value(
                int(tags[carrier.identity]), fits, dtype="int64",
            ),
        ],
        tag,
        attributes={"binding": "int_overflow_tag"},
    )
    tag.accounting.update({
        "native_type_tag_for_value_id": int(value_id),
        "native_type_alternatives": tuple(getattr(fact, "alternatives", ())),
        "native_type_tags": tuple(sorted(tags.items())),
    })
    builder.native_value_type_tags[int(value_id)] = tag
    builder.native_value_union_payloads[int(value_id)] = {
        "builtins.int": (low,),
        carrier.identity: (packed,),
    }
    builder.post_native_type_tag(int(value_id), tag, fact, wide)


def python_int_from_carrier(limbs: Iterable[float]) -> int:
    """The Python ``int`` a carrier's limbs denote (host side, exact)."""

    total = 0
    for limb in limbs:
        value = float(limb)
        if value != value or not value.is_integer():
            raise ValueError(f"int carrier limb is not an integer: {value!r}")
        total += int(value)
    return total

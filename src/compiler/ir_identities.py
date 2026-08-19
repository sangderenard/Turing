"""Backend-neutral arithmetic identities over repository SSA.

The first inhabitant of the seam `SSA_IDENTITY_CATALOGUE.md` names: every
`ssa_*` backend consumes the same ``IRModule``, so an identity applied here is
inherited by all seven, including the six with no ``-O2`` behind them. This
module starts with the single highest-value item, constant-exponent ``Pow``
strength reduction (catalogue section 2.1) -- on the reference fluid kernel the
24 ``Pow`` calls with compile-time-constant exponents are the entire measured
cost.

The exact/inexact split is a policy boundary, not an implementation detail:

* exact (fires by default): ``x**2`` -> ``Mul(x, x)`` and ``x**-1`` ->
  ``Div(1, x)``. Both are bit-identical to a correctly rounded ``pow``.
* inexact (fires only when asked): ``x**0.5`` -> ``Sqrt(x)`` (differs at
  ``-0.0``/``-inf``), ``x**-2`` -> ``Div(1, Mul(x, x))`` and ``x**-0.5`` ->
  ``Div(1, Sqrt(x))`` (one extra rounding each). Until a value can carry a
  proven ``positive`` fact (catalogue section 5), these change bits and must
  be opted into, so their numeric consequence is a measured delta rather than
  a silent default.

``x**1``, ``x**0`` and integer exponents beyond ``2`` are deliberately absent:
``x**1`` does not occur in any measured workload and needs use-rewriting,
``x**0`` differs at ``NaN``, and ``x**3`` as ``x*x*x`` rounds twice where a
correctly rounded ``pow`` rounds once.
"""
from __future__ import annotations

import dataclasses

from ..transmogrifier.ssa import Instr, SSAValue

_POW = ("Pow", "pow")

# Exponents whose reduction is bit-identical to a correctly rounded pow.
_EXACT_EXPONENTS = (2.0, -1.0)
# Exponents whose reduction changes bits; opt-in only (see module docstring).
_INEXACT_EXPONENTS = (0.5, -0.5, -2.0)


def _module_watermark(functions) -> int:
    """One past the largest SSA id anywhere in the module.

    Same discipline as ``ir_indexing``: a planner region shares its caller's
    value space, so fresh ids must clear the MODULE's maximum, not one
    function's.
    """

    highest = -1
    for function in functions.values():
        for value in function.args:
            highest = max(highest, int(value.id))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    highest = max(highest, int(instruction.res.id))
                for argument in instruction.args:
                    highest = max(highest, int(argument.id))
    return highest + 1


def _constant_value(instruction) -> float | None:
    """The numeric payload of a ``Const``, or ``None`` if it has none."""

    payload = instruction.attributes.get("constant")
    if payload is None:
        payload = instruction.attributes.get("values")
    if payload is None and "value" in instruction.attributes:
        payload = instruction.attributes.get("value")
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return None
    return float(payload)


def reduce_constant_exponent_pow(functions, inexact: bool | None = None) -> dict:
    """Rewrite ``Pow(x, Const c)`` to cheaper exact primitives, in place.

    ``functions`` is a mapping of name -> repository SSA ``Function``.
    ``inexact=True`` additionally fires the bit-changing set; the default asks
    the active :mod:`work_contract` (which still honors the legacy
    ``TURING_POW_INEXACT`` variable as an override), so no flag threads
    through the compile entry points. Returns a count per rewrite kind, for
    logging and for tests.
    """

    if inexact is None:
        from .work_contract import active_contract

        inexact = active_contract().inexact_identities
    allowed = _EXACT_EXPONENTS + (_INEXACT_EXPONENTS if inexact else ())

    next_id = _module_watermark(functions)

    def fresh() -> SSAValue:
        nonlocal next_id
        value = SSAValue(next_id, dtype="float64")
        next_id += 1
        return value

    counts = {"pow_to_mul": 0, "pow_to_reciprocal": 0, "pow_to_sqrt": 0}

    for function in functions.values():
        # Exponent constants live in the same function as their Pow; ids are
        # only comparable within one function's value space.
        constants: dict[int, float] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    value = _constant_value(instruction)
                    if value is not None:
                        constants[int(instruction.res.id)] = value

        # One shared literal 1.0 per function. Minted lazily, but INSERTED only
        # after every block is rewritten: inserting into the entry block while
        # it is itself being rebuilt would be overwritten by its new list.
        one_value: SSAValue | None = None

        def unit() -> SSAValue:
            nonlocal one_value
            if one_value is None:
                one_value = fresh()
            return one_value

        def stripped(instruction) -> dict:
            # A backend that prefers `tensor_operation` over `op` must not
            # keep reading "Pow" off the rewritten instruction.
            return {
                key: value
                for key, value in instruction.attributes.items()
                if key not in ("tensor_operation", "tensor")
            }

        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                canonical_op = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.op
                )
                exponent = None
                if (
                    canonical_op in _POW
                    and len(instruction.args) == 2
                    and instruction.res is not None
                ):
                    exponent = constants.get(int(instruction.args[1].id))
                if exponent is None or exponent not in allowed:
                    rewritten.append(instruction)
                    continue

                base = instruction.args[0]
                attributes = stripped(instruction)
                if exponent == 2.0:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Mul", args=[base, base],
                        attributes=attributes,
                    ))
                    counts["pow_to_mul"] += 1
                elif exponent == -1.0:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), base],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
                elif exponent == 0.5:
                    rewritten.append(dataclasses.replace(
                        instruction, op="Sqrt", args=[base],
                        attributes=attributes,
                    ))
                    counts["pow_to_sqrt"] += 1
                elif exponent == -0.5:
                    root = fresh()
                    rewritten.append(
                        Instr("Sqrt", [base], root, attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), root],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
                elif exponent == -2.0:
                    square = fresh()
                    rewritten.append(
                        Instr("Mul", [base, base], square,
                              attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(), square],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
            block.instrs = rewritten

        if one_value is not None:
            # Entry block is first in insertion order and dominates the rest.
            entry = next(iter(function.blocks.values()))
            entry.instrs.insert(
                0, Instr("Const", [], one_value, attributes={"constant": 1.0})
            )

    return counts

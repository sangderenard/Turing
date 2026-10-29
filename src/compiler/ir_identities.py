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

import contextlib as _contextlib
import functools as _functools
import dataclasses
import math
from collections.abc import Mapping

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

    from .concordance_declarations import (
        PRECISION_IDENTITIES, PRECISION_IDENTITY_VALUE,
    )
    from .ssa_record_return_state import mint_ssa_value

    def fresh(function, *operands) -> SSAValue:
        # Minted through the book: a NOVEL ``ssa_value`` row from the cell of
        # the operand the identity makes it from (``mint_ssa_value``).
        return SSAValue(mint_ssa_value(
            function, PRECISION_IDENTITY_VALUE, operands,
            dtype="float64", shape=(), stage=PRECISION_IDENTITIES,
        ), dtype="float64")

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

        def unit(base) -> SSAValue:
            nonlocal one_value
            if one_value is None:
                one_value = fresh(function, base)
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
                        instruction, op="Div", args=[unit(base), base],
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
                    root = fresh(function, base)
                    rewritten.append(
                        Instr("Sqrt", [base], root, attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(base), root],
                        attributes=attributes,
                    ))
                    counts["pow_to_reciprocal"] += 1
                elif exponent == -2.0:
                    square = fresh(function, base)
                    rewritten.append(
                        Instr("Mul", [base, base], square,
                              attributes=dict(attributes))
                    )
                    rewritten.append(dataclasses.replace(
                        instruction, op="Div", args=[unit(base), square],
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


def _python_int_dtype(dtype) -> bool:
    return str(dtype) == "int64"


def _integer_constant(instruction) -> int | None:
    """The exact int payload of an integer ``Const``, else ``None``."""

    for key in ("constant", "values", "value"):
        payload = instruction.attributes.get(key)
        if payload is not None:
            break
    if isinstance(payload, bool) or not isinstance(payload, int):
        return None
    return int(payload)


def _integer_pow_site(instruction) -> str | None:
    """Which exact power reduction an instruction takes, else ``None``.

    ``"int"``: ``a ** b`` / ``pow(a, b)`` on int64 with an int64 result.
    ``"modular"``: ``pow(a, b, m)`` on int64.
    """

    canonical = str(
        instruction.attributes.get("tensor_operation")
        or instruction.attributes.get("tensor")
        or instruction.op
    )
    result = instruction.res
    if (
        canonical not in _POW or result is None
        or tuple(getattr(result, "shape", None) or ())
        or len(instruction.args) not in (2, 3)
        or not all(_python_int_dtype(value.dtype) for value in instruction.args)
        or any(tuple(value.shape or ()) for value in instruction.args)
        or not _python_int_dtype(result.dtype)
    ):
        return None
    return "int" if len(instruction.args) == 2 else "modular"


def reduce_integer_pow(functions) -> dict:
    """The shared exact power reduction: Python ``int`` powers, in place.

    Program: ``a ** b``, ``pow(a, b)``, ``pow(a, b, m)`` on ints.  Wrong
    record: C and LLVM refused every int power but ``x ** 2``, and a float
    ``pow`` of an int is not Python's exact int.  Rule: square-and-multiply
    in ordinary repository SSA, so every backend inherits it -- each
    multiply is Python ``int`` arithmetic (``source_type`` ``BinOp``),
    which C and LLVM overflow-check (``native_faults.INT_OVERFLOW``).
    A constant exponent unrolls straight-line; a variable one becomes a
    loop over its bits (at most 63 trips).  ``pow(a, b, m)`` reduces each
    product by Python's floored ``%``, so intermediates stay below ``m**2``.

    Python's other outcomes are named faults: ``0 ** -n`` raises
    ``ZeroDivisionError`` (``INT_ZERO_DIVISION``), ``pow(a, b, 0)``
    ``ValueError`` (``POW_MODULUS_ZERO``).  ``a ** -n`` is a float in
    Python and ``pow(a, -n, m)`` a modular inverse; an int-typed power
    reaching a negative exponent reports ``INT_POWER_NEGATIVE_EXPONENT``
    rather than return an int Python never computes: the frontend types
    every int ** int as int (``a ** -2`` too), and a float arm would need
    an int | float result.
    """

    from .concordance_declarations import (
        PRECISION_IDENTITIES, PRECISION_IDENTITY_VALUE, SSA_BLOCK,
        SSABlockKind,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import function_scope_of, identity_cells
    from .native_faults import (
        INT_POWER_NEGATIVE_EXPONENT, INT_ZERO_DIVISION, POW_MODULUS_ZERO,
    )
    from .ssa_record_return_state import mint_ssa_value
    from ..transmogrifier.ssa import BasicBlock

    counts = {"int_pow_unrolled": 0, "int_pow_loop": 0, "int_pow_modular": 0}

    def rewrite(function, block_name: str, index: int) -> None:
        block = function.blocks[block_name]
        power = block.instrs[index]
        site = _integer_pow_site(power)
        post = block.instrs[index + 1:]
        old_successors = list(block.successors)
        block.instrs = block.instrs[:index]
        anchor = power.args[0]
        constants = {
            int(instruction.res.id): _integer_constant(instruction)
            for candidate in function.blocks.values()
            for instruction in candidate.instrs
            if instruction.op == "Const" and instruction.res is not None
        }
        added: list[BasicBlock] = []
        taken = set(function.blocks)
        state = {"current": block}

        def new_block(stem: str, kind) -> BasicBlock:
            serial = 0
            name = f"{block_name}.{stem}"
            while name in taken:
                serial += 1
                name = f"{block_name}.{stem}.{serial}"
            taken.add(name)
            created = BasicBlock(name)
            added.append(created)
            # The block's ``ssa_block`` row, from the power it spells.
            owners = identity_cells(function, power.res, *power.args)
            book = current_identity_book()
            row = (function_scope_of(function), str(function.name), name)
            if owners and book.latest_ref(SSA_BLOCK, row) is None:
                book.post(
                    SSA_BLOCK, row, kind, stage=PRECISION_IDENTITIES,
                    provenance=Derived(owners), mode=Mode.CONCORD,
                )
            return created

        def mint(dtype: str) -> SSAValue:
            return SSAValue(mint_ssa_value(
                function, PRECISION_IDENTITY_VALUE, (anchor,),
                dtype=dtype, shape=(), stage=PRECISION_IDENTITIES,
            ), dtype=dtype)

        def put(op, args, dtype="int64", attributes=None, res=None) -> SSAValue:
            res = mint(dtype) if res is None else res
            state["current"].instrs.append(
                Instr(op, list(args), res, attributes=dict(attributes or {}))
            )
            return res

        def arithmetic(op, args, res=None) -> SSAValue:
            # Python ``int`` arithmetic: the backends' checked spelling.
            return put(op, args, attributes={
                "source_type": "BinOp", "integer_power": True,
            }, res=res)

        def const(value: int) -> SSAValue:
            return put("Const", [], attributes={"value": int(value)})

        def jump(target: BasicBlock) -> None:
            current = state["current"]
            current.instrs.append(
                Instr("Br", [], None, attributes={"target": target.name})
            )
            current.successors = [target.name]

        def guard(condition: SSAValue, code: int, failure: str) -> None:
            fault = new_block("pow_fault", SSABlockKind.VALIDATION_FAIL)
            passing = new_block("pow_checked", SSABlockKind.VALIDATION_PASS)
            current = state["current"]
            current.instrs.append(Instr("CondBr", [condition], None, attributes={
                "true_target": fault.name, "false_target": passing.name,
            }))
            current.successors = [fault.name, passing.name]
            fault.instrs.append(Instr("Call", [], None, attributes={
                "callee": "turing_validation_error", "error_code": int(code),
                "failure": failure,
            }))
            state["current"] = fault
            jump(passing)
            state["current"] = passing

        base, exponent = power.args[0], power.args[1]
        literal = constants.get(int(exponent.id))
        zero, one = const(0), const(1)
        unrolled = site == "int" and literal is not None and literal >= 0
        negative = None if unrolled else put("Lt", [exponent, zero], "bool")
        modulus = None
        if site == "modular":
            modulus = power.args[2]
            modulus_zero = put("Eq", [modulus, zero], "bool")
            guard(modulus_zero, POW_MODULUS_ZERO, "pow_modulus_zero")
            guard(negative, INT_POWER_NEGATIVE_EXPONENT,
                  "pow_negative_exponent")
            modulus = put("Select", [modulus_zero, one, modulus])
        elif not unrolled:
            zero_base = put("Eq", [base, zero], "bool")
            guard(put("And", [negative, zero_base], "bool"),
                  INT_ZERO_DIVISION, "zero_to_negative_power")
            guard(negative, INT_POWER_NEGATIVE_EXPONENT,
                  "pow_negative_exponent")
        if unrolled:
            # Straight-line square-and-multiply over the literal's bits.
            square, result, bits = base, None, int(literal)
            if bits == 0:
                put("Const", [], attributes={"value": 1}, res=power.res)
            while bits:
                last_bit = bits == 1
                if bits & 1:
                    result = square if result is None else arithmetic(
                        "Mul", [result, square],
                        res=power.res if last_bit else None,
                    )
                    if result is square and last_bit:
                        arithmetic("Mul", [square, one], res=power.res)
                bits >>= 1
                if bits:
                    square = arithmetic("Mul", [square, square])
            counts["int_pow_unrolled"] += 1
        else:
            count = put("Select", [negative, zero, exponent])
            if modulus is not None:
                start = arithmetic("Mod", [base, modulus])
                first = arithmetic("Mod", [one, modulus])
            else:
                start, first = base, one
            entry = state["current"]
            header = new_block("pow_loop", SSABlockKind.LOOP_HEADER)
            step = new_block("pow_step", SSABlockKind.LOOP_BODY)
            done = new_block("pow_done", SSABlockKind.LOOP_EXIT)
            jump(header)
            result, square, remaining = power.res, mint("int64"), mint("int64")
            next_result, next_square, next_remaining = (
                mint("int64"), mint("int64"), mint("int64"),
            )
            incoming = {"incoming_blocks": (entry.name, step.name)}
            state["current"] = header
            put("Phi", [first, next_result], attributes=incoming, res=result)
            put("Phi", [start, next_square], attributes=incoming, res=square)
            put("Phi", [count, next_remaining], attributes=incoming,
                res=remaining)
            going = put("Gt", [remaining, zero], "bool")
            header.instrs.append(Instr("CondBr", [going], None, attributes={
                "true_target": step.name, "false_target": done.name,
            }))
            header.successors = [step.name, done.name]
            state["current"] = step
            odd = put("Ne", [put("BitAnd", [remaining, one]), zero], "bool")
            factor = put("Select", [odd, square, one])
            put("Shr", [remaining, one], res=next_remaining)
            more = put("Gt", [next_remaining, zero], "bool")
            # The square is taken only while bits remain, so it
            # overflows only where the power itself does.
            kept = put("Select", [more, square, one])
            if modulus is None:
                arithmetic("Mul", [result, factor], res=next_result)
                arithmetic("Mul", [kept, kept], res=next_square)
            else:
                arithmetic("Mod", [arithmetic("Mul", [result, factor]),
                                   modulus], res=next_result)
                arithmetic("Mod", [arithmetic("Mul", [kept, kept]),
                                   modulus], res=next_square)
            jump(header)
            state["current"] = done
            counts["int_pow_modular" if modulus is not None
                   else "int_pow_loop"] += 1

        tail = state["current"]
        tail.instrs.extend(post)
        tail.successors = old_successors
        if tail is not block:
            # Successor Phis now enter from the block holding the tail.
            for successor in old_successors:
                target = function.blocks.get(successor)
                for instruction in (target.instrs if target else ()):
                    blocks = instruction.attributes.get("incoming_blocks")
                    if str(instruction.op) == "Phi" and blocks:
                        instruction.attributes["incoming_blocks"] = tuple(
                            tail.name if str(origin) == block_name else origin
                            for origin in blocks
                        )
        ordered = []
        for name, existing in function.blocks.items():
            ordered.append((name, existing))
            if name == block_name:
                ordered.extend((created.name, created) for created in added)
        function.blocks.clear()
        function.blocks.update(ordered)

    for function in functions.values():
        while True:
            site = next(
                (
                    (name, position)
                    for name, candidate in function.blocks.items()
                    for position, instruction in enumerate(candidate.instrs)
                    if _integer_pow_site(instruction) is not None
                ),
                None,
            )
            if site is None:
                break
            rewrite(function, *site)
    return counts


#: The scalar float64 transcendental SSA ops the signal cores route, by the
#: name the planner leaves on them: ``hierarchical_plan``'s
#: ``TENSOR_OPERATION_SCALAR_SPELLING`` respells ``exp``/``log``/``pow`` to
#: ``Exp``/``Log``/``Pow``; every other ``math`` name keeps its lowercase
#: tensor-vocabulary spelling.  ``Sqrt`` is not here: IEEE square root is
#: correctly rounded in every backend and stays native.
TRANSCENDENTAL_OPS = frozenset({
    "Exp", "Log", "Pow",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
})


@dataclasses.dataclass(frozen=True)
class SignalCoreRoute:
    """One row of ``TRANSCENDENTAL_SIGNAL_CORES``: how an op reaches its core.

    ``method`` names a ``WideExpansionEmitter`` method, called as
    ``method(*leading, *operands, **dict(keywords))`` with the op's float64
    operands, or is a function ``method(emitter, instruction)`` for a core
    that reads the op's own record (``Pow``'s operator/``math.pow`` form).
    The cores answer in three shapes, and the pass reads each one:

    * an expansion (a list of limbs, high first) -- collapsed to one double;
    * ``(value, faults)`` -- ``value`` a double or an expansion, ``faults``
      a sequence of ``(predicate, code, name)`` rows: a ``bool`` SSA value
      true exactly where Python raises, the ``native_faults`` code it
      reports, and the failure's name;
    * ``(value, predicate)`` -- one bare ``bool`` predicate, reported as
      this row's ``fault`` (a ``(code, name)`` pair).

    A predicate of ``None`` is a fault the core proves cannot happen.
    """

    method: object
    leading: tuple = ()
    keywords: tuple = ()
    fault: tuple | None = None


def _exp_signal_core(emitter, instruction):
    """``math.exp``: the full-range core, plus Python's ``OverflowError``.

    The predicate is read from the limbs the core returned
    (``exp_overflow_predicate``), so the fault and the value agree at the
    last finite binade (709.782712893384 finite, 709.7827128933841 raises).
    """

    from .native_faults import FLOAT_MATH_RANGE

    source = emitter.expanded(instruction.args[0], 2)
    result = emitter.exp_expansion(source, 2)
    return result, ((
        emitter.exp_overflow_predicate(source, result),
        FLOAT_MATH_RANGE, "math_exp_overflow",
    ),)


def _log_signal_core(emitter, instruction):
    """``math.log``: the full-range core; ``x <= 0`` raises ``ValueError``."""

    from .native_faults import FLOAT_MATH_DOMAIN

    source = emitter.expanded(instruction.args[0], 2)
    return emitter.log_expansion(source, 2), ((
        emitter.log_domain_predicate(source),
        FLOAT_MATH_DOMAIN, "math_log_domain",
    ),)


def _pow_signal_core(emitter, instruction):
    """``x ** y`` or ``math.pow(x, y)``, by the record that wrote the op.

    The planner spells both ``Pow``; Python faults them differently
    (``0.0 ** -1`` is ``ZeroDivisionError``, ``math.pow(0.0, -1)`` is
    ``ValueError``), and ``pow_expansion`` returns the rows for the form it
    is told.  The extraction contract names ``math.pow`` on the call; the
    operator carries ``source_type`` ``BinOp``.  Neither is a refusal by
    name rather than a guessed form.
    """

    attributes = instruction.attributes or {}
    contract = attributes.get("extraction_contract") or {}
    identity = (
        contract.get("identity") if isinstance(contract, Mapping) else None
    )
    if identity == "math.pow":
        form = "math"
    elif attributes.get("source_type") == "BinOp":
        form = "operator"
    else:
        raise ValueError(
            "float Pow carries neither a math.pow extraction contract nor "
            f"an operator source_type ({dict(attributes)!r}); its Python "
            "faults cannot be chosen"
        )
    return emitter.pow_expansion(*instruction.args, 2, form=form)


def _circular_route(name: str) -> SignalCoreRoute:
    from .native_faults import FLOAT_MATH_DOMAIN

    # sin/cos/tan(+-inf): Python raises ValueError('math domain error').
    return SignalCoreRoute(
        "circular_double", (name,),
        fault=(FLOAT_MATH_DOMAIN, f"math_{name}_infinite"),
    )


#: SSA op -> its signal core, for ``transcendentals: signal_cores``.  Every
#: other op in ``TRANSCENDENTAL_OPS`` (``asin``, ``acos``, ``sinh``,
#: ``cosh``) has no core yet and is refused by name under that policy.
#: c22, c913-c915 and c925 hold these on C, LLVM and Fortran.
TRANSCENDENTAL_SIGNAL_CORES: dict[str, SignalCoreRoute] = {
    "Exp": SignalCoreRoute(_exp_signal_core),
    "Log": SignalCoreRoute(_log_signal_core),
    "Pow": SignalCoreRoute(_pow_signal_core),
    "sin": _circular_route("sin"),
    "cos": _circular_route("cos"),
    "tan": _circular_route("tan"),
    "atan": SignalCoreRoute("atan_expansion"),
    "atan2": SignalCoreRoute("atan2_expansion"),
    "tanh": SignalCoreRoute("tanh_expansion"),
}

_TRANSCENDENTAL_FLOAT64 = frozenset({"float64", "float", "double", "f64"})


def _scalar_float64(value) -> bool:
    return (
        value is not None
        and str(value.dtype or "").casefold() in _TRANSCENDENTAL_FLOAT64
        and not tuple(value.shape or ())
    )


def _shaped(value) -> bool:
    """A tensor/span value: a static shape, or a declared runtime rank."""

    if value is None:
        return False
    if tuple(value.shape or ()):
        return True
    accounting = getattr(value, "accounting", None) or {}
    return any(
        int(accounting.get(key, 0) or 0) > 0
        for key in ("program_abi_rank", "ssa_call_rank")
    )


def _elementwise_transcendental(instruction) -> str | None:
    """The scalar op one element of a shaped transcendental is, else None.

    Read from the records, never from a list kept here: the op's catalogue
    row (``ssa_numeric_operators.TENSOR_SSA_OPERATOR_BY_NAME``) must be
    ``elementwise_unary``/``elementwise_binary``; its scalar spelling is the
    planner's own (``hierarchical_plan.TENSOR_OPERATION_SCALAR_SPELLING``:
    ``exp`` -> ``Exp``, ``pow`` -> ``Pow``; every other name keeps its
    spelling); and that scalar op must be one of ``TRANSCENDENTAL_OPS``, the
    set the scalar route already names (an op already spelled as one of
    those, shaped, is its own scalar op).  The result must be shaped
    (a static shape or a declared runtime rank).
    """

    from .hierarchical_plan import TENSOR_OPERATION_SCALAR_SPELLING
    from .ssa_numeric_operators import TENSOR_SSA_OPERATOR_BY_NAME

    name = str(
        instruction.attributes.get("tensor_operation") or instruction.op
    )
    if name in TRANSCENDENTAL_OPS:
        # Already the scalar spelling (the planner writes ``a ** 1.5`` on a
        # tensor as a shaped ``Pow``); on a shaped value every backend
        # spells it elementwise (``scalar_likeness``).
        scalar_op = name
    else:
        row = TENSOR_SSA_OPERATOR_BY_NAME.get(name)
        if row is None or row.category not in {
            "elementwise_unary", "elementwise_binary",
        }:
            return None
        scalar_op = TENSOR_OPERATION_SCALAR_SPELLING.get(name, name)
    if scalar_op not in TRANSCENDENTAL_OPS or not _shaped(instruction.res):
        return None
    return scalar_op


def scalarize_elementwise_transcendentals(function, block_name, index):
    """One elementwise tensor/span transcendental -> a loop of scalar ops.

    Program: ``np.exp(a)``, ``a.sin()``, ``a ** b`` on a float64 tensor.
    Wrong record: the planner keeps the lowercase tensor op, and each
    backend spelled it with host math per element -- C libm in
    ``emit_elementwise_span``, LLVM's ``unary_double`` kernel (fused to
    ``llvm.exp.f64``), Fortran's elemental ``exp()`` -- under a contract
    whose ``transcendentals`` is ``signal_cores``.  Rule: the result is a
    fresh buffer (a ``clone`` of an operand of the result's shape) filled
    by an SSA loop over its elements: ``GetElementPtr``/``Load`` each shaped
    operand's element (a rank-0 operand is the same value for every
    element), apply the SCALAR op the signal cores route
    (``_elementwise_transcendental``), ``GetElementPtr``/``Store`` it.  The
    caller (``lower_transcendentals_to_signal_cores``) then rewrites that
    scalar op as it does any scalar one, so every element is the core's
    correctly rounded double and every element carries the core's fault
    guards (``native_faults``).

    Refused by name: an op with no core, an operand or result that is not
    float64, operand shapes that differ (a broadcast), a
    runtime-sized span (no static element count), and a ``Pow`` whose
    Python form ``_pow_signal_core`` cannot read.  None of these is left
    on host math.
    """

    from .concordance_declarations import (
        PRECISION_IDENTITIES, PRECISION_IDENTITY_VALUE, SSA_BLOCK,
        SSABlockKind,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import (
        function_scope_of, identity_cells, mint_ssa_value,
    )
    from ..transmogrifier.ssa import BasicBlock

    block = function.blocks[block_name]
    call = block.instrs[index]
    scalar_op = _elementwise_transcendental(call)
    name = str(call.attributes.get("tensor_operation") or call.op)
    where = f"{call.op} -> %{call.res.id} in {function.name}"
    contract = (
        "; the numeric contract's transcendentals is 'signal_cores', which "
        "never leaves a math function on the host library (numeric mode "
        "'compute' selects backend_intrinsic)"
    )
    if scalar_op not in TRANSCENDENTAL_SIGNAL_CORES:
        raise ValueError(
            f"elementwise {name} has no signal core ({where}){contract}"
        )
    result = call.res
    shape = tuple(result.shape or ())
    if not shape or not all(isinstance(extent, int) for extent in shape):
        raise ValueError(
            f"elementwise {name} over a runtime-sized span ({where}) has no "
            f"static element count for its signal-core loop{contract}"
        )
    if not call.args or not all(
        str(value.dtype or "").casefold() in _TRANSCENDENTAL_FLOAT64
        for value in (result, *call.args)
    ):
        raise ValueError(
            f"elementwise {name} ({where}) is not float64 on every operand "
            "and its result "
            f"({[str(value.dtype) for value in (result, *call.args)]}); the "
            f"signal cores compute float64{contract}"
        )
    for value in call.args:
        if _shaped(value) and tuple(value.shape or ()) != shape:
            raise ValueError(
                f"elementwise {name} ({where}) broadcasts operand %{value.id} "
                f"of shape {tuple(value.shape or ())!r} to {shape!r}; the "
                f"signal-core loop spells only equal shapes{contract}"
            )
    template = next(value for value in call.args if _shaped(value))

    post = block.instrs[index + 1:]
    old_successors = list(block.successors)
    block.instrs = block.instrs[:index]
    added: list[BasicBlock] = []
    taken = set(function.blocks)
    state = {"current": block}
    owners = identity_cells(function, call.res, *call.args)

    def new_block(stem: str, kind) -> BasicBlock:
        serial = 0
        label = f"{block_name}.{stem}"
        while label in taken:
            serial += 1
            label = f"{block_name}.{stem}.{serial}"
        taken.add(label)
        created = BasicBlock(label)
        added.append(created)
        book = current_identity_book()
        row = (function_scope_of(function), str(function.name), label)
        if owners and book.latest_ref(SSA_BLOCK, row) is None:
            book.post(
                SSA_BLOCK, row, kind, stage=PRECISION_IDENTITIES,
                provenance=Derived(owners), mode=Mode.CONCORD,
            )
        return created

    def mint(dtype: str) -> SSAValue:
        return SSAValue(mint_ssa_value(
            function, PRECISION_IDENTITY_VALUE, (call.res,),
            dtype=dtype, shape=(), stage=PRECISION_IDENTITIES,
        ), dtype=dtype, shape=())

    lowered = {"lowered_from": f"elementwise.{name}"}

    def put(op, args, dtype, attributes=None, res=None) -> SSAValue:
        # ``dtype`` None: an effect (``Store``) with no result.
        if res is None and dtype is not None:
            res = mint(dtype)
        state["current"].instrs.append(Instr(
            op, list(args), res,
            attributes={**lowered, **dict(attributes or {})},
        ))
        return res

    def jump(target: BasicBlock) -> None:
        current = state["current"]
        current.instrs.append(Instr(
            "Br", [], None, attributes={**lowered, "target": target.name},
        ))
        current.successors = [target.name]

    # The result's own id names the fresh buffer the loop fills: a
    # ``clone`` of an operand of its shape (C and LLVM copy into the
    # result's storage, Fortran assigns the array), every element of which
    # the loop overwrites.  ``zeros_like`` has no module-lane C spelling.
    put("clone", [template], "float64", attributes={
        "tensor_operation": "clone", "tensor_candidate": "clone",
    }, res=result)
    zero = put("Const", [], "int64", attributes={"value": 0})
    one = put("Const", [], "int64", attributes={"value": 1})
    count = put("Const", [], "int64", attributes={
        "value": int(math.prod(shape)),
    })
    entry = state["current"]
    header = new_block("ew_loop", SSABlockKind.LOOP_HEADER)
    body = new_block("ew_body", SSABlockKind.LOOP_BODY)
    done = new_block("ew_done", SSABlockKind.LOOP_EXIT)
    jump(header)
    position, following = mint("int64"), mint("int64")
    state["current"] = header
    put("Phi", [zero, following], "int64", attributes={
        "incoming_blocks": (entry.name, body.name),
    }, res=position)
    going = put("Lt", [position, count], "bool")
    header.instrs.append(Instr("CondBr", [going], None, attributes={
        **lowered, "true_target": body.name, "false_target": done.name,
    }))
    header.successors = [body.name, done.name]
    state["current"] = body

    def element(value) -> SSAValue:
        if not _shaped(value):
            return value
        address = put("GetElementPtr", [value, position], "ptr")
        return put("Load", [address], "float64")

    operands = [element(value) for value in call.args]
    # The op's own record (``extraction_contract``, ``source_type``) rides
    # on the element op: ``Pow`` reads its Python form from it.
    attributes = {
        key: value for key, value in dict(call.attributes or {}).items()
        if key not in {"tensor_operation", "tensor_candidate", "tensor"}
    }
    computed = put(scalar_op, operands, "float64", attributes=attributes)
    target = put("GetElementPtr", [result, position], "ptr")
    put("Store", [computed, target], None)
    put("Add", [position, one], "int64", res=following)
    jump(header)
    state["current"] = done
    done.instrs.extend(post)
    done.successors = old_successors
    # Successor Phis now enter from the block holding the tail.
    for successor in old_successors:
        target_block = function.blocks.get(successor)
        for instruction in (target_block.instrs if target_block else ()):
            blocks = instruction.attributes.get("incoming_blocks")
            if str(instruction.op) == "Phi" and blocks:
                instruction.attributes["incoming_blocks"] = tuple(
                    done.name if str(origin) == block_name else origin
                    for origin in blocks
                )
    ordered = []
    for label, existing in function.blocks.items():
        ordered.append((label, existing))
        if label == block_name:
            ordered.extend((made.name, made) for made in added)
    function.blocks.clear()
    function.blocks.update(ordered)
    return name


def lower_transcendentals_to_signal_cores(functions, *, mode=None) -> dict:
    """Route scalar float64 ``math`` transcendentals through the signal cores.

    Program: any ``math.exp``/``math.log``/``math.sin``/... on a float.
    Wrong record: each backend spelled the op through its own table -- C
    libm, ``llvm.*`` intrinsics (``atan``/``asin`` refused), Fortran
    intrinsics and a 1e-6 sine LUT -- so the native value depended on the
    host library, not on the repository.  Rule: the numeric contract's
    ``transcendentals`` (``ssa_call_input_adapters.transcendental_policy``)
    names the route; ``signal_cores`` rewrites each op that has a row in
    ``TRANSCENDENTAL_SIGNAL_CORES`` into that ``WideExpansionEmitter``
    method's instructions, in place, so C, LLVM and Fortran all compile the
    same exact primitive ops (Add/Sub/Mul/Div/Fma/Floor/compare/Select).

    Python's raising outcomes are named faults: each predicate the core
    returns is guarded with ``CondBr`` to a ``turing_validation_error``
    block, as ``reduce_integer_pow`` guards its own.  A scalar
    transcendental with no core is refused by name: under ``signal_cores``
    it is never left on host math.

    Elementwise tensor/span transcendentals take the same route: under
    ``signal_cores`` each becomes a loop whose body is the scalar op
    (``scalarize_elementwise_transcendentals``), rewritten here like any
    other, so every element is the core's double behind the core's guards;
    the receipt counts them (``elementwise``).

    ``backend_intrinsic`` (the ``compute`` mode) rewrites nothing: each
    backend spells the op with its own math library (C libm, ``llvm.*`` or
    libm calls, gfortran intrinsics), scalar or elementwise.  The receipt
    names those ops (``left_native``).
    """

    from .concordance_declarations import (
        PRECISION_IDENTITIES, SSA_BLOCK, SSABlockKind,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_call_input_adapters import transcendental_policy
    from .ssa_record_return_state import function_scope_of, identity_cells
    from ..transmogrifier.ssa import BasicBlock

    policy = transcendental_policy(mode)
    receipt = {
        "policy": policy, "lowered": {}, "elementwise": {}, "left_native": {},
    }
    flavor = active_two_product_flavor()

    def scalar(instruction) -> bool:
        return (
            str(instruction.op) in TRANSCENDENTAL_OPS
            and _scalar_float64(instruction.res)
            and bool(instruction.args)
            and all(_scalar_float64(value) for value in instruction.args)
        )

    def routed(instruction) -> bool:
        if policy != "signal_cores" or not scalar(instruction):
            return False
        if str(instruction.op) not in TRANSCENDENTAL_SIGNAL_CORES:
            raise ValueError(
                f"scalar float {instruction.op} has no signal core; the "
                "numeric contract's transcendentals is 'signal_cores', "
                "which never leaves a math function on the host library "
                "(numeric mode 'compute' selects backend_intrinsic)"
            )
        return True

    def rewrite(function, block_name: str, index: int) -> None:
        block = function.blocks[block_name]
        call = block.instrs[index]
        op = str(call.op)
        row = TRANSCENDENTAL_SIGNAL_CORES[op]
        method = row.method
        post = block.instrs[index + 1:]
        old_successors = list(block.successors)
        emitted: list = []
        emitter = WideExpansionEmitter(
            function, emitted, two_product_flavor=flavor,
        )
        answer = (
            method(emitter, call) if callable(method)
            else getattr(emitter, method)(
                *row.leading, *call.args, **dict(row.keywords)
            )
        )
        faults = ()
        if isinstance(answer, tuple):
            answer, faults = answer
        if not isinstance(answer, (SSAValue, list, tuple)):
            raise ValueError(
                f"signal core {method!r} for {op} answered {answer!r}; "
                "expected a double, an expansion, or (value, faults)"
            )
        # An expansion is renormalised (|low| <= ulp(high)/2), so one
        # rounded add of its two leading limbs is its correctly rounded
        # double.
        result = (
            emitter.lead(list(answer)) if isinstance(answer, (list, tuple))
            else answer
        )
        if isinstance(faults, SSAValue):
            if row.fault is None:
                raise ValueError(
                    f"signal core {method!r} for {op} returned a fault "
                    "predicate and its route names no fault code for it"
                )
            faults = ((faults, *row.fault),)
        faults = [
            (predicate, int(code), str(name))
            for predicate, code, name in faults
            if predicate is not None
        ]
        producer = next(
            (
                instruction for instruction in emitted
                if instruction.res is not None
                and int(instruction.res.id) == int(result.id)
            ),
            None,
        )
        if producer is None:
            raise ValueError(
                f"signal core {method!r} for {op} returned a value it did "
                "not compute; the rewrite has no instruction to bind the "
                "op's result to"
            )
        # The op's own result id becomes the core's final value, so every
        # reader of the op reads the collapsed signal-core double.
        producer.res = call.res

        def bound(value):
            return call.res if int(value.id) == int(result.id) else value

        for instruction in emitted:
            instruction.args = [bound(value) for value in instruction.args]
            instruction.attributes["lowered_from"] = f"signal_core.{op}"
        faults = [
            (bound(predicate), code, name)
            for predicate, code, name in faults
        ]
        block.instrs = block.instrs[:index] + emitted
        added: list[BasicBlock] = []
        taken = set(function.blocks)
        current = block

        def new_block(stem: str, kind) -> BasicBlock:
            serial = 0
            name = f"{block_name}.{stem}"
            while name in taken:
                serial += 1
                name = f"{block_name}.{stem}.{serial}"
            taken.add(name)
            created = BasicBlock(name)
            added.append(created)
            # The block's ``ssa_block`` row, from the op it spells.
            owners = identity_cells(function, call.res, *call.args)
            book = current_identity_book()
            row = (function_scope_of(function), str(function.name), name)
            if owners and book.latest_ref(SSA_BLOCK, row) is None:
                book.post(
                    SSA_BLOCK, row, kind, stage=PRECISION_IDENTITIES,
                    provenance=Derived(owners), mode=Mode.CONCORD,
                )
            return created

        for predicate, code, name in faults:
            fault = new_block("math_fault", SSABlockKind.VALIDATION_FAIL)
            passing = new_block("math_checked", SSABlockKind.VALIDATION_PASS)
            current.instrs.append(Instr("CondBr", [predicate], None, attributes={
                "true_target": fault.name, "false_target": passing.name,
            }))
            current.successors = [fault.name, passing.name]
            fault.instrs.append(Instr("Call", [], None, attributes={
                "callee": "turing_validation_error", "error_code": code,
                "failure": name,
            }))
            fault.instrs.append(
                Instr("Br", [], None, attributes={"target": passing.name})
            )
            fault.successors = [passing.name]
            current = passing
        current.instrs.extend(post)
        current.successors = old_successors
        if current is not block:
            # Successor Phis now enter from the block holding the tail.
            for successor in old_successors:
                target = function.blocks.get(successor)
                for instruction in (target.instrs if target else ()):
                    blocks = instruction.attributes.get("incoming_blocks")
                    if str(instruction.op) == "Phi" and blocks:
                        instruction.attributes["incoming_blocks"] = tuple(
                            current.name if str(origin) == block_name
                            else origin
                            for origin in blocks
                        )
            ordered = []
            for name, existing in function.blocks.items():
                ordered.append((name, existing))
                if name == block_name:
                    ordered.extend((made.name, made) for made in added)
            function.blocks.clear()
            function.blocks.update(ordered)
        lowered = receipt["lowered"].setdefault(str(function.name), {})
        lowered[op] = lowered.get(op, 0) + 1

    for function in functions.values():
        # Elementwise tensor/span transcendentals first: each becomes a loop
        # whose body holds one scalar op, which the loop below then routes
        # like any scalar one.  Same contract, same cores, same guards.
        while policy == "signal_cores":
            site = next(
                (
                    (name, position)
                    for name, candidate in function.blocks.items()
                    for position, instruction in enumerate(candidate.instrs)
                    if _elementwise_transcendental(instruction) is not None
                ),
                None,
            )
            if site is None:
                break
            elementwise = scalarize_elementwise_transcendentals(
                function, *site,
            )
            counted = receipt["elementwise"].setdefault(
                str(function.name), {},
            )
            counted[elementwise] = counted.get(elementwise, 0) + 1
        while True:
            site = next(
                (
                    (name, position)
                    for name, candidate in function.blocks.items()
                    for position, instruction in enumerate(candidate.instrs)
                    if routed(instruction)
                ),
                None,
            )
            if site is None:
                break
            rewrite(function, *site)
        left = sorted({
            str(instruction.op)
            for candidate in function.blocks.values()
            for instruction in candidate.instrs
            if (
                str(instruction.op) in TRANSCENDENTAL_OPS
                or _elementwise_transcendental(instruction) is not None
            )
            and not str(
                instruction.attributes.get("lowered_from") or ""
            ).startswith("signal_core.")
        })
        if left:
            receipt["left_native"][str(function.name)] = left
        if (
            str(function.name) in receipt["lowered"]
            or str(function.name) in receipt["elementwise"] or left
        ):
            function.metadata["transcendental_pipeline"] = {
                "policy": policy,
                "lowered": dict(receipt["lowered"].get(str(function.name), {})),
                "elementwise": dict(
                    receipt["elementwise"].get(str(function.name), {})
                ),
                "left_native": tuple(left),
            }
    return receipt


# Operations a region may contain and still be removable when nothing reads
# its results: value construction and arithmetic only -- no stores, no
# calls, no table traffic.  ``LNot``, ``Select`` and ``isfinite`` are the
# operations the structural recovery itself builds a predicate chain from
# (``not``, the ``and``/``or`` operand selection, ``math.isfinite``); the
# sweep below is that recovery's counterpart, so it must know every operation
# the recovery emits or the chain's tail outlives its consumers.
_PURE_REGION_OPS = frozenset({
    "Const", "Neg", "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Abs",
    "Max", "Min", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "FloorDiv", "Mod",
    "Shl", "Shr", "bitand", "bitor", "bitxor", "invert", "LAnd", "LOr",
    "LNot", "Select", "isfinite",
    "Cast", "range", "string_token", "Ret",
})


def _post_dead_structural_retirement(function, instruction, *, sources=()) -> None:
    """Post one ``dead_structural_retirement`` row for a swept value.

    DERIVED(the value's ``ssa_value`` cell; the function's root cell when
    the value was never posted -- a structural value recovered after the
    control lowering).  A function with neither writes through the raw
    primitive, which the book tags ``Unsourced`` itself.
    """

    from .concordance_declarations import (
        CELL_SET, DEAD_STRUCTURAL_RETIREMENT, DEAD_STRUCTURAL_SWEEP,
    )
    from .identity_concordance import Mode, current_identity_book
    from .precompile_to_ssa import _post_derived_or_raw
    from .ssa_record_return_state import function_scope_of, identity_cells

    book = current_identity_book()
    cells = tuple(dict.fromkeys((*identity_cells(function, instruction.res), *sources)))
    if not cells:
        root = book.latest_ref(CELL_SET, (function_scope_of(function), 0))
        cells = () if root is None else (root,)
    _post_derived_or_raw(
        book, DEAD_STRUCTURAL_RETIREMENT,
        (function_scope_of(function), int(instruction.res.id)),
        (
            str(instruction.op),
            str(instruction.attributes.get("structural_operation")),
        ),
        cells, stage=DEAD_STRUCTURAL_SWEEP, mode=Mode.REVISE,
    )


def drop_dead_pure_structural_instructions(functions) -> int:
    """Remove unconsumed frontend structural values with no side effects."""

    from .identity_concordance import scope_local_accounting_ids

    removed = 0
    for function in functions.values():
        protected: set[int] = set()
        for key in (
            "source_output_value_ids", "required_source_value_ids"
        ):
            protected.update(map(int, function.metadata.get(key) or ()))
        for named_output in function.metadata.get("named_outputs") or ():
            try:
                protected.add(int(named_output[1]))
            except (TypeError, ValueError, IndexError):
                continue
        # Dynamic tensor descriptors refer to their shape/rank/count through
        # SSA accounting.  Those are real uses even though they are not
        # ordinary instruction operands; deleting their Const producers leaves
        # the descriptor naming an unavailable value at backend emission.
        for value in (
            *function.args,
            *(
                occurrence
                for block in function.blocks.values()
                for instruction in block.instrs
                for occurrence in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                )
            ),
        ):
            # Every scope-local id the value's accounting names (tensor
            # shape/rank/count, type tag, heap owner, optional payload).
            protected.update(scope_local_accounting_ids(value))
        changed = True
        while changed:
            changed = False
            consumed = {
                int(argument.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            }
            for block in function.blocks.values():
                retained = []
                for instruction in block.instrs:
                    result_id = (
                        None if instruction.res is None
                        else int(instruction.res.id)
                    )
                    is_structural = (
                        instruction.attributes.get("structural_operation")
                        is not None
                    )
                    if (
                        result_id is not None
                        and result_id not in consumed
                        and result_id not in protected
                        # Const is intrinsically pure.  Frontend-only values
                        # such as a legalized slice token do not always carry
                        # a structural_operation tag, but once unconsumed they
                        # must disappear here rather than reach a numerical
                        # backend as an impossible literal.
                        and (is_structural or instruction.op == "Const")
                        and instruction.op in _PURE_REGION_OPS
                    ):
                        _post_dead_structural_retirement(function, instruction)
                        removed += 1
                        changed = True
                        continue
                    retained.append(instruction)
                block.instrs = retained
    return removed


def drop_dead_pure_region_calls(functions) -> int:
    """Remove pure calls whose scalar results or aggregate projections are dead.

    Catalogue section 2.2's first load-bearing inhabitant, motivated by
    completeness rather than speed: the planner occasionally carves a
    value's construction into its own region, and the caller's loop
    machinery then never reads the projected results (the materialized
    ``range`` of a comprehension is the observed case, in ``re``'s
    ``_mk_bitmap``). The dead group still emits, so a backend without a
    spelling for the region's payload reports a shortfall for code nothing
    runs.

    Removal is conservative: the callee body must be pure (every op in
    ``_PURE_REGION_OPS``), the aggregate and every projection pointer must
    be consumed only inside the group, and every projected ``Load`` result
    must be unconsumed and absent from the caller's declared outputs.
    Returns the number of removed call groups.
    """

    removed_total = 0
    removed_callees: set[str] = set()
    for function in functions.values():
        protected: set[int] = set()
        for key in (
            "source_output_value_ids",
            # Source-linked calls and structural record construction are
            # installed after region carving.  Their exact authored feeds are
            # a liveness contract even if a late control rewrite has not yet
            # placed the consuming instruction beside the region projection.
            "required_source_value_ids",
        ):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue
        # Loop-carried ports are consumed through metadata, not necessarily
        # as instruction operands; a projected value serving as a port must
        # never be swept.
        for port_id, port_value in dict(
            function.metadata.get("carried_port_values") or {}
        ).items():
            protected.add(int(port_id))
            try:
                protected.add(int(port_value.id))
            except (AttributeError, TypeError, ValueError):
                continue

        consumers: dict[int, int] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    identity = int(argument.id)
                    consumers[identity] = consumers.get(identity, 0) + 1
                if instruction.op in ("Phi", "phi"):
                    # Phi incoming values may live only in the attribute
                    # record, not in args; the backends' latch copies are
                    # generated from exactly that record, so it consumes.
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            identity = int(value.id)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        consumers[identity] = consumers.get(identity, 0) + 1

        for block in function.blocks.values():
            drop: set[int] = set()
            for index, instruction in enumerate(block.instrs):
                if (
                    instruction.op not in ("Call", "call")
                    or instruction.res is None
                ):
                    continue
                callee = functions.get(
                    str(instruction.attributes.get("callee") or "")
                )
                if callee is None or any(
                    body_instruction.op not in _PURE_REGION_OPS
                    for body_block in callee.blocks.values()
                    for body_instruction in body_block.instrs
                ):
                    continue
                aggregate_id = int(instruction.res.id)
                if instruction.attributes.get("result_convention") != "ssa.aggregate":
                    # Record specialization can reduce a normalization helper
                    # to ``return receiver``. Its fields already alias their
                    # declared spans; an unused conceptual handle must not keep
                    # an otherwise dead receiver in the physical call frame.
                    # Only a bare return is sufficient proof here: numerical
                    # instructions can define in/out formals without a Store.
                    if any(body_instruction.op != "Ret"
                           for body_block in callee.blocks.values()
                           for body_instruction in body_block.instrs):
                        continue
                    if consumers.get(aggregate_id, 0) or aggregate_id in protected:
                        continue
                    function.metadata["discarded_pure_call_sites"] = (
                        *function.metadata.get("discarded_pure_call_sites", ()),
                        (instruction.attributes.get("plan_callsite_id"), callee.name),
                    )
                    drop.add(index)
                    removed_total += 1
                    removed_callees.add(str(instruction.attributes.get("callee") or ""))
                    continue
                group = [index]
                pointer_ids: set[int] = set()
                load_results: list = []
                index_const_ids: set[int] = set()
                for follow_index in range(index + 1, len(block.instrs)):
                    follower = block.instrs[follow_index]
                    if follower.res is None:
                        continue
                    if follower.op == "GetElementPtr" and follower.args and (
                        int(follower.args[0].id) == aggregate_id
                    ):
                        group.append(follow_index)
                        pointer_ids.add(int(follower.res.id))
                        for operand in follower.args[1:]:
                            index_const_ids.add(int(operand.id))
                    elif follower.op == "Load" and follower.args and (
                        int(follower.args[0].id) in pointer_ids
                    ):
                        group.append(follow_index)
                        load_results.append(follower.res)
                group_indices = set(group)
                # Group-internal consumption of the aggregate and pointers.
                internal: dict[int, int] = {}
                for member_index in group:
                    for operand in block.instrs[member_index].args:
                        identity = int(operand.id)
                        internal[identity] = internal.get(identity, 0) + 1
                if consumers.get(aggregate_id, 0) != internal.get(
                    aggregate_id, 0
                ):
                    continue
                if any(
                    consumers.get(pointer, 0) != internal.get(pointer, 0)
                    for pointer in pointer_ids
                ):
                    continue
                if any(
                    consumers.get(int(value.id), 0) > 0
                    or int(value.id) in protected
                    for value in load_results
                ):
                    continue
                drop.update(group_indices)
                # An index constant consumed only by the removed GEPs goes
                # with them; one shared elsewhere stays.
                for const_index, candidate in enumerate(block.instrs):
                    if (
                        candidate.op == "Const"
                        and candidate.res is not None
                        and int(candidate.res.id) in index_const_ids
                        and consumers.get(int(candidate.res.id), 0)
                        == internal.get(int(candidate.res.id), 0)
                    ):
                        drop.add(const_index)
                removed_total += 1
                removed_callees.add(
                    str(instruction.attributes.get("callee") or "")
                )
            if drop:
                block.instrs = [
                    instruction
                    for index, instruction in enumerate(block.instrs)
                    if index not in drop
                ]
    # A planned region whose last call site was removed would still be emitted
    # as an export root.  The planner can also leave a structural constant in
    # an independently minted region after the owner has materialized the same
    # deterministic source value directly (``b"\0asm"`` in build_module).
    # Such an uncalled region is not a compilation unit when its own explicit
    # source-integral contract publishes no outputs.  Sweep precisely that
    # case and receipt it on the owner; authored functions, effectful regions,
    # and independently useful regions with outputs all remain untouched.
    still_called = {
        str(instruction.attributes.get("callee") or "")
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in ("Call", "call")
    }
    if removed_callees:
        for callee_name in removed_callees:
            if (
                callee_name in functions
                and callee_name not in still_called
                and "__planned_region_" in callee_name
            ):
                del functions[callee_name]
    for region_name, region in tuple(functions.items()):
        integral = dict(region.metadata.get("source_region_integral") or {})
        if (
            "__planned_region_" not in str(region_name)
            or region_name in still_called
            or "output_value_ids" not in integral
            or tuple(integral.get("output_value_ids") or ())
            or any(
                instruction.op not in _PURE_REGION_OPS
                for block in region.blocks.values()
                for instruction in block.instrs
            )
        ):
            continue
        owner_name = str(integral.get("owner") or "")
        owner = functions.get(owner_name)
        if owner is not None:
            receipts = list(owner.metadata.get(
                "discarded_outputless_source_regions", ()
            ))
            receipts.append({
                "function": str(region_name),
                "identity_token_chain": tuple(
                    integral.get("identity_token_chain") or ()
                ),
                "reason": "uncalled-pure-no-published-outputs",
            })
            owner.metadata["discarded_outputless_source_regions"] = tuple(
                receipts
            )
        del functions[region_name]
        removed_total += 1
    return removed_total



#: Limb elements narrower than the binary64 default. Carrying one of these
#: retypes the value, because on a lane with no f64 the two statements
#: cannot be allowed to disagree.
_NARROW_LIMB_ELEMENTS = frozenset({"float32", "f32", "single"})

#: The largest finite double, for exact finiteness tests (``|v| <= max``).
_DOUBLE_MAX = 1.7976931348623157e308

_FLOAT64_SPELLINGS = frozenset({"float64"})
_INT64_SPELLINGS = frozenset({"int64"})


def narrow_float64_to_float32(module) -> int:
    """Retype every remaining binary64 value as binary32, once it is safe.

    The GPU lanes have no f64 AT ALL -- WGSL does not define the type --
    so a float64 value reaching them is not a precision choice, it is an
    impossibility. The choice that IS available is the one the ladder was
    built for: two float32 limbs in place of one float64, which carries
    more significand than the f64 it replaces rather than less.

    The precision expansion already types its own limbs from the declared
    element, so after it runs what is left at float64 is the surrounding
    scaffolding -- formals, the loads through them, and the exact integer
    floors of range reduction, none of which hold a limbed quantity. This
    narrows that scaffolding so the module speaks one element throughout.

    REFUSED when any precision section declares an element other than
    float32, because then the float64 values are load-bearing and
    narrowing them would be exactly the silent precision loss the whole
    pipeline exists to prevent. A module with no sections at all is also
    refused: nothing there is carrying precision in limbs, so narrowing it
    would simply be throwing digits away where the caller asked for none
    of this.

    Returns how many values were retyped.
    """

    receipt = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    ) or {}
    contracts = tuple(receipt.get("section_contracts", ()))
    if not contracts:
        raise ValueError(
            "narrow_float64_to_float32 refuses a module with no precision "
            "sections: there are no limbs carrying the precision that "
            "narrowing would rely on, so this would only lose digits"
        )
    for contract in contracts:
        if not contract.get("element"):
            raise ValueError(
                "narrow_float64_to_float32 refuses a precision section "
                f"contract with no recorded limb element: {contract!r}"
            )
    declared = {str(contract["element"]) for contract in contracts}
    unsupported = {
        element for element in declared
        if element not in _NARROW_LIMB_ELEMENTS
    }
    if unsupported:
        raise ValueError(
            "narrow_float64_to_float32 refuses a module whose sections "
            f"declare {sorted(unsupported)}: those limbs ARE float64 and "
            "narrowing them would silently halve the precision the caller "
            "declared"
        )

    seen: set[int] = set()
    narrowed = 0

    def retype(value) -> None:
        nonlocal narrowed
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        dtype = str(getattr(value, "dtype", None) or "")
        if dtype in _FLOAT64_SPELLINGS:
            value.dtype = "float32"
            narrowed += 1
        elif dtype in _INT64_SPELLINGS:
            # WGSL has no 64-bit integer either. These are the index and
            # quarter-turn counts around the arithmetic, not quantities --
            # a field addressed by an i32 index is bounded by the same
            # limit its buffers already are.
            value.dtype = "int32"
            narrowed += 1

    for function in (getattr(module, "functions", module) or {}).values():
        for formal in getattr(function, "args", ()):
            retype(formal)
        for block in getattr(function, "blocks", {}).values():
            for instruction in getattr(block, "instrs", ()):
                retype(getattr(instruction, "res", None))
                for argument in getattr(instruction, "args", ()):
                    retype(argument)
    return narrowed


#: Operations whose result is a truth value rather than a quantity.
_PREDICATE_NAMES = frozenset({
    "Lt", "Le", "Gt", "Ge", "Eq", "Ne",
    "lt", "le", "gt", "ge", "eq", "ne",
})



def _is_predicate_derived(instruction, definitions, predicates,
                          depth: int = 4) -> bool:
    """Whether this value descends from a comparison within a few steps.

    Shallow on purpose: a mask is formed close to the comparison that
    made it, and a deep search would start refusing genuine arithmetic
    that merely happens to sit downstream of a branch condition.
    """

    if depth <= 0:
        return False
    for argument in getattr(instruction, "args", ()):
        identifier = int(argument.id)
        if identifier in predicates:
            return True
        producer = definitions.get(identifier)
        if producer is not None and _is_predicate_derived(
            producer, definitions, predicates, depth - 1
        ):
            return True
    return False


def _source_numeric_scopes(function) -> tuple:
    """The numeric scopes the reducer keyed ``function``'s source facts by
    (its authored identity and its source numeric scope), as
    ``lower_precision_operations`` reads them."""

    metadata = dict(getattr(function, "metadata", {}) or {})
    return tuple(dict.fromkeys(
        scope for scope in (
            metadata.get("authored_function_identity"),
            metadata.get("source_numeric_scope"),
        )
        if scope is not None
    ))


def _source_precision_fact_cells(book, function, *values) -> tuple:
    """The reducer's precision facts on ``values`` under ``function``'s
    numeric scopes: the ``source_precision_boundary`` row a promoted value
    entered at and the ``source_precision_operator`` row a result was
    declared with.  These are the cells a width read off a value or an
    instruction attribute was written from."""

    from .concordance_declarations import (
        SOURCE_PRECISION_BOUNDARY, SOURCE_PRECISION_OPERATOR,
    )

    return tuple(dict.fromkeys(
        cell
        for scope in _source_numeric_scopes(function)
        for value in values
        if value is not None
        for cell in (
            book.latest_ref(SOURCE_PRECISION_BOUNDARY, (scope, int(value.id))),
            book.latest_ref(SOURCE_PRECISION_OPERATOR, (scope, int(value.id))),
        )
        if cell is not None
    ))


def carry_precision_through_ssa(functions) -> int:
    """Make a precision result self-describing, then rename what reads it.

    Not an identity -- nothing here is rewritten into a cheaper equivalent.
    It lives beside them because it consumes repository SSA under the same
    ``(functions) -> count`` shape, and because every identity written for
    the precision operations needs this to have run first: until it does,
    precision stops at the first undeclared temporary and a chain is only
    ever visible as isolated operations.

    Ingestion can name an operation ``precision_*`` only where the source
    DECLARED an operand's width, because it walks the AST and no SSA value
    exists yet to ask. The result of such an operation is a genuine limbed
    quantity, but it is emitted as a bare scalar -- ``float64``, shape
    ``()`` -- so the next operation cannot tell it apart from ordinary
    arithmetic and lowers as ordinary arithmetic.

    Here the values exist, so the property can live on the value the way it
    lives on an ``AbstractTensor``: the limb count becomes a channel in the
    LAST dimension, which is the same layout the tensor uses and the same
    one a backend will stride. Shape alone is ambiguous -- a trailing axis
    of 2 could be a real tensor axis -- so ``accounting`` records that the
    axis IS limbs. The shape is the layout; the accounting is the fact.

    Propagation then stops being inference and becomes a read: an ordinary
    operation one of whose operands carries the limb fact is a precision
    operation, and is renamed to say so. Iterated to a fixed point, since
    renaming one instruction makes its result the evidence for the next.
    """

    from ..common.tensors.topological_reducer import (
        PRECISION_CLOSED_OPERATIONS, PRECISION_SINGULAR_NAMES,
    )

    singular = dict(PRECISION_SINGULAR_NAMES)
    planted = {name: operation for operation, name in singular.items()}
    changed_total = 0
    from .concordance_declarations import (
        PRECISION_CARRY, PRECISION_CHANNEL_SHAPE,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import identity_cells

    book = current_identity_book()
    channel_shape_page = book.page(PRECISION_CHANNEL_SHAPE)

    def limbs_of(value) -> int:
        try:
            record = value.accounting or {}
        except AttributeError:
            return 1
        return max(int(record.get("precision_limbs") or 1), 1)

    def carry(
        function_name: str,
        function,
        instruction,
        limbs: int,
        element: str | None,
    ) -> bool:
        """Give one result value the limb channel and the fact."""

        value = instruction.res
        if value is None or limbs <= 1 or limbs_of(value) == limbs:
            return False
        value.accounting["precision_limbs"] = int(limbs)
        resolved = str(element or value.dtype or "") or None
        value.accounting["precision_element"] = resolved
        # The element is what the limbs ARE, so it is the value's dtype and
        # not merely a note beside it. Leaving the two to disagree is
        # invisible on a lane that has both widths and fatal on one that
        # does not: a float32 expansion kept saying float64, and the
        # WebGPU emitter -- where there IS no f64 -- refused sixty-seven
        # values whose limbs were already 24-bit. Only a NARROWING is
        # written here; the default element is the dtype the value already
        # carries, so a binary64 program is untouched.
        if resolved in _NARROW_LIMB_ELEMENTS and value.dtype != resolved:
            value.dtype = resolved
        existing = tuple(value.shape or ())
        # ``limbs_of`` above is the idempotence fact.  Extent equality is
        # not: a logical vector may naturally have length equal to its limb
        # width.  Treating `(2,)` as though it already contained the channel
        # made a two-element, two-limb value collapse to rank zero later.
        channel_shape = (*existing, int(limbs))
        row = (str(function_name), int(value.id))
        proposed = (existing, channel_shape, int(limbs))
        incumbent = channel_shape_page.latest(row)
        if incumbent is not None and tuple(incumbent) != proposed:
            raise ValueError(
                "precision channel shape concordance disagreement for "
                f"{row!r}: recorded={incumbent!r}, proposed={proposed!r}"
            )
        # The channel is read from the value and from the operands whose
        # limbs decided it, and those limbs entered at the reducer's
        # boundary / operator facts: DERIVED from exactly those cells.  A
        # same-fact re-post is CONCORD (the edge is recorded, no cell
        # written); a disagreeing fact raised above.
        book.post(
            PRECISION_CHANNEL_SHAPE, row, proposed, stage=PRECISION_CARRY,
            provenance=Derived(tuple(dict.fromkeys((
                *identity_cells(function, value, *instruction.args),
                *_source_precision_fact_cells(
                    book, function, value, *instruction.args,
                ),
            )))),
            mode=Mode.CONCORD,
        )
        value.accounting["precision_logical_shape"] = existing
        value.shape = channel_shape
        return True

    for function_name, function in functions.items():
        while True:
            changed = 0
            for block in function.blocks.values():
                for instruction in block.instrs:
                    operation = str(instruction.op)
                    attributes = instruction.attributes
                    if operation in planted:
                        limbs = max(
                            int(attributes.get("precision_limbs") or 1), 1
                        )
                        element = attributes.get("precision_element")
                    elif operation in PRECISION_CLOSED_OPERATIONS:
                        # An ordinary operation reading a limbed operand.
                        # Most limbs and highest precision decide, exactly
                        # as they do at ingestion.
                        limbs = max(
                            (limbs_of(value) for value in instruction.args),
                            default=1,
                        )
                        if limbs <= 1:
                            continue
                        element = next(
                            (
                                (value.accounting or {}).get(
                                    "precision_element"
                                )
                                for value in instruction.args
                                if limbs_of(value) > 1
                            ),
                            None,
                        )
                        instruction.op = singular[operation]
                        attributes["precision_limbs"] = int(limbs)
                        attributes["precision_element"] = element
                        attributes["lowered_from"] = operation
                        changed += 1
                    else:
                        continue
                    if carry(
                        str(function_name), function, instruction, limbs,
                        element,
                    ):
                        changed += 1
            # The backward half of the same fact: a value CONSUMED at width
            # n must be PRODUCED at width n. Forward renaming alone spreads
            # only from limbed values outward, which strands any plain
            # subchain that merely feeds a precision operation -- the even
            # cores' ``s = z * z`` computed both operands unlimbed, so the
            # squaring stayed ordinary, the loads behind ``z`` were never
            # recognised as limb-strided, and a two-limb kernel silently
            # read the wrong elements. Renaming the defining instruction
            # here makes its operands evidence for the next forward sweep,
            # so the recognition walks all the way back to the loads.
            definitions = {
                int(instruction.res.id): instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }
            predicates = {
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
                and str(instruction.op) in _PREDICATE_NAMES
            }
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if str(instruction.op) not in planted:
                        continue
                    limbs = max(int(
                        instruction.attributes.get("precision_limbs") or 1
                    ), 1)
                    if limbs <= 1:
                        continue
                    element = instruction.attributes.get("precision_element")
                    for argument in instruction.args:
                        if limbs_of(argument) > 1:
                            continue
                        producer = definitions.get(int(argument.id))
                        if (
                            producer is None
                            or str(producer.op)
                            not in PRECISION_CLOSED_OPERATIONS
                        ):
                            continue
                        # A value built from a COMPARISON is a mask: it is
                        # zero or one, exactly representable in one limb,
                        # and there is nothing for a second limb to hold.
                        # Widening it anyway is not merely wasted work --
                        # it puts an error-free transformation on a
                        # boolean, and the LLVM lane then tracks the same
                        # register as both i1 and double and emits a
                        # two_product residual over a predicate. Measured,
                        # a quadrant blend that the C lane computed
                        # exactly came back off by half on LLVM. The
                        # forward pass never reached these because a mask
                        # has no limbed operand; only this backward half
                        # can, so the guard belongs here.
                        if _is_predicate_derived(
                            producer, definitions, predicates
                        ):
                            continue
                        producer.attributes["precision_limbs"] = int(limbs)
                        producer.attributes["precision_element"] = element
                        producer.attributes["lowered_from"] = str(producer.op)
                        producer.op = singular[str(producer.op)]
                        changed += 1
            if not changed:
                break
            changed_total += changed
    return changed_total


@dataclasses.dataclass(frozen=True)
class PrecisionIdentity:
    """One reduction the precision operations admit, and why it is true.

    ``requires`` is what must exist before it may fire -- a proven fact on a
    value, or a kernel the bank can serve. An empty tuple is unconditional.
    ``exact`` means the rewrite changes no bit of the result; an inexact one
    would need the work contract's licence, and there are none here on
    purpose: every entry below is a theorem, not a trade.
    """

    name: str
    matches: tuple[str, ...]
    exact: bool
    requires: tuple[str, ...]
    effect: str
    justification: str


#: The identity table for the precision reduction phase.
#:
#: Ordered by what they save, cheapest evidence first. An expansion operation
#: is expensive -- a two-limb product is seven flops before renormalisation --
#: so the identities proving one ISN'T NEEDED are worth more than any that
#: make one faster, and they come first.
PRECISION_IDENTITIES: tuple[PrecisionIdentity, ...] = (
    PrecisionIdentity(
        name="negation_is_per_limb",
        matches=("precision_neg",),
        exact=True,
        requires=(),
        effect="negate each limb; no error-free transformation at all",
        justification=(
            "Negation is exact in every binary floating-point format: it "
            "flips a sign bit and cannot round. An expansion is the sum of "
            "its limbs, so negating each limb negates the sum exactly and "
            "preserves nonoverlapping-ness. precision_neg is therefore "
            "never an expansion operation, only a channel-wise sign flip."
        ),
    ),
    PrecisionIdentity(
        name="subtraction_is_addition_of_negation",
        matches=("precision_sub",),
        exact=True,
        requires=("negation_is_per_limb",),
        effect="rewrite sub(a, b) to add(a, neg(b))",
        justification=(
            "Exact because the negation it introduces is exact. Its value "
            "is not speed -- it is that a destination implements four "
            "operations instead of five, and the subtraction path cannot "
            "drift from the addition path because there is only one."
        ),
    ),
    PrecisionIdentity(
        name="exact_identity_element",
        matches=("precision_add", "precision_mul"),
        exact=True,
        requires=("constant_operand", "operand_special_value_fact"),
        effect="add(x, 0) -> x; mul(x, 1) -> x",
        justification=(
            "Only against a literal exact zero or one AND proved operand "
            "special-value facts. A value that merely "
            "ROUNDS to zero or one is not an identity element: adding a "
            "denormal changes the low limb even when it cannot change the "
            "high one, and that low limb is the entire point of the "
            "representation. Signed zero and NaN payload behaviour also "
            "disqualify a blind use rewrite: even x + (+0.0) changes -0.0."
        ),
    ),
    PrecisionIdentity(
        name="scaling_by_power_of_two",
        matches=("precision_mul",),
        exact=True,
        requires=("constant_operand", "exponent_range_fact"),
        effect="scale every limb by the power of two; no transformation",
        justification=(
            "Within a proved normal exponent range, multiplication by a "
            "power of two only shifts the exponent, so it is exact for every "
            "limb independently and preserves the "
            "nonoverlapping property that makes the limbs a valid "
            "expansion. This is what makes Cody-Waite argument reduction "
            "affordable. Without that range fact, overflow and gradual "
            "underflow make the shortcut false and it must not fire."
        ),
    ),
    PrecisionIdentity(
        name="sterbenz_cancellation",
        matches=("precision_sub",),
        exact=True,
        requires=("operand_range_fact",),
        effect="collapse to an ordinary Sub",
        justification=(
            "Sterbenz's lemma: if b/2 <= a <= 2b then a - b is exactly "
            "representable, so the dual is provably zero and computing it "
            "is computing zero the long way. This is the identity that took "
            "expm1 to 1.75 ulp with no core at all. It needs a proven range "
            "on both operands -- catalogue section 5's fact slot -- and must "
            "never fire on a guess, because where it is wrong it is wrong "
            "silently and by the whole residual."
        ),
    ),
    PrecisionIdentity(
        name="single_renormalisation_per_chain",
        matches=("precision_add", "precision_sub"),
        exact=True,
        requires=("chain_internal_consumers", "fixed_width_chain_proof"),
        effect="renormalise once at the end of a chain, not per operation",
        justification=(
            "Renormalisation itself preserves an expansion's exact sum, but "
            "a FIXED-width chain may discard different trailing terms when "
            "the intermediate renormalisation moves. The shortcut therefore "
            "requires a proof that the chain's retained width contains every "
            "intermediate term; internal-consumer shape alone is not proof."
        ),
    ),
    PrecisionIdentity(
        name="exact_accumulation_over_long_chain",
        matches=("precision_add",),
        exact=True,
        requires=("chain_internal_consumers", "kernel:superaccumulator"),
        effect="accumulate into a fixed-point superaccumulator",
        justification=(
            "Past some chain length, not renormalising at all beats "
            "renormalising once: an accumulator wide enough to span the "
            "exponent range absorbs each term in constant time with no "
            "rounding whatsoever, and is read out once. It is exact for ANY "
            "number of terms and any conditioning, which a fixed-width "
            "expansion is not -- so this is also the only entry here that "
            "improves accuracy rather than merely preserving it."
        ),
    ),
    PrecisionIdentity(
        name="two_product_kernel",
        matches=("precision_mul",),
        exact=True,
        requires=("kernel:two_product",),
        effect="call the banked two_product kernel for the dual",
        justification=(
            "The dual of a product is Dekker's splitting -- six multiplies "
            "and several adds -- or, where the hardware has a fused "
            "multiply-add, the single instruction fma(a, b, -p). Both are "
            "exact and they agree bit for bit, so this is variant selection "
            "rather than a licence to be inexact. Making it a BANKED KERNEL "
            "rather than a backend rule is what makes that claim checkable: "
            "the bank verifies every variant against the spec's oracle "
            "before serving it, so the fused variant is proven equal to the "
            "split one rather than trusted because a capability flag was set."
        ),
    ),
)


#: Identities true over the reals and false over floating point.
#:
#: Recorded because a reduction phase that does not know them will find them:
#: each is a rewrite some optimizer already performs, and each destroys the
#: representation silently. This is the table's other half -- knowing what
#: must NOT reduce is the same knowledge as knowing what may.
PRECISION_REFUSALS: tuple[PrecisionIdentity, ...] = (
    PrecisionIdentity(
        name="cancellation_of_added_term",
        matches=("precision_add", "precision_sub"),
        exact=False,
        requires=("never",),
        effect="REFUSED: (a + b) - b is not a",
        justification=(
            "The whole representation exists because a + b loses bits that "
            "b cannot give back. This rewrite is exactly the assumption "
            "that no bits were lost, which is the assumption the limbs "
            "exist to deny."
        ),
    ),
    PrecisionIdentity(
        name="reassociation_of_the_dual",
        matches=("precision_add", "precision_sub", "precision_mul"),
        exact=False,
        requires=("never",),
        effect="REFUSED: the error term must keep its written association",
        justification=(
            "A dual expression is algebraically zero -- that is what makes "
            "it the error term -- so a reassociating simplifier that sees "
            "it whole can fold it away entirely. Knuth's and Dekker's "
            "proofs hold for one specific order of operations and for no "
            "other; the primal may be reassociated freely, the dual may not "
            "be touched at all. "
            "MEASURED, because the stronger claim is easy to assume and was "
            "asserted here before it was checked: gfortran at -ffast-math "
            "did NOT fold a raw two_sum residual, on inputs that genuinely "
            "lose bits. Lowering one SSA value per statement, through named "
            "temporaries loaded from and stored to arrays, appears to be "
            "why -- no simplifier ever sees the residual as one expression. "
            "So this refusal is a standing hazard rather than an observed "
            "failure, and the structure is carrying more of the protection "
            "than the isolation flags are. Treat a destination that inlines "
            "more aggressively as untested, not as safe."
        ),
    ),
    PrecisionIdentity(
        name="contraction_across_the_primal",
        matches=("precision_add", "precision_sub"),
        exact=False,
        requires=("never",),
        effect="REFUSED: the primal must not fuse with its own producers",
        justification=(
            "two_sum requires s to be the correctly rounded sum of exactly "
            "its two operands. If an operand was itself a product and the "
            "multiply-add contracts, s becomes the rounded value of a "
            "different expression while the dual computes its residual "
            "against the materialised operand, and the pair no longer sums "
            "to what it claims. Reassociation of the primal is safe; "
            "contraction across it is not, and the distinction is exactly "
            "one flag on one instruction."
        ),
    ),
    PrecisionIdentity(
        name="distribution_over_addition",
        matches=("precision_mul",),
        exact=False,
        requires=("never",),
        effect="REFUSED: a * (b + c) is not a * b + a * c",
        justification=(
            "Two roundings on the right against one on the left, and the "
            "limb counts differ: the distributed form needs a wider "
            "expansion to hold the same value. Over the reals it is free; "
            "here it changes both the result and the storage it requires."
        ),
    ),
)


def reduce_precision_operations(functions) -> dict:
    """Fire the identities whose evidence is already in the SSA.

    Every rewrite here is exact, so none consults the work contract. Most
    do not restructure anything: they record HOW an operation is to be
    realised -- per-limb, scaled, renormalised or not -- as a fact on the
    instruction. That is the honest shape while destinations are still
    unwritten, because the decision is the identity's to make and the
    spelling is theirs, and it keeps the reduction phase from having to
    know either.

    Absent on purpose: ``exact_identity_element``. Dropping ``add(x, 0)``
    means pointing every consumer at ``x``, and this module deliberately
    avoids use-rewriting -- it is why ``x**1`` is missing from the ``Pow``
    reduction. It needs the same machinery that entry needs, and should
    arrive with it rather than growing a private copy here.

    ``scaling_by_power_of_two`` and ``sterbenz_cancellation`` wait on proven
    range facts; delayed renormalisation waits on a fixed-width chain proof;
    the two kernel entries wait on the bank. Pattern shape is never promoted
    into numerical evidence here.

    Returns a count per identity name.
    """

    from math import frexp

    from ..common.tensors.topological_reducer import (
        PRECISION_PACK_NAME,
        PRECISION_SINGULAR_NAMES,
    )

    add_name = PRECISION_SINGULAR_NAMES["Add"]
    sub_name = PRECISION_SINGULAR_NAMES["Sub"]
    mul_name = PRECISION_SINGULAR_NAMES["Mul"]
    neg_name = PRECISION_SINGULAR_NAMES["neg"]

    counts = {
        "negation_is_per_limb": 0,
        "subtraction_is_addition_of_negation": 0,
        "scaling_by_power_of_two": 0,
        "single_renormalisation_per_chain": 0,
    }
    from .concordance_declarations import (
        PRECISION_IDENTITIES, PRECISION_IDENTITY_VALUE,
    )
    from .ssa_record_return_state import mint_ssa_value

    def fresh(function, like) -> SSAValue:
        # Minted through the book from the cell of the value it is made
        # like (the subtrahend the identity negates).
        value = SSAValue(
            mint_ssa_value(
                function, PRECISION_IDENTITY_VALUE, (like,),
                dtype=like.dtype, shape=tuple(like.shape or ()),
                stage=PRECISION_IDENTITIES,
            ),
            dtype=like.dtype,
            shape=tuple(like.shape or ()),
            device=like.device, accounting=dict(like.accounting or {}),
        )
        return value

    def is_power_of_two(value: float) -> bool:
        # Exactly a power of two, and finite: frexp puts every such value
        # at a mantissa of exactly 0.5. Zero and the specials are excluded
        # because scaling by them is not an exponent shift.
        if not value or value != value or value in (float("inf"), -float("inf")):
            return False
        return frexp(abs(value))[0] == 0.5

    for function in functions.values():
        # -- negation is a sign flip, never an expansion -------------------
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) == neg_name:
                    instruction.attributes["precision_form"] = "per_limb"
                    counts["negation_is_per_limb"] += 1

        # -- subtraction is addition of a negation -------------------------
        # The result value is reused, so no consumer needs rewriting: only
        # a negation is inserted ahead of the operation that keeps the id.
        for block in function.blocks.values():
            if not any(
                str(instruction.op) == sub_name
                for instruction in block.instrs
            ):
                continue
            rewritten = []
            for instruction in block.instrs:
                if str(instruction.op) != sub_name or len(instruction.args) != 2:
                    rewritten.append(instruction)
                    continue
                left, right = instruction.args
                negated = fresh(function, right)
                rewritten.append(Instr(
                    neg_name, [right], negated,
                    attributes={
                        "precision_form": "per_limb",
                        "precision_limbs": instruction.attributes.get(
                            "precision_limbs"
                        ),
                        "precision_element": instruction.attributes.get(
                            "precision_element"
                        ),
                        "lowered_from": sub_name,
                    },
                ))
                instruction.op = add_name
                instruction.args = [left, negated]
                instruction.attributes["lowered_from"] = sub_name
                rewritten.append(instruction)
                counts["negation_is_per_limb"] += 1
                counts["subtraction_is_addition_of_negation"] += 1
            block.instrs = rewritten

        # -- scaling by a power of two is exact per limb -------------------
        constants: dict[int, float] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.op == "Const" and instruction.res is not None:
                    value = _constant_value(instruction)
                    if value is not None:
                        constants[int(instruction.res.id)] = value
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) != mul_name:
                    continue
                for operand in instruction.args:
                    scale = constants.get(int(operand.id))
                    if scale is None or not is_power_of_two(scale):
                        continue
                    if not instruction.attributes.get(
                        "precision_scale_exponent_range_proven"
                    ):
                        continue
                    instruction.attributes["precision_form"] = "scale_per_limb"
                    instruction.attributes["precision_scale"] = scale
                    counts["scaling_by_power_of_two"] += 1
                    break

        # -- one renormalisation per chain ---------------------------------
        # An intermediate expansion needs renormalising only if something
        # READS it. Anything reached through metadata counts as a reader:
        # a loop-carried port or a declared output is consumed through a
        # channel no scan of operands can see.
        protected: set[int] = set()
        for key in (
            "source_output_value_ids", "required_source_value_ids",
        ):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue
        for port_id, port_value in dict(
            function.metadata.get("carried_port_values") or {}
        ).items():
            protected.add(int(port_id))
            try:
                protected.add(int(port_value.id))
            except (AttributeError, TypeError, ValueError):
                continue

        readers: dict[int, list] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    readers.setdefault(int(argument.id), []).append(instruction)
                if instruction.op in ("Phi", "phi"):
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            readers.setdefault(int(value.id), []).append(
                                instruction
                            )
                        except (AttributeError, TypeError, ValueError):
                            continue

        precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    str(instruction.op) not in precision_ops
                    or instruction.res is None
                ):
                    continue
                result = int(instruction.res.id)
                consumers = readers.get(result, ())
                # A renormalisation is needed where a value ESCAPES, not
                # between every pair of operations.
                #
                # The rule used to require the single consumer to be another
                # ADD, which is the shape a sum chain has and the shape a
                # Horner never has: there every add feeds a multiply, so the
                # identity matched nothing on the most common expression in
                # the whole signal pack -- 0 of 8 identities fired on a sine
                # core. Any precision operation can consume an unrenormalised
                # pair; what cannot is anything else, because outside the
                # section the pair is just two doubles whose sum is the
                # value.
                internal = (
                    result not in protected
                    and len(consumers) == 1
                    and str(consumers[0].op) in precision_ops
                    and bool(instruction.attributes.get(
                        "precision_fixed_width_chain_proven"
                    ))
                )
                instruction.attributes["precision_renormalise"] = not internal
                if internal:
                    counts["single_renormalisation_per_chain"] += 1
    return counts


#: The fused multiply-add, as repository SSA owns it.
#:
#: ``Fma(x, y, z)`` is ``x * y + z`` under EXACTLY ONE rounding. That single
#: rounding is the entire content of the operation: a separate multiply and
#: add round twice, so the two are different functions, not the same function
#: at different speeds. Every destination has a spelling -- C's ``fma``,
#: GLSL's and WGSL's ``fma``, LLVM's ``llvm.fma`` -- so this is a real
#: operation to be implemented per backend rather than a hint dropped for an
#: optimizer to notice.
#:
#: Naming it here rather than leaving it to LLVM's ``contract`` flag buys
#: determinism: a flag PERMITS fusion, so whether it happens depends on the
#: toolchain, the named target, and the optimizer's mood, and the six
#: backends with no ``-O2`` behind them never see it at all. An instruction
#: either is an Fma or is not.
FMA = "Fma"


def contract_multiply_add_to_fma(functions, licensed: bool | None = None) -> dict:
    """Fuse ``Mul`` feeding an ``Add``/``Sub`` into :data:`FMA`, in place.

    INEXACT in general, and gated accordingly -- the default asks the active
    work contract's ``contract_multiply_add``, the same switch the LLVM
    backend consults. Fusing removes a rounding, which USUALLY improves the
    result and always changes it, so it is a licensed change of bits rather
    than a free win.

    The exception is the reason this exists. Where the multiply and the
    subtraction form an error term -- ``a * b - fl(a * b)`` -- the fused
    form is not merely more accurate but EXACTLY the residual Dekker's
    splitting computes the long way, six multiplies and several adds
    replaced by one instruction. There the rewrite is exact, and the
    licence is beside the point; it needs the dual to be marked as such,
    which is deferred work, so today it rides the general licence.

    Only fires where the product is consumed by that one operation.
    Otherwise the multiply must be kept for its other readers and fusing
    computes it twice -- once fused, once not, at two different roundings,
    which is worse than not fusing at all.

    Returns a count per rewrite kind.
    """

    if licensed is None:
        from .work_contract import active_contract

        licensed = bool(active_contract().contract_multiply_add)
    counts = {"add_to_fma": 0, "sub_to_fma": 0}
    if not licensed:
        return counts

    for function in functions.values():
        protected: set[int] = set()
        for key in ("source_output_value_ids", "required_source_value_ids"):
            protected.update(
                int(value_id)
                for value_id in (function.metadata.get(key) or ())
            )
        for name_value in (function.metadata.get("named_outputs") or ()):
            try:
                protected.add(int(name_value[1]))
            except (TypeError, ValueError, IndexError):
                continue

        readers: dict[int, int] = {}
        products = {}
        fused: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    identity = int(argument.id)
                    readers[identity] = readers.get(identity, 0) + 1
                if instruction.op in ("Phi", "phi"):
                    for _predecessor, value in (
                        instruction.attributes.get("incoming") or ()
                    ):
                        try:
                            identity = int(value.id)
                        except (AttributeError, TypeError, ValueError):
                            continue
                        readers[identity] = readers.get(identity, 0) + 1
                if (
                    str(instruction.op) == "Mul"
                    and len(instruction.args) == 2
                    and instruction.res is not None
                ):
                    products[int(instruction.res.id)] = instruction

        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation not in ("Add", "Sub") or len(instruction.args) != 2:
                    rewritten.append(instruction)
                    continue
                left, right = instruction.args
                # `Sub(c, Mul(a, b))` is deliberately not matched: it is
                # `c - a * b`, which needs a negated multiplicand to reach
                # this form, and that is a second rewrite wearing the first
                # one's name.
                product = products.get(int(left.id))
                addend = right
                if product is None and operation == "Add":
                    product = products.get(int(right.id))
                    addend = left
                if product is None:
                    rewritten.append(instruction)
                    continue
                carrier = int(product.res.id)
                if readers.get(carrier, 0) != 1 or carrier in protected:
                    rewritten.append(instruction)
                    continue

                factors = list(product.args)
                if operation == "Sub":
                    from .concordance_declarations import (
                        PRECISION_IDENTITIES, PRECISION_IDENTITY_VALUE,
                    )
                    from .ssa_record_return_state import mint_ssa_value

                    # Minted through the book from the addend it negates.
                    negated = SSAValue(
                        mint_ssa_value(
                            function, PRECISION_IDENTITY_VALUE, (addend,),
                            dtype=addend.dtype,
                            shape=tuple(addend.shape or ()),
                            stage=PRECISION_IDENTITIES,
                        ),
                        dtype=addend.dtype,
                        shape=tuple(addend.shape or ()),
                        device=addend.device,
                        accounting=dict(addend.accounting or {}),
                    )
                    rewritten.append(Instr(
                        "Neg", [addend], negated,
                        attributes={"lowered_from": "Sub"},
                    ))
                    addend = negated
                    counts["sub_to_fma"] += 1
                else:
                    counts["add_to_fma"] += 1
                instruction.op = FMA
                instruction.args = [*factors, addend]
                instruction.attributes["lowered_from"] = operation
                instruction.attributes["roundings"] = 1
                rewritten.append(instruction)
                fused.add(carrier)
                products.pop(carrier, None)
            block.instrs = rewritten

        # The fused multiplies are dead -- their one reader absorbed them.
        # Established by recounting what the rewritten code actually reads,
        # rather than by reasoning about what should now be unreferenced.
        if fused:
            live: set[int] = set()
            for block in function.blocks.values():
                for instruction in block.instrs:
                    for argument in instruction.args:
                        live.add(int(argument.id))
                    if instruction.op in ("Phi", "phi"):
                        for _predecessor, value in (
                            instruction.attributes.get("incoming") or ()
                        ):
                            try:
                                live.add(int(value.id))
                            except (AttributeError, TypeError, ValueError):
                                continue
            for block in function.blocks.values():
                block.instrs = [
                    each for each in block.instrs
                    if not (
                        each.res is not None
                        and int(each.res.id) in fused
                        and int(each.res.id) not in live
                        and int(each.res.id) not in protected
                    )
                ]
    return counts


# ---------------------------------------------------------------------------
# The API between SSA Fma and precision sections.
#
# A precision section is a stretch of SSA whose operations carry limbs. An
# `Fma` inside one is not the same request as an `Fma` in ordinary code: in
# ordinary code the single rounding is an improvement a contract licenses,
# and emitting a multiply and an add instead is merely a slower, slightly
# different answer. Inside a precision section the single rounding IS the
# operation -- `a * b - fl(a * b)` under two roundings is not a worse
# residual, it is zero -- so the same instruction carries an obligation the
# ordinary one does not.
#
# This is what the two sides have to agree on, stated once here so each
# backend can later be tuned against it rather than each inventing its own
# reading. Nothing below emits or lowers anything; it is the description a
# destination is measured against.
# ---------------------------------------------------------------------------

#: The section's `Fma` instructions are MANDATORY, not licensed.
#:
#: Deliberately not "Fma rounds once" -- that is what `Fma` means, it is true
#: in ordinary code as well, and a destination that cannot spell the op
#: shortfalls through the ordinary mechanism without any help from here.
#: What is section-specific is the opposite of the usual policy: an `Fma`
#: elsewhere APPEARS ONLY WHEN `contract_multiply_add` licenses it and must
#: not appear under `prove` at all, whereas here it is the residual itself,
#: so it has to be emitted whatever the contract says. A destination that
#: drops it because the contract forbids contraction has not been
#: conservative; it has computed zero.
FMA_MANDATORY = "fma.mandatory_regardless_of_contract"

#: A destination must not reassociate or contract within the section, except
#: at instructions the section itself declares as `Fma`. Both halves matter
#: and they pull opposite ways: the error terms are algebraically zero, so
#: any reassociating simplifier deletes them, while the declared Fma is a
#: contraction that must happen. "Optimize nothing here" and "optimize
#: everything here" are both wrong; the section says which instructions are
#: which.
SECTION_ISOLATION = "section.no_reassociation_except_declared_fma"

#: A destination may stage the section's lanes concurrently, or serially.
#: Unlike the other two this is permission, never obligation -- serial is
#: always a correct answer -- so a backend that ignores it is conforming,
#: not failing.
LANE_STAGING = "lanes.optional_concurrency"

#: Every obligation a precision section can place on a destination.
PRECISION_SECTION_OBLIGATIONS = (
    FMA_MANDATORY, SECTION_ISOLATION, LANE_STAGING,
)


#: The two exact spellings of ``a * b == p + e``, and who needs which.
#:
#: ``"fma"`` is the two-instruction form: ``e = fma(a, b, -p)``. It is the
#: cheap spelling, and the one that makes FMA_MANDATORY appear on a section's
#: contract, because the single rounding IS the residual.
#:
#: ``"split"`` is Dekker's spelling through Veltkamp splitting -- seventeen
#: ordinary operations, no fma anywhere. Same exact residual; the theorem
#: needs only IEEE mul/add/sub. Choosing it is how a destination with no
#: single-rounding fma (WebAssembly; WGSL and GLSL, whose ``fma()`` is
#: permitted to round twice) hosts a section HONESTLY: the obligation is
#: discharged by never being incurred, and the price is instruction count,
#: not accuracy. The eager twin lives at
#: ``common.tensors.extended_precision.two_product`` and the two must stay
#: instruction-for-instruction alike.
TWO_PRODUCT_FLAVORS = ("fma", "split")


#: The flavour the completed-module seam lowers with when its caller has no
#: way to say. The seam sits inside the source compiler, many layers below
#: anyone choosing a destination -- mirroring how ``active_contract()``
#: reaches the emitters -- so the choice travels the same way: a scoped
#: ambient value, defaulting to the cheap spelling.
_ACTIVE_TWO_PRODUCT_FLAVOR = "fma"


def active_two_product_flavor() -> str:
    """The flavour ``apply_precision_pipeline`` uses when not told."""

    return _ACTIVE_TWO_PRODUCT_FLAVOR


@_contextlib.contextmanager
def two_product_flavor_scope(flavor: str):
    """Lower every module built inside this scope with ``flavor``."""

    global _ACTIVE_TWO_PRODUCT_FLAVOR
    if flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    held = _ACTIVE_TWO_PRODUCT_FLAVOR
    _ACTIVE_TWO_PRODUCT_FLAVOR = str(flavor)
    try:
        yield
    finally:
        _ACTIVE_TWO_PRODUCT_FLAVOR = held


def _veltkamp_constant(element) -> float:
    """The splitting constant for a limb element.

    Sourced from ``extended_precision.LIMB_ELEMENTS`` so the compiler and
    the eager twin can never disagree about it. Splitting with the wrong
    constant does not degrade the residual, it invalidates the theorem --
    so an unknown element is a refusal, never a default. An ABSENT element
    is the binary64 default the whole scalar path assumes.
    """

    from ..common.tensors.extended_precision import limb_element_facts

    return float(limb_element_facts(element)["split"])


#: What each destination actually delivers, as opposed to what it emits.
#:
#: The four lanes present a UNIFIED FRONT: every one of them accepts `Fma`
#: and produces something, so a program compiles on all four and no caller
#: has to write four versions. This table is what keeps that front from
#: becoming a lie. Emitting is not meeting: WebAssembly has no fma
#: instruction and expands into a multiply and an add, which rounds twice,
#: so it does not declare FMA_MANDATORY -- and a precision section asking
#: for one is refused HERE, before emission, rather than discovered later
#: in a residual that came back zero.
#:
#: Ordinary code is unaffected by that refusal. Code that merely wanted the
#: accuracy of an fma gets the arithmetic on every lane; only code whose
#: correctness depends on the single rounding is turned away, and only where
#: the single rounding is not available.
BACKEND_PRECISION_CAPABILITIES: dict[str, tuple[str, ...]] = {
    # C99 `fma`, and `#pragma STDC FP_CONTRACT OFF`. Both in the language.
    "c": (FMA_MANDATORY, SECTION_ISOLATION),
    # @llvm.fma.f64 is the operation, not a licence to fuse. Isolation is
    # withholding the fast-math flags, which are already per-instruction.
    "llvm": (FMA_MANDATORY, SECTION_ISOLATION),
    # IEEE_FMA is F2018 and the lane already uses IEEE_ARITHMETIC. Fortran
    # forbids reassociating a parenthesised expression -- and every emitted
    # binary is parenthesised -- while contraction, the one rewrite the
    # language leaves to the processor, is withdrawn per-artifact by the
    # toolchain: emit_module marks a module that carries sections and
    # aggressive_fortran_flags adds -ffp-contract=off for exactly those.
    # An explicit ieee_fma call is an intrinsic invocation, not a
    # contraction, so it survives the flag. Both halves together are the
    # whole of isolation, so the claim is now whole too.
    "fortran": (FMA_MANDATORY, SECTION_ISOLATION),
    # No fma instruction exists -- a section that CONTAINS one is refused,
    # because the expansion rounds twice and the residual comes back zero.
    # Isolation, though, wasm delivers by construction: its floating point
    # is deterministic IEEE 754 and an engine has no licence to reassociate
    # or contract anything. So a section lowered with the "split" flavour of
    # two_product -- no Fma to spell -- runs here honestly, just longer.
    "wasm": (SECTION_ISOLATION,),
    # SPIR-V spells the single rounding as GLSL.std.450 Fma and withdraws
    # contraction per-instruction with the NoContraction decoration; both
    # are in the assembler. Float64 limbs are fine: OpCapability Float64 is
    # emitted on demand, and the f32-only restriction covers only the
    # extended-instruction transcendentals a limb section never uses.
    "spirv": (FMA_MANDATORY, SECTION_ISOLATION),
    # WGSL's fma() is explicitly permitted to round twice, so FMA_MANDATORY
    # is not claimed and never will be -- a section containing an Fma is
    # still refused here, which is what the fma_value_ids gate in
    # precision_backend_shortfalls already decides on its own.
    #
    # SECTION_ISOLATION is claimed for the "split" flavour, which is the
    # route this comment always named. Split two_product spells no fma at
    # all: Veltkamp halves at 4097 (2**12 + 1), and for a 24-bit f32
    # significand the 12-bit half products are exact, so the residual is
    # recovered by subtraction rather than by a fused rounding. That
    # removes the failure this lane was guarding against -- an fma-shaped
    # residual coming back algebraically zero -- because there is no fused
    # operation left to double-round.
    #
    # What is NOT settled by the language is reassociation: two_sum's
    # `(a - (s - b')) + (b - b')` is only an error-free transformation if
    # it is evaluated as written. So this claim is carried by MEASUREMENT
    # rather than by a spec argument -- tools/probe_webgpu_precision.py
    # scores the emitted shader against exact rational truth in a real
    # browser, and a lane that reassociated would show it immediately as a
    # width that stops buying precision.
    "webgpu": (SECTION_ISOLATION,),
    # Desktop GLSL compute (GL 4.3 + GL_ARB_gpu_shader5): the `precise`
    # qualifier is exactly contraction-and-reassociation control -- values
    # so qualified must be computed as written, and an fma() call on
    # precise operands is required to be the single-rounding operation.
    # The lane's emitter qualifies every section temporary `precise` and
    # spells Fma as fma(), which is what makes both claims true. This key
    # is the DESKTOP lane; the browser fragment lane is "webgl" below.
    "glsl": (FMA_MANDATORY, SECTION_ISOLATION),
    # GLSL ES 3.0 fragment shaders: f32 only, no fma(), `precise` is
    # desktop 4.40+. Nothing claimable.
    "webgl": (),
}


def unmet_precision_obligations(contract, backend: str) -> tuple[str, ...]:
    """What ``backend`` cannot honour for one section. Empty means it may."""

    return contract.unmet_by(
        BACKEND_PRECISION_CAPABILITIES.get(str(backend).casefold(), ())
    )


@dataclasses.dataclass(frozen=True)
class PrecisionSectionContract:
    """What one precision section presents, and what it requires of a host.

    Derived from the instruction stream rather than declared by an author,
    so it cannot drift from the code it describes.
    """

    function: str
    #: Result ids of the section's instructions, in program order.
    value_ids: tuple[int, ...]
    #: The precision operation names present.
    operations: tuple[str, ...]
    #: Result ids of explicit `Fma` instructions, or abstract multiplication
    #: operations whose exact lowering necessarily introduces one.
    fma_value_ids: tuple[int, ...]
    #: Widest limb count and element type in the section.
    limbs: int
    element: str | None
    obligations: tuple[str, ...]

    def as_record(self) -> dict:
        return {
            "function": self.function,
            "value_ids": list(self.value_ids),
            "operations": list(self.operations),
            "fma_value_ids": list(self.fma_value_ids),
            "limbs": self.limbs,
            "element": self.element,
            "obligations": list(self.obligations),
        }

    def unmet_by(self, capabilities) -> tuple[str, ...]:
        """Obligations a destination declaring ``capabilities`` cannot meet.

        Empty means the section may be lowered there. Anything else is a
        shortfall to report BEFORE emitting, not a degradation to discover
        in the numbers afterwards -- the failures here do not announce
        themselves, they return plausible numbers whose residual is zero.
        ``LANE_STAGING`` never appears: it is permission, so no destination
        can fail it. ``FMA_MANDATORY`` appears only for a section that
        actually contains one.
        """

        declared = frozenset(str(each) for each in capabilities)
        required = set(self.obligations) - {LANE_STAGING}
        if not self.fma_value_ids:
            required.discard(FMA_MANDATORY)
        return tuple(sorted(required - declared))


def precision_section_contracts(
    functions, two_product_flavor: str = "fma",
) -> tuple[PrecisionSectionContract, ...]:
    """Read the contract each precision section places on a destination.

    A section is the maximal run of consecutive instructions in one block
    that carry limbs, plus any `Fma` sitting among them. Adjacency is the
    right criterion here rather than dataflow reachability: what a
    destination must refrain from reoptimizing is a contiguous stretch of
    emitted code, and an `Fma` between two limbed operations is inside the
    stretch whether or not it consumes one of them.

    ``two_product_flavor`` must match what ``lower_precision_operations``
    will actually emit, since the contract is recorded before the lowering
    runs and describes the instructions that WILL exist.
    """

    from ..common.tensors.topological_reducer import (
        PRECISION_PACK_NAME,
        PRECISION_SINGULAR_NAMES,
    )

    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
    # Under the "fma" flavour, exact lowering of a multiplication or a
    # division necessarily introduces an Fma, so those operations carry the
    # obligation before the instruction exists. Under "split" they lower to
    # ordinary arithmetic and incur nothing; only an EXPLICIT Fma in the
    # stream still demands the real single rounding, because no expansion
    # can substitute for a request the author spelled out.
    precision_fma_ops = {
        PRECISION_SINGULAR_NAMES["Mul"],
        PRECISION_SINGULAR_NAMES["Div"],
        # The symbolic Exp/Log cores are Horner chains over wide
        # multiplication. Their source operation therefore carries the same
        # exact-product obligation as an explicit wide multiply, before the
        # core is materialised into individual SSA operations below.
        PRECISION_SINGULAR_NAMES["Exp"],
        PRECISION_SINGULAR_NAMES["Log"],
    } if two_product_flavor == "fma" else set()
    contracts: list[PrecisionSectionContract] = []

    for name, function in functions.items():
        for block in function.blocks.values():
            runs: list[list] = []
            current: list = []
            for instruction in block.instrs:
                if str(instruction.op) in precision_ops or (
                    current and str(instruction.op) == FMA
                ):
                    current.append(instruction)
                    continue
                if current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)

            for run in runs:
                if not any(str(each.op) in precision_ops for each in run):
                    continue
                limbs, element = 1, None
                for each in run:
                    limbs = max(
                        limbs, int(each.attributes.get("precision_limbs") or 1)
                    )
                    # Taken independently of the limb count: an operation can
                    # carry a width without yet knowing its element type, and
                    # reading the element off whichever instruction happened
                    # to be widest loses one that a neighbour does know.
                    #
                    # The RESULT VALUE is asked before the instruction. An
                    # operation named at ingestion keeps whatever element its
                    # attributes were stamped with -- usually none, since the
                    # declaration gives limbs and no type -- while the carry
                    # pass resolves the element onto the value from its dtype.
                    # Reading only the attributes reports None for a section
                    # whose element is perfectly well known.
                    if element is None:
                        result = each.res
                        element = (
                            (result.accounting or {}).get("precision_element")
                            or (result.dtype if result is not None else None)
                            if result is not None else None
                        ) or each.attributes.get("precision_element")
                contracts.append(PrecisionSectionContract(
                    function=str(name),
                    value_ids=tuple(
                        int(each.res.id) for each in run
                        if each.res is not None
                    ),
                    operations=tuple(sorted({str(each.op) for each in run})),
                    fma_value_ids=tuple(
                        int(each.res.id) for each in run
                        if (
                            str(each.op) == FMA
                            or str(each.op) in precision_fma_ops
                        ) and each.res is not None
                    ),
                    limbs=limbs,
                    element=None if element is None else str(element),
                    obligations=PRECISION_SECTION_OBLIGATIONS,
                ))
    return tuple(contracts)


#: Stamped on every instruction inside a precision section, and the one thing
#: about the section that must reach a destination.
PRECISION_SECTION_ATTRIBUTE = "precision_section"


def mark_precision_sections(functions) -> int:
    """Stamp each precision section's instructions so emission can see it.

    The OPERATOR does not survive: `precision_mul` and its siblings exist to
    be recognised, propagated along, and reduced against, and are then
    expanded into an error-free transformation before any destination is
    reached. That is deliberate -- it is why no backend implements them.

    But the expansion is plain Add/Sub/Mul that looks like everybody else's,
    and an optimiser has no way to tell a Dekker split from the same
    expression written by someone who would be glad to see it folded. So the
    section boundary has to survive in the operator's place, and this is what
    carries it.

    Stamping the instruction rather than recording a range is what makes it
    survive: an expansion that inherits its source's attributes stays marked
    without anything having to re-derive where the section went, and a pass
    that moves an instruction cannot silently move it out of its section.

    Returns the number of instructions marked.
    """

    wanted: dict[str, set[int]] = {}
    for contract in precision_section_contracts(functions):
        wanted.setdefault(contract.function, set()).update(contract.value_ids)

    marked = 0
    for name, function in functions.items():
        inside = wanted.get(str(name))
        if not inside:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.res is not None
                    and int(instruction.res.id) in inside
                    and not instruction.attributes.get(
                        PRECISION_SECTION_ATTRIBUTE
                    )
                ):
                    instruction.attributes[PRECISION_SECTION_ATTRIBUTE] = True
                    marked += 1
    return marked


class WideExpansionEmitter:
    """Spell width-N expansion arithmetic as repository SSA, one limb per value.

    This is the expansion builder ``lower_precision_operations`` uses, lifted
    out of that pass so the insertion context is explicit: ``function`` is the
    function values are minted in, ``emitted`` is the instruction list new
    instructions are appended to, and ``limbs`` maps a value id to its limbs
    (high first; an absent id is an ordinary one-limb value). Any pass that
    holds those three can materialise the same ``signal_symbolic`` proofs and
    Precision arithmetic -- ``proof_core``, ``exp_expansion``,
    ``log_expansion`` -- outside a Precision section.

    ``element`` is the limb element the section being expanded declared
    (``None`` for the value's own dtype); a caller moving across instructions
    sets it per instruction. ``known_scalar_ids`` is the set of rank-0
    constant ids ``put`` reads to choose which operand carries the extent;
    the emitter adds every constant it mints. ``two_product_flavor`` selects
    the exact-product spelling (see ``TWO_PRODUCT_FLAVORS``).

    Every instruction it emits is stamped ``PRECISION_SECTION_ATTRIBUTE`` with
    a ``lowered_from`` naming its origin, exactly as the in-pass closures did,
    so the SSA ``lower_precision_operations`` produces is unchanged.
    """

    def __init__(
        self, function, emitted: list, *, two_product_flavor: str = "fma",
        limbs: dict | None = None, known_scalar_ids: set | None = None,
        element=None,
    ) -> None:
        if two_product_flavor not in TWO_PRODUCT_FLAVORS:
            raise ValueError(
                f"unknown two_product flavour {two_product_flavor!r}; "
                f"expected one of {TWO_PRODUCT_FLAVORS}"
            )
        from .concordance_declarations import (
            PRECISION_CONSTANT, PRECISION_EXPANSION, PRECISION_LIMB,
        )
        from .ssa_record_return_state import mint_ssa_value

        self._mint = mint_ssa_value
        self._limb_kind = PRECISION_LIMB
        self._constant_kind = PRECISION_CONSTANT
        self._stage = PRECISION_EXPANSION
        self.function = function
        self.emitted = emitted
        self.two_product_flavor = two_product_flavor
        self.limbs = {} if limbs is None else limbs
        self.known_scalar_ids = (
            set() if known_scalar_ids is None else known_scalar_ids
        )
        # The element the section being expanded declared, when it is
        # narrower than the value it is expanding. An expansion mints
        # many values and every one of them is a LIMB, so the element
        # is what they are -- inheriting the pre-expansion dtype makes
        # a float32 expansion claim to be float64, which is wrong in
        # two ways at once. It refuses to emit on a lane that has no
        # f64 at all, and, worse, ``splitter_for`` reads the same
        # dtype to choose Veltkamp's constant: a float32 limb split at
        # 2**27 + 1 instead of 4097 does not halve a 24-bit
        # significand, so the half products stop being exact and the
        # residual is quietly wrong rather than loudly absent.
        self.element = element
        self._splitters: dict[float, object] = {}

    # -- value and instruction minting --------------------------------------

    def fresh(self, like, extent_like=None):
        declared = self.element
        extent_like = like if extent_like is None else extent_like
        like_dtype = str(like.dtype or "")
        extent_dtype = str(extent_like.dtype or "")
        dtype = (
            declared if declared in _NARROW_LIMB_ELEMENTS
            else extent_like.dtype
            if (
                extent_dtype.casefold() in {
                    "float", "float16", "float32", "float64",
                    "double", "f16", "f32", "f64",
                }
                and like_dtype.casefold() not in {
                    "float", "float16", "float32", "float64",
                    "double", "f16", "f32", "f64",
                }
            )
            else like.dtype
        )
        logical_shape = tuple(
            (extent_like.accounting or {}).get(
                "precision_logical_shape"
            )
            or tuple(extent_like.shape or ())
        )
        carried_width = max(int(
            (extent_like.accounting or {}).get(
                "precision_limbs"
            ) or 1
        ), 1)
        if (
            not (extent_like.accounting or {}).get(
                "precision_logical_shape"
            )
            and
            carried_width > 1
            and logical_shape
            and logical_shape[-1] == carried_width
        ):
            logical_shape = logical_shape[:-1]
        # Minted through the book from the cells of the value it is
        # shaped like and the one whose extent it carries.
        value = SSAValue(
            self._mint(
                self.function, self._limb_kind, (like, extent_like),
                dtype=dtype, shape=logical_shape,
                stage=self._stage,
            ),
            dtype=dtype,
            shape=logical_shape,
            device=like.device,
            accounting={
                "precision_extent_source_id": int(extent_like.id),
            },
        )
        return value

    def put(self, op, args, like):
        known_scalar_ids = self.known_scalar_ids

        def extent_strength(value):
            accounting = value.accounting or {}
            shape = tuple(value.shape or ())
            carried_width = max(int(
                accounting.get("precision_limbs") or 1
            ), 1)
            declared_logical_shape = accounting.get(
                "precision_logical_shape"
            )
            logical_rank = (
                len(tuple(declared_logical_shape))
                if declared_logical_shape is not None
                else len(shape) - int(
                    carried_width > 1
                    and bool(shape)
                    and shape[-1] == carried_width
                )
            )
            declared_rank = max(
                logical_rank,
                int(accounting.get("program_abi_rank", 0) or 0),
                int(accounting.get("ssa_call_rank", 0) or 0),
            )
            return (
                declared_rank,
                int(value.id) not in known_scalar_ids,
                bool(shape),
            )

        extent_like = max(
            tuple(args) or (like,),
            key=extent_strength,
        )
        result = self.fresh(like, extent_like)
        self.emitted.append(Instr(op, list(args), result, attributes={
            PRECISION_SECTION_ATTRIBUTE: True,
            "lowered_from": "precision",
        }))
        return result

    def parts(self, value):
        return self.limbs.get(int(value.id), [value])

    def zero(self, like):
        """Materialise an exact low limb for scalar promotion.

        Repeating the high limb would represent ``x + x`` rather
        than ``x``.  A plain scalar is the expansion ``[x, +0, ...]``;
        IEEE positive zero is exact in every supported element
        format and therefore needs no numerical precondition.
        """

        result = self.fresh(like)
        self.emitted.append(Instr("Const", [], result, attributes={
            "constant": 0.0,
            PRECISION_SECTION_ATTRIBUTE: True,
            "lowered_from": "precision.scalar_promotion",
        }))
        self.known_scalar_ids.add(int(result.id))
        return result

    def scalar_constant(self, number, like):
        # Minted through the book from the value it is emitted for.
        result = SSAValue(
            self._mint(
                self.function, self._constant_kind, (like,),
                dtype=like.dtype, shape=(), stage=self._stage,
            ),
            dtype=like.dtype,
            shape=(),
            device=like.device,
        )
        self.emitted.append(Instr("Const", [], result, attributes={
            "constant": float(number),
            PRECISION_SECTION_ATTRIBUTE: True,
            "lowered_from": "precision.scalar_constant",
        }))
        self.known_scalar_ids.add(int(result.id))
        return result

    # -- error-free transformations -----------------------------------------

    def two_sum(self, a, b):
        put = self.put
        s = put("Add", (a, b), a)
        bb = put("Sub", (s, a), a)
        left = put("Sub", (a, put("Sub", (s, bb), a)), a)
        right = put("Sub", (b, bb), a)
        return s, put("Add", (left, right), a)

    def splitter_for(self, value):
        """One Const per emitter per element, not one per product."""

        magnitude = _veltkamp_constant(value.dtype)
        held = self._splitters.get(magnitude)
        if held is None:
            held = self.fresh(value)
            self.emitted.append(Instr("Const", [], held, attributes={
                "constant": float(magnitude),
                PRECISION_SECTION_ATTRIBUTE: True,
                "lowered_from": "precision.split_constant",
            }))
            self._splitters[magnitude] = held
        return held

    def two_product(self, a, b):
        put = self.put
        product = put("Mul", (a, b), a)
        if self.two_product_flavor == "fma":
            negated = put("Neg", (product,), a)
            return product, put(FMA, (a, b, negated), a)
        # Dekker through Veltkamp splitting: the same exact residual
        # with no fma anywhere, for destinations that cannot deliver
        # a single rounding. The eager twin is
        # ``extended_precision.two_product`` / ``_split``; the
        # operation order here mirrors it exactly so the two stay
        # provably the same algorithm.
        constant = self.splitter_for(a)

        def split(value):
            scaled = put("Mul", (value, constant), value)
            high = put(
                "Sub",
                (scaled, put("Sub", (scaled, value), value)),
                value,
            )
            return high, put("Sub", (value, high), value)

        a_high, a_low = split(a)
        b_high, b_low = split(b)
        error = put(
            "Sub", (put("Mul", (a_high, b_high), a), product), a
        )
        error = put(
            "Add", (error, put("Mul", (a_high, b_low), a)), a
        )
        error = put(
            "Add", (error, put("Mul", (a_low, b_high), a)), a
        )
        return product, put(
            "Add", (error, put("Mul", (a_low, b_low), a)), a
        )

    # -- expansion arithmetic -----------------------------------------------

    def expanded(self, value, width: int):
        held = list(self.parts(value))
        if len(held) < width:
            held.extend(self.zero(value) for _ in range(width - len(held)))
        return held[:width]

    def renormalise(self, terms, width: int):
        """Distil arbitrary exact terms to ``width`` SSA limbs."""

        rest = list(terms)
        if not rest:
            raise ValueError("cannot renormalise an empty expansion")
        kept = []
        for _index in range(int(width)):
            if not rest:
                break
            carry = rest[-1]
            tail = [None] * (len(rest) - 1)
            for position in range(len(rest) - 2, -1, -1):
                carry, tail[position] = self.two_sum(
                    rest[position], carry
                )
            kept.append(carry)
            rest = tail
        while len(kept) < int(width):
            kept.append(self.zero(kept[-1]))
        for leftover in rest:
            kept[-1] = self.put("Add", (kept[-1], leftover), kept[-1])
        return kept

    def add_expansions(self, left, right, width: int):
        return self.renormalise([*left, *right], width)

    def multiply_expansions(self, left, right, width: int):
        products = []
        for left_limb in left:
            for right_limb in right:
                high, low = self.two_product(left_limb, right_limb)
                products.extend((high, low))
        return self.renormalise(products, width)

    def lead(self, expansion):
        return (
            expansion[0] if len(expansion) == 1
            else self.put(
                "Add", (expansion[0], expansion[1]), expansion[0]
            )
        )

    def divide_expansions(self, left, right, width: int):
        put = self.put
        remainder = list(left)
        quotient = []
        for _index in range(int(width) + 1):
            digit = put(
                "Div", (self.lead(remainder), self.lead(right)), left[0]
            )
            quotient.append(digit)
            product = self.multiply_expansions(
                [digit], right, int(width) + 2
            )
            remainder = self.add_expansions(
                remainder,
                [put("Neg", (term,), term) for term in product],
                int(width) + 2,
            )
        return self.renormalise(quotient, width)

    def sum_expansion(self, source):
        """The ordinary value used only for a range/seed decision."""

        total = source[0]
        for term in source[1:]:
            total = self.put("Add", (total, term), total)
        return total

    def constant_expansion(self, parts_, like):
        return [self.scalar_constant(part, like) for part in parts_]

    # -- transcendental expansions ------------------------------------------

    def sqrt_expansion(self, source, width: int):
        """The eager Precision Newton law over repository limbs."""

        put = self.put
        collapsed = self.sum_expansion(source)
        seed = put("Sqrt", (collapsed,), collapsed)
        root = self.expanded(seed, width)
        half = self.scalar_constant(0.5, seed)
        steps = max(1, (int(width) - 1).bit_length()) + 1
        for _step in range(steps):
            quotient = self.divide_expansions(source, root, width)
            root = self.add_expansions(root, quotient, width)
            root = [put("Mul", (term, half), term) for term in root]
        return root

    def proof_core(self, name: str, argument, width: int):
        """Materialise the registered symbolic proof as wide SSA.

        The identity, interval-derived order, exact rational
        coefficients, and structural parity all come from
        signal_symbolic. This pass only spells that already-authored
        program with the expansion arithmetic above.
        """

        from ..common.tensors import signal_symbolic as proof

        count = proof.order_for(
            name, proof.CORE_RADII[name], digits=16 * int(width),
        )
        degree = proof.order_to_degree(name, count)
        coefficients = proof.structured_coefficients(name, degree)
        element = self.element
        coefficient_values = [
            self.constant_expansion(
                proof.limb_decomposition(
                    coefficient, width, element=element,
                ),
                argument[0],
            )
            for coefficient in coefficients
        ]
        structure = proof.TRANSCENDENTALS[name]["structure"]
        base = (
            self.multiply_expansions(argument, argument, width)
            if structure in ("odd", "even") else list(argument)
        )
        result = coefficient_values[-1]
        for coefficient in reversed(coefficient_values[:-1]):
            result = self.add_expansions(
                coefficient,
                self.multiply_expansions(base, result, width),
                width,
            )
        if structure in ("odd", "factored"):
            result = self.multiply_expansions(argument, result, width)
        return result

    def _ln2_head(self, like):
        """ln(2) rounded once to a double, from the exact rational.

        ``signal_symbolic.constant_limbs`` derives it from the proof's own
        constant, so no host library supplies a core constant; it is the
        same double ``math.log(2.0)`` returns.
        """

        from ..common.tensors.signal_symbolic import constant_limbs

        return self.scalar_constant(constant_limbs("ln2", 1)[0], like)

    # -- library-free range decisions --------------------------------------
    #
    # exp and log choose a binade and rescale by it. Those decisions used to
    # be spelled ``Pow(2, k)`` and ``Log`` -- host libm in every backend --
    # and ``2**k`` multiplied inside the expansion overflowed for k >= 1024
    # and flushed subnormal results. Everything below is compares, ``Select``
    # and multiplies by power-of-two constants, all exact in IEEE binary64.
    # Measured on ~20k arguments each against mpmath: exp 99.12% -> 100%
    # correctly rounded (it returned 0 for subnormal results and was 1 ulp
    # off below 2**-968), log 80.8% -> 100% (NaN for every subnormal
    # input). The edge points are pinned by
    # test_precision_exp_log_range_decisions_are_exact_and_library_free.

    def power_of_two_factors(self, count, like):
        """Exact factors whose product is ``2**count``, all on one side of 1.

        ``count`` is an integer-valued float with ``|count| < 2048``. Its
        magnitude is peeled bit by bit with compares against power-of-two
        constants; bit ``i`` selects ``2**(+-2**i)`` or 1. Every factor is a
        representable power of two (bit 10 is two ``2**+-512``), so a value
        multiplied by them in turn is exact for as long as it stays in the
        normal range, and rounds exactly once when the last one leaves it.
        Bits 0..8 are folded into one factor (at most ``2**+-511``).
        """

        put = self.put
        constant = self.scalar_constant
        zero = constant(0.0, like)
        one = constant(1.0, like)
        negative = self.compare("Lt", count, zero)
        remaining = self.select(
            negative, put("Neg", (count,), count), count,
        )
        factors = []
        low = None
        for bit in range(10, -1, -1):
            step = float(2 ** bit)
            taken = self.compare("Ge", remaining, constant(step, like))
            remaining = put(
                "Sub",
                (remaining, self.select(taken, constant(step, like), zero)),
                remaining,
            )
            exponent = min(2 ** bit, 512)
            factor = self.select(
                taken,
                self.select(
                    negative,
                    constant(2.0 ** -exponent, like),
                    constant(2.0 ** exponent, like),
                ),
                one,
            )
            if bit == 10:
                factors.extend((factor, factor))
            elif bit == 9:
                factors.append(factor)
            else:
                low = factor if low is None else put(
                    "Mul", (low, factor), factor,
                )
        factors.append(low)
        return factors

    def scale_by_factors(self, value, factors):
        for factor in factors:
            value = self.put("Mul", (value, factor), value)
        return value

    def binade(self, collapsed):
        """``(e, factors)`` with ``collapsed * 2**-e`` in ``[0.75, 1.5)``.

        For a positive finite ``collapsed`` only; the factors rescale any
        limb exactly (applied in order). Subnormal inputs are first lifted by
        ``2**600``; then a greedy ladder divides by ``2**s`` while the value
        is ``>= 2**s`` and multiplies by ``2**s`` while it is ``< 2**(1-s)``
        (s = 512 ... 1), leaving ``[1, 2)``; a last halving above 1.5
        centres the mantissa on 1 so ``log`` near 1 loses nothing to
        ``m - 1``. Every step is a compare, a ``Select`` and an exact
        power-of-two multiply.
        """

        put = self.put
        constant = self.scalar_constant
        like = collapsed
        one = constant(1.0, like)
        zero = constant(0.0, like)
        tiny = self.compare("Lt", collapsed, constant(2.0 ** -1000, like))
        lift = self.select(tiny, constant(2.0 ** 600, like), one)
        exponent = self.select(tiny, constant(-600.0, like), zero)
        current = put("Mul", (collapsed, lift), collapsed)
        down = None
        up = None
        for bit in range(9, -1, -1):
            step = 2 ** bit
            taken = self.compare(
                "Ge", current, constant(2.0 ** step, like),
            )
            factor = self.select(taken, constant(2.0 ** -step, like), one)
            current = put("Mul", (current, factor), current)
            exponent = put(
                "Add",
                (exponent, self.select(taken, constant(float(step), like), zero)),
                exponent,
            )
            down = factor if down is None else put(
                "Mul", (down, factor), factor,
            )
        for bit in range(9, -1, -1):
            step = 2 ** bit
            taken = self.compare(
                "Lt", current, constant(2.0 ** (1 - step), like),
            )
            factor = self.select(taken, constant(2.0 ** step, like), one)
            current = put("Mul", (current, factor), current)
            exponent = put(
                "Sub",
                (exponent, self.select(taken, constant(float(step), like), zero)),
                exponent,
            )
            up = factor if up is None else put("Mul", (up, factor), factor)
        high = self.compare("Ge", current, constant(1.5, like))
        half = self.select(high, constant(0.5, like), one)
        exponent = put(
            "Add", (exponent, self.select(high, one, zero)), exponent,
        )
        return exponent, (lift, up, down, half)

    def exp_expansion(self, source, width: int):
        """exp by Cody-Waite ln(2) reduction, the proof core, exact rescale.

        ``k = nearest(x / ln2)`` on the collapsed value (clamped to
        ``[-746, 710]``, beyond which exp is 0 or inf), ``r = x - k*ln2``
        with ``ln2`` as ``constant_limbs`` limbs, ``exp(r)`` from the proof
        core, then ``* 2**k`` by exact power-of-two factors -- after the core
        is canonicalised to ``(s, t)``, ``s = fl(s + t)``. A normal result is
        ``s*2**k + t*2**k``, exact until it overflows to inf (the tail is
        dropped below ``2**-968``, where it could itself round). A subnormal
        result is rounded ONCE to the ``2**-1074`` grid: ``H = s*2**(k+1074)``
        (``< 2**52``) is rounded to an integer by ``2**52`` and the tie the
        exact tail ``t`` breaks is fixed up, then scaled back exactly.
        """

        from ..common.tensors.signal_symbolic import (
            INTEGER_SHIFTER, constant_limbs,
        )

        put = self.put
        constant = self.scalar_constant
        collapsed = self.sum_expansion(source)
        like = collapsed
        zero = constant(0.0, like)
        one = constant(1.0, like)
        upper = constant(710.0, like)
        lower = constant(-746.0, like)
        overflow = self.compare("Gt", collapsed, upper)
        underflow = self.compare("Lt", collapsed, lower)
        clamped = self.select(
            overflow, upper, self.select(underflow, lower, collapsed),
        )
        ln2_parts = self.constant_expansion(
            constant_limbs("ln2", width, element=self.element),
            source[0],
        )
        ln2_head = self._ln2_head(source[0])
        # Nearest integer by the shifter (``signal_symbolic``'s own
        # ``INTEGER_SHIFTER``): two IEEE adds every backend spells. ``Round``
        # has no Fortran spelling and its tie rule differs between C
        # (``nearbyint``) and LLVM (``roundeven``); any integer within one
        # of x/ln2 keeps the remainder inside the core's interval.
        shifter = constant(INTEGER_SHIFTER, like)
        binade = put(
            "Sub",
            (put(
                "Add",
                (put("Div", (clamped, ln2_head), clamped), shifter),
                clamped,
            ), shifter),
            clamped,
        )
        remainder = self.add_expansions(
            source,
            [
                put("Neg", (term,), term)
                for term in self.multiply_expansions(
                    ln2_parts, [binade], width,
                )
            ],
            width,
        )
        core = self.proof_core("exp", remainder, width)
        head, tail = self.two_sum(core[0], self.sum_expansion(core[1:]))
        # exp(r) lies in [0.70, 1.42]: the result is below 2**-1022 exactly
        # when k < -1022, or k == -1022 with s < 1.
        floor_binade = constant(-1022.0, like)
        subnormal = self.select(
            self.compare("Lt", binade, floor_binade),
            one,
            self.select(
                self.compare("Eq", binade, floor_binade),
                self.select(self.compare("Lt", head, one), one, zero),
                zero,
            ),
        )
        subnormal = self.compare("Eq", subnormal, one)
        shift = self.select(
            subnormal,
            put("Add", (binade, constant(1074.0, like)), binade),
            binade,
        )
        factors = self.power_of_two_factors(shift, like)
        scaled_head = self.scale_by_factors(head, factors)
        scaled_tail = self.scale_by_factors(tail, factors)
        # Subnormal: scaled_head = H in [0, 2**52), scaled_tail = L exact.
        grid = constant(2.0 ** 52, like)
        integral = put(
            "Sub", (put("Add", (scaled_head, grid), scaled_head), grid),
            scaled_head,
        )
        fraction = put("Sub", (scaled_head, integral), scaled_head)
        half = constant(0.5, like)
        adjust = self.select(
            self.compare("Eq", fraction, half),
            self.select(self.compare("Gt", scaled_tail, zero), one, zero),
            self.select(
                self.compare("Eq", fraction, put("Neg", (half,), half)),
                self.select(
                    self.compare("Lt", scaled_tail, zero),
                    put("Neg", (one,), one),
                    zero,
                ),
                zero,
            ),
        )
        back = constant(2.0 ** -537, like)
        rounded = put(
            "Mul",
            (put("Mul", (put("Add", (integral, adjust), integral), back),
                 integral), back),
            integral,
        )
        # Below 2**-968 the scaled tail can itself be subnormal, so it is
        # rounded to the 2**-1074 grid and can land exactly on half an ulp
        # of the head, where the collapse's tie-to-even then moves the
        # correctly rounded head by one ulp (x = -706.2225020067164 gave
        # 1.9564303102951204e-307, want ...121e-307). There the head alone
        # is the correctly rounded result -- ``s = fl(s + t)`` scaled
        # exactly -- and a tail under 2**-1021 cannot be carried anyway.
        low = self.compare("Lt", scaled_head, constant(2.0 ** -968, like))
        result = [
            self.select(subnormal, rounded, scaled_head),
            self.select(
                subnormal, zero, self.select(low, zero, scaled_tail),
            ),
        ]
        result.extend(zero for _ in range(int(width) - 2))
        infinity = constant(float("inf"), like)
        return [
            self.select(
                overflow, infinity if index == 0 else zero,
                self.select(underflow, zero, limb),
            )
            for index, limb in enumerate(result)
        ]

    def exp_overflow_predicate(self, source, result):
        """True where Python's ``math.exp`` raises ``OverflowError``.

        That is exactly a finite argument whose correctly rounded exp is
        inf, read from ``result`` -- the limbs ``exp_expansion`` returned for
        ``source`` -- rather than from a hand threshold, so the predicate
        and the value can never disagree (709.782712893384 is finite,
        709.7827128933841 overflows). ``+inf`` gives inf without raising in
        Python and is not in it; NaN is not in it. Like
        ``log_domain_predicate`` the emitter does not fault: the pass
        lowering a Python ``math.exp`` guards on this.
        """

        collapsed = self.sum_expansion(source)
        value = self.sum_expansion(result)
        infinity = self.scalar_constant(float("inf"), collapsed)
        finite_value = self.select(
            self.compare("Lt", collapsed, infinity),
            value,
            self.scalar_constant(0.0, collapsed),
        )
        return self.compare("Eq", finite_value, infinity)

    def log_domain_predicate(self, source):
        """True where ``log`` has no real value: ``x <= 0`` (incl. -0, -inf).

        Python raises ``ValueError`` there. The emitter does not choose
        between a fault and an IEEE value: ``log_expansion`` leaves the IEEE
        limit (-inf at zero, NaN below) in the limbs, and the pass lowering a
        Python ``math.log`` guards on this predicate. NaN is not in it.
        """

        collapsed = self.sum_expansion(source)
        return self.compare(
            "Le", collapsed, self.scalar_constant(0.0, collapsed),
        )

    def log_expansion(self, source, width: int):
        """log by an exact binade ladder plus the log1p proof core.

        ``x = m * 2**e`` with ``m`` in ``[0.75, 1.5)`` (``binade``), the
        limbs rescaled exactly. Near 1 (``m <= 1.25``) the core takes
        ``m - 1`` directly, so ``log(1 - 2**-53)`` keeps every bit; above
        it two wide square roots bring the mantissa into the core's
        interval and the core is scaled by 4. ``e*ln2`` is restored with
        ``constant_limbs`` limbs. ``+inf`` gives ``+inf``; ``x <= 0`` gives
        the IEEE limit (see ``log_domain_predicate``); NaN propagates.
        """

        from ..common.tensors.signal_symbolic import constant_limbs

        put = self.put
        constant = self.scalar_constant
        collapsed = self.sum_expansion(source)
        like = collapsed
        exponent, factors = self.binade(collapsed)
        mantissa = [self.scale_by_factors(limb, factors) for limb in source]
        one = self.constant_expansion(
            (1.0, *((0.0,) * (int(width) - 1))), source[0],
        )
        minus_one = [put("Neg", (term,), term) for term in one]
        reduced = self.sqrt_expansion(
            self.sqrt_expansion(mantissa, width), width,
        )
        near = self.compare(
            "Le", self.sum_expansion(mantissa), constant(1.25, like),
        )
        direct = self.add_expansions(mantissa, minus_one, width)
        rooted = self.add_expansions(reduced, minus_one, width)
        offset = [
            self.select(near, each, other)
            for each, other in zip(direct, rooted)
        ]
        core = self.proof_core("log1p", offset, width)
        multiplier = self.select(near, constant(1.0, like), constant(4.0, like))
        core = [put("Mul", (term, multiplier), term) for term in core]
        ln2_parts = self.constant_expansion(
            constant_limbs("ln2", width, element=self.element),
            source[0],
        )
        restored = self.multiply_expansions(ln2_parts, [exponent], width)
        result = self.add_expansions(core, restored, width)
        zero = constant(0.0, like)
        infinity = constant(float("inf"), like)
        nan = constant(float("nan"), like)
        positive_infinity = self.compare("Eq", collapsed, infinity)
        at_zero = self.compare("Eq", collapsed, zero)
        negative = self.compare("Lt", collapsed, zero)
        head_special = self.select(
            positive_infinity, infinity,
            self.select(
                at_zero, put("Neg", (infinity,), infinity),
                self.select(negative, nan, result[0]),
            ),
        )
        special = self.select(
            positive_infinity, constant(1.0, like),
            self.select(
                at_zero, constant(1.0, like),
                self.select(negative, constant(1.0, like), zero),
            ),
        )
        special = self.compare("Eq", special, constant(1.0, like))
        return [head_special, *(
            self.select(special, zero, limb) for limb in result[1:]
        )]

    # -- circular functions over the whole double range ----------------------
    #
    # sin, cos and tan of one float64, correctly rounded, as straight-line
    # SSA: an exact reduction onto [-pi/4, pi/4] carried in limbs, the
    # ``sin``/``cos`` proof cores, a quadrant ``Select``, and one collapse.
    # Every op is IEEE-exact or a correctly rounded + - * / fma; the only
    # constants are exact rationals from ``signal_symbolic``. No host library
    # makes a value or a range decision. (trig cores, package S2b; the
    # measurement in route_math_to_signal_cores.md found the Cody-Waite
    # split refused beyond ~2**40, so large arguments take the windowed
    # 2/pi reduction below.)

    def compare(self, op: str, left, right):
        """A ``bool`` predicate over two limbs (``Lt``/``Le``/``Eq``/...)."""

        shape = tuple(
            (left.accounting or {}).get("precision_logical_shape")
            or tuple(left.shape or ())
        )
        result = SSAValue(
            self._mint(
                self.function, self._limb_kind, (left, right),
                dtype="bool", shape=shape, stage=self._stage,
            ),
            dtype="bool", shape=shape, device=left.device,
        )
        self.emitted.append(Instr(op, [left, right], result, attributes={
            PRECISION_SECTION_ATTRIBUTE: True,
            "lowered_from": "precision.compare",
        }))
        return result

    def select(self, condition, if_true, if_false):
        """A true select -- never the arithmetic blend that cancels."""

        return self.put("Select", (condition, if_true, if_false), if_true)

    def _modulo(self, value, modulus: float, centred: bool):
        """``value`` reduced by an exact multiple of ``modulus`` (a power of 2).

        ``value * (1/modulus)`` is exact, ``Floor``/``Round`` of it is an
        exact integer and the product back is exact, so the difference is
        exact whenever it is representable -- and it is: it keeps only bits
        ``value`` already had, below ``modulus``. The centred form keeps a
        small NEGATIVE value as it is (a floored one would need
        ``modulus - tiny``, which rounds).
        """

        put = self.put
        scaled = put(
            "Mul", (value, self.scalar_constant(1.0 / modulus, value)), value,
        )
        whole = put("Round" if centred else "Floor", (scaled,), value)
        return put("Sub", (
            value,
            put("Mul", (whole, self.scalar_constant(modulus, value)), value),
        ), value)

    def _cody_waite_reduction(self, magnitude, width: int):
        """``|x| - k*pi/2`` for ``|x| < CIRCULAR_CODY_WAITE_LIMIT``.

        The split is ``cody_waite_for`` at the limit, so every ``k*piece`` is
        exact by construction and the residual accumulates in the limbs;
        the quotient is the shifter's exact nearest integer, as
        ``signal_symbolic.reduce_argument`` takes it.
        """

        from fractions import Fraction

        from ..common.tensors.signal_symbolic import (
            INTEGER_SHIFTER, cody_waite_for,
        )

        put = self.put
        pieces = cody_waite_for(
            CIRCULAR_CODY_WAITE_LIMIT, "pi", Fraction(1, 2),
        )
        whole = sum(Fraction(piece) for piece in pieces)
        shifter = self.scalar_constant(INTEGER_SHIFTER, magnitude)
        quotient = put("Sub", (
            put("Add", (
                put("Mul", (
                    magnitude, self.scalar_constant(float(1 / whole), magnitude),
                ), magnitude),
                shifter,
            ), magnitude),
            shifter,
        ), magnitude)
        carried = int(width) + 1
        reduced = self.expanded(magnitude, carried)
        for piece in pieces:
            exact = put(
                "Mul", (quotient, self.scalar_constant(piece, magnitude)),
                magnitude,
            )
            reduced = self.add_expansions(
                reduced, [put("Neg", (exact,), exact)], carried,
            )
        return (
            self.renormalise(reduced, width),
            self._modulo(quotient, 4.0, centred=False),
        )

    def _payne_hanek_reduction(self, magnitude, width: int):
        """``|x| - k*pi/2`` for any finite ``|x| >= 1``, from windows of 2/pi.

        ``|x| = m * 2**(24*j)`` with ``1 <= m < 2**24`` (an exact compare /
        ``Select`` ladder of power-of-two scalings). ``m``'s ulp is at least
        2**-52, so ``m * I`` is a multiple of 4 for any multiple ``I`` of
        2**54: of ``2**(24*j) * 2/pi`` only ``F_j = (2**(24*j) * 2/pi) mod
        2**54`` matters for the quadrant and the fraction. ``F_j`` is an
        exact rational cut into double limbs at build time
        (``circular_payne_hanek_windows``) and chosen by ``j`` with
        ``Select``. Each ``m * f`` is an exact fma product pair, each half
        is reduced mod 4 exactly, and the sum of those small terms is
        ``|x| * 2/pi`` mod 4 to far below the ~2**-61 closest approach of a
        double to a multiple of pi/2.
        """

        from fractions import Fraction

        from ..common.tensors.signal_symbolic import constant_limbs

        put = self.put
        step = CIRCULAR_PAYNE_HANEK_STEP
        windows = circular_payne_hanek_windows()
        scaled = magnitude
        window = self.scalar_constant(0.0, magnitude)
        for bit in range(len(windows).bit_length() - 1, -1, -1):
            shift = step * (1 << bit)
            if shift > 1023:
                continue
            above = self.compare(
                "Ge", scaled, self.scalar_constant(2.0 ** shift, magnitude),
            )
            scaled = self.select(above, put("Mul", (
                scaled, self.scalar_constant(2.0 ** -shift, magnitude),
            ), magnitude), scaled)
            window = self.select(above, put("Add", (
                window, self.scalar_constant(float(1 << bit), magnitude),
            ), magnitude), window)
        chosen = [
            self.compare(
                "Eq", window, self.scalar_constant(float(index), magnitude),
            )
            for index in range(1, len(windows))
        ]
        terms = []
        for position in range(len(windows[0])):
            limb = self.scalar_constant(windows[0][position], magnitude)
            for index, predicate in enumerate(chosen, start=1):
                limb = self.select(
                    predicate,
                    self.scalar_constant(windows[index][position], magnitude),
                    limb,
                )
            high, low = self.two_product(scaled, limb)
            terms.append(self._modulo(high, 4.0, centred=True))
            terms.append(self._modulo(low, 4.0, centred=True))
        carried = int(width) + 2
        turns = self.renormalise(terms, carried)
        quotient = put("Round", (turns[0],), turns[0])
        fraction = self.add_expansions(
            turns, [put("Neg", (quotient,), quotient)], carried,
        )
        half_pi = self.constant_expansion(
            constant_limbs("pi", carried, scale=Fraction(1, 2)), magnitude,
        )
        return (
            self.multiply_expansions(fraction, half_pi, width),
            self._modulo(quotient, 4.0, centred=False),
        )

    def circular_reduction(self, magnitude, width: int):
        """``(r, q)``: ``|x| = q*pi/2 + r (mod 2*pi)``, ``r`` in ``width`` limbs.

        ``magnitude`` is ``|x|``. Both reductions are spelled and the exact
        compare picks one, so the SSA stays one straight-line block.
        """

        moderate, moderate_quadrant = self._cody_waite_reduction(
            magnitude, width,
        )
        large, large_quadrant = self._payne_hanek_reduction(magnitude, width)
        beyond = self.compare("Ge", magnitude, self.scalar_constant(
            CIRCULAR_CODY_WAITE_LIMIT, magnitude,
        ))
        reduced = [
            self.select(beyond, wide, narrow)
            for wide, narrow in zip(large, moderate)
        ]
        return reduced, self.select(beyond, large_quadrant, moderate_quadrant)

    def sin_cos_expansion(self, magnitude, width: int):
        """``(sin(r), cos(r), q)`` for ``|x| = q*pi/2 + r``: both proof cores."""

        reduced, quadrant = self.circular_reduction(magnitude, width)
        return (
            self.proof_core("sin", reduced, width),
            self.proof_core("cos", reduced, width),
            quadrant,
        )

    def circular_double(self, name: str, source, width: int = 2):
        """Correctly rounded ``sin``/``cos``/``tan`` of one float64 value.

        Returns ``(value, domain_error)``. ``domain_error`` is the ``bool``
        "``source`` is +-inf": Python raises ``ValueError('math domain
        error')`` there, which a straight-line emitter cannot spell, so the
        lowering that places this must guard it; ``value`` is NaN in that
        lane. NaN in gives NaN out (``x - x`` keeps the payload), and
        ``sin``/``tan`` return a zero argument itself, so ``-0.0`` stays
        ``-0.0``. The result is the existing collapse of the limbs (their
        sum, high first), negated exactly where the symmetry says so.
        """

        if name not in ("sin", "cos", "tan"):
            raise ValueError(f"circular_double has no {name!r}")
        put = self.put
        magnitude = put("Abs", (source,), source)
        sine, cosine, quadrant = self.sin_cos_expansion(magnitude, width)
        if name == "cos":
            # cos(|x|) = sin(|x| + pi/2): the same table one quadrant on.
            quadrant = self._modulo(put("Add", (
                quadrant, self.scalar_constant(1.0, quadrant),
            ), quadrant), 4.0, centred=False)
        parity = self._modulo(quadrant, 2.0, centred=False)
        odd = self.compare(
            "Eq", parity, self.scalar_constant(1.0, quadrant),
        )
        if name == "tan":
            numerator = [
                self.select(odd, c, s) for s, c in zip(sine, cosine)
            ]
            denominator = [
                self.select(odd, s, c) for s, c in zip(sine, cosine)
            ]
            value = self.sum_expansion(
                self.divide_expansions(numerator, denominator, width)
            )
            negate = odd
        else:
            value = self.sum_expansion([
                self.select(odd, c, s) for s, c in zip(sine, cosine)
            ])
            negate = self.compare(
                "Ge", quadrant, self.scalar_constant(2.0, quadrant),
            )
        value = self.select(negate, put("Neg", (value,), value), value)
        zero = self.scalar_constant(0.0, source)
        if name != "cos":
            below = self.compare("Lt", source, zero)
            value = self.select(below, put("Neg", (value,), value), value)
            value = self.select(
                self.compare("Eq", source, zero), source, value,
            )
        largest = self.scalar_constant(_FLOAT64_MAX, source)
        finite = self.compare("Le", magnitude, largest)
        value = self.select(
            finite, value, put("Sub", (source, source), source),
        )
        return value, self.compare("Gt", magnitude, largest)

    # -- exact lane machinery for the full-range compositions ----------------
    #
    # Package S2c (atan, atan2, tanh, pow).  Every range decision below is
    # an IEEE-exact compare, ``Select``, ``Floor``/``Round``, ``Sqrt`` or a
    # multiply by a power-of-two constant; no host libm, no ``Pow``/``Log``
    # spelling.  A lane that is decided by a special value (nan, inf, a
    # signed zero, a saturated result) carries that IEEE value in its high
    # limb and ``-0.0`` in every lower limb: ``v + -0.0 == v`` for every v,
    # so the sum-of-limbs collapse keeps it, sign of zero included.

    def _predicate(self, op: str, args):
        """A ``bool`` value: comparisons and logical And/Or/Not of tests."""

        like = args[0]
        shape = tuple(
            (like.accounting or {}).get("precision_logical_shape")
            or tuple(like.shape or ())
        )
        result = SSAValue(
            self._mint(
                self.function, self._limb_kind, tuple(args),
                dtype="bool", shape=shape, stage=self._stage,
            ),
            dtype="bool", shape=shape, device=like.device,
        )
        self.emitted.append(Instr(op, list(args), result, attributes={
            PRECISION_SECTION_ATTRIBUTE: True,
            "lowered_from": "precision.lane_test",
        }))
        return result

    def _all(self, *tests):
        held = tests[0]
        for test in tests[1:]:
            held = self._predicate("And", (held, test))
        return held

    def _any(self, *tests):
        held = tests[0]
        for test in tests[1:]:
            held = self._predicate("Or", (held, test))
        return held

    def _not(self, test):
        return self._predicate("Not", (test,))

    def _select(self, test, when_true, when_false):
        return self.put("Select", (test, when_true, when_false), when_true)

    def _select_expansion(self, test, when_true, when_false):
        return [
            self._select(test, high, low)
            for high, low in zip(when_true, when_false)
        ]

    def _negated(self, expansion):
        return [self.put("Neg", (term,), term) for term in expansion]

    def _scaled(self, expansion, factor):
        """Multiply every limb by an exact power of two (no rounding while
        the limbs stay normal, which the callers' ranges guarantee)."""

        return [self.put("Mul", (term, factor), term) for term in expansion]

    def _special(self, value, width: int):
        """A lane whose answer is the IEEE value ``value`` itself."""

        tail = self.scalar_constant(-0.0, value)
        return [value, *([tail] * (int(width) - 1))]

    def _is_finite(self, value):
        largest = self.scalar_constant(_DOUBLE_MAX, value)
        return self._predicate(
            "Le", (self.put("Abs", (value,), value), largest),
        )

    def _is_nan(self, value):
        return self._predicate("Ne", (value, value))

    def _sign_bit(self, value):
        """IEEE signbit for a non-nan value: ``1/±0`` tells the zeros."""

        zero = self.scalar_constant(0.0, value)
        one = self.scalar_constant(1.0, value)
        return self._select_bool(
            self._predicate("Eq", (value, zero)),
            self._predicate(
                "Lt", (self.put("Div", (one, value), value), zero),
            ),
            self._predicate("Lt", (value, zero)),
        )

    def _select_bool(self, test, when_true, when_false):
        return self._any(
            self._all(test, when_true),
            self._all(self._not(test), when_false),
        )

    def _odd_integer(self, value):
        """``value`` is an odd integer (finite input; |v| >= 2**53 is even)."""

        integral = self._predicate(
            "Eq", (value, self.put("Floor", (value,), value)),
        )
        half = self.put(
            "Mul", (value, self.scalar_constant(0.5, value)), value,
        )
        return self._all(integral, self._predicate(
            "Ne", (half, self.put("Floor", (half,), half)),
        ))

    def _power_of_two(self, exponent, like):
        """``2**n`` for an integer-valued ``n`` in [-1022, 1023], exactly.

        Ten selected factors ``2**(±2**b)``; every partial product is a
        normal power of two, so no multiply rounds.
        """

        zero = self.scalar_constant(0.0, like)
        negative = self._predicate("Lt", (exponent, zero))
        remaining = self.put("Abs", (exponent,), exponent)
        result = self.scalar_constant(1.0, like)
        for bit in range(9, -1, -1):
            step = self.scalar_constant(float(2 ** bit), like)
            take = self._predicate("Ge", (remaining, step))
            remaining = self._select(
                take, self.put("Sub", (remaining, step), remaining),
                remaining,
            )
            factor = self._select(
                negative,
                self.scalar_constant(2.0 ** -(2 ** bit), like),
                self.scalar_constant(2.0 ** (2 ** bit), like),
            )
            result = self._select(
                take, self.put("Mul", (result, factor), result), result,
            )
        return result

    def _binade(self, value, companions=()):
        """``value = m * 2**e`` with ``m`` in [1, 2), for finite ``value > 0``.

        A compare ladder over power-of-two constants: every multiply is by
        an exact power of two, so ``m`` is exact and ``e`` is an exact
        integer-valued double.  ``companions`` (the lower limbs of an
        expansion led by ``value``) are scaled by the same factors.
        """

        like = value
        mantissa, others = value, list(companions)
        exponent = self.scalar_constant(0.0, like)

        def apply(test, power: int):
            nonlocal mantissa, others, exponent
            factor = self.scalar_constant(2.0 ** power, like)
            mantissa = self._select(
                test, self.put("Mul", (mantissa, factor), mantissa),
                mantissa,
            )
            others = [
                self._select(test, self.put("Mul", (term, factor), term), term)
                for term in others
            ]
            exponent = self._select(
                test,
                self.put(
                    "Sub",
                    (exponent, self.scalar_constant(float(power), like)),
                    exponent,
                ),
                exponent,
            )

        apply(self._predicate(
            "Lt", (mantissa, self.scalar_constant(2.0 ** -1022, like)),
        ), 64)
        for bit in range(9, -1, -1):
            step = 2 ** bit
            apply(self._predicate(
                "Ge", (mantissa, self.scalar_constant(2.0 ** step, like)),
            ), -step)
        for bit in range(9, -1, -1):
            step = 2 ** bit
            apply(self._predicate(
                "Lt",
                (mantissa, self.scalar_constant(2.0 ** (1 - step), like)),
            ), step)
        return mantissa, exponent, others

    def round_scaled(self, expansion, exponent, sticky: float = 0.0):
        """One correctly rounded double for ``(sum(expansion) + δ) * 2**K``.

        ``expansion`` is a renormalised expansion with a positive finite
        high limb; ``exponent`` is the integer-valued double ``K`` (any
        size); ``δ`` is an infinitesimal of sign ``sticky`` (the part of a
        value below every limb, e.g. ``atan(v) < v``) that only breaks an
        exact tie.  The value is normalised to ``m * 2**T`` with ``m`` in
        [1, 2); ``T > 1023`` is +inf, ``T < -1075`` is +0, and otherwise
        ``m`` is rounded by one IEEE multiply onto the result grid (the
        subnormal grid below 2**-1022), after which the exact residual
        against the high limb, plus the lower limbs, moves the result one
        grid step when the true value lies past the midpoint -- so a
        subnormal result is rounded once, not twice.
        """

        like = expansion[0]
        tail = expansion[1] if len(expansion) == 2 else (
            self.sum_expansion(expansion[1:]) if len(expansion) > 2
            else self.scalar_constant(0.0, like)
        )
        mantissa, binade, (low,) = self._binade(expansion[0], (tail,))
        total = self.put("Add", (exponent, binade), exponent)
        lowest = self.scalar_constant(-1075.0, like)
        highest = self.scalar_constant(1023.0, like)
        clamped = self._select(
            self._predicate("Lt", (total, lowest)), lowest,
            self._select(
                self._predicate("Gt", (total, highest)), highest, total,
            ),
        )
        normal_floor = self.scalar_constant(-1022.0, like)
        first = self._select(
            self._predicate("Lt", (clamped, normal_floor)),
            normal_floor, clamped,
        )
        second = self.put("Sub", (clamped, first), clamped)
        result = self.put("Mul", (
            self.put(
                "Mul", (mantissa, self._power_of_two(first, like)), mantissa,
            ),
            self._power_of_two(second, like),
        ), mantissa)
        back = self.put("Mul", (
            self.put("Mul", (
                result,
                self._power_of_two(self.put("Neg", (second,), second), like),
            ), result),
            self._power_of_two(self.put("Neg", (first,), first), like),
        ), result)
        residual = self.put("Sub", (mantissa, back), mantissa)
        grid = self.put("Mul", (
            self.scalar_constant(2.0 ** -52, like),
            self._power_of_two(self.put("Neg", (second,), second), like),
        ), mantissa)
        half = self.put(
            "Mul", (grid, self.scalar_constant(0.5, like)), grid,
        )
        zero = self.scalar_constant(0.0, like)
        above = self.put("Add", (
            self.put("Sub", (residual, half), residual), low,
        ), residual)
        below = self.put("Add", (
            self.put("Add", (residual, half), residual), low,
        ), residual)
        up = self._predicate("Gt", (above, zero))
        down = self._predicate("Lt", (below, zero))
        if sticky > 0:
            up = self._any(up, self._predicate("Eq", (above, zero)))
        elif sticky < 0:
            down = self._any(down, self._predicate("Eq", (below, zero)))
        step = self.put("Mul", (
            self.scalar_constant(2.0 ** -52, like),
            self._power_of_two(first, like),
        ), like)
        result = self._select(
            up, self.put("Add", (result, step), result),
            self._select(
                down, self.put("Sub", (result, step), result), result,
            ),
        )
        largest = self.scalar_constant(_DOUBLE_MAX, like)
        infinity = self.put("Mul", (largest, largest), largest)
        result = self._select(
            self._predicate("Gt", (total, highest)), infinity, result,
        )
        return self._select(
            self._predicate("Lt", (total, lowest)), zero, result,
        )

    def _pi_multiple(self, scale, like, width: int):
        from fractions import Fraction

        from ..common.tensors.signal_symbolic import constant_limbs

        return self.constant_expansion(
            constant_limbs(
                "pi", width, scale=Fraction(scale), element=self.element,
            ),
            like,
        )

    def _one(self, like, width: int):
        return self.constant_expansion(
            (1.0, *((0.0,) * (int(width) - 1))), like,
        )

    def _exp_reduced(self, source, width: int):
        """``exp(source) = core * 2**k`` for |source| <= ~760.

        The ln(2) reduction of ``exp_expansion`` (round to the nearest
        multiple, one-limb-wider ln(2) so ``k*ln2`` is exact to well below
        the result's last limb) and its ``exp`` proof core, with the scale
        left to the caller as the integer ``k`` -- the caller applies it
        exactly (``_scaled``) or with one rounding (``round_scaled``).
        """

        from ..common.tensors.signal_symbolic import constant_limbs

        put = self.put
        lead = source[0]
        ln2_head = self._ln2_head(lead)
        binade = put(
            "Round", (put("Div", (lead, ln2_head), lead),), lead,
        )
        ln2_parts = self.constant_expansion(
            constant_limbs("ln2", int(width) + 1, element=self.element),
            lead,
        )
        remainder = self.add_expansions(
            source,
            self._negated(self.multiply_expansions(
                ln2_parts, [binade], int(width) + 1,
            )),
            width,
        )
        return self.proof_core("exp", remainder, width), binade

    def _log_positive(self, value, width: int):
        """``log(value)`` for a finite double ``value > 0``, relatively exact.

        ``value = m * 2**e`` with ``m`` in [sqrt(1/2), sqrt(2)) by the exact
        binade ladder, then ``log(m) = 2*log1p(u)`` with
        ``u = (m - 1)/(sqrt(m) + 1)`` -- ``m - 1`` is exact (Sterbenz), so
        ``u`` keeps its relative accuracy as ``m`` approaches 1 and nothing
        cancels.  ``|u| <= 0.19`` sits inside the ``log1p`` proof radius.
        (``log_expansion``'s binade law ``Floor(Log(x)/ln2)`` puts x just
        below 1 at binade -1 and cancels ``4*log1p(...)`` against ``ln2``,
        which pow's ``y*log(x)`` cannot afford.)
        """

        from ..common.tensors.signal_symbolic import constant_limbs

        put = self.put
        mantissa, binade, _ = self._binade(value)
        fold = self._predicate(
            "Ge", (mantissa, self.scalar_constant(1.4142135623730951, value)),
        )
        mantissa = self._select(fold, put("Mul", (
            mantissa, self.scalar_constant(0.5, value),
        ), mantissa), mantissa)
        binade = self._select(fold, put("Add", (
            binade, self.scalar_constant(1.0, value),
        ), binade), binade)
        one = self._one(value, width)
        root = self.sqrt_expansion(self.expanded(mantissa, width), width)
        offset = self.divide_expansions(
            [put("Sub", (mantissa, one[0]), mantissa)],
            self.add_expansions(root, one, width),
            width,
        )
        core = self.proof_core("log1p", offset, width)
        two = self.scalar_constant(2.0, value)
        core = [put("Mul", (term, two), term) for term in core]
        ln2_parts = self.constant_expansion(
            constant_limbs("ln2", int(width) + 1, element=self.element),
            value,
        )
        return self.add_expansions(
            core,
            self.multiply_expansions(ln2_parts, [binade], int(width) + 1),
            width,
        )

    # -- atan, atan2, tanh ----------------------------------------------------

    def _atan_unit(self, folded, width: int):
        """``atan(v)`` for an expansion ``v`` in [0, 1].

        Above the proof radius (tan(pi/8)) the identity
        ``atan(v) = pi/4 + atan((v-1)/(v+1))`` folds ``v`` back inside it
        (no table).  The decision reads the high limb against the radius
        itself, so both sides land on the proven interval.
        """

        from ..common.tensors import signal_symbolic as proof

        like = folded[0]
        radius = self.scalar_constant(proof.CORE_RADII["atan"], like)
        middle = self._predicate("Gt", (folded[0], radius))
        one = self._one(like, width)
        shifted = self.divide_expansions(
            self.add_expansions(folded, self._negated(one), width),
            self.add_expansions(folded, one, width),
            width,
        )
        core = self.proof_core(
            "atan", self._select_expansion(middle, shifted, folded), width,
        )
        return self._select_expansion(
            middle,
            self.add_expansions(
                core, self._pi_multiple("1/4", like, width), width,
            ),
            core,
        )

    def atan_expansion(self, value, width: int = 2):
        """Correctly rounded ``math.atan`` of the double ``value``, as limbs.

        ``|x| > 1`` reflects through ``atan(x) = pi/2 - atan(1/x)``
        (``1/x`` as an expansion quotient), then ``_atan_unit``; the sign
        is restored by negating limbs (atan is odd).  nan, ±inf and ±0 are
        decided lanes.
        """

        like = value
        magnitude = self.put("Abs", (value,), value)
        one = self.scalar_constant(1.0, like)
        finite = self._is_finite(value)
        safe = self._select(finite, magnitude, one)
        large = self._predicate("Gt", (safe, one))
        inverse = self.divide_expansions(
            self._one(like, width),
            self.expanded(self._select(large, safe, one), width),
            width,
        )
        folded = self._select_expansion(
            large, inverse, self.expanded(safe, width),
        )
        core = self._atan_unit(folded, width)
        half_pi = self._pi_multiple("1/2", like, width)
        result = self._select_expansion(
            large,
            self.add_expansions(half_pi, self._negated(core), width),
            core,
        )
        zero = self.scalar_constant(0.0, like)
        result = self._select_expansion(
            self._predicate("Lt", (value, zero)), self._negated(result),
            result,
        )
        # ±inf -> ±pi/2 (the half-pi limbs, signed); nan and ±0 -> value.
        signed_half_pi = self._select_expansion(
            self._predicate("Lt", (value, zero)),
            self._negated(half_pi), half_pi,
        )
        result = self._select_expansion(
            self._any(self._is_nan(value), self._predicate("Eq", (value, zero))),
            self._special(value, width),
            self._select_expansion(finite, result, signed_half_pi),
        )
        return result

    def atan2_expansion(self, y, x, width: int = 2):
        """Correctly rounded ``math.atan2(y, x)``, as limbs.

        The ratio is formed as ``min/max`` of the magnitudes from their
        exact binade mantissas (``q = m_num/m_den`` in (1/2, 2), scale
        ``2**(e_num - e_den)``), so it neither overflows nor loses limbs to
        the subnormal range; ``_atan_unit`` takes it, ``pi/2 - ...`` undoes
        the swap and ``pi - ...`` the negative ``x`` (signbit, so -0.0
        counts).  A ratio below ``2**-600`` with ``x > 0`` and no swap is
        the result itself: ``atan(v) = v - v**3/3`` lies just under ``v``,
        so it is rounded once by ``round_scaled`` with a negative sticky
        (a subnormal quotient exactly on a midpoint rounds down, not to
        even).  C99/Python special cases are decided lanes.
        """

        like = y
        zero = self.scalar_constant(0.0, like)
        one = self.scalar_constant(1.0, like)
        y_magnitude = self.put("Abs", (y,), y)
        x_magnitude = self.put("Abs", (x,), x)
        y_finite, x_finite = self._is_finite(y), self._is_finite(x)
        y_zero = self._predicate("Eq", (y, zero))
        x_zero = self._predicate("Eq", (x, zero))
        y_safe = self._select(
            self._all(y_finite, self._not(y_zero)), y_magnitude, one,
        )
        x_safe = self._select(
            self._all(x_finite, self._not(x_zero)), x_magnitude, one,
        )
        swap = self._predicate("Gt", (y_safe, x_safe))
        numerator = self._select(swap, x_safe, y_safe)
        denominator = self._select(swap, y_safe, x_safe)
        top, top_binade, _ = self._binade(numerator)
        bottom, bottom_binade, _ = self._binade(denominator)
        quotient = self.divide_expansions(
            self.expanded(top, width), self.expanded(bottom, width), width,
        )
        shift = self.put("Sub", (top_binade, bottom_binade), top_binade)
        tiny = self._predicate(
            "Lt", (shift, self.scalar_constant(-600.0, like)),
        )
        ratio = self._scaled(
            quotient,
            self._power_of_two(self._select(tiny, zero, shift), like),
        )
        ratio = self._select_expansion(
            tiny, self.expanded(zero, width), ratio,
        )
        angle = self._atan_unit(ratio, width)
        half_pi = self._pi_multiple("1/2", like, width)
        pi = self._pi_multiple(1, like, width)
        angle = self._select_expansion(
            swap, self.add_expansions(half_pi, self._negated(angle), width),
            angle,
        )
        x_negative = self._sign_bit(x)
        angle = self._select_expansion(
            x_negative, self.add_expansions(pi, self._negated(angle), width),
            angle,
        )
        # Special lanes, lowest priority first (the later select wins).
        quarter_pi = self._pi_multiple("1/4", like, width)
        three_quarter_pi = self._pi_multiple("3/4", like, width)
        y_infinite = self._not(self._any(y_finite, self._is_nan(y)))
        x_infinite = self._not(self._any(x_finite, self._is_nan(x)))
        angle = self._select_expansion(
            self._all(x_zero, self._not(y_zero)), half_pi, angle,
        )
        angle = self._select_expansion(
            x_infinite,
            self._select_expansion(
                x_negative, pi, self.expanded(zero, width),
            ),
            angle,
        )
        angle = self._select_expansion(
            y_infinite,
            self._select_expansion(
                x_infinite,
                self._select_expansion(
                    x_negative, three_quarter_pi, quarter_pi,
                ),
                half_pi,
            ),
            angle,
        )
        y_negative = self._sign_bit(y)
        angle = self._select_expansion(
            y_negative, self._negated(angle), angle,
        )
        # The tiny direct lane is signed before it becomes a special lane:
        # negating ``[v, -0.0]`` would turn the tail into +0.0 and lose -0.
        direct = self.round_scaled(quotient, shift, sticky=-1.0)
        direct = self._select(
            y_negative, self.put("Neg", (direct,), direct), direct,
        )
        angle = self._select_expansion(
            self._all(
                tiny, self._not(swap), self._not(x_negative),
                y_finite, x_finite, self._not(y_zero), self._not(x_zero),
            ),
            self._special(direct, width), angle,
        )
        # y = ±0 (x not nan): ±0 for x >= +0, ±pi for x <= -0.
        signed_pi = self._select_expansion(
            y_negative, self._negated(pi), pi,
        )
        angle = self._select_expansion(
            y_zero,
            self._select_expansion(
                x_negative, signed_pi, self._special(y, width),
            ),
            angle,
        )
        either_nan = self._any(self._is_nan(y), self._is_nan(x))
        return self._select_expansion(
            either_nan,
            self._special(self.put("Add", (y, x), y), width),
            angle,
        )

    def tanh_expansion(self, value, width: int = 2):
        """Correctly rounded ``math.tanh`` of the double ``value``, as limbs.

        |x| <= 1/2 is the ``tanh`` proof core on x itself (odd series, so
        tiny x keeps its relative accuracy -- the expm1-free route, nothing
        cancels).  1/2 < |x| <= 22 is ``1 - 2/(exp(2|x|) + 1)`` with
        ``exp`` reduced exactly (``2|x|`` is exact, the 2**k scale is an
        exact power-of-two multiply); there tanh >= 0.46, so the
        subtraction cannot cancel.  Beyond 22, 1 - tanh < 2**-62 and the
        result is ±1 (inf included).  nan and ±0 are the value itself.
        """

        like = value
        zero = self.scalar_constant(0.0, like)
        half = self.scalar_constant(0.5, like)
        saturation = self.scalar_constant(22.0, like)
        magnitude = self.put("Abs", (value,), value)
        small = self._predicate("Le", (magnitude, half))
        saturated = self._predicate("Gt", (magnitude, saturation))
        core = self.proof_core(
            "tanh", self.expanded(self._select(small, value, zero), width),
            width,
        )
        doubled = self.put("Add", (magnitude, magnitude), magnitude)
        middle_argument = self._select(
            self._any(small, saturated, self._is_nan(value)),
            self.scalar_constant(1.0, like), doubled,
        )
        scaled_core, binade = self._exp_reduced(
            self.expanded(middle_argument, width), width,
        )
        grown = self._scaled(scaled_core, self._power_of_two(binade, like))
        one = self._one(like, width)
        two = self.constant_expansion(
            (2.0, *((0.0,) * (int(width) - 1))), like,
        )
        middle = self.add_expansions(
            one,
            self._negated(self.divide_expansions(
                two, self.add_expansions(grown, one, width), width,
            )),
            width,
        )
        unit = self.expanded(self.scalar_constant(1.0, like), width)
        positive = self._select_expansion(saturated, unit, middle)
        positive = self._select_expansion(
            self._predicate("Lt", (value, zero)),
            self._negated(positive), positive,
        )
        result = self._select_expansion(small, core, positive)
        return self._select_expansion(
            self._any(self._is_nan(value), self._predicate("Eq", (value, zero))),
            self._special(value, width), result,
        )

    # -- pow ------------------------------------------------------------------

    def _exact_power(self, base, exponent, width: int):
        """``(m, n, e)`` with ``base**exponent == m**n * 2**(e*n)`` exactly
        when ``exponent * 2**j`` is an integer ``n`` for some j <= 5 and
        ``base`` is a perfect ``2**j``-th power (the smallest such j).

        ``base`` is a finite double > 0.  Each level takes ``Sqrt`` of a
        mantissa in [1, 4) (the exponent made even first), so the root is a
        normal double in [1, 2) and its exactness is an exact test: the
        two-product of the root with itself is the mantissa with a zero
        residual.  Returns ``(mantissa, exponent_n, binade, usable)``.
        """

        like = base
        two = self.scalar_constant(2.0, like)
        half = self.scalar_constant(0.5, like)
        mantissa, binade, _ = self._binade(base)
        exact = self._predicate("Eq", (two, two))
        scaled = exponent
        levels = []
        for level in range(6):
            if level:
                halved = self.put("Mul", (binade, half), binade)
                odd = self._predicate(
                    "Ne", (halved, self.put("Floor", (halved,), halved)),
                )
                widened = self._select(
                    odd, self.put("Mul", (mantissa, two), mantissa), mantissa,
                )
                binade = self.put("Floor", (halved,), halved)
                root = self.put("Sqrt", (widened,), widened)
                square, residual = self.two_product(root, root)
                exact = self._all(
                    exact,
                    self._predicate("Eq", (square, widened)),
                    self._predicate(
                        "Eq", (residual, self.scalar_constant(0.0, like)),
                    ),
                )
                mantissa = root
                scaled = self.put("Mul", (scaled, two), scaled)
            integral = self._predicate(
                "Eq", (scaled, self.put("Floor", (scaled,), scaled)),
            )
            levels.append((integral, mantissa, binade, exact, scaled))
        found, mantissa, binade, exact, count = levels[-1]
        for integral, level_mantissa, level_binade, level_exact, level_count in (
            reversed(levels[:-1])
        ):
            mantissa = self._select(integral, level_mantissa, mantissa)
            binade = self._select(integral, level_binade, binade)
            exact = self._select_bool(integral, level_exact, exact)
            count = self._select(integral, level_count, count)
            found = self._any(found, integral)
        one = self.scalar_constant(1.0, like)
        in_range = self._all(
            self._predicate("Ge", (count, one)),
            self._predicate(
                "Le", (count, self.scalar_constant(64.0, like)),
            ),
        )
        usable = self._all(found, exact, self._any(
            in_range, self._predicate("Eq", (mantissa, one)),
        ))
        return mantissa, count, binade, usable

    def pow_expansion(self, x, y, width: int = 2, *, form: str = "operator"):
        """Correctly rounded float ``x ** y`` (``form="operator"``) or
        ``math.pow(x, y)`` (``form="math"``), as limbs, plus its faults.

        Returns ``(limbs, faults)``; ``faults`` is a tuple of
        ``(predicate, native_faults code, failure name)`` rows for the
        lowering pass to guard (a validation-fail block calling
        ``turing_validation_error``, as ``reduce_integer_pow`` does).  The
        rows are: ``0.0 ** negative`` (ZeroDivisionError for ``**``,
        ValueError for ``math.pow``), negative ``**`` non-integer (complex
        in Python: refused by name as ``FLOAT_POW_COMPLEX_RESULT`` for
        ``**``, ValueError for ``math.pow``) and overflow (OverflowError).

        The general lane is ``exp(y*log|x|)``: ``_log_positive`` (exact
        binade, relatively exact near 1), the product by ``y`` in limbs,
        ``_exp_reduced`` and one rounding by ``round_scaled`` (overflow to
        inf and subnormal results rounded once).  A result that may be
        exactly representable or exactly a rounding midpoint
        (``3.0**34``, ``(m*m)**1.5``, ``2.0**-1075``) is a dyadic power: when
        ``_exact_power`` finds ``y*2**j = n`` with ``|x|`` a perfect
        ``2**j``-th power and ``1 <= n <= 64`` (or a power-of-two mantissa),
        ``m**n`` is formed by square-and-multiply in limbs -- exact for every
        such result that fits a double -- and rounded by ``round_scaled``
        with scale ``2**(e*n)``, so ties go to even exactly as Python's.
        Python's special cases (nan, inf, signed zeros, odd integers) are
        decided lanes in Python's own order.
        """

        from .native_faults import (
            FLOAT_MATH_DOMAIN, FLOAT_MATH_RANGE, FLOAT_POW_COMPLEX_RESULT,
            FLOAT_POW_OVERFLOW, FLOAT_POW_ZERO_NEGATIVE,
        )

        if form not in ("operator", "math"):
            raise ValueError(
                f"pow form {form!r}: expected 'operator' (x ** y) or "
                "'math' (math.pow)"
            )
        like = x
        put = self.put
        zero = self.scalar_constant(0.0, like)
        one = self.scalar_constant(1.0, like)
        largest = self.scalar_constant(_DOUBLE_MAX, like)
        infinity = put("Mul", (largest, largest), largest)
        not_a_number = put("Mul", (zero, infinity), zero)
        x_nan, y_nan = self._is_nan(x), self._is_nan(y)
        x_finite, y_finite = self._is_finite(x), self._is_finite(y)
        x_zero = self._predicate("Eq", (x, zero))
        y_zero = self._predicate("Eq", (y, zero))
        x_negative = self._predicate("Lt", (x, zero))
        y_negative = self._predicate("Lt", (y, zero))
        y_positive = self._predicate("Gt", (y, zero))
        y_safe = self._select(y_finite, y, one)
        y_odd = self._all(y_finite, self._odd_integer(y_safe))
        y_integral = self._predicate(
            "Eq", (y_safe, put("Floor", (y_safe,), y_safe)),
        )
        magnitude = put("Abs", (x,), x)
        general = self._all(
            x_finite, y_finite, self._not(x_zero), self._not(y_zero),
        )
        base = self._select(general, magnitude, one)
        exponent = self._select(general, y, one)
        # Approximation lane: exp(y * log|x|).
        logarithm = self._log_positive(base, width)
        product = self.multiply_expansions(logarithm, [exponent], width)
        # The range decision reads the one-double product: it is ±inf, never
        # nan, where the limb product overflows (|y| near the largest double).
        estimate = put("Mul", (logarithm[0], exponent), exponent)
        over = self._predicate(
            "Gt", (estimate, self.scalar_constant(720.0, like)),
        )
        under = self._predicate(
            "Lt", (estimate, self.scalar_constant(-760.0, like)),
        )
        product = self._select_expansion(
            self._any(over, under), self.expanded(zero, width), product,
        )
        grown, binade = self._exp_reduced(product, width)
        # Exact-power lane: m**n * 2**(e*n).
        mantissa, count, level_binade, usable = self._exact_power(
            base, exponent, width,
        )
        steps = self._select(usable, count, one)
        steps = self._select(
            self._predicate("Eq", (mantissa, one)), one, steps,
        )
        power = self._one(like, width)
        for bit in range(6, -1, -1):
            power = self.multiply_expansions(power, power, width)
            step = self.scalar_constant(float(2 ** bit), like)
            take = self._predicate("Ge", (steps, step))
            steps = self._select(take, put("Sub", (steps, step), steps), steps)
            power = self._select_expansion(
                take, self.multiply_expansions(power, [mantissa], width),
                power,
            )
        scale = self._select(
            usable, put("Mul", (level_binade, count), level_binade), binade,
        )
        rounded = self.round_scaled(
            self._select_expansion(usable, power, grown), scale,
        )
        rounded = self._select(
            usable, rounded,
            self._select(over, infinity, self._select(under, zero, rounded)),
        )
        negate = self._all(x_negative, y_odd)
        value = self._select(negate, put("Neg", (rounded,), rounded), rounded)
        # Python's float_pow order; the later select wins.
        negative_fraction = self._all(
            x_finite, x_negative, y_finite, self._not(y_integral),
        )
        value = self._select(negative_fraction, not_a_number, value)
        reciprocal = put("Div", (one, x), x)
        zero_value = self._select(
            y_negative,
            self._select(y_odd, reciprocal, infinity),
            self._select(y_odd, x, zero),
        )
        value = self._select(
            self._all(x_zero, y_finite, self._not(y_zero)), zero_value, value,
        )
        negative_zero = put("Neg", (zero,), zero)
        infinite_value = self._select(
            y_positive,
            self._select(y_odd, x, magnitude),
            self._select(
                y_odd,
                self._select(x_negative, negative_zero, zero),
                zero,
            ),
        )
        x_infinite = self._not(self._any(x_finite, x_nan))
        y_infinite = self._not(self._any(y_finite, y_nan))
        value = self._select(
            self._all(x_infinite, y_finite, self._not(y_zero)),
            infinite_value, value,
        )
        unit = self._predicate("Eq", (magnitude, one))
        beyond = self._select_bool(
            y_positive,
            self._predicate("Gt", (magnitude, one)),
            self._predicate("Lt", (magnitude, one)),
        )
        value = self._select(
            y_infinite,
            self._select(
                unit, one,
                self._select(beyond, put("Abs", (y,), y), zero),
            ),
            value,
        )
        value = self._select(
            y_nan, self._select(self._predicate("Eq", (x, one)), one, y),
            value,
        )
        value = self._select(x_nan, x, value)
        value = self._select(y_zero, one, value)
        operator = form == "operator"
        faults = (
            (
                self._all(x_zero, y_finite, y_negative),
                FLOAT_POW_ZERO_NEGATIVE if operator else FLOAT_MATH_DOMAIN,
                "float_pow_zero_to_negative_power",
            ),
            (
                negative_fraction,
                FLOAT_POW_COMPLEX_RESULT if operator else FLOAT_MATH_DOMAIN,
                "float_pow_negative_base_fractional_exponent",
            ),
            (
                self._all(
                    general, self._not(negative_fraction),
                    self._predicate(
                        "Gt", (put("Abs", (rounded,), rounded), largest),
                    ),
                ),
                FLOAT_POW_OVERFLOW if operator else FLOAT_MATH_RANGE,
                "float_pow_overflow",
            ),
        )
        return self._special(value, width), faults


#: Below this ``|x|`` the circular reduction is Cody-Waite (the split
#: ``cody_waite_for`` derives for it); at and above it, the windowed 2/pi
#: reduction. The split's own precision falls as its reach grows (each
#: piece keeps ``53 - log2(k) - guard`` bits), so the limit stays where
#: ``k * (pi/2)`` carries ~150 bits; the windowed reduction has no such
#: limit and covers everything above. (trig cores, package S2b)
CIRCULAR_CODY_WAITE_LIMIT = 2.0 ** 10

#: Exponent step between the 2/pi windows: ``|x| = m * 2**(24*j)``.
CIRCULAR_PAYNE_HANEK_STEP = 24

#: Double limbs per window of ``(2**(24*j) * 2/pi) mod 2**54``.
CIRCULAR_PAYNE_HANEK_LIMBS = 6

#: The largest finite binary64, spelled exactly (``(2 - 2**-52) * 2**1023``).
_FLOAT64_MAX = (2.0 - 2.0 ** -52) * 2.0 ** 1023


@_functools.lru_cache(maxsize=1)
def circular_payne_hanek_windows() -> tuple:
    """``(2**(24*j) * 2/pi) mod 2**54`` as exact double limbs, per window.

    One row per ``j`` with ``24*j <= 1023``. 2/pi is the exact rational
    ``signal_symbolic.constant_rational('pi')`` gives, to enough digits that
    the deepest window's last limb is still exact; each row is cut by
    nearest-double heads with the remainder carried, as
    ``limb_decomposition`` cuts any rational.
    """

    from fractions import Fraction

    from ..common.tensors.signal_symbolic import constant_rational

    step = CIRCULAR_PAYNE_HANEK_STEP
    count = 1023 // step + 1
    # Bits wanted: the deepest window's shift, the 54 kept above the
    # point, the limbs' own ~53 bits each, and a margin.
    bits = step * (count - 1) + 54 + 53 * CIRCULAR_PAYNE_HANEK_LIMBS + 64
    two_over_pi = Fraction(2) / constant_rational(
        "pi", int(bits * 0.30103) + 20,
    )
    modulus = 1 << 54
    rows = []
    for index in range(count):
        window = two_over_pi * (1 << (step * index))
        rest = window - modulus * (window.numerator // (
            window.denominator * modulus
        ))
        row = []
        for _ in range(CIRCULAR_PAYNE_HANEK_LIMBS):
            head = float(rest)
            row.append(head)
            rest -= Fraction(head)
        rows.append(tuple(row))
    return tuple(rows)


def lower_precision_operations(
    functions, two_product_flavor: str = "fma",
) -> dict:
    """Expand every `precision_*` into arithmetic a destination already has.

    A precision value becomes ONE SSA VALUE PER LIMB, not one value with a
    limb axis. That is the representation the working kernels use --
    `two_product(a, b, p, e, n)` writes its two limbs to two arrays -- and it
    is the one that survives, because each limb is then an ordinary scalar
    that every backend can already hold, load and store. The channel-shaped
    alternative needs a destination to understand an aggregate before it can
    do arithmetic on one.

    After this runs no `precision_*` remains, which is the point: the
    operators exist to be recognised and reduced against, and their identity
    is not meant to reach a backend. The section attribute is stamped on
    every instruction produced here so the boundary survives where the
    operator does not.

    Expansions are the same width-N algorithms as the eager tensor path:
    Knuth ``two_sum`` distillation for addition, every limb-pair through an
    error-free ``two_product`` for multiplication, and digit-at-a-time
    expansion division. Width is an authored integer rather than a fixed
    catalogue tier; the recommended ladders remain policy guidance because
    multiplication and renormalisation grow quadratically.
    Returns a count per operation.

    ``two_product_flavor`` selects the exact-product spelling (see
    ``TWO_PRODUCT_FLAVORS``): ``"fma"`` emits the two-instruction residual,
    ``"split"`` emits Dekker's fma-free seventeen-operation form for
    destinations that cannot deliver a single rounding.
    """

    from ..common.tensors.topological_reducer import (
        PRECISION_PACK_NAME,
        PRECISION_SINGULAR_NAMES,
    )

    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )

    add_name = PRECISION_SINGULAR_NAMES["Add"]
    sub_name = PRECISION_SINGULAR_NAMES["Sub"]
    mul_name = PRECISION_SINGULAR_NAMES["Mul"]
    div_name = PRECISION_SINGULAR_NAMES["Div"]
    sqrt_name = PRECISION_SINGULAR_NAMES["Sqrt"]
    exp_name = PRECISION_SINGULAR_NAMES["Exp"]
    log_name = PRECISION_SINGULAR_NAMES["Log"]
    neg_name = PRECISION_SINGULAR_NAMES["neg"]
    expandable = {
        add_name, sub_name, mul_name, div_name, sqrt_name, exp_name,
        log_name, neg_name, PRECISION_PACK_NAME,
    }

    counts = {name: 0 for name in expandable}
    original_formals = {
        str(name): tuple(function.args) for name, function in functions.items()
    }
    from .concordance_declarations import (
        PRECISION_CHANNEL_SHAPE, PRECISION_CONSTANT,
        PRECISION_DECLARED_FORMAL_WIDTH, PRECISION_EXPANSION,
        PRECISION_FORMAL_LIMB, PRECISION_LIMB, SOURCE_PRECISION_BOUNDARY,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import identity_cells, mint_ssa_value

    book = current_identity_book()
    promotion_page = book.page(SOURCE_PRECISION_BOUNDARY)
    channel_shape_page = book.page(PRECISION_CHANNEL_SHAPE)
    promoted_scalar_formals: dict[str, set[int]] = {}
    for name, function in functions.items():
        metadata = dict(getattr(function, "metadata", {}) or {})
        scopes = {
            metadata.get("authored_function_identity"),
            metadata.get("source_numeric_scope"),
        } - {None}
        promoted_scalar_formals[str(name)] = {
            int(value_id)
            for row in promotion_page.rows()
            for fact in (promotion_page.latest(row),)
            if isinstance(row, tuple)
            and len(row) == 2
            and row[0] in scopes
            and isinstance(fact, tuple)
            and len(fact) == 3
            and str(fact[0]) == "promote"
            for value_id in (row[1], fact[1])
        }

        # Region calls retain the ProcessGraph feed identity separately from
        # the settled SSA actual.  Structural views can therefore spell the
        # same promoted tensor as (for example) feed ``flat`` / actual
        # ``points``.  Carry the concorded promotion through that exact
        # positional receipt before deciding which function formals need
        # physical limb parameters; a promoted scalar's low limbs are zeros,
        # never extra ABI inputs.
        promoted = promoted_scalar_formals[str(name)]
        aliases = metadata.get("value_aliases", ()) or ()
        alias_pairs = (
            aliases.items() if isinstance(aliases, Mapping) else aliases
        )
        changed_promotions = True
        while changed_promotions:
            changed_promotions = False
            for alias, owner in alias_pairs:
                alias, owner = int(alias), int(owner)
                if alias in promoted or owner in promoted:
                    before = len(promoted)
                    promoted.update((alias, owner))
                    changed_promotions |= len(promoted) != before
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if str(instruction.op) not in {"Call", "call"}:
                        continue
                    feed_ids = tuple(map(int,
                        instruction.attributes.get("feed_ids") or ()
                    ))
                    for feed_id, actual in zip(
                        feed_ids, instruction.args, strict=False,
                    ):
                        if feed_id not in promoted:
                            continue
                        actual_id = int(actual.id)
                        if actual_id not in promoted:
                            promoted.add(actual_id)
                            changed_promotions = True

    # -- the boundary, brought into line with the interior ----------------
    #
    # A precision value is n scalars everywhere inside a function, but a
    # declared parameter still arrived as ONE. The two representations
    # disagreeing at the boundary is what let a four-limb coefficient be
    # undeliverable while the arithmetic that would consume it already
    # worked, so a parameter used at width n becomes n formals: the
    # declared one carries the high limb and the rest are appended.
    #
    # Appending rather than interleaving keeps every existing position
    # valid, so a caller that does not know about limbs is unaffected.
    #
    # Width is read from the operations rather than from a declaration.
    # The declaration is gone by now -- it lives in the AST -- but an
    # operation records the width it was recognised at, and a parameter
    # feeding a width-n operation is a width-n parameter. That also keeps
    # this honest about parameters that are declared and never used at
    # precision: they stay one scalar, because nothing needs more.
    # -- precision arrays: channel-strided limbs ---------------------------
    #
    # An ARRAY of precision values is one ordinary array whose element i,
    # limb k lives at flat index i * limbs + k. One SSA value, so it can be
    # passed and RETURNED like any other array -- which is what the scalar
    # representation cannot do, and why every precision result until now had
    # to collapse into a single double on the way out.
    #
    # The arithmetic is unchanged: limbs are loaded into scalars, operated on
    # per limb, and stored back. Only the accessors widen. That keeps the one
    # representation every backend can already hold and avoids asking any of
    # them to understand an aggregate.
    #
    # An array is recognised by USE, not by shape -- inside a region a passed
    # array has shape () and is told apart only by being the base of a
    # GetElementPtr. A base whose Load feeds a precision operation is a
    # precision array; so is one whose Store carries a limbed value.
    precision_arrays: dict[str, dict[int, int]] = {}
    for name, function in functions.items():
        bases: dict[int, int] = {}
        for block in function.blocks.values():
            pointers: dict[int, int] = {}
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "GetElementPtr"
                    and instruction.args and instruction.res is not None
                ):
                    pointers[int(instruction.res.id)] = int(
                        instruction.args[0].id
                    )
            loaded: dict[int, int] = {}
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "Load" and instruction.args
                    and instruction.res is not None
                    and int(instruction.args[0].id) in pointers
                ):
                    loaded[int(instruction.res.id)] = pointers[
                        int(instruction.args[0].id)
                    ]
            produced: dict[int, int] = {}
            for instruction in block.instrs:
                width = max(
                    int(instruction.attributes.get("precision_limbs") or 1), 1
                )
                if str(instruction.op) in expandable and width > 1:
                    for argument in instruction.args:
                        base = loaded.get(int(argument.id))
                        if base is not None:
                            bases[base] = max(bases.get(base, 1), width)
                    if instruction.res is not None:
                        produced[int(instruction.res.id)] = width
            # The destination side. An array that is only ever WRITTEN --
            # every output array is -- has no Load to be recognised by, and
            # would otherwise collapse its limbs on the way out, which is the
            # single thing that has capped every precision result so far.
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "Store"
                    and len(instruction.args) >= 2
                    and int(instruction.args[0].id) in produced
                    and int(instruction.args[1].id) in pointers
                ):
                    base = pointers[int(instruction.args[1].id)]
                    bases[base] = max(
                        bases.get(base, 1),
                        produced[int(instruction.args[0].id)],
                    )
        if bases:
            precision_arrays[str(name)] = bases

    # -- declared precision arrays, across the planned-region seam ---------
    #
    # Recognition by use stops at a call. The planner puts a pack loop's
    # ``z = x[i]`` and ``y[i] = ...`` in regions of their own, so the load
    # of x feeds only a Ret and the store into y stores a formal: neither
    # region holds a precision operation, and both indexed element i as one
    # double -- x[i] read element i/2's limbs as two elements, and y[i]
    # kept only the leading limb. The parameter's declaration says what
    # the array IS: a ``Precision[k]`` array parameter of the authored
    # function is limb-strided at width k wherever it is addressed, and a
    # region formal bound to it by position is the same array.
    declared_arrays: dict[str, dict[int, int]] = {}
    for name, function in functions.items():
        for row in (
            function.metadata.get("numeric_parameter_descriptors", ()) or ()
        ):
            if not isinstance(row, Mapping):
                continue
            descriptor = dict(row.get("descriptor") or {})
            limbs_declared = int(descriptor.get("limbs") or 1)
            if (
                limbs_declared > 1
                and "precision" in tuple(descriptor.get("features") or ())
                and row.get("value_id") is not None
            ):
                declared_arrays.setdefault(str(name), {})[
                    int(row["value_id"])
                ] = limbs_declared
    changed = True
    while changed:
        changed = False
        for name, function in functions.items():
            known = declared_arrays.get(str(name)) or {}
            if not known:
                continue
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in ("Call", "call"):
                        continue
                    callee = str(instruction.attributes.get("callee") or "")
                    formals = original_formals.get(callee)
                    if not formals:
                        continue
                    there = declared_arrays.setdefault(callee, {})
                    for formal, actual in zip(formals, instruction.args):
                        width = known.get(int(actual.id), 1)
                        if width > there.get(int(formal.id), 1):
                            there[int(formal.id)] = width
                            changed = True
    for name, function in functions.items():
        known = declared_arrays.get(str(name)) or {}
        if not known:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    str(instruction.op) == "GetElementPtr" and instruction.args
                    and int(instruction.args[0].id) in known
                    and int(instruction.args[0].id)
                    not in promoted_scalar_formals[str(name)]
                ):
                    base = int(instruction.args[0].id)
                    bases = precision_arrays.setdefault(str(name), {})
                    bases[base] = max(bases.get(base, 1), known[base])

    parameter_widths: dict[str, dict[int, int]] = {}
    for name, function in functions.items():
        formal_ids = {int(value.id) for value in function.args}
        # A formal whose declared type is a Precision value of some width
        # (for example a numeral's coefficient leaf) is that many limbs,
        # whether or not a Precision operation in this body touches it.
        declared_width_page = book.page(PRECISION_DECLARED_FORMAL_WIDTH)
        widths: dict[int, int] = {}
        for value in function.args:
            declared = int((value.accounting or {}).get("precision_limbs") or 1)
            if (
                declared <= 1
                or int(value.id) in promoted_scalar_formals[str(name)]
            ):
                continue
            row = (str(name), int(value.id))
            incumbent = declared_width_page.latest(row)
            if incumbent is not None and int(incumbent) != declared:
                raise ValueError(
                    f"{PRECISION_DECLARED_FORMAL_WIDTH.name} disagreement "
                    f"for {row!r}: recorded={incumbent!r}, "
                    f"proposed={declared!r}"
                )
            # The width is read off the formal's own accounting: DERIVED
            # from the formal's ``ssa_value`` cell and the reducer's facts
            # that put the width there.
            book.post(
                PRECISION_DECLARED_FORMAL_WIDTH, row, declared,
                stage=PRECISION_EXPANSION,
                provenance=Derived(tuple(dict.fromkeys((
                    *identity_cells(function, value),
                    *_source_precision_fact_cells(book, function, value),
                )))),
                mode=Mode.CONCORD,
            )
            widths[int(value.id)] = declared
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) not in expandable:
                    continue
                width = max(
                    int(instruction.attributes.get("precision_limbs") or 1), 1
                )
                for argument in instruction.args:
                    if (
                        int(argument.id) in formal_ids
                        and int(argument.id) not in promoted_scalar_formals[
                            str(name)
                        ]
                        and width > 1
                    ):
                        widths[int(argument.id)] = max(
                            widths.get(int(argument.id), 1), width
                        )
        # A formal stored into a precision array is an element of it: the
        # store region of a pack loop receives the value to write as a
        # formal, and keeping it one scalar wrote only the leading limb.
        arrays_here = precision_arrays.get(str(name)) or {}
        for block in function.blocks.values():
            pointers = {
                int(instruction.res.id): int(instruction.args[0].id)
                for instruction in block.instrs
                if str(instruction.op) == "GetElementPtr"
                and instruction.args and instruction.res is not None
            }
            for instruction in block.instrs:
                if str(instruction.op) != "Store" or len(instruction.args) < 2:
                    continue
                stored = int(instruction.args[0].id)
                width = arrays_here.get(
                    pointers.get(int(instruction.args[1].id), -1), 1
                )
                if (
                    width > 1 and stored in formal_ids
                    and stored not in promoted_scalar_formals[str(name)]
                ):
                    widths[stored] = max(widths.get(stored, 1), width)
        if widths:
            parameter_widths[str(name)] = widths

    # A callee whose formals grew needs its callers to supply the new ones,
    # and a caller can only do that if ITS matching formal grew too. Pushing
    # the requirement back through call sites to a fixed point is what makes
    # the change survive the wrapper/region pair the planner emits.
    changed = True
    while changed:
        changed = False
        for name, function in functions.items():
            for block in function.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in ("Call", "call"):
                        continue
                    callee = str(instruction.attributes.get("callee") or "")
                    wanted = parameter_widths.get(callee)
                    if not wanted:
                        continue
                    target = functions.get(callee)
                    if target is None:
                        continue
                    formals = original_formals.get(callee, tuple(target.args))
                    caller_formals = {int(v.id) for v in function.args}
                    here = parameter_widths.setdefault(str(name), {})
                    for position, formal in enumerate(formals):
                        width = wanted.get(int(formal.id), 1)
                        if width <= 1 or position >= len(instruction.args):
                            continue
                        actual = int(instruction.args[position].id)
                        if (
                            actual in caller_formals
                            and actual not in promoted_scalar_formals[
                                str(name)
                            ]
                            and here.get(actual, 1) < width
                        ):
                            here[actual] = width
                            changed = True

    seeded: dict[str, list[list]] = {}
    for name, function in functions.items():
        widths = parameter_widths.get(str(name)) or {}
        if not widths:
            continue
        rows: list[list] = []
        parameter_name_by_id = {
            int(value_id): str(label)
            for label, value_id in (
                function.metadata.get("parameter_names", ()) or ()
            )
        }
        member_receipts = list(
            function.metadata.get("parameter_member_formals", ()) or ()
        )
        for formal in list(function.args):
            width = widths.get(int(formal.id), 1)
            if width <= 1:
                continue
            row = [formal]
            for limb_index in range(1, width):
                # Minted through the book from the declared formal's cell.
                extra = SSAValue(
                    mint_ssa_value(
                        function, PRECISION_FORMAL_LIMB, (formal,),
                        dtype=formal.dtype, shape=tuple(formal.shape or ()),
                        stage=PRECISION_EXPANSION,
                    ),
                    dtype=formal.dtype,
                    shape=tuple(formal.shape or ()),
                    device=formal.device,
                    accounting={
                        "precision_formal_source_id": int(formal.id),
                        "precision_limb_index": int(limb_index),
                        "precision_extent_source_id": int(formal.id),
                    },
                )
                function.args.append(extra)
                row.append(extra)
                parameter_name = parameter_name_by_id.get(int(formal.id))
                if parameter_name is not None:
                    member_receipts.append({
                        "value_id": int(extra.id),
                        "parameter": parameter_name,
                        "path": (int(limb_index),),
                    })
            rows.append(row)
        if member_receipts:
            function.metadata["parameter_member_formals"] = tuple(
                member_receipts
            )
        seeded[str(name)] = rows

    # Callees first: a caller reading a region's result needs to know which
    # aggregate fields the region appended for that result's limbs.
    callee_first: list[str] = []

    def visit(name: str, seen: set[str]) -> None:
        if name in seen or name not in functions:
            return
        seen.add(name)
        for block in functions[name].blocks.values():
            for instruction in block.instrs:
                if instruction.op in ("Call", "call"):
                    visit(str(instruction.attributes.get("callee") or ""), seen)
        callee_first.append(name)

    aggregate_callees = {
        str(instruction.attributes.get("callee") or "")
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in ("Call", "call")
        and instruction.res is not None
        and instruction.attributes.get("result_convention") == "ssa.aggregate"
    }
    visited: set[str] = set()
    for name in functions:
        visit(str(name), visited)
    # result field -> the fields appended for its limbs, per region
    returned_limb_fields: dict[str, dict[int, tuple[tuple[int, object], ...]]] = {}

    for function_name in callee_first:
        function = functions[function_name]
        # value id -> its limbs, high first. Absent means an ordinary value,
        # which is simply a one-limb expansion whose low limb is zero.
        limbs: dict[int, list] = {}
        lowered_limb_records: dict[int, tuple[int, ...]] = {}
        for row in seeded.get(str(function_name), ()):  # declared parameters
            limbs[int(row[0].id)] = list(row)
            lowered_limb_records[int(row[0].id)] = tuple(
                int(value.id) for value in row
            )

        for block in function.blocks.values():
            emitted: list = []
            known_scalar_ids = {
                int(instruction.res.id)
                for candidate_block in function.blocks.values()
                for instruction in candidate_block.instrs
                if instruction.op == "Const"
                and instruction.res is not None
                and not tuple(instruction.res.shape or ())
                and not int((instruction.res.accounting or {}).get(
                    "program_abi_rank", 0
                ) or 0)
            }

            # The expansion builder, with this block as its insertion point.
            # ``element`` is set per instruction below, from the section the
            # instruction belongs to.
            emitter = WideExpansionEmitter(
                function, emitted, two_product_flavor=two_product_flavor,
                limbs=limbs, known_scalar_ids=known_scalar_ids,
            )
            fresh = emitter.fresh
            put = emitter.put
            parts = emitter.parts
            zero = emitter.zero
            expanded = emitter.expanded
            add_expansions = emitter.add_expansions
            multiply_expansions = emitter.multiply_expansions
            divide_expansions = emitter.divide_expansions
            sqrt_expansion = emitter.sqrt_expansion
            exp_expansion = emitter.exp_expansion
            log_expansion = emitter.log_expansion

            def collapse(parts_of, original):
                """Define the ORIGINAL value as the sum of its limbs.

                The expanded operation drops out of the stream, so nothing
                would define the value it used to produce -- and a declared
                output, a buffer order or a Ret still names that id. An
                undefined output buffer reads as zero, which is a plausible
                number and therefore the worst possible failure.

                Reusing the id rather than rewriting every consumer means
                metadata that names values, and not just instruction
                operands, keeps working without this pass having to know
                what all of it is.
                """

                total = parts_of[0]
                for each in parts_of[1:-1]:
                    total = put("Add", (total, each), total)
                emitted.append(Instr(
                    "Add", [total, parts_of[-1]], original,
                    attributes={
                        PRECISION_SECTION_ATTRIBUTE: True,
                        "lowered_from": "precision.collapse",
                    },
                ))
                original.accounting[
                    "precision_extent_source_id"
                ] = int(parts_of[0].id)
                # The limb channel the carry pass put on this value is now
                # STALE and actively harmful. It described a value whose
                # limbs lived on a trailing axis; lowering has just made
                # them separate scalars, and the value left here holds one
                # number. A backend that believes the shape builds an ABI
                # around an array -- Fortran demanded an integer extent
                # parameter per lowered value and the call wrote through a
                # null pointer. LLVM ignored the shape and worked, which is
                # exactly how a disagreement like this stays hidden.
                physical_shape = tuple(original.shape or ())
                carried_width = max(int(
                    (original.accounting or {}).get("precision_limbs") or 1
                ), 1)
                # Only the historical channel-shaped representation owns a
                # trailing limb axis.  A specialized Precision value now
                # carries its ordinary logical extents while the limbs are
                # separate SSA values; removing the last logical extent here
                # turned a `(2,)` result into a scalar at the LLVM boundary.
                shape_fact = channel_shape_page.latest((
                    str(function_name), int(original.id),
                ))
                if shape_fact is not None:
                    logical_shape, channel_shape, recorded_width = shape_fact
                    if int(recorded_width) != carried_width:
                        raise ValueError(
                            "precision collapse width disagrees with its "
                            "channel-shape concordance for "
                            f"{(str(function_name), int(original.id))!r}: "
                            f"width={carried_width}, recorded={shape_fact!r}"
                        )
                    if physical_shape not in {
                        tuple(logical_shape), tuple(channel_shape),
                    }:
                        raise ValueError(
                            "precision collapse shape disagrees with its "
                            "channel-shape concordance for "
                            f"{(str(function_name), int(original.id))!r}: "
                            f"shape={physical_shape!r}, recorded={shape_fact!r}"
                        )
                    original.shape = tuple(logical_shape)
                elif (
                    carried_width > 1
                    and physical_shape
                    and physical_shape[-1] == carried_width
                ):
                    original.shape = physical_shape[:-1]
                else:
                    original.shape = physical_shape
                if original.accounting:
                    original.accounting.pop("precision_limbs", None)
                    original.accounting.pop("precision_element", None)
                    original.accounting.pop("precision_logical_shape", None)

            arrays = precision_arrays.get(str(function_name)) or {}
            # pointer value id -> (base array id, index value)
            addressed: dict[int, tuple] = {}
            # aggregate value id -> the region that returned it, and a field
            # pointer's id -> (its GetElementPtr, the field it addresses)
            aggregates: dict[int, str] = {}
            fields_of: dict[int, tuple] = {}

            def constant(number: int, like):
                # Minted through the book from the index it strides.
                value = SSAValue(
                    mint_ssa_value(
                        function, PRECISION_CONSTANT, (like,),
                        dtype="int", shape=(), stage=PRECISION_EXPANSION,
                    ),
                    dtype="int",
                    shape=(),
                )
                emitted.append(Instr("Const", [], value, attributes={
                    "constant": int(number),
                    PRECISION_SECTION_ATTRIBUTE: True,
                }))
                return value

            def channel(base, index, width: int, position: int):
                """Address element ``index`` limb ``position``, stride ``width``.

                Flat index ``index * width + position``. Said with Mul, Add
                and Const, which every destination already supplies, rather
                than with a new operator seven backends would each have to
                implement.
                """

                scaled = put("Mul", (index, constant(width, index)), index)
                flat = (
                    scaled if position == 0
                    else put("Add", (scaled, constant(position, index)), index)
                )
                return put("GetElementPtr", (base, flat), index)

            for instruction in block.instrs:
                operation = str(instruction.op)
                emitter.element = instruction.attributes.get(
                    "precision_element"
                )

                if operation in ("Call", "call"):
                    callee = str(instruction.attributes.get("callee") or "")
                    rows = seeded.get(callee)
                    if rows and not instruction.attributes.get(
                        "precision_actuals_bound"
                    ):
                        target_formals = original_formals.get(callee, ())
                        by_head = {int(row[0].id): row for row in rows}
                        original_actuals = list(instruction.args)
                        additions = []
                        declared_inputs = instruction.attributes.get(
                            "callee_input_ids"
                        )
                        added_formal_ids = []
                        for position, formal in enumerate(target_formals):
                            row = by_head.get(int(formal.id))
                            if row is None or position >= len(original_actuals):
                                continue
                            actual = original_actuals[position]
                            supplied = parts(actual)
                            for index in range(1, len(row)):
                                additions.append(
                                    supplied[index]
                                    if index < len(supplied)
                                    else zero(actual)
                                )
                                added_formal_ids.append(int(row[index].id))
                        instruction.args.extend(additions)
                        if declared_inputs is not None:
                            declared = tuple(map(int, declared_inputs))
                            if len(declared) != len(original_actuals):
                                raise ValueError(
                                    "precision call input concordance starts "
                                    "from a receipt that does not cover its "
                                    f"actuals: {callee!r} receipt={declared!r}, "
                                    f"actuals={len(original_actuals)}"
                                )
                            instruction.attributes["callee_input_ids"] = (
                                *declared, *added_formal_ids,
                            )
                        instruction.attributes["precision_actuals_bound"] = True
                        instruction.attributes[
                            "precision_actual_count"
                        ] = len(additions)
                        instruction.attributes[
                            "precision_actual_formal_ids"
                        ] = tuple(added_formal_ids)
                    appended = returned_limb_fields.get(callee)
                    if appended and instruction.res is not None:
                        aggregates[int(instruction.res.id)] = callee
                        if (
                            "output_ids" in instruction.attributes
                            and not instruction.attributes.get(
                                "precision_result_limbs_bound"
                            )
                        ):
                            instruction.attributes["output_ids"] = (
                                *tuple(instruction.attributes["output_ids"]),
                                *(
                                    int(limb.id)
                                    for _field, fields in sorted(
                                        appended.items()
                                    )
                                    for _index, limb in fields
                                ),
                            )
                            instruction.attributes[
                                "precision_result_limbs_bound"
                            ] = True
                    emitted.append(instruction)
                    continue

                if (
                    operation == "GetElementPtr" and len(instruction.args) >= 2
                    and instruction.res is not None
                ):
                    addressed[int(instruction.res.id)] = (
                        instruction.args[0], instruction.args[1],
                    )
                    if instruction.attributes.get("aggregate_index") is not None:
                        fields_of[int(instruction.res.id)] = (
                            instruction,
                            int(instruction.attributes["aggregate_index"]),
                        )
                    emitted.append(instruction)
                    continue

                if (
                    operation == "Load" and instruction.args
                    and instruction.res is not None
                ):
                    where = addressed.get(int(instruction.args[0].id))
                    width = arrays.get(int(where[0].id)) if where else None
                    if width and width > 1:
                        base, index = where
                        loaded_limbs = [
                            put("Load", (channel(base, index, width, position),),
                                instruction.res)
                            for position in range(width)
                        ]
                        limbs[int(instruction.res.id)] = loaded_limbs
                        lowered_limb_records[int(instruction.res.id)] = tuple(
                            int(value.id) for value in loaded_limbs
                        )
                        # The original value must still be DEFINED. Not every
                        # consumer of a load from a precision array is a
                        # precision operation -- a shell computing `s = z * z`
                        # from an undeclared local reads it as ordinary
                        # arithmetic -- and such a consumer still names the id
                        # this expansion just removed the definition of.
                        # Leaving it dangling produced NaN, because the
                        # instruction read whatever the slot happened to hold.
                        collapse(loaded_limbs, instruction.res)
                        continue
                    field = fields_of.get(int(instruction.args[0].id))
                    callee = aggregates.get(int(where[0].id)) if where else None
                    appended = (
                        returned_limb_fields.get(callee, {}).get(field[1])
                        if callee and field else None
                    )
                    if appended:
                        # The region returned this result's limbs in fields
                        # of their own; read each one back under the id the
                        # region gave it, as the head field already is.
                        emitted.append(instruction)
                        pointer_instruction = field[0]
                        received = []
                        for index, limb in appended:
                            pointer = fresh(pointer_instruction.res)
                            pointer.dtype = pointer_instruction.res.dtype
                            pointer.shape = tuple(
                                pointer_instruction.res.shape or ()
                            )
                            emitted.append(Instr(
                                "GetElementPtr",
                                [where[0], constant(index, where[1])],
                                pointer,
                                attributes={
                                    **dict(pointer_instruction.attributes),
                                    "aggregate_index": int(index),
                                    "source_output_id": int(limb.id),
                                    PRECISION_SECTION_ATTRIBUTE: True,
                                },
                            ))
                            value = SSAValue(
                                int(limb.id), dtype=limb.dtype,
                                shape=tuple(limb.shape or ()),
                                device=limb.device,
                                accounting={
                                    "ssa_call_result_from": (
                                        callee, int(limb.id),
                                    ),
                                },
                            )
                            emitted.append(Instr(
                                "Load", [pointer], value,
                                attributes={
                                    **dict(instruction.attributes),
                                    "aggregate_index": int(index),
                                    "source_output_id": int(limb.id),
                                    PRECISION_SECTION_ATTRIBUTE: True,
                                },
                            ))
                            received.append(value)
                        limbs[int(instruction.res.id)] = received
                        lowered_limb_records[int(instruction.res.id)] = tuple(
                            int(value.id) for value in received
                        )
                        continue
                    emitted.append(instruction)
                    continue

                if operation == "Store" and len(instruction.args) >= 2:
                    where = addressed.get(int(instruction.args[1].id))
                    width = arrays.get(int(where[0].id)) if where else None
                    stored = limbs.get(int(instruction.args[0].id))
                    if width and width > 1 and stored:
                        base, index = where
                        for position in range(width):
                            emitted.append(Instr(
                                "Store",
                                [
                                    stored[position] if position < len(stored)
                                    else zero(stored[0]),
                                    channel(base, index, width, position),
                                ],
                                None,
                                attributes={
                                    PRECISION_SECTION_ATTRIBUTE: True,
                                    "lowered_from": "precision.store",
                                },
                            ))
                        continue
                    emitted.append(instruction)
                    continue

                if (
                    operation in ("Ret", "ret")
                    and str(function_name) in aggregate_callees
                    and not instruction.attributes.get(
                        "precision_result_limbs_bound"
                    )
                    and any(
                        len(limbs.get(int(value.id), ())) > 1
                        for value in instruction.args
                    )
                ):
                    # A region hands its results back as an aggregate the
                    # caller takes apart field by field. A limbed result
                    # keeps its collapsed head in its own field, for any
                    # reader that wants one number, and its limbs follow in
                    # fields appended after every existing one -- so no
                    # existing field moves.
                    appended: dict[int, tuple] = {}
                    extra = []
                    for position, value in enumerate(list(instruction.args)):
                        held = limbs.get(int(value.id), ())
                        if len(held) <= 1:
                            continue
                        rows = []
                        for limb in held:
                            rows.append((
                                len(instruction.args) + len(extra), limb,
                            ))
                            extra.append(limb)
                        appended[position] = tuple(rows)
                    instruction.args.extend(extra)
                    instruction.attributes["precision_result_limbs_bound"] = True
                    returned_limb_fields[str(function_name)] = appended
                    function.metadata["precision_returned_limb_fields"] = tuple(
                        (position, tuple(
                            (index, int(limb.id)) for index, limb in rows
                        ))
                        for position, rows in sorted(appended.items())
                    )
                    emitted.append(instruction)
                    continue

                if operation not in expandable or instruction.res is None:
                    emitted.append(instruction)
                    continue

                if operation == PRECISION_PACK_NAME:
                    width = max(int(
                        instruction.attributes.get("precision_limbs") or 2
                    ), 2)
                    source = list(instruction.args[:width])
                    if len(source) != width:
                        raise ValueError(
                            "precision pack does not carry its declared "
                            f"limbs: width={width}, operands={len(source)}"
                        )
                    limbs[int(instruction.res.id)] = source
                    lowered_limb_records[int(instruction.res.id)] = tuple(
                        int(value.id) for value in source
                    )
                    collapse(source, instruction.res)
                    counts[operation] += 1
                    continue

                if operation == neg_name:
                    # Exact per limb: a sign flip cannot round.
                    width = max(int(
                        instruction.attributes.get("precision_limbs") or 2
                    ), 2)
                    source = expanded(instruction.args[0], width)
                    negated = [put("Neg", (each,), each) for each in source]
                    limbs[int(instruction.res.id)] = negated
                    lowered_limb_records[int(instruction.res.id)] = tuple(
                        int(value.id) for value in negated
                    )
                    collapse(negated, instruction.res)
                    counts[operation] += 1
                    continue

                if operation == sqrt_name:
                    width = max(int(
                        instruction.attributes.get("precision_limbs") or 2
                    ), 2)
                    source = expanded(instruction.args[0], width)
                    root = sqrt_expansion(source, width)
                    limbs[int(instruction.res.id)] = root
                    lowered_limb_records[int(instruction.res.id)] = tuple(
                        int(value.id) for value in root
                    )
                    collapse(root, instruction.res)
                    counts[operation] += 1
                    continue

                if operation in {exp_name, log_name}:
                    width = max(int(
                        instruction.attributes.get("precision_limbs") or 2
                    ), 2)
                    source = expanded(instruction.args[0], width)
                    result_limbs = (
                        exp_expansion(source, width)
                        if operation == exp_name
                        else log_expansion(source, width)
                    )
                    limbs[int(instruction.res.id)] = result_limbs
                    lowered_limb_records[int(instruction.res.id)] = tuple(
                        int(value.id) for value in result_limbs
                    )
                    collapse(result_limbs, instruction.res)
                    counts[operation] += 1
                    continue

                width = max(int(
                    instruction.attributes.get("precision_limbs") or 2
                ), 2)
                left = expanded(instruction.args[0], width)
                right = expanded(instruction.args[1], width)
                if operation == sub_name:
                    right = [put("Neg", (each,), each) for each in right]

                if operation == mul_name:
                    result_limbs = multiply_expansions(left, right, width)
                elif operation == div_name:
                    result_limbs = divide_expansions(left, right, width)
                else:
                    result_limbs = add_expansions(left, right, width)

                limbs[int(instruction.res.id)] = result_limbs
                lowered_limb_records[int(instruction.res.id)] = tuple(
                    int(value.id) for value in result_limbs
                )
                collapse(result_limbs, instruction.res)
                counts[operation] += 1

            block.instrs = emitted

        if lowered_limb_records:
            function.metadata["precision_lowered_values"] = tuple(
                (value_id, limb_ids)
                for value_id, limb_ids in sorted(lowered_limb_records.items())
            )

    return counts


PRECISION_PIPELINE_METADATA = "precision_pipeline"
NUMERIC_FEATURE_PIPELINE_METADATA = "numeric_feature_pipeline"


def apply_numeric_feature_pipeline(module) -> dict:
    """Validate composite descriptors at the completed repository-SSA seam.

    Annotation ingestion and topology reduction have already selected the
    numeric feature algebra. This seam proves that the same descriptor reached
    repository SSA before any backend or the inner Precision transaction sees
    the operation. Until component expansion is installed, a composite section
    refuses here rather than degrading to its scalar spelling.
    """

    operation_names = {
        "Add": "add",
        "Sub": "sub",
        "Mul": "mul",
        "Div": "truediv",
        "Neg": "neg",
        "Pow": "pow_integer",
    }
    sections = []
    for function_name, function in module.functions.items():
        for block_name, block in function.blocks.items():
            for instruction in block.instrs:
                if not instruction.attributes.get("numeric_composite_pending"):
                    continue
                descriptor = dict(
                    instruction.attributes.get(
                        "numeric_feature_descriptor"
                    ) or {}
                )
                operation = operation_names.get(str(instruction.op))
                supported = tuple(map(str, descriptor.get("operators") or ()))
                if operation is None or operation not in supported:
                    raise ValueError(
                        "numeric feature descriptor does not cover its SSA "
                        f"operator: function={function_name!r}, "
                        f"op={instruction.op!s}, descriptor={descriptor!r}"
                    )
                sections.append({
                    "function": str(function_name),
                    "block": str(block_name),
                    "value_id": (
                        None if instruction.res is None
                        else int(instruction.res.id)
                    ),
                    "operand_ids": tuple(
                        int(value.id) for value in instruction.args
                    ),
                    "operation": operation,
                    "descriptor": descriptor,
                })
    receipt = {
        "schema": "numeric-feature-pipeline-v1",
        "status": (
            "outer-lowering-required" if sections else "not-present"
        ),
        "sections": sections,
    }
    module.metadata[NUMERIC_FEATURE_PIPELINE_METADATA] = receipt
    if sections:
        summary = tuple(
            (
                section["function"], section["value_id"],
                section["descriptor"].get("type_name"),
                section["operation"],
            )
            for section in sections
        )
        raise NotImplementedError(
            "numeric composite repository SSA reached the outer-algebra "
            "lowering seam but component expansion is not installed: "
            + repr(summary)
        )
    return receipt


def _refresh_precision_call_records(module) -> int:
    """Extend canonical call records after precision grows a scalar ABI.

    The source linker owns every pre-existing binding.  Precision lowering
    only appends formals and actuals, so this refresh preserves those records
    byte-for-byte and adds exact caller-value bindings for the appended limb
    positions.  No name matching or positional reconstruction of old state is
    permitted here.
    """

    refreshed = 0
    call_table = dict(getattr(module, "call_table", {}) or {})
    for caller_name, records in tuple(call_table.items()):
        caller = module.functions.get(str(caller_name))
        if caller is None:
            continue
        instructions = tuple(
            instruction
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.op in ("Call", "call")
        )
        updated = []
        for record in records:
            call = next((
                instruction for instruction in instructions
                if str(instruction.attributes.get("callee") or "")
                == str(record.callee_symbol or "")
                and (
                    instruction.attributes.get("plan_callsite_id") is None
                    or int(instruction.attributes["plan_callsite_id"])
                    == int(record.callsite_id)
                )
            ), None)
            callee = module.functions.get(str(record.callee_symbol or ""))
            if call is None or callee is None:
                updated.append(record)
                continue

            existing_arguments = {
                (int(actual), int(formal))
                for actual, formal in record.argument_bindings
            }
            existing_frames = {
                int(formal): (str(kind), source)
                for formal, kind, source in record.frame_bindings
            }
            additions = 0
            for actual, formal in zip(call.args, callee.args):
                binding = (int(actual.id), int(formal.id))
                if binding not in existing_arguments:
                    existing_arguments.add(binding)
                    additions += 1
                existing_frames.setdefault(
                    int(formal.id), ("caller_value", int(actual.id))
                )
            if not additions:
                updated.append(record)
                continue
            storage = tuple(dict.fromkeys((
                *map(int, record.callee_storage_value_ids),
                *(int(value.id) for value in callee.args),
            )))
            bound = set(existing_frames)
            updated.append(dataclasses.replace(
                record,
                argument_bindings=tuple(sorted(existing_arguments)),
                callee_storage_value_ids=storage,
                frame_bindings=tuple(
                    (formal, kind, source)
                    for formal, (kind, source) in sorted(existing_frames.items())
                ),
                unresolved_frame_value_ids=tuple(
                    value for value in record.unresolved_frame_value_ids
                    if int(value) not in bound
                ),
            ))
            refreshed += 1
        call_table[str(caller_name)] = tuple(updated)
    module.call_table = call_table
    return refreshed


def apply_precision_pipeline(
    module, two_product_flavor: str | None = None,
) -> dict:
    """Run the complete exact precision lowering transaction on ``module``.

    This is the sole production entry point.  It records the pre-lowering
    section contracts before their ``precision_*`` operators disappear,
    performs only catalogue identities marked exact, expands the arithmetic,
    repairs the canonical call ABI, and refuses to return if an abstract
    precision operator survived.  A durable receipt makes the transaction
    idempotent and gives emitters the obligations they must validate.

    ``two_product_flavor`` (see ``TWO_PRODUCT_FLAVORS``) chooses the exact
    product spelling and thereby which obligations the sections record:
    ``"fma"`` for destinations with a single-rounding fma, ``"split"`` for
    those without one.  ``None`` defers to ``active_two_product_flavor()``,
    which is how a caller above the completed-module seam chooses without
    threading an argument through the source compiler.  A module is lowered
    in ONE flavour; asking again with a different one is refused rather
    than answered with the wrong receipt, because the instructions already
    emitted cannot change.
    """
    from .identity_concordance import (
        _ACTIVE_IDENTITY_BOOK,
        begin_identity_book,
        end_identity_book,
    )

    active = _ACTIVE_IDENTITY_BOOK.get()
    if active is None or getattr(active, "detached", False):
        # Outside a compile this call is its own transaction: its facts
        # describe this module only, so they go in the module's own book
        # (``identity_book(module)`` reads them back after the fact).
        # A detached book attached to the module counts as no book (the
        # ``emission_concordance.emission_book`` rule): re-activating it would
        # land back in this branch forever.
        metadata = getattr(module, "metadata", None)
        own = None if metadata is None else metadata.get("identity_book")
        if own is not None and getattr(own, "detached", False):
            own = None
        token = _ACTIVE_IDENTITY_BOOK.set(own) if own is not None else None
        if token is None:
            own, token = begin_identity_book()
            if metadata is not None:
                metadata["identity_book"] = own
        try:
            return apply_precision_pipeline(module, two_product_flavor)
        finally:
            end_identity_book(token)

    from ..common.tensors.topological_reducer import PRECISION_SINGULAR_NAMES

    if two_product_flavor is None:
        two_product_flavor = active_two_product_flavor()
    if two_product_flavor not in TWO_PRODUCT_FLAVORS:
        raise ValueError(
            f"unknown two_product flavour {two_product_flavor!r}; "
            f"expected one of {TWO_PRODUCT_FLAVORS}"
        )
    functions = module.functions
    prior = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    )
    if prior and prior.get("status") == "lowered":
        held = str(prior.get("two_product_flavor") or "fma")
        if held != two_product_flavor:
            raise ValueError(
                f"module already lowered with two_product flavour {held!r}; "
                f"cannot re-lower as {two_product_flavor!r} -- lower a fresh "
                "module for the other flavour"
            )
        return prior

    precision_ops = frozenset(PRECISION_SINGULAR_NAMES.values())
    before = sum(
        str(instruction.op) in precision_ops
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    if not before:
        return {
            "schema": "precision-pipeline-v1",
            "status": "not-present",
            "exact_only": True,
            "source_operations": 0,
            "section_contracts": [],
        }

    carried = carry_precision_through_ssa(functions)
    identity_counts = reduce_precision_operations(functions)
    contracts = precision_section_contracts(functions, two_product_flavor)
    marked = mark_precision_sections(functions)
    lowered = lower_precision_operations(functions, two_product_flavor)
    phi_descriptors = settle_single_input_phi_descriptors(functions)
    calls_refreshed = _refresh_precision_call_records(module)
    remaining = tuple(
        (str(name), int(instruction.res.id) if instruction.res else None,
         str(instruction.op))
        for name, function in functions.items()
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in precision_ops
    )
    if remaining:
        raise ValueError(
            "precision lowering left abstract operators: " + repr(remaining)
        )

    receipt = {
        "schema": "precision-pipeline-v1",
        "status": "lowered",
        "exact_only": True,
        "two_product_flavor": two_product_flavor,
        "source_operations": int(before),
        "carried_facts": int(carried),
        "identity_counts": dict(identity_counts),
        "identity_catalogue": [
            {
                "name": identity.name,
                "exact": bool(identity.exact),
                "requires": list(identity.requires),
                "fired": int(identity_counts.get(identity.name, 0)),
            }
            for identity in PRECISION_IDENTITIES
        ],
        "refused_identities": [identity.name for identity in PRECISION_REFUSALS],
        "section_contracts": [contract.as_record() for contract in contracts],
        "marked_instructions": int(marked),
        "lowered_operations": dict(lowered),
        "single_input_phi_descriptors": int(phi_descriptors),
        "call_records_refreshed": int(calls_refreshed),
    }
    module.metadata[PRECISION_PIPELINE_METADATA] = receipt
    return receipt


def settle_single_input_phi_descriptors(functions) -> int:
    """Carry an exact sole incoming descriptor through a structural Phi.

    Static specialization and loop evaporation can leave a join with one
    surviving arm.  Its value identity remains useful for control provenance,
    but its descriptor is no longer a merge problem: the sole incoming value
    is authoritative.  This runs after precision collapse because that is
    when the incoming logical shape is restored.
    """
    from .concordance_declarations import (
        SINGLE_INPUT_PHI_DESCRIPTOR, SINGLE_INPUT_PHI_SETTLEMENT,
    )
    from .identity_concordance import Derived, Mode, current_identity_book
    from .ssa_record_return_state import identity_cells

    book = current_identity_book()
    page = book.page(SINGLE_INPUT_PHI_DESCRIPTOR)
    changed = 0
    for function_name, function in functions.items():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    str(instruction.op).casefold() != "phi"
                    or instruction.res is None
                    or len(instruction.args) != 1
                ):
                    continue
                source = instruction.args[0]
                result = instruction.res
                source_shape = tuple(source.shape or ())
                source_dtype = str(source.dtype or "")
                row = (str(function_name), int(result.id))
                proposed = (
                    int(source.id), source_dtype, source_shape,
                )
                incumbent = page.latest(row)
                if incumbent is not None and tuple(incumbent) != proposed:
                    raise ValueError(
                        "single-input Phi descriptor concordance disagreement "
                        f"for {row!r}: recorded={incumbent!r}, "
                        f"proposed={proposed!r}"
                    )
                # The descriptor is the sole arm's, read off that value for
                # this Phi: DERIVED from both values' ``ssa_value`` cells.
                book.post(
                    SINGLE_INPUT_PHI_DESCRIPTOR, row, proposed,
                    stage=SINGLE_INPUT_PHI_SETTLEMENT,
                    provenance=Derived(identity_cells(function, result, source)),
                    mode=Mode.CONCORD,
                )
                result_shape = tuple(result.shape or ())
                if result_shape and result_shape != source_shape:
                    raise ValueError(
                        "single-input Phi shape disagrees with its sole "
                        f"incoming value for {row!r}: result={result_shape!r}, "
                        f"source={source_shape!r}"
                    )
                result_dtype = str(result.dtype or "")
                if (
                    result_dtype not in {"", "none", "unknown", "ssa.aggregate"}
                    and source_dtype
                    and result_dtype != source_dtype
                ):
                    raise ValueError(
                        "single-input Phi dtype disagrees with its sole "
                        f"incoming value for {row!r}: result={result_dtype!r}, "
                        f"source={source_dtype!r}"
                    )
                before = (result_dtype, result_shape, result.device)
                if source_dtype:
                    result.dtype = source.dtype
                result.shape = source_shape
                if result.device is None:
                    result.device = source.device
                result.accounting = {
                    **dict(result.accounting or {}),
                    "single_input_phi_source_id": int(source.id),
                }
                after = (
                    str(result.dtype or ""), tuple(result.shape or ()),
                    result.device,
                )
                changed += before != after
    return changed


def precision_backend_shortfalls(
    module, backend: str, function_names=None,
) -> tuple[dict, ...]:
    """Return persisted precision obligations ``backend`` cannot deliver."""

    receipt = (getattr(module, "metadata", {}) or {}).get(
        PRECISION_PIPELINE_METADATA
    ) or {}
    selected = None if function_names is None else {
        str(name) for name in function_names
    }
    capabilities = frozenset(BACKEND_PRECISION_CAPABILITIES.get(
        str(backend).casefold(), ()
    ))
    shortfalls = []
    for contract in receipt.get("section_contracts", ()):
        if selected is not None and str(contract.get("function")) not in selected:
            continue
        required = set(map(str, contract.get("obligations", ()))) - {
            LANE_STAGING
        }
        if not contract.get("fma_value_ids"):
            required.discard(FMA_MANDATORY)
        missing = tuple(sorted(required - capabilities))
        if missing:
            shortfalls.append({
                "function": str(contract.get("function")),
                "value_ids": tuple(map(int, contract.get("value_ids", ()))),
                "missing": missing,
            })
    return tuple(shortfalls)


def inline_host_linear_source_regions(functions) -> tuple[dict, tuple[dict, ...]]:
    """Return a host-lane view with safe one-call source regions spliced in.

    The repository keeps planned regions separate because shader deployment
    needs a dispatchable subgraph.  A native host lane has the opposite cost:
    an outputless loop-body region becomes one subroutine call per element.
    This identity does not alter the repository module.  It creates replacement
    caller/block/instruction containers only where every legality condition is
    proved, and returns an ordinary mapping for the host emitter.

    Eligible means: a compile-complementary source region, exactly one callsite,
    one straight-line block, no published outputs, no nested call/control, an
    exact formal/actual arity match, and no produced-ID collision in the caller.
    Anything less stays a call.  The receipt states every splice explicitly.
    """

    from .work_contract import active_contract
    if active_contract().functions.inline_depth == 0:
        return dict(functions), ()

    view = dict(functions)
    calls_by_callee: dict[str, list[tuple[str, str, int, object]]] = {}
    for caller_name, caller in functions.items():
        for block_name, block in caller.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.op not in ("Call", "call"):
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                if callee:
                    calls_by_callee.setdefault(callee, []).append((
                        str(caller_name), str(block_name), index, instruction,
                    ))

    receipts = []
    forbidden = {
        "Call", "call", "Ret", "ret", "Return", "return", "Br", "br",
        "Branch", "branch", "CondBr", "condbr", "Phi", "phi",
    }
    replacements: dict[tuple[str, str, int], list] = {}
    removable = set()
    for callee_name, occurrences in calls_by_callee.items():
        callee = functions.get(callee_name)
        if callee is None or len(occurrences) != 1:
            continue
        integral = dict(callee.metadata.get("source_region_integral") or {})
        # PUBLISHED OUTPUTS ARE FINE. The splice clones each instruction
        # with its own result value, ids and all, so a value the region
        # published is produced in the caller at the same id the caller
        # already refers to -- there is nothing left to publish through.
        # The one hazard that would make this unsafe, an id in the region
        # colliding with one the caller already produces, is checked
        # separately below and still rejects the splice.
        #
        # Refusing them was costing a real destination: a two-limb result
        # publishes two values, and WGSL has no multi-value helper, so the
        # GPU lane could not take a wide kernel at all while the region
        # stayed a call.
        if not integral:
            continue
        if set(callee.blocks) != {"entry"}:
            continue
        body = callee.blocks["entry"]
        if any(str(instruction.op) in forbidden for instruction in body.instrs):
            continue
        caller_name, block_name, index, call = occurrences[0]
        if len(call.args) != len(callee.args):
            continue
        # An AGGREGATE result is not a reason to refuse. The call hands
        # back a span the caller immediately takes apart, and every read
        # of it carries `source_output_id` naming the value that slot
        # stands for -- so once the producing instructions live in the
        # caller, the span and the reads of it are both redundant and are
        # dropped below. Refusing this kept the region alive as a
        # separate function, and a value produced inside it could only
        # leave through the published set: the limbs of a wide result are
        # not in that set, so a two-limb kernel published its collapsed
        # head and the field it fed was single precision between steps.
        aggregate = (
            int(call.res.id) if call.res is not None
            and str(call.attributes.get("result_convention") or "")
            == "ssa.aggregate" else None
        )
        if call.res is not None and aggregate is None:
            continue
        produced = {
            int(instruction.res.id)
            for instruction in body.instrs if instruction.res is not None
        }
        formal_ids = {int(value.id) for value in callee.args}
        available = set(formal_ids)
        valid = True
        for instruction in body.instrs:
            if any(int(argument.id) not in available for argument in instruction.args):
                valid = False
                break
            if instruction.res is not None:
                available.add(int(instruction.res.id))
        if not valid:
            continue
        caller = functions[caller_name]
        # Instructions that only take the aggregate apart. They are keyed
        # by position so they can be deleted, and they are excluded from
        # the occupancy test below: a Load whose result IS the id the
        # spliced body produces is not a collision, it is the same value
        # arriving by a route that is about to disappear.
        aggregate_reads: dict[tuple, tuple] = {}
        if aggregate is not None:
            pointers = {aggregate}
            for caller_block_name, caller_block in caller.blocks.items():
                for position, instruction in enumerate(caller_block.instrs):
                    operation = str(instruction.op)
                    reads_aggregate = any(
                        int(argument.id) in pointers
                        for argument in instruction.args
                    )
                    if not reads_aggregate:
                        continue
                    if operation not in (
                        "GetElementPtr", "getelementptr", "Load", "load",
                    ):
                        aggregate_reads.clear()
                        break
                    # Only a GetElementPtr yields another POINTER into the
                    # span. A Load yields the value itself, and whatever
                    # consumes that -- the Ret, most importantly -- is an
                    # ordinary use and not an illegal reader of the
                    # aggregate. Treating a Load result as a pointer made
                    # the function's own return look like a violation and
                    # aborted every splice.
                    if instruction.res is not None and operation in (
                        "GetElementPtr", "getelementptr",
                    ):
                        pointers.add(int(instruction.res.id))
                    aggregate_reads[
                        (caller_name, str(caller_block_name), position)
                    ] = ()
                else:
                    continue
                break
            # Every read must be accounted for, and each published slot
            # must actually be produced by the body, or the caller would
            # be left referring to something no longer defined.
            named = {
                int(instruction.attributes.get("source_output_id"))
                for caller_block in caller.blocks.values()
                for instruction in caller_block.instrs
                if instruction.attributes.get("source_output_id") is not None
                and str(instruction.op) in ("Load", "load")
            }
            if not aggregate_reads or not named <= produced:
                continue
        read_results = {
            int(caller.blocks[key[1]].instrs[key[2]].res.id)
            for key in aggregate_reads
            if caller.blocks[key[1]].instrs[key[2]].res is not None
        }
        occupied = {
            int(value.id) for value in caller.args
        } | {
            int(instruction.res.id)
            for caller_block in caller.blocks.values()
            for instruction in caller_block.instrs
            if instruction.res is not None and instruction is not call
            and int(instruction.res.id) not in read_results
        }
        if produced & occupied:
            continue
        substitution = {
            int(formal.id): actual
            for formal, actual in zip(callee.args, call.args)
        }
        cloned = []
        for instruction in body.instrs:
            cloned.append(Instr(
                instruction.op,
                [substitution.get(int(argument.id), argument)
                 for argument in instruction.args],
                instruction.res,
                arg_roles=list(instruction.arg_roles),
                attributes={
                    **dict(instruction.attributes),
                    "inlined_source_region": str(callee_name),
                },
                source_span=(
                    None if instruction.source_span is None
                    else dict(instruction.source_span)
                ),
            ))
        replacements[(caller_name, block_name, index)] = cloned
        replacements.update(aggregate_reads)
        removable.add(callee_name)
        receipts.append({
            "caller": caller_name,
            "block": block_name,
            "callee": callee_name,
            "instruction_count": len(cloned),
            "identity_token_chain": list(
                integral.get("identity_token_chain") or ()
            ),
        })

    for caller_name in {key[0] for key in replacements}:
        caller = functions[caller_name]
        blocks = {}
        for block_name, block in caller.blocks.items():
            instructions = []
            for index, instruction in enumerate(block.instrs):
                instructions.extend(replacements.get(
                    (caller_name, str(block_name), index), (instruction,)
                ))
            blocks[block_name] = dataclasses.replace(
                block, instrs=instructions
            )
        metadata = dict(caller.metadata)
        own_receipts = tuple(
            receipt for receipt in receipts
            if receipt["caller"] == caller_name
        )
        metadata["host_linear_region_inlining"] = own_receipts
        # The limb rows move WITH the instructions. A value's row says
        # which SSA values are its limbs, and splicing relocates the
        # instructions that produce them into this caller -- so leaving
        # the rows behind in a function that no longer exists loses the
        # only record that a result is wide. Measured, the returned value
        # of a two-limb kernel had a row of (22418, 22437) inside the
        # region and none at all after inlining, so the low limb was
        # simply never published and the field it fed was single
        # precision between steps however many limbs the arithmetic used.
        rows = dict(metadata.get("precision_lowered_values") or ())
        for receipt in own_receipts:
            spliced = functions.get(str(receipt["callee"]))
            if spliced is None:
                continue
            # The caller's own rows WIN. A region shares its caller's
            # formals by identity, so it records rows for them too --
            # naming its own low-limb values, which are not the caller's.
            # Overwriting cost the caller the names of eight formals and
            # left them unbound at the boundary.
            for value_id, limb_ids in (
                spliced.metadata.get("precision_lowered_values") or ()
            ):
                rows.setdefault(value_id, limb_ids)
        if rows:
            metadata["precision_lowered_values"] = tuple(sorted(rows.items()))
        view[caller_name] = dataclasses.replace(
            caller, blocks=blocks, metadata=metadata
        )
    for callee_name in removable:
        view.pop(callee_name, None)
    return view, tuple(receipts)

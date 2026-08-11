"""Backend-neutral decomposition of derived tensor ops into primitives.

Many canonical tensor operators are compositions of a smaller primitive set
every backend already lowers: ``square`` is a multiply, ``reciprocal`` a
divide, ``cbrt`` a signed cube power, a float cast a copy, an int cast a
truncation. Spelling each one per backend duplicates the same algebra in every
target. Instead, decompose them ONCE here, at the SSA level, into
``mul``/``truediv``/``pow``/``abs``/``sign``/``copy``/``trunc`` + constants --
ops the Fortran, WASM, C and LLVM emitters all already know -- so the primitive
set stays small and a new backend inherits every derived op for free.

This is the op analogue of ``ir_indexing`` (subscripts -> address primitives):
one lowering, at the shared level, available to all backends.
"""
from __future__ import annotations

from ..transmogrifier.ssa import Instr, SSAValue


def _square(args, res, fresh):
    (x,) = args[:1]
    return [Instr("mul", [x, x], res)]


def _reciprocal(args, res, fresh):
    (x,) = args[:1]
    one = SSAValue(fresh(), dtype="float64")
    return [
        Instr("Const", [], one, attributes={"value": 1.0}),
        Instr("truediv", [one, x], res),
    ]


def _cbrt(args, res, fresh):
    # cbrt(x) = sign(x) * |x|**(1/3): the real cube root, sign-preserving (a bare
    # ** on a negative base is not real), and correct at 0 (sign(0)=+1, 0**k=0).
    (x,) = args[:1]
    magnitude = SSAValue(fresh(), dtype="float64")
    third = SSAValue(fresh(), dtype="float64")
    powered = SSAValue(fresh(), dtype="float64")
    signed = SSAValue(fresh(), dtype="float64")
    return [
        Instr("abs", [x], magnitude),
        Instr("Const", [], third, attributes={"value": 1.0 / 3.0}),
        Instr("pow", [magnitude, third], powered),
        Instr("sign", [x], signed),
        Instr("mul", [signed, powered], res),
    ]


def _copy_first(args, res, fresh):
    # Identity over the tensor operand, dropping any trailing shape/dtype/device
    # operands (which then fall out of the dependency walk as dead). A view op
    # (reshape/flatten/unsqueeze/...) only relabels shape on a contiguous buffer,
    # and a float/double/to cast in the f64 working kernel is the working type
    # already -- both are identity on the data. This mirrors the WASM backend's
    # _lower_view_ops (view -> add(x, 0)) and the WebGPU _SHAPE_ONLY set, but at
    # the SSA level so every backend inherits it.
    return [Instr("copy", [args[0]], res)]


def _trunc_first(args, res, fresh):
    # An integer cast truncates; the value stays in the f64 working type so it
    # flows through the numeric ABI unchanged.
    return [Instr("trunc", [args[0]], res)]


#: op name -> recipe(args, res, fresh) -> list[Instr]. The universal derived-op
#: translation matrix; every recipe emits only primitives all backends lower.
# View ops -- shape relabels that leave contiguous data untouched -- are
# identity on the data (a reshape only changes the shape label). This is the one
# canonical set, defined here at the SSA level; backends consume it instead of
# each carrying their own copy (the WASM backend's _lower_view_ops, the WebGPU
# _SHAPE_ONLY, ...). Data-REORDERING ops (transpose, permute, repeat_interleave)
# are deliberately NOT here: they move data and need index remapping no backend
# lowers yet.
VIEW_OPS = frozenset({
    "reshape", "view", "clone", "flatten", "ravel",
    "unsqueeze", "squeeze", "contiguous",
})

_RECIPES = {
    "square": _square,
    "reciprocal": _reciprocal,
    "cbrt": _cbrt,
    "float": _copy_first,
    "double": _copy_first,
    "to": _copy_first,
    "long": _trunc_first,
    "int": _trunc_first,
    **{name: _copy_first for name in VIEW_OPS},
}


def _op_name(instruction: Instr) -> str:
    # A tensor op may arrive wrapped as a Call carrying its canonical name under
    # ``tensor_operation`` (the shared LLVM/numeric lowering does this).
    if instruction.op in ("Call", "call"):
        tensor_operation = instruction.attributes.get("tensor_operation")
        if tensor_operation:
            return str(tensor_operation)
    return instruction.op


def lower_derived_ops_to_ssa(functions) -> None:
    """Decompose derived tensor ops into primitives across the SSA functions,
    in place. ``functions`` is a mapping of name -> repository SSA ``Function``.
    """

    for function in functions.values():
        next_id = -1
        for value in function.args:
            next_id = max(next_id, int(value.id))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    next_id = max(next_id, int(instruction.res.id))
                for argument in instruction.args:
                    next_id = max(next_id, int(argument.id))
        next_id += 1

        def fresh() -> int:
            nonlocal next_id
            value_id = next_id
            next_id += 1
            return value_id

        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                recipe = _RECIPES.get(_op_name(instruction))
                if (
                    recipe is not None
                    and instruction.res is not None
                    and instruction.args
                ):
                    rewritten.extend(
                        recipe(list(instruction.args), instruction.res, fresh)
                    )
                else:
                    rewritten.append(instruction)
            block.instrs = rewritten

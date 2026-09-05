"""Backend-neutral lowering of subscript ops to SSA address primitives.

``d[i]`` and ``d[i] = v`` arrive as the ops ``Indexed(base, index...) -> res``
and ``IndexedStore(base, index..., value) -> res``. Neither is a primitive any
backend should special-case: both are an *address* into ``base`` followed by a
load or a store. This module rewrites them to exactly that -- ``GetElementPtr``
(base + indices) then ``Load`` / ``Store`` -- the same address vocabulary the
repository SSA and every backend already speak (the WASM backend computes the
address from the same selectors; the Fortran backend renders ``GetElementPtr``+
``Load`` as ``base(i+1)`` and ``GetElementPtr``+``Store`` as ``base(i+1) = v``).
So the subscript lowering lives once, at the SSA level, not once per backend.

A store mutates ``base`` in place, so its result value (the "returned array")
aliases ``base``: uses of the result are rewritten to ``base`` rather than
carrying a fresh SSA name for the same storage.
"""
from __future__ import annotations

import dataclasses

from ..transmogrifier.ssa import Instr, SSAValue

_GATHER = ("Indexed", "gather")
_SCATTER = ("IndexedStore", "index_set")


def lower_indexing_to_ssa_addressing(functions) -> None:
    """Rewrite ``Indexed``/``IndexedStore`` ops to ``GetElementPtr``+``Load``/
    ``Store`` across the given SSA functions, in place.

    ``functions`` is a mapping of name -> repository SSA ``Function``.
    """

    # Address temporaries are minted above every id in the MODULE, not above
    # each function's own maximum. A planner region is carved out of its caller
    # and shares the caller's value space, so a per-function allocator hands a
    # region-internal address the very integer the caller uses for one of its
    # own scalars -- and the whole-program structural-output recovery, which
    # matches a caller's desired id against any id a callee produces, then
    # binds the caller's scalar to that address. That is how the fluid advance
    # read a height cell where ``tracer_diffusivity`` belonged: region_2's
    # address landed on 80, region_1's on 89 (same number, different space).
    # One module-wide watermark makes the collision unrepresentable.
    next_id = -1
    for function in functions.values():
        for value in function.args:
            next_id = max(next_id, int(value.id))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is not None:
                    next_id = max(next_id, int(instruction.res.id))
                for argument in instruction.args:
                    next_id = max(next_id, int(argument.id))
    next_id += 1

    def fresh() -> SSAValue:
        nonlocal next_id
        value = SSAValue(next_id)
        next_id += 1
        return value

    for function in functions.values():

        # base value each store's result aliases, so later uses read the same
        # storage the store mutated in place.
        aliases: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                canonical_op = str(
                    instruction.attributes.get("tensor_operation")
                    or instruction.attributes.get("tensor")
                    or instruction.op
                )
                if canonical_op in _GATHER and len(instruction.args) >= 2:
                    base, *indices = instruction.args
                    address = fresh()
                    rewritten.append(
                        Instr("GetElementPtr", [base, *indices], address)
                    )
                    rewritten.append(Instr("Load", [address], instruction.res))
                elif canonical_op in _SCATTER and len(instruction.args) >= 3:
                    base = instruction.args[0]
                    value = instruction.args[-1]
                    indices = instruction.args[1:-1]
                    address = fresh()
                    rewritten.append(
                        Instr("GetElementPtr", [base, *indices], address)
                    )
                    rewritten.append(Instr("Store", [value, address], None))
                    if instruction.res is not None:
                        aliases[int(instruction.res.id)] = base
                else:
                    rewritten.append(instruction)
            block.instrs = rewritten

        if not aliases:
            continue

        # A store's result is its mutated base: point every later use at
        # base. Bases CHAIN: when stores to one array are sequential in a
        # block -- which is what every unrolled loop and every size-baked
        # kernel produces -- store #2's base is store #1's result, so the
        # map holds 189 -> 146 -> 2. Resolving one level leaves a use
        # pointing at 146, a version id no instruction defines: the SSA is
        # then use-before-def and the evaluator (honestly) refuses while
        # some emitters (dishonestly) emitted it. Resolve every alias to
        # its ROOT storage.
        def root(value: SSAValue) -> SSAValue:
            seen: set[int] = set()
            while int(value.id) in aliases and int(value.id) not in seen:
                seen.add(int(value.id))
                value = aliases[int(value.id)]
            return value

        for block in function.blocks.values():
            for index, instruction in enumerate(block.instrs):
                if any(int(a.id) in aliases for a in instruction.args):
                    block.instrs[index] = dataclasses.replace(
                        instruction,
                        args=[root(a) if int(a.id) in aliases else a
                              for a in instruction.args],
                    )

    _propagate_scalar_dtypes(functions)


def _propagate_scalar_dtypes(functions) -> None:
    """Settle scalar contracts after structural indexing is expanded.

    ``Indexed`` knows the element dtype of its base, while the universal
    ``GetElementPtr`` temporary intentionally does not.  Preserve that fact
    across the rewrite, then carry casts and integer arithmetic through the
    one function's SSA namespace.  Caller/callee agreement is handled by the
    explicit call signature; a bare integer ID is never used to conflate two
    function-local values.  This is type accounting only: IDs and instruction
    order are unchanged.
    """

    integers = {"int", "int8", "int16", "int32", "int64", "i32", "i64"}
    floats = {"float", "float16", "float32", "float64", "double", "f32", "f64"}
    preserving = {
        "Add", "Sub", "Mul", "FloorDiv", "Mod", "Pow", "Min", "Max",
        "BitAnd", "BitOr", "BitXor", "Shl", "Shr", "Neg", "Abs",
    }
    predicates = {
        "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "ULt", "ULe",
        "LAnd", "LOr", "LNot", "LXor",
    }

    for function in functions.values():
        values: dict[int, list[SSAValue]] = {}
        instructions = []
        for value in function.args:
            values.setdefault(int(value.id), []).append(value)
        for block in function.blocks.values():
            for instruction in block.instrs:
                instructions.append(instruction)
                for value in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                ):
                    values.setdefault(int(value.id), []).append(value)

        dtype_of: dict[int, str] = {}
        for value_id, occurrences in values.items():
            declared = [str(value.dtype) for value in occurrences if value.dtype]
            if "int64" in declared or "i64" in declared:
                dtype_of[value_id] = "int64"
            elif declared:
                dtype_of[value_id] = declared[0]

        address_base: dict[int, int] = {}
        for instruction in instructions:
            if (
                instruction.op == "GetElementPtr"
                and instruction.res is not None
                and instruction.args
            ):
                address_base[int(instruction.res.id)] = int(
                    instruction.args[0].id
                )

        for _ in range(max(1, len(instructions))):
            changed = False
            for instruction in instructions:
                if instruction.res is None:
                    continue
                result_id = int(instruction.res.id)
                operand_dtypes = tuple(
                    dtype_of.get(int(value.id)) for value in instruction.args
                )
                inferred = None
                if instruction.op == "Const":
                    literal = instruction.attributes.get("value")
                    inferred = (
                        "bool" if isinstance(literal, bool)
                        else "int64" if isinstance(literal, int)
                        else "float64" if isinstance(literal, float)
                        else None
                    )
                elif instruction.op == "Cast":
                    inferred = instruction.attributes.get("target_dtype")
                elif instruction.op == "Load" and instruction.args:
                    declared = str(instruction.res.dtype or "")
                    if declared and "ptr" not in declared.casefold():
                        # LLVM opaque pointers do not state a pointee type.
                        # The Load result does, as does the original Indexed
                        # result retained by address lowering; never replace
                        # that explicit scalar contract with ``ptr``.
                        inferred = declared
                    else:
                        base_id = address_base.get(
                            int(instruction.args[0].id)
                        )
                        candidate = (
                            None if base_id is None
                            else dtype_of.get(base_id)
                        )
                        inferred = (
                            None
                            if candidate is not None
                            and "ptr" in str(candidate).casefold()
                            else candidate
                        )
                elif instruction.op == "BitLength":
                    inferred = "int64"
                elif instruction.op in predicates:
                    inferred = "bool"
                elif instruction.op in preserving and operand_dtypes and all(
                    candidate is not None for candidate in operand_dtypes
                ):
                    if any(candidate in floats for candidate in operand_dtypes):
                        inferred = "float64"
                    elif all(
                        candidate in integers for candidate in operand_dtypes
                    ):
                        inferred = "int64"
                elif instruction.op in {"Div", "Sqrt", "Exp", "Log"}:
                    inferred = "float64"
                if inferred in {
                    "int", "int8", "int16", "int32", "i32", "i64",
                }:
                    inferred = "int64"
                if inferred is not None and dtype_of.get(result_id) != inferred:
                    dtype_of[result_id] = str(inferred)
                    changed = True
            if not changed:
                break

        for value_id, dtype in dtype_of.items():
            for value in values.get(value_id, ()):
                value.dtype = dtype

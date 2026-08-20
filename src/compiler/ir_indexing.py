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

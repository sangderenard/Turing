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

        def fresh() -> SSAValue:
            nonlocal next_id
            value = SSAValue(next_id)
            next_id += 1
            return value

        # base value each store's result aliases, so later uses read the same
        # storage the store mutated in place.
        aliases: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                if instruction.op in _GATHER and len(instruction.args) >= 2:
                    base, *indices = instruction.args
                    address = fresh()
                    rewritten.append(
                        Instr("GetElementPtr", [base, *indices], address)
                    )
                    rewritten.append(Instr("Load", [address], instruction.res))
                elif instruction.op in _SCATTER and len(instruction.args) >= 3:
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
        # A store's result is its mutated base: point every later use at base.
        for block in function.blocks.values():
            for index, instruction in enumerate(block.instrs):
                if any(int(a.id) in aliases for a in instruction.args):
                    block.instrs[index] = dataclasses.replace(
                        instruction,
                        args=[aliases.get(int(a.id), a) for a in instruction.args],
                    )

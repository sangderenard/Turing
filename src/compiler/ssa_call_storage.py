"""Target-neutral storage classification for repository-SSA call arguments.

Backends must distinguish a scalar cell (which is converted to the callee's
declared dtype at a call) from a span (whose address is forwarded unchanged).
The evidence is repository SSA itself: explicit rank/accounting and ordinary
GetElementPtr use, propagated positionally through the call graph.  Value ids
remain function-local throughout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def call_array_argument_ids(
    functions: Mapping[str, object],
    function_names: Iterable[str] | None = None,
) -> dict[str, frozenset[int]]:
    """Return array-valued formal ids for each selected function.

    This is the call-argument portion of the array-contract analysis used by
    the Fortran backend.  It intentionally does not infer from dtype: scalar
    and span values may share a dtype, while a rankless helper formal becomes
    a span as soon as its body indexes it.
    """

    selected = tuple(
        str(name) for name in (
            function_names if function_names is not None else functions
        )
        if str(name) in functions
    )
    arrays: dict[str, set[int]] = {name: set() for name in selected}

    for name in selected:
        function = functions[name]
        held = arrays[name]
        held.update(
            int(argument.id)
            for argument in function.args
            if (
                tuple(argument.shape or ())
                or str((argument.accounting or {}).get("program_abi_storage"))
                == "span"
                or int((argument.accounting or {}).get("program_abi_rank", 0) or 0) > 0
                or int((argument.accounting or {}).get("ssa_call_rank", 0) or 0) > 0
            )
        )
        held.update(map(
            int, function.metadata.get("sequence_array_argument_ids", ()),
        ))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op in {"GetElementPtr", "getelementptr"}
                    and instruction.args
                    and instruction.attributes.get("aggregate_index") is None
                ):
                    held.add(int(instruction.args[0].id))

    changed = True
    while changed:
        changed = False
        for caller_name in selected:
            caller = functions[caller_name]
            caller_arrays = arrays[caller_name]
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in {"Call", "call"}:
                        continue
                    callee_name = str(
                        instruction.attributes.get("callee") or ""
                    )
                    if callee_name not in arrays:
                        continue
                    callee = functions[callee_name]
                    callee_arrays = arrays[callee_name]
                    for actual, formal in zip(instruction.args, callee.args):
                        actual_id = int(actual.id)
                        formal_id = int(formal.id)
                        if (
                            actual_id not in caller_arrays
                            and formal_id not in callee_arrays
                        ):
                            continue
                        if actual_id not in caller_arrays:
                            caller_arrays.add(actual_id)
                            changed = True
                        if formal_id not in callee_arrays:
                            callee_arrays.add(formal_id)
                            changed = True

    return {name: frozenset(values) for name, values in arrays.items()}

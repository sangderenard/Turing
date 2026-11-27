"""Which SSA functions can leave through a fault exit, and which callers follow.

A checked fault records its validation code and leaves its function through
the one ``Trap`` marked ``fault_exit`` (``precompile_to_ssa._finish_fault_exits``;
the runtime broadcast guards of ``tensor_ssa_lowering``).  The exit publishes
no outputs.

Program: declared ``Sum(k * x, (k, 1, n))`` over a tensor ``x`` (LLVM, O3).
Wrong record: planned region 6's runtime broadcast guard failed (code 100)
and left the region, but its owner went on to call regions 7 and 8 on the
shape, rank and heap cells region 6 never wrote.  Reading that unwritten
stack storage is undefined, and the loops sized from it never finished (2.5
CPU-hours on one call); at O0 the same garbage happened to end and Python
raised code 100 afterwards.  Rule: a fault exit leaves the whole SSA call
chain, as a Python raise unwinds.  After a call to a function that can fault
exit, the caller continues only when no fault is pending, and otherwise
leaves through its own fault exit.

A function that handles exceptions itself (it takes the active record,
``ExceptionTake``) keeps its authored guards: its protected statements read
the record and select the handler, so the call there must continue.
"""

from __future__ import annotations

from typing import Any


def is_fault_exit(instruction: Any) -> bool:
    return (
        str(getattr(instruction, "op", "")) == "Trap"
        and bool((getattr(instruction, "attributes", None) or {}).get("fault_exit"))
    )


def handles_faults(function: Any) -> bool:
    """The function takes the active exception record (an authored try)."""

    return any(
        str(instruction.op) == "ExceptionTake"
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def fault_exit_functions(module: Any) -> frozenset[str]:
    """Functions that can leave through a fault exit, directly or by a call.

    A caller that handles faults itself does not propagate them, so it is a
    fault-exit function only through its own ``Trap``.
    """

    functions = getattr(module, "functions", {}) or {}
    exits: set[str] = set()
    callees: dict[str, set[str]] = {}
    handlers: set[str] = set()
    for name, function in functions.items():
        called: set[str] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                if is_fault_exit(instruction):
                    exits.add(str(name))
                elif str(instruction.op) in {"Call", "call"}:
                    callee = (instruction.attributes or {}).get("callee")
                    if callee is not None and str(callee) in functions:
                        called.add(str(callee))
                elif str(instruction.op) == "ExceptionTake":
                    handlers.add(str(name))
        callees[str(name)] = called
    changed = True
    while changed:
        changed = False
        for name, called in callees.items():
            if name in exits or name in handlers:
                continue
            if called & exits:
                exits.add(name)
                changed = True
    return frozenset(exits)


def propagates_fault_after_call(
    module: Any, caller: str, callee: str,
    exits: frozenset[str] | None = None,
) -> bool:
    """The caller must leave after this call when a fault is pending."""

    functions = getattr(module, "functions", {}) or {}
    exits = fault_exit_functions(module) if exits is None else exits
    function = functions.get(caller)
    return (
        str(callee) in exits
        and function is not None
        and not handles_faults(function)
    )

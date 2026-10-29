"""Where a promoted numeric join may be read.

Program: ``if c == 2: s = n else: s = n * 0.5`` (corpus c770).  Python
never promotes at a join: ``s`` is an int on one path and a float on the
other.  The SSA join carries one physical class (the widest float, or
int64 for ints and bools), converting the narrower arm on its own edge
(``_ControlSSABuilder._numeric_join_arms``), and marks the Phi
``numeric_join_promoted``.  That is exact only where every use of the
joined value gives the same result for either arm's class: arithmetic
with an operand of the join's class that is not itself a join, which
converts the narrower operand the same way Python does (``s + 0.25``,
``acc += s``; ``s + s`` is int + int on one path).  A ``For`` header that
carries its update's class because the loop provably runs is such a join
inside the body (the seed on the first trip) and exact at the exit when
its update is not a join.  So is the promoted read view of a numeric
union call result (``_promote_union_call_result``: a ``Select`` on the tag
over each arm's own payload), which is followed like a Phi.  Anything else --
the value leaving the program, a comparison, ``//`` or ``%`` with an int,
a store, ``type()`` -- can tell the classes apart, so it is refused by
name.  A join or call result that leaves the program, or a function, as
its value-union frame keeps each arm's exact class, and so does a region
the tag dispatches per arm (``_emit_region_call_by_arm``: each copy reads
its own arm's payload); the readers left here are the ones no per-arm copy
covers.
"""

from __future__ import annotations

from typing import Any


PROMOTED = "numeric_join_promoted"

_ARITHMETIC_OPS = {"Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "Pow"}
_PASS_THROUGH_OPS = {"Phi", "Cast"}
_NUMERIC_CLASS = {
    "bool": "bool", "int": "int", "int32": "int", "int64": "int",
    "float": "float", "float16": "float", "float32": "float",
    "float64": "float", "double": "float",
}


def refuse_type_visible_numeric_joins(module: Any) -> int:
    """Raise when a promoted join reaches a use that can see its class.

    Returns how many promoted joins were followed.  Values are followed
    through Phi and Cast, into callees by argument position, and out of a
    callee's ``Ret`` into each caller's projection of the call aggregate.
    A ``Ret`` of a function no other function calls is a program output.
    """

    functions = dict(getattr(module, "functions", module) or {})
    callers: dict[str, list[tuple[str, Any]]] = {}
    for name, function in functions.items():
        for block in function.blocks.values():
            for instruction in block.instrs:
                callee = instruction.attributes.get("callee")
                if instruction.op == "Call" and callee in functions:
                    callers.setdefault(str(callee), []).append(
                        (name, instruction)
                    )

    def producers(function: Any) -> dict[int, Any]:
        return {
            int(instruction.res.id): instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }

    def uses(function: Any, value_id: int):
        for block in function.blocks.values():
            for instruction in block.instrs:
                if any(int(arg.id) == value_id for arg in instruction.args):
                    yield instruction

    def _refuse(function_name: str, origin: tuple[str, int], why: str):
        raise ValueError(
            f"a join of int and float arms ({origin[0]} value {origin[1]}) "
            f"{why} in {function_name!r}; Python keeps each arm's class "
            "there, and the int-or-float union lowering is not built"
        )

    roots = [
        (name, int(instruction.res.id), str(instruction.res.dtype))
        for name, function in functions.items()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get(PROMOTED)
        and instruction.res is not None
    ]
    # A ``For`` header typed with its update's class because the loop
    # provably runs (``_carried_run_dtype``): the seed's class is visible
    # only to body reads on the first trip.  Its exit is exact when its
    # update is not itself a join (see ``tainted``).
    run_headers = {
        (name, int(instruction.res.id)): int(instruction.args[1].id)
        for name, function in functions.items()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.attributes.get(PROMOTED)
        and instruction.attributes.get("binding") == "loop_carried"
        and instruction.res is not None and len(instruction.args) == 2
    }

    def walk(tainted: set[tuple[str, int]] | None) -> set[tuple[str, int]]:
        """Follow every join; with ``tainted``, refuse class-visible uses.

        Without it, only collect what every join reaches, so that an
        arithmetic use whose other operand is itself a join (``s + s``:
        int + int on one path) is not mistaken for a converting one.
        """

        reached: set[tuple[str, int]] = set()
        checking = tainted is not None

        def refuse(function_name, origin, why):
            if checking:
                _refuse(function_name, origin, why)

        for origin_function, origin_id, join_dtype in roots:
            join_class = _NUMERIC_CLASS.get(join_dtype)
            origin = (origin_function, origin_id)
            exact_exit = (
                checking and origin in run_headers
                and (origin_function, run_headers[origin]) not in tainted
            )
            pending = [(origin_function, origin_id)]
            seen: set[tuple[str, int]] = set()
            while pending:
                function_name, value_id = pending.pop()
                if (function_name, value_id) in seen:
                    continue
                seen.add((function_name, value_id))
                function = functions[function_name]
                for instruction in uses(function, value_id):
                    op = str(instruction.op)
                    if (
                        exact_exit and (function_name, value_id) == origin
                        and instruction.attributes.get("binding")
                        == "loop_result_port"
                    ):
                        # The loop provably ran and its update is exact:
                        # the exit is the update's class, the header's.
                        continue
                    if (
                        op in _PASS_THROUGH_OPS
                        or instruction.attributes.get("binding") in {
                            "numeric_union_call_result",
                            "numeric_union_arm_result",
                        }
                    ) and instruction.res is not None:
                        pending.append((function_name, int(instruction.res.id)))
                    elif op in _ARITHMETIC_OPS:
                        others = [
                            arg for arg in instruction.args
                            if int(arg.id) != value_id
                        ]
                        if len(others) != 1 or _NUMERIC_CLASS.get(
                            str(others[0].dtype)
                        ) != join_class or (
                            checking
                            and (function_name, int(others[0].id)) in tainted
                        ):
                            refuse(
                                function_name, origin,
                                f"is an operand of {op} with an operand not "
                                f"of its class {join_dtype!r}",
                            )
                    elif op == "Call" and instruction.attributes.get(
                        "union_arm_dispatch"
                    ):
                        # Called only where the tag names this payload's
                        # arm: the payload is that arm's exact value there.
                        continue
                    elif op == "Call":
                        callee = str(instruction.attributes.get("callee"))
                        if callee not in functions:
                            refuse(function_name, origin,
                                   f"is passed to {callee!r}")
                            continue
                        formals = functions[callee].args
                        for position, arg in enumerate(instruction.args):
                            if int(arg.id) == value_id:
                                pending.append(
                                    (callee, int(formals[position].id))
                                )
                    elif op == "Ret":
                        sites = callers.get(function_name, [])
                        if not sites:
                            refuse(function_name, origin,
                                   "is returned from the program")
                        slots = {
                            position for position, arg in enumerate(
                                instruction.args
                            ) if int(arg.id) == value_id
                        }
                        for caller_name, call in sites:
                            caller = functions[caller_name]
                            if call.res is None:
                                continue
                            if str(call.res.dtype) != "ssa.aggregate":
                                # A one-result call is its result.
                                if slots == {0}:
                                    pending.append(
                                        (caller_name, int(call.res.id))
                                    )
                                continue
                            made = producers(caller)
                            for load in caller.blocks.values():
                                for read in load.instrs:
                                    if read.op != "Load" or read.res is None:
                                        continue
                                    address = made.get(int(read.args[0].id))
                                    if (
                                        address is None
                                        or address.op != "GetElementPtr"
                                        or int(address.args[0].id)
                                        != int(call.res.id)
                                    ):
                                        continue
                                    index = made.get(int(address.args[1].id))
                                    slot = (
                                        None if index is None
                                        else index.attributes.get("value")
                                    )
                                    if slot is None:
                                        refuse(caller_name, origin,
                                               "is read at an unproven slot")
                                        continue
                                    if int(slot) in slots:
                                        pending.append(
                                            (caller_name, int(read.res.id))
                                        )
                    else:
                        refuse(function_name, origin, f"is read by {op}")
            reached |= seen
        return reached

    walk(walk(None))
    return len(roots)

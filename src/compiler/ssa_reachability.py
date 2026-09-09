"""Remove proven dead CFG edges without treating unused calls as pure.

This pass deliberately leaves signatures alone. The repository linker owns
the transaction that removes unused captures and their actual call operands.
"""

from __future__ import annotations

from collections import Counter


def hoist_nondominating_constants(function) -> tuple[dict[str, object], ...]:
    """Place unique operand-free constants before every reachable use.

    Region/control reconstruction can attach a literal definition to a late
    exit compartment even though the same immutable SSA object is consumed on
    an earlier loop path.  Constants have no operands or effects, so moving a
    uniquely-defined object to entry preserves its value and makes its lifetime
    explicit.  Reused numeric identities remain untouched because their
    meaning is ambiguous and must be freshened by the identity passes.
    """
    blocks = function.blocks
    if not blocks:
        return ()
    entry_name = "entry" if "entry" in blocks else next(iter(blocks))
    reachable, pending = set(), [entry_name]
    predecessors = {name: set() for name in blocks}
    while pending:
        name = pending.pop()
        if name in reachable or name not in blocks:
            continue
        reachable.add(name)
        for successor in blocks[name].successors:
            if successor in blocks:
                predecessors[successor].add(name)
                pending.append(successor)
    dominators = {
        name: ({entry_name} if name == entry_name else set(reachable))
        for name in reachable
    }
    changed = True
    while changed:
        changed = False
        for name in reachable - {entry_name}:
            incoming = predecessors[name] & reachable
            common = (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            updated = common | {name}
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    definitions = Counter(
        int(instruction.res.id)
        for block in blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    )
    formals = {int(value.id) for value in function.args}
    candidates = []
    for block_name, block in blocks.items():
        if block_name not in reachable or block_name == entry_name:
            continue
        for index, instruction in enumerate(block.instrs):
            if (
                str(instruction.op).casefold() != "const"
                or instruction.args
                or instruction.res is None
                or int(instruction.res.id) in formals
                or definitions[int(instruction.res.id)] != 1
            ):
                continue
            result = instruction.res
            nondominating_use = False
            for use_block_name in reachable:
                use_block = blocks[use_block_name]
                for use_index, use in enumerate(use_block.instrs):
                    for position, argument in enumerate(use.args):
                        if argument is not result:
                            continue
                        at_block, at_index = use_block_name, use_index
                        if str(use.op).casefold() == "phi":
                            incoming = tuple((use.attributes or {}).get(
                                "incoming_blocks", ()
                            ))
                            if position >= len(incoming):
                                continue
                            at_block = incoming[position]
                            at_index = len(blocks.get(at_block, ()).instrs) if at_block in blocks else 0
                        if (
                            block_name not in dominators.get(at_block, set())
                            or (block_name == at_block and index >= at_index)
                        ):
                            nondominating_use = True
                            break
                    if nondominating_use:
                        break
                if nondominating_use:
                    break
            if nondominating_use:
                candidates.append((block_name, index, instruction))

    if not candidates:
        return ()
    for block_name, _index, instruction in candidates:
        blocks[block_name].instrs.remove(instruction)
    entry = blocks[entry_name]
    insertion_index = 0
    while (
        insertion_index < len(entry.instrs)
        and str(entry.instrs[insertion_index].op).casefold() in {
            "const", "nonevalue"
        }
        and not entry.instrs[insertion_index].args
    ):
        insertion_index += 1
    entry.instrs[insertion_index:insertion_index] = [
        instruction for _block, _index, instruction in candidates
    ]
    receipts = tuple({
        "value_id": int(instruction.res.id),
        "from_block": str(block_name),
        "from_index": int(index),
        "to_block": str(entry_name),
        "priority": "operand_free_immutable_definition",
        "tie_policy": "incumbent",
    } for block_name, index, instruction in candidates)
    prior = tuple(function.metadata.get("constant_hoist_receipts", ()))
    function.metadata["constant_hoist_receipts"] = (*prior, *(
        receipt for receipt in receipts if receipt not in prior
    ))
    return receipts


def prune_constant_control_flow(function) -> int:
    """Fold scalar Boolean branches and repair predecessor-labelled Phis.

Only literal scalars and Boolean operations are evaluated. In particular,
loads, calls, and record projections are never interpreted as constants.
Instructions in reachable blocks retain their order and effects.
"""
    blocks = function.blocks
    if not blocks:
        return 0
    entry = "entry" if "entry" in blocks else next(iter(blocks))
    changes = 0
    while True:
        constants = {}
        definitions = Counter(
            instruction.res.id
            for block in blocks.values() for instruction in block.instrs
            if instruction.res is not None
        )
        mutable = {
            value.id for value in function.args
            if (value.accounting or {}).get("program_abi_mutable")
        }
        # Iterate because block insertion order need not be dominance order.
        while True:
            added = False
            for block in blocks.values():
                for instruction in block.instrs:
                    result = instruction.res
                    if (result is None or result.shape or result.id in constants
                            or definitions[result.id] != 1 or result.id in mutable
                            or any(arg.shape for arg in instruction.args)):
                        continue
                    attrs = instruction.attributes or {}
                    known = [constants.get(arg.id) for arg in instruction.args]
                    value = None
                    if instruction.op == "Const":
                        literal = attrs.get("value")
                        if type(literal) in (bool, int, float):
                            value = literal
                    elif instruction.op == "Phi" and known and all(
                        item is not None for item in known
                    ):
                        if all(type(item) is type(known[0]) and item == known[0]
                               for item in known):
                            value = known[0]
                    elif instruction.op in {"LAnd", "LOr"} and known:
                        absorbing = instruction.op == "LOr"
                        if any(item is not None and bool(item) == absorbing
                               for item in known):
                            value = absorbing
                        elif all(item is not None for item in known):
                            value = not absorbing
                    elif instruction.op == "LNot" and len(known) == 1:
                        if known[0] is not None:
                            value = not known[0]
                    if value is not None:
                        constants[result.id] = value
                        added = True
            if not added:
                break

        changed = 0
        for block in blocks.values():
            if not block.instrs:
                continue
            terminator = block.instrs[-1]
            attrs = terminator.attributes or {}
            if (terminator.op == "CondBr" and len(terminator.args) == 1
                    and terminator.args[0].id in constants):
                target = attrs["true_target" if bool(
                    constants[terminator.args[0].id]
                ) else "false_target"]
                terminator.op = "Br"
                terminator.args = []
                terminator.arg_roles = []
                terminator.attributes = {"target": target}
                changed += 1
            if terminator.op == "Br":
                block.successors = [terminator.attributes["target"]]
            elif terminator.op == "CondBr":
                block.successors = [attrs["true_target"], attrs["false_target"]]
            elif terminator.op == "Ret":
                block.successors = []

        reachable = set()
        pending = [entry]
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            if name not in blocks:
                raise ValueError(f"{function.name}: missing CFG target {name!r}")
            reachable.add(name)
            pending.extend(blocks[name].successors)
        dead = set(blocks) - reachable
        for name in dead:
            del blocks[name]
        changed += len(dead)
        predecessors = {name: set() for name in blocks}
        for name, block in blocks.items():
            for successor in block.successors:
                predecessors[successor].add(name)
        for name, block in blocks.items():
            for instruction in block.instrs:
                if instruction.op != "Phi":
                    continue
                incoming = (instruction.attributes or {}).get("incoming_blocks")
                if incoming is None:
                    continue
                if len(incoming) != len(instruction.args):
                    raise ValueError(f"{function.name}: malformed Phi in {name}")
                receivers = instruction.attributes.get("record_return_receivers")
                if receivers is not None and len(receivers) != len(incoming):
                    raise ValueError(f"{function.name}: malformed record Phi provenance in {name}")
                keep = [index for index, parent in enumerate(incoming)
                        if parent in predecessors[name]]
                if len(keep) == len(incoming):
                    continue
                if not keep:
                    raise ValueError(f"{function.name}: Phi without incoming edge in {name}")
                instruction.args = [instruction.args[index] for index in keep]
                instruction.attributes["incoming_blocks"] = tuple(incoming[index] for index in keep)
                if receivers is not None:
                    instruction.attributes["record_return_receivers"] = tuple(receivers[index] for index in keep)
                if instruction.attributes.get("record_return_scalar"):
                    instruction.attributes["initial_value_id"] = int(instruction.args[0].id)
                if instruction.arg_roles:
                    instruction.arg_roles = [instruction.arg_roles[index] for index in keep]
                changed += 1
        changes += changed
        if not changed:
            return changes

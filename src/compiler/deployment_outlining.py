"""Outline proven independent-iteration deployment lanes into callable closures.

``precompile_to_ssa`` mints a retained loop's deployment permission as ONE
lane: lane zero is the SSA template for every independent iteration, and the
region's ``iteration_space`` says how to fan it out.  Backends cannot deploy
that template while it lives inline in the parent's loop body -- there is no
callable to hand a worker.  This pass makes the deployment demand concrete
*in SSA*, where every backend sees it:

- the lane's body blocks move into a new module function whose formals are
  the lane's live-ins (the induction value first);
- the parent's loop body becomes one internal ``Call`` to that function, so
  the serial instruction stream remains a byte-equivalent fallback;
- the region record and ``module.metadata["deployment_outlines"]`` carry the
  outline so a backend can replace the whole loop execution with a native
  span deploy (``turing_pool_deploy_span``) and jump to the join.

The pass PROVES before it moves.  A lane is refused, with a named reason,
when the honest checks fail:

- the loop shape is not the canonical header(Phi/compare/CondBr) +
  latch(Add/Br) counted form;
- a lane-defined value is consumed outside the lane (live-outs need an
  aggregate return lowering that does not exist yet);
- a lane instruction mutates a sequence whose handle lives outside the lane
  (append order is iteration order; an ordered-join lowering is required);
- a lane ``Store`` targets memory rooted outside the lane without the
  induction in its address chain (an unproven shared-store shuffle).

Refusals never mutate anything.  They are the deployment counterpart of
every backend's shortfall dataclass: the exact next compiler frontier,
named, instead of a silently serial product.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from ..transmogrifier.ssa import BasicBlock, Function, Instr

#: Sequence helper verbs that only read; anything else on a shared handle is
#: an ordered mutation the barrier join cannot legalize.
_SEQUENCE_READ_VERBS = frozenset({
    "lookup", "lookup_or_default", "get", "getitem", "length", "contains",
    "read", "peek",
})


@dataclass(frozen=True)
class LaneOutlineRecord:
    """One outlined lane, with everything a backend needs to deploy it."""

    function: str
    region_id: int
    outline_name: str
    induction_id: int
    start_id: int
    stop_id: int
    step_id: int
    argument_ids: tuple[int, ...]
    entry_block: str
    header_block: str
    exit_block: str
    continuation_block: str
    comparison: str
    #: Blocks the backend must wrap in the pool's effect lock: they hold an
    #: order-insensitive shared append (lane-invariant operands only).
    guarded_blocks: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "region_id": self.region_id,
            "outline": self.outline_name,
            "induction": self.induction_id,
            "iteration_space": [self.start_id, self.stop_id, self.step_id],
            "arguments": list(self.argument_ids),
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class OutlineReport:
    outlined: tuple[LaneOutlineRecord, ...]
    refused: tuple[tuple[str, int, str], ...]

    @property
    def complete(self) -> bool:
        return not self.refused


def _sequence_helper_verb(callee: str) -> tuple[int, str] | None:
    """``(sequence_id, verb)`` when *callee* is a sequence helper."""

    parts = str(callee).split("_")
    # ssa_sequence_<id>_<verb...>
    if len(parts) >= 4 and parts[0] == "ssa" and parts[1] == "sequence":
        try:
            return int(parts[2]), "_".join(parts[3:])
        except ValueError:
            return None
    return None


def _instruction_stream(function: Function):
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            yield block_name, index, instruction


def outline_independent_iteration_lanes(module: Any) -> OutlineReport:
    """Outline every provable single-lane iteration region in *module*."""

    outlined: list[LaneOutlineRecord] = []
    refused: list[tuple[str, int, str]] = []
    outlines = module.metadata.setdefault("deployment_outlines", {})
    for function_name in list(module.deployment_table):
        function = module.functions.get(str(function_name))
        if function is None:
            continue
        regions = list(module.deployment_table[function_name])
        for position, region in enumerate(regions):
            if (function_name, int(region.region_id)) in outlines:
                continue
            if (
                str(region.schedule) != "independent_iterations"
                or len(region.lanes) != 1
                or region.iteration_space is None
            ):
                continue
            result = _outline_one(module, function, region)
            if isinstance(result, str):
                refused.append((str(function_name), int(region.region_id), result))
                continue
            record, replacement_region = result
            regions[position] = replacement_region
            module.deployment_table[function_name] = tuple(regions)
            outlines[(str(function_name), int(region.region_id))] = record
            outlined.append(record)
    return OutlineReport(tuple(outlined), tuple(refused))


def _outline_one(
    module: Any, function: Function, region: Any,
) -> tuple[LaneOutlineRecord, Any] | str:
    region_id = int(region.region_id)
    lane_index = int(region.lanes[0].index)

    # -- membership is read from the live instruction stream, never from the
    # possibly stale sites on the record.
    member_instrs: set[int] = set()
    member_blocks: dict[str, list[int]] = {}
    deploy_site = None
    for block_name, index, instruction in _instruction_stream(function):
        attributes = instruction.attributes or {}
        if (
            attributes.get("deployment_frame")
            and instruction.op == "Deploy"
            and int(attributes.get("region_id", -1)) == region_id
        ):
            deploy_site = (block_name, index)
        memberships = attributes.get("deployment_memberships") or ()
        if any(
            int(mr) == region_id and int(ml) == lane_index
            for mr, ml in memberships
        ):
            member_instrs.add(id(instruction))
            member_blocks.setdefault(block_name, []).append(index)
    if deploy_site is None:
        return "region has no Deploy marker in the emitted stream"
    if not member_blocks:
        return "region lane has no member instructions"

    # -- every member block must be wholly lane-owned. Constant hoisting
    # moves pure ``Const`` members out of the loop while keeping their
    # membership attribute; those are loop-invariant live-ins, not lane
    # body, so they are reclassified rather than refused.
    for block_name in list(member_blocks):
        indices = member_blocks[block_name]
        block = function.blocks[block_name]
        if len(indices) == len(block.instrs):
            continue
        stray = [block.instrs[index] for index in indices]
        if all(instruction.op in {"Const", "const"} for instruction in stray):
            for instruction in stray:
                member_instrs.discard(id(instruction))
            del member_blocks[block_name]
            continue
        return (
            f"block {block_name!r} mixes lane and non-lane instructions; "
            "partial-block outlining is not implemented"
        )
    if not member_blocks:
        return "region lane holds only hoisted constants"

    # -- canonical counted-loop shape discovery.
    deploy_block = function.blocks[deploy_site[0]]
    terminator = deploy_block.instrs[-1]
    if terminator.op not in {"Br", "br"}:
        return "deploy block does not fall through to a loop header"
    header_name = str(terminator.attributes.get("target"))
    header = function.blocks.get(header_name)
    if header is None or len(header.instrs) != 3:
        return "loop header is not the canonical Phi/compare/CondBr triple"
    phi, compare, cond = header.instrs
    if phi.op not in {"Phi", "phi"} or phi.res is None:
        return "loop header does not begin with the induction Phi"
    if compare.op not in {"Lt", "Gt"} or compare.res is None:
        return "loop condition is not a Lt/Gt comparison"
    if cond.op not in {"CondBr", "condbr"}:
        return "loop header does not end with CondBr"
    body_entry = str(cond.attributes.get("true_target"))
    exit_block_name = str(cond.attributes.get("false_target"))
    if body_entry not in member_blocks:
        return "loop body entry is not lane-owned"
    incoming = tuple(phi.attributes.get("incoming_blocks") or ())
    if len(incoming) != 2 or len(phi.args) != 2:
        return "induction Phi does not have exactly two incoming edges"
    if incoming[0] == deploy_site[0]:
        start_value, latch_value = phi.args
        latch_name = str(incoming[1])
    elif incoming[1] == deploy_site[0]:
        latch_value, start_value = phi.args
        latch_name = str(incoming[0])
    else:
        return "induction Phi has no incoming edge from the deploy block"
    latch = function.blocks.get(latch_name)
    if latch is None or len(latch.instrs) != 2:
        return "loop latch is not the canonical Add/Br pair"
    advance, latch_br = latch.instrs
    if (
        advance.op != "Add" or advance.res is None
        or int(advance.res.id) != int(latch_value.id)
        or not advance.args
        or int(advance.args[0].id) != int(phi.res.id)
    ):
        return "loop latch does not advance the induction by a step"
    step_value = advance.args[1]
    if str(compare.op) != "Lt":
        return "only ascending (Lt) iteration spaces are outlined"
    stop_value = compare.args[1]
    if int(compare.args[0].id) != int(phi.res.id):
        return "loop condition does not test the induction"

    exit_block = function.blocks.get(exit_block_name)
    if exit_block is None:
        return "loop exit block is missing"
    if any(ins.op in {"Phi", "phi"} for ins in exit_block.instrs):
        return "loop exit carries Phi values; the pooled jump would skip them"

    # -- lane-defined values and their outside consumers.
    defined_ids: set[int] = set()
    for block_name in member_blocks:
        for instruction in function.blocks[block_name].instrs:
            if instruction.res is not None:
                defined_ids.add(int(instruction.res.id))
    for block_name, _index, instruction in _instruction_stream(function):
        if id(instruction) in member_instrs:
            continue
        for argument in instruction.args:
            if int(argument.id) in defined_ids:
                return (
                    f"lane value %t{argument.id} is consumed outside the "
                    "lane; live-out outlining is not implemented"
                )
    # The induction and condition must stay loop-internal for the pooled
    # jump to the exit to be safe.
    loop_internal = set(member_blocks) | {header_name, latch_name}
    for check_id, label in (
        (int(phi.res.id), "induction"),
        (int(compare.res.id), "loop condition"),
    ):
        for block_name, _index, instruction in _instruction_stream(function):
            if block_name in loop_internal:
                continue
            if any(int(argument.id) == check_id for argument in instruction.args):
                return f"{label} %t{check_id} is used outside the loop"

    # -- side-effect gates.
    #
    # A shared-sequence append whose every operand is lane-INVARIANT pushes
    # an identical element on every passing iteration: only the count is
    # observable, so atomicity alone (the pool's effect lock) preserves
    # serial semantics and the block is marked as a guarded critical
    # section.  A lane-dependent append stays refused -- its order IS
    # iteration order and needs an indexed/ordered join lowering.
    guarded_blocks: list[str] = []
    guarded_result_ids: dict[str, set[int]] = {}
    for block_name in member_blocks:
        for instruction in function.blocks[block_name].instrs:
            if instruction.op in {"Call", "call"}:
                callee = str((instruction.attributes or {}).get("callee") or "")
                helper = _sequence_helper_verb(callee)
                if helper is not None:
                    sequence_id, verb = helper
                    if verb in _SEQUENCE_READ_VERBS:
                        continue
                    if sequence_id in defined_ids:
                        continue  # lane-local sequence
                    if (
                        verb == "append"
                        and all(
                            int(argument.id) not in defined_ids
                            for argument in instruction.args
                        )
                        and not _lane_touches_sequence_elsewhere(
                            function, member_blocks, sequence_id, instruction,
                        )
                    ):
                        if block_name not in guarded_blocks:
                            guarded_blocks.append(block_name)
                        if instruction.res is not None:
                            guarded_result_ids.setdefault(
                                block_name, set(),
                            ).add(int(instruction.res.id))
                        continue
                    return (
                        f"lane mutates shared sequence %t{sequence_id} via "
                        f"{callee!r}; append order is iteration order, so an "
                        "ordered-join lowering is required"
                    )
            if instruction.op in {"Store", "store"} and len(instruction.args) >= 2:
                if block_name in guarded_blocks:
                    stored = instruction.args[0]
                    if (
                        int(stored.id) not in defined_ids
                        or int(stored.id)
                        in guarded_result_ids.get(block_name, ())
                    ):
                        # Inside the critical section, storing an invariant
                        # or the append's own result is last-write-wins with
                        # a deterministic final value (the final count).
                        continue
                if not _store_is_iteration_private(
                    function, member_instrs, instruction,
                    defined_ids, int(phi.res.id),
                ):
                    address = instruction.args[1]
                    return (
                        f"lane store through %t{address.id} targets shared "
                        "memory without the induction in its address chain"
                    )

    # -- continuation: the single non-member target the lane exits to.
    continuations: set[str] = set()
    for block_name in member_blocks:
        for target in _branch_targets(function.blocks[block_name].instrs[-1]):
            if target not in member_blocks:
                continuations.add(target)
    if len(continuations) != 1:
        return (
            "lane does not exit to exactly one continuation block: "
            f"{sorted(continuations)!r}"
        )
    continuation = continuations.pop()

    # -- live-ins, induction first.
    live_ids: list[int] = [int(phi.res.id)]
    seen: set[int] = set(live_ids)
    for block_name in member_blocks:
        for instruction in function.blocks[block_name].instrs:
            for argument in instruction.args:
                value_id = int(argument.id)
                if value_id in defined_ids or value_id in seen:
                    continue
                seen.add(value_id)
                live_ids.append(value_id)
    value_index = {int(value.id): value for value in function.args}
    for _block, _index, instruction in _instruction_stream(function):
        if instruction.res is not None:
            value_index.setdefault(int(instruction.res.id), instruction.res)
    formals = [value_index[value_id] for value_id in live_ids]

    # -- build the outlined function; blocks MOVE, they are not copied.
    outline_name = f"{function.name}__deploy_region_{region_id}_lane{lane_index}"
    if outline_name in module.functions:
        return f"outline name {outline_name!r} already exists"
    ordered_members = [
        name for name in function.blocks if name in member_blocks
    ]
    ordered_members.remove(body_entry)
    ordered_members.insert(0, body_entry)
    outline_blocks: dict[str, BasicBlock] = {}
    for name in ordered_members:
        block = function.blocks[name]
        _retarget(block.instrs[-1], continuation, "lane_return")
        block.successors = [
            "lane_return" if successor == continuation else successor
            for successor in block.successors
        ]
        outline_blocks[name] = block
    outline_blocks["lane_return"] = BasicBlock(
        "lane_return", [Instr("Ret", [], None)], [],
    )
    outline = Function(
        outline_name,
        formals,
        outline_blocks,
        metadata={
            "argument_names": tuple(
                f"lane_arg{index}" for index in range(len(formals))
            ),
            "output_names": (),
            "deployment_outline_of": (function.name, region_id),
            "pool_effect_guarded_blocks": tuple(guarded_blocks),
        },
    )
    module.functions[outline_name] = outline
    for table_name in (
        "tensor_tables", "sequence_tables", "record_tables",
        "reference_tables",
    ):
        table = getattr(module, table_name, None)
        if isinstance(table, dict) and function.name in table:
            table[outline_name] = table[function.name]

    # -- the parent's lane collapses to one call plus the continuation edge.
    call = Instr(
        "Call",
        formals,
        None,
        attributes={
            "callee": outline_name,
            "deployment_memberships": ((region_id, lane_index),),
            "deployment_outline": True,
        },
    )
    branch = Instr("Br", [], None, attributes={"target": continuation})
    function.blocks[body_entry] = BasicBlock(
        body_entry, [call, branch], [continuation],
    )
    for name in ordered_members:
        if name != body_entry:
            del function.blocks[name]

    record = LaneOutlineRecord(
        function=function.name,
        region_id=region_id,
        outline_name=outline_name,
        induction_id=int(phi.res.id),
        start_id=int(start_value.id),
        stop_id=int(stop_value.id),
        step_id=int(step_value.id),
        argument_ids=tuple(live_ids),
        entry_block=body_entry,
        header_block=header_name,
        exit_block=exit_block_name,
        continuation_block=continuation,
        comparison=str(compare.op),
        guarded_blocks=tuple(guarded_blocks),
    )
    lane = region.lanes[0]
    replacement_region = dataclasses.replace(
        region,
        lanes=(dataclasses.replace(
            lane,
            callees=(outline_name, *lane.callees),
            instruction_sites=((body_entry, 0),),
        ),),
    )
    return record, replacement_region


def _branch_targets(instruction: Instr) -> tuple[str, ...]:
    attributes = instruction.attributes or {}
    if instruction.op in {"Br", "br"}:
        return (str(attributes.get("target")),)
    if instruction.op in {"CondBr", "condbr"}:
        return (
            str(attributes.get("true_target")),
            str(attributes.get("false_target")),
        )
    return ()


def _retarget(instruction: Instr, old: str, new: str) -> None:
    attributes = instruction.attributes or {}
    for key in ("target", "true_target", "false_target"):
        if str(attributes.get(key)) == old:
            attributes[key] = new
    instruction.attributes = attributes


def _lane_touches_sequence_elsewhere(
    function: Function,
    member_blocks: dict[str, list[int]],
    sequence_id: int,
    append_instruction: Instr,
) -> bool:
    """Whether any OTHER lane instruction reads or mutates *sequence_id*.

    A concurrent read would observe a nondeterministic intermediate length;
    a second mutation site would interleave.  Either forfeits the
    order-insensitive-append argument.
    """

    for block_name in member_blocks:
        for instruction in function.blocks[block_name].instrs:
            if instruction is append_instruction:
                continue
            if instruction.op not in {"Call", "call"}:
                continue
            callee = str((instruction.attributes or {}).get("callee") or "")
            helper = _sequence_helper_verb(callee)
            if helper is not None and helper[0] == sequence_id:
                return True
    return False


def _store_is_iteration_private(
    function: Function,
    member_instrs: set[int],
    store: Instr,
    defined_ids: set[int],
    induction_id: int,
) -> bool:
    """Whether a lane store provably touches per-iteration memory.

    True when the stored-through address is lane-defined and its GEP chain
    reaches the induction (a per-iteration slot), or when the address itself
    is created entirely inside the lane from lane-defined roots.
    """

    producers: dict[int, Instr] = {}
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                producers[int(instruction.res.id)] = instruction
    frontier = [int(store.args[1].id)]
    visited: set[int] = set()
    saw_induction = False
    rooted_outside = False
    while frontier:
        value_id = frontier.pop()
        if value_id in visited:
            continue
        visited.add(value_id)
        if value_id == induction_id:
            saw_induction = True
            continue
        producer = producers.get(value_id)
        if producer is None or id(producer) not in member_instrs:
            if value_id not in defined_ids and value_id != induction_id:
                rooted_outside = True
            continue
        for argument in producer.args:
            frontier.append(int(argument.id))
    return saw_induction or not rooted_outside


__all__ = [
    "LaneOutlineRecord",
    "OutlineReport",
    "outline_independent_iteration_lanes",
]

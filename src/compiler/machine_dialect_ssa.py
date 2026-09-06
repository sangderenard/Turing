"""Non-lossy machine-state SSA derived from the decoded machine program.

This is a dialect inside the repository SSA containers, not a replacement for
the owning machine token graph.  Each instruction is one exact, named state
transition.  Later legalization may decompose a transition into ordinary SSA
arithmetic, memory and control operations; until then the operation remains
visible and compilability is not falsely claimed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, Instr, SSAValue
from ..transmogrifier.ssa_registry import Handler
from .machine_reference_vocabulary import (
    DecodedInstruction, MachineSemanticToken, RelativeAddressOperand,
)
from .machine_symbolic_effects import symbolic_effect_for_instruction


MACHINE_SSA_DIALECT = "turing.machine-state-ssa.amd64.v1"


def machine_dialect_occurrences(function: Function) -> tuple[tuple[str, str], ...]:
    """Return every decompiler-specific operation retained in ``function``.

    Ordinary repository operations may still carry machine provenance.  The
    boundary is the operation/dialect itself, not whether an instruction was
    originally decoded from a PE image.
    """

    occurrences = []
    if function.metadata.get("dialect") == MACHINE_SSA_DIALECT:
        occurrences.append(("<function>", MACHINE_SSA_DIALECT))
    for block_name, block in function.blocks.items():
        for instruction in block.instrs:
            if (
                str(instruction.op).startswith("machine.")
                or instruction.attributes.get("machine_dialect")
                == MACHINE_SSA_DIALECT
            ):
                occurrences.append((str(block_name), str(instruction.op)))
    return tuple(occurrences)


def repository_ssa_legalized(function: Function) -> bool:
    """Whether no decompiler-dialect transition remains in the function."""

    return not machine_dialect_occurrences(function)


def module_machine_dialect_occurrences(
    module: "IRModule | Mapping[str, Function]",
) -> tuple[tuple[str, str, str], ...]:
    """Return the complete target-facing residual ledger for one module."""

    functions = module.functions if hasattr(module, "functions") else module
    return tuple(
        (str(function_name), str(block_name), str(operation))
        for function_name, function in functions.items()
        for block_name, operation in machine_dialect_occurrences(function)
    )


def format_machine_dialect_occurrences(
    occurrences: Sequence[tuple[str, str, str]],
) -> str:
    return "; ".join(
        f"{function}:{block}:{operation}"
        for function, block, operation in occurrences
    )


def _target(instruction: DecodedInstruction) -> int | None:
    operand = next((
        item for item in instruction.operands
        if isinstance(item, RelativeAddressOperand)
    ), None)
    return None if operand is None else int(operand.target_address)


def _is_terminal(semantic: MachineSemanticToken) -> bool:
    return semantic in {
        MachineSemanticToken.RETURN,
        MachineSemanticToken.DIRECT_RELATIVE_JUMP,
        MachineSemanticToken.INDIRECT_JUMP,
        MachineSemanticToken.BREAKPOINT_TRAP,
        MachineSemanticToken.SOFTWARE_INTERRUPT,
    }


def decoded_function_to_machine_ssa(
    name: str,
    decoded: Sequence[DecodedInstruction],
    *,
    external_fallthrough_address: int | None = None,
) -> Function:
    """Retain a completely decoded function as explicit machine-state SSA."""

    instructions = tuple(decoded)
    if not instructions:
        raise ValueError("cannot build machine SSA from no instructions")
    by_address = {int(item.address): item for item in instructions}
    if len(by_address) != len(instructions):
        raise ValueError("duplicate decoded machine instruction address")
    ordered = tuple(sorted(by_address))
    start, end = ordered[0], (
        int(by_address[ordered[-1]].address)
        + len(by_address[ordered[-1]].encoded)
    )

    leaders = {start}
    for address in ordered:
        instruction = by_address[address]
        semantic = MachineSemanticToken(instruction.semantic)
        target = _target(instruction)
        fallthrough = address + len(instruction.encoded)
        if target in by_address:
            leaders.add(int(target))
        if (
            semantic in {
                MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
                MachineSemanticToken.DIRECT_RELATIVE_CALL,
                MachineSemanticToken.INDIRECT_CALL,
            }
            and fallthrough in by_address
        ):
            leaders.add(fallthrough)

    ordered_leaders = tuple(sorted(leaders))
    labels = {
        address: f"block_{address:016x}" for address in ordered_leaders
    }
    sources: dict[str, tuple[DecodedInstruction, ...]] = {}
    for index, address in enumerate(ordered_leaders):
        stop = ordered_leaders[index + 1] if index + 1 < len(ordered_leaders) else end
        sources[labels[address]] = tuple(
            by_address[item] for item in ordered if address <= item < stop
        )

    successors: dict[str, list[str]] = {}
    external_successors: dict[str, tuple[int, ...]] = {}
    conditional_successors: dict[
        str, tuple[tuple[str | None, int | None], tuple[str | None, int | None]]
    ] = {}
    predecessors: dict[str, list[str]] = defaultdict(list)
    for index, address in enumerate(ordered_leaders):
        label = labels[address]
        last = sources[label][-1]
        semantic = MachineSemanticToken(last.semantic)
        target = _target(last)
        fallthrough = int(last.address) + len(last.encoded)
        local: list[str] = []
        external: list[int] = []
        if semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
            ordered_destinations = []
            for destination in (target, fallthrough):
                if destination in labels:
                    destination_label = labels[int(destination)]
                    local.append(destination_label)
                    ordered_destinations.append((destination_label, None))
                elif destination is not None:
                    external.append(int(destination))
                    ordered_destinations.append((None, int(destination)))
                else:
                    ordered_destinations.append((None, None))
            conditional_successors[label] = tuple(ordered_destinations)
        elif semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
            if target in labels:
                local.append(labels[int(target)])
            elif target is not None:
                external.append(int(target))
        elif not _is_terminal(semantic) and fallthrough in labels:
            local.append(labels[fallthrough])
        elif (
            not _is_terminal(semantic)
            and index + 1 < len(ordered_leaders)
        ):
            local.append(labels[ordered_leaders[index + 1]])
        elif not _is_terminal(semantic) and external_fallthrough_address is not None:
            external.append(int(external_fallthrough_address))
        successors[label] = local
        external_successors[label] = tuple(external)
        for destination in local:
            predecessors[destination].append(label)

    next_value = 0

    def fresh(dtype: str) -> SSAValue:
        nonlocal next_value
        value = SSAValue(next_value, dtype=dtype)
        next_value += 1
        return value

    initial_state = fresh("machine.state.amd64")
    first_label = labels[start]
    predecessors[first_label].insert(0, "entry")
    block_inputs = {
        label: fresh("machine.state.amd64")
        for label in sources
    }
    block_outputs: dict[str, SSAValue] = {"entry": initial_state}
    blocks: dict[str, BasicBlock] = {
        "entry": BasicBlock(
            "entry",
            [Instr(
                Handler.Br.value, [], None,
                attributes={
                    "target": first_label,
                    "machine_dialect": MACHINE_SSA_DIALECT,
                    "machine_preheader": True,
                },
            )],
            [first_label],
        ),
    }
    phi_instructions: dict[str, Instr] = {}

    for label, block_sources in sources.items():
        active = block_inputs[label]
        emitted: list[Instr] = []
        phi = Instr(
            "machine.PhiState", [], active,
            attributes={
                "incoming_blocks": tuple(predecessors[label]),
                "machine_dialect": MACHINE_SSA_DIALECT,
            },
        )
        emitted.append(phi)
        phi_instructions[label] = phi
        for instruction in block_sources:
            effect = symbolic_effect_for_instruction(instruction)
            result = fresh("machine.state.amd64")
            emitted.append(Instr(
                f"machine.{effect.semantic.name.lower()}", [active], result,
                attributes={
                    "machine_dialect": MACHINE_SSA_DIALECT,
                    "machine_address": int(instruction.address),
                    "machine_token": instruction.token.name,
                    "machine_semantic": effect.semantic.name,
                    "machine_encoded": bytes(instruction.encoded).hex(),
                    "machine_operands": tuple(repr(item) for item in instruction.operands),
                    "machine_reads": effect.reads,
                    "machine_writes": effect.writes,
                    "effect_domains": effect.effect_domains,
                    "may_trap": effect.may_trap,
                    "conditional": effect.conditional,
                },
            ))
            active = result
        block_outputs[label] = active

        last = block_sources[-1]
        semantic = MachineSemanticToken(last.semantic)
        local = successors[label]
        external = external_successors[label]
        if semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
            true_destination, false_destination = conditional_successors[label]
            predicate = fresh("machine.predicate")
            emitted.append(Instr(
                "machine.condition", [active], predicate,
                attributes={
                    "machine_dialect": MACHINE_SSA_DIALECT,
                    "machine_token": last.token.name,
                    "machine_address": int(last.address),
                },
            ))
            emitted.append(Instr(
                Handler.CondBr.value, [predicate], None,
                attributes={
                    "true_target": true_destination[0],
                    "false_target": false_destination[0],
                    "true_target_address": true_destination[1],
                    "false_target_address": false_destination[1],
                    "machine_dialect": MACHINE_SSA_DIALECT,
                    "machine_address": int(last.address),
                },
            ))
        elif local:
            emitted.append(Instr(
                Handler.Br.value, [], None,
                attributes={"target": local[0], "machine_dialect": MACHINE_SSA_DIALECT},
            ))
        elif external:
            emitted.append(Instr(
                "machine.ExternalBr", [active], None,
                attributes={
                    "target_addresses": external,
                    "machine_dialect": MACHINE_SSA_DIALECT,
                    "machine_address": int(last.address),
                },
            ))
        elif semantic is MachineSemanticToken.RETURN:
            emitted.append(Instr(
                Handler.Ret.value, [active], None,
                attributes={"machine_dialect": MACHINE_SSA_DIALECT},
            ))
        elif semantic in {
            MachineSemanticToken.BREAKPOINT_TRAP,
            MachineSemanticToken.SOFTWARE_INTERRUPT,
            MachineSemanticToken.INDIRECT_JUMP,
        }:
            emitted.append(Instr(
                "machine.Terminate", [active], None,
                attributes={
                    "machine_semantic": semantic.name,
                    "machine_dialect": MACHINE_SSA_DIALECT,
                },
            ))
        blocks[label] = BasicBlock(label, emitted, list(local))

    for label, phi in phi_instructions.items():
        incoming_blocks = tuple(predecessors[label])
        phi.args[:] = [block_outputs[item] for item in incoming_blocks]

    return Function(
        name, [initial_state], blocks,
        metadata={
            "dialect": MACHINE_SSA_DIALECT,
            "lifted_from": "decoded-machine-program",
            "entry_block": "entry",
            "argument_names": ("__machine_state",),
            "machine_instruction_count": len(instructions),
            "machine_region": (start, end),
            "requires_machine_legalization": True,
        },
    )


__all__ = [
    "MACHINE_SSA_DIALECT", "decoded_function_to_machine_ssa",
    "format_machine_dialect_occurrences", "machine_dialect_occurrences",
    "module_machine_dialect_occurrences", "repository_ssa_legalized",
]

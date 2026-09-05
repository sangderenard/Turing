"""Address-linked live SSA inspection for decompile/edit/recompile execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from ..transmogrifier.ssa import Function, IRModule
from .machine_execution import (
    MachineExecutionOrchestrator, MachineExecutionResult, MachineExecutionState,
)


@dataclass(frozen=True, slots=True)
class AddressLinkedSSALine:
    line_id: str
    function_name: str
    block_name: str
    ordinal: int
    machine_address: int | None
    operation: str
    text: str
    constant_value: object | None = None
    machine_operand_role: str | None = None


@dataclass(frozen=True, slots=True)
class LiveSSAViewport:
    sequence: int
    machine_address: int
    highlighted_line_ids: tuple[str, ...]
    lines: tuple[AddressLinkedSSALine, ...]
    instruction_token: str | None
    semantic_token: str | None
    encoded: bytes
    changed_registers: Mapping[str, tuple[int, int]]
    result: MachineExecutionResult
    source_machine_address: int | None = None
    source_encoded: bytes = b""
    stream_name: str | None = None
    stream_route: str | None = None
    external_reference_id: int | None = None
    source_read_head: str | None = None
    read_head: str | None = None
    source_read_head_microsteps: int = 0
    read_head_microsteps: int = 0


def _line_text(instruction) -> str:
    result = "" if instruction.res is None else f"{instruction.res.name()} = "
    arguments = ", ".join(item.name() for item in instruction.args)
    detail = ""
    if "value" in instruction.attributes:
        detail = f" value={instruction.attributes['value']!r}"
    return f"{result}{instruction.op} {arguments}{detail}".rstrip()


def address_linked_ssa_lines(module: IRModule) -> tuple[AddressLinkedSSALine, ...]:
    """Flatten authored SSA without losing function/block/occurrence identity."""

    lines = []
    for function_name, function in module.functions.items():
        for block_name, block in function.blocks.items():
            for ordinal, instruction in enumerate(block.instrs):
                raw_address = instruction.attributes.get("machine_address")
                lines.append(AddressLinkedSSALine(
                    f"{function_name}:{block_name}:{ordinal}",
                    str(function_name), str(block_name), ordinal,
                    None if raw_address is None else int(raw_address),
                    str(instruction.op), _line_text(instruction),
                    instruction.attributes.get("value"),
                    instruction.attributes.get("machine_operand_role"),
                ))
    return tuple(lines)


def ssa_viewport(
    lines: Iterable[AddressLinkedSSALine],
    machine_address: int,
    *,
    before: int = 8,
    after: int = 8,
    search: str | None = None,
) -> tuple[tuple[AddressLinkedSSALine, ...], tuple[str, ...]]:
    """Select a configurable address-centred window with optional filtering."""

    if before < 0 or after < 0:
        raise ValueError("SSA viewport extents must be non-negative")
    catalogue = tuple(lines)
    highlighted = tuple(
        item.line_id for item in catalogue
        if item.machine_address == int(machine_address)
    )
    if not highlighted:
        return (), ()
    if search is not None:
        needle = str(search).casefold()
        visible = tuple(item for item in catalogue if needle in item.text.casefold())
        return visible, highlighted
    indices = tuple(
        index for index, item in enumerate(catalogue)
        if item.line_id in highlighted
    )
    left = max(0, min(indices) - before)
    right = min(len(catalogue), max(indices) + after + 1)
    return catalogue[left:right], highlighted


class LiveSSAExecutionSession:
    """Step a VM while publishing the SSA correlated to its current PC."""

    def __init__(
        self,
        module: IRModule,
        executor: MachineExecutionOrchestrator,
        state: MachineExecutionState,
        *,
        before: int = 8,
        after: int = 8,
        search: str | None = None,
    ) -> None:
        self.module = module
        self.executor = executor
        self.state = state
        self.before = int(before)
        self.after = int(after)
        self.search = search
        self.lines = address_linked_ssa_lines(module)
        self.sequence = 0

    def step(self) -> LiveSSAViewport:
        source_address = int(self.state.pc)
        before_registers = self.state.register_contents()
        result = self.executor.step(self.state)
        stream_event = getattr(self.executor, "last_stream_event", None)
        address = (
            source_address if stream_event is None
            else int(stream_event.instruction_address)
        )
        after_registers = result.state.register_contents()
        changed = MappingProxyType({
            name: (int(before_registers[name]), int(after_registers[name]))
            for name in before_registers
            if before_registers[name] != after_registers[name]
        })
        visible, highlighted = ssa_viewport(
            self.lines, address, before=self.before, after=self.after,
            search=self.search,
        )
        instruction = result.instruction
        self.state = result.state
        self.sequence += 1
        return LiveSSAViewport(
            self.sequence, address, highlighted, visible,
            None if instruction is None else instruction.token.name,
            None if instruction is None else instruction.semantic.name,
            b"" if instruction is None else bytes(instruction.encoded),
            changed, result,
            source_address,
            (
                b"" if stream_event is None
                else bytes(stream_event.source_encoded)
            ),
            None if stream_event is None else str(stream_event.stream_name),
            (
                None if stream_event is None
                else stream_event.route.name
            ),
            (
                None if stream_event is None
                else stream_event.external_reference_id
            ),
            (
                None if stream_event is None
                else str(stream_event.source_read_head)
            ),
            (
                None if stream_event is None
                else str(stream_event.read_head)
            ),
            (
                0 if stream_event is None
                else int(stream_event.source_read_head_microsteps)
            ),
            (
                0 if stream_event is None
                else int(stream_event.read_head_microsteps)
            ),
        )


@dataclass(frozen=True, slots=True)
class SSAOperationReplacement:
    line_id: str
    old_operation: str
    new_operation: str


@dataclass(frozen=True, slots=True)
class SSAConstantReplacement:
    line_id: str
    old_value: object
    new_value: object


@dataclass(frozen=True, slots=True)
class SSAEditValidation:
    replacements: tuple[SSAOperationReplacement | SSAConstantReplacement, ...]
    executable: bool
    ledger: object


class SSAEditTransaction:
    """Reversible SSA text-operation edits with a mandatory emission gate."""

    def __init__(self, module: IRModule) -> None:
        self.module = module
        self._original_instructions = {
            f"{function_name}:{block_name}:{ordinal}": (
                str(instruction.op), deepcopy(instruction.attributes),
            )
            for function_name, function in module.functions.items()
            for block_name, block in function.blocks.items()
            for ordinal, instruction in enumerate(block.instrs)
        }
        self._replacements: list[SSAOperationReplacement | SSAConstantReplacement] = []

    @property
    def replacements(self) -> tuple[SSAOperationReplacement | SSAConstantReplacement, ...]:
        return tuple(self._replacements)

    def replace(
        self,
        search: str,
        replacement: str,
        *,
        line_ids: Iterable[str] | None = None,
    ) -> tuple[SSAOperationReplacement, ...]:
        edits = replace_ssa_operations(
            self.module, search, replacement, line_ids=line_ids,
        )
        self._replacements.extend(edits)
        return edits

    def replace_constant(self, line_id: str, value: object) -> SSAConstantReplacement:
        """Replace one visible Const value by stable viewport occurrence ID."""

        for function_name, function in self.module.functions.items():
            for block_name, block in function.blocks.items():
                for ordinal, instruction in enumerate(block.instrs):
                    candidate = f"{function_name}:{block_name}:{ordinal}"
                    if candidate != str(line_id):
                        continue
                    if str(instruction.op) != "Const" or "value" not in instruction.attributes:
                        raise ValueError(f"SSA line {line_id!r} is not an authored constant")
                    old = instruction.attributes["value"]
                    instruction.attributes["value"] = value
                    edit = SSAConstantReplacement(candidate, old, value)
                    self._replacements.append(edit)
                    return edit
        raise KeyError(f"unknown SSA line {line_id!r}")

    def validate_pe(self, **ledger_options) -> SSAEditValidation:
        """Run the proof-gated reverse compiler; completeness grants execution."""

        from .pe_recompilation import build_pe_recompilation_ledger

        ledger = build_pe_recompilation_ledger(self.module, **ledger_options)
        return SSAEditValidation(
            self.replacements, bool(ledger.complete), ledger,
        )

    def rollback(self) -> None:
        for function_name, function in self.module.functions.items():
            for block_name, block in function.blocks.items():
                for ordinal, instruction in enumerate(block.instrs):
                    line_id = f"{function_name}:{block_name}:{ordinal}"
                    operation, attributes = self._original_instructions[line_id]
                    instruction.op = operation
                    instruction.attributes.clear()
                    instruction.attributes.update(deepcopy(attributes))
        self._replacements.clear()


def replace_ssa_operations(
    module: IRModule,
    search: str,
    replacement: str,
    *,
    line_ids: Iterable[str] | None = None,
) -> tuple[SSAOperationReplacement, ...]:
    """Apply an explicit operation-name edit while retaining occurrence identity.

    Recompilation remains a separate mandatory validation step; this function
    cannot mark edited SSA executable or attach machine bytes.
    """

    if not search:
        raise ValueError("SSA operation search cannot be empty")
    selected = None if line_ids is None else frozenset(str(item) for item in line_ids)
    edits = []
    for function_name, function in module.functions.items():
        for block_name, block in function.blocks.items():
            for ordinal, instruction in enumerate(block.instrs):
                line_id = f"{function_name}:{block_name}:{ordinal}"
                if selected is not None and line_id not in selected:
                    continue
                old = str(instruction.op)
                new = old.replace(search, replacement)
                if new == old:
                    continue
                instruction.op = new
                edits.append(SSAOperationReplacement(line_id, old, new))
    return tuple(edits)


__all__ = [
    "AddressLinkedSSALine", "LiveSSAExecutionSession", "LiveSSAViewport",
    "SSAConstantReplacement", "SSAEditTransaction", "SSAEditValidation",
    "SSAOperationReplacement",
    "address_linked_ssa_lines",
    "replace_ssa_operations", "ssa_viewport",
]

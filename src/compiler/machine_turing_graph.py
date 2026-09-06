"""Direct reachable x86 decoding into the Turing bit-calculus graph.

This module deliberately sits before repository SSA.  The x86 decoder owns
reachability and instruction boundaries; :class:`MachineTuringWriter` owns
versioned register bit-carriers and expands supported pure value semantics
through the existing Turing recipes.  Control topology remains a separate
envelope because calls, effects, and branch scheduling are not bit operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable

import networkx as nx

from .bitops_translator import BitOpsTranslator, BitStr
from .machine_program_graph import (
    MachineFunctionDecodeReport,
    decode_reachable_region,
)
from .machine_reference_vocabulary import (
    DecodedInstruction,
    ImmediateOperand,
    MachineSemanticToken,
    RegisterOperand,
    RelativeAddressOperand,
    X86ReferenceDecoder,
    X86Register,
)
from ..transmogrifier.ssa import Instr, SSAValue


class TuringOperatorToken(IntEnum):
    """Stable numeric vocabulary for the reduced operator graph."""

    INPUT = 0
    CONSTANT = 1
    NAND = 2
    MOTION_LEFT = 3
    MOTION_RIGHT = 4
    CONCAT = 5
    SLICE = 6
    SELECT = 7
    LENGTH = 8
    ZEROS = 9


_OPERATOR_TOKENS = {
    "nand": TuringOperatorToken.NAND,
    "sigma_L": TuringOperatorToken.MOTION_LEFT,
    "sigma_R": TuringOperatorToken.MOTION_RIGHT,
    "concat": TuringOperatorToken.CONCAT,
    "slice": TuringOperatorToken.SLICE,
    "mu": TuringOperatorToken.SELECT,
    "length": TuringOperatorToken.LENGTH,
    "zeros": TuringOperatorToken.ZEROS,
}


@dataclass(frozen=True, slots=True)
class TopologicalReduction:
    graph: nx.MultiDiGraph
    node_map: dict[int, int]

    def verify_quotient(self, original: nx.MultiDiGraph) -> bool:
        return is_turing_topological_quotient(original, self)


@dataclass(frozen=True, slots=True)
class TuringSSAProjection:
    """Repository SSA instructions projected from one Turing operator DAG."""

    instructions: tuple[Instr, ...]
    register_outputs: dict[str, SSAValue]


@dataclass(frozen=True, slots=True)
class MachineTuringGraph:
    """One direct binary-to-calculus result plus its control envelope."""

    operator_graph: nx.MultiDiGraph
    control_graph: nx.MultiDiGraph
    report: MachineFunctionDecodeReport
    discovery_order: tuple[int, ...]
    register_outputs: dict[X86Register, int]
    register_values: dict[X86Register, int]
    unsupported_semantics: tuple[tuple[int, MachineSemanticToken, str], ...]

    @property
    def complete(self) -> bool:
        return self.report.complete and not self.unsupported_semantics

    def topologically_reduce(self) -> TopologicalReduction:
        return reduce_turing_operator_graph(self.operator_graph)

    def to_ssa(self, *, reduce: bool = True) -> TuringSSAProjection:
        graph = (
            self.topologically_reduce().graph
            if reduce
            else self.operator_graph
        )
        return turing_operator_graph_to_ssa(graph)


class MachineTuringWriter:
    """Append-only writer for the pure register/value portion of x86 state."""

    _BINARY_RECIPES = {
        MachineSemanticToken.INTEGER_ADD: "add",
        MachineSemanticToken.INTEGER_SUBTRACT: "sub",
        MachineSemanticToken.INTEGER_MULTIPLY: "mul",
        MachineSemanticToken.BITWISE_AND: "bitand",
        MachineSemanticToken.BITWISE_OR: "bitor",
        MachineSemanticToken.BITWISE_XOR: "bitxor",
    }

    def __init__(self, bit_width: int = 64):
        if bit_width <= 0:
            raise ValueError("bit_width must be positive")
        self.bit_width = bit_width
        self.translator = BitOpsTranslator(bit_width)
        self.registers: dict[X86Register, BitStr] = {}
        self.initial_registers: dict[X86Register, BitStr] = {}
        self.constants: dict[int, BitStr] = {}
        self.unsupported: list[tuple[int, MachineSemanticToken, str]] = []
        self.instruction_addresses: list[int] = []
        self.instructions: list[DecodedInstruction] = []

    def _bits_for_integer(self, value: int) -> BitStr:
        normalized = int(value) & ((1 << self.bit_width) - 1)
        existing = self.constants.get(normalized)
        if existing is not None:
            return existing
        bits = self.translator.bits_from_int(normalized)
        self.translator.graph.bind_input(
            bits,
            name=f"constant:{normalized}",
            metadata={
                "kind": "constant",
                "value": normalized,
                "width": self.bit_width,
            },
        )
        self.constants[normalized] = bits
        return bits

    def _initial_register_value(self, operand: RegisterOperand) -> BitStr:
        if operand.width != self.bit_width:
            raise NotImplementedError(
                f"{operand.width}-bit register views require explicit "
                f"extension/truncation in a {self.bit_width}-bit writer"
            )
        existing = self.initial_registers.get(operand.register)
        if existing is not None:
            return existing
        bits = self.translator.bits_from_int(0)
        self.translator.graph.bind_input(
            bits,
            name=f"register:{operand.register.name}",
            metadata={
                "kind": "register",
                "register": operand.register.name,
                "width": self.bit_width,
            },
        )
        self.initial_registers[operand.register] = bits
        return bits

    def _register_value(
        self,
        operand: RegisterOperand,
        state: dict[X86Register, BitStr],
    ) -> BitStr:
        existing = state.get(operand.register)
        return (
            existing
            if existing is not None
            else self._initial_register_value(operand)
        )

    def _operand_value(
        self,
        operand: Any,
        state: dict[X86Register, BitStr],
    ) -> BitStr:
        if isinstance(operand, RegisterOperand):
            return self._register_value(operand, state)
        if isinstance(operand, ImmediateOperand):
            return self._bits_for_integer(operand.value)
        raise NotImplementedError(
            f"pure Turing writer does not yet materialize "
            f"{type(operand).__name__}"
        )

    def _record_unsupported(
        self,
        instruction: DecodedInstruction,
        reason: str,
    ) -> None:
        self.unsupported.append(
            (instruction.address, instruction.semantic, reason)
        )

    def consume(self, instruction: DecodedInstruction) -> None:
        """Append one decoder event without routing through repository SSA."""

        self.instruction_addresses.append(instruction.address)
        self.instructions.append(instruction)

    def _apply(
        self,
        instruction: DecodedInstruction,
        incoming: dict[X86Register, BitStr],
    ) -> dict[X86Register, BitStr]:
        """Apply one pure instruction to a copy of its incoming state."""

        state = dict(incoming)
        semantic = instruction.semantic
        if semantic in {
            MachineSemanticToken.NO_OPERATION,
            MachineSemanticToken.RETURN,
            MachineSemanticToken.DIRECT_RELATIVE_JUMP,
            MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP,
        }:
            return state
        if semantic is MachineSemanticToken.DIRECT_RELATIVE_CALL:
            self._record_unsupported(
                instruction,
                "call requires an explicit continuation and effect state",
            )
            return state
        try:
            if semantic is MachineSemanticToken.REGISTER_WRITE_IMMEDIATE:
                destination, source = instruction.operands
                if not isinstance(destination, RegisterOperand):
                    raise NotImplementedError("immediate destination is not a register")
                if destination.width != self.bit_width:
                    raise NotImplementedError(
                        f"{destination.width}-bit immediate write needs an explicit "
                        "register-view policy"
                    )
                state[destination.register] = self._operand_value(source, state)
                return state

            recipe = self._BINARY_RECIPES.get(semantic)
            if recipe is not None:
                destination, source = instruction.operands[:2]
                if not isinstance(destination, RegisterOperand):
                    raise NotImplementedError("memory destinations are effect nodes")
                left = self._register_value(destination, state)
                right = self._operand_value(source, state)
                state[destination.register] = self.translator.apply_bits(
                    recipe, left, right,
                )
                return state

            if semantic is MachineSemanticToken.BITWISE_NOT:
                (destination,) = instruction.operands
                if not isinstance(destination, RegisterOperand):
                    raise NotImplementedError("memory destinations are effect nodes")
                value = self._register_value(destination, state)
                state[destination.register] = self.translator.apply_bits(
                    "invert", value,
                )
                return state

            raise NotImplementedError(
                "semantic belongs in the control/effect envelope or has no "
                "Turing recipe yet"
            )
        except (NotImplementedError, TypeError, ValueError) as error:
            self._record_unsupported(instruction, str(error))
        return state

    def _join_selector(
        self,
        *,
        address: int,
        predecessor: int,
        ordinal: int,
    ) -> BitStr:
        selector = self.translator.bits_from_int(0)
        self.translator.graph.bind_input(
            selector,
            name=f"join:{address:#x}:predecessor:{predecessor:#x}",
            metadata={
                "kind": "control-selector",
                "join_address": address,
                "predecessor": predecessor,
                "ordinal": ordinal,
                "width": self.bit_width,
            },
        )
        return selector

    def _merge_states(
        self,
        address: int,
        incoming: list[tuple[int, dict[X86Register, BitStr]]],
    ) -> dict[X86Register, BitStr]:
        if not incoming:
            return {}
        merged = dict(incoming[0][1])
        known_registers = set().union(*(state for _pred, state in incoming))
        for ordinal, (predecessor, state) in enumerate(incoming[1:], start=1):
            selector: BitStr | None = None
            for register in sorted(known_registers, key=int):
                operand = RegisterOperand(register, self.bit_width)
                left = merged.get(register)
                if left is None:
                    left = self._initial_register_value(operand)
                right = state.get(register)
                if right is None:
                    right = self._initial_register_value(operand)
                if left is right:
                    merged[register] = left
                    continue
                if selector is None:
                    selector = self._join_selector(
                        address=address,
                        predecessor=predecessor,
                        ordinal=ordinal,
                    )
                result = self.translator.tm.mu(left, right, selector)
                producer = self.translator.graph.producer_index(result)
                if producer is not None:
                    self.translator.graph.nodes[producer].metadata.update({
                        "kind": "state-join",
                        "join_address": address,
                        "predecessors": tuple(pred for pred, _state in incoming),
                        "register": register.name,
                    })
                merged[register] = result
        return merged

    def lower_control_graph(self, control_graph: nx.MultiDiGraph) -> None:
        """Solve acyclic instruction states and emit explicit Turing joins."""

        if not self.instructions:
            self.registers = {}
            return
        by_address = {
            instruction.address: instruction for instruction in self.instructions
        }
        if not nx.is_directed_acyclic_graph(control_graph):
            instruction = min(self.instructions, key=lambda item: item.address)
            self._record_unsupported(
                instruction,
                "cyclic control requires loop-header state fixed points",
            )
            self.registers = {}
            return
        outgoing_states: dict[int, dict[X86Register, BitStr]] = {}
        for address in nx.topological_sort(control_graph):
            predecessors = sorted(control_graph.predecessors(address))
            incoming = self._merge_states(
                address,
                [(pred, outgoing_states[pred]) for pred in predecessors],
            )
            instruction = by_address[address]
            first_emitted = len(self.translator.graph.nodes)
            outgoing_states[address] = self._apply(instruction, incoming)
            for node in self.translator.graph.nodes[first_emitted:]:
                node.metadata.setdefault(
                    "machine_instruction_address", instruction.address,
                )
                node.metadata.setdefault(
                    "machine_semantic_token", int(instruction.semantic),
                )
                node.metadata.setdefault(
                    "machine_mnemonic", instruction.token.name.lower(),
                )
        sinks = sorted(
            address for address in control_graph if control_graph.out_degree(address) == 0
        )
        self.registers = self._merge_states(
            max(by_address) + 1,
            [(address, outgoing_states[address]) for address in sinks],
        )

    def operator_graph(self) -> tuple[nx.MultiDiGraph, dict[X86Register, int]]:
        graph = nx.MultiDiGraph(
            layer="turing-operator",
            bit_width=self.bit_width,
        )
        for node in self.translator.graph.nodes:
            metadata = dict(node.metadata)
            if node.op == "input":
                kind = metadata.get("kind", "input")
                token = (
                    TuringOperatorToken.CONSTANT
                    if kind == "constant"
                    else TuringOperatorToken.INPUT
                )
            else:
                token = _OPERATOR_TOKENS[node.op]
            graph.add_node(
                node.idx,
                op=node.op,
                token_id=int(token),
                kwargs=dict(node.kwargs),
                metadata=metadata,
            )
        for edge in self.translator.graph.edges:
            graph.add_edge(
                edge.src_idx,
                edge.dst_idx,
                role=f"arg:{edge.arg_pos}",
                arg_pos=edge.arg_pos,
            )
        outputs: dict[X86Register, int] = {}
        for register, bits in self.registers.items():
            producer = self.translator.graph.producer_index(bits)
            if producer is not None:
                outputs[register] = producer
                graph.nodes[producer].setdefault("output_registers", []).append(
                    register.name
                )
        graph.graph["register_outputs"] = {
            register.name: node for register, node in outputs.items()
        }
        return graph, outputs


def _control_graph(
    instructions: Iterable[DecodedInstruction],
) -> nx.MultiDiGraph:
    ordered = tuple(sorted(instructions, key=lambda item: item.address))
    by_address = {instruction.address: instruction for instruction in ordered}
    graph = nx.MultiDiGraph(layer="machine-control")
    for instruction in ordered:
        graph.add_node(
            instruction.address,
            token_id=int(instruction.token),
            semantic_token_id=int(instruction.semantic),
            token=instruction.token.name,
            semantic=instruction.semantic.name,
            encoded=instruction.encoded,
        )
    for instruction in ordered:
        next_address = instruction.address + len(instruction.encoded)
        targets = tuple(
            operand.target_address
            for operand in instruction.operands
            if isinstance(operand, RelativeAddressOperand)
        )
        semantic = instruction.semantic
        if semantic is MachineSemanticToken.CONDITIONAL_RELATIVE_JUMP:
            if next_address in by_address:
                graph.add_edge(instruction.address, next_address, role="fallthrough")
            for target in targets:
                if target in by_address:
                    graph.add_edge(instruction.address, target, role="branch")
        elif semantic is MachineSemanticToken.DIRECT_RELATIVE_JUMP:
            for target in targets:
                if target in by_address:
                    graph.add_edge(instruction.address, target, role="branch")
        elif semantic is MachineSemanticToken.DIRECT_RELATIVE_CALL:
            if next_address in by_address:
                graph.add_edge(instruction.address, next_address, role="continuation")
            for target in targets:
                if target in by_address:
                    graph.add_edge(instruction.address, target, role="call")
        elif semantic not in {
            MachineSemanticToken.RETURN,
            MachineSemanticToken.INDIRECT_JUMP,
            MachineSemanticToken.BREAKPOINT_TRAP,
            MachineSemanticToken.SOFTWARE_INTERRUPT,
        } and next_address in by_address:
            graph.add_edge(instruction.address, next_address, role="sequence")
    return graph


def raise_binary_to_turing_graph(
    binary_region: bytes | bytearray | memoryview,
    *,
    base_address: int = 0,
    entry_offsets: tuple[int, ...] = (0,),
    bit_width: int = 64,
    decoder: X86ReferenceDecoder | None = None,
) -> MachineTuringGraph:
    """Stream one bounded x86 region directly into a Turing operator graph."""

    materialized = bytes(binary_region)
    writer = MachineTuringWriter(bit_width)
    report = decode_reachable_region(
        decoder or X86ReferenceDecoder(),
        materialized,
        base_address=base_address,
        entry_offsets=entry_offsets,
        instruction_sink=writer.consume,
    )
    control_graph = _control_graph(report.instructions)
    writer.lower_control_graph(control_graph)
    operator_graph, outputs = writer.operator_graph()
    values = {
        register: writer.translator.int_from_bits(bits)
        for register, bits in writer.registers.items()
    }
    return MachineTuringGraph(
        operator_graph=operator_graph,
        control_graph=control_graph,
        report=report,
        discovery_order=tuple(writer.instruction_addresses),
        register_outputs=outputs,
        register_values=values,
        unsupported_semantics=tuple(writer.unsupported),
    )


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _frozen(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_frozen(item) for item in value)
    return value


def reduce_turing_operator_graph(graph: nx.MultiDiGraph) -> TopologicalReduction:
    """Hash-cons equivalent pure nodes without changing dependency topology."""

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Turing operator reduction requires a DAG")
    reduced = nx.MultiDiGraph(**graph.graph)
    old_to_new: dict[int, int] = {}
    signatures: dict[Any, int] = {}
    next_id = 0
    for old_id in nx.topological_sort(graph):
        data = dict(graph.nodes[old_id])
        inputs = tuple(sorted(
            (
                int(edge_data.get("arg_pos", 0)),
                old_to_new[source],
            )
            for source, _target, _key, edge_data in graph.in_edges(
                old_id, keys=True, data=True,
            )
        ))
        metadata = dict(data.get("metadata", {}))
        signature = (
            data.get("token_id"),
            inputs,
            _frozen(data.get("kwargs", {})),
            _frozen(metadata),
        )
        existing = signatures.get(signature)
        if existing is not None:
            old_to_new[old_id] = existing
            for register in data.get("output_registers", ()):
                reduced.nodes[existing].setdefault("output_registers", []).append(
                    register
                )
            continue
        new_id = next_id
        next_id += 1
        signatures[signature] = new_id
        old_to_new[old_id] = new_id
        reduced.add_node(new_id, **data)
        for arg_pos, source in inputs:
            reduced.add_edge(
                source, new_id, role=f"arg:{arg_pos}", arg_pos=arg_pos,
            )
    reduced.graph["register_outputs"] = {
        name: old_to_new[node]
        for name, node in graph.graph.get("register_outputs", {}).items()
    }
    return TopologicalReduction(reduced, old_to_new)


def is_turing_topological_quotient(
    original: nx.MultiDiGraph,
    reduction: TopologicalReduction,
) -> bool:
    """Verify that ``node_map`` is a token- and edge-preserving quotient."""

    reduced = reduction.graph
    mapping = reduction.node_map
    if set(mapping) != set(original):
        return False
    if set(mapping.values()) != set(reduced):
        return False
    for original_id, reduced_id in mapping.items():
        if reduced_id not in reduced:
            return False
        if (
            original.nodes[original_id].get("token_id")
            != reduced.nodes[reduced_id].get("token_id")
        ):
            return False
    for source, target, edge_payload in original.edges(data=True):
        reduced_source = mapping[source]
        reduced_target = mapping[target]
        if reduced_source == reduced_target:
            return False
        candidates = reduced.get_edge_data(
            reduced_source, reduced_target, default={},
        )
        if not any(
            payload.get("arg_pos") == edge_payload.get("arg_pos")
            for payload in candidates.values()
        ):
            return False
    expected_outputs = {
        name: mapping[node_id]
        for name, node_id in original.graph.get("register_outputs", {}).items()
    }
    return expected_outputs == reduced.graph.get("register_outputs", {})


def turing_operator_graph_to_ssa(
    graph: nx.MultiDiGraph,
) -> TuringSSAProjection:
    """Project a reduced calculus DAG into ordinary repository SSA values."""

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Turing-to-SSA projection requires a DAG")
    bit_width = int(graph.graph.get("bit_width", 0))
    values: dict[int, SSAValue] = {}
    instructions: list[Instr] = []
    for node_id in nx.topological_sort(graph):
        payload = graph.nodes[node_id]
        result = SSAValue(
            int(node_id),
            dtype=f"bits{bit_width}" if bit_width else "bits",
            shape=(bit_width,) if bit_width else (),
        )
        values[node_id] = result
        incoming = sorted(
            (
                int(edge_payload.get("arg_pos", 0)),
                source,
            )
            for source, _target, _key, edge_payload in graph.in_edges(
                node_id, keys=True, data=True,
            )
        )
        metadata = dict(payload.get("metadata", {}))
        kind = metadata.get("kind")
        op = (
            "constant" if payload.get("op") == "input" and kind == "constant"
            else payload.get("op", "input")
        )
        instructions.append(Instr(
            str(op),
            [values[source] for _position, source in incoming],
            result,
            arg_roles=[f"arg:{position}" for position, _source in incoming],
            attributes={
                "token_id": int(payload["token_id"]),
                "metadata": metadata,
                "kwargs": dict(payload.get("kwargs", {})),
            },
        ))
    outputs = {
        name: values[node_id]
        for name, node_id in graph.graph.get("register_outputs", {}).items()
    }
    return TuringSSAProjection(tuple(instructions), outputs)


__all__ = [
    "MachineTuringGraph",
    "MachineTuringWriter",
    "TopologicalReduction",
    "TuringSSAProjection",
    "TuringOperatorToken",
    "is_turing_topological_quotient",
    "raise_binary_to_turing_graph",
    "reduce_turing_operator_graph",
    "turing_operator_graph_to_ssa",
]

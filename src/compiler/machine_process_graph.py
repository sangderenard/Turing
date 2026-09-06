"""Direct MachineProgramGraph ingestion into the full ProcessGraph schema.

The adapter is a structural view over already decoded machine records.  It
does not execute, decode again, numerically project, or route through
``FusedProgram``.  Each machine instruction is one ProcessGraph operation
whose input/output is the complete versioned machine state; exact bytes,
operands, effects, and control destinations remain attached as schema data.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .machine_reference_vocabulary import (
    MachineSemanticToken, RelativeAddressOperand,
)
from .machine_symbolic_effects import symbolic_effect_for_instruction
from .semantic_translation import (
    SemanticOperationIdentity, SemanticRepresentation, semantic_identity,
)
from ..transmogrifier.graph.graph_express2 import ProcessGraph


MACHINE_PROCESS_SCHEMA_VERSION = 1


def machine_process_operation(semantic: MachineSemanticToken | int) -> str:
    return f"machine.{MachineSemanticToken(semantic).name.lower()}"


def machine_process_role_schemas() -> dict[str, dict[str, dict[str, str]]]:
    """Schemas accepted by the ordinary ProcessGraph ingestion machinery."""

    return {
        machine_process_operation(semantic): {
            "up": {"machine_state": "one"},
            "down": {"machine_state": "many"},
        }
        for semantic in MachineSemanticToken
    }


def install_machine_role_schemas(graph: ProcessGraph) -> None:
    """Customize one graph without mutating global language schemas."""

    graph.role_schemas = {
        **graph.role_schemas,
        **machine_process_role_schemas(),
    }


def _operand_record(operand: Any) -> dict[str, Any]:
    if is_dataclass(operand):
        values = asdict(operand)
    else:
        values = dict(vars(operand)) if hasattr(operand, "__dict__") else {
            "repr": repr(operand),
        }
    normalized = {}
    for key, value in values.items():
        if hasattr(value, "name"):
            normalized[key] = value.name
        else:
            normalized[key] = value
    normalized["kind"] = type(operand).__name__
    return normalized


def machine_program_to_process_graph(program: Any) -> ProcessGraph:
    """Import an owning MachineProgramGraph as one full ProcessGraph.

    Functions remain separate state chains.  Local branch/call targets are
    represented as non-value ``control-target`` edges in addition to the
    sequential machine-state dependency, so topological/control consumers can
    retain loops without confusing target identity with a numeric operand.
    """

    graph = ProcessGraph(materialize_memory=False)
    install_machine_role_schemas(graph)
    next_node = 0
    address_nodes: dict[int, int] = {}
    pending_control: list[tuple[int, int]] = []

    def allocate() -> int:
        nonlocal next_node
        node = next_node
        next_node += 1
        return node

    for function_index, record in enumerate(program.functions):
        entry = allocate()
        function_rva = int(record.function.begin_rva)
        graph.G.add_node(
            entry,
            op="machine.state.input",
            label=f"machine state for sub_{function_rva:08x}",
            type="machine.state.input",
            parents=[], children=[],
            input_roles=(), output_roles=("machine_state",),
            schema_version=MACHINE_PROCESS_SCHEMA_VERSION,
            attributes={
                "machine_function_rva": function_rva,
                "machine_function_index": function_index,
                "semantic_family": "machine.state",
                "semantic_representation": (
                    SemanticRepresentation.PROCESS_GRAPH.value
                ),
            },
            tensor={}, control={"entry": True}, source_span=None,
        )
        previous = entry
        for instruction_index, instruction in enumerate(record.report.instructions):
            node = allocate()
            effect = symbolic_effect_for_instruction(instruction)
            facets = {
                "machine_semantic": effect.semantic.name,
                "instruction_token": instruction.token.name,
                "encoded": bytes(instruction.encoded).hex(),
                "reads": effect.reads,
                "writes": effect.writes,
                "effect_domains": effect.effect_domains,
                "may_trap": effect.may_trap,
                "conditional": effect.conditional,
            }
            machine_identity = semantic_identity(
                effect.semantic.name,
                SemanticRepresentation.MACHINE_GRAPH,
                facets=facets,
            )
            identity = SemanticOperationIdentity(
                machine_identity.family,
                SemanticRepresentation.PROCESS_GRAPH,
                machine_process_operation(effect.semantic),
                machine_identity.facets,
            )
            attributes = {
                **identity.attributes(),
                "machine_source_representation": (
                    SemanticRepresentation.MACHINE_GRAPH.value
                ),
                "machine_address": int(instruction.address),
                "machine_encoded": bytes(instruction.encoded).hex(),
                "machine_token": instruction.token.name,
                "machine_semantic": effect.semantic.name,
                "machine_operands": tuple(
                    _operand_record(operand) for operand in instruction.operands
                ),
                "machine_reads": effect.reads,
                "machine_writes": effect.writes,
                "effect_domains": effect.effect_domains,
                "may_trap": effect.may_trap,
                "conditional": effect.conditional,
                "machine_function_rva": function_rva,
                "machine_instruction_index": instruction_index,
            }
            parent = (previous, "machine_state")
            graph.G.add_node(
                node,
                op=machine_process_operation(effect.semantic),
                label=instruction.token.name,
                type="machine.instruction",
                parents=[parent], children=[],
                input_roles=("machine_state",),
                output_roles=("machine_state",),
                schema_version=MACHINE_PROCESS_SCHEMA_VERSION,
                attributes=attributes,
                extra_args=attributes,
                tensor={},
                control={
                    "semantic": effect.semantic.name,
                    "conditional": effect.conditional,
                },
                source_span={
                    "machine_address": int(instruction.address),
                    "encoded_size": len(instruction.encoded),
                },
            )
            graph.G.add_edge(previous, node, role="machine_state")
            graph.G.nodes[previous]["children"].append((node, "machine_state"))
            address_nodes[int(instruction.address)] = node
            for operand in instruction.operands:
                if isinstance(operand, RelativeAddressOperand):
                    pending_control.append((node, int(operand.target_address)))
            previous = node
        graph.roots.append(previous)

    for source, target_address in pending_control:
        target = address_nodes.get(target_address)
        if target is None:
            graph.G.nodes[source]["control"].setdefault(
                "external_target_addresses", [],
            ).append(target_address)
            continue
        graph.G.add_edge(source, target, role="control-target")
        graph.G.nodes[source]["control"].setdefault(
            "target_nodes", [],
        ).append(target)

    graph.domain_shape = ()
    graph.G.graph.update({
        "source_ir": "MachineProgramGraph",
        "machine_process_schema_version": MACHINE_PROCESS_SCHEMA_VERSION,
        "machine_image_base": int(program.image.image_base),
        "machine_state_model": "complete-amd64-state",
        "numeric_projection": False,
    })
    return graph


__all__ = [
    "MACHINE_PROCESS_SCHEMA_VERSION", "install_machine_role_schemas",
    "machine_process_operation", "machine_process_role_schemas",
    "machine_program_to_process_graph",
]

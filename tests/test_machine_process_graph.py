from types import SimpleNamespace

from src.compiler.machine_process_graph import (
    machine_process_operation, machine_program_to_process_graph,
)
from src.compiler.machine_program_graph import decode_reachable_region
from src.compiler.machine_reference_vocabulary import (
    MachineSemanticToken, X86ReferenceDecoder,
)
from src.compiler.ssa_builder import process_graph_to_ssa_instrs


def _program(encoded: bytes):
    base = 0x140001000
    report = decode_reachable_region(
        X86ReferenceDecoder(), encoded, base_address=base,
    )
    function = SimpleNamespace(begin_rva=0x1000, end_rva=0x1000 + len(encoded))
    return SimpleNamespace(
        image=SimpleNamespace(image_base=0x140000000),
        functions=(SimpleNamespace(function=function, report=report),),
    )


def test_machine_program_ingests_directly_as_full_process_graph_schema():
    graph = machine_program_to_process_graph(_program(b"\x48\x01\xd8\xc3"))

    assert graph.G.graph == {
        "source_ir": "MachineProgramGraph",
        "machine_process_schema_version": 1,
        "machine_image_base": 0x140000000,
        "machine_state_model": "complete-amd64-state",
        "numeric_projection": False,
    }
    instructions = [
        data for _node, data in graph.G.nodes(data=True)
        if data.get("type") == "machine.instruction"
    ]
    assert [item["op"] for item in instructions] == [
        machine_process_operation(MachineSemanticToken.INTEGER_ADD),
        machine_process_operation(MachineSemanticToken.RETURN),
    ]
    add = instructions[0]
    assert add["parents"] == [(0, "machine_state")]
    assert add["input_roles"] == ("machine_state",)
    assert add["output_roles"] == ("machine_state",)
    assert add["attributes"]["semantic_family"] == "arithmetic.add"
    assert add["attributes"]["machine_encoded"] == "4801d8"
    assert add["attributes"]["machine_reads"]
    assert add["attributes"]["machine_writes"]
    assert len(add["attributes"]["machine_operands"]) == 2
    assert graph.role_schemas[add["op"]] == {
        "up": {"machine_state": "one"},
        "down": {"machine_state": "many"},
    }


def test_machine_control_target_is_structural_not_numeric_operand():
    # jne -2 loops to itself; both the state chain and control ownership remain.
    graph = machine_program_to_process_graph(_program(b"\x75\xfe"))
    instruction_node = next(
        node for node, data in graph.G.nodes(data=True)
        if data.get("type") == "machine.instruction"
    )
    assert graph.G.nodes[instruction_node]["parents"] == [(0, "machine_state")]
    assert graph.G.has_edge(instruction_node, instruction_node)
    assert graph.G.edges[instruction_node, instruction_node]["role"] == "control-target"
    assert graph.G.nodes[instruction_node]["control"]["target_nodes"] == [
        instruction_node,
    ]


def test_machine_family_survives_generic_process_graph_ssa_view():
    graph = machine_program_to_process_graph(_program(b"\x48\x01\xd8\xc3"))
    # The linear helper is only a view test here; retained control is handled
    # by the graph-to-section renderer.  Supply the already-linear state order
    # while excluding the structural control-target role from numeric inputs.
    graph.compute_levels = lambda **_kwargs: {
        node: index for index, node in enumerate(graph.G.nodes)
    }
    instructions = process_graph_to_ssa_instrs(graph)
    add = next(
        item for item in instructions
        if item.op == machine_process_operation(MachineSemanticToken.INTEGER_ADD)
    )
    assert add.attributes["semantic_family"] == "arithmetic.add"
    assert add.attributes["semantic_source_representation"] == "process-graph"
    assert add.attributes["semantic_representation"] == "repository-ssa"
    assert add.attributes["machine_encoded"] == "4801d8"

import networkx as nx
import pytest

from src.compiler.machine_reference_vocabulary import X86Register
from src.compiler.machine_turing_graph import (
    TuringOperatorToken,
    raise_binary_to_turing_graph,
)
from src.compiler.recursive_reduction import (
    TapeCostVector,
    assemble_scalar_machine_tape_program,
    estimate_terminal_tape_execution_cost,
    execute_machine_turing_graph,
)
from src.hardware.constants import REGISTERS


def test_binary_streams_directly_to_reduced_turing_operator_graph():
    # mov eax, 5; add eax, 3; and eax, 15; ret
    binary = bytes.fromhex(
        "b8 05 00 00 00 "
        "83 c0 03 "
        "83 e0 0f "
        "c3"
    )

    raised = raise_binary_to_turing_graph(binary, bit_width=32)

    assert raised.complete
    assert [
        instruction.semantic.name for instruction in raised.report.instructions
    ] == [
        "REGISTER_WRITE_IMMEDIATE",
        "INTEGER_ADD",
        "BITWISE_AND",
        "RETURN",
    ]
    assert raised.register_values[X86Register.RAX] == 8
    assert nx.is_directed_acyclic_graph(raised.operator_graph)
    assert {
        payload["op"] for _, payload in raised.operator_graph.nodes(data=True)
    } >= {"input", "nand", "slice", "mu"}
    assert all(
        isinstance(payload["token_id"], int)
        for _, payload in raised.operator_graph.nodes(data=True)
    )
    assert {
        payload["token_id"]
        for _, payload in raised.operator_graph.nodes(data=True)
        if payload["op"] == "nand"
    } == {int(TuringOperatorToken.NAND)}

    reduced = raised.topologically_reduce()
    assert nx.is_directed_acyclic_graph(reduced.graph)
    assert reduced.graph.number_of_nodes() < raised.operator_graph.number_of_nodes()
    assert reduced.verify_quotient(raised.operator_graph)
    assert reduced.graph.graph["register_outputs"]["RAX"] == reduced.node_map[
        raised.register_outputs[X86Register.RAX]
    ]

    broken = reduced.graph.copy()
    broken.remove_edge(*next(iter(broken.edges(keys=True))))
    assert not type(reduced)(broken, reduced.node_map).verify_quotient(
        raised.operator_graph
    )

    ssa = raised.to_ssa()
    assert len(ssa.instructions) == reduced.graph.number_of_nodes()
    assert ssa.register_outputs["RAX"].id == reduced.graph.graph[
        "register_outputs"
    ]["RAX"]
    assert any(instruction.op == "nand" for instruction in ssa.instructions)
    assert all(
        instruction.attributes["token_id"] >= 0
        for instruction in ssa.instructions
    )


def test_bfs_discovers_branch_topology_and_writer_emits_versioned_join_state():
    # mov eax, 1
    # je target
    # add eax, 1
    # jmp target
    # target: ret
    binary = bytes.fromhex(
        "b8 01 00 00 00 "
        "74 05 "
        "83 c0 01 "
        "eb 00 "
        "c3"
    )

    raised = raise_binary_to_turing_graph(
        binary, base_address=0x1000, bit_width=32,
    )

    assert raised.discovery_order == (
        0x1000, 0x1005, 0x1007, 0x100A, 0x100C,
    )
    branch_edges = {
        (source, target, payload["role"])
        for source, target, payload in raised.control_graph.edges(data=True)
        if source == 0x1005
    }
    assert branch_edges == {
        (0x1005, 0x1007, "fallthrough"),
        (0x1005, 0x100C, "branch"),
    }
    assert raised.report.complete
    assert raised.complete
    assert raised.unsupported_semantics == ()
    joins = [
        (node, payload)
        for node, payload in raised.operator_graph.nodes(data=True)
        if payload["metadata"].get("kind") == "state-join"
    ]
    assert len(joins) == 1
    join_node, join = joins[0]
    assert join["op"] == "mu"
    assert join["metadata"] == {
        "result_length": 32,
        "kind": "state-join",
        "join_address": 0x100C,
        "predecessors": (0x1005, 0x100A),
        "register": "RAX",
    }
    assert raised.register_outputs[X86Register.RAX] == join_node
    selectors = [
        payload["metadata"]
        for _, payload in raised.operator_graph.nodes(data=True)
        if payload["metadata"].get("kind") == "control-selector"
    ]
    assert selectors == [{
        "kind": "control-selector",
        "join_address": 0x100C,
        "predecessor": 0x100A,
        "ordinal": 1,
        "width": 32,
        "name": "join:0x100c:predecessor:0x100a",
    }]


def test_unknown_opcode_stops_before_turing_writer_invents_semantics():
    raised = raise_binary_to_turing_graph(b"\x0f\x0b", bit_width=32)

    assert not raised.complete
    assert not raised.report.complete
    assert raised.report.instructions == ()
    assert raised.operator_graph.number_of_nodes() == 0
    assert raised.control_graph.number_of_nodes() == 0


def test_control_cycle_is_preserved_and_requires_a_state_fixed_point():
    # mov eax, 0; loop: add eax, 1; jmp loop
    binary = bytes.fromhex(
        "b8 00 00 00 00 "
        "83 c0 01 "
        "eb fb"
    )

    raised = raise_binary_to_turing_graph(binary, bit_width=32)

    assert raised.report.complete
    assert nx.is_directed_acyclic_graph(raised.control_graph) is False
    assert not raised.complete
    assert len(raised.unsupported_semantics) == 1
    assert "loop-header state fixed points" in raised.unsupported_semantics[0][2]
    assert raised.operator_graph.number_of_nodes() == 0


def test_machine_owned_nand_cone_executes_to_physical_cassette_events():
    # mov eax, 5; xor eax, eax; ret
    raised = raise_binary_to_turing_graph(
        bytes.fromhex("b8 05 00 00 00 31 c0 c3"),
        bit_width=32,
    )

    execution = execute_machine_turing_graph(
        raised,
        output_register=X86Register.RAX,
        per_source_error_probability_upper_bound=1e-12,
    )

    assert execution.witness.halted
    assert set(execution.witness.outputs.values()) == {0}
    assert execution.program.storage_mode == "registers"
    assert len(execution.witness.events) == 11
    # The XOR at address 5 owns the ten emitted NAND instructions.  MOV's
    # initialized constant and RET's control event require no physical op.
    assert execution.physical_descendants(0) == ()
    assert execution.physical_descendants(5) == tuple(range(10))
    assert execution.physical_descendants(7) == ()
    assert all(
        execution.provenance.nodes[("physical", index)]["token_id"] == 3
        for index in range(10)
    )
    assert execution.witness.total_cost.tape_distance_frames > 0
    assert execution.witness.reliability.success_probability_lower_bound > 0


def test_lifted_bitwise_program_reaches_physical_tape_with_ownership():
    # mov eax, 5; or eax, 3; and eax, 15; ret
    raised = raise_binary_to_turing_graph(
        bytes.fromhex(
            "b8 05 00 00 00 "
            "83 c8 03 "
            "83 e0 0f "
            "c3"
        ),
        bit_width=32,
    )

    execution = execute_machine_turing_graph(
        raised,
        output_register=X86Register.RAX,
        per_source_error_probability_upper_bound=1e-12,
    )

    assert raised.register_values[X86Register.RAX] == 7
    assert set(execution.witness.outputs.values()) == {7}
    assert execution.witness.halted
    assert execution.physical_descendants(0) == ()
    assert execution.physical_descendants(5) == tuple(range(5))
    assert execution.physical_descendants(8) == (5, 6)
    assert execution.physical_descendants(11) == ()
    assert execution.witness.total_cost.operator_events == 8
    estimate = estimate_terminal_tape_execution_cost(execution.program)
    observed = execution.witness.execution_cost
    assert estimate.tape_distance_frames == observed.tape_distance_frames
    assert estimate.seeks == observed.seeks
    assert estimate.read_frames == observed.read_frames
    assert estimate.write_frames == observed.write_frames
    assert estimate.operator_events == observed.operator_events
    assert estimate.storage_frames == observed.storage_frames
    or_cost = execution.cost_for_instruction(5)
    and_cost = execution.cost_for_instruction(8)
    assert or_cost.operator_events == 5
    assert and_cost.operator_events == 2
    assert or_cost.signal_energy_frame_units > and_cost.signal_energy_frame_units
    assert execution.reliability_for_instruction(
        5,
        per_source_error_probability_upper_bound=1e-12,
    ).exposed_frame_sources == or_cost.noise_exposure_frame_sources


def test_structural_arguments_are_literals_not_interned_identity_edges():
    # ADD exercises length results and many small integer slice bounds.  Those
    # integers are commonly interned by CPython and must never become carrier
    # dependencies merely because their object identities coincide.
    raised = raise_binary_to_turing_graph(
        bytes.fromhex("b8 05 00 00 00 83 c0 03 c3"),
        bit_width=32,
    )
    graph = raised.operator_graph
    assert any(payload["op"] == "slice" for _, payload in graph.nodes(data=True))

    for node, payload in graph.nodes(data=True):
        op = payload["op"]
        incoming_positions = {
            edge["arg_pos"] for *_, edge in graph.in_edges(node, data=True)
        }
        literals = (payload.get("metadata") or {}).get("literal_args", {})
        if op == "slice":
            assert incoming_positions == {0}
            assert set(literals) == {1, 2}
        elif op in {"sigma_L", "sigma_R"}:
            assert incoming_positions == {0}
            assert set(literals) == {1}
        elif op == "zeros":
            assert incoming_positions == set()
            assert set(literals) == {0}


def test_lifted_add_scalarizes_to_nand_and_fits_the_spill_envelope():
    # add eax, 1; ret -- RAX remains a genuine input rather than a folded MOV.
    raised = raise_binary_to_turing_graph(
        bytes.fromhex("83 c0 01 c3"),
        bit_width=32,
    )
    assembly = assemble_scalar_machine_tape_program(
        raised,
        output_register=X86Register.RAX,
        input_register_values={X86Register.RAX: 41},
    )
    scalarized = assembly.scalarized
    output_bits = assembly.output_bits
    values = dict(assembly.input_bit_values)
    for node in nx.topological_sort(scalarized.graph):
        payload = scalarized.graph.nodes[node]
        if payload["op"] == "constant":
            values[node] = payload["metadata"]["value"]
        elif payload["op"] == "nand":
            parents = [
                source for source, _target, edge in sorted(
                    scalarized.graph.in_edges(node, data=True),
                    key=lambda item: item[2]["arg_pos"],
                )
            ]
            values[node] = 1 - (values[parents[0]] & values[parents[1]])
    result = assembly.pack_output(values)
    program = assembly.program

    assert result == 42
    assert {payload["op"] for _, payload in scalarized.graph.nodes(data=True)} <= {
        "input", "constant", "nand",
    }
    assert len(output_bits) == 32
    assert max(program.spill_slots.values()) < 64
    assert assembly.spill_slot_count == 34
    assert len(program.output_registers) == 3
    assert len(program.output_spill_slots) == 29
    assert assembly.tape_descendants(0) == tuple(
        range(len(program.instructions) - 1)
    )
    assert assembly.tape_descendants(3) == ()
    assert sum(assembly.opcode_counts.values()) == len(program.instructions)
    estimate = assembly.execution_cost_estimate
    assert estimate.operator_events == len(program.instructions)
    assert estimate.storage_frames == (
        program.tape_map.data_start
        + (REGISTERS + assembly.spill_slot_count) * program.bit_width
        + 1
    )
    assert estimate.tape_distance_frames > 0
    assert estimate.signal_energy_frame_units > 0
    assert estimate.noise_exposure_frame_sources > 0
    reliability = assembly.estimate_reliability(1e-12)
    assert reliability.exposed_frame_sources == (
        estimate.noise_exposure_frame_sources
    )
    assert reliability.success_probability_lower_bound > 0.999
    concurrency = assembly.concurrency_profile
    assert concurrency.operator_nodes == 1776
    assert concurrency.critical_path_operator_events == 258
    assert concurrency.maximum_parallel_operator_events == 32
    assert concurrency.average_available_parallelism == pytest.approx(
        1776 / 258,
    )
    assert concurrency.physical_parallel_lanes == 1
    assert estimate.peak_parallel_lanes == 1
    add_cost = assembly.cost_for_instruction(0)
    assert add_cost.operator_events == len(program.instructions) - 1
    assert assembly.cost_for_instruction(3) == TapeCostVector()
    assert assembly.reliability_for_instruction(
        0,
        per_source_error_probability_upper_bound=1e-12,
    ).exposed_frame_sources == add_cost.noise_exposure_frame_sources

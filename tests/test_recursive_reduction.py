import networkx as nx
import pytest

from src.compiler.recursive_reduction import (
    ReductionLayer,
    ReductionRank,
    ReductionRule,
    TapeCostVector,
    TapePlacement,
    analyze_graph_concurrency,
    assemble_nand_terminal_tape_program,
    bitops_turing_reduction_catalog,
    estimate_turing_tape_feasibility,
    estimate_tape_reliability,
    execute_object_method_source,
    execute_terminal_tape_program,
    reduce_bitops_process_graph,
)
from src.compiler.machine_turing_graph import raise_binary_to_turing_graph
from src.hardware.analog_spec import Opcode
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _process_graph(source: str) -> ProcessGraph:
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(source)
    return graph


def test_catalog_is_numeric_self_describing_and_well_founded():
    catalog = bitops_turing_reduction_catalog()
    rules = catalog.rules()

    assert {rule.source_spelling for rule in rules} == {
        "bitand", "bitor", "bitxor", "invert", "add", "sub", "mul",
    }
    assert all(isinstance(rule.token_id, int) for rule in rules)
    assert all(isinstance(rule.source_token_id, int) for rule in rules)
    assert all(rule.target_rank < rule.source_rank for rule in rules)
    assert all(rule.reducer == "BitOpsTranslator.apply_bits" for rule in rules)
    assert all(rule.target_token_ids for rule in rules)
    assert catalog.resolve_spelling(ReductionLayer.BITOPS, "add") is not None

    with pytest.raises(ValueError, match="strictly decrease"):
        ReductionRule(
            token_id=99,
            source_token_id=98,
            source_spelling="bad",
            source_rank=ReductionRank(ReductionLayer.BITOPS),
            target_rank=ReductionRank(ReductionLayer.BITOPS),
            input_roles=("value",),
        )


def test_concurrency_profile_preserves_independent_nand_frontiers():
    graph = nx.MultiDiGraph()
    for node in range(4):
        graph.add_node(node, op="input")
    for node in (4, 5, 6):
        graph.add_node(node, op="nand")
    graph.add_edge(0, 4, arg_pos=0)
    graph.add_edge(1, 4, arg_pos=1)
    graph.add_edge(2, 5, arg_pos=0)
    graph.add_edge(3, 5, arg_pos=1)
    graph.add_edge(4, 6, arg_pos=0)
    graph.add_edge(5, 6, arg_pos=1)

    profile = analyze_graph_concurrency(
        graph,
        output_nodes=(6,),
        physical_parallel_lanes=1,
    )

    assert profile.logical_nodes == 7
    assert profile.operator_nodes == 3
    assert profile.critical_path_operator_events == 2
    assert profile.operator_width_by_level == (2, 1)
    assert tuple(map(set, profile.operator_frontiers)) == ({4, 5}, {6})
    assert profile.operator_levels == {4: 1, 5: 1, 6: 2}
    assert profile.maximum_parallel_operator_events == 2
    assert profile.average_available_parallelism == pytest.approx(1.5)
    assert profile.serial_to_critical_path_ratio == pytest.approx(1.5)
    assert profile.physical_parallel_lanes == 1


def test_process_graph_bitops_reduction_retains_parent_child_morphism():
    source = _process_graph(
        """
def kernel(x, y):
    return x ^ y
"""
    )
    source_bitxor = next(
        node for node, payload in source.G.nodes(data=True)
        if payload["op"] == "bitxor"
    )

    artifact = reduce_bitops_process_graph(source, bit_width=4)

    assert artifact.verify_lineage()
    children = artifact.lineage[source_bitxor]
    assert children
    assert all(
        artifact.target.G.nodes[child]["control"]["source_node"]
        == source_bitxor
        for child in children
    )
    assert any(
        artifact.target.G.nodes[child]["op"] == "nand"
        for child in children
    )
    handoffs = [
        event for event in artifact.metagraph.snapshot().events
        if event.kind == "component-handoff"
        and event.detail["transformation"].startswith("reduction-rule:")
    ]
    assert handoffs
    journey = artifact.journey()
    assert journey.descendants(0, source_bitxor) == children
    assert journey.stages[1].rank < journey.stages[0].rank
    assert artifact.tape.complete
    assert artifact.tape.instruction_nodes
    assert artifact.tape.cost.operator_events == len(
        artifact.tape.instruction_nodes
    )
    assert artifact.tape.cost.tape_distance_frames > 0


def test_tape_cost_vector_distinguishes_memory_placements():
    graph = nx.MultiDiGraph(bit_width=4)
    graph.add_node(0, op="input", metadata={"result_length": 4})
    graph.add_node(1, op="input", metadata={"result_length": 4})
    graph.add_node(2, op="nand", metadata={"result_length": 4})
    graph.add_edge(0, 2, arg_pos=0)
    graph.add_edge(1, 2, arg_pos=1)

    compact = TapePlacement(
        offsets={0: 0, 1: 4, 2: 8},
        extents={0: 4, 1: 4, 2: 4},
        total_frames=12,
    )
    scattered = TapePlacement(
        offsets={0: 0, 1: 100, 2: 200},
        extents={0: 4, 1: 4, 2: 4},
        total_frames=204,
    )

    compact_report = estimate_turing_tape_feasibility(
        graph, placement=compact,
    )
    scattered_report = estimate_turing_tape_feasibility(
        graph, placement=scattered,
    )

    assert compact_report.complete and scattered_report.complete
    assert compact_report.cost.tape_distance_frames < (
        scattered_report.cost.tape_distance_frames
    )
    assert compact_report.cost.mechanical_work_units < (
        scattered_report.cost.mechanical_work_units
    )
    assert compact_report.cost.latency_seconds < (
        scattered_report.cost.latency_seconds
    )
    assert compact_report.cost.storage_frames < scattered_report.cost.storage_frames


def test_cost_algebra_serial_adds_work_and_parallel_takes_critical_time():
    left = TapeCostVector(
        read_frames=4,
        latency_seconds=2.0,
        mechanical_work_units=3.0,
        peak_parallel_lanes=2,
    )
    right = TapeCostVector(
        write_frames=5,
        latency_seconds=3.0,
        mechanical_work_units=7.0,
        peak_parallel_lanes=4,
    )

    serial = left.serial(right)
    parallel = left.parallel(right)

    assert serial.latency_seconds == 5.0
    assert parallel.latency_seconds == 3.0
    assert serial.mechanical_work_units == parallel.mechanical_work_units == 10.0
    assert parallel.peak_parallel_lanes == 4

    reliability = estimate_tape_reliability(
        serial,
        per_source_error_probability_upper_bound=0.25,
    )
    assert reliability.failure_probability_upper_bound == 0.0
    exposed = TapeCostVector(noise_exposure_frame_sources=3)
    bounded = estimate_tape_reliability(
        exposed,
        per_source_error_probability_upper_bound=0.1,
    )
    assert bounded.failure_probability_upper_bound == pytest.approx(0.3)
    assert bounded.success_probability_lower_bound == pytest.approx(0.7)


def test_x86_derived_turing_graph_has_a_direct_tape_feasibility_projection():
    # mov eax, 5; xor eax, eax; ret
    raised = raise_binary_to_turing_graph(
        bytes.fromhex("b8 05 00 00 00 31 c0 c3"),
        bit_width=32,
    )

    report = estimate_turing_tape_feasibility(raised.operator_graph)

    assert raised.complete and report.complete
    assert report.initialized_data_nodes
    assert report.instruction_nodes
    assert set(report.opcodes) == set(report.instruction_nodes)
    assert set(report.opcodes.values()) <= {opcode.value for opcode in Opcode}
    assert Opcode.NAND.value in report.opcodes.values()
    assert report.cost.storage_frames > 0
    assert report.cost.noise_exposure_frame_sources > 0


def test_four_input_nand_dag_spills_to_explicit_tape_slots_and_executes():
    graph = nx.MultiDiGraph(bit_width=4)
    for node, label in enumerate(("a", "b", "c", "d")):
        graph.add_node(node, op="input", label=label)
    graph.add_node(4, op="nand", label="nand")
    graph.add_node(5, op="nand", label="nand")
    graph.add_node(6, op="nand", label="nand")
    graph.add_edge(0, 4, arg_pos=0)
    graph.add_edge(1, 4, arg_pos=1)
    graph.add_edge(2, 5, arg_pos=0)
    graph.add_edge(3, 5, arg_pos=1)
    graph.add_edge(4, 6, arg_pos=0)
    graph.add_edge(5, 6, arg_pos=1)
    values = {0: 0b1100, 1: 0b1010, 2: 0b1111, 3: 0b0011}

    program = assemble_nand_terminal_tape_program(
        graph,
        bit_width=4,
        input_values=values,
        output_nodes=(6,),
    )
    witness = execute_terminal_tape_program(program)

    first = (~(values[0] & values[1])) & 0b1111
    second = (~(values[2] & values[3])) & 0b1111
    expected = (~(first & second)) & 0b1111
    assert program.storage_mode == "spilled"
    assert len(program.spill_slots) == len(graph)
    assert set(program.initial_spill_values) == {
        program.spill_slots[node] for node in values
    }
    assert Opcode.LOAD in {instruction.opcode for instruction in program.instructions}
    assert Opcode.STORE in {instruction.opcode for instruction in program.instructions}
    assert witness.halted
    assert witness.outputs == {6: expected}
    assert witness.execution_cost.operator_events == len(program.instructions)
    assert witness.execution_cost.tape_distance_frames > 0


def test_object_method_xor_executes_with_six_stage_event_provenance():
    executed_object = execute_object_method_source(
        """
class WordOps:
    def xor(self, x, y):
        return x ^ y
""",
        class_name="WordOps",
        method_name="xor",
        bit_width=4,
        input_values_by_name={"x": 0b1010, "y": 0b1100},
        source_filename="word_ops.py",
        per_source_error_probability_upper_bound=1e-12,
    )
    object_reduction = executed_object.object_reduction
    artifact = object_reduction.reduction
    source = artifact.source
    source_bitxor = next(
        node for node, payload in source.G.nodes(data=True)
        if payload["op"] == "bitxor"
    )
    executed = executed_object.execution
    program = executed.program
    witness = executed.witness

    assert witness.halted
    assert set(witness.outputs.values()) == {0b0110}
    assert witness.audio_samples > 0
    assert len(witness.events) == len(program.instructions)
    assert witness.events[-1].opcode == Opcode.HALT.value
    assert witness.events[-1].source_node is None
    physical_sources = {
        event.source_node for event in witness.events
        if event.source_node is not None
    }
    assert physical_sources == set(artifact.lineage[source_bitxor])
    assert all(
        artifact.target.G.nodes[node]["op"] == "nand"
        for node in physical_sources
    )
    assert set(program.node_registers.values()) <= {0, 1, 2}
    assert [stage.rank.layer for stage in executed.journey.stages] == [
        ReductionLayer.OBJECT,
        ReductionLayer.PROCESS,
        ReductionLayer.BITOPS,
        ReductionLayer.TURING,
        ReductionLayer.TAPE,
        ReductionLayer.PHYSICAL,
    ]
    physical_descendants = executed.journey.descendants(
        1,
        source_bitxor,
        target_stage=5,
    )
    assert set(physical_descendants) == {
        event.instruction_index
        for event in witness.events
        if event.source_node is not None
    }
    assert set(executed.journey.descendants(
        0,
        "WordOps.xor",
        target_stage=5,
    )) == set(physical_descendants)
    assert object_reduction.raised.identity.graph_identity == "WordOps.xor"
    assert artifact.source.G.graph["object_origin"]["source_filename"] == (
        "word_ops.py"
    )
    assert witness.execution_cost.operator_events == len(witness.events)
    assert witness.execution_cost.read_frames > 0
    assert witness.execution_cost.write_frames > 0
    assert witness.execution_cost.tape_distance_frames > 0
    assert witness.total_cost.read_frames >= witness.execution_cost.read_frames
    assert witness.total_cost.latency_seconds == pytest.approx(
        witness.audio_samples / 44_100,
    )
    assert witness.reliability.exposed_frame_sources == (
        witness.total_cost.noise_exposure_frame_sources
    )
    assert witness.reliability.success_probability_lower_bound > 0.0
    object_cost = executed.cost_for_ancestor(0, "WordOps.xor")
    assert object_cost.operator_events == len(physical_descendants)
    assert object_cost.tape_distance_frames > 0
    assert executed.reliability_for_ancestor(
        0,
        "WordOps.xor",
        per_source_error_probability_upper_bound=1e-12,
    ).exposed_frame_sources == object_cost.noise_exposure_frame_sources
    assert executed.concurrency_profile.maximum_parallel_operator_events > 1


def test_object_method_add_executes_through_visible_scalar_turing_stage():
    executed_object = execute_object_method_source(
        """
class WordOps:
    def add(self, x, y):
        return x + y
""",
        class_name="WordOps",
        method_name="add",
        bit_width=2,
        input_values_by_name={"x": 1, "y": 1},
        source_filename="word_add.py",
        per_source_error_probability_upper_bound=1e-12,
    )
    execution = executed_object.execution
    scalarized = execution.scalarized

    assert scalarized is not None
    assert execution.scalar_graph_ref is not None
    assert set(execution.witness.outputs.values()) == {2}
    assert execution.witness.halted
    assert len(execution.program.instructions) == 165
    assert [stage.rank.layer for stage in execution.journey.stages] == [
        ReductionLayer.OBJECT,
        ReductionLayer.PROCESS,
        ReductionLayer.BITOPS,
        ReductionLayer.TURING,
        ReductionLayer.TURING,
        ReductionLayer.TAPE,
        ReductionLayer.PHYSICAL,
    ]
    assert [stage.rank.structural_depth for stage in execution.journey.stages] == [
        0, 0, 0, 1, 0, 0, 0,
    ]
    physical_descendants = execution.journey.descendants(
        0,
        "WordOps.add",
        target_stage=6,
    )
    assert set(physical_descendants) == {
        event.instruction_index
        for event in execution.witness.events
        if event.source_node is not None
    }
    assert execution.physical_events_for_ancestor(
        0, "WordOps.add",
    ) == tuple(sorted(set(physical_descendants)))
    object_cost = execution.cost_for_ancestor(0, "WordOps.add")
    assert object_cost.operator_events == len(set(physical_descendants))
    assert object_cost.tape_distance_frames > 0
    object_reliability = execution.reliability_for_ancestor(
        0,
        "WordOps.add",
        per_source_error_probability_upper_bound=1e-12,
    )
    assert object_reliability.exposed_frame_sources == (
        object_cost.noise_exposure_frame_sources
    )
    assert any(
        (payload.get("metadata") or {}).get("literal_args")
        for _, payload in executed_object.object_reduction.reduction.target.G.nodes(
            data=True,
        )
    )
    concurrency = analyze_graph_concurrency(
        scalarized.graph,
        output_nodes=tuple(
            bit
            for bits in scalarized.output_bits.values()
            for bit in bits
        ),
    )
    assert concurrency.maximum_parallel_operator_events > 1
    assert concurrency.physical_parallel_lanes == 1
    assert execution.concurrency_profile == concurrency
    assert execution.witness.total_cost.tape_distance_frames > 0
    assert execution.witness.reliability.success_probability_lower_bound > 0


def test_spilled_terminal_maps_outputs_beyond_three_registers():
    graph = nx.MultiDiGraph()
    for node, value in enumerate((1, 0, 1, 1)):
        graph.add_node(
            node,
            op="constant",
            metadata={"kind": "constant", "value": value, "result_length": 1},
        )
    program = assemble_nand_terminal_tape_program(
        graph,
        bit_width=1,
        input_values={},
        output_nodes=(0, 1, 2, 3),
    )
    assert program.storage_mode == "spilled"
    assert len(program.output_registers) == 3
    assert len(program.output_spill_slots) == 1
    assert set(program.output_registers) | set(program.output_spill_slots) == {
        0, 1, 2, 3,
    }

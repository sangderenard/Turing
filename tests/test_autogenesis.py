import numpy as np

from src.compiler.autogenesis import (
    compile_source_autogenesis,
    compile_sympy_autogenesis,
)


def test_sympy_string_autogenesis_uses_process_graph_and_ssa_route():
    result = compile_sympy_autogenesis("sin(x) + (x + 2*y)**2")
    snapshot = result.metagraph.snapshot()
    process_graph = next(
        graph for graph in snapshot.graphs if graph.stage == "process-graph"
    )

    assert result.aot.G.graph["symbolic_source"] == "sympy"
    assert len(result.ssa.instructions) == result.aot.G.number_of_nodes()
    assert sum(
        component.ref.graph_id == process_graph.id
        for component in snapshot.components
    ) == result.aot.G.number_of_nodes()
    assert {graph.stage for graph in snapshot.graphs} >= {
        "process-graph", "ssa", "ir-package"
    }
    assert any(
        event.detail.get("transformation") == "sympy-process-graph-to-ssa"
        for event in snapshot.events
    )


def test_sympy_pi_remains_an_ssa_operation_with_bounded_policy():
    result = compile_sympy_autogenesis(
        "sin(pi*x)", pi_solver="machin", pi_epsilon=1.0e-10,
    )
    pi_instruction = next(
        instruction for instruction in result.ssa.instructions
        if instruction.op == "Pi"
    )

    assert pi_instruction.attributes["constant_identity"] == "pi"
    assert pi_instruction.attributes["constant_solver"] == "machin"
    assert pi_instruction.attributes["absolute_error_bound"] <= 1.0e-10
    assert pi_instruction.res.dtype == "float64"


def test_autogenesis_without_entrypoint_compiles_whole_source_to_ssa():
    result = compile_source_autogenesis(
        """
class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value = self.value + 1

def utility(value):
    return value * 2
""",
        None,
        {},
        final_target=None,
    )

    counter = result.ssa.module.class_table.by_identity("Counter")
    assert result.aot is None
    assert counter is not None
    assert {method.name for method in counter.methods} == {
        "__init__", "increment"
    }
    assert any(name.endswith("__utility") for name in result.ssa.exports)
    assert {graph.stage for graph in result.metagraph.snapshot().graphs} >= {
        "process-graph", "control-ir", "ssa", "ir-package"
    }


def test_real_compiler_run_records_ingestion_precompile_ssa_and_package_handoffs():
    result = compile_source_autogenesis(
        """
def kernel(x, gain):
    scaled = x * gain
    shifted = scaled + 1.0
    return shifted.sin()
""",
        "kernel",
        {
            "x": np.ones(4),
            "gain": np.full(4, 2.0),
        },
    )

    snapshot = result.metagraph.snapshot()
    stages = {graph.stage for graph in snapshot.graphs}
    transformations = {
        event.detail.get("transformation")
        for event in snapshot.events
        if event.kind == "component-handoff"
    }
    assert result.ssa.complete
    assert {
        "process-graph",
        "precompile",
        "ssa",
        "ir-package",
        "backend-adapter:webgl",
        "backend:webgl",
    } <= stages
    assert "process-graph-to-precompile" in transformations
    assert "precompile-to-ssa" in transformations
    assert "ssa-to-package" in transformations
    assert "ssa-to-webgl-adapter" in transformations
    assert "webgl-adapter-to-glsl-es" in transformations
    webgl_handoffs = [
        event
        for event in snapshot.events
        if event.kind == "component-handoff"
        and event.detail.get("transformation")
        == "webgl-adapter-to-glsl-es"
    ]
    assert webgl_handoffs
    assert all(
        event.detail.get("granularity") == "exact-value"
        for event in webgl_handoffs
    )
    assert result.final_artifact.complete


def test_autogenesis_retains_and_legalizes_reachable_object_geometry():
    result = compile_source_autogenesis(
        """
class Gain:
    def __init__(self, gain):
        self.gain = gain

    def apply(self, value):
        return value * self.gain

def train(value):
    return Gain(2.0).apply(value)
""",
        "train",
        {"value": 3.0},
        final_target=None,
    )

    module = result.ssa.module
    assert module.class_table.by_identity("Gain") is not None
    assert sum(len(table.records) for table in module.record_tables.values()) == 2
    assert not any(
        instruction.op.casefold() == "getattr"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )
    assert not any(
        record.resolution == "unresolved"
        for records in module.call_table.values()
        for record in records
    )


def test_autogenesis_propagates_returned_record_storage_through_factory_call():
    result = compile_source_autogenesis(
        """
class Gain:
    def __init__(self, gain):
        self.gain = gain

    def apply(self, value):
        return value * self.gain

def make_gain():
    return Gain(2.0)

def train(value):
    gain = make_gain()
    return gain.apply(value)
""",
        "train",
        {"value": 3.0},
        final_target=None,
    )

    module = result.ssa.module
    train_symbol = next(
        name for name in module.functions
        if name.startswith("train__train")
    )
    train_records = tuple(module.record_tables[train_symbol].records.values())
    assert len(train_records) == 1
    assert train_records[0].identity == "Gain"
    assert train_records[0].fields[0].storage_identity == "Gain.gain"
    assert not any(
        instruction.op.casefold() == "getattr"
        for function in module.functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
    )


def test_xor_whole_object_call_table_has_no_unresolved_occurrences():
    from pathlib import Path

    source = Path("examples/xor_project/train_xor.py").read_text()
    result = compile_source_autogenesis(
        source,
        "train",
        {},
        final_target=None,
        extraction_contract="extraction_contracts/program_extraction.yaml",
    )

    calls = tuple(
        record
        for records in result.ssa.module.call_table.values()
        for record in records
    )
    assert calls
    assert not [
        record for record in calls if record.resolution == "unresolved"
    ]
    assert {record.resolution for record in calls} <= {
        "native_call", "decomposed"
    }

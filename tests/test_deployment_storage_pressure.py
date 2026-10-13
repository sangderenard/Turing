"""Small ownership/eviction regressions, never a full bootstrap or orbital run."""
import dataclasses

import pytest

from src.compiler import memory_regulation as regulation
from src.compiler.identity_concordance import IdentityBook
from src.compiler.live_tiles import LiveTileArena, TiledDict, live_digraph
from src.compiler.work_contract import active_contract, scoped_work_contract


@pytest.fixture(autouse=True)
def small_working_contract():
    with scoped_work_contract(dataclasses.replace(
        active_contract(), memory_budget_bytes=1 << 30,
        memory_elastic_available_percent=0,
    )):
        yield


def test_initial_adoption_evicts_payloads_as_they_are_registered(monkeypatch):
    resident = [1]
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: resident[0])
    monkeypatch.setattr(regulation, "system_available_memory_bytes", lambda: None)
    with scoped_work_contract(dataclasses.replace(active_contract(),
            memory_budget_bytes=100, live_tile_rows=4, live_resident_tiles=1,
            live_bucket_rows=4, live_decoded_budget_bytes=1024)):
        regulator, token, owns = regulation.begin_regulation(IdentityBook())
        try:
            graph = live_digraph()
            graph.add_nodes_from((i, {"payload": str(i) + "repeat " * 100})
                                 for i in range(64))
            assert regulator.live_arena is None
            resident[0] = 500
            regulator.pressure_checkpoint("selection fixture")
            owner = regulator.live_arena
            assert owner._payload_hot_peak < 20
            assert isinstance(graph._node, TiledDict)
            owner.flush()
            assert graph.nodes[53]["payload"] == "53" + "repeat " * 100
            assert owner.stats().disk_tiles > 0
        finally:
            regulator.budget_bytes = None
            regulation.end_regulation(regulator, token, releases_book_state=owns)


def test_generated_class_keeps_exact_plans_in_compressed_storage(monkeypatch):
    from src.compiler.glsl_deployment_strategy import ProcessGraphGLSLDeployment
    from src.compiler.process_graph_fusion import ScheduledProcessGraphDispatchPlan
    from src.compiler.loop_composer import LoopDescriptor, LoopPlan, LoopStrategy
    from src.compiler.loop_ir import ConditionDomain, LoopDomainKind, SemanticLoop
    resident = [1]
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: resident[0])
    monkeypatch.setattr(regulation, "system_available_memory_bytes", lambda: None)
    with scoped_work_contract(dataclasses.replace(active_contract(),
            memory_budget_bytes=100, live_tile_rows=4, live_resident_tiles=1,
            live_bucket_rows=4, live_decoded_budget_bytes=1024)):
        regulator, token, owns = regulation.begin_regulation(IdentityBook())
        try:
            dispatch = ScheduledProcessGraphDispatchPlan((), (), (), (),
                                                        {i: (i, 0) for i in range(64)})
            descriptor = LoopDescriptor(7, "While", "", "condition",
                                        tuple(range(64)), (65,))
            semantic = SemanticLoop(7, LoopDomainKind.CONDITION,
                                    ConditionDomain(65), tuple(range(64)))
            loop = LoopPlan(descriptor, LoopStrategy.NATIVE_SOURCE, "authored loop", semantic)
            shell_type = type("SelectedStorageFixture", (ProcessGraphGLSLDeployment,),
                              {"dispatch_plan": dispatch, "loop_plans": (loop,),
                               "deployment_batches": dispatch.node_locations,
                               "function_shell_types": {3: ProcessGraphGLSLDeployment}})
            regulator.register_live_root(shell_type)
            resident[0] = 500
            regulator.pressure_checkpoint("selected plan fixture")
            owner = regulator.live_arena
            assert isinstance(shell_type.function_shell_types, TiledDict)
            assert isinstance(dispatch.node_locations, TiledDict)
            assert shell_type.deployment_batches is dispatch.node_locations
            assert object.__getattribute__(descriptor, "__dict__").get("_tile_arena") is owner
            assert object.__getattribute__(semantic, "__dict__").get("_tile_arena") is owner
            class_identity, dispatch_identity, loop_identity = id(shell_type), id(dispatch), id(loop)
            owner.flush()
            assert (id(shell_type), id(shell_type.dispatch_plan), id(shell_type.loop_plans[0])) == (
                class_identity, dispatch_identity, loop_identity)
            assert shell_type.dispatch_plan.node_locations[53] == (53, 0)
            assert tuple(shell_type.loop_plans[0].loop.body_nodes) == tuple(range(64))
            assert shell_type.loop_plans[0].semantic.domain.condition_value_id == 65
            assert shell_type.function_shell_types[3] is ProcessGraphGLSLDeployment
            assert owner.stats().payload_reloads > 0
        finally:
            regulator.budget_bytes = None
            regulation.end_regulation(regulator, token, releases_book_state=owns)


def test_decoded_front_cannot_expand_when_system_reserve_is_exhausted(monkeypatch):
    gib = 1 << 30
    owner = LiveTileArena(decoded_budget_bytes=128 << 20)
    owner.compress_resident_ram()
    owner.expand_decoded_front(gib)
    regulator = regulation.MemoryRegulator(IdentityBook(), budget_bytes=10 * gib,
        compress_policy=True, tile_disk_policy=True, live_arena=owner)
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 8 * gib)
    monkeypatch.setattr(regulation, "system_available_memory_bytes", lambda: gib)
    regulator.pressure_checkpoint("paged resident footprint fixture")
    assert owner.decoded_budget_bytes == owner.base_decoded_budget_bytes


def test_late_networkx_adoption_keeps_new_node_and_edge_writes(tmp_path):
    import networkx as nx
    import pickle
    from src.compiler.live_tiles import adopt_digraph
    graph = nx.DiGraph()
    graph.add_edge(1, 2, role="original")
    owner = LiveTileArena(tile_rows=4, hot_tiles=2, bucket_rows=4, directory=tmp_path)
    owner.compress_resident_ram()
    identity = id(graph)
    adopt_digraph(graph, owner)
    graph.add_node(3, type="Store", children=[])
    graph.add_edge(2, 3, role="result")
    owner.flush()
    assert id(graph) == identity
    assert graph.nodes[3]["type"] == "Store"
    assert graph._succ[2][3] is graph._pred[3][2]
    graph._succ[2][3]["role"] = "updated"
    assert graph._pred[3][2]["role"] == "updated"
    restored = pickle.loads(pickle.dumps(graph))
    assert restored.nodes[3]["type"] == "Store"
    assert restored.edges[2, 3]["role"] == "updated"


@pytest.mark.parametrize("source", (
    "def twice(x):\n    return x * 2.0\n\ndef root(x):\n    return twice(x) + 1.0\n",
    "def twice(x):\n    for index in range(2):\n        x = x + 1.0\n    return x\n\ndef root(x):\n    return twice(x) + 1.0\n",
), ids=("calls", "loop"))
def test_source_selection_checks_pressure_and_keeps_copied_graphs_readable(source, tmp_path, monkeypatch):
    from src.compiler import glsl_deployment_strategy as strategy
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
    )
    in_selection = [False]
    selected = []
    ordinary_selection = strategy.strategize_shell_deployment
    def resident():
        regulator = regulation._ACTIVE.get()
        if not in_selection[0]:
            return 1
        # Simulate a breach that cold-tail adoption actually resolves. The
        # separate storage tests exercise the one-tile maximum tier; this
        # source/native gate uses a bounded active cache over compressed data.
        return 500 if regulator.live_arena is None else 90

    monkeypatch.setattr(regulation, "process_rss_bytes", resident)
    monkeypatch.setattr(regulation, "system_available_memory_bytes", lambda: None)

    def observe_selection(*args, **kwargs):
        in_selection[0] = True
        shell_type = ordinary_selection(*args, **kwargs)
        owner = regulation._ACTIVE.get().live_arena
        assert owner is not None, "selection completed without checking pressure"
        owner.flush(disk=True)
        graph = shell_type.process_graph
        assert isinstance(graph.G._node, TiledDict)
        assert isinstance(shell_type.dispatch_plan.node_locations, TiledDict)
        selected.append(shell_type)
        return shell_type

    monkeypatch.setattr(strategy, "strategize_shell_deployment", observe_selection)
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml").with_program_abi({
        "records": {}, "bindings": [],
        "values": [{"function": "*", "parameter": "x", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "python_type": "builtins.float"}],
    })
    with scoped_work_contract(dataclasses.replace(active_contract(), memory_budget_bytes=100,
            live_tile_rows=8, live_resident_tiles=32, live_bucket_rows=8,
            live_decoded_budget_bytes=64 << 10)):
        module, outputs, exports = lower_ast_source_to_ssa(source, "root",
            extraction_contract=contract, runtime_closure_only=True,
            progress=lambda _: None, progress_bars=False)
    owner = object.__getattribute__(module, "__dict__")["_tile_arena"]
    snapshots = [(shell, shell.process_graph, tuple(shell.process_graph.G)) for shell in selected]
    owner.flush()
    for shell, graph, nodes in snapshots:
        assert shell.process_graph is graph
        assert tuple(graph.G) == nodes
    report = concordance_report(module)
    assert report.splitlines()[0].endswith("0 finding(s)"), report
    artifact = emit_ssa_function_to_llvm(module, exports[0])
    assert not artifact.shortfalls
    artifact = compile_artifact(artifact, directory=tmp_path, optimization="O0")
    function = module.functions[exports[0]]
    parameter = dict(function.metadata["parameter_names"])["x"]
    namespace = {}
    exec(source, namespace)
    for value in (-2.5, 0.0, 3.25):
        execution = prepare_artifact_execution(artifact, {parameter: value}).run()
        actual = execution.buffers[outputs[function.name][0].id].reshape(-1)[0]
        assert actual == namespace["root"](value)


def test_selection_roots_stay_plain_without_pressure(monkeypatch):
    from src.compiler.glsl_deployment_strategy import ProcessGraphGLSLDeployment
    from src.compiler.process_graph_fusion import ScheduledProcessGraphDispatchPlan
    monkeypatch.setattr(regulation, "process_rss_bytes", lambda: 1)
    monkeypatch.setattr(regulation, "system_available_memory_bytes", lambda: None)
    regulator, token, owns = regulation.begin_regulation(IdentityBook())
    try:
        dispatch = ScheduledProcessGraphDispatchPlan((), (), (), (), {3: (0, 1)})
        shell = type("DecodedSelectionFixture", (ProcessGraphGLSLDeployment,),
                     {"dispatch_plan": dispatch, "function_shell_types": {}})
        regulator.register_live_root(shell)
        regulator.pressure_checkpoint("ordinary selection")
        assert regulator.live_arena is None
        assert type(dispatch.node_locations) is dict
        assert type(shell.function_shell_types) is dict
    finally:
        regulation.end_regulation(regulator, token, releases_book_state=owns)

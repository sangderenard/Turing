from __future__ import annotations

import json
import ast
import pickle
import dataclasses
import subprocess
import sys
import weakref
from pathlib import Path

import networkx as nx
import pytest

from src.compiler import project_compilation_product as product
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


SOURCE = '''
def left(x):
    return x * 2.0 + 1.0

def right(x):
    return x * 3.0 - 1.0

def root(x):
    return left(x) + right(x)

def unrelated(x):
    return x / 7.0
'''


def test_native_growth_resumes_one_cut_per_stage_and_executes_binary_calls(tmp_path):
    import ctypes
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "shape": [], "python_type": "builtins.float"}
                   for name in ("left", "right", "root", "unrelated")],
        "records": {}, "bindings": [],
    })
    planned = product.compile_partitioned_source(
        SOURCE, tmp_path / "growth", extraction_contract=contract,
        interface_tiles=True, plan_only=True, max_unit_source_nodes=8,
    )
    assert not (tmp_path / "growth/product").exists()
    products = []
    events = []
    for stage in range(4):
        manifest = product.compile_resolved_process_graph_plan(
            planned["resolved_process_graph"], planned["process_graph_unit_plan"],
            tmp_path / "growth/product", jobs=2, max_stage_units=1,
            worker_resident_reservation_bytes=256 * 1024 ** 2,
            max_worker_memory_bytes=4 * 1024 ** 3, progress=events.append,
        )
        assert manifest["stage_admitted_units"] == 1
        assert len(manifest["stage_admitted_unit_indices"]) == 1
        assert manifest["stage_admitted_unit_indices"][0] in {
            unit["unit_index"] for unit in manifest["units"]
        }
        if products:
            assert manifest["stage_admitted_unit_indices"][0] not in {
                unit["unit_index"] for unit in products[-1]["units"]
            }
        assert len(manifest["units"]) == stage + 1
        assert manifest["counts"]["failed"] == manifest["counts"]["partial"] == 0
        products.append(manifest)
    assert [event["qualified_names"][0] for event in events
            if event["stage"] == "resolved_unit_start"][:3] in (
                ["left", "right", "root"], ["right", "left", "root"])
    assert products[-1]["status"] == "terminal"
    assert len({unit["worker_process_id"] for unit in products[-1]["units"]}) == 4
    # Resumption retained the accepted leaf artifact rather than recompiling it.
    first = products[0]["units"][0]
    retained = next(unit for unit in products[-1]["units"] if unit["unit_index"] == first["unit_index"])
    assert retained["artifact_sha256"] == first["artifact_sha256"]
    assert retained["worker_process_id"] == first["worker_process_id"]
    assert retained["growth_priority"] == first["growth_priority"]
    assert all(row["remaining_source_nodes"] == 0 for row in products[-1]["growth_priorities"])
    module, outputs, _exports = product.load_partitioned_source_interfaces(products[-1])
    assert all(not function.blocks for function in module.functions.values())
    root = next(function for function in module.functions.values()
                if function.metadata.get("source_qualified_name") == "root")
    proxy = emit_ssa_module_to_c(module, root.name, entry_name="growth_check")
    assert proxy.complete and len(proxy.linked_objects) == 3
    built = proxy.compile(tmp_path / "check")
    library = ctypes.CDLL(str(built.library_path))
    entry = library.growth_check
    entry.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong)]
    for value in (-3.0, 0.0, 21.0):
        incoming, outgoing = ctypes.c_double(value), ctypes.c_double()
        pointers = (ctypes.c_void_p * 2)(ctypes.addressof(incoming), ctypes.addressof(outgoing))
        entry(pointers, (ctypes.c_longlong * 1)(0))
        assert outgoing.value == value * 5.0
    # A same-path modified graph cannot be mistaken for the frozen input.
    Path(planned["resolved_process_graph"]).write_bytes(b"changed input")
    with pytest.raises(ValueError, match="pinned inputs"):
        product.compile_resolved_process_graph_plan(
            planned["resolved_process_graph"], planned["process_graph_unit_plan"],
            tmp_path / "growth/product", max_stage_units=1,
        )


def test_ssa_growth_composes_interiors_before_one_native_emission(tmp_path):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    import ctypes
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "scalar", "dtype": "float64",
                    "rank": 0, "shape": [], "python_type": "builtins.float"}
                   for name in ("left", "right", "root", "unrelated")],
        "records": {}, "bindings": [],
    })
    planned = product.compile_partitioned_source(
        SOURCE, tmp_path / "ssa-growth", extraction_contract=contract,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=32,
    )
    events = []
    stages = []
    for _stage in range(2):
        manifest = product.compile_resolved_process_graph_plan(
            planned["resolved_process_graph"], planned["process_graph_unit_plan"],
            tmp_path / "ssa-growth/product", max_stage_units=2, jobs=2,
            worker_resident_reservation_bytes=256 << 20, progress=events.append,
        )
        assert manifest["counts"]["failed"] == manifest["counts"]["partial"] == 0, manifest
        assert all(unit["native_interface"] is None for unit in manifest["units"])
        stages.append(manifest)
    assert stages[0]["concurrent_workers_peak"] == 2
    assert all(stage["stage_admitted_units"] == 2 for stage in stages)
    assert all(len(stage["stage_admitted_unit_indices"]) == 2 for stage in stages)
    starts = [event for event in events if event["stage"] == "resolved_unit_start"]
    assert {event["qualified_names"][0] for event in starts[:2]} == {"left", "right"}
    assert manifest["status"] == "terminal"
    caller = next(unit for unit in manifest["units"] if unit["qualified_names"] == ["root"])
    assert len(caller["composed_ssa_units"]) == 2
    caller_path = tmp_path / "ssa-growth/product/units" / f'unit_{caller["unit_index"]:03d}' / caller["artifact"]
    enclosure, published, roots = product._load_pickle_artifact(caller_path)
    assert len(enclosure.functions) >= 3
    assert all(function.blocks for function in enclosure.functions.values())
    module, outputs, exports = product.load_composed_source_units(manifest)
    from src.compiler.identity_concordance import CorrelationTable, _UNSOURCED_KINDS
    assert not [finding for finding in CorrelationTable.build(module).findings(module)
                if finding.kind not in _UNSOURCED_KINDS]
    # The composition audit consumes exact component owners and signatures.
    original_owners = module.metadata["ssa_composition_function_books"]
    module.metadata["ssa_composition_function_books"] = {}
    assert any(finding.kind == "ssa-composition-book-owner-disagreement"
               for finding in CorrelationTable.build(module).findings(module))
    module.metadata["ssa_composition_function_books"] = original_owners
    signature = module.functions[roots[0]].metadata["repository_ssa_signature"]
    original_arguments = signature["argument_ids"]
    signature["argument_ids"] = (999999,)
    assert any(finding.kind == "repository-ssa-signature-disagreement"
               for finding in CorrelationTable.build(module).findings(module))
    signature["argument_ids"] = original_arguments
    entry = roots[0]
    assert entry in exports
    artifact = emit_ssa_module_to_c(module, entry, entry_name="composed_check")
    assert artifact.complete and not artifact.linked_objects
    artifact.compile(tmp_path / "one-native-unit")
    library = ctypes.CDLL(str(artifact.library_path))
    call = library.composed_check
    call.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong)]
    for value in (-3.0, 0.0, 21.0):
        incoming, outgoing = ctypes.c_double(value), ctypes.c_double()
        pointers = (ctypes.c_void_p * 2)(ctypes.addressof(incoming), ctypes.addressof(outgoing))
        call(pointers, (ctypes.c_longlong * 1)(0))
        assert outgoing.value == value * 5.0


def test_public_partitioned_source_accepts_ssa_composition(tmp_path):
    from src.compiler.extraction_contract import ExtractionContract
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "shape": [],
                    "python_type": "builtins.float"}
                   for name in ("left", "right", "root", "unrelated")],
        "records": {}, "bindings": [],
    })
    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE, partitioned_directory=tmp_path / "public-ssa",
        extraction_contract=contract,
        partitioned_options={"ssa_composition": True, "jobs": 1,
                             "max_unit_source_nodes": 32},
        progress=lambda message: None,
    )
    assert module.metadata["composed_source_product"]["counts"]["compiled-unverified"] == 4
    assert all(function.blocks for function in module.functions.values())
    assert exports and outputs


def _inputs(tmp_path):
    from src.compiler.extraction_contract import ExtractionContract
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "shape": [],
                    "python_type": "builtins.float"}
                   for name in ("left", "right", "root", "unrelated")],
        "records": {}, "bindings": [],
    })
    graphs, plans = [], []
    lower_ast_source_to_ssa(
        SOURCE, None, name="scoped_units", progress=lambda message: None,
        extraction_contract=contract,
        resolved_process_graph_sink=graphs.append,
        compilation_unit_plan_sink=plans.append,
        stop_after_compilation_unit_plan=True,
    )
    graph_path, plan_path = tmp_path / "resolved.pkl", tmp_path / "plan.json"
    with graph_path.open("wb") as stream:
        product._dump_resolved_process_graph(graphs[0], stream)
    plan_path.write_text(json.dumps(plans[0]), encoding="utf-8")
    return graph_path, plan_path, plans[0], graphs[0]


def test_unit_worker_does_not_load_unrelated_function_bodies(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell
    graph_path, plan_path, plan, original = _inputs(tmp_path)
    index = next(i for i, unit in enumerate(plan["units"])
                 if unit["qualified_names"] == ["left"])
    selected = set(plan["units"][index]["function_references"])
    original_counts = {entry.reference.address: len(entry.graph.G)
                       for entry in original.function_table if entry.graph is not None}

    class ObservedInput(Exception):
        pass

    def observe(graph, references, **kwargs):
        retained = {entry.reference.address: len(entry.graph.G)
                    for entry in graph.function_table if entry.graph is not None}
        assert set(retained) == selected, retained
        assert all(retained[ref] == original_counts[ref] for ref in retained)
        assert len(graph.G) == 0
        raise ObservedInput()

    monkeypatch.setattr(fortran_c_shell, "lower_resolved_process_graph_unit_to_ssa", observe)
    with pytest.raises(ObservedInput):
        product.compile_resolved_process_graph_unit(
            graph_path, plan_path, index, tmp_path / "unit",
        )


def test_published_cuts_preserve_exact_nodes_edges_scopes_and_addresses(tmp_path):
    _graph_path, _plan_path, plan, original = _inputs(tmp_path)
    events = []
    published = product.publish_resolved_process_graph_inputs(
        original, plan, tmp_path / "cuts", progress=events.append,
    )
    snapshots = [event for event in events
                 if event["stage"] == "resolved_input_snapshot"]
    assert [event["completed_count"] for event in snapshots] == list(
        range(len(plan["units"]) + 1)
    )
    assert all(event["count"] == len(plan["units"]) for event in snapshots)
    assert events[-1] == {"stage": "resolved_input_complete",
                          "count": len(plan["units"])}
    for item in published["scoped_inputs"]:
        cut = product._load_pickle_artifact(item["path"])
        expected = set(product._resolved_unit_references(plan["units"], item["unit_index"]))
        assert len(cut.G) == 0
        assert [entry.reference for entry in cut.function_table] == [entry.reference for entry in original.function_table]
        for entry in cut.function_table:
            if entry.reference.address not in expected:
                assert entry.graph is None
                continue
            source = original.function_table.entry(entry.reference).graph
            assert entry.graph.function_table is cut.function_table
            assert set(entry.graph.G) == set(source.G)
            for node_id in source.G:
                before, after = source.G.nodes[node_id], entry.graph.G.nodes[node_id]
                assert {key: value for key, value in after.items() if key != "expr_obj"} == {
                    key: value for key, value in before.items() if key != "expr_obj"}
                if isinstance(before.get("expr_obj"), ast.AST):
                    assert ast.dump(after["expr_obj"], include_attributes=True) == ast.dump(before["expr_obj"], include_attributes=True)
            assert entry.graph.G.edges == source.G.edges
            assert entry.graph.G.graph["lexical_read_scope"] == source.G.graph["lexical_read_scope"]
        assert item["sha256"] == product._file_sha256(Path(item["path"]))
    assert all(entry.graph is not None for entry in original.function_table)


def test_published_cut_retains_cached_property_function_identity(tmp_path):
    _graph_path, _plan_path, plan, original = _inputs(tmp_path)
    selected = next(
        entry.graph for entry in original.function_table
        if entry.qualified_name == "left"
    )
    selected.G.graph["descriptor_function"] = nx.Graph.nodes.func

    published = product.publish_resolved_process_graph_inputs(
        original, plan, tmp_path / "cuts",
    )
    item = next(
        item for item in published["scoped_inputs"]
        if "left" in plan["units"][item["unit_index"]]["qualified_names"]
    )
    cut = product._load_pickle_artifact(item["path"])
    restored = cut.function_table.entry("left").graph
    assert restored.G.graph["descriptor_function"] is nx.Graph.nodes.func


def test_scoped_worker_never_reads_complete_archive_and_rejects_drift(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell
    graph_path, plan_path, plan, original = _inputs(tmp_path)
    published = product.publish_resolved_process_graph_inputs(original, plan, tmp_path / "cuts")
    index = next(i for i, unit in enumerate(plan["units"]) if unit["qualified_names"] == ["left"])
    load = product._load_pickle_artifact
    read_paths = []

    def observed_load(path):
        assert Path(path) != graph_path
        read_paths.append(Path(path))
        return load(path)

    class ObservedInput(Exception):
        pass

    def observed_lower(*args, **kwargs):
        raise ObservedInput()

    monkeypatch.setattr(product, "_load_pickle_artifact", observed_load)
    monkeypatch.setattr(fortran_c_shell, "lower_resolved_process_graph_unit_to_ssa", observed_lower)
    with pytest.raises(ObservedInput):
        product.compile_resolved_process_graph_unit(graph_path, tmp_path / "cuts/plan.json", index, tmp_path / "out")
    assert read_paths == [Path(published["scoped_inputs"][index]["path"])]
    published["scoped_inputs"][index]["sha256"] = "0" * 64
    (tmp_path / "cuts/plan.json").write_text(json.dumps(published), encoding="utf-8")
    with pytest.raises(ValueError, match="digest disagrees"):
        product.compile_resolved_process_graph_unit(graph_path, tmp_path / "cuts/plan.json", index, tmp_path / "out")


def test_real_two_process_source_unit_compilation(tmp_path, monkeypatch):
    from src.compiler.extraction_contract import ExtractionContract
    events = []
    graph_refs = []
    publish = product.publish_resolved_process_graph_inputs

    def observe_publish(graph, plan, directory, **kwargs):
        graph_refs.extend(weakref.ref(item) for item in
                          (graph, *(entry.graph for entry in graph.function_table if entry.graph is not None)))
        return publish(graph, plan, directory, **kwargs)

    def observe_progress(event):
        events.append(event)
        if event["stage"] == "resolved_unit_start":
            assert graph_refs and all(ref() is None for ref in graph_refs)

    monkeypatch.setattr(product, "publish_resolved_process_graph_inputs", observe_publish)
    source = SOURCE.replace("def root(x):\n    return left(x) + right(x)\n", "")
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "scalar",
                    "dtype": "float64", "rank": 0, "shape": [], "python_type": "builtins.float"}
                   for name in ("left", "right", "unrelated")], "records": {}, "bindings": [],
    })
    manifest = product.compile_partitioned_source(
        source, tmp_path / "run", jobs=2, extraction_contract=contract,
        max_unit_source_nodes=5,
        worker_resident_reservation_bytes=256 * 1024 ** 2,
        max_total_resident_bytes=512 * 1024 ** 2,
        max_worker_memory_bytes=4 * 1024 ** 3,
        progress=observe_progress,
    )
    assert manifest["counts"]["failed"] == 0, manifest
    assert manifest["counts"]["compiled-unverified"] == 3, manifest
    assert manifest["concurrent_workers_peak"] == 2
    released = next(i for i, event in enumerate(events) if event["stage"] == "source_plan_released")
    assert all(i > released for i, event in enumerate(events) if event["stage"] == "resolved_unit_start")
    assert len({unit["worker_process_id"] for unit in manifest["units"]}) == 3
    assert all(len(unit["scoped_input"]["function_references"]) == 1 for unit in manifest["units"])
    assert all(unit["scoped_input"]["source_nodes"] < unit["scoped_input"]["project_source_nodes"] for unit in manifest["units"])
    plan = json.loads((tmp_path / "run/inputs/plan.json").read_text(encoding="utf-8"))
    archived = plan["source_concordance"]
    assert archived["sha256"] == product._file_sha256(Path(archived["path"]))
    source_book = product._load_pickle_artifact(archived["path"])
    assert sum(page.cell_count() for page in dict.values(source_book.pages)) == archived["cells"] > 0
    assert source_book.clock[0] == archived["clock"]
    from src.compiler.identity_concordance import concordance_report
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    import numpy as np
    expected = {"left": 43.0, "right": 62.0, "unrelated": 3.0}
    for unit in manifest["units"]:
        unit_root = tmp_path / "run/product/units" / f"unit_{unit['unit_index']:03d}"
        module, outputs, exports = product._load_pickle_artifact(unit_root / unit["artifact"])
        report = concordance_report(module)
        assert report.splitlines()[0].endswith(", 0 finding(s)"), report
        assert module.metadata["source_concordance_archive"] == archived
        root = module.functions[exports[0]]
        artifact = emit_ssa_function_to_llvm(module, root.name)
        assert artifact.complete, artifact.shortfalls
        artifact = compile_artifact(artifact, directory=unit_root / "native")
        runtime = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
                                      _validation_error_reset=None, _validation_error_take=None)
        payload = unit_root / "native-check.pkl"
        product._dump_pickle_artifact(payload, (runtime, {}, ("program_outputs",), {
            "inputs": {int(root.args[0].id): np.asarray(21.0)},
            "outputs": tuple(int(value.id) for value in outputs[root.name]),
        }))
        child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
                               "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
                               capture_output=True, text=True, timeout=10)
        assert child.returncode == 0, child.stdout + child.stderr
        line = next(line for line in child.stdout.splitlines() if line.startswith("GRAPH_PROGRAM_OUTPUTS="))
        assert json.loads(line.split("=", 1)[1]) == [expected[unit["qualified_names"][0]]]


def test_oversized_dependency_closure_is_not_serialized_or_admitted(tmp_path, monkeypatch):
    graph_path, _plan_path, plan, graph = _inputs(tmp_path)
    published = product.publish_resolved_process_graph_inputs(
        graph, plan, tmp_path / "cuts", max_unit_source_nodes=2,
    )
    assert all(item["status"] == "requires-smaller-cut" for item in published["scoped_inputs"])
    assert not list((tmp_path / "cuts").glob("*.pkl"))

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized source tile launched a worker")

    monkeypatch.setattr(product.subprocess, "Popen", forbidden)
    manifest = product.compile_resolved_process_graph_plan(
        graph_path, tmp_path / "cuts/plan.json", tmp_path / "product", jobs=2,
        stop_after_failure=True,
    )
    assert manifest["concurrent_workers_peak"] == 0
    assert manifest["stopped_after_failure"] is True
    assert manifest["blocking_failure_unit_indices"] == list(range(len(plan["units"])))
    assert manifest["counts"]["failed"] == len(plan["units"])
    assert all(unit["resource"] == "source-node-tile" for unit in manifest["units"])
    assert manifest["subdivision_integral_count"] == len(plan["units"])


@pytest.mark.parametrize("shape", [(), (4,)])
def test_native_interface_tiles_compile_real_caller_without_callee_bodies(tmp_path, shape):
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.ssa_llvm_backend import compile_artifact, emit_ssa_function_to_llvm
    import numpy as np
    contract = ExtractionContract("extraction_contracts/program_extraction.yaml").with_program_abi({
        "values": [{"function": name, "parameter": "x", "storage": "span" if shape else "scalar",
                    "dtype": "float64", "rank": len(shape), "shape": list(shape),
                    "python_type": "src.common.tensors.abstraction.AbstractTensor" if shape else "builtins.float"}
                   for name in ("left", "right", "root", "unrelated", "top")], "records": {}, "bindings": [],
    })
    declarations, declared_outputs, declared_exports = lower_ast_source_to_ssa(
        SOURCE + "\ndef top(x):\n    return root(x) * 2.0\n", "top" if shape else None,
        partitioned_directory=tmp_path / "tiles", extraction_contract=contract,
        partitioned_options={"jobs": 2, "max_unit_source_nodes": 8,
            "worker_resident_reservation_bytes": 256 * 1024 ** 2,
            "max_worker_memory_bytes": 4 * 1024 ** 3},
        progress=lambda message: None,
    )
    manifest = declarations.metadata["partitioned_source_product"]
    assert all(not function.blocks for function in declarations.functions.values())
    assert manifest["counts"]["failed"] == manifest["counts"]["partial"] == 0, manifest
    assert manifest["concurrent_workers_peak"] == 2
    caller = next(unit for unit in manifest["units"] if unit["qualified_names"] == ["root"])
    assert caller["scoped_input"]["function_references"] == caller["unit"]["function_references"]
    assert len(caller["linked_native_interfaces"]) == 2
    for unit in manifest["units"]:
        header_path = tmp_path / "tiles/product/units" / f"unit_{unit['unit_index']:03d}" / unit["native_interface"]["artifact"]
        header, _out, _exports = product._load_native_unit_interface(header_path)
        assert all(not fn.blocks for fn in header.functions.values())
        assert sum(page.cell_count() for page in dict.values(header.struct_table.book.pages)) < 32
    cut = product._load_pickle_artifact(tmp_path / "tiles/inputs" / f"unit_{caller['unit_index']:03d}.pkl")
    assert {int(entry.reference.address) for entry in cut.function_table if entry.graph is not None} == set(caller["unit"]["function_references"])
    unit_root = tmp_path / "tiles/product/units" / f"unit_{caller['unit_index']:03d}"
    module, outputs, exports = product._load_pickle_artifact(unit_root / caller["artifact"])
    imported = [fn for fn in module.functions.values() if fn.metadata.get("native_interface")]
    assert len(imported) == 2
    assert all(not fn.blocks for fn in imported)
    assert all(entry.graph is None for entry in module.function_table)
    root = module.functions[exports[0]]
    artifact = emit_ssa_function_to_llvm(module, root.name, entry_name="linked_tile_test")
    assert artifact.complete, artifact.shortfalls
    assert len(artifact.linked_objects) == 2
    artifact = compile_artifact(artifact, directory=unit_root / "linked")
    runtime = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
                                  _validation_error_reset=None, _validation_error_take=None)
    payload = unit_root / "native-check.pkl"
    product._dump_pickle_artifact(payload, (runtime, {}, ("program_outputs",), {
        "inputs": {int(root.args[0].id): np.full(shape, 21.0)},
        "outputs": tuple(int(value.id) for value in outputs[root.name]),
    }))
    child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
                           "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    line = next(line for line in child.stdout.splitlines() if line.startswith("GRAPH_PROGRAM_OUTPUTS="))
    assert json.loads(line.split("=", 1)[1]) == [np.full(shape, 105.0).tolist()]
    public_artifact = emit_ssa_function_to_llvm(declarations, root.name, entry_name="public_llvm_tiles")
    assert public_artifact.complete and len(public_artifact.linked_objects) == 3
    public_artifact = compile_artifact(public_artifact, directory=unit_root / "public-llvm")
    product._dump_pickle_artifact(payload, (public_artifact, {}, ("program_outputs",), {
        "inputs": {int(root.args[0].id): np.full(shape, 21.0)},
        "outputs": tuple(int(value.id) for value in outputs[root.name]),
    }))
    child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
                           "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    line = next(line for line in child.stdout.splitlines() if line.startswith("GRAPH_PROGRAM_OUTPUTS="))
    assert json.loads(line.split("=", 1)[1]) == [np.full(shape, 105.0).tolist()]
    from src.compiler.ssa_c_backend import emit_ssa_module_to_c
    from src.compiler.identity_concordance import concordance_report, CorrelationTable
    assert concordance_report(declarations).splitlines()[0].endswith(", 0 finding(s)")
    proxy = emit_ssa_module_to_c(declarations, root.name, entry_name="partitioned_api")
    assert proxy.complete, proxy.shortfalls
    assert len(proxy.linked_objects) == 3
    proxy.compile(unit_root / "proxy")
    native = subprocess.run([sys.executable, "-u", "-c", '''
import ctypes, sys
library = ctypes.CDLL(sys.argv[1])
entry = library.partitioned_api
entry.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_longlong)]
count = int(sys.argv[2])
x, output = (ctypes.c_double * count)(*([21.0] * count)), (ctypes.c_double * count)()
pointers = (ctypes.c_void_p * 2)(ctypes.addressof(x), ctypes.addressof(output))
entry(pointers, (ctypes.c_longlong * 1)(0))
assert list(output) == [105.0] * count, list(output)
print('NATIVE_PUBLIC_INTERFACE=105.0')
''', str(proxy.library_path), str(4 if shape else 1)], capture_output=True, text=True, timeout=10)
    assert native.returncode == 0, native.stdout + native.stderr
    print(native.stdout.strip())
    standalone = proxy.compile_standalone(unit_root / "standalone", {
        int(root.args[0].id): np.full(shape, 21.0)})
    native = subprocess.run([str(standalone.executable_path), "1", "0"],
                            capture_output=True, text=True, timeout=10)
    assert native.returncode == 0, native.stdout + native.stderr
    saved = np.fromfile(standalone.final_outputs_path, dtype=np.float64)
    count = 4 if shape else 1
    assert saved[count:].tolist() == [105.0] * count
    top_unit = next(unit for unit in manifest["units"] if unit["qualified_names"] == ["top"])
    assert len(top_unit["scoped_input"]["function_references"]) == 1
    top_name = top_unit["exports"][0]
    if shape:
        assert declared_exports == (top_name,)
    top = emit_ssa_function_to_llvm(declarations, top_name, entry_name="deep_tiles")
    assert len(top.linked_objects) == 4  # top + root + the two leaf objects.
    top = compile_artifact(top, directory=unit_root / "deep")
    product._dump_pickle_artifact(payload, (top, {}, ("program_outputs",), {
        "inputs": {int(declarations.functions[top_name].args[0].id): np.full(shape, 21.0)},
        "outputs": tuple(int(value.id) for value in declared_outputs[top_name]),
    }))
    child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve().parents[1] /
                           "tools/compiler_probes/graph_kernel_native_worker.py"), str(payload)],
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 0, child.stdout + child.stderr
    line = next(line for line in child.stdout.splitlines() if line.startswith("GRAPH_PROGRAM_OUTPUTS="))
    assert json.loads(line.split("=", 1)[1]) == [np.full(shape, 210.0).tolist()]
    from src.compiler.ssa_fortran_backend import emit_module, FortranEmissionError
    with pytest.raises(FortranEmissionError, match="C/LLVM backend"):
        emit_module(declarations)
    with pytest.raises(ValueError, match="fresh output directory"):
        product.compile_partitioned_source(SOURCE, tmp_path / "tiles")
    bad_manifest = {**manifest, "units": [{**unit, "process_graph_unit_plan_sha256": "0" * 64}
                                          for unit in manifest["units"]]}
    with pytest.raises(ValueError, match="different frozen plan"):
        product.load_partitioned_source_interfaces(bad_manifest)
    changed = next(iter(declarations.functions.values()))
    changed.metadata["native_interface"]["argument_ids"] = (999,)
    assert any(f.kind == "native-interface-disagreement"
               for f in CorrelationTable.build(declarations).findings(declarations))
    stale = dataclasses.replace(artifact, linked_objects=((artifact.linked_objects[0][0], "0" * 64),))
    with pytest.raises(ValueError, match="object digest changed"):
        compile_artifact(stale, directory=unit_root / "stale")


def test_native_declaration_effect_summary_keeps_writes_and_refuses_unknown_feeds():
    from src.transmogrifier.ssa import Function, BasicBlock, Instr, SSAValue
    from src.compiler.ssa_call_input_adapters import _read_only_feed
    destination, source = SSAValue(0, "float64", (4,)), SSAValue(1, "float64")
    body = Function("mutates", [destination, source], {
        "entry": BasicBlock("entry", [Instr("Store", [source, destination], None)])})
    summary = {int(value.id): _read_only_feed(body, value.id) for value in body.args}
    assert summary == {0: False, 1: True}
    declaration = Function("mutates", body.args, {}, {
        "native_interface": {"read_only_arguments": summary}})
    assert not _read_only_feed(declaration, 0)
    assert _read_only_feed(declaration, 1)
    assert not _read_only_feed(declaration, 999)


def test_interface_tile_refuses_precompiled_body_payloads(tmp_path):
    from src.transmogrifier.ssa import IRModule, Function, BasicBlock
    _graph_path, _plan_path, plan, graph = _inputs(tmp_path)
    graph.function_table.entry("left").metadata["host_ssa_module"] = IRModule({
        "old_piece": Function("old_piece", [], {"entry": BasicBlock("entry", [])})})
    with pytest.raises(ValueError, match="precompiled SSA import needs a bodyless interface"):
        product.publish_resolved_process_graph_inputs(graph, plan, tmp_path / "cuts", interface_tiles=True)

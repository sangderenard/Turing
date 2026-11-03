"""The linked packet splitter copies each authored byte slice into its own row."""

import ast
import inspect
import json
from pathlib import Path

from src.common.tensors.compression import octet_codec, octet_lz
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import SOURCE_VALUE_CLASS
from src.compiler.concordance_declarations import INGESTION_VALUE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.project_compilation_product import (
    _load_pickle_artifact, compile_partitioned_source,
    compile_resolved_process_graph_unit,
)
from src.common.tensors.topological_reducer import node_identity_cell


def test_linked_split_has_three_typed_fresh_byte_slices(tmp_path):
    source = (inspect.getsource(octet_lz._read_header) + "\n"
              + inspect.getsource(octet_codec._split))
    bindings = dict(vars(octet_lz))
    bindings.update(vars(octet_codec))
    bindings.pop("_read_header", None)
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "values": [
            {"function": name, "parameter": "packet", "storage": "span",
             "dtype": "int64", "rank": 1, "python_type": "builtins.bytes"}
            for name in ("_read_header", "_split")
        ] + [
            {"function": "_read_header", "parameter": "count",
             "storage": "scalar", "dtype": "int64", "rank": 0,
             "shape": [], "python_type": "builtins.int"},
        ],
        "records": {}, "bindings": [],
    })
    planned = compile_partitioned_source(
        source, tmp_path / "plan", entrypoint="_split",
        extraction_contract=contract, python_bindings=bindings,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=4096,
        source_options={"tensor_ssa_reference": c_backend_repository_ssa_reference()},
        progress=lambda _: None,
    )
    units = json.loads(Path(planned["process_graph_unit_plan"]).read_text(
        encoding="utf-8"
    ))["units"]
    leaf_index = next(i for i, unit in enumerate(units)
                      if unit["qualified_names"] == ["_read_header"])
    caller_index = next(i for i, unit in enumerate(units)
                        if unit["qualified_names"] == ["_split"])
    assert leaf_index in units[caller_index]["dependency_units"]
    unit_graph = _load_pickle_artifact(
        tmp_path / f"plan/inputs/unit_{caller_index:03d}.pkl"
    )
    graph = unit_graph.function_table.entry(
        unit_graph.G.graph["compilation_unit_input"]["function_references"][0]
    ).graph
    book = unit_graph.G.graph["compilation_unit_identity_book"]
    scope = tuple(graph.G.graph["authored_function_identity"])
    packet_id = next(node_id for node_id, data in graph.G.nodes(data=True)
                     if data.get("type") == "Input"
                     and (data.get("attributes") or {}).get("binding_name")
                     == "packet")
    assert graph.G.nodes[packet_id]["attributes"]["sequence_writable"] is False
    packet_class = book.latest_ref(SOURCE_VALUE_CLASS, (scope, int(packet_id)))
    assert packet_class is not None
    packet_origins = tuple(source for source, stage in book.edges_into(packet_class)
                           if source.page == SOURCE_VALUE_CLASS
                           and stage.name == "canonical_relabel")
    assert len(packet_origins) == 1
    slices = tuple(node_id for node_id, data in graph.G.nodes(data=True)
                   if isinstance(data.get("expr_obj"), ast.Subscript)
                   and isinstance(data["expr_obj"].slice, ast.Slice))
    assert len(slices) == 3
    for node_id in slices:
        attrs = graph.G.nodes[node_id]["attributes"]
        assert (attrs["producer_kind"], attrs["aggregate_kind"],
                attrs["sequence_writable"]) == ("sequence_slice", "bytes", True)
        class_ref = book.latest_ref(SOURCE_VALUE_CLASS, (scope, int(node_id)))
        assert class_ref is not None
        assert book.page(SOURCE_VALUE_CLASS).cell(
            class_ref.row, class_ref.column
        )[0] == "builtins.bytes"
        class_origins = tuple(source for source, stage in book.edges_into(class_ref)
                              if source.page == SOURCE_VALUE_CLASS
                              and stage.name == "canonical_relabel")
        assert len(class_origins) == 1
        causes = {source for source, _stage in book.edges_into(class_origins[0])}
        assert packet_origins[0] in causes
        ingestion = {source for source in causes if source.page == INGESTION_VALUE}
        assert len(ingestion) == 1
        assert ingestion <= {source for source, _stage in book.edges_into(
            node_identity_cell(graph, int(node_id), book=book)
        )}
    leaf_root = tmp_path / "leaf"
    leaf = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], planned["process_graph_unit_plan"],
        leaf_index, leaf_root, progress=lambda _: None,
    )
    assert leaf["repository_ssa_complete"] is True
    caller_root = tmp_path / "caller"
    caller = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], planned["process_graph_unit_plan"],
        caller_index, caller_root, linked_units={leaf_index: leaf_root},
        progress=lambda _: None,
    )
    assert caller["repository_ssa_complete"] is True
    module, _, _ = _load_pickle_artifact(caller_root / "repository-ssa.pkl")
    root = module.functions["_split___split"]
    table = module.sequence_tables[root.name]
    source = table.by_id(packet_id)
    assert source.column_dtypes == ("int64",) and source.writable is False
    for node_id in slices:
        result = table.by_id(int(node_id))
        assert result.column_dtypes == ("int64",) and result.writable is True
        assert any(name.startswith(f"{root.name}__ssa_sequence_{node_id}_append_slice")
                   for name in module.functions)

"""C emission cites accepted leaf SSA and exception origins by owner import."""

import inspect
import json
from pathlib import Path

from src.common.tensors.compression import octet_codec, octet_lz
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import (
    Backend, C_EXCEPTION_OPERATION, EMISSION_FUNCTION, SSA_BLOCK,
)
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.identity_concordance import (
    CorrelationTable, IMPORTED_CELL, Ref, identity_book,
)
from src.compiler.project_compilation_product import (
    _load_pickle_artifact, compile_partitioned_source,
    compile_resolved_process_graph_unit,
)
from src.compiler.ssa_c_backend import emit_ssa_module_to_c


def test_linked_split_c_emission_imports_accepted_leaf_origins(tmp_path):
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
        ] + [{
            "function": "_read_header", "parameter": "count",
            "storage": "scalar", "dtype": "int64", "rank": 0,
            "shape": [], "python_type": "builtins.int",
        }],
        "records": {}, "bindings": [],
    })
    planned = compile_partitioned_source(
        source, tmp_path / "plan", entrypoint="_split",
        extraction_contract=contract, python_bindings=bindings,
        ssa_composition=True, plan_only=True, max_unit_source_nodes=4096,
        source_options={
            "tensor_ssa_reference": c_backend_repository_ssa_reference(),
        },
        progress=lambda _: None,
    )
    unit_plan = planned["process_graph_unit_plan"]
    units = json.loads(Path(unit_plan).read_text(encoding="utf-8"))["units"]
    leaf_index = next(i for i, unit in enumerate(units)
                      if unit["qualified_names"] == ["_read_header"])
    caller_index = next(i for i, unit in enumerate(units)
                        if unit["qualified_names"] == ["_split"])
    assert leaf_index in units[caller_index]["dependency_units"]
    leaf_root = tmp_path / "leaf"
    leaf = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan, leaf_index,
        leaf_root, progress=lambda _: None,
    )
    assert leaf["repository_ssa_complete"] is True
    caller_root = tmp_path / "caller"
    caller = compile_resolved_process_graph_unit(
        planned["resolved_process_graph"], unit_plan, caller_index,
        caller_root, linked_units={leaf_index: leaf_root},
        progress=lambda _: None,
    )
    assert caller["repository_ssa_complete"] is True
    module, _outputs, _exports = _load_pickle_artifact(
        caller_root / "repository-ssa.pkl"
    )
    artifact = emit_ssa_module_to_c(module, "_split___split")
    book = identity_book(module)
    rows = tuple(book.page(C_EXCEPTION_OPERATION).rows())
    assert len(rows) == 3
    for row in rows:
        source_cell, function_name, _result = row
        accepted = module.functions[function_name].metadata[
            "accepted_ssa_component"
        ]
        receipt = book.latest_ref(
            IMPORTED_CELL, (accepted["artifact_sha256"], source_cell.key),
        )
        policy = book.latest_ref(C_EXCEPTION_OPERATION, row)
        assert receipt is not None and policy is not None
        assert receipt in {source for source, _stage in book.edges_into(policy)}
    helper = next(name for name, function in module.functions.items()
                  if function.metadata.get("ssa_sequence_operation") == "append"
                  and function.metadata.get("accepted_ssa_component"))
    # The pending function cell owns the source; its completed revision owns
    # the emitted units and therefore has a different direct edge set.
    header = Ref(EMISSION_FUNCTION, (helper, Backend.C_MODULE), 0)
    assert header is not None
    assert any(
        source.page == IMPORTED_CELL
        and book.page(IMPORTED_CELL).cell(
            source.row, source.column,
        ).source.page == SSA_BLOCK
        for source, _stage in book.edges_into(header)
    )
    assert not any(
        finding.kind == "c-exception-provenance"
        for finding in CorrelationTable._operator_safety_findings(module)
    )
    assert not any("emission_unit: no source cell" in shortfall.reason
                   for shortfall in artifact.shortfalls)
    assert artifact.complete, artifact.shortfalls
    import numpy as np
    from src.compiler.ssa_storage_requirements import public_sequence_capacity_sources

    relations = public_sequence_capacity_sources(module, "_split___split")
    assert len(relations) == 3
    artifact.compile(tmp_path / "native_split")
    packet_id = relations[0].input_data_id
    for commands, literals in ((b"", b""), (b"\x00\xff", b"\x80\x7f"),
                               (bytes(range(130)), bytes(range(255, -1, -1)))):
        packet = octet_lz._header(len(literals), 1, len(commands)) + commands + literals
        native = artifact.prepare_execution({packet_id: np.frombuffer(packet, dtype=np.uint8)}).run()
        assert tuple(bytes(native.buffers[item.output_data_id].tolist())
                     for item in relations) == octet_codec._split(packet)

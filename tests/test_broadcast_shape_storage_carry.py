"""The authored broadcast loop keeps one output arena through its stores."""

import ast
import contextlib
import io
from pathlib import Path

import numpy as np

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.concordance_declarations import LOOP_STORAGE_VERSION
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable, identity_book
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_authored_broadcast_shape_extents_preserves_output_storage(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/compiler/graph_refinement_kernels.py").read_text()
    names = ("rank", "start", "left", "right", "output")
    contract = ExtractionContract(
        root / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [],
        "values": [
            {"function": "broadcast_shape_extents", "parameter": name,
             "storage": "scalar" if name in {"rank", "start"} else "span",
             "dtype": "int32", "rank": 0 if name in {"rank", "start"} else 1,
             "python_type": "builtins.int" if name in {"rank", "start"}
             else "src.common.tensors.abstraction.AbstractTensor",
             "mutable": name == "output"}
            for name in names
        ],
    })
    module, _, _ = lower_ast_source_to_ssa(
        source, "broadcast_shape_extents", name="broadcast_storage_carry",
        extraction_contract=contract, runtime_closure_only=True,
        python_bindings={},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _: None, progress_bars=False,
    )
    book = identity_book(module)
    rows = tuple(book.page(LOOP_STORAGE_VERSION).rows())
    assert rows
    assert not CorrelationTable._loop_storage_version_findings(module)

    function = module.functions[
        "broadcast_storage_carry__broadcast_shape_extents"
    ]
    parameters = dict(function.metadata["parameter_names"])
    output_id = parameters["output"]
    assert all(not (
        instruction.op == "Phi" and instruction.attributes.get("binding")
        == "loop_carried" and any(arg.id == output_id for arg in instruction.args)
    ) for block in function.blocks.values() for instruction in block.instrs)

    artifact = emit_ssa_module_to_c(module, function.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "broadcast_storage_carry")
    status_id, = (item["value_id"] for item in artifact.runtime_abi()["buffers"]
                  if item["role"] == "output")
    for start, left, right, expected, status in (
        (0, [0, 1], [1, 1], [0, 1], 1),
        (0, [2, 1], [2, 3], [2, 3], 1),
        (1, [99, 0, 1], [99, 1, 1], [-9, 0, 1], 1),
        (0, [2, 3], [4, 3], [-9, -9], 0),
    ):
        output = np.full(len(left), -9, dtype=np.int32)
        execution = artifact.prepare_execution({
            parameters["rank"]: np.array(2, dtype=np.int32),
            parameters["start"]: np.array(start, dtype=np.int32),
            parameters["left"]: np.asarray(left, dtype=np.int32),
            parameters["right"]: np.asarray(right, dtype=np.int32),
            output_id: output,
        }).run()
        assert output.tolist() == expected
        assert int(np.asarray(execution.buffers[status_id]).flat[0]) == status

    row = next(row for row in rows if len(book.page(LOOP_STORAGE_VERSION).latest(row)) > 1)
    ref = book.latest_ref(LOOP_STORAGE_VERSION, row)
    page = book.page(LOOP_STORAGE_VERSION)
    old = page.cells[(ref.row, ref.column)]
    node, kind, edges = old[0]
    page.cells[(ref.row, ref.column)] = (
        (node, kind, ((edges[0][0], row[0]), *edges[1:])), *old[1:],
    )
    try:
        assert CorrelationTable._loop_storage_version_findings(module)
    finally:
        page.cells[(ref.row, ref.column)] = old


def test_actual_loop_rebinding_still_has_a_value_carry():
    source = """def root(rank, left, right, output):
    axis = 0
    while axis < rank:
        if axis == 0:
            output = left
        else:
            output = right
        axis = axis + 1
    return output[0]
"""
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry(
        graph.function_table.reference("root")
    ).graph.G
    loops = tuple(data for _node_id, data in executable.nodes(data=True)
                  if data.get("type") == "While")
    assert len(loops) == 1
    carried = loops[0]["attributes"]["loop_carried_bindings"]
    assert "output" in carried
    initial, updated = carried["output"]
    assert initial != updated

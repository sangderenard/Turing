from __future__ import annotations

import ast
import contextlib
import io
import sqlite3

import pytest

from src.common.tensors.abstract_nn.graph_translation_network import (
    CompilerTransformationRoute,
    GraphTranslationNetwork,
    TransformationWeightMatrix,
    TransformerUnavailableError,
)
from src.common.tensors.abstract_nn.training_data_store import (
    CompilerTrainingDatabase,
    put_reduced_graph_view,
)
from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_training_database_retains_views_tokens_lineage_and_commands(tmp_path):
    path = tmp_path / "compiler-training.sqlite3"
    with CompilerTrainingDatabase(path) as database:
        program_id = database.put_program(
            "def kernel(x):\n    return x + 1\n", "kernel",
            metadata={"fixture": True},
        )
        source = database.put_view(
            program_id, "source", {"text": "kernel"},
            ("def", "kernel", "x"), token_ids=(11, 12, 13),
            generator="test",
        )
        ssa = database.put_view(
            program_id, "ssa", {"nodes": ((0, "Input"), (1, "Add"))},
            ("Input", "Add"), token_ids=(21, 22), generator="test",
        )
        database.link_views(
            program_id, source.id, ssa.id, "compile_ssa",
            weight_key="source->ssa@0",
            compiler_command={"command": "compile_ssa"},
        )
        request = database.request_compiler_view(
            program_id, "ssa", "fortran", "emit_fortran",
            {"entry": "kernel"},
        )
        duplicate = database.request_compiler_view(
            program_id, "ssa", "fortran", "emit_fortran",
            {"entry": "kernel"},
        )
        assert duplicate.id == request.id
        assert database.forms(program_id) == ("source", "ssa")
        assert len(database.pending_commands()) == 1
        database.complete_command(request.id, ssa.id)
        assert database.pending_commands() == ()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM programs").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM views").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM token_events").fetchone()[0] == 5
        assert connection.execute("SELECT count(*) FROM transformations").fetchone()[0] == 1
    finally:
        connection.close()


def test_weight_matrix_routes_missing_views_and_refuses_fake_prediction():
    with CompilerTrainingDatabase() as database:
        program_id = database.put_program("x = 1\n", "module")
        database.put_view(
            program_id, "source", {"text": "x = 1"},
            ("x", "=", "1"), token_ids=(1, 2, 3), generator="test",
        )
        weights = TransformationWeightMatrix(("source", "python_ast", "ssa"))
        weights.populate_stubs(vocabulary_size=128)
        weights.persist(database)
        network = GraphTranslationNetwork(
            database,
            weights,
            (
                CompilerTransformationRoute("source", "python_ast", "parse_python"),
                CompilerTransformationRoute("source", "ssa", "compile_ssa"),
            ),
        )

        requests = network.densify(program_id, arguments={"entry": "module"})
        assert {(item.source_form, item.target_form) for item in requests} == {
            ("source", "python_ast"), ("source", "ssa"),
        }
        assert network.densify(program_id) == ()
        with pytest.raises(TransformerUnavailableError):
            network.predict("source", "ssa", (1, 2, 3))
        statuses = {
            row[0]
            for row in database.connection.execute("SELECT status FROM weight_sets")
        }
        assert statuses == {"stub"}


def test_reduced_graph_capture_retains_dense_tokens_without_object_addresses():
    source = "def kernel(x):\n    value = x + 1\n    return value\n"
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry("kernel").graph.G

    with CompilerTrainingDatabase() as database:
        program_id = database.put_program(source, "kernel")
        view = put_reduced_graph_view(database, program_id, executable)
        row = database.connection.execute(
            "SELECT payload_json, tokens_json FROM views WHERE id=?", (view.id,),
        ).fetchone()
        assert "ssa_identity_tokens" in row[0]
        assert "slot:start" in row[1]
        assert "0x" not in row[0]
        assert database.connection.execute(
            "SELECT count(*) FROM token_events WHERE view_id=?", (view.id,),
        ).fetchone()[0] == len(view.tokens)

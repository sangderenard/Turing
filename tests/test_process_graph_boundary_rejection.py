"""A callable attribute cannot authorize a hidden Python execution path."""

import ast
from types import SimpleNamespace

import pytest

from src.compiler.glsl_deployment_strategy import _compile_whole_process_graph
from src.transmogrifier.graph.graph_express2 import _expand_unresolved_ast_parents


def _marked_source(value):
    return value + 1


def test_pursued_callable_marker_cannot_replace_authored_source():
    _marked_source.__process_graph_boundary__ = "host_materialization"
    try:
        with pytest.raises(ValueError, match="no native SSA or extraction-contract authority"):
            _expand_unresolved_ast_parents(
                ast.parse("def entry(value):\n    return _marked_source(value)\n"),
                {"_marked_source": _marked_source},
                include=lambda *_args, **_kwargs: True,
                pursuit_roots=("entry",),
            )
    finally:
        del _marked_source.__process_graph_boundary__


def test_marked_shell_cannot_report_graph_compilation_complete():
    graph = SimpleNamespace(G=SimpleNamespace(graph={"function_name": "marked"}))
    shell = SimpleNamespace(process_graph_boundary="host_materialization", process_graph=graph)
    with pytest.raises(RuntimeError, match="cannot count as compiled"):
        _compile_whole_process_graph(shell)
    assert not hasattr(shell, "whole_program_compiled")

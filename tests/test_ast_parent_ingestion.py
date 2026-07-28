import ast
import contextlib
import io

from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _source_helper(value):
    return value * 2 + 1


def _recursive_source_helper(value):
    if value <= 0:
        return 0
    return _recursive_source_helper(value - 1) + 1


def _dependency_leaf(value):
    return value + 1


def _dependency_middle(value):
    return _dependency_leaf(value) * 2


def _dependency_root(value):
    return _dependency_middle(value) - 3


class _WriterSource:
    def append_frame(self, frame, *, keyframe=False):
        if keyframe:
            return frame + 1
        return frame


def _ingest(source, bindings):
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = dict(bindings)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(source),
            resolve_unresolved_parents=True,
        )
    return graph


def _definitions(graph, name):
    return [
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(
            data.get("expr_obj"),
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and data["expr_obj"].name == name
    ]


def test_ingestion_builds_source_parent_as_process_graph_nodes():
    graph = _ingest(
        """
def root(value):
    return _source_helper(value)
""",
        {"_source_helper": _source_helper},
    )

    helper_id, _helper = _definitions(graph, "_source_helper")[0]
    call_id, call = next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Name)
        and data["expr_obj"].func.id == "_source_helper"
    )

    assert graph.G.has_edge(helper_id, call_id)
    assert call["attributes"]["resolved_ast_parent"] == helper_id
    assert len(graph.function_table) == 0


def test_ingestion_resolves_attribute_parent_to_method_ast():
    graph = _ingest(
        """
def record(writer, frame):
    return writer.append_frame(frame, keyframe=True)
""",
        {"writer": _WriterSource()},
    )

    method_id, _method = _definitions(graph, "append_frame")[0]
    call_id, call = next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "append_frame"
    )

    assert graph.G.has_edge(method_id, call_id)
    assert call["attributes"]["resolved_ast_parent"] == method_id
    assert len(graph.function_table) == 0


def test_ingestion_recursion_adds_one_definition_and_links_both_calls():
    graph = _ingest(
        """
def root(value):
    return _recursive_source_helper(value)
""",
        {"_recursive_source_helper": _recursive_source_helper},
    )

    definitions = _definitions(graph, "_recursive_source_helper")
    assert len(definitions) == 1
    helper_id = definitions[0][0]
    linked_calls = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and graph.G.has_edge(helper_id, node_id)
    ]
    assert len(linked_calls) == 2
    assert len(graph.function_table) == 0


def test_ingestion_reaches_transitive_dependency_fixed_point():
    graph = _ingest(
        """
def entry(value):
    return _dependency_root(value)
""",
        {"_dependency_root": _dependency_root},
    )

    for name in (
        "_dependency_root",
        "_dependency_middle",
        "_dependency_leaf",
    ):
        definition_id, _definition = _definitions(graph, name)[0]
        calls = [
            node_id
            for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Call)
            and (
                (
                    isinstance(data["expr_obj"].func, ast.Name)
                    and data["expr_obj"].func.id == name
                )
                or (
                    isinstance(data["expr_obj"].func, ast.Attribute)
                    and data["expr_obj"].func.attr == name
                )
            )
        ]
        assert calls
        assert all(graph.G.has_edge(definition_id, call) for call in calls)
    assert graph.G.graph["unresolved_ast_calls"] == ()
    assert graph.G.graph["missing_ast_parent_calls"] == ()
    assert graph.G.graph["ast_parent_closure_complete"]
    assert len(graph.function_table) == 0


def test_ingestion_leaves_source_less_parent_unresolved():
    graph = _ingest(
        """
def root(value):
    return len(value)
""",
        {"len": len},
    )

    assert graph.G.graph["resolved_ast_parent_count"] == 0
    assert graph.G.graph["unresolved_ast_calls"] == (
        {
            "name": "len",
            "line": 3,
            "column": 11,
            "reason": "source_unavailable",
        },
    )
    assert graph.G.graph["missing_ast_parent_calls"] == ()
    assert not _definitions(graph, "len")
    assert len(graph.function_table) == 0

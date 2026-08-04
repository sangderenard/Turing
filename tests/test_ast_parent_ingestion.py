import ast
import contextlib
import io
import types

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
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

    def unrelated_method(self, frame):
        return _unrelated_dependency(frame)


def _unrelated_dependency(value):
    return value - 1


class _ConstructedSource:
    def __init__(self, value):
        self.value = value

    def unrelated_method(self):
        return _unrelated_dependency(self.value)


class _FieldParameterCollisionSource:
    symbols = None

    def encode(self, symbols):
        return symbols + 1


def _scoped_dependency_a(value):
    return value + 10


def _scoped_dependency_b(value):
    return value - 10


def _scoped_root_a(value):
    return dependency(value)


def _scoped_root_b(value):
    return dependency(value)


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


def test_profile_verbose_emits_per_item_ast_and_graph_logs(capsys):
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"_source_helper": _source_helper}
    graph.build_from_ast(
        ast.parse(
            "def root(value):\n"
            "    return _source_helper(value)\n"
        ),
        resolve_unresolved_parents=True,
        profile_verbose=True,
    )
    captured = capsys.readouterr().out

    assert "[ast-parent]" in captured
    assert "[graph-build" in captured


def test_progress_callback_emits_graph_build_logs_without_profile_verbose():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"_source_helper": _source_helper}
    messages = []
    graph.build_from_ast(
        ast.parse(
            "def root(value):\n"
            "    return _source_helper(value)\n"
        ),
        resolve_unresolved_parents=True,
        progress=messages.append,
    )

    assert any("[ast-parent]" in message for message in messages)
    assert any("[graph-build" in message for message in messages)


def test_literal_module_constants_are_static_function_bindings():
    graph = _ingest(
        "CARD_SIZE = 2000\n\ndef build():\n    return CARD_SIZE\n",
        {},
    )
    (_node_id, data), = _definitions(graph, "build")

    assert data["expr_obj"]._python_bindings["CARD_SIZE"] == 2000
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("build").graph
    assert function_graph.python_bindings["CARD_SIZE"] == 2000

    direct = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        direct.build_from_ast(ast.parse(
            "CARD_SIZE = 2000\n\ndef build():\n    return CARD_SIZE\n"
        ))
    reduce_abstract_tensor_topology(direct)
    direct_function = direct.function_table.entry("build").graph
    assert direct_function.python_bindings["CARD_SIZE"] == 2000


def test_class_field_default_does_not_shadow_method_parameter():
    graph = _ingest(
        """
def build(symbols):
    table = _FieldParameterCollisionSource()
    return table.encode(symbols)
""",
        {"_FieldParameterCollisionSource": _FieldParameterCollisionSource},
    )
    reduce_abstract_tensor_topology(graph)

    method = graph.function_table.entry("encode").graph
    symbol_ids = method.G.graph["identity_table"]["symbols"]
    assert symbol_ids
    assert method.G.nodes[symbol_ids[0]]["type"] == "Input"
    assert not any(
        data.get("type") == "Constant"
        and (data.get("attributes") or {}).get("binding_name") == "symbols"
        for _node_id, data in method.G.nodes(data=True)
    )


def test_imported_callable_is_resolved_in_external_function_table():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(
            "from pathlib import Path\n\ndef build(value):\n"
            "    return Path(value)\n"
        ))
    reduce_abstract_tensor_topology(graph)

    entry = graph.external_function_table.entry("Path")
    assert entry.python_callable is not None
    assert entry.python_callable("example").name == "example"


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
    assert not _definitions(graph, "unrelated_method")
    assert not _definitions(graph, "_unrelated_dependency")
    assert len(graph.function_table) == 0


def test_ingestion_filters_unreferenced_class_methods_before_recursing():
    graph = _ingest(
        """
def construct(value):
    return _ConstructedSource(value)
""",
        {"_ConstructedSource": _ConstructedSource},
    )

    assert len(_definitions(graph, "__init__")) == 1
    assert not _definitions(graph, "unrelated_method")
    assert not _definitions(graph, "_unrelated_dependency")


def test_actual_ast_ingestion_records_dummy_class_schema_without_process_inference():
    graph = _ingest(
        '''
class Thermostat:
    manufacturer: str = "Turing"

    def __init__(self, reading: float):
        self.reading: float = reading
        self.target = 21.0

    def adjust(self, amount: float) -> float:
        return self.reading + amount
''',
        {},
    )

    map_ir = graph.G.graph["map_ir"]
    (schema,) = map_ir["objects"]
    assert schema["class_name"] == "Thermostat"
    assert schema["attributes"] == (
        {"name": "manufacturer", "identity": "Thermostat.manufacturer", "storage": "class", "annotation": "str", "permissions": ()},
        {"name": "reading", "identity": "Thermostat.reading", "storage": "instance", "annotation": "float", "permissions": ()},
        {"name": "target", "identity": "Thermostat.target", "storage": "instance", "annotation": None, "permissions": ()},
    )
    assert [method["graph_identity"] for method in schema["methods"]] == [
        "Thermostat.__init__", "Thermostat.adjust",
    ]
    assert [item["identity"] for item in map_ir["graphs"]] == [
        "Thermostat.__init__", "Thermostat.adjust",
    ]
    assert map_ir["permissions"] == ()
    assert all(item["permissions"] == () for item in map_ir["graphs"])


def test_annotated_assignments_split_schema_from_runtime_assignment():
    graph = _ingest(
        '''
ModuleAlias: type = int

class Accumulator:
    scale: float = 2.0

    def apply(self, value: float) -> float:
        result: float = value * self.scale
        return result
''',
        {},
    )

    schema = graph.G.graph["map_ir"]["schema"]
    assert schema["module"]["annotations"][0]["name"] == "ModuleAlias"
    assert schema["classes"][0]["members"][0]["identity"] == "Accumulator.scale"
    function = next(
        item for item in schema["functions"]
        if item["identity"] == "Accumulator.apply"
    )
    assert function["locals"][0]["name"] == "result"
    assert function["locals"][0]["annotation"] == "float"

    from src.common.tensors.topological_reducer import (
        reduce_abstract_tensor_topology,
    )

    reduce_abstract_tensor_topology(graph)
    apply_graph = graph.function_table.entry("Accumulator.apply").graph
    assert not any(
        isinstance(data.get("expr_obj"), ast.AnnAssign)
        for _node_id, data in apply_graph.G.nodes(data=True)
    )
    assert any(
        isinstance(data.get("expr_obj"), ast.BinOp)
        for _node_id, data in apply_graph.G.nodes(data=True)
    )


def test_dynamic_attribute_call_does_not_alias_method_by_basename():
    graph = _ingest(
        """
class Accumulator:
    def append(self, value):
        return value

def collect(values, value):
    values.append(value)
""",
        {},
    )

    method_id, _method = _definitions(graph, "append")[0]
    call_id, call = next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "append"
    )
    assert not graph.G.has_edge(method_id, call_id)
    assert "resolved_ast_parent" not in call.get("attributes", {})


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


def test_ingestion_keeps_each_definition_globals_lexically_scoped():
    root_a = types.FunctionType(
        _scoped_root_a.__code__,
        {"dependency": _scoped_dependency_a},
        name=_scoped_root_a.__name__,
    )
    root_b = types.FunctionType(
        _scoped_root_b.__code__,
        {"dependency": _scoped_dependency_b},
        name=_scoped_root_b.__name__,
    )
    graph = _ingest(
        """
def entry(value):
    return _scoped_root_a(value) + _scoped_root_b(value)
""",
        {
            "_scoped_root_a": root_a,
            "_scoped_root_b": root_b,
        },
    )

    assert len(_definitions(graph, "_scoped_dependency_a")) == 1
    assert len(_definitions(graph, "_scoped_dependency_b")) == 1


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

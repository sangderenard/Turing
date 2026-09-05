import ast
import contextlib
import io
import re
import types

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.common.tensors.accelerator_backends.aot_compile import (
    _source_dependency_is_not_tensor_primitive,
)
from src.compiler.loop_composer import LoopComposer, LoopBackendCapabilities
from src.compiler.process_graph_fusion import extract_clean_process_subgraph
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.node_special_cases import tensor_operation_name
from src.transmogrifier.function_table import (
    ParameterAccess,
    ParameterScope,
    ParameterStorage,
    ParameterTransfer,
)


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


def _extend_rows_into(rows, destination):
    for row in rows:
        destination.extend(row)
    return destination


def _first_byte(bits):
    return bits[0]


def _neg_tensor_code_reference(value):
    return -value


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


class _RemovableStorageSource:
    def remove_node(self, nid):
        return nid + 1


class _ReceiverForwardingSource:
    def __init__(self):
        self.G = _RemovableStorageSource()

    def callee(self, G, nid):
        return G.remove_node(nid)

    def run(self, nid):
        return self.callee(self.G, nid)

    def sweep(self, G, nodes):
        for nid in nodes:
            G.remove_node(nid)


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


def test_literal_module_table_can_reference_earlier_literal_constants():
    graph = _ingest(
        "I32 = 0x7F\nI64 = 0x7E\n"
        "VALUE_TYPE = {'i32': I32, 'i64': I64}\n\n"
        "def encode(name):\n    return VALUE_TYPE[name]\n",
        {},
    )
    (_node_id, data), = _definitions(graph, "encode")

    assert data["expr_obj"]._python_bindings["VALUE_TYPE"] == {
        "i32": 0x7F,
        "i64": 0x7E,
    }
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("encode").graph
    assert function_graph.python_bindings["VALUE_TYPE"] == {
        "i32": 0x7F,
        "i64": 0x7E,
    }
    assert not any(
        str((node.get("attributes") or {}).get("binding_name")) == "VALUE_TYPE"
        and str((node.get("attributes") or {}).get("binding_kind")) == "external"
        for _node_id, node in function_graph.G.nodes(data=True)
    )


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


def test_tensor_code_reference_is_ingested_as_a_process_graph_definition():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                "def root(value):\n"
                "    return value.neg()\n"
            ),
            resolve_unresolved_parents=True,
            tensor_code_references={"neg": _neg_tensor_code_reference},
        )

    reference_id, _definition = _definitions(
        graph, "_neg_tensor_code_reference"
    )[0]
    call_id, call = next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and tensor_operation_name(data["expr_obj"]) == "neg"
    )
    assert graph.G.has_edge(reference_id, call_id)
    assert call["attributes"]["resolved_ast_parent"] == reference_id

    reduce_abstract_tensor_topology(graph)
    root = graph.function_table.entry("root").graph
    call = next(
        data
        for _node_id, data in root.G.nodes(data=True)
        if data.get("op") == "neg"
    )
    # Reduction preserves the original graph-node correlation for provenance;
    # the callable itself has an independent opaque FunctionTable reference.
    callee = graph.function_table.entry("_neg_tensor_code_reference")
    assert callee.name == "_neg_tensor_code_reference"
    assert callee.graph is not None
    assert any(
        data.get("type") in {"Neg", "UnaryOp"}
        or data.get("op") in {"neg", "Neg"}
        for _node_id, data in callee.graph.G.nodes(data=True)
    )


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


def test_ingestion_propagates_field_receiver_identity_through_call_parameter():
    graph = _ingest(
        """
def entry(source, nid):
    return source.run(nid)
""",
        {"source": _ReceiverForwardingSource()},
    )

    remove_id, _definition = _definitions(graph, "remove_node")[0]
    call_id, call = next(
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "remove_node"
    )
    assert graph.G.has_edge(remove_id, call_id)
    assert call["attributes"]["resolved_ast_parent"] == remove_id
    assert graph.G.graph["missing_ast_parent_calls"] == ()

    reduce_abstract_tensor_topology(graph)
    contracts = graph.function_table.entry("callee").parameter_contracts
    assert tuple(contract.name for contract in contracts) == (
        "self", "G", "nid",
    )
    assert contracts[1].transfer is ParameterTransfer.ALIAS
    assert contracts[1].access is ParameterAccess.INOUT
    assert contracts[1].storage is ParameterStorage.RECORD
    assert contracts[1].scope is ParameterScope.CALLER
    assert contracts[2].storage is ParameterStorage.SCALAR


def test_source_linked_argument_method_is_not_an_opaque_loop_effect():
    graph = _ingest(
        """
def entry(source, storage, nodes):
    source.sweep(storage, nodes)
""",
        {
            "source": _ReceiverForwardingSource(),
            "storage": _RemovableStorageSource(),
        },
    )
    reduce_abstract_tensor_topology(graph)

    sweep = graph.function_table.entry("sweep").graph
    call = next(
        data
        for _node_id, data in sweep.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "remove_node"
    )
    assert call["attributes"]["method_ref"] == call["attributes"]["callee_ref"]
    assert not any(
        effect["operator"] == "remove_node"
        for _node_id, data in sweep.G.nodes(data=True)
        for effect in (data.get("attributes") or {}).get(
            "loop_state_effects", ()
        )
    )


def test_process_graph_field_provenance_pursues_digraph_remove_node_source():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"owner": ProcessGraph}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("""
def entry(owner, nid):
    owner.deduplicate_node(owner.G, nid)
"""),
            resolve_unresolved_parents=True,
            parent_include=_source_dependency_is_not_tensor_primitive,
        )

    remove_id, definition = _definitions(graph, "remove_node")[0]
    assert getattr(definition["expr_obj"], "_python_source_identity")[1].endswith(
        "DiGraph.remove_node"
    )
    call_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "remove_node"
    )
    assert graph.G.has_edge(remove_id, call_id)

    reduce_abstract_tensor_topology(graph)
    original = graph.function_table.entry("deduplicate_node").graph
    specialized = extract_clean_process_subgraph(original, original.G)
    call = next(
        data
        for _node_id, data in specialized.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "remove_node"
    )
    assert call["attributes"]["method_ref"] == call["attributes"]["callee_ref"]
    composer = LoopComposer(LoopBackendCapabilities(
        backend="fortran",
        native_for=True,
        native_while=True,
        dynamic_bounds=True,
    ))
    plans = composer.compose(specialized)
    assert not any(
        effect.operator == "remove_node"
        for plan in plans
        for effect in plan.loop.state_effects
    )


def test_set_add_is_collection_candidate_not_tensor_operation():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("""
def collect(value):
    present = set()
    for item in value:
        present.add(item)
    return present
"""),
            resolve_unresolved_parents=True,
            tensor_code_references={"add": _neg_tensor_code_reference},
        )
    call = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "add"
    )
    attributes = call.get("attributes", {})
    assert attributes["tensor_candidate"] == "add"
    assert "tensor" not in attributes
    assert getattr(call["expr_obj"], "_tensor_code_reference", None) is None
    assert not _definitions(graph, "_neg_tensor_code_reference")

    reduce_abstract_tensor_topology(graph)
    collect = graph.function_table.entry("collect").graph
    sequence = next(
        data
        for _node_id, data in collect.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Name)
        and data["expr_obj"].func.id == "set"
    )
    sequence_attributes = sequence.get("attributes", {})
    assert sequence_attributes["aggregate_kind"] == "set"
    assert sequence_attributes["sequence_key_columns"] == (0,)
    assert sequence_attributes["sequence_writable"] is True
    loop = next(
        data
        for _node_id, data in collect.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effects = loop.get("attributes", {})["loop_state_effects"]
    assert len(effects) == 1
    assert effects[0]["effect_mode"] == "sequence_mutation"
    assert effects[0]["sequence_policy"] == "unique"


def test_literal_aggregate_identity_reaches_pursued_callee_formal():
    graph = _ingest(
        """
def root(rows):
    destination = []
    return _extend_rows_into(rows, destination)
""",
        {"_extend_rows_into": _extend_rows_into},
    )

    pursued = next(
        data["expr_obj"]
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.FunctionDef)
        and data["expr_obj"].name == "_extend_rows_into"
    )
    assert pursued._python_aggregate_binding_kinds["destination"] == "list"

    reduce_abstract_tensor_topology(graph)
    callee = graph.function_table.entry("_extend_rows_into").graph
    destination = next(
        data
        for _node_id, data in callee.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "destination"
    )
    assert destination["attributes"]["aggregate_kind"] == "list"
    loop = next(
        data
        for _node_id, data in callee.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effects = loop["attributes"]["loop_state_effects"]
    assert [effect["effect_mode"] for effect in effects] == [
        "sequence_mutation"
    ]


def test_tuple_loop_target_survives_body_rebinding_of_one_member():
    graph = _ingest(
        """
def rewrite(rows, replacement):
    result = []
    for op, value in rows:
        if value:
            op = replacement
        result.append((op, value))
    return result
""",
        {},
    )

    reduce_abstract_tensor_topology(graph)
    reduced = graph.function_table.entry("rewrite").graph
    loop = next(
        data
        for _node_id, data in reduced.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    bindings = loop["attributes"]["loop_target_bindings"]
    assert tuple(bindings) == ("op", "value")
    assert bindings["op"] != bindings["value"]


def test_bytearray_identity_reaches_pursued_span_parameter():
    graph = _ingest(
        """
def root():
    data = bytearray(16)
    return _first_byte(data)
""",
        {"_first_byte": _first_byte},
    )

    reduce_abstract_tensor_topology(graph)
    callee = graph.function_table.entry("_first_byte").graph
    bits = next(
        data
        for _node_id, data in callee.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "bits"
    )
    assert bits["attributes"]["aggregate_kind"] == "bytearray"
    assert bits["attributes"]["sequence_writable"] is True


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


def test_real_networkx_constructor_carries_source_factory_field_storage():
    import networkx as nx
    from src.common.tensors.topological_reducer import (
        reduce_abstract_tensor_topology,
    )

    graph = _ingest(
        "import networkx as nx\n"
        "def make_graph():\n"
        "    return nx.DiGraph()\n",
        {"nx": nx},
    )
    reduce_abstract_tensor_topology(graph)
    constructor = next(
        entry.graph
        for entry in graph.function_table
        if entry.name == "__init__"
        and entry.graph is not None
        and entry.graph.G.graph.get("method_owner") == "DiGraph"
    )

    contracts = constructor.G.graph["class_field_aggregate_kinds"]
    assert contracts == {
        "graph": "dict",
        "_node": "dict",
        "_adj": "dict",
        "_succ": "dict",
        "_pred": "dict",
        "__networkx_cache__": "dict",
    }
    assert constructor.G.graph["class_field_aliases"] == {"_succ": "_adj"}
    assert constructor.G.graph["class_field_value_aggregate_kinds"] == {
        "_node": "dict",
        "_adj": "dict",
        "_succ": "dict",
        "_pred": "dict",
    }


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


def test_pursuit_roots_exclude_unreachable_module_calls_without_bounding_closure():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"len": len}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                "def leaf(value):\n"
                "    return value\n\n"
                "def middle(value):\n"
                "    return leaf(value)\n\n"
                "def wanted(value):\n"
                "    return middle(value)\n\n"
                "def unrelated(values):\n"
                "    return len(values)\n"
            ),
            resolve_unresolved_parents=True,
            pursuit_roots=("wanted",),
        )

    assert graph.G.graph["unresolved_ast_calls"] == ()
    linked = {
        data["expr_obj"].func.id
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Name)
        and (data.get("attributes") or {}).get("resolved_ast_parent") is not None
    }
    assert linked == {"middle", "leaf"}


def test_direct_filtered_generator_iteration_becomes_ordinary_loop_control():
    graph = _ingest(
        """
def calls(module):
    retained = []
    for call in (
        node for node in module if isinstance(node, int)
    ):
        retained.append(call)
    return retained
""",
        {},
    )

    assert not any(
        isinstance(data.get("expr_obj"), (ast.GeneratorExp, ast.comprehension))
        for _node_id, data in graph.G.nodes(data=True)
    )
    loop = next(
        data["expr_obj"]
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    assert isinstance(loop.iter, ast.Name)
    assert loop.iter.id == "module"
    guard = loop.body[0]
    assert isinstance(guard, ast.If)
    assert isinstance(guard.body[0], ast.Assign)


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
            "target_module": "builtins",
            "target_qualname": "len",
            "owner_name": "root",
            "owner_source_identity": None,
        },
    )
    assert graph.G.graph["missing_ast_parent_calls"] == ()
    assert not _definitions(graph, "len")
    assert len(graph.function_table) == 0


def test_pursued_nested_dict_capture_lowers_sympy_find_opts_stores():
    sympy_cse = pytest.importorskip("sympy.simplify.cse_main")
    graph = _ingest(
        """
def entry(exprs):
    return opt_cse(exprs)
""",
        {"opt_cse": sympy_cse.opt_cse},
    )

    reduce_abstract_tensor_topology(graph)
    nested = graph.function_table.entry("_find_opts").graph
    capture_inputs = [
        data
        for _node_id, data in nested.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "opt_subs"
    ]
    assert capture_inputs
    assert all(
        (data.get("attributes") or {}).get("binding_kind") == "closure"
        and (data.get("attributes") or {}).get("aggregate_kind") == "dict"
        and (data.get("attributes") or {}).get("sequence_key_columns") == (0,)
        and (data.get("attributes") or {}).get("sequence_writable") is True
        for data in capture_inputs
    )
    assert sum(
        data.get("type") == "IndexedStore"
        for _node_id, data in nested.G.nodes(data=True)
    ) == 2


def test_pursued_module_dict_keeps_external_table_kind_for_re_compile():
    graph = _ingest(
        """
def entry(pattern, flags):
    return compile_re(pattern, flags)
""",
        {"compile_re": re._compile},
    )

    reduce_abstract_tensor_topology(graph)
    compiled = graph.function_table.entry("_compile").graph
    cache_inputs = [
        data
        for _node_id, data in compiled.G.nodes(data=True)
        if data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "_cache"
    ]
    assert cache_inputs
    assert all(
        (data.get("attributes") or {}).get("binding_kind") == "external"
        and (data.get("attributes") or {}).get("aggregate_kind") == "dict"
        and (data.get("attributes") or {}).get("sequence_key_columns") == (0,)
        and (data.get("attributes") or {}).get("sequence_writable") is True
        for data in cache_inputs
    )


def test_worklist_resolves_sympy_instance_assigned_method_sources():
    dpll2 = pytest.importorskip("sympy.logic.algorithms.dpll2")
    graph = _ingest(
        """
def entry(clauses, variables):
    return SATSolver(clauses, variables, set())._find_model()
""",
        {"SATSolver": dpll2.SATSolver},
    )

    assert graph.G.graph["missing_ast_parent_calls"] == ()
    expected = {
        "heur_lit_assigned": "_vsids_lit_assigned",
        "heur_lit_unset": "_vsids_lit_unset",
        "heur_clause_added": "_vsids_clause_added",
    }
    definitions = {
        data["expr_obj"].name: node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.FunctionDef)
    }
    for call_name, implementation_name in expected.items():
        implementation_id = definitions[implementation_name]
        calls = [
            node_id
            for node_id, data in graph.G.nodes(data=True)
            if isinstance(data.get("expr_obj"), ast.Call)
            and isinstance(data["expr_obj"].func, ast.Attribute)
            and data["expr_obj"].func.attr == call_name
        ]
        assert calls
        assert all(
            graph.G.has_edge(implementation_id, call_id)
            for call_id in calls
        )


def test_worklist_resolves_sympy_same_class_method_source():
    enumerative = pytest.importorskip("sympy.utilities.enumerative")
    graph = _ingest(
        """
def entry():
    return MultisetPartitionTraverser().enum_range([1], 1, 2)
""",
        {
            "MultisetPartitionTraverser": (
                enumerative.MultisetPartitionTraverser
            )
        },
    )

    decrement_id = next(
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.FunctionDef)
        and data["expr_obj"].name == "decrement_part_small"
    )
    calls = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "decrement_part_small"
    ]
    assert calls
    assert all(graph.G.has_edge(decrement_id, call_id) for call_id in calls)
    assert graph.G.graph["missing_ast_parent_calls"] == ()

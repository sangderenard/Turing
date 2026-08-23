from __future__ import annotations

import ast
import contextlib
import io
import re._constants
import subprocess
import sys

import networkx as nx

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.common.tensors.abstract_nn.token_encoder import decode_identity_tokens
from src.common.tensors.abstract_nn.token_lexicon import CompilerTokenLexicon
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_python_named_integer_becomes_plain_constant_with_origin_metadata():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {
        "ATOMIC_GROUP": re._constants.ATOMIC_GROUP,
        "OPCODES": (
            re._constants.ATOMIC_GROUP,
            re._constants.SUBPATTERN,
        ),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
def classify(value):
    return value == ATOMIC_GROUP
"""))

    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry("classify").graph.G
    constant = next(
        data
        for _node_id, data in executable.nodes(data=True)
        if (data.get("attributes") or {}).get("binding_name")
        == "ATOMIC_GROUP"
    )

    assert type(constant["attributes"]["value"]) is int
    assert constant["attributes"]["value"] == int(
        re._constants.ATOMIC_GROUP
    )
    assert constant["attributes"]["python_static_origins"] == ({
        "schema": "turing.python-named-integer.v1",
        "path": "ATOMIC_GROUP",
        "module": "re._constants",
        "type": "_NamedIntConstant",
        "name": "ATOMIC_GROUP",
        "integer_value": int(re._constants.ATOMIC_GROUP),
    },)
    assert graph.python_bindings["OPCODES"] == (
        int(re._constants.ATOMIC_GROUP),
        int(re._constants.SUBPATTERN),
    )
    assert all(type(value) is int for value in graph.python_bindings["OPCODES"])

    from src.compiler.project_compilation_product import (
        _dump_resolved_process_graph,
    )

    serialized = io.BytesIO()
    _dump_resolved_process_graph(graph, serialized)
    assert serialized.tell() > 0


def test_tuple_comprehension_publishes_fixed_resident_row_width():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
def rows(values: list[int]):
    return [(value, value + 1) for value in values]
"""))

    reduce_abstract_tensor_topology(graph)
    executable = graph.function_table.entry("rows").graph.G
    materializer_id, materializer = next(
        (node_id, data)
        for node_id, data in executable.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.ListComp)
    )

    assert materializer["attributes"]["sequence_column_count"] == 2

    from src.compiler.fortran_c_shell import _field_slot_ops

    declarations = _field_slot_ops(executable)[8]
    assert next(
        columns
        for sequence_id, _policy, columns, _writable in declarations
        if sequence_id == materializer_id
    ) == 2


def test_ingestion_identity_ledger_versions_authored_rebindings_before_ssa():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("""
def kernel(x):
    signal = x + 1
    signal = signal * 2
    return signal
"""))
    reduce_abstract_tensor_topology(graph)

    executable = graph.function_table.entry(
        graph.function_table.reference("kernel")
    ).graph.G
    records = executable.graph["ingestion_identity_table"]["signal"]
    assert [record["version"] for record in records] == [0, 1]
    assert len({record["value_id"] for record in records}) == 2
    assert all(record["spelling"] == "signal" for record in records)
    assert all(len(record["context_sha256"]) == 64 for record in records)
    assert all(record["context_tokens"] for record in records)
    assert records[0]["context"]["producer_op"] == "Add"
    assert records[0]["context"]["dependency_shape"]
    for record in records:
        assert tuple(
            decode_identity_tokens(token_id)["token"]
            for token_id in record["context_token_ids"]
        ) == record["context_tokens"]


def test_unchanged_source_rebuilds_the_same_dense_token_ordered_ids():
    source = """
def kernel(x):
    signal = x + 1
    signal = signal * 2
    return signal
"""

    def snapshot():
        graph = ProcessGraph(materialize_memory=False)
        with contextlib.redirect_stdout(io.StringIO()):
            graph.build_from_ast(ast.parse(source))
        reduce_abstract_tensor_topology(graph)
        executable = graph.function_table.entry(
            graph.function_table.reference("kernel")
        ).graph.G
        return {
            "ids": tuple(executable.nodes),
            "tokens": dict(executable.graph["ssa_identity_tokens"]),
            "ingestion": dict(executable.graph["ingestion_identity_table"]),
            "ops": tuple(
                (
                    value_id,
                    data.get("op"),
                    tuple(data.get("parents") or ()),
                )
                for value_id, data in executable.nodes(data=True)
            ),
        }

    first = snapshot()
    second = snapshot()
    assert first == second
    assert first["ids"] == tuple(range(len(first["ids"])))


def test_unchanged_source_ids_are_stable_across_python_processes():
    script = r'''
import ast
import contextlib
import io
import json

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.transmogrifier.graph.graph_express2 import ProcessGraph

source = """
def kernel(x):
    signal = x + 1
    signal = signal * 2
    return signal
"""
graph = ProcessGraph(materialize_memory=False)
with contextlib.redirect_stdout(io.StringIO()):
    graph.build_from_ast(ast.parse(source))
reduce_abstract_tensor_topology(graph)
executable = graph.function_table.entry(
    graph.function_table.reference("kernel")
).graph.G
print(json.dumps({
    "ids": list(executable.nodes),
    "tokens": executable.graph["ssa_identity_tokens"],
    "ops": [
        [value_id, str(data.get("op")), list(data.get("parents") or ())]
        for value_id, data in executable.nodes(data=True)
    ],
}, sort_keys=True))
'''
    first = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True,
    ).stdout
    second = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True,
    ).stdout

    assert first == second


def test_compiler_token_lexicon_learns_labeled_structural_contexts():
    context = {
        "node_kind": "Name",
        "role": "assignment_target",
        "target": "Name(id='signal', ctx=Store())",
        "dependency_shape": ("Input", "Add"),
    }
    lexicon = CompilerTokenLexicon().observe_contexts((context,))

    assert "field:role" in lexicon.token_ids
    assert "value:assignment_target" in lexicon.token_ids
    assert lexicon.counts["field:role"] == 1
    assert lexicon.token_id("field:role") == lexicon.token_ids["field:role"]
    statistics = next(iter(lexicon.context_statistics.values()))
    assert statistics["count"] == 1
    assert statistics["token_ids"]

    upgraded = lexicon.upgrade_document({
        "lexicon_revision": 0,
        "context": context,
        "context_token_ids": statistics["token_ids"],
    })
    assert upgraded["lexicon_revision"] == lexicon.revision
    assert upgraded["context_token_ids"] == statistics["token_ids"]


def test_schema_guard_constructor_idiom_becomes_one_normalization_operator():
    module = ast.parse(
        """
def kernel(x):
    return x if isinstance(x, AbstractTensor) else AbstractTensor.tensor(x)
"""
    )
    conditional = next(
        node for node in ast.walk(module) if isinstance(node, ast.IfExp)
    )
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    guard = next(
        node for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance"
    )
    normalizer = next(
        node for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "tensor"
    )
    guard._extraction_contract = {
        "identity": "builtins.isinstance", "action": "intrinsic",
    }
    normalizer._extraction_contract = {
        "identity": (
            "src.common.tensors.abstraction.AbstractTensor.tensor"
        ),
        "action": "intrinsic",
    }
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)

    reduce_abstract_tensor_topology(graph)

    executable = graph.function_table.entry(
        graph.function_table.reference("kernel")
    ).graph.G
    normalized = next(
        data
        for _node_id, data in executable.nodes(data=True)
        if (data.get("attributes") or {}).get(
            "source_type_normalization"
        )
    )
    assert not any(
        isinstance(data.get("expr_obj"), ast.IfExp)
        for _node_id, data in executable.nodes(data=True)
    )
    assert normalized["op"] == "tensor"
    assert normalized["attributes"]["source_type_normalization"] == {
        "guard": "schema_type_guard",
        "schema_type": (
            "src.common.tensors.abstraction.AbstractTensor"
        ),
        "source_ifexp": id(conditional),
    }


def test_annotated_receiver_discovers_authored_method_without_instance():
    class Receiver:
        def scale(self, value):
            return value * 2.0

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"Receiver": Receiver}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            "def kernel(receiver: Receiver, value):\n"
            "    return receiver.scale(value)\n",
            resolve_unresolved_parents=True,
            pursuit_roots=("kernel",),
        )
    reduce_abstract_tensor_topology(graph)

    scale = graph.function_table.reference("scale")
    assert scale is not None
    kernel = graph.function_table.entry("kernel").graph
    call = next(
        data
        for _node_id, data in kernel.G.nodes(data=True)
        if (data.get("attributes") or {}).get("callee_ref") is not None
    )
    assert call["attributes"]["callee_ref"] == scale.address


def test_descendant_loop_targets_are_not_enclosing_loop_carried_state():
    module = ast.parse(
        """
def kernel(rows):
    i = 99
    size = 88
    for row in rows:
        for i, size in enumerate(row):
            pass
    return 0
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)

    reduce_abstract_tensor_topology(graph)

    outer = module.body[0].body[2]
    carried = (
        graph.G.nodes[id(outer)].get("attributes") or {}
    ).get("loop_carried_bindings", {})
    assert "i" not in carried
    assert "size" not in carried


def test_augmented_assignment_is_exact_loop_carried_state():
    module = ast.parse(
        """
def kernel(max_iters):
    iters = 0
    while iters < max_iters:
        iters += 1
    return iters
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)

    reduce_abstract_tensor_topology(graph)

    loop = module.body[0].body[1]
    carried = (
        graph.G.nodes[id(loop)].get("attributes") or {}
    ).get("loop_carried_bindings", {})
    assert "iters" in carried
    initial, updated = carried["iters"]
    assert initial != updated


def test_only_name_assign_and_call_receive_existing_process_graph_aliases():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def kernel(x):
    y = helper(x)
"""
            )
        )

    original_types = {
        node_id: data["type"] for node_id, data in graph.G.nodes(data=True)
    }
    reduced = reduce_abstract_tensor_topology(graph)

    assert reduced is graph
    aliases = {"Name": "Load", "Assign": "Store", "Call": "Call"}
    for node_id, original_type in original_types.items():
        data = graph.G.nodes[node_id]
        if original_type in aliases:
            assert data["type"] == aliases[original_type]
            assert data["op"] == aliases[original_type]
            assert data["attributes"]["source_type"] == original_type
        else:
            assert data["type"] == original_type


def test_annotated_assignments_use_one_ingestion_operator_in_functions_only():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(
            """
class Example:
    class_value: float = 1.0

    def method(self, value):
        local_value: float = value
        return local_value
"""
        ))

    # A class body's own ``AnnAssign`` is its field schema -- the class-table
    # builder (topological_reducer.py's ``class_field_defaults``/``fields``)
    # reads it directly off the untouched ``ClassDef`` AST, so it must
    # survive ingestion unnormalized.  Only a method body's local-variable
    # annotation is executable code, and that one is normalized to ``Assign``
    # the same way it always was.
    class_body_ann_assigns = [
        data
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AnnAssign)
    ]
    assert len(class_body_ann_assigns) == 1

    assignments = [
        data
        for _node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Assign)
    ]
    assert len(assignments) == 1

    reduce_abstract_tensor_topology(graph)

    assert all(data["type"] == "Store" for data in assignments)
    assert graph.G.graph["map_ir"]["schema"]["classes"][0][
        "members"
    ][0]["annotation"] == "float"


def test_expr_wrapper_is_removed_without_removing_its_interior():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("helper(3)\n"))

    expr_ids = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data["type"] == "Expr"
    ]
    interior_ids = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data["type"] == "Call"
    ]
    assert expr_ids
    assert interior_ids
    reduce_abstract_tensor_topology(graph)

    assert not any(node_id in graph.G for node_id in expr_ids)
    assert all(node_id in graph.G for node_id in interior_ids)


def test_nested_matrix_multiplication_is_numerical_topology_not_a_call():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def dct(blocks, transform):
    return transform @ blocks @ transform
"""
    )
    matrix_products = [
        node for node in ast.walk(module) if isinstance(node, ast.BinOp)
    ]
    assert len(matrix_products) == 2

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    for expression in matrix_products:
        data = graph.G.nodes[id(expression)]
        assert data["type"] == "matmul"
        assert data["op"] == "matmul"
        assert data["parents"] == [
            (id(expression.left), "lhs"),
            (id(expression.right), "rhs"),
        ]


def test_ellipsis_index_is_lowered_as_one_parent_aware_index_operation():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(tensor, zigzag):
    return tensor[..., zigzag]
"""
    )
    subscript = next(
        node for node in ast.walk(module) if isinstance(node, ast.Subscript)
    )
    authored_index_tuple = subscript.slice
    assert isinstance(authored_index_tuple, ast.Tuple)

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    indexed = graph.G.nodes[id(subscript)]
    assert indexed["type"] == "Indexed"
    # Ellipsis is expanded before ingestion into one ndim-driven index value.
    # The authored Subscript identity remains stable, while its former Tuple
    # and Ellipsis leaves are deliberately replaced by the explicit index DAG.
    assert indexed["parents"] == [
        (id(subscript.value), "base"),
        (id(subscript.slice), "index"),
    ]
    assert not any(
        isinstance(node, ast.Constant) and node.value is Ellipsis
        for node in ast.walk(subscript)
    )
    assert id(authored_index_tuple) not in graph.G


def test_imports_are_retained_as_logged_contextual_requirements(caplog):
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    from package.math import helper as operation
    return operation(x)
"""
    )
    imported = next(
        node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
    )
    imported_name = imported.names[0]

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    with caplog.at_level(
        "INFO",
        logger="src.common.tensors.topological_reducer",
    ):
        reduce_abstract_tensor_topology(graph)

    requirement = {
        "kind": "import_from",
        "module": "package.math",
        "level": 0,
        "names": (("helper", "operation"),),
    }
    assert graph.G.graph["contextual_requirements"] == (requirement,)
    assert (
        graph.G.nodes[id(imported)]["attributes"][
            "contextual_requirement"
        ]
        == requirement
    )
    assert (
        graph.G.nodes[id(imported_name)]["attributes"][
            "contextual_requirement"
        ]
        == requirement
    )
    assert "retaining ProcessGraph import" in caplog.text


def test_imported_calls_use_distinct_external_python_table():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
from operator import add

def combine(left, right):
    return add(left, right)
"""
            )
        )
    reduce_abstract_tensor_topology(graph)

    assert graph.function_table.reference("combine") is not None
    assert graph.function_table.reference("add") is None
    external_reference = graph.external_function_table.reference("add")
    assert external_reference is not None
    call = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("external_callee_ref")
        == external_reference.address
    )
    assert "callee_ref" not in call["attributes"]

    graph.external_function_table.resolve_imports()
    assert graph.external_function_table.invoke(
        external_reference,
        4,
        7,
    ) == 11


def test_calls_reference_separate_local_function_subgraphs():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def helper(x):
    return x + 1

def kernel(x):
    return helper(x)
"""
    )
    helper = module.body[0]
    call = next(
        node for node in ast.walk(module.body[1])
        if isinstance(node, ast.Call)
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    reference = graph.function_table.reference("helper")
    assert reference is not None
    assert (
        graph.G.nodes[id(call)]["attributes"]["callee_ref"]
        == reference.address
    )
    entry = graph.function_table.entry(reference)
    assert entry.graph is not graph
    assert id(helper) not in entry.graph.G
    assert any(
        data.get("type") == "Input" and data.get("label") == "x"
        for _node_id, data in entry.graph.G.nodes(data=True)
    )
    assert any(
        data.get("type") == "Add"
        for _node_id, data in entry.graph.G.nodes(data=True)
    )
    assert entry.graph.function_table is graph.function_table


def test_bound_method_dereference_is_an_explicit_ssa_accessor():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Graph:
    def connect(self, value):
        self.last = value

    def build(self, values):
        for value in values:
            self.connect(value)
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    build_graph = graph.function_table.entry("build").graph
    call_id, call = next(
        (node_id, data)
        for node_id, data in build_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    accessor_id, accessor = next(
        (node_id, data)
        for node_id, data in build_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Attribute)
    )
    method_ref = graph.G.graph["class_table"]["Graph"]["methods"][
        "connect"
    ]

    assert call["attributes"]["method_ref"] == method_ref
    assert accessor["attributes"]["accessor_kind"] == "method"
    assert accessor["attributes"]["method_ref"] == method_ref
    assert (accessor_id, "callee") in call["parents"]
    loop = next(
        data
        for _node_id, data in build_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    assert not (loop.get("attributes") or {}).get("loop_state_effects")


def test_method_resolution_follows_an_authored_function_returned_class():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Worker:
    def apply(self, value):
        return value + 1

def build_worker() -> Worker:
    return Worker()

def run(value):
    worker = build_worker()
    return worker.apply(value)
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    run_graph = graph.function_table.entry("run").graph
    calls = {
        ast.unparse(data["expr_obj"]): data
        for _node_id, data in run_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    }
    method_ref = graph.G.graph["class_table"]["Worker"]["methods"][
        "apply"
    ]
    assert calls["build_worker()"]["attributes"]["result_class_ref"] == (
        "Worker"
    )
    assert calls["worker.apply(value)"]["attributes"]["method_ref"] == (
        method_ref
    )


def test_field_aggregate_contract_follows_an_authored_method_result():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Value:
    def __init__(self):
        self.accounting = {}

class Builder:
    def fresh_value(self) -> Value:
        return Value()

    def lower(self, rows):
        for row in rows:
            value = self.fresh_value()
            value.accounting.update({"row": row})
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("lower").graph
    field = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Attribute)
        and ast.unparse(data["expr_obj"]) == "value.accounting"
    )

    assert field["attributes"]["aggregate_kind"] == "dict"
    assert field["attributes"]["record_field"] == ("Value", "accounting")
    loop = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effect, = loop["attributes"]["loop_state_effects"]
    assert effect["operator"] == "update"
    assert effect["effect_mode"] == "mapping_mutation"
    assert effect["sequence_policy"] == "unique"


def test_setdefault_result_retains_nested_collection_identity_in_a_loop():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def group_rows(rows):
    groups = {}
    for key, value in rows:
        group = groups.setdefault(key, set())
        group.add(value)
    return groups
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("group_rows").graph
    loop = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effects = {
        effect["operator"]: effect
        for effect in loop["attributes"]["loop_state_effects"]
    }

    assert effects["setdefault"]["effect_mode"] == "mapping_mutation"
    assert effects["add"]["effect_mode"] == "sequence_mutation"
    assert effects["add"]["sequence_policy"] == "unique"


def test_imported_dataclass_field_factory_is_a_record_aggregate_contract():
    from src.transmogrifier.ssa import SSAValue

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"SSAValue": SSAValue}
    module = ast.parse(
        """
from src.transmogrifier.ssa import SSAValue

def make_value() -> SSAValue:
    return SSAValue(1)

def kernel(rows):
    for row in rows:
        value = make_value()
        value.accounting.update({"row": row})
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("kernel").graph
    field = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Attribute)
        and ast.unparse(data["expr_obj"]) == "value.accounting"
    )
    loop = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effect, = loop["attributes"]["loop_state_effects"]

    assert field["attributes"]["aggregate_kind"] == "dict"
    assert field["attributes"]["record_field"] == ("SSAValue", "accounting")
    assert effect["effect_mode"] == "mapping_mutation"


def test_annotated_mapping_field_retains_key_and_record_value_contract():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class SSAValue:
    pass

class Builder:
    def __init__(self):
        self.external_values: dict[int, SSAValue | None] = {}

    def restore(self, key: int, previous: SSAValue | None):
        self.external_values[key] = previous
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("restore").graph
    field = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Attribute)
        and ast.unparse(data["expr_obj"]) == "self.external_values"
    )

    assert field["attributes"]["mapping_key_dtype"] == "int64"
    assert field["attributes"]["mapping_value_dtype"] == "int64"
    assert field["attributes"]["mapping_value_record"] == "SSAValue"
    assert field["attributes"]["mapping_value_optional"] is True


def test_lazy_class_mapping_initialization_types_phi_and_setdefault():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Groups:
    def add_rows(self, rows):
        groups = getattr(self, "_groups", None)
        if groups is None:
            groups = {}
            self._groups = groups
        for key, value in rows:
            group = groups.setdefault(key, set())
            group.add(value)
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("add_rows").graph
    phi = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Phi" and data.get("label") == "groups"
    )
    loop = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    )
    effects = {
        effect["operator"]: effect
        for effect in loop["attributes"]["loop_state_effects"]
    }

    assert phi["attributes"]["aggregate_kind"] == "dict"
    assert phi["attributes"]["mapping_value_aggregate_kind"] == "set"
    assert effects["setdefault"]["effect_mode"] == "mapping_mutation"
    assert effects["add"]["effect_mode"] == "sequence_mutation"


def test_mapping_pop_requires_tombstone_storage_without_duplicate_deletion():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Builder:
    def __init__(self):
        self.external_values = {}

    def lower(self, keys):
        for key in keys:
            previous = self.external_values.pop(key, None)
            if previous is not None:
                pass
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    executable = graph.function_table.entry("lower").graph.G
    external_values = next(
        int(data.get("value_id", node_id))
        for node_id, data in executable.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Attribute)
        and ast.unparse(data["expr_obj"]) == "self.external_values"
    )

    from src.compiler.fortran_c_shell import _field_slot_ops

    contract = _field_slot_ops(executable)
    table_deletions = contract[13]
    tombstone_sequence_ids = contract[17]

    assert table_deletions == ()
    assert tombstone_sequence_ids == (external_values,)


def test_descendant_loop_owns_its_sequence_mutation_effect():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(rows):
    results = []
    for row in rows:
        for value in row:
            results.extend((value,))
    return results
"""
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("kernel").graph
    loops = {
        data["expr_obj"].lineno: data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    }
    assert not (loops[4].get("attributes") or {}).get(
        "loop_state_effects"
    )
    inner_effect, = loops[5]["attributes"]["loop_state_effects"]
    assert inner_effect["operator"] == "extend"
    assert inner_effect["effect_mode"] == "sequence_mutation"
    assert inner_effect["sequence_policy"] == "duplicates"


def test_imported_callee_is_an_external_function_reference():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    from package.math import helper as operation
    return operation(x)
"""
    )
    call = next(node for node in ast.walk(module) if isinstance(node, ast.Call))

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    assert graph.function_table.reference("operation") is None
    reference = graph.external_function_table.reference("operation")
    assert reference is not None
    assert (
        graph.G.nodes[id(call)]["attributes"]["external_callee_ref"]
        == reference.address
    )
    assert graph.external_function_table.entry(reference).qualified_name == (
        "package.math.helper"
    )


def test_recursive_call_is_retained_as_a_function_table_backedge():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def recur(x):
    return recur(x)
"""
    )
    call = next(node for node in ast.walk(module) if isinstance(node, ast.Call))

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    reference = graph.function_table.reference("recur")
    assert reference is not None
    assert graph.function_table.entry(reference).recursive
    assert graph.G.nodes[id(call)]["attributes"]["recursive_backedge"] is True


def test_lexical_occurrences_are_unique_then_reduce_to_monotonic_fanout():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    first = x + 1
    second = first * first
    return second
"""
    )
    first_loads = [
        node
        for node in ast.walk(module)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "first"
        )
    ]
    assert len(first_loads) == 2
    assert id(first_loads[0]) != id(first_loads[1])

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    assert all(id(node) in graph.G for node in first_loads)

    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert list(function_graph.G) == list(range(len(function_graph.G)))
    assert all(
        data["value_id"] == node_id
        for node_id, data in function_graph.G.nodes(data=True)
    )
    assert not any(
        data.get("type") in {"Load", "Store"}
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    inputs = [
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ]
    assert len(inputs) == 1
    add = next(
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    multiply = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Mul"
    )
    assert multiply["parents"] == [(add, "lhs"), (add, "rhs")]


def test_static_python_reference_chain_collapses_without_becoming_input():
    class StaticTensorAPI:
        @staticmethod
        def stack(values, dim=0):
            raise AssertionError("static collapse must not execute the method")

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"StaticTensorAPI": StaticTensorAPI}
    module = ast.parse(
        """
def kernel(values):
    return StaticTensorAPI.stack(values, dim=0)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert not any(
        data.get("label") == "StaticTensorAPI"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    operation = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "stack"
    )
    assert operation["attributes"]["static_python_reference"] == (
        "StaticTensorAPI.stack"
    )
    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["values"]


def test_static_python_reference_alias_uses_integer_wrapper_node():
    class StaticTensorAPI:
        @staticmethod
        def stack(values, dim=0):
            raise AssertionError("static collapse must not execute the method")

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"StaticTensorAPI": StaticTensorAPI}
    module = ast.parse(
        """
def kernel(values):
    api = StaticTensorAPI
    operation = api.stack
    return operation(values, dim=0)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert all(isinstance(node_id, int) for node_id in function_graph.G)
    reference_id, reference = next(
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "StaticReference"
    )
    assert reference["attributes"]["static_python_reference"] == (
        "StaticTensorAPI.stack"
    )
    operation = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Call"
    )
    assert (reference_id, "callee") in operation["parents"]
    assert not any(
        data.get("label") in {"api", "operation"}
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_lambda_receives_an_anonymous_function_subgraph():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(value):
    transform = lambda item: item + 1
    return transform(value)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    lambda_entry = next(
        entry
        for entry in graph.function_table
        if entry.metadata.get("source_type") == "Lambda"
    )
    assert lambda_entry.name.startswith("<lambda:")
    assert lambda_entry.graph is not None
    assert lambda_entry.graph.G.graph["function_parameters"] == ("item",)
    call = next(
        data
        for _node_id, data in graph.function_table.entry("kernel").graph.G.nodes(
            data=True
        )
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert call["attributes"]["callee_ref"] == lambda_entry.reference.address


def test_augmented_assignment_lowers_to_value_operation():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def accumulate(value, increment):
    value += increment
    return value
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("accumulate").graph

    assert not any(
        isinstance(data.get("expr_obj"), ast.AugAssign)
        and data.get("type") == "AugAssign"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    add = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    assert len(add["parents"]) == 2


def test_named_expression_binds_and_forwards_its_value():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def select(value):
    return (saved := value) + saved
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("select").graph

    assert not any(
        isinstance(data.get("expr_obj"), ast.NamedExpr)
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    add = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    assert len(add["parents"]) == 2
    assert add["parents"][0][0] == add["parents"][1][0]


def test_unary_plus_forwards_the_existing_value():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("def positive(value):\n    return +value\n"))
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("positive").graph

    assert not any(
        data.get("type") == "Cast"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["value"]


def test_referenced_builtin_is_not_mislabeled_as_runtime_input():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def freeze(values):\n    return tuple(values)\n")
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("freeze").graph

    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["values"]
    call = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert call["attributes"]["static_python_reference"] == "tuple"


def test_branch_assignments_merge_before_later_use():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def choose(condition, left, right):
    if condition:
        selected = left
    else:
        selected = right
    return selected
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("choose").graph

    phi = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Phi"
    )
    assert {role for _parent, role in phi["parents"]} == {
        "test",
        "body",
        "orelse",
    }
    assert not any(
        data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "selected"
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_referenced_module_literal_becomes_constant_not_input():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"TABLE": ((1, 2), (3, 4))}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def lookup():\n    return TABLE\n")
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("lookup").graph

    assert not any(
        data.get("type") == "Input"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    constant = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Constant"
    )
    assert constant["constant"] == ((1, 2), (3, 4))


def test_referenced_module_literal_remains_a_call_argument():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"TABLE": (1, 2, 3)}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def consume(value):
    return value

def entry():
    return consume(TABLE)
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("entry").graph
    call = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert any(
        function_graph.G.nodes[parent].get("constant") == (1, 2, 3)
        for parent, role in call["parents"]
        if role == "arg:0"
    )


def test_reducer_preserves_every_python_constant_literal():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def constants():
    return (8, 0.5, None, ..., b"jpeg")
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("constants").graph

    literals = [
        data["constant"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Constant"
    ]
    assert literals == [8, 0.5, None, Ellipsis, b"jpeg"]


def test_floor_division_remains_distinct_from_true_division():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def divide(x):
    return (x / 2, x // 2)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("divide").graph

    operations = [
        data["type"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") in {"Div", "FloorDiv"}
    ]
    assert operations == ["Div", "FloorDiv"]


def test_attribute_assignment_lowers_to_setattr_with_object_and_value():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Accumulator:
    def append(self, value):
        self._pending = value
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("Accumulator.append").graph

    set_attr = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "SetAttr"
    )
    assert set_attr["attributes"]["attribute"] == "_pending"
    assert {role for _parent, role in set_attr["parents"]} == {
        "object",
        "value",
    }
    assert not any(
        data.get("type") == "Attribute"
        and isinstance(data.get("expr_obj"), ast.Attribute)
        and isinstance(data["expr_obj"].ctx, ast.Store)
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_static_python_attribute_value_remains_an_explicit_host_boundary():
    class Registry:
        handler = object()

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"registry": Registry}
    module = ast.parse(
        """
def attach(parameter):
    parameter.callback = registry.handler
    return parameter
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("attach").graph

    set_attr = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "SetAttr"
    )
    value_id = next(
        parent for parent, role in set_attr["parents"] if role == "value"
    )
    assert function_graph.G.nodes[value_id]["type"] == "StaticReference"
    assert set_attr["attributes"]["static_value_boundary"] == "registry.handler"


def test_autograd_tape_assignment_remains_an_explicit_recursive_seam():
    class AutogradState:
        tape = object()

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"autograd": AutogradState}
    module = ast.parse(
        """
def attach(parameter):
    parameter._tape = autograd.tape
    return parameter
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("attach").graph

    set_attr = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "SetAttr"
        and data.get("attributes", {}).get("attribute") == "_tape"
    )
    value_id = next(
        parent for parent, role in set_attr["parents"] if role == "value"
    )
    assert function_graph.G.nodes[value_id]["type"] == "StaticReference"
    assert set_attr["attributes"]["static_value_boundary"] == "autograd.tape"


def test_static_class_attribute_update_uses_reference_and_latest_ssa_value():
    class Counter:
        value = 0

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"Counter": Counter}
    module = ast.parse(
        """
def increment():
    Counter.value += 1
    return Counter.value
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("increment").graph

    reference_id, reference = next(
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "StaticReference"
    )
    set_attr_id, set_attr = next(
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "SetAttr"
    )
    updated_value = next(
        parent
        for parent, role in set_attr["parents"]
        if role == "value"
    )

    assert reference["attributes"]["static_python_reference"] == "Counter"
    assert (reference_id, "object") in set_attr["parents"]
    assert set_attr["attributes"]["attribute"] == "value"
    assert function_graph.G.graph["identity_table"]["result_0"] == (
        updated_value,
    )
    assert set_attr_id != updated_value


def test_scope_declarations_do_not_become_runtime_operators():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
shared = 0

def outer(value):
    captured = value

    def update():
        nonlocal captured
        global shared
        captured += 1
        shared = captured
        return captured

    return update()
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("update").graph

    for candidate in (graph, function_graph):
        assert not any(
            data.get("type") in {"Nonlocal", "Global"}
            for _node_id, data in candidate.G.nodes(data=True)
        )


def test_delete_lowers_names_away_and_preserves_object_effects():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def discard(mapping, key, owner, temporary):
    del temporary
    del mapping[key]
    del owner.cached
    return mapping
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("discard").graph

    for candidate in (graph, function_graph):
        assert not any(
            data.get("type") == "Delete"
            for _node_id, data in candidate.G.nodes(data=True)
        )

    del_item = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "DelItem"
    )
    del_attr = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "DelAttr"
    )
    assert {role for _parent, role in del_item["parents"]} == {
        "base",
        "index",
    }
    assert {role for _parent, role in del_attr["parents"]} == {"object"}
    assert del_attr["attributes"]["attribute"] == "cached"


def test_try_value_survives_when_every_exception_handler_terminates():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def convert(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid") from error
    return numeric
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("convert").graph

    assert not any(
        data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "numeric"
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_try_else_consumes_the_successful_body_value_before_path_merge():
    """A try's structural else edge must not feed back through its value merge."""

    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def report(value, sink):
    computed = None
    try:
        computed = int(value)
    except Exception as error:
        detail = str(error)
    else:
        sink(computed)
    return computed
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("report").graph.G

    assert nx.is_directed_acyclic_graph(function_graph)
    try_node = next(
        node_id for node_id, data in function_graph.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Try)
    )
    else_call = next(
        (node_id, data) for node_id, data in function_graph.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and getattr(data["expr_obj"].func, "id", None) == "sink"
    )
    assert try_node not in {
        parent for parent, _role in else_call[1].get("parents", ())
    }


def test_generator_expression_target_does_not_leak_into_later_same_named_loop():
    # A comprehension/generator-expression `for` target has owned its own
    # scope since Python 3.0 -- it is never visible outside the
    # comprehension, unlike a bare `for` statement's target. Before this was
    # fixed, `environment["address"]` set while resolving
    # `tuple(... for address in xs)` was never undone, so the *unrelated*
    # `for address in ys:` statement two lines later picked up the
    # comprehension's own (later-evaporated) node as its "value of address
    # before this loop" -- surfacing much later, once that node was deleted,
    # as an unrelated "missing ProcessGraph input" KeyError with no mention
    # of a comprehension anywhere nearby.
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def repro(xs, ys):
    tagged = tuple(int(address) for address in xs)
    total = 0
    for address in ys:
        total = total + address
    return total, tagged
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("repro").graph

    for_loops = [
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    ]
    assert len(for_loops) == 1
    _node_id, loop_data = for_loops[0]
    attributes = loop_data.get("attributes") or {}
    # The real `for address in ys:` has nothing bound before it -- the
    # comprehension's own `address` must not appear as its "initial" value.
    assert "address" not in attributes.get("loop_target_initials", {})


def test_generator_expression_materialization_pass_also_does_not_leak_target():
    # Same as above but for tuple()/list() around the generator expression,
    # which topological_reducer.py resolves the generator's `elt` a second
    # time (to attach loop_iteration_outputs metadata) through a separate,
    # originally-unscoped code path -- fixing only the first pass left this
    # second one still leaking.
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def repro(xs, ys):
    tagged = tuple(int(address) & 1 for address in xs)
    total = 0
    for address in ys:
        total = total + address
    return total, tagged
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("repro").graph

    for_loops = [
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.For)
    ]
    assert len(for_loops) == 1
    _node_id, loop_data = for_loops[0]
    attributes = loop_data.get("attributes") or {}
    assert "address" not in attributes.get("loop_target_initials", {})


def test_generator_target_leak_no_longer_crashes_the_real_aot_pipeline():
    # End-to-end regression for the bug these two tests isolate: compiling
    # the exact repro shape through the real compiler entrypoint (not just
    # the reducer) used to fail with a bare `KeyError` from deep inside
    # capture, referencing a node number with no readable connection to a
    # comprehension five lines above the code that actually failed.
    from src.common.tensors.accelerator_backends.aot_compile import (
        compile_ast_aot,
    )

    source = """
def repro(xs, ys, flag):
    tagged = tuple(int(address) for address in xs)
    total = 0
    for address in ys:
        total = total + address
    if flag:
        return total + len(tagged)
    return total
"""
    for xs, ys, flag in [
        ((), (), True),
        ((1, 2), (), True),
        ((), (3, 4), True),
        ((1, 2), (3, 4), False),
    ]:
        compile_ast_aot(
            source, "repro", {"xs": xs, "ys": ys, "flag": flag},
            precompile_only=True,
        )


def test_subscript_write_to_attribute_field_orders_before_later_bare_read():
    """``obj.field[i, j] = ...`` must order before a later bare ``obj.field``
    read of the same field -- see ``tools/HANDOFF_2026-08-17_CRASH.md``. The
    read node used to depend only on the receiver (``state``), never on the
    element-wise write, so nothing stopped the scheduler from placing the
    read before the loop that produces the field's contents.
    """
    module = ast.parse(
        """
def kernel(state):
    for row in range(2):
        for col in range(2):
            state.next_height[row, col] = 1.0
    state.height = state.next_height + 0.0
    return state
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)

    reduce_abstract_tensor_topology(graph)

    function_graph = graph.function_table.entry("kernel").graph

    indexed_store = next(
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "IndexedStore"
    )
    next_height_reads = [
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "GetAttr"
        and (data.get("attributes") or {}).get("attribute") == "next_height"
    ]
    # One GetAttr resolves ``state.next_height`` as the IndexedStore's own
    # ``base`` (inside the loop); the other is the post-loop bare read
    # feeding ``state.next_height + 0.0``. Exclude the store's own base.
    base_parents = {
        parent
        for parent, role in function_graph.G.nodes[indexed_store]["parents"]
        if role == "base"
    }
    post_loop_reads = [
        node_id for node_id in next_height_reads if node_id not in base_parents
    ]
    assert post_loop_reads, "expected a bare post-loop read of state.next_height"
    for read_id in post_loop_reads:
        assert nx.has_path(function_graph.G, indexed_store, read_id), (
            "the post-loop read of state.next_height must be ordered after "
            "the loop's element-wise write to it"
        )

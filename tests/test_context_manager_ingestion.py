import ast

import pytest

from src.transmogrifier.graph.node_special_cases import inline_context_managers
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.python_special_cases import (
    interpret_python_special_case,
    lower_python_shell_file_contexts,
)


def _definition(source: str) -> ast.FunctionDef:
    return ast.parse(source).body[0]


def test_context_template_binds_yield_and_closes_on_body_failure():
    tree = ast.parse("""
events = []
with managed(events, 7) as value:
    events.append(value)
    raise RuntimeError("body")
""")
    manager = _definition("""
def managed(events, value):
    events.append("enter")
    try:
        yield value
    finally:
        events.append("exit")
""")

    inline_context_managers(tree, lambda call: manager)

    assert not any(isinstance(node, ast.With) for node in ast.walk(tree))
    namespace = {}
    with pytest.raises(RuntimeError, match="body"):
        exec(compile(tree, "<context-template>", "exec"), namespace)
    assert namespace["events"] == ["enter", 7, "exit"]


def test_context_templates_nest_cleanup_around_inner_acquisition():
    tree = ast.parse("""
events = []
with managed(events, "outer"), broken(events):
    events.append("body")
""")
    managed = _definition("""
def managed(events, name):
    events.append("enter-" + name)
    try:
        yield name
    finally:
        events.append("exit-" + name)
""")
    broken = _definition("""
def broken(events):
    events.append("broken")
    raise RuntimeError("acquire")
    yield
""")
    definitions = {"managed": managed, "broken": broken}

    inline_context_managers(
        tree,
        lambda call: definitions[call.func.id],
    )

    namespace = {}
    with pytest.raises(RuntimeError, match="acquire"):
        exec(compile(tree, "<nested-context-template>", "exec"), namespace)
    assert namespace["events"] == ["enter-outer", "broken", "exit-outer"]


def test_context_template_keeps_statements_after_a_direct_yield():
    tree = ast.parse("""
events = []
with managed(events):
    events.append("body")
""")
    manager = _definition("""
def managed(events):
    events.append("enter")
    yield
    events.append("exit")
""")

    inline_context_managers(tree, lambda call: manager)

    namespace = {}
    exec(compile(tree, "<direct-yield-template>", "exec"), namespace)
    assert namespace["events"] == ["enter", "body", "exit"]


def test_unreadable_file_manager_remains_for_the_shell_boundary():
    tree = ast.parse("""
with artifact.open("rb") as stream:
    payload = stream.read()
""")

    inline_context_managers(tree, lambda call: None)

    assert any(isinstance(node, ast.With) for node in ast.walk(tree))


def _file_receipt(identity: str) -> dict:
    return {
        "action": "python_host_call",
        "rule_id": "python-filesystem-is-a-shell-boundary",
        "identity": identity,
        "classification": "python_source",
        "parameters": {
            "execution": "shell_io.file_broker",
            "shell_capability": "files",
            "shell_abi": "turing-shell-io-abi.files",
        },
    }


def test_resolved_file_context_becomes_ordered_shell_operations():
    tree = ast.parse("""
def save(path, payload):
    with open(path, "wb") as stream:
        count = stream.write(payload)
        stream.flush()
    return count
""")
    opened = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
    )
    opened._extraction_contract = _file_receipt("builtins.open")

    lower_python_shell_file_contexts(tree)

    assert not any(isinstance(node, ast.With) for node in ast.walk(tree))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert [node.func.id for node in calls] == [
        "__turing_shell_file_open",
        "__turing_shell_file_write",
        "__turing_shell_file_flush",
        "__turing_shell_file_close",
    ]
    assert [
        node._extraction_contract["parameters"]["operation"]
        for node in calls
    ] == ["open", "write", "flush", "close"]
    special_cases = [interpret_python_special_case(node) for node in calls]
    assert [special.type for special in special_cases] == ["Call"] * 4
    assert all(special.attributes["shell_boundary"] for special in special_cases)
    assert all(
        special.attributes["deployment_owner"] == "shell_io.file_broker"
        for special in special_cases
    )
    assert tree._turing_shell_file_contexts[0]["cleanup_policy"] == (
        "ordered-scope-exit"
    )
    assert tree._turing_shell_file_contexts[0]["operation_identities"] == (
        "turing.shell.files.open",
        "turing.shell.files.write",
        "turing.shell.files.flush",
        "turing.shell.files.close",
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.build_graph(tree)
    shell_nodes = [
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("shell_boundary")
    ]
    assert len(shell_nodes) == 4
    assert all(data["type"] == "Call" for _node_id, data in shell_nodes)
    assert all(
        all(str(role) != "func" for _parent, role in data.get("parents", ()))
        for _node_id, data in shell_nodes
    )
    assert not any(
        isinstance(data.get("expr_obj"), ast.Name)
        and data["expr_obj"].id.startswith("__turing_shell_file_")
        for _node_id, data in graph.G.nodes(data=True)
    )

    from src.compiler.glsl_deployment_strategy import (
        _is_dispatch_metadata_node,
    )

    assert all(
        _is_dispatch_metadata_node(graph, node_id)
        for node_id, _data in shell_nodes
    )


def test_file_context_with_escaped_stream_remains_explicitly_unlowered():
    tree = ast.parse("""
def save(path, payload):
    with open(path, "wb") as stream:
        pickle.dump(payload, stream)
""")
    opened = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
    )
    opened._extraction_contract = _file_receipt("builtins.open")

    lower_python_shell_file_contexts(tree)

    assert any(isinstance(node, ast.With) for node in ast.walk(tree))


def test_text_file_context_remains_for_a_future_encoding_aware_shell_operation():
    tree = ast.parse("""
def save(path, payload):
    with open(path, "w") as stream:
        stream.write(payload)
""")
    opened = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "open"
    )
    opened._extraction_contract = _file_receipt("builtins.open")

    lower_python_shell_file_contexts(tree)

    assert any(isinstance(node, ast.With) for node in ast.walk(tree))


def test_unresolved_context_is_a_total_compilation_failure():
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.compiler.exception_graph import MissingResourceControlTopology

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "def root(manager):\n"
            "    with manager() as value:\n"
            "        return value\n"
        ))
        with pytest.raises(
            MissingResourceControlTopology,
            match="native resource scope requires concorded entry, exit and cleanup edges",
        ):
            reduce_abstract_tensor_topology(graph)
    finally:
        end_identity_book(token)


def test_context_call_identity_follows_concorded_factory_result_and_dynamic_argument():
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.concordance_declarations import NATIVE_EXCEPTION_CONTROL_GAP, SOURCE_PYTHON_IDENTITY
    from src.compiler.exception_graph import MissingResourceControlTopology
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "class IdentityBook:\n"
            "    def borrowed(self, name):\n"
            "        return name\n"
            "def identity_book(module) -> IdentityBook:\n"
            "    return IdentityBook()\n"
            "def root(module, names):\n"
            "    book = identity_book(module)\n"
            "    for name in names:\n"
            "        with book.borrowed(name) as page:\n"
            "            return page\n"
        ))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("root").graph
        context = next(
            node for node in ast.walk(function.G.graph["function_body"][1])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "borrowed"
        )
        fact = book.page(SOURCE_PYTHON_IDENTITY).latest((
            function.G.graph["authored_function_identity"], id(context),
        ))
        assert fact[0] == "concorded_context_call"
        source = next(row[0] for row in book.page(NATIVE_EXCEPTION_CONTROL_GAP).rows()
                      if book.page(NATIVE_EXCEPTION_CONTROL_GAP).latest(row)[0] == "With")
        assert book.page(NATIVE_EXCEPTION_CONTROL_GAP).latest((source,))[1] == (
            "resource_enter", "resource_exit", "cleanup",
        )
    finally:
        end_identity_book(token)


def test_mapping_get_uses_declared_value_class_after_presence_guard():
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.concordance_declarations import SOURCE_PYTHON_IDENTITY
    from src.compiler.exception_graph import MissingResourceControlTopology
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "class IdentityBook:\n"
            "    def borrowed(self, name):\n"
            "        return name\n"
            "def root(books: dict[str, IdentityBook], name: str):\n"
            "    book = books.get(name)\n"
            "    if book is None:\n"
            "        return 0\n"
            "    with book.borrowed(name) as page:\n"
            "        return 1\n"
        ))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        function = graph.function_table.entry("root").graph
        context = next(
            node
            for statement in function.G.graph["function_body"]
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "borrowed"
        )
        fact = book.page(SOURCE_PYTHON_IDENTITY).latest((
            function.G.graph["authored_function_identity"], id(context),
        ))
        assert fact[0] == "concorded_context_call"
    finally:
        end_identity_book(token)


def test_resource_demand_finalization_records_all_refusals_and_is_order_independent():
    from types import SimpleNamespace
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.concordance_declarations import (
        NATIVE_EXCEPTION_CONTROL_GAP, SOURCE_RESOURCE_ADMISSION_DEMAND,
    )
    from src.compiler.exception_graph import (
        MissingResourceControlTopology, finalize_resource_admission_demands,
    )
    from src.compiler.identity_concordance import (
        CorrelationTable, begin_identity_book, end_identity_book,
    )

    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "def first(manager):\n"
            "    with manager() as value:\n"
            "        return value\n"
            "def second(manager):\n"
            "    with manager() as value:\n"
            "        return value\n"
        ))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        demand_rows = tuple(book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).rows())
        gap_rows = tuple(book.page(NATIVE_EXCEPTION_CONTROL_GAP).rows())
        assert len(demand_rows) == len(gap_rows) == 2
        assert all(book.page(NATIVE_EXCEPTION_CONTROL_GAP).latest(row)[1] == (
            "missing_callable_identity", "resource_enter", "resource_exit", "cleanup",
        ) for row in gap_rows)
        findings = CorrelationTable._control_activity_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        )
        assert not [finding for finding in findings
                    if finding.kind == "resource-admission-demand"]
        original = tuple(book.latest_ref(NATIVE_EXCEPTION_CONTROL_GAP, row)
                         for row in gap_rows)
        table = graph.function_table
        table._entries = dict(reversed(tuple(table._entries.items())))
        with pytest.raises(MissingResourceControlTopology):
            finalize_resource_admission_demands(graph)
        assert original == tuple(book.latest_ref(NATIVE_EXCEPTION_CONTROL_GAP, row)
                                 for row in gap_rows)
    finally:
        end_identity_book(token)

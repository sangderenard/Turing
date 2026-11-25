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


def _pickle_receipt(identity: str) -> dict:
    return {
        "action": "use_native", "rule_id": "native-pickle-object-result", "identity": identity,
        "classification": "native_extension",
        "parameters": {"execution": "shell_io.external_references", "result_dtype": "opaque_ref"},
    }


def _pickle_scopes(source: str) -> ast.AST:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            node._extraction_contract = _file_receipt("io.open")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("dump", "load"):
            node._extraction_contract = _pickle_receipt(f"_pickle.{node.func.attr}")
    return lower_python_shell_file_contexts(tree)


def test_a_file_scope_that_only_pickle_uses_becomes_tensor_payload_calls():
    tree = _pickle_scopes("""
def save_map(file_path, map_data):
    with open(file_path, 'wb') as f:
        pickle.dump(map_data, f)

def load_map(file_path):
    with open(file_path, 'rb') as f:
        log('loaded')
        return to_tensor(pickle.load(f))
""")
    assert not any(isinstance(node, ast.With) for node in ast.walk(tree))
    calls = {
        node.func.id: node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("__turing_")
    }
    assert set(calls) == {"__turing_tensor_payload_save", "__turing_tensor_payload_load"}
    save, load = calls["__turing_tensor_payload_save"], calls["__turing_tensor_payload_load"]
    assert [ast.unparse(a) for a in save.args] == ["turing_payload_path_1", "map_data"]
    assert [ast.unparse(a) for a in load.args] == ["turing_payload_path_2"]
    assert save._extraction_contract["identity"] == "src.common.tensors.compression.tensor_payload.save"
    assert load._extraction_contract["parameters"]["result_dtype"] == "tensor_region"
    assert save._extraction_contract["parameters"]["argument_sequences"] == [
        {"position": 0, "encoding": "unicode-codepoints"}, {"position": 1, "encoding": "tensor-region"},
    ]
    unparsed = ast.unparse(tree)
    assert "turing_payload_path_2 = file_path" in unparsed and "log('loaded')" in unparsed
    assert "return to_tensor(__turing_tensor_payload_load(turing_payload_path_2))" in unparsed
    scopes = tree._turing_tensor_payload_scopes
    assert [s["mode"] for s in scopes] == ["wb", "rb"]
    assert scopes[0]["replaced"][0]["identity"] == "_pickle.dump"
    assert scopes[1]["replaced"][0]["becomes"] == "src.common.tensors.compression.tensor_payload.load"


@pytest.mark.parametrize("body", [
    "pickle.dump(m, f)\n        f.flush()",        # the stream used for something else too
    "pickle.dump(m, f, protocol=4)",              # a keyword changes the protocol
    "pickle.load(f)",                             # loading from a stream opened to write
])
def test_a_pickle_scope_with_any_other_stream_use_is_left_as_it_was(body):
    tree = _pickle_scopes(f"""
def save_map(p, m):
    with open(p, 'wb') as f:
        {body}
""")
    assert any(isinstance(node, ast.With) for node in ast.walk(tree))
    assert not hasattr(tree, "_turing_tensor_payload_scopes")


def _file_plan(source: str) -> dict:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
        ):
            node._extraction_contract = _file_receipt("builtins.open")
    lower_python_shell_file_contexts(tree)
    (plan,) = tree._turing_shell_file_contexts
    return plan


def test_file_plan_of_parameters_and_literals_is_pre_entry_eligible():
    # The acceptance program of case 6b7d429f (ticket b043a7c9).
    plan = _file_plan("""
def entry(p):
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + 1
""")

    assert plan["phase"] == "pre_entry"
    assert plan["phase_reason"] is None
    assert [operation["result"] for operation in plan["operations"]] == [
        "turing_file_handle_1", "n", None,
    ]


@pytest.mark.parametrize("helper", [
    # Neither store can run before the entry's first statement, so the
    # first-statement plan reads p as the caller's value either way
    # (review a04ac60a, comments 600 to 605).
    """
    def helper():
        p = 'shadow'
        return p
""",
    """
    def helper():
        nonlocal p
        p = 'shadow'
""",
])
def test_later_nested_store_to_a_parameter_keeps_a_first_statement_plan(
    helper,
):
    plan = _file_plan(
        "def entry(p):\n"
        "    with open(p, 'wb') as f:\n"
        "        n: int = f.write(b'ab')\n"
        + helper
        + "    return n + 1\n"
    )

    assert plan["phase"] == "pre_entry"
    assert plan["phase_reason"] is None


def test_plan_operands_resolve_in_operation_order():
    # `as p` names the stream p after open has read the parameter p: the
    # open's path operand is the entry parameter, and the write's handle
    # operand is the open's result (review a04ac60a, comments 604 and 606).
    plan = _file_plan("""
def entry(p):
    with open(p, 'wb') as p:
        n: int = p.write(b'ab')
    return n + 1
""")

    opened, written, _closed = plan["operations"]
    assert opened["arguments"][0] == {"kind": "name", "name": "p"}
    assert written["arguments"][0] == {
        "kind": "name", "name": opened["result"],
    }
    assert opened["result"] != "p"
    assert plan["phase"] == "pre_entry"


@pytest.mark.parametrize(("source", "reason"), [
    # Anything before the scope, effectful or not, keeps it post_entry: the
    # predicate is FunctionDef.body[0] (review a04ac60a, comments 596-597).
    ("""
def entry(p):
    print(p)
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + 1
""", "scope-not-first-statement"),
    ("""
def entry(p):
    k = len(p)
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + k
""", "scope-not-first-statement"),
    ("""
def entry(p, k):
    q = 1 // k
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + q
""", "scope-not-first-statement"),
    ("""
def entry(p):
    return 0
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + 1
""", "scope-not-first-statement"),
    ('''
def entry(p):
    """Write two bytes."""
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + 1
''', "scope-not-first-statement"),
    ("""
def entry(p):
    p = p + '.bin'
    with open(p, 'wb') as f:
        n: int = f.write(b'ab')
    return n + 1
""", "scope-not-first-statement"),
    ("""
DATA = b'ab'
def entry(p):
    with open(p, 'wb') as f:
        n: int = f.write(DATA)
    return n + 1
""", "operand-not-available-before-entry"),
    ("""
def entry(p, k):
    if k:
        with open(p, 'wb') as f:
            n: int = f.write(b'ab')
        return n + 1
    return 0
""", "scope-not-a-direct-function-statement"),
    ("""
def entry(p, k):
    with open(p, 'wb') as f:
        k = k + 1
        n: int = f.write(b'ab')
    return n + k
""", "scope-body-is-not-plan-operations"),
    ("""
with open('out.bin', 'wb') as f:
    f.write(b'ab')
""", "no-enclosing-function"),
])
def test_file_plan_that_cannot_run_before_the_entry_stays_post_entry(
    source, reason,
):
    plan = _file_plan(source)

    assert plan["phase"] == "post_entry"
    assert plan["phase_reason"] == reason


@pytest.mark.parametrize(("source", "phase", "reason"), [
    ("""
def entry(p):
    with open(p, 'wb') as f:
        f.write(b'ab')
    return 1
""", "pre_entry", None),
    ('''
def entry(p):
    """Write two bytes."""
    with open(p, 'wb') as f:
        f.write(b'ab')
    return 1
''', "post_entry", "scope-not-first-statement"),
])
def test_compiled_module_carries_the_plan_phase(source, phase, reason):
    # The phase must survive ingestion: read it from the plan the compiler
    # publishes, not only from the raw-AST lowering (review a04ac60a,
    # comments 596 to 599).
    import contextlib
    import io
    from pathlib import Path

    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        module, _, _ = lower_ast_source_to_ssa(
            source, "entry", name="file_plan_phase",
            extraction_contract=contract,
            progress=lambda _message: None, progress_bars=False,
        )

    (plan,) = module.metadata["shell_io"]["boundary_plans"]
    assert plan["function"] == "entry"
    assert plan["phase"] == phase
    assert plan["phase_reason"] == reason


_OWNER_HELPER = """
def helper(q):
    with open(q, 'wb') as g:
        g.write(b'zz')
    return 2
"""


@pytest.mark.parametrize(("entry", "owners"), [
    ("""
def entry(p):
    with open(p, 'wb') as f:
        f.write(b'ab')
    return 1
""", ["entry"]),
    ("""
def entry(p, q):
    with open(p, 'wb') as f:
        f.write(b'ab')
    return helper(q)
""", ["entry", "helper"]),
])
def test_published_plans_name_their_own_function_owner(entry, owners):
    # Lexical eligibility does not decide which plans run before the entry:
    # a helper whose first statement is a file scope is pre_entry too. Each
    # published plan must name its own function so the entry ABI can keep
    # only the entry's plans (review 301e340f, comments 613 to 615). An
    # uncalled helper's plan is not published at all.
    import contextlib
    import io
    from pathlib import Path

    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    )
    with contextlib.redirect_stdout(io.StringIO()):
        module, _, _ = lower_ast_source_to_ssa(
            _OWNER_HELPER + entry, "entry", name="file_plan_owner",
            extraction_contract=contract,
            progress=lambda _message: None, progress_bars=False,
        )

    plans = module.metadata["shell_io"]["boundary_plans"]
    assert sorted(plan["function"] for plan in plans) == owners
    assert all(plan["phase"] == "pre_entry" for plan in plans)

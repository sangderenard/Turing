import ast

from src.compiler.concordance_declarations import PARAMETER_ANNOTATION
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_lifted_method_annotation_retains_its_authored_qualified_identity():
    # The exact first failure of compiling all of identity_concordance.py:
    # pursuit has lifted the method without renaming its Python AST spelling.
    tree = ast.parse(
        "def mint_scope(stage: Any):\n    return stage\n"
        "def mint_scope(stage: Stage | None):\n    return stage\n"
    )
    for node, name in zip(tree.body, ("mint_scope", "IdentityBook.mint_scope")):
        node._python_source_identity = ("nativebooks", name)
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(tree, filename="nativebooks", resolve_unresolved_parents=False)
        assert book.page(PARAMETER_ANNOTATION).latest(("nativebooks", "mint_scope", "stage")) == "Any"
        assert book.page(PARAMETER_ANNOTATION).latest(("nativebooks", "IdentityBook.mint_scope", "stage")) == "Stage | None"
        assert graph.G.graph["function_parameter_annotations"] == {
            ("nativebooks", "mint_scope"): {"stage": "Any"},
            ("nativebooks", "IdentityBook.mint_scope"): {"stage": "Stage | None"},
        }
    finally:
        end_identity_book(token)


def test_annotation_accessor_requires_exact_module_and_definition_identity():
    from types import SimpleNamespace
    from src.compiler.authored_parameter_abi import _current_authored_parameter_annotations
    catalogue = {
        ("first", "Book.read"): {"key": "int"},
        ("second", "Book.read"): {"key": "str"},
        ("first", "Other.read"): {"key": "float"},
    }
    graph = SimpleNamespace(graph={"function_name": "read", "method_owner": "Book",
                                   "function_parameter_annotations": catalogue})
    assert _current_authored_parameter_annotations(graph) == {}
    for identity, expected in catalogue.items():
        graph.graph["authored_function_identity"] = identity
        assert _current_authored_parameter_annotations(graph) == expected


def test_same_method_names_across_classes_and_modules_stay_distinct():
    identities = (("one", "First.read"), ("one", "Second.read"),
                  ("two", "First.read"))
    tree = ast.parse("\n".join(
        f"def read(value: {kind}):\n    return value\n"
        for kind in ("int", "float", "str")))
    for node, identity in zip(tree.body, identities):
        node._python_source_identity = identity
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(tree, filename="caller", resolve_unresolved_parents=False)
        annotations = graph.G.graph["function_parameter_annotations"]
        assert set(annotations) == set(identities)
        for identity, spelling in zip(identities, ("int", "float", "str")):
            assert annotations[identity] == {"value": spelling}
            assert book.page(PARAMETER_ANNOTATION).latest((*identity, "value")) == spelling
    finally:
        end_identity_book(token)


def test_source_graph_lookup_never_uses_dotted_suffix():
    from types import SimpleNamespace
    from src.compiler.fortran_c_shell import _source_graph_for_lowered_function
    wrong = object()
    function = SimpleNamespace(metadata={"source_qualified_name": "module.Book.read"})
    assert _source_graph_for_lowered_function({"read": wrong}, "compiled_read", function) is None


def test_source_graph_index_preserves_modules_and_rejects_conflicting_identity():
    import pytest
    from types import SimpleNamespace
    from src.compiler.fortran_c_shell import (
        _resolved_source_graphs_by_symbol, _source_graph_for_lowered_function,
    )
    entries = []
    for module in ("one", "two"):
        graph = SimpleNamespace(graph={
            "function_name": "read", "authored_function_identity": (module, "Book.read"),
        })
        entries.append(SimpleNamespace(name="read", qualified_name=f"{module}.Book.read",
                                       graph=SimpleNamespace(G=graph)))
    indexed = _resolved_source_graphs_by_symbol(SimpleNamespace(function_table=entries))
    assert "read" not in indexed
    assert "Book.read" not in indexed
    for entry in entries:
        identity = entry.graph.G.graph["authored_function_identity"]
        function = SimpleNamespace(metadata={"authored_function_identity": identity})
        assert _source_graph_for_lowered_function(indexed, "compiled", function) is entry.graph.G
    duplicate = SimpleNamespace(name="read", qualified_name="one.Book.read",
                                graph=SimpleNamespace(G=SimpleNamespace(graph={
                                    "authored_function_identity": ("one", "Book.read")})))
    with pytest.raises(ValueError, match="conflicting source graphs"):
        _resolved_source_graphs_by_symbol(SimpleNamespace(function_table=(*entries, duplicate)))


def test_numeric_scopes_use_authored_identity_and_mint_unnamed_graphs():
    from types import SimpleNamespace
    from src.common.tensors.topological_reducer import _source_numeric_scope
    book, token = begin_identity_book()
    try:
        first = SimpleNamespace(graph={"function_name": "read", "method_owner": "Book",
                                       "authored_function_identity": ("one", "Book.read")})
        second = SimpleNamespace(graph={**first.graph,
                                        "authored_function_identity": ("two", "Book.read")})
        assert _source_numeric_scope(first) == ("one", "Book.read")
        assert _source_numeric_scope(second) == ("two", "Book.read")
        unnamed = [SimpleNamespace(graph={"function_name": "read", "method_owner": "Book"})
                   for _ in range(2)]
        scopes = [_source_numeric_scope(graph) for graph in unnamed]
        assert scopes[0] != scopes[1]
        assert _source_numeric_scope(unnamed[0], "different display name") == scopes[0]
    finally:
        end_identity_book(token)


def test_lambdas_retain_their_full_authored_source_paths():
    from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
    from src.compiler.fortran_c_shell import _resolved_source_graphs_by_symbol
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(
            "def owner(x):\n    first = lambda y: y + x\n    second = lambda y: y - x\n    return first(x) + second(x)\n"), filename="lambda_source")
        reduce_abstract_tensor_topology(graph)
        lambdas = [entry.graph.G for entry in graph.function_table
                   if entry.graph is not None and entry.name.startswith("<lambda:")]
        assert len(lambdas) == 2
        identities = [item.graph["authored_function_identity"] for item in lambdas]
        assert identities[0] != identities[1]
        assert all(len(identity) == 3 and identity[2] for identity in identities)
        indexed = _resolved_source_graphs_by_symbol(graph)
        for identity, item in zip(identities, lambdas):
            assert indexed[("source-function-instance", identity, item.graph["function_ref"])] is item
    finally:
        end_identity_book(token)


def test_imported_views_of_one_definition_keep_exact_function_instances():
    from types import SimpleNamespace
    from src.compiler.fortran_c_shell import (
        _resolved_source_graphs_by_symbol, _source_graph_for_lowered_function,
    )
    identity = ("books", "Registry.__init__")
    entries = [SimpleNamespace(
        name="__init__", qualified_name=spelling,
        graph=SimpleNamespace(G=SimpleNamespace(graph={
            "authored_function_identity": identity, "function_ref": reference})))
        for spelling, reference in (("Registry.__init__", 69), ("books.Registry.__init__", 235))]
    indexed = _resolved_source_graphs_by_symbol(SimpleNamespace(function_table=entries))
    assert identity not in indexed
    for entry in entries:
        metadata = {"authored_function_identity": identity,
                    "source_function_reference": entry.graph.G.graph["function_ref"]}
        assert _source_graph_for_lowered_function(indexed, "compiled", SimpleNamespace(metadata=metadata)) is entry.graph.G
    assert _source_graph_for_lowered_function(indexed, "compiled", SimpleNamespace(
        metadata={"authored_function_identity": identity})) is None

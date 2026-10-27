"""A selected call result shape retains its exact return and call causes."""
from types import SimpleNamespace
import networkx as nx
import pytest
from src.compiler.glsl_deployment_strategy import _callsite_return_shape_sources
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book, Novel, Derived, Mode,
    record_shape_transformation, record_proven_shape, PROVEN_SHAPE_PAGE, SHAPE_STATE_PAGE,
)
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE, FUNCTION_ADDRESS, SOURCE_FUNCTION_REACHABILITY, INGESTION, INGEST_SOURCE,
)


@pytest.mark.parametrize("poison", [None, "callee", "return", "shape", "binding"])
def test_selected_return_shape_proves_exact_call_and_shape(poison):
    book, token = begin_identity_book()
    try:
        def graph(scope, lexical):
            g = nx.DiGraph()
            g.graph.update(authored_function_identity=scope, lexical_read_scope=lexical,
                           canonical_value_ids=True, shape_scope=(scope, 0))
            return SimpleNamespace(G=g)
        caller = graph(("fixture", "caller"), ("caller_reads", 0))
        original = graph(("fixture", "callee"), ("callee_reads", 0))
        selected = graph(("fixture", "callee"), ("selected_reads", 0))
        caller.G.add_node(4, type="Call", value_id=4, attributes={})
        selected.G.add_node(3, type="Mul", value_id=3, attributes={})
        selected.G.graph.update(function_outputs=("result",), identity_table={"result": (3,)})
        call = book.post(CANONICAL_VALUE, (("caller_reads", 0), 4), ("call",),
                         stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        result = book.post(CANONICAL_VALUE, (("selected_reads", 0), 3), ("return",),
                           stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        address = book.post(FUNCTION_ADDRESS, ("fixture.callee",), 7,
                            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        if poison != "binding":
            book.post(SOURCE_FUNCTION_REACHABILITY,
                      (("fixture", "caller"), 4, ("fixture", "callee")), True,
                      stage=INGESTION, provenance=Derived((call, address)), mode=Mode.CONCORD)
        geometry = ((15,), "float64", 1, "static", None)
        shape_scope = selected.G.graph["shape_scope"]
        record_shape_transformation(shape_scope, 3, shape_scope, 3,
                                    stage="graph_tensor_descriptor", operation="Mul", role="lhs",
                                    source_state=geometry, target_state=geometry, source_cells=(result,))
        descriptor = {"shape": (15,), "dtype": "float64", "rank": 1}
        if poison == "callee":
            selected.G.graph["authored_function_identity"] = ("fixture", "foreign")
        if poison == "shape":
            descriptor["shape"] = (16,)
        if poison is not None:
            with pytest.raises(ValueError):
                _callsite_return_shape_sources(caller, original, selected, 4,
                                               8 if poison == "return" else 3, descriptor)
        else:
            sources = _callsite_return_shape_sources(caller, original, selected, 4, 3, descriptor)
            record_proven_shape(caller.G.graph["shape_scope"], 4, (15,), "float64", sources=sources)
            ref = book.latest_ref(PROVEN_SHAPE_PAGE, (caller.G.graph["shape_scope"], 4))
            assert {parent for parent, _ in book.edges_into(ref)} == set(sources)
            assert book.latest_ref(SHAPE_STATE_PAGE, (shape_scope, 3)) in sources
    finally:
        end_identity_book(token)

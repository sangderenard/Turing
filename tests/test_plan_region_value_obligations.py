"""Region dtype obligations: posted unresolved, settled citing the producer."""

import warnings
from pathlib import Path

import pytest

import src.compiler.glsl_deployment_strategy as strategy
from src.compiler.concordance_declarations import (
    CANONICAL_VALUE,
    INGEST_SOURCE,
    INGESTION,
    PLAN_REGION_VALUE_DTYPE,
    SOURCE_VALUE_CLASS,
)
from src.compiler.identity_concordance import (
    Mode,
    Novel,
    Unresolved,
    begin_identity_book,
    current_identity_book,
    end_identity_book,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph

ROOT = Path(__file__).resolve().parents[1]


def _graph_with_identity(value_id):
    graph = ProcessGraph(materialize_memory=False)
    scope = ("callers", 0)
    graph.G.graph.update({
        "function_name": "reader",
        "lexical_read_scope": scope,
        "canonical_value_ids": True,
        "authored_function_identity": ("module", "reader"),
    })
    graph.G.add_node(
        value_id, type="Indexed", op="Indexed", value_id=value_id,
        attributes={}, parents=[], children=[],
    )
    current_identity_book().post(
        CANONICAL_VALUE, (scope, value_id), ("Indexed", "Indexed", "x"),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
        mode=Mode.CONCORD,
    )
    return graph


def test_obligation_is_posted_unresolved_then_settled_with_producer():
    _book, token = begin_identity_book()
    try:
        graph = _graph_with_identity(5)
        strategy._post_region_value_obligation(graph, 5)
        book = current_identity_book()
        row = strategy._region_value_dtype_row(graph, 5, mint=False)
        assert isinstance(
            book.page(PLAN_REGION_VALUE_DTYPE).latest(row), Unresolved
        )
        assert strategy.outstanding_region_value_obligations(graph) == (5,)
        obligation = book.latest_ref(PLAN_REGION_VALUE_DTYPE, row)

        # The producer: the value's source class proved int.
        book.post(
            SOURCE_VALUE_CLASS, (("module", "reader"), 5),
            ("builtins.int", 1, "test producer"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        producer = book.latest_ref(SOURCE_VALUE_CLASS, (("module", "reader"), 5))
        cells = strategy._dtype_producer_cells(graph, 5)
        assert producer in cells

        strategy._settle_region_value_obligation(graph, 5, "int64", cells)
        assert book.page(PLAN_REGION_VALUE_DTYPE).latest(row) == "int64"
        assert strategy.outstanding_region_value_obligations(graph) == ()
        settled = book.latest_ref(PLAN_REGION_VALUE_DTYPE, row)
        sources = {ref for ref, _stage in book.edges_into(settled)}
        assert obligation in sources
        assert producer in sources
    finally:
        end_identity_book(token)


def test_every_typed_return_settles_its_obligation(monkeypatch):
    """Through the single exit, each obligation a non-terminal freeze posted
    is closed by the terminal freeze, citing at least one producer cell."""

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    posted, settled = set(), {}
    original_post = strategy._post_region_value_obligation
    original_settle = strategy._settle_region_value_obligation

    def post(graph, value_id):
        posted.add((id(graph.G), int(value_id)))
        return original_post(graph, value_id)

    def settle(graph, value_id, dtype, producer_cells=()):
        settled[(id(graph.G), int(value_id))] = tuple(producer_cells)
        return original_settle(graph, value_id, dtype, producer_cells)

    monkeypatch.setattr(strategy, "_post_region_value_obligation", post)
    monkeypatch.setattr(strategy, "_settle_region_value_obligation", settle)
    source = (ROOT / "tests/native_semantics_corpus"
              / "c55_list_filled_by_callee.py").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lower_ast_source_to_ssa(
            source, "root", name="obligation_probe",
            extraction_contract=str(
                ROOT / "extraction_contracts/program_extraction.yaml"
            ),
            progress=lambda _: None,
        )
    assert posted, "the probe must exercise a non-terminal obligation"
    assert posted <= set(settled)
    assert all(settled[key] for key in posted)


def _owed_host_scalar_graph(value_id):
    graph = _graph_with_identity(value_id)
    graph.G.graph["source_numeric_scope"] = ("module", "reader")
    graph.G.nodes[value_id]["attributes"] = {"binding_name": "acc"}
    assert strategy._owe_host_scalar_class(graph, value_id)
    assert strategy.outstanding_region_value_obligations(graph) == (value_id,)
    return graph


def test_terminal_freeze_settles_owed_host_scalar_from_class_or_planned_dtype():
    _book, token = begin_identity_book()
    try:
        by_class = _owed_host_scalar_graph(5)
        current_identity_book().post(
            SOURCE_VALUE_CLASS, (("module", "reader"), 5),
            ("builtins.float", 1, "test producer"),
            stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()),
            mode=Mode.CONCORD,
        )
        strategy._refuse_unsettled_host_scalars(by_class, {})
        row = strategy._region_value_dtype_row(by_class, 5, mint=False)
        assert current_identity_book().page(
            PLAN_REGION_VALUE_DTYPE).latest(row) == "float64"
        assert strategy.outstanding_region_value_obligations(by_class) == ()

        by_plan = _owed_host_scalar_graph(6)
        strategy._refuse_unsettled_host_scalars(by_plan, {6: "int64"})
        row = strategy._region_value_dtype_row(by_plan, 6, mint=False)
        assert current_identity_book().page(
            PLAN_REGION_VALUE_DTYPE).latest(row) == "int64"
    finally:
        end_identity_book(token)


def test_terminal_freeze_refuses_unsettled_host_scalar_by_name():
    _book, token = begin_identity_book()
    try:
        graph = _owed_host_scalar_graph(7)
        with pytest.raises(ValueError) as refusal:
            strategy._refuse_unsettled_host_scalars(graph, {})
        message = str(refusal.value)
        assert "'reader'" in message and "Indexed" in message
        assert "'acc'" in message and "value 7" in message
    finally:
        end_identity_book(token)

"""Core contract tests; neither replaces the bootstrap or managed tire gate."""
import pickle

import pytest

from src.compiler.concordance_declarations import (
    BOUNDARY_TRANSPORT, INGESTION, INGEST_SOURCE, SSA_VALUE, SSAValueFact,
    SSAValueOrigin, STRUCTURAL_VALUE, CALL_EDGE, OPTIONAL_VALUE_STRUCTURE,
)
from src.compiler.identity_concordance import (
    ConcordanceRefusal, CorrelationTable, Derived, Mode, Novel,
    begin_identity_book, end_identity_book,
)
from src.compiler.structural_concordance import ssa_structure, boundary_transport, optional_structure
from src.transmogrifier.ssa import IRModule


def test_live_transport_tracks_exact_revisions_and_retains_history():
    book, token = begin_identity_book()
    try:
        def publish(scope, dtype, previous=None):
            return book.post(
                SSA_VALUE, (scope, 295),
                SSAValueFact(dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                stage=INGESTION,
                provenance=(Novel(INGEST_SOURCE, ()) if previous is None else Derived((previous,))),
                mode=Mode.CONCORD if previous is None else Mode.REVISE,
            )

        produced = publish("coerce_metrics", "float64")
        consumed = publish("region", "float64")
        _, source = ssa_structure("coerce_metrics", 295, book=book)
        _, target = ssa_structure("region", 295, book=book)
        binding = book.post(CALL_EDGE, ("region", 295, "coerce_metrics", 295), "argument",
                            stage=INGESTION, provenance=Derived((produced, consumed)), mode=Mode.CONCORD)
        transport = boundary_transport("owner", "hard_failure", source, target, (((), ()),),
                                       bindings=(binding,), book=book)
        historical = book.page(BOUNDARY_TRANSPORT).cell(transport.row, transport.column)
        publish("coerce_metrics", "bool", produced)
        findings = CorrelationTable._live_structure_findings(IRModule({}, metadata={"identity_book": book}))
        assert len(findings) == 2  # producer view and its transitive transport
        with pytest.raises(ConcordanceRefusal, match="physical transport disagreement"):
            book.current_ref(BOUNDARY_TRANSPORT, transport.row)
        publish("region", "bool", consumed)
        book.settle_derivations()
        current = book.current_ref(BOUNDARY_TRANSPORT, transport.row)
        assert current.column > transport.column
        assert not book.stale_derivations()
        assert book.page(BOUNDARY_TRANSPORT).cell(transport.row, transport.column) == historical
        restored = pickle.loads(pickle.dumps(book))
        assert restored.current_ref(BOUNDARY_TRANSPORT, transport.row) == current
    finally:
        end_identity_book(token)


def test_transport_refuses_partial_path_mapping():
    book, token = begin_identity_book()
    try:
        book.post(SSA_VALUE, ("physical", 30),
                  SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                  stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        _, cell = ssa_structure("physical", 30, book=book)
        with pytest.raises(ConcordanceRefusal, match="incomplete structural transport"):
            boundary_transport("owner", "return", cell, cell, (), book=book)
        assert book.latest_ref(STRUCTURAL_VALUE, cell.row) == cell
        assert not tuple(book.page(BOUNDARY_TRANSPORT).rows())
    finally:
        end_identity_book(token)


def test_projected_live_view_keeps_recipe_and_historical_cell():
    book, token = begin_identity_book()
    try:
        original = book.post(SSA_VALUE, ("owner", 30),
                             SSAValueFact("int64", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                             stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
        fact, view = ssa_structure("owner", 30, book=book)
        projected = pickle.loads(pickle.dumps(book.project_cells((view,))))
        projected.post(SSA_VALUE, ("owner", 30),
                       SSAValueFact("bool", (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                       stage=INGESTION, provenance=Derived((original,)), mode=Mode.REVISE)
        assert projected.stale_derivations()
        current = projected.current_ref(STRUCTURAL_VALUE, view.row)
        assert projected.page(STRUCTURAL_VALUE).cell(current.row, current.column).dtype == "bool"
        assert projected.page(STRUCTURAL_VALUE).cell(view.row, view.column) == fact
        assert not book.stale_derivations()
    finally:
        end_identity_book(token)


def test_optional_semantics_do_not_expand_one_physical_operand():
    book, token = begin_identity_book()
    try:
        cells = tuple(book.post(SSA_VALUE, ("owner", value),
                                SSAValueFact(dtype, (), SSAValueOrigin.ADOPTED_GRAPH_ID),
                                stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
                      for value, dtype in ((30, "int64"), (31, "bool")))
        book.post(OPTIONAL_VALUE_STRUCTURE, ("owner", 30), (30, 31),
                  stage=INGESTION, provenance=Derived(cells), mode=Mode.CONCORD)
        physical, _ = ssa_structure("owner", 30, book=book)
        semantic, _ = optional_structure("owner", 30, book=book)
        assert tuple(path for path, _ in physical.leaves()) == ((),)
        assert tuple((path, leaf.value_id) for path, leaf in semantic.leaves()) == (
            (("payload",), 30), (("presence",), 31),
        )
    finally:
        end_identity_book(token)


def _post_fact(book, scope, value_id, **fields):
    return book.post(
        SSA_VALUE, (scope, value_id),
        SSAValueFact(fields.pop("dtype", "bool"), (), SSAValueOrigin.ADOPTED_GRAPH_ID, **fields),
        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD,
    )


def _transport_pair(book, producer_fields, consumer_fields):
    produced = _post_fact(book, "producer", 7, **producer_fields)
    consumed = _post_fact(book, "consumer", 7, **consumer_fields)
    _, source = ssa_structure("producer", 7, book=book)
    _, target = ssa_structure("consumer", 7, book=book)
    binding = book.post(CALL_EDGE, ("consumer", 7, "producer", 7), "argument",
                        stage=INGESTION, provenance=Derived((produced, consumed)),
                        mode=Mode.CONCORD)
    return boundary_transport("owner", "physical", source, target, (((), ()),),
                              bindings=(binding,), book=book)


def test_transport_refuses_physical_dtype_difference():
    book, token = begin_identity_book()
    try:
        with pytest.raises(ConcordanceRefusal, match="physical transport disagreement"):
            _transport_pair(book, dict(physical_dtype="float64"), {})
    finally:
        end_identity_book(token)


def test_transport_passes_equal_physical_fields():
    book, token = begin_identity_book()
    try:
        fields = dict(physical_dtype="float64", storage="scalar", rank=0)
        transport = _transport_pair(book, fields, dict(fields))
        assert book.current_ref(BOUNDARY_TRANSPORT, transport.row) is not None
    finally:
        end_identity_book(token)


def test_transport_refuses_declared_storage_difference():
    book, token = begin_identity_book()
    try:
        with pytest.raises(ConcordanceRefusal, match="physical transport disagreement"):
            _transport_pair(book, dict(dtype="float64", storage="reference"),
                            dict(dtype="float64", storage="scalar"))
    finally:
        end_identity_book(token)


def test_revise_of_physical_dtype_invalidates_transport():
    book, token = begin_identity_book()
    try:
        transport = _transport_pair(book, dict(physical_dtype="float64"),
                                    dict(physical_dtype="float64"))
        previous = book.latest_ref(SSA_VALUE, ("producer", 7))
        book.post(SSA_VALUE, ("producer", 7),
                  SSAValueFact("bool", (), SSAValueOrigin.ADOPTED_GRAPH_ID,
                               physical_dtype="int32"),
                  stage=INGESTION, provenance=Derived((previous,)), mode=Mode.REVISE)
        with pytest.raises(ConcordanceRefusal, match="physical transport disagreement"):
            book.current_ref(BOUNDARY_TRANSPORT, transport.row)
    finally:
        end_identity_book(token)


def _audited(accounting, **fact_fields):
    from src.transmogrifier.ssa import BasicBlock, Function, SSAValue

    book, token = begin_identity_book()
    try:
        _post_fact(book, "audited", 5, **fact_fields)
        value = SSAValue(5, "bool", (), accounting=accounting)
        function = Function("audited", [value], {"entry": BasicBlock("entry", [])})
        module = IRModule({"audited": function}, metadata={"identity_book": book})
        return CorrelationTable._ssa_publication_findings(module)
    finally:
        end_identity_book(token)


def test_audit_reports_physical_unposted_not_disagreement():
    found = _audited({"physical_dtype": "float64"})
    assert [f.kind for f in found] == ["ssa-physical-unposted"]


def test_audit_reports_physical_disagreement_when_both_set():
    found = _audited({"physical_dtype": "float64"}, physical_dtype="int32")
    assert [f.kind for f in found] == ["ssa-publication-type-disagreement"]
    assert not _audited({"physical_dtype": "float64"}, physical_dtype="float64")

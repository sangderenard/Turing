"""Iterable domains are proved by exact subject/class cells, not local IDs."""
import hashlib
from types import SimpleNamespace

import pytest
from tests.test_specialized_parameter_class_projection import _input, _loop_target
from src.common.tensors.topological_reducer import specialize_python_precision_widths
from src.compiler.identity_concordance import (
    begin_identity_book, end_identity_book, CorrelationTable, Derived, Novel, Mode,
)
from src.compiler.concordance_declarations import (
    SOURCE_VALUE_CLASS, SOURCE_NUMERIC_SPECIALIZATION_PAGE,
    SOURCE_NUMERIC_SPECIALIZATION, CELL_SET, INGESTION, CANONICAL_VALUE, INGEST_SOURCE,
)


@pytest.mark.parametrize("domain", ["builtins.bool", ""])
@pytest.mark.parametrize("poison", [None, "domain", "target", "members", "address"])
def test_iterable_receipt_requires_exact_class_subject_pack(domain, poison):
    book, token = begin_identity_book()
    try:
        graph, _, _ = _input(book, "builtins.list", "builtins.list")
        target = _loop_target(book, graph)
        actual = book.post(SOURCE_VALUE_CLASS, (("prior", "call"), 1),
                           (domain, 1, "callsite iterable element class"),
                           stage=INGESTION, provenance=Derived((target,)), mode=Mode.CONCORD)
        graph.nodes[1]["attributes"]["callsite_iterable_element_class_cell"] = actual
        specialize_python_precision_widths(graph)
        module = SimpleNamespace(metadata={"identity_book": book})
        assert CorrelationTable._source_numeric_specialization_findings(module) == []
        if poison is None:
            return
        scope = graph.graph["source_numeric_scope"]
        row = (scope[0], scope[2])
        old = book.latest_ref(SOURCE_NUMERIC_SPECIALIZATION_PAGE, row)
        pack = next(cell for cell, _ in [(cell, None) for cell in book.mint_of(old)[1]] if cell.page == CELL_SET)
        parents = tuple(cell for cell, _ in book.edges_into(pack))
        receipt = list(book.page(SOURCE_NUMERIC_SPECIALIZATION_PAGE).latest(row)[1])
        receipt.append(("audit_probe", poison))
        for index, (key, value) in enumerate(receipt):
            if isinstance(key, tuple) and key[0] == "iterated_class":
                if poison == "domain":
                    receipt[index] = (key, ("builtins.float", 1))
                elif poison == "target":
                    foreign = book.post(CANONICAL_VALUE, (("foreign", 0), 1), ("foreign target",),
                                        stage=INGESTION, provenance=Novel(INGEST_SOURCE, ()), mode=Mode.CONCORD)
                    parents = (*parents, foreign)
                    receipt[index] = ((key[0], foreign.key), value)
        if poison == "address":
            parents = tuple(cell for cell in parents if cell.page.name != "function_address")
        receipt = tuple(receipt)
        digest = hashlib.sha256(repr(receipt).encode()).hexdigest()[:16]
        new_scope = (scope[0], "specialization", digest)
        pack_fact = tuple(cell.key for cell in parents)
        if poison == "members":
            pack_fact = ()
        bad_pack = book.post(CELL_SET, (new_scope, 1), pack_fact,
                             stage=INGESTION, provenance=Derived(parents), mode=Mode.CONCORD)
        book.post(SOURCE_NUMERIC_SPECIALIZATION_PAGE, (scope[0], digest), (new_scope, receipt),
                  stage=INGESTION, provenance=Novel(SOURCE_NUMERIC_SPECIALIZATION, (bad_pack,)), mode=Mode.CONCORD)
        findings = CorrelationTable._source_numeric_specialization_findings(module)
        assert len(findings) == 1
        assert findings[0].kind == "source-numeric-specialization-disagreement"
    finally:
        end_identity_book(token)

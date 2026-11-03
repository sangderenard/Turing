"""A specialized integer width reads its exact source range cell."""

from pathlib import Path

from src.compiler.concordance_declarations import SOURCE_INT_RANGE
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.source_evidence_resolver import (
    IntegerDomainEvidence, UnresolvedSourceDemand, require_integer_domain,
)
from src.compiler import source_evidence_resolver


SOURCE = 'def root(x:int)->int:\n    return x + 1\n'
CONTRACT = Path(__file__).resolve().parents[1] / 'extraction_contracts/program_extraction.yaml'


def test_specialized_integer_width_reads_exact_domain(monkeypatch):
    original = source_evidence_resolver.require_integer_domain
    observed = []

    def read(scope, value_id, *, book=None):
        answer = original(scope, value_id, book=book)
        if int(value_id) == 1:
            observed.append(answer)
        return answer

    monkeypatch.setattr(source_evidence_resolver, 'require_integer_domain', read)
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, 'root', name='integer_domain_probe',
        extraction_contract=CONTRACT, progress=lambda _message: None,
    )
    assert observed
    specialized = [answer for answer in observed
                   if isinstance(answer, IntegerDomainEvidence)
                   and isinstance(answer.scope, tuple)
                   and 'specialization' in answer.scope]
    assert specialized
    assert all((answer.lower, answer.upper) == (1, 1)
               and answer.subject_cell.page == SOURCE_INT_RANGE
               and any(cell.page == SOURCE_INT_RANGE
                       for cell in answer.proof_cells)
               for answer in specialized)
    assert not CorrelationTable.build(module).findings(module)


def test_missing_exact_integer_domain_is_named():
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, 'root', name='integer_domain_missing_probe',
        extraction_contract=CONTRACT, progress=lambda _message: None,
    )
    book = module.metadata['identity_book']
    scope = next(row[0] for row, _column in book.page(SOURCE_INT_RANGE).cells
                 if isinstance(row[0], tuple) and 'specialization' in row[0])
    missing = require_integer_domain(scope, 999999, book=book)
    assert isinstance(missing, UnresolvedSourceDemand)
    assert missing.missing == 'missing_source_int_range'
    assert missing.scope == scope
    subject = book.latest_ref(SOURCE_INT_RANGE, (scope, 1))
    assert subject is not None
    original_edges_into = book.edges_into
    try:
        book.edges_into = lambda cell: (() if cell == subject
                                        else original_edges_into(cell))
        unsourced = require_integer_domain(scope, 1, book=book)
    finally:
        book.edges_into = original_edges_into
    assert isinstance(unsourced, UnresolvedSourceDemand)
    assert unsourced.missing == 'missing_integer_domain_provenance'
    assert unsourced.subject_cell == subject

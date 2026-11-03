"""An explicit source call feed reads its posted SSA physical width."""

from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import ConcordanceRefusal
from src.compiler.source_evidence_resolver import (
    PhysicalLayoutEvidence, UnresolvedSourceDemand,
)
from src.compiler import source_evidence_resolver


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    'def helper(xs: list[float]) -> float:\n'
    '    return xs[0]\n'
    'def root(xs: list[float]) -> float:\n'
    '    return helper(xs)\n'
)


def _lower(name):
    contract = ExtractionContract(
        ROOT / 'extraction_contracts/program_extraction.yaml'
    ).with_execution_file(
        ROOT / 'extraction_contracts/vehicle_full_native_execution.yaml'
    )
    return lower_ast_source_to_ssa(
        SOURCE, 'root', name=name, extraction_contract=contract,
        progress=lambda _message: None,
    )


def test_explicit_call_feed_width_uses_abi_backed_ssa_evidence(monkeypatch):
    original = source_evidence_resolver.require_physical_layout
    source_observed = []
    formal_observed = []

    def read(scope, value_id, *, book=None):
        answer = original(scope, value_id, book=book)
        if str(scope).endswith('__root@control:0') and int(value_id) == 0:
            source_observed.append(answer)
        if str(scope).endswith('__helper@control:0') and int(value_id) == 0:
            formal_observed.append(answer)
        return answer

    monkeypatch.setattr(source_evidence_resolver, 'require_physical_layout', read)
    _module, _outputs, _exports = _lower('physical_call_width')
    assert source_observed and formal_observed
    assert all(isinstance(answer, PhysicalLayoutEvidence)
               and answer.physical_dtype == 'float64'
               and answer.storage == 'span'
               and answer.rank == 1
               for answer in (*source_observed, *formal_observed))


def test_explicit_call_feed_refuses_missing_physical_proof(monkeypatch):
    original = source_evidence_resolver.require_physical_layout

    def read(scope, value_id, *, book=None):
        if str(scope).endswith('__root@control:0') and int(value_id) == 0:
            return UnresolvedSourceDemand(
                scope, int(value_id), 'physical_layout',
                'missing_physical_dtype',
            )
        return original(scope, value_id, book=book)

    monkeypatch.setattr(source_evidence_resolver, 'require_physical_layout', read)
    with pytest.raises(
        ConcordanceRefusal,
        match='physical-call-input-unproven: .*missing_physical_dtype',
    ):
        _lower('physical_call_missing')


def test_explicit_formal_refuses_missing_target_physical_proof(monkeypatch):
    original = source_evidence_resolver.require_physical_layout

    def read(scope, value_id, *, book=None):
        if str(scope).endswith('__helper@control:0') and int(value_id) == 0:
            return UnresolvedSourceDemand(
                scope, int(value_id), 'physical_layout',
                'missing_physical_dtype',
            )
        return original(scope, value_id, book=book)

    monkeypatch.setattr(source_evidence_resolver, 'require_physical_layout', read)
    with pytest.raises(
        ConcordanceRefusal,
        match='physical-call-formal-unproven: .*missing_physical_dtype',
    ):
        _lower('physical_formal_missing')

"""Actual export selection and source identity, not symbol spelling, prove a root."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from tests.test_owned_record_return_source import SOURCE
from src.compiler.extraction_contract import ExtractionContract
from tests.owned_record_test_support import lower_before_owned_record_boundary as lower_ast_source_to_ssa
from src.compiler.source_owned_record_projection import (public_source_entry,
    public_source_entry_findings, validated_public_source_entry)


def test_public_source_root_requires_exact_authorities_and_no_internal_callers(monkeypatch):
    module, _, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    proof = public_source_entry(module, name)
    assert proof is not None
    cell, fact = proof
    assert public_source_entry_findings(module) == ()
    book = module.metadata['identity_book']
    page = book.page(cell.page)
    key = (cell.row, cell.column)
    for field in ('function_address_cell', 'root_demand_cell', 'source_definition_cell'):
        original = page.cells[key]
        page.cells[key] = replace(fact, **{field:fact.activation_cell})
        try:
            assert public_source_entry(module, name) is None
        finally:
            page.cells[key] = original
    activation = fact.activation_cell
    activation_page = book.page(activation.page)
    activation_key = (activation.row, activation.column)
    original = activation_page.cells[activation_key]
    activation_page.cells[activation_key] = False
    try:
        assert public_source_entry(module, name) is None
    finally:
        activation_page.cells[activation_key] = original
    # Same qualname in another module is not the selected definition span.
    from src.compiler.identity_concordance import Ref
    from src.compiler.concordance_declarations import SOURCE_SPAN
    original_edges = book.edges_into
    address_edges = original_edges(fact.function_address_cell)
    def wrong_definition(ref):
        edges = original_edges(ref)
        if ref != fact.function_address_cell:
            return edges
        return tuple((Ref(SOURCE_SPAN, ('different-module', source.row[1], ()), source.column)
                      if source.page == SOURCE_SPAN else source, stage)
                     for source, stage in edges)
    with monkeypatch.context() as patch:
        patch.setattr(book, 'edges_into', wrong_definition)
        assert public_source_entry(module, name) is None
        assert public_source_entry_findings(module)
    assert book.edges_into(fact.function_address_cell) == address_edges
    def absent_demand(ref):
        edges = original_edges(ref)
        return tuple((source, stage) for source, stage in edges
                     if ref != fact.activation_cell or source != fact.root_demand_cell)
    with monkeypatch.context() as patch:
        patch.setattr(book, 'edges_into', absent_demand)
        assert public_source_entry(module, name) is None
        assert public_source_entry_findings(module)
    # A recursive call is an internal caller even though the symbol is public.
    block = next(iter(module.functions[name].blocks.values()))
    op = SimpleNamespace(op='Call', attributes={'callee':name})
    block.instrs.append(op)
    try:
        assert public_source_entry(module, name) is None
        assert validated_public_source_entry(module, name) == proof
        assert public_source_entry_findings(module) == ()
    finally:
        block.instrs.pop()
    proxy = SimpleNamespace(functions=module.functions, metadata=module.metadata,
                            call_table={'caller':(SimpleNamespace(callee_symbol=name, resolution='resolved'),)})
    assert public_source_entry(proxy, name) is None
    assert public_source_entry(module, name) == proof
    # Lowered compiler helpers do not inherit the caller's public boundary.
    for other in module.functions:
        if other not in exports:
            assert public_source_entry(module, other) is None

from pathlib import Path
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.source_owned_record_returns import owned_record_return_sources, validate_owned_record_return_source
from src.compiler.source_owned_record_elements import owned_record_glyph_element, owned_record_primitive_element
from src.compiler.concordance_declarations import CANONICAL_VALUE, SEQUENCE_CONTRACT_CONCORDANCE

SOURCE = '''from dataclasses import dataclass
@dataclass
class Node:
    key: str
@dataclass
class Output:
    names: list[str]
def root(nodes: list[Node]) -> Output:
    out: list[str] = []
    for node in nodes:
        out.append(node.key)
    return Output(names=out)
'''


def lower_source():
    module, _, exports = lower_ast_source_to_ssa(SOURCE, 'root',
        extraction_contract=ExtractionContract(Path(__file__).parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    return module, exports[0]


def test_owned_return_uses_exact_current_declaration_not_mutation_history(monkeypatch):
    module, name = lower_source()
    source, = owned_record_return_sources(module, name)
    assert validate_owned_record_return_source(module, name, source)
    element = owned_record_glyph_element(module, name, source)
    assert element is not None and element.semantic_class == 'builtins.str'
    assert owned_record_primitive_element(module, name, source) is None
    table = module.sequence_tables[name]
    assert table.sequences[source.sequence_id].child_table_pool is not None
    book = table.book
    contract, = (c for c, _ in book.edges_into(source.descriptor_cell)
                 if c.page == SEQUENCE_CONTRACT_CONCORDANCE)
    edges = tuple(book.edges_into(contract))
    canonical, = (c for c, _ in edges if c.page == CANONICAL_VALUE)
    assert int(canonical.row[-1]) == source.sequence_id
    history = tuple(e for e in edges if e[0].page == SEQUENCE_CONTRACT_CONCORDANCE)
    assert history
    foreign = next(c for row in book.page(CANONICAL_VALUE).rows()
                   if row[0] == canonical.row[0] and row[-1] != canonical.row[-1]
                   for c in (book.latest_ref(CANONICAL_VALUE, row),))
    stage = edges[0][1]
    from src.compiler.concorded_literals import origin_cell
    foreign_same_id = next(c for row in book.page(CANONICAL_VALUE).rows()
                          if row[-1] == canonical.row[-1] and row[0] != canonical.row[0]
                          for c in (book.latest_ref(CANONICAL_VALUE, row),)
                          if origin_cell(book, c) != origin_cell(book, canonical))
    variants = (
        tuple(e for e in edges if e[0].page != CANONICAL_VALUE), # history alone is not authority
        tuple((foreign_same_id, s) if c == canonical else (c, s) for c, s in edges),
        (*edges, (foreign, stage)), # ambiguous current identity
        tuple((foreign, s) if c == canonical else (c, s) for c, s in edges),
    )
    original_edges = book.edges_into
    for altered in variants:
        with monkeypatch.context() as patch:
            patch.setattr(book, 'edges_into', lambda c, altered=altered:
                          altered if c == contract else original_edges(c))
            assert owned_record_return_sources(module, name) == ()
            assert not validate_owned_record_return_source(module, name, source)
            assert owned_record_glyph_element(module, name, source) is None
    from src.compiler.identity_concordance import Ref
    wrong_scope = Ref(contract.page, ('other_function', source.sequence_id), contract.column)
    descriptor_edges = tuple(book.edges_into(source.descriptor_cell))
    with monkeypatch.context() as patch:
        patch.setattr(book, 'edges_into', lambda c:
                      tuple((wrong_scope, s) if p == contract else (p, s)
                            for p, s in descriptor_edges)
                      if c == source.descriptor_cell else original_edges(c))
        assert owned_record_return_sources(module, name) == ()
        assert not validate_owned_record_return_source(module, name, source)
    page = book.page(SEQUENCE_CONTRACT_CONCORDANCE)
    original_cell = page.cell
    fact = original_cell(contract.row, contract.column)
    assert fact[:3] == ('duplicates', 1, True)
    for altered in (('unique', *fact[1:]),
                    (fact[0], 2, *fact[2:]),
                    (*fact[:2], False, fact[3])):
        with monkeypatch.context() as patch:
            patch.setattr(page, 'cell', lambda row, column, altered=altered:
                          altered if (row, column) == (contract.row, contract.column)
                          else original_cell(row, column))
            assert owned_record_return_sources(module, name) == ()
            assert not validate_owned_record_return_source(module, name, source)
    assert owned_record_return_sources(module, name) == (source,)
    # No public ownership or writer approval was introduced by semantic str.
    from src.compiler.ssa_storage_requirements import public_owned_sequence_returns
    assert public_owned_sequence_returns(module, name) == ()

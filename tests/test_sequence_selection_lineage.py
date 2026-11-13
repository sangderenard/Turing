from pathlib import Path
from tests.test_owned_record_return_source import SOURCE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.source_owned_record_returns import owned_record_return_sources
from src.compiler.source_sequence_selection import sequence_selection


def test_nested_selection_requires_exact_phi_geometry(monkeypatch):
    module, _, exports = lower_ast_source_to_ssa(
        SOURCE, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    name = exports[0]
    source, = owned_record_return_sources(module, name)
    tree = sequence_selection(module, name, source.sequence_id)
    assert tree is not None
    assert len(tree.arms) == 2
    assert sum(len(arm.arms) for arm in tree.arms) == 2
    assert tree.geometry_cell is not None
    function = module.functions[name]
    owner_phi = next(op for block in function.blocks.values() for op in block.instrs
                     if op.res is not None and int(op.res.id) == source.owner_id)
    original = owner_phi.attributes['incoming_blocks']
    owner_phi.attributes['incoming_blocks'] = tuple(reversed(original))
    try:
        assert sequence_selection(module, name, source.sequence_id) is None
    finally:
        owner_phi.attributes['incoming_blocks'] = original
    book = module.sequence_tables[name].book
    page = book.page(tree.geometry_cell.page)
    key = (tree.geometry_cell.row, tree.geometry_cell.column)
    fact = page.cells.pop(key)
    try:
        assert sequence_selection(module, name, source.sequence_id) is None
    finally:
        page.cells[key] = fact
    assert sequence_selection(module, name, source.sequence_id) == tree

    from src.compiler.concordance_declarations import SEQUENCE_CONTRACT_CONCORDANCE, CANONICAL_VALUE
    terminal = next(arm for arm in tree.arms if not arm.arms)
    contract, = [p for p, _ in book.edges_into(terminal.descriptor_cell)
                  if p.page == SEQUENCE_CONTRACT_CONCORDANCE]
    original_edges = type(book).edges_into
    wrong = next(p for p, _ in book.edges_into(
        next(p for p, _ in book.edges_into(tree.descriptor_cell)
             if p.page == SEQUENCE_CONTRACT_CONCORDANCE)) if p.page == CANONICAL_VALUE)
    def replace_direct(self, ref):
        edges = original_edges(self, ref)
        return tuple((wrong if p.page == CANONICAL_VALUE else p, stage)
                     for p, stage in edges) if ref == contract else edges
    with monkeypatch.context() as patch:
        patch.setattr(type(book), 'edges_into', replace_direct)
        assert sequence_selection(module, name, source.sequence_id) is None
    def ambiguous_direct(self, ref):
        edges = tuple(original_edges(self, ref))
        if ref == contract:
            edges += ((wrong, edges[0][1]),)
        return edges
    with monkeypatch.context() as patch:
        patch.setattr(type(book), 'edges_into', ambiguous_direct)
        assert sequence_selection(module, name, source.sequence_id) is None
    assert sequence_selection(module, name, source.sequence_id) == tree

    from src.compiler.source_sequence_selection import local_selection_lifetimes
    lifetime = local_selection_lifetimes(module, name, source.sequence_id)
    assert lifetime is not None
    assert len(lifetime.original_owner_ids) == 3
    assert len(lifetime.terminal_cells) == 3
    # A missing lifetime receipt is not inferred from address-Phi geometry.
    lc = lifetime.terminal_cells[0]
    lp = book.page(lc.page)
    lk = (lc.row, lc.column)
    lf = lp.cells.pop(lk)
    try:
        assert local_selection_lifetimes(module, name, source.sequence_id) is None
    finally:
        lp.cells[lk] = lf

"""Ordinary loop targets retain their authored control identity relation."""
from pathlib import Path
from src.compiler.concordance_declarations import CANONICAL_VALUE, CONTROL_PROGRAM, ControlProgramFact


def test_ordinary_iterable_binding_is_published_with_both_causes(monkeypatch):
    import src.compiler.control_source as control
    from src.compiler.identity_concordance import current_identity_book
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    original = control.post_control_program
    observed = []

    def post(graph, program, **kwargs):
        cell = original(graph, program, **kwargs)
        if cell is not None and program is not None and program.iterable_bindings:
            book = current_identity_book()
            fact = book.page(CONTROL_PROGRAM).cell(cell.row, cell.column)
            assert len(fact.iterable_bindings) == len(program.iterable_bindings)
            causes = {parent for parent, _ in book.edges_into(cell)}
            for (source, target, induction), authored in zip(fact.iterable_bindings, program.iterable_bindings):
                assert source.page == target.page == CANONICAL_VALUE
                assert (source.row[-1], target.row[-1], induction) == authored
                assert source in causes and target in causes
                assert source.row[0] == target.row[0] == cell.row[0]
            observed.append(fact.iterable_bindings)
        return cell

    monkeypatch.setattr(control, 'post_control_program', post)
    lower_ast_source_to_ssa('''def root(values: list[int]) -> int:
    total = 0
    for value in values:
        total += value
    return total
''', 'root', extraction_contract=Path(__file__).parents[1]/'extraction_contracts/program_extraction.yaml',
        progress=lambda _: None, progress_bars=False)
    assert observed


def test_legacy_control_program_seed_has_no_invented_iterable_relation():
    assert ControlProgramFact((), (), (), None, (), ()).iterable_bindings == ()

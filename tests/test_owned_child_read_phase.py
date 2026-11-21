"""Source-only read-phase gates; unsafe cases never emit or execute native code."""
from pathlib import Path
import pytest

from src.compiler.source_owned_child_read_phase import owned_child_read_phase_sources

PREFIX = '''from dataclasses import dataclass
@dataclass
class Node:
    key: str
'''
BODY = '''def root(nodes: list[Node]) -> int:
    out: list[str] = []
    for node in nodes:
        out.append(node.key)
    total = 0
    for word in out:
        total += len(word)
    return total
'''


class Captured(BaseException):
    pass


def capture(source, monkeypatch):
    from src.compiler.precompile_to_ssa import _ControlSSABuilder
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.concordance_declarations import CANONICAL_VALUE, CONTROL_BLOCK, ControlBlockKind
    from src.compiler.source_sequence_lifetime import _ast_kind
    from tests.owned_record_test_support import lower_before_owned_record_boundary
    captured = []

    def inspect(builder, loop, **kwargs):
        book, scope = builder._book(), builder.lexical_read_scope
        for sid, target_id, induction in builder.program.iterable_bindings:
            origin = book.latest_ref(CANONICAL_VALUE, (scope, int(sid)))
            if origin is None or _ast_kind(book, origin) != 'value:List':
                continue
            loops = [book.latest_ref(CONTROL_BLOCK, row)
                     for row in book.page(CONTROL_BLOCK).rows()
                     if row[0] == scope and row[1] == ControlBlockKind.LOOP
                     and book.page(CONTROL_BLOCK).cell(row, book.latest_ref(CONTROL_BLOCK,row).column).extra[0] == induction]
            assert len(loops) == 1
            captured.append((book, scope, int(sid), loops[0].row[2], int(target_id)))
        raise Captured()

    with monkeypatch.context() as patch:
        patch.setattr(_ControlSSABuilder, 'lower_loop', inspect)
        with pytest.raises(Captured):
            lower_before_owned_record_boundary(source, 'root',
                extraction_contract=ExtractionContract(Path(__file__).parents[1] / 'extraction_contracts/program_extraction.yaml'),
                progress=lambda _: None, progress_bars=False)
    assert captured, 'source census was not reached'
    return captured


def test_build_then_read_has_exact_closed_phase(monkeypatch):
    arguments, = capture(PREFIX + BODY, monkeypatch)
    assert owned_child_read_phase_sources(*arguments)


def test_comparison_only_consumer_uses_exact_shell_program(monkeypatch):
    body=BODY.replace('        total += len(word)',"        if word == 'a':\n            total += 1")
    arguments, = capture(PREFIX + body,monkeypatch)
    assert owned_child_read_phase_sources(*arguments)


@pytest.mark.parametrize('body', [
    BODY.replace('        total += len(word)', '        out.clear()\n        total += len(word)'),
    BODY.replace('        total += len(word)', '        for node in nodes:\n            out.append(node.key)\n        total += len(word)'),
    BODY.replace('    return total', '    out.clear()\n    for node in nodes:\n        out.append(node.key)\n    return total'),
    BODY.replace('    total = 0', '    alias = out\n    total = 0').replace('    return total', '    alias.clear()\n    return total'),
    BODY.replace('def root(nodes: list[Node]) -> int:', 'def root(nodes: list[Node], count: int) -> int:').replace(
        '    out: list[str] = []', '    for repeat in range(count):\n        out: list[str] = []').replace(
        '    for node in nodes:\n        out.append(node.key)\n    total = 0\n    for word in out:\n        total += len(word)',
        '        for node in nodes:\n            out.append(node.key)\n        total = 0\n        for word in out:\n            total += len(word)'),
])
def test_mutation_or_enclosing_backedge_refuses_without_native(body, monkeypatch):
    arguments, = capture(PREFIX + body, monkeypatch)
    assert owned_child_read_phase_sources(*arguments) is None


def test_unknown_pool_use_refuses_without_native(monkeypatch):
    from src.compiler.concordance_declarations import PROCESS_GRAPH_OPERANDS, CANONICAL_VALUE
    arguments, = capture(PREFIX + BODY, monkeypatch)
    assert owned_child_read_phase_sources(*arguments)
    book, scope, sid, _, _ = arguments
    resident = book.latest_ref(CANONICAL_VALUE, (scope, sid))
    # A real scalar computation in this source is not an admitted pool
    # operation. Make its actual source operands include the resident; the
    # placement/alias prerequisites remain untouched.
    candidates = []
    for row in book.page(PROCESS_GRAPH_OPERANDS).rows():
        if row[0] != scope:
            continue
        cell = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
        source = book.latest_ref(CANONICAL_VALUE, row)
        from src.compiler.source_sequence_lifetime import _ast_kind
        if _ast_kind(book, source) == 'value:AugAssign':
            candidates.append(cell)
    assert len(candidates) == 1
    cell = candidates[0]
    with monkeypatch.context() as patch:
        page = book.page(PROCESS_GRAPH_OPERANDS)
        patch.setitem(page.cells, (cell.row,cell.column), (*page.cell(cell.row,cell.column),(resident,'arg:0')))
        assert owned_child_read_phase_sources(*arguments) is None


@pytest.mark.parametrize('value,parameter', [('out','list[str]'),('word','str')])
def test_authored_helper_escape_refuses_without_native(value, parameter, monkeypatch):
    helper = f'def sink(values: {parameter}) -> int:\n    return len(values)\n'
    if value == 'out':
        body = BODY.replace('    return total','    total += sink(out)\n    return total')
    else:
        body = BODY.replace('        total += len(word)','        total += sink(word)')
    arguments, = capture(PREFIX + helper + body, monkeypatch)
    assert owned_child_read_phase_sources(*arguments) is None


@pytest.mark.parametrize('body',[
    BODY.replace('    total = 0','    def reset():\n        out.clear()\n    total = 0').replace('    return total','    reset()\n    return total'),
    BODY.replace('        total += len(word)','        def hold():\n            return word\n        total += len(word)').replace('    return total','    return hold'),
])
def test_exact_cross_scope_capture_and_missing_inventory_refuse(body,monkeypatch):
    from src.compiler.concordance_declarations import SOURCE_CLOSURE_ENVIRONMENT,SOURCE_CLOSURE_CAPTURE
    arguments, = capture(PREFIX+body,monkeypatch)
    assert owned_child_read_phase_sources(*arguments) is None
    book=arguments[0]
    original_latest=book.latest_ref
    for page in (SOURCE_CLOSURE_ENVIRONMENT,SOURCE_CLOSURE_CAPTURE):
        rows=book.page(page).rows()
        assert rows
        with monkeypatch.context() as patch:
            patch.setattr(book,'latest_ref',lambda p,row: None if p==page else original_latest(p,row))
            assert owned_child_read_phase_sources(*arguments) is None


@pytest.mark.parametrize('body',[
    BODY.replace('        total += len(word)','        def keep(value: str = word) -> str:\n            return value\n        total += len(word)').replace('    return total','    return keep'),
    BODY.replace('        total += len(word)','        class Keep:\n            value: str = word\n        total += len(word)').replace('    return total','    return Keep'),
])
def test_nested_definition_default_or_class_capture_refuses(body,monkeypatch):
    from src.compiler.concordance_declarations import SOURCE_CLOSURE_FREE_NAMES
    arguments, = capture(PREFIX+body,monkeypatch)
    assert owned_child_read_phase_sources(*arguments) is None
    book=arguments[0]
    # The independent source activity declaration remains present when its
    # optional closure-name inventory is hidden. No empty-inventory grant.
    original_latest=book.latest_ref
    with monkeypatch.context() as patch:
        patch.setattr(book.page(SOURCE_CLOSURE_FREE_NAMES),'rows',lambda: ())
        patch.setattr(book,'latest_ref',lambda page,row: None if page==SOURCE_CLOSURE_FREE_NAMES else original_latest(page,row))
        assert owned_child_read_phase_sources(*arguments) is None


def test_missing_operand_census_or_swapped_placement_refuses(monkeypatch):
    from dataclasses import replace
    from src.compiler.concordance_declarations import PROCESS_GRAPH_OPERANDS, CONTROL_BLOCK_PLACEMENT, CONTROL_PROGRAM
    arguments, = capture(PREFIX + BODY, monkeypatch)
    assert owned_child_read_phase_sources(*arguments)
    book, scope, *_ = arguments
    rows = [row for row in book.page(PROCESS_GRAPH_OPERANDS).rows() if row[0] == scope]
    original_latest = book.latest_ref
    for missing in rows:
        with monkeypatch.context() as patch:
            patch.setattr(book,'latest_ref',lambda page,row: None if page==PROCESS_GRAPH_OPERANDS and row==missing else original_latest(page,row))
            assert owned_child_read_phase_sources(*arguments) is None
    program = next(book.latest_ref(CONTROL_PROGRAM,row) for row in book.page(CONTROL_PROGRAM).rows() if row[0]==scope)
    roots = book.page(CONTROL_PROGRAM).cell(program.row,program.column).root_blocks
    cells = [book.latest_ref(CONTROL_BLOCK_PLACEMENT,(scope,root)) for root in roots]
    assert len(cells)==2
    with monkeypatch.context() as patch:
        page=book.page(CONTROL_BLOCK_PLACEMENT)
        for cell in cells:
            fact=page.cell(cell.row,cell.column)
            patch.setitem(page.cells,(cell.row,cell.column),replace(fact,ordinal=1-fact.ordinal))
        assert owned_child_read_phase_sources(*arguments) is None


def test_missing_or_tampered_activity_identity_refuses(monkeypatch):
    from src.compiler.concordance_declarations import SOURCE_ACTIVITY_SCOPE,SOURCE_ACTIVITY_NODE
    arguments, = capture(PREFIX+BODY,monkeypatch)
    proof=owned_child_read_phase_sources(*arguments)
    assert proof
    book,scope,*_=arguments
    original_latest=book.latest_ref
    for missing in [cell for cell in proof if cell.page in (SOURCE_ACTIVITY_SCOPE,SOURCE_ACTIVITY_NODE)]:
        with monkeypatch.context() as patch:
            patch.setattr(book,'latest_ref',lambda page,row: None if page==missing.page and row==missing.row else original_latest(page,row))
            assert owned_child_read_phase_sources(*arguments) is None
    scope_cell=book.latest_ref(SOURCE_ACTIVITY_SCOPE,(scope,))
    with monkeypatch.context() as patch:
        patch.setitem(book.page(SOURCE_ACTIVITY_SCOPE).cells,(scope_cell.row,scope_cell.column),('foreign','scope'))
        assert owned_child_read_phase_sources(*arguments) is None
    point=next(cell for cell in proof if cell.page==SOURCE_ACTIVITY_NODE)
    page=book.page(SOURCE_ACTIVITY_NODE);kind,owners=page.cell(point.row,point.column)
    with monkeypatch.context() as patch:
        patch.setitem(page.cells,(point.row,point.column),('ClassDef' if kind!='ClassDef' else 'Name',owners))
        assert owned_child_read_phase_sources(*arguments) is None

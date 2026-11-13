"""Semantic item receipts do not collapse glyph or nested spans to int64."""
import ast
from dataclasses import replace
from pathlib import Path
import pytest
from src.compiler.extraction_contract import ExtractionContract
from tests.owned_record_test_support import lower_before_owned_record_boundary as lower_ast_source_to_ssa
from src.compiler.source_owned_record_returns import owned_record_return_sources
from src.compiler.source_owned_record_elements import owned_record_primitive_element

from tests.test_owned_record_return_source import SOURCE as SELECTED_SOURCE
SOURCE = SELECTED_SOURCE.replace('list[int]', 'list[ITEM]').replace('out.append(i)', 'out.append(VALUE)').replace('out.append(-i)', 'out.append(VALUE)').replace('out.append(-7)', 'out.append(VALUE)')


def lower(source):
    module, _, exports = lower_ast_source_to_ssa(source, 'root',
        extraction_contract=ExtractionContract(Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _:None, progress_bars=False)
    proof, = owned_record_return_sources(module, exports[0])
    return module, exports[0], proof

@pytest.mark.parametrize(('item','value'), [('int','7'), ('float','1.5'), ('bool','True')])
def test_primitive_element_is_exact_authored_class(item, value):
    module, name, source = lower(SOURCE.replace('ITEM',item).replace('VALUE',value))
    from src.compiler.source_owned_record_elements import source_field_sequence_element_findings
    assert source_field_sequence_element_findings(module) == ()
    result = owned_record_primitive_element(module,name,source)
    assert result is not None
    assert result.semantic_class == 'builtins.'+item
    assert result.element_cell.row == ('builtins',item,'')
    book=module.sequence_tables[name].book
    page=book.page(result.binding_cell.page)
    key=(result.binding_cell.row,result.binding_cell.column)
    fact=page.cells[key]
    try:
        page.cells[key]=replace(fact,annotation_cell=fact.field_occurrence_cell)
        assert owned_record_primitive_element(module,name,source) is None
        assert source_field_sequence_element_findings(module)
        page.cells[key]=replace(fact,module_cell=fact.annotation_cell)
        assert owned_record_primitive_element(module,name,source) is None
        page.cells[key]=replace(fact,rebound_names=('list',))
        assert owned_record_primitive_element(module,name,source) is None
        page.cells[key]=replace(fact,element_cell=fact.container_cell)
        assert owned_record_primitive_element(module,name,source) is None
    finally:
        page.cells[key]=fact
    assert owned_record_primitive_element(module,name,source)==result

@pytest.mark.parametrize('prefix, annotation, expected', [
    ('', 'list[int]', True), ('', 'list[float]', True), ('', 'list[bool]', True),
    ('', 'list[str]', True), ('', 'list[list[int]]', False),
    ('class int: pass\n', 'list[int]', False),
    ('int = unknown()\n', 'list[int]', False),
    ('list = unknown()\n', 'list[int]', False),
    ('from elsewhere import int\n', 'list[int]', False),
    ('from elsewhere import *\n', 'list[int]', False),
    ('', 'list[Thing]', False),
    ('', 'list[int | None]', False),
    ('def other(int): pass\n', 'list[int]', False),
    ('', 'list[thing.int]', False), ('', 'thing.list[int]', False),
    ('from builtins import int as I\n', 'list[I]', False),
])
def test_ingestion_uses_existing_builtin_rebound_guard(prefix, annotation, expected):
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.compiler.concordance_declarations import SOURCE_FIELD_SEQUENCE_ELEMENT
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    book, token=begin_identity_book()
    try:
        tree=ast.parse(prefix+'class Result:\n    items: '+annotation+'\n')
        graph=ProcessGraph(materialize_memory=False)
        graph.build_from_ast(tree, filename='elementproof', resolve_unresolved_parents=False)
        rows=tuple(book.page(SOURCE_FIELD_SEQUENCE_ELEMENT).rows())
        assert bool(rows) is expected
    finally:
        end_identity_book(token)


def test_string_receipt_is_not_primitive_even_when_physical_lane_is_int64():
    module,name,source=lower(SOURCE.replace('ITEM','str').replace('VALUE',"'a'"))
    from src.compiler.concordance_declarations import SOURCE_FIELD_SEQUENCE_ELEMENT
    book=module.sequence_tables[name].book
    assert any(book.page(SOURCE_FIELD_SEQUENCE_ELEMENT).latest(row).element_cell.row == ('builtins','str','')
               for row in book.page(SOURCE_FIELD_SEQUENCE_ELEMENT).rows())
    assert owned_record_primitive_element(module,name,source) is None


def test_foreign_same_named_class_is_not_returned_field_authority():
    from src.compiler.identity_concordance import begin_identity_book, end_identity_book
    from src.compiler.source_owned_record_elements import validated_field_sequence_element
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    from src.compiler.concordance_declarations import SOURCE_FIELD_SEQUENCE_ELEMENT
    module,name,source=lower(SOURCE.replace('ITEM','int').replace('VALUE','7'))
    book=module.sequence_tables[name].book
    actual=owned_record_primitive_element(module,name,source)
    assert actual is not None
    _,token=begin_identity_book(book)
    try:
        graph=ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse('class Result:\n    items: list[int]\n'),
                             filename='foreign_module',resolve_unresolved_parents=False)
    finally:
        end_identity_book(token)
    foreign=[book.latest_ref(SOURCE_FIELD_SEQUENCE_ELEMENT,row)
             for row in book.page(SOURCE_FIELD_SEQUENCE_ELEMENT).rows()
             if row[0].row[0]=='foreign_module']
    assert len(foreign)==1 and validated_field_sequence_element(book,foreign[0]) is not None
    page=book.page(actual.binding_cell.page)
    key=(actual.binding_cell.row,actual.binding_cell.column)
    original=page.cells[key]
    try:
        page.cells[key]=None
        assert owned_record_primitive_element(module,name,source) is None
    finally:
        page.cells[key]=original
    assert owned_record_primitive_element(module,name,source)==actual

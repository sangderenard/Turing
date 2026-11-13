"""Authored sequence item identity, distinct from its native lane dtype."""
import ast
import builtins
from dataclasses import dataclass


def publish_field_sequence_elements(tree):
    """Admit only unshadowed, direct builtin annotations in submitted modules.

    The established whole-source rebound census deliberately over-refuses:
    imports, aliases and even unrelated local shadows require a richer lexical
    receipt, rather than an interpretation of the last component of a name.
    """
    from .ast_control_normalization import _rebound_names
    from .concordance_declarations import (
        CLASS_FIELD_DECLARATION, SOURCE_FIELD_SEQUENCE_ELEMENT,
        SourceFieldSequenceElement, INGESTION,
    )
    from .identity_concordance import current_identity_book, Derived, Mode
    from ..transmogrifier.graph.graph_express2 import (
        post_source_span, _resolve_ast_parent_reference,
    )
    from ..common.tensors.topological_reducer import _static_symbol_definition_cell
    if not isinstance(tree, ast.Module):
        return
    rebound = _rebound_names(tree)
    book = current_identity_book()
    module = post_source_span(tree, tree)
    if module is None:
        return
    for owner in tree.body:
        if not isinstance(owner, ast.ClassDef) or getattr(owner, '_python_source_identity', None):
            continue
        for statement in owner.body:
            if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
                continue
            annotation = statement.annotation
            if not (isinstance(annotation, ast.Subscript)
                    and isinstance(annotation.value, ast.Name)
                    and isinstance(annotation.slice, ast.Name)):
                continue
            names = (annotation.value.id, annotation.slice.id)
            if '*' in rebound or names[0] != 'list' or names[1] not in {'int', 'float', 'bool', 'str'} or any(n in rebound for n in names):
                continue
            bindings = {n: vars(builtins)[n] for n in names}
            inherited = getattr(owner, '_python_bindings', {}) or {}
            if any(n in inherited and inherited[n] is not bindings[n] for n in names):
                continue
            container = _resolve_ast_parent_reference(annotation.value, bindings)
            element = _resolve_ast_parent_reference(annotation.slice, bindings)
            if container is not list or not any(element is t for t in (int, float, bool, str)):
                continue
            owner_span = post_source_span(owner, owner)
            field = None if owner_span is None else book.latest_ref(
                CLASS_FIELD_DECLARATION, (*owner_span.row[:2], statement.target.id))
            if field is None:
                continue
            annotation_cell = post_source_span(owner, annotation)
            container_cell = _static_symbol_definition_cell(container)
            element_cell = _static_symbol_definition_cell(element)
            if None in (annotation_cell, container_cell, element_cell):
                continue
            occurrence = post_source_span(owner, statement)
            fact = SourceFieldSequenceElement(annotation_cell, module, occurrence, container_cell, element_cell, tuple(sorted(rebound)))
            book.post(SOURCE_FIELD_SEQUENCE_ELEMENT, (field,), fact,
                      stage=INGESTION, provenance=Derived((field, annotation_cell, module, occurrence, container_cell, element_cell)),
                      mode=Mode.CONCORD)


@dataclass(frozen=True)
class OwnedRecordPrimitiveElement:
    field_cell: object
    binding_cell: object
    container_cell: object
    element_cell: object
    semantic_class: str


def validated_field_sequence_element(book, binding):
    """Validate binding authority separately from primitive transfer eligibility."""
    from .concordance_declarations import (
        CLASS_FIELD_DECLARATION, SOURCE_FIELD_SEQUENCE_ELEMENT,
        SourceFieldSequenceElement, STATIC_SYMBOL_DEFINITION, StaticSymbolKind,
        SOURCE_SPAN,
    )
    from ..transmogrifier.graph.graph_express2 import _span_fact
    if binding.page != SOURCE_FIELD_SEQUENCE_ELEMENT or len(binding.row) != 1:
        return None
    field, = binding.row
    if field.page != CLASS_FIELD_DECLARATION or book.latest_ref(field.page, field.row) != field:
        return None
    fact = book.page(binding.page).cell(binding.row, binding.column)
    if not isinstance(fact, SourceFieldSequenceElement):
        return None
    if set(p for p, _ in book.edges_into(binding)) != {field, fact.annotation_cell, fact.module_cell, fact.field_occurrence_cell, fact.container_cell, fact.element_cell}:
        return None
    if fact.annotation_cell.page != SOURCE_SPAN or fact.module_cell.page != SOURCE_SPAN:
        return None
    if (fact.field_occurrence_cell.page != SOURCE_SPAN
            or fact.field_occurrence_cell.row[:2] != field.row[:2]
            or book.page(SOURCE_SPAN).cell(fact.field_occurrence_cell.row, fact.field_occurrence_cell.column).kind != 'AnnAssign'
            or fact.annotation_cell.row[:2] != field.row[:2]):
        return None
    if book.page(SOURCE_SPAN).cell(fact.module_cell.row, fact.module_cell.column).kind != 'Module':
        return None
    if (fact.field_occurrence_cell not in {p for p, _ in book.edges_into(field)}
            or fact.module_cell.row[0] != field.row[0]
            or tuple(sorted(set(fact.rebound_names))) != fact.rebound_names):
        return None
    declaration = book.page(field.page).cell(field.row, field.column)
    try:
        annotation = ast.parse(declaration.annotation, mode='eval').body
    except (SyntaxError, TypeError):
        return None
    if not (isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name)
            and annotation.value.id == 'list' and isinstance(annotation.slice, ast.Name)
            and annotation.slice.id in {'int', 'float', 'bool', 'str'}):
        return None
    span = book.page(SOURCE_SPAN).cell(fact.annotation_cell.row, fact.annotation_cell.column)
    if span.dump_sha256 != _span_fact(annotation).dump_sha256:
        return None
    if '*' in fact.rebound_names or any(name in fact.rebound_names for name in ('list', annotation.slice.id)):
        return None
    expected = ((fact.container_cell, 'list'), (fact.element_cell, annotation.slice.id))
    if any(cell.page != STATIC_SYMBOL_DEFINITION or cell.row != ('builtins', name, '')
           or book.page(cell.page).cell(cell.row, cell.column).kind != StaticSymbolKind.CLASS
           for cell, name in expected):
        return None
    return OwnedRecordPrimitiveElement(field, binding, fact.container_cell, fact.element_cell,
                                       'builtins.' + annotation.slice.id)


def owned_record_primitive_element(module, function_name, source):
    """An exact scalar-item contract, not a physical int64 interpretation."""
    from .source_owned_record_returns import validate_owned_record_return_source
    from .concordance_declarations import CLASS_FIELD_DECLARATION, SOURCE_FIELD_SEQUENCE_ELEMENT
    if not validate_owned_record_return_source(module, function_name, source):
        return None
    book = module.sequence_tables[function_name].book
    # Constructor materialization cites its precise field declarations directly.
    # Do not search the book by an unqualified class spelling.
    fields = tuple(cell for cell, _ in book.edges_into(source.record_cell)
                   if cell.page == CLASS_FIELD_DECLARATION
                   and cell.row[2] == source.field_name)
    if len(fields) != 1:
        return None
    field, = fields
    if book.latest_ref(field.page, field.row) != field:
        return None
    matches=[]
    for field in fields:
        binding=book.latest_ref(SOURCE_FIELD_SEQUENCE_ELEMENT,(field,))
        if binding is None:
            continue
        proof=validated_field_sequence_element(book,binding)
        if proof is not None and proof.semantic_class in {'builtins.int','builtins.float','builtins.bool'}:
            matches.append(proof)
    return matches[0] if len(matches)==1 else None


def source_field_sequence_element_findings(module):
    from .concordance_declarations import SOURCE_FIELD_SEQUENCE_ELEMENT
    from .identity_concordance import Finding
    book=dict(getattr(module,'metadata',{}) or {}).get('identity_book')
    if book is None:
        return ()
    findings=[]
    for row in book.page(SOURCE_FIELD_SEQUENCE_ELEMENT).rows():
        try:
            if validated_field_sequence_element(book,book.latest_ref(SOURCE_FIELD_SEQUENCE_ELEMENT,row)) is None:
                raise ValueError('field sequence annotation binding is unproven')
        except (ValueError,KeyError,AttributeError,TypeError) as error:
            findings.append(Finding('field-sequence-element-unproven',str(row),None,str(error)))
    return tuple(findings)

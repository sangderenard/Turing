"""Exact public source-entry evidence for owned record result projection.

An entry receipt is not an ownership transfer.  This module presently only
joins the existing source activation and materialization authorities.
"""
from .identity_concordance import Ref


def _entry_authorities(book, function_name):
    from .concordance_declarations import (
        LOWERED_SOURCE_FUNCTION, FUNCTION_ADDRESS, SOURCE_DEFINITION,
        SOURCE_PURSUIT_ACTIVATION, CONTRACT_DEMAND_PAGE, DemandKind, SOURCE_SPAN,
    )
    lowered = book.latest_ref(LOWERED_SOURCE_FUNCTION, (str(function_name),))
    if lowered is None:
        return None
    fact = book.page(lowered.page).cell(lowered.row, lowered.column)
    if not isinstance(fact, tuple) or len(fact) != 4 or fact[1]:
        return None
    address = fact[0]
    if not isinstance(address, Ref) or address.page != FUNCTION_ADDRESS:
        return None
    if address not in {source for source, _ in book.edges_into(lowered)}:
        return None
    if book.latest_ref(address.page, address.row) != address:
        return None
    spans = {source for source, _ in book.edges_into(address)
             if source.page == SOURCE_SPAN}
    matches = []
    for span in spans:
        if len(span.row) != 3 or span.row[2] != ():
            continue
        definition = book.latest_ref(SOURCE_DEFINITION, span.row[:2])
        activation = book.latest_ref(SOURCE_PURSUIT_ACTIVATION, span.row[:2])
        if definition is None or activation is None:
            continue
        if book.page(definition.page).cell(definition.row, definition.column) not in {'FunctionDef', 'AsyncFunctionDef'}:
            continue
        if book.page(activation.page).cell(activation.row, activation.column) is not True:
            continue
        if span not in {source for source, _ in book.edges_into(definition)}:
            continue
        sources = {source for source, _ in book.edges_into(activation)}
        if span not in sources:
            continue
        demands = [source for source in sources
                   if source.page == CONTRACT_DEMAND_PAGE
                   and source.row[0] == DemandKind.PURSUIT_ROOT]
        if len(demands) == 1:
            matches.append((lowered, address, definition, activation, demands[0]))
    return matches[0] if len(matches) == 1 else None


def publish_public_source_entry(function):
    """Called only by the existing non-private export-selection writer."""
    from .concordance_declarations import PUBLIC_SOURCE_ENTRY, PublicSourceEntry, FRAME_BINDING
    from .identity_concordance import current_identity_book, Derived, Mode
    book = current_identity_book()
    refs = _entry_authorities(book, function.name)
    if refs is None:
        return None
    return book.post(PUBLIC_SOURCE_ENTRY, (str(function.name),), PublicSourceEntry(*refs),
                     stage=FRAME_BINDING, provenance=Derived(refs), mode=Mode.CONCORD)


def validated_public_source_entry(module, function_name):
    """Validate export authority independently of ownership eligibility."""
    from .concordance_declarations import PUBLIC_SOURCE_ENTRY, PublicSourceEntry
    name = str(function_name)
    if name not in module.functions:
        return None
    book = module.metadata.get('identity_book')
    if book is None:
        return None
    cell = book.latest_ref(PUBLIC_SOURCE_ENTRY, (name,))
    if cell is None:
        return None
    fact = book.page(cell.page).cell(cell.row, cell.column)
    refs = _entry_authorities(book, name)
    if refs is None or not isinstance(fact, PublicSourceEntry) or fact != PublicSourceEntry(*refs):
        return None
    if not set(refs).issubset({source for source, _ in book.edges_into(cell)}):
        return None
    return cell, fact


def public_source_entry(module, function_name):
    """Read an authored public root with no internal or recursive callers."""
    validated = validated_public_source_entry(module, function_name)
    if validated is None:
        return None
    name = str(function_name)
    for function in module.functions.values():
        for block in function.blocks.values():
            for op in block.instrs:
                if op.op in {'Call', 'call'} and str(op.attributes.get('callee') or '') == name:
                    return None
    for records in (getattr(module, 'call_table', {}) or {}).values():
        for record in records:
            if str(record.callee_symbol) == name and record.resolution != 'decomposed':
                return None
    return validated


def public_source_entry_findings(module):
    """Audit authority rows without imposing projection's no-caller restriction."""
    from .concordance_declarations import PUBLIC_SOURCE_ENTRY
    from .identity_concordance import Finding
    book = dict(getattr(module, 'metadata', {}) or {}).get('identity_book')
    if book is None:
        return ()
    findings = []
    for row in book.page(PUBLIC_SOURCE_ENTRY).rows():
        try:
            if validated_public_source_entry(module, row[0]) is None:
                raise ValueError('public source entry does not match exact current source authorities')
        except (ValueError, KeyError, AttributeError, TypeError) as error:
            findings.append(Finding('public-source-entry-unproven', str(row[0]), None, str(error)))
    return tuple(findings)

"""Conservative source bounds for single-pass mapping comprehensions.

This proves the producer, not arbitrary later calls. Final output allocation
must additionally join the exact returned descriptor and reject unknown runtime
call escapes/mutations of its storage. Capacity is an upper bound, never the
result length. The control ledger owns
publication cardinality; graph attributes only nominate candidates to check.
"""
from dataclasses import fields

from .concordance_declarations import (
    CANONICAL_VALUE, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT,
    ControlBlockKind, NATIVE_OBJECT_SOURCE, PROCESS_GRAPH_OPERANDS, ROOT,
    SOURCE_COMPREHENSION_CAPACITY_BOUND, SourceComprehensionCapacityBound,
    SHELL_HANDOFF, LOOP_RESULT_PORT, Arm,
)


def _fact(book, cell):
    if cell is None:
        raise ValueError("missing source cell")
    if book.latest_ref(cell.page, cell.row) != cell:
        raise ValueError("superseded source cell")
    return book.page(cell.page).cell(cell.row, cell.column)


def _sources(fact):
    return tuple(cell for field in fields(fact)
                 for cell in (getattr(fact, field.name)
                              if field.name == "publication_path"
                              else (getattr(fact, field.name),)))



def _depends_on(book, value, destination):
    """Conservative identity/operand closure for possible storage aliases."""
    pending, seen = [value], set()
    while pending:
        current = pending.pop()
        if current == destination:
            return True
        if current is None or current in seen:
            continue
        seen.add(current)
        pending.extend(source for source, _stage in book.edges_into(current))
        mint = book.mint_of(current)
        if mint is not None:
            pending.extend(mint[1])
        if current.page == CANONICAL_VALUE:
            operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, current.row)
            if operands is not None:
                pending.extend(source for source, _role in _fact(book, operands))
    return False


def _publication_path(book, scope, publication_control, loop_control):
    """A separate publication may be guarded, never repeated per source row."""
    path, seen = [], set()
    current = publication_control
    while current != loop_control:
        if current in seen:
            raise ValueError("cyclic publication placement")
        seen.add(current)
        control = _fact(book, current)
        permitted = (ControlBlockKind.SEQUENCE_MUTATION if not path
                     else ControlBlockKind.CONDITIONAL)
        if control.kind != permitted:
            raise ValueError("publication path contains repeatable or unsupported control")
        placement = book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, current))
        placed = _fact(book, placement)
        if placed.parent == ROOT:
            raise ValueError("publication is outside its source loop")
        if placed.arm not in {Arm.BODY, Arm.ORELSE}:
            raise ValueError("publication is not in a single-execution branch")
        if placed.parent == loop_control and placed.arm != Arm.BODY:
            raise ValueError("publication is not in source loop body")
        path.extend((current, placement))
        current = placed.parent
    return tuple(path)


def _validate(book, fact):
    if not isinstance(fact, SourceComprehensionCapacityBound):
        raise ValueError("invalid comprehension bound fact")
    for cell in _sources(fact):
        _fact(book, cell)
    identities = (fact.destination_cell, fact.source_cell,
                  fact.loop_cell, fact.materializer_cell)
    if any(cell.page != CANONICAL_VALUE for cell in identities):
        raise ValueError("bound lacks canonical source identities")
    scope = fact.destination_cell.row[0]
    if any(cell.row[0] != scope for cell in identities):
        raise ValueError("cross-scope comprehension identities")
    if len(set(identities)) != 4:
        raise ValueError("aliased comprehension identities")
    if book.mint_of(fact.destination_cell) != (LOOP_RESULT_PORT, (fact.loop_cell,)):
        raise ValueError("destination lacks exact loop-result mint")
    destination = _fact(book, fact.destination_cell)
    if not isinstance(destination, tuple) or destination[0] != "LoopResult":
        raise ValueError("destination is not a loop result")
    if fact.loop_operands_cell.page != PROCESS_GRAPH_OPERANDS or fact.loop_operands_cell.row != (scope, fact.loop_cell.row[1]):
        raise ValueError("wrong loop operand receipt")
    operands = _fact(book, fact.loop_operands_cell)
    iterators = tuple(cell for cell, role in operands if role in {"iterable", "iter"})
    if iterators != (fact.source_cell,):
        raise ValueError("loop does not iterate this exact source")
    if fact.control_cell.page != CONTROL_BLOCK or fact.control_cell.row != (scope, ControlBlockKind.LOOP, fact.loop_cell):
        raise ValueError("wrong control loop receipt")
    control = _fact(book, fact.control_cell)
    if control.kind != ControlBlockKind.LOOP:
        raise ValueError("not a counted loop")
    if len(control.extra) != 6 or tuple(control.extra[1:5]) != ("0", f"__iterable_extent_{fact.source_cell.row[1]}__", "1", "lt"):
        raise ValueError("loop does not visit each source row at most once")
    if fact.placement_cell.page != CONTROL_BLOCK_PLACEMENT or fact.placement_cell.row != (scope, fact.control_cell):
        raise ValueError("wrong loop placement receipt")
    placement = _fact(book, fact.placement_cell)
    if placement.parent != ROOT:
        raise ValueError("nested or repeatable comprehension placement")
    mutations = []
    for row in book.page(CONTROL_BLOCK).rows():
        if row[0] != scope:
            continue
        cell = book.latest_ref(CONTROL_BLOCK, row)
        placed = book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, cell))
        if placed is None:
            continue
        for mutation in getattr(_fact(book, cell), "sequence_mutations", ()):
            if mutation and _depends_on(book, mutation[0], fact.destination_cell):
                mutations.append((cell, mutation))
    if len(mutations) != 1:
        raise ValueError("destination has missing or additional publications")
    publication_control = mutations[0][0]
    if _publication_path(book, scope, publication_control, fact.control_cell) != fact.publication_path:
        raise ValueError("publication placement proof disagrees")
    target, operation, arguments, effect, policy, argument_kind, publication = mutations[0][1]
    if (operation, policy, argument_kind) != ("update", "unique", "mapping_items") or len(arguments) != 2:
        raise ValueError("not a single key/value publication")
    if effect != fact.loop_cell or publication != fact.publication_cell:
        raise ValueError("publication belongs to a different loop")
    if any(cell is None or cell.page != CANONICAL_VALUE or cell.row[0] != scope for cell in arguments):
        raise ValueError("publication arguments lack exact identities")
    if publication.page != NATIVE_OBJECT_SOURCE or publication.row != (scope, fact.materializer_cell.row[1], "publication"):
        raise ValueError("wrong materializer publication receipt")
    data = dict(_fact(book, publication))
    if len(tuple(data.get("source_argument_value_ids", ()))) != 1 or tuple(data.get("projected_argument_value_ids", ())) != tuple(cell.row[1] for cell in arguments):
        raise ValueError("publication projection disagrees with its source")
    publication_sources = {cell for cell, _stage in book.edges_into(publication)}
    if not {fact.materializer_cell, *arguments}.issubset(publication_sources):
        raise ValueError("publication lost source provenance")
    control_sources = {cell for cell, _stage in book.edges_into(publication_control)}
    if not {target, effect, publication, *arguments}.issubset(control_sources):
        raise ValueError("control mutation lost source provenance")
    return fact


def publish_source_comprehension_capacity(graph, control, *, book=None):
    """Call after the FINAL control posting, before native SSA construction.

    Unsupported candidates are left unproved. Neither storage nor annotations
    are modified. The control argument makes the required ordering explicit.
    """
    from .identity_concordance import current_identity_book, Derived, Mode
    from .control_source import LoopBlock, _flatten_control_sequence, control_block_cell
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book() if book is None else book
    graph = getattr(graph, "G", graph)
    if graph is None:
        return ()
    scope = graph.graph.get("lexical_read_scope")
    if scope is None or control is None:
        return ()
    scope = tuple(scope)
    loops = tuple(block for block in _flatten_control_sequence(control.root)
                  if isinstance(block, LoopBlock))
    root_loops = {
        block.source_loop_node_id: block for block in loops
        if sum(other.source_loop_node_id == block.source_loop_node_id
               for other in loops) == 1
    }
    result = []
    for node_id, data in graph.nodes(data=True):
        attrs = data.get("attributes") or {}
        loop_id = attrs.get("loop_id")
        if data.get("type") != "LoopResult" or attrs.get("result_kind") != "collection" or attrs.get("aggregate_kind") != "dict" or loop_id not in root_loops:
            continue
        source_id = attrs.get("collection_iterable_value_id")
        materializer_id = attrs.get("materializer_node_id")
        if source_id is None or materializer_id is None:
            continue
        try:
            control_cell = control_block_cell(book, scope, root_loops[loop_id])
            fact = SourceComprehensionCapacityBound(
                *(node_identity_cell(graph, int(value), book=book) for value in (node_id, source_id, loop_id, materializer_id)),
                book.latest_ref(NATIVE_OBJECT_SOURCE, (scope, int(materializer_id), "publication")),
                book.latest_ref(PROCESS_GRAPH_OPERANDS, (scope, int(loop_id))),
                control_cell,
                book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, control_cell)),
            )
            from dataclasses import replace
            publications = []
            for row in book.page(CONTROL_BLOCK).rows():
                if row[0] != scope:
                    continue
                candidate = book.latest_ref(CONTROL_BLOCK, row)
                if book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, candidate)) is None:
                    continue
                for mutation in getattr(_fact(book, candidate), "sequence_mutations", ()):
                    if mutation and _depends_on(book, mutation[0], fact.destination_cell):
                        publications.append(candidate)
            if len(publications) != 1:
                continue
            fact = replace(fact, publication_path=_publication_path(
                book, scope, publications[0], control_cell))
            _validate(book, fact)
        except (ValueError, KeyError, TypeError):
            continue
        row = (scope, int(node_id))
        previous = book.latest_ref(SOURCE_COMPREHENSION_CAPACITY_BOUND, row)
        if previous is not None and _fact(book, previous) == fact:
            result.append(previous)
            continue
        result.append(book.post(SOURCE_COMPREHENSION_CAPACITY_BOUND,
            (scope, int(node_id)), fact, stage=SHELL_HANDOFF,
            provenance=Derived(_sources(fact)), mode=Mode.REVISE))
    return tuple(result)


def source_comprehension_capacity_findings(module):
    """Audit receipt contents, current revisions and exact direct causes."""
    from .identity_concordance import Finding
    book = (getattr(module, "metadata", {}) or {}).get("identity_book")
    if book is None or SOURCE_COMPREHENSION_CAPACITY_BOUND.name not in book.pages:
        return ()
    findings = []
    page = book.page(SOURCE_COMPREHENSION_CAPACITY_BOUND)
    for row in page.rows():
        cell = book.latest_ref(SOURCE_COMPREHENSION_CAPACITY_BOUND, row)
        try:
            fact = _validate(book, _fact(book, cell))
            if row != (fact.destination_cell.row[0], fact.destination_cell.row[1]):
                raise ValueError("bound row differs from destination identity")
            if {source for source, _stage in book.edges_into(cell)} != set(_sources(fact)):
                raise ValueError("bound lacks exact source provenance")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            findings.append(Finding("source-comprehension-capacity", str(row[0]), int(row[1]), str(error)))
    return tuple(findings)

"""SOURCE_CALL_FIELD_STATE_TRANSFER generalized: a field written once reaches its observers through receiver aliases.

The incumbent writer (topological_reducer, after ``post_call_activity_corridors``) carries one call's completed
keyed mapping write to a later call's observed formal field through a certified corridor.  This carries the plain
case the AbstractTensor YoungMan needs (owner, 2026-11-12): ``self.precision = precision`` in ``__init__``, then
``self._load_and_set_geometry(...)`` -> ... -> ``_load_map``, which reads ``dtype=self.precision``.

Rule.  A field of class C whose canonical REDUCER_FIELD_STATE holds exactly one WRITTEN version in the whole program
(one row, one revision, no other kind of write of that field on a C receiver) has that write's value on every
successful read: Python raises before a read of an unset attribute, and there is no other writer.  From the write's
receiver, every SOURCE_CALL_OBJECT_ALIAS that passes that receiver into a callee posts one transfer row, and the
callee's formal carries the chain on.  Where a reached formal has an OBSERVED state of the field, the written value's
literal -- a literal itself, or a formal whose every planner specialization agrees on one literal -- is posted as
SOURCE_LITERAL_VALUE at the observed read's canonical row, so ``concorded_literals.literal_from_identity`` (and the
deployment's ``dtype=`` projection) reads it.  Nothing is guessed: no proof, no row.
"""

from __future__ import annotations

from typing import Any

TAG = "single-program-write-receiver-chain"


def _canonical_row(row: Any) -> bool:
    scope = row[0] if isinstance(row, tuple) and row else None
    return isinstance(scope, tuple) and len(scope) == 2 and isinstance(scope[1], int)


def _written_literal(book: Any, value: Any) -> tuple[Any, tuple]:
    """The one literal the written value has everywhere, and the cells that prove it; (None, ()) when unproven."""

    from .concordance_declarations import (
        CANONICAL_VALUE, NAME_BINDING, PLANNER_SPECIALIZATION_PAGE, SpecializationSource,
    )
    from .concorded_literals import MISSING, literal_from_identity

    literal, cells = literal_from_identity(book, value)
    if literal is not MISSING:
        return literal, tuple(cells)
    if value.page != CANONICAL_VALUE:
        return None, ()
    read_scope, value_id = value.row[0], int(value.row[1])
    names = tuple(
        row[1] for row in book.page(NAME_BINDING).scope_rows(read_scope)
        if (fact := book.page(NAME_BINDING).latest(row)) is not None
        and getattr(fact, "authored", False) and int(fact.value_id) == value_id
    )
    if len(names) != 1:
        return None, ()
    forks = tuple(
        target for target, stage in book.edges_out_of(value)
        if target.page == CANONICAL_VALUE and getattr(stage, "name", None) == "read_scope_fork"
    )
    page = book.page(PLANNER_SPECIALIZATION_PAGE)
    facts, proof = set(), []
    for fork in forks:
        cell = book.latest_ref(PLANNER_SPECIALIZATION_PAGE, (tuple(fork.row[0]), str(names[0])))
        if cell is None:
            continue
        fact = page.cell(cell.row, cell.column)
        if getattr(fact, "source", None) is not SpecializationSource.LITERAL:
            return None, ()
        facts.add(repr(fact.value))
        proof.extend((fork, cell))
        literal = fact.value
    if len(facts) != 1:
        return None, ()
    return literal, tuple(proof)


def post_receiver_field_transfers(book: Any = None) -> int:
    """Post the generalized transfers and observed-read literals; the number of literals posted."""

    from .concordance_declarations import (
        CANONICAL_VALUE, CLASS_FIELD_DECLARATION, NAME_BINDING, REDUCER_FIELD_STATE, REDUCTION,
        SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_CALL_OBJECT_ALIAS,
        SOURCE_LITERAL_VALUE, SOURCE_VALUE_CLASS, FieldState, FieldStateKind,
    )
    from .identity_concordance import Derived, Mode, current_identity_book

    book = current_identity_book() if book is None else book
    states = book.page(REDUCER_FIELD_STATE)
    activity_scopes = book.page(SOURCE_ACTIVITY_SCOPE)
    value_classes = book.page(SOURCE_VALUE_CLASS)

    def authored_scope(read_scope):
        cell = book.latest_ref(SOURCE_ACTIVITY_SCOPE, (read_scope,))
        return None if cell is None else activity_scopes.cell(cell.row, cell.column)

    def receiver_class(read_scope, receiver):
        scope = authored_scope(read_scope)
        if scope is None:
            return None, None
        cell = book.latest_ref(SOURCE_VALUE_CLASS, (tuple(scope), int(receiver)))
        fact = None if cell is None else value_classes.cell(cell.row, cell.column)
        return (fact[0] if isinstance(fact, tuple) and fact else None), cell

    # 1. Every canonical state of every field, by (receiver class, field).
    writes: dict[tuple[str, str], list] = {}
    observed: dict[tuple, Any] = {}
    other_writes: set[tuple[str, str]] = set()
    for row in states.rows():
        if not _canonical_row(row):
            continue
        cell = book.latest_ref(REDUCER_FIELD_STATE, row)
        state = states.cell(cell.row, cell.column)
        if not isinstance(state, FieldState):
            continue
        field = str(row[2])
        if state.kind is FieldStateKind.OBSERVED:
            observed[(tuple(row[0]), int(row[1]), field)] = cell
            continue
        class_name, class_cell = receiver_class(tuple(row[0]), int(row[1]))
        if class_name is None:
            continue
        if state.kind is FieldStateKind.WRITTEN and cell.column == 0:
            writes.setdefault((str(class_name), field), []).append((row, cell, state, class_cell))
        else:
            other_writes.add((str(class_name), field))

    # 2. Aliases by the caller-side canonical value they pass.
    alias_page = book.page(SOURCE_CALL_OBJECT_ALIAS)
    names = book.page(NAME_BINDING)
    by_actual: dict[tuple, list] = {}
    for row in alias_page.rows():
        alias = book.latest_ref(SOURCE_CALL_OBJECT_ALIAS, row)
        fact = alias_page.cell(alias.row, alias.column)
        if not (isinstance(fact, tuple) and len(fact) == 5):
            continue
        actual, formal_binding, _memory, _class_cell, declaration = fact
        if actual.page != CANONICAL_VALUE:
            continue
        binding_fact = names.cell(formal_binding.row, formal_binding.column)
        consumer = (tuple(formal_binding.row[0]), int(binding_fact.value_id))
        by_actual.setdefault((tuple(actual.row[0]), int(actual.row[1])), []).append(
            (row, alias, declaration, consumer)
        )

    posted = 0
    for (class_name, field), found in writes.items():
        if len(found) != 1 or (class_name, field) in other_writes:
            continue
        write_row, write_cell, write_state, class_cell = found[0]
        literal, literal_proof = _written_literal(book, write_state.value)
        start = (tuple(write_row[0]), int(write_row[1]))
        frontier, seen = [(start, None)], {start}
        while frontier:
            receiver, incoming = frontier.pop()
            observed_cell = observed.get((*receiver, field))
            if observed_cell is not None and literal is not None and incoming is not None:
                observed_state = states.cell(observed_cell.row, observed_cell.column)
                read = observed_state.value
                if read.page == CANONICAL_VALUE and book.latest_ref(SOURCE_LITERAL_VALUE, read.row) is None:
                    book.post(
                        SOURCE_LITERAL_VALUE, read.row, literal, stage=REDUCTION,
                        provenance=Derived(tuple(dict.fromkeys((
                            read, observed_cell, incoming, write_cell, *literal_proof,
                        )))), mode=Mode.CONCORD,
                    )
                    posted += 1
            for alias_row, alias, declaration, consumer in by_actual.get(receiver, ()):
                if str(declaration.row[1]) != class_name or consumer in seen:
                    continue
                field_declaration = book.latest_ref(
                    CLASS_FIELD_DECLARATION, (*declaration.row, field),
                )
                if field_declaration is None:
                    continue
                transfer = book.post(
                    SOURCE_CALL_FIELD_STATE_TRANSFER,
                    (alias_row[0], int(receiver[1]), int(alias_row[1]), alias_row[2], int(alias_row[3]), field),
                    (TAG, write_cell, incoming, alias, field_declaration, observed.get((*consumer, field))),
                    stage=REDUCTION,
                    provenance=Derived(tuple(filter(None, (
                        write_cell, incoming, alias, field_declaration, class_cell,
                        observed.get((*consumer, field)),
                    )))), mode=Mode.CONCORD,
                )
                seen.add(consumer)
                frontier.append((consumer, transfer))
    return posted


__all__ = ["TAG", "post_receiver_field_transfers"]

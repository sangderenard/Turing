"""SOURCE_CALL_FIELD_STATE_TRANSFER generalized: a field written once reaches its observers through receiver aliases.

The incumbent writer (topological_reducer, after ``post_call_activity_corridors``) carries one call's completed
keyed mapping write to a later call's observed formal field through a certified corridor.  This carries the plain
case the AbstractTensor YoungMan needs (owner, 2026-11-12): ``self.precision = precision`` in ``__init__``, then
``self._load_and_set_geometry(...)`` -> ... -> ``_load_map``, which reads ``dtype=self.precision``.

Rule.  A field of class C whose canonical REDUCER_FIELD_STATE holds exactly one WRITTEN version in the whole program
(one row, one revision, no other kind of write of that field on a C receiver), and that the class body does not
declare (CLASS_FIELD_DECLARATION storage ``instance``: no class attribute a read could fall back to), has that
write's value on every successful read: Python raises before a read of an unset attribute, and there is no other
writer.  From the write's receiver, every SOURCE_CALL_OBJECT_ALIAS that passes that receiver into a callee posts one
transfer row, and the callee's formal carries the chain on.  Where a reached formal has an OBSERVED state of the
field, the written value's literal -- a literal itself, or a formal whose every planner specialization agrees on one
literal -- is posted as SOURCE_LITERAL_VALUE at the observed read's canonical row, so
``concorded_literals.literal_from_identity`` (and the deployment's ``dtype=`` projection) reads it.  Nothing is
guessed: no proof, no row.

Two object steps extend the chain (owner, 2026-11-13: extend the alias chain); each is a
SOURCE_FIELD_TRANSFER_OBJECT_STEP row.

- ``constructed``: a write on ``__init__``'s ``self`` reaches the constructed value through the class Call's
  constructor alias (source_constructor_field_demand.candidate_constructor_receiver_aliases).  YoungMan:
  ``CompositeGeometry.__init__`` writes ``self.precision``; its ``self`` IS the value of
  ``CompositeGeometry(geometry, precision, device)``.
- ``field_held``: when that value is the one written value of another single-written object field, the holding
  field's own transfer reaches its reads, and each read value is that same object.  YoungMan:
  ``self.geometry_wrapper = CompositeGeometry(...)`` in ``YoungManAlgorithm.__init__``; ``_configure_geometry`` reads
  ``self.geometry_wrapper`` and passes it as ``configure_geometry``'s ``self`` -> ``define_offsets`` ->
  ``dtype=self.precision``.

Reach is computed first, to a fixed point of the held steps, then posted field by field -- a field only after every
holding field whose reads it crosses -- so each step cites the receipt that reached its source.
"""

from __future__ import annotations

from typing import Any

TAG = "single-program-write-receiver-chain"


def _canonical_row(row: Any) -> bool:
    scope = row[0] if isinstance(row, tuple) and row else None
    return isinstance(scope, tuple) and len(scope) == 2 and isinstance(scope[1], int)


def _written_literal(book: Any, value: Any, _seen: frozenset = frozenset()) -> tuple[Any, tuple]:
    """The one literal the written value has everywhere, and the cells that prove it; (None, ()) when unproven.

    Program: corpus c424 ``p = P(y, x); q = P(x - 1, 2)`` with ``self.y = y`` in ``__init__``.  Wrong record: the
    written formal's literal was read from its copies' planner specializations, skipping a copy with none, so
    ``q``'s ``y = 2`` stood for ``p.y`` too (WRONG once the constructor step reached ``dot``'s reads).  Rule: a
    formal's literal is proven only when every call that reaches its function (SOURCE_FUNCTION_REACHABILITY) binds
    it (SOURCE_CALL_ACTUAL_FORMAL -- a defaulted argument is no binding) and every bound actual is the same literal:
    a literal itself, or a formal proven the same way.
    """

    from .concordance_declarations import (
        CANONICAL_VALUE, SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_ACTUAL_FORMAL, SOURCE_FUNCTION_REACHABILITY,
    )
    from .concorded_literals import MISSING, literal_from_identity

    literal, cells = literal_from_identity(book, value)
    if literal is not MISSING:
        return literal, tuple(cells)
    if value.page != CANONICAL_VALUE or value in _seen:
        return None, ()
    read_scope, value_id = tuple(value.row[0]), int(value.row[1])
    scope_cell = book.latest_ref(SOURCE_ACTIVITY_SCOPE, (read_scope,))
    if scope_cell is None:
        return None, ()
    callee = tuple(book.page(SOURCE_ACTIVITY_SCOPE).cell(scope_cell.row, scope_cell.column))
    reaching = tuple(row for row in book.page(SOURCE_FUNCTION_REACHABILITY).rows() if tuple(row[2]) == callee)
    if not reaching:
        return None, ()
    bindings = {
        (tuple(row[0]), int(row[1])): row
        for row in book.page(SOURCE_CALL_ACTUAL_FORMAL).rows()
        if tuple(row[2]) == callee and int(row[3]) == value_id
    }
    facts, proof = set(), []
    for reach in reaching:
        row = bindings.get((tuple(reach[0]), int(reach[1])))
        if row is None:
            return None, ()
        binding = book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL, row)
        actual = book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(binding.row, binding.column)[0]
        actual_literal, actual_proof = _written_literal(book, actual, _seen | {value})
        if not actual_proof:
            return None, ()
        facts.add(repr(actual_literal))
        proof.extend((book.latest_ref(SOURCE_FUNCTION_REACHABILITY, reach), binding, *actual_proof))
        literal = actual_literal
    if len(facts) != 1:
        return None, ()
    return literal, tuple(dict.fromkeys(proof))


def post_receiver_field_transfers(book: Any = None) -> int:
    """Post the generalized transfers and observed-read literals; the number of literals posted."""

    from collections import deque

    from .concordance_declarations import (
        CANONICAL_VALUE, CLASS_DECLARATION, CLASS_FIELD_DECLARATION, NAME_BINDING, REDUCER_FIELD_STATE,
        REDUCTION, SOURCE_ACTIVITY_SCOPE, SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_CALL_OBJECT_ALIAS,
        SOURCE_FIELD_TRANSFER_OBJECT_STEP, SOURCE_LITERAL_VALUE, SOURCE_VALUE_CLASS, FieldFact, FieldState,
        FieldStateKind,
    )
    from .identity_concordance import Derived, Mode, Ref, current_identity_book

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

    def canonical_key(ref):
        return (tuple(ref.row[0]), int(ref.row[1])) if ref.page == CANONICAL_VALUE else None

    # 1. Every canonical state of every field, by (receiver class, field) -- every revision, not only the latest.
    #    Program: CompositeGeometry.define_offsets reads ``dtype=self.precision`` in each geometry arm; its OBSERVED
    #    row has one revision per read (nodes 52, 80, 102, 135, 196, ...), and the latest (228) is not the
    #    tetrahedron arm's read (80) that the tetrahedron copy keeps.  Each OBSERVED revision is one read of the
    #    same field; a write revision anywhere in a row counts as a write.
    writes: dict[tuple[str, str], list] = {}
    observed: dict[tuple, tuple] = {}
    other_writes: set[tuple[str, str]] = set()
    for row in states.rows():
        if not _canonical_row(row):
            continue
        latest = book.latest_ref(REDUCER_FIELD_STATE, row)
        field = str(row[2])
        reads = []
        for column in range(latest.column + 1):
            cell = Ref(REDUCER_FIELD_STATE, latest.row, column)
            state = states.cell(cell.row, cell.column)
            if not isinstance(state, FieldState):
                continue
            if state.kind is FieldStateKind.OBSERVED:
                reads.append(cell)
                continue
            class_name, class_cell = receiver_class(tuple(row[0]), int(row[1]))
            if class_name is None:
                continue
            if state.kind is FieldStateKind.WRITTEN and cell.column == 0:
                writes.setdefault((str(class_name), field), []).append((row, cell, state, class_cell))
            else:
                other_writes.add((str(class_name), field))
        if reads:
            observed[(tuple(row[0]), int(row[1]), field)] = tuple(reads)

    # 2. Aliases by the caller-side canonical value they pass; a constructor alias (its actual is the class Call's
    #    own value) also by the ``__init__`` ``self`` it binds.
    alias_page = book.page(SOURCE_CALL_OBJECT_ALIAS)
    names = book.page(NAME_BINDING)
    by_actual: dict[tuple, list] = {}
    by_constructed: dict[tuple, list] = {}
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
        actual_key = canonical_key(actual)
        by_actual.setdefault(actual_key, []).append((row, alias, declaration, consumer))
        if actual_key[1] == int(row[1]):
            by_constructed.setdefault(consumer, []).append((row, alias, declaration, actual_key))

    # 3. The single-written instance fields.
    singles: dict[tuple[str, str], dict] = {}
    for (class_name, field), found in writes.items():
        if len(found) != 1 or (class_name, field) in other_writes:
            continue
        write_row, write_cell, write_state, class_cell = found[0]
        write_scope = authored_scope(tuple(write_row[0]))
        if write_scope is None:
            continue
        field_declaration = book.latest_ref(CLASS_FIELD_DECLARATION, (write_scope[0], class_name, field))
        field_fact = (None if field_declaration is None else
                      book.page(CLASS_FIELD_DECLARATION).cell(field_declaration.row, field_declaration.column))
        if not isinstance(field_fact, FieldFact) or field_fact.storage != "instance":
            continue
        written = canonical_key(write_state.value)
        held_class = None if written is None else receiver_class(*written)[0]
        literal, literal_proof = _written_literal(book, write_state.value)
        singles[(class_name, field)] = {
            "write_cell": write_cell, "class_cell": class_cell, "field_declaration": field_declaration,
            "start": (tuple(write_row[0]), int(write_row[1])), "literal": literal, "literal_proof": literal_proof,
            # The written value is an object a held step may carry: an authored class's instance.
            "holds": written if held_class is not None and book.latest_ref(
                CLASS_DECLARATION, (write_scope[0], str(held_class)),
            ) is not None else None,
        }

    def reach(key, held):
        class_name, field = key
        start = singles[key]["start"]
        order, parent, queue = [start], {start: None}, deque([start])
        while queue:
            node = queue.popleft()
            for alias_row, alias, declaration, consumer in by_actual.get(node, ()):
                if (str(declaration.row[1]) != class_name or consumer in parent
                        or book.latest_ref(CLASS_FIELD_DECLARATION, (*declaration.row, field)) is None):
                    continue
                parent[consumer] = ("alias", node, alias_row, alias)
                order.append(consumer)
                queue.append(consumer)
            for alias_row, alias, declaration, constructed in by_constructed.get(node, ()):
                if str(declaration.row[1]) != class_name or constructed in parent:
                    continue
                parent[constructed] = ("constructed", node, alias_row, alias)
                order.append(constructed)
                queue.append(constructed)
            for read, holder, holder_node, holder_observed in sorted(held.get(node, ()), key=repr):
                if read in parent:
                    continue
                parent[read] = ("field_held", node, holder, holder_node, holder_observed)
                order.append(read)
                queue.append(read)
        return order, parent

    # 4. Reach to a fixed point of the held steps.
    held: dict[tuple, set] = {}
    while True:
        reached = {key: reach(key, held) for key in singles}
        derived: dict[tuple, set] = {}
        for key, info in singles.items():
            if info["holds"] is None:
                continue
            for node in reached[key][0][1:]:
                for observed_cell in observed.get((*node, key[1]), ()):
                    read = canonical_key(states.cell(observed_cell.row, observed_cell.column).value)
                    if read is not None and read != info["holds"]:
                        derived.setdefault(info["holds"], set()).add((read, key, node, observed_cell))
        if derived == held:
            break
        held = derived

    # 5. Post, a field only after every holding field whose reads it crosses.
    receipts: dict[tuple, Any] = {}
    pending = list(singles)
    posted = 0
    while pending:
        ready = [
            key for key in pending
            if all(step[0] != "field_held" or (step[2], step[3]) in receipts
                   for step in reached[key][1].values() if step is not None)
        ]
        if not ready:
            break
        for key in ready:
            pending.remove(key)
            _class_name, field = key
            info = singles[key]
            write_cell, field_declaration = info["write_cell"], info["field_declaration"]
            order, parent = reached[key]
            receipts[(key, order[0])] = None
            for node in order[1:]:
                step = parent[node]
                previous = step[1]
                incoming = receipts[(key, previous)]
                reads = observed.get((*node, field), ())
                if step[0] == "alias":
                    _kind, _previous, alias_row, alias = step
                    receipt = book.post(
                        SOURCE_CALL_FIELD_STATE_TRANSFER,
                        (alias_row[0], int(previous[1]), int(alias_row[1]), alias_row[2], int(alias_row[3]), field),
                        (TAG, write_cell, incoming, alias, field_declaration, reads[-1] if reads else None),
                        stage=REDUCTION,
                        provenance=Derived(tuple(filter(None, (
                            write_cell, incoming, alias, field_declaration, info["class_cell"], *reads,
                        )))), mode=Mode.CONCORD,
                    )
                else:
                    if step[0] == "constructed":
                        proof, holder_write, holder_receipt = step[3], None, None
                    else:
                        _kind, _previous, holder, holder_node, proof = step
                        holder_write = singles[holder]["write_cell"]
                        holder_receipt = receipts[(holder, holder_node)]
                    receipt = book.post(
                        SOURCE_FIELD_TRANSFER_OBJECT_STEP, (*previous, *node, field),
                        (step[0], write_cell, incoming, proof, field_declaration, holder_write, holder_receipt),
                        stage=REDUCTION,
                        provenance=Derived(tuple(filter(None, (
                            write_cell, incoming, proof, field_declaration, info["class_cell"], holder_write,
                            holder_receipt,
                        )))), mode=Mode.CONCORD,
                    )
                receipts[(key, node)] = receipt
                if info["literal"] is None:
                    continue
                for observed_cell in reads:
                    read = states.cell(observed_cell.row, observed_cell.column).value
                    if read.page == CANONICAL_VALUE and book.latest_ref(SOURCE_LITERAL_VALUE, read.row) is None:
                        book.post(
                            SOURCE_LITERAL_VALUE, read.row, info["literal"], stage=REDUCTION,
                            provenance=Derived(tuple(dict.fromkeys((
                                read, observed_cell, receipt, write_cell, *info["literal_proof"],
                            )))), mode=Mode.CONCORD,
                        )
                        posted += 1
    return posted


__all__ = ["TAG", "post_receiver_field_transfers"]

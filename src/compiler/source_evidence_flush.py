"""Settle the already-authored whole-source evidence publishers."""

import hashlib
import os
import time

from .bounded_fixed_point import BoundedFixedPoint, bound_for
from .concordance_declarations import (
    CALLABLE_IDENTITY, CANONICAL_VALUE, CLASS_DECLARATION,
    CLASS_FIELD_DECLARATION,
    CLASS_METHOD_DECLARATION, NAME_BINDING, PROCESS_GRAPH_OPERANDS,
    FIXED_POINT_ROUND, FIXED_POINT_ROUND_STAGE, FUNCTION_ADDRESS,
    REDUCER_FIELD_STATE,
    INGESTION_VALUE, SOURCE_ACTIVITY_BINDING, SOURCE_ACTIVITY_CALL,
    SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_NODE,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_CALL_ACTIVITY_CORRIDOR,
    SOURCE_CALL_ACTUAL_FORMAL, SOURCE_CALL_FIELD_EFFECT_COMPLETION,
    SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_CALL_RESULT_IDENTITY, SEMANTIC_OUTPUT, SOURCE_ACTIVITY_SCOPE,
    SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND,
    SOURCE_CALL_FIELD_STATE_TRANSFER, SOURCE_CALL_OBJECT_ALIAS,
    SOURCE_FUNCTION_EXIT_FIELD_STATE, SOURCE_FUNCTION_REACHABILITY,
    SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_EVIDENCE_FLUSH_ROUND,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_LITERAL_VALUE, SOURCE_MAPPING_GET_READ, SOURCE_OPERATOR_KIND,
    SOURCE_PARAMETER_MEMORY_CONTRACT, SOURCE_RESOURCE_ADMISSION_DEMAND,
    SOURCE_SEQUENCE_MUTATION, SOURCE_SPAN, SOURCE_TERMINAL_RETURN,
    SOURCE_VALUE_CLASS,
)
from .identity_concordance import Derived, Mode, current_identity_book
from .source_guarded_callable import post_guarded_callables
from .source_class_domain_forward import post_formal_return_class_domains
from .source_bytes_concat_class import post_bytes_concat_classes
from .source_call_return_class import post_reached_bytes_return_classes
from .source_tuple_member_demand import post_tuple_member_demands
from .source_guarded_payload import post_guarded_mapping_payloads
from .source_mapping_get import post_mapping_get_reads


_FAMILY = (
    SOURCE_FUNCTION_REACHABILITY, SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_RESULT_IDENTITY, SEMANTIC_OUTPUT, SOURCE_ACTIVITY_SCOPE,
    SOURCE_PARAMETER_MEMORY_CONTRACT, SOURCE_CALL_FIELD_EFFECT_COMPLETION,
    SOURCE_CALL_OBJECT_ALIAS,
    SOURCE_FUNCTION_EXIT_FIELD_STATE, REDUCER_FIELD_STATE,
    SOURCE_SPAN, INGESTION_VALUE, CANONICAL_VALUE,
    SOURCE_ACTIVITY_NODE, SOURCE_ACTIVITY_EDGE, SOURCE_ACTIVITY_BINDING,
    SOURCE_ACTIVITY_CALL,
    SOURCE_CALL_ACTIVITY_CORRIDOR, SOURCE_CALL_FIELD_STATE_TRANSFER,
    SOURCE_VALUE_CLASS, CLASS_DECLARATION, CLASS_FIELD_DECLARATION,
    CLASS_METHOD_DECLARATION, FUNCTION_ADDRESS, SOURCE_SEQUENCE_MUTATION,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_LITERAL_VALUE, PROCESS_GRAPH_OPERANDS,
    NAME_BINDING, SOURCE_OPERATOR_KIND, SOURCE_TERMINAL_RETURN,
    SOURCE_RESOURCE_ADMISSION_DEMAND, SOURCE_MAPPING_GET_READ,
    SOURCE_GUARDED_MAPPING_PAYLOAD, CALLABLE_IDENTITY,
    SOURCE_CALL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_FORMAL_CLOSED_CLASS_ALTERNATIVES,
    SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND,
)


def _snapshot(book):
    """Sorted exact cells and direct provenance, including added edges."""
    return tuple(sorted(
        ((cell, tuple(sorted(book.edges_into(cell), key=repr)))
         for page in _FAMILY for row in book.page(page).rows()
         for cell in (book.latest_ref(page, row),)),
        key=lambda entry: repr(entry[0]),
    ))


def _entry_digest(entry):
    return int.from_bytes(hashlib.sha256(repr(entry).encode('utf-8')).digest(), 'big')


def _digest(cells):
    """Commutative fingerprint so a round can prove its small exact delta."""
    value = 0
    for entry in cells:
        value ^= _entry_digest(entry)
    return f'{value:064x}'


def _apply_changes(before_digest, changes):
    value = int(before_digest, 16)
    for old, new in changes:
        if old is not None:
            value ^= _entry_digest(old)
        if new is not None:
            value ^= _entry_digest(new)
    return f'{value:064x}'


def flush_source_evidence(source_graph, *, book=None):
    """Run existing post-transfer readers until no family row changes."""
    book = current_identity_book() if book is None else book
    function_table = source_graph.function_table
    entries = tuple(entry for entry in function_table
                    if getattr(entry, 'graph', None) is not None)
    scopes = tuple(sorted({tuple(entry.graph.G.graph.get(
        'authored_function_identity') or ())[:1]
        for entry in entries}, key=repr))
    scope = ('source-evidence', *scopes)
    name = 'whole-source-evidence-flush'
    initial = _digest(_snapshot(book))
    previous = tuple((row[2], book.latest_ref(FIXED_POINT_ROUND, row),
                      book.page(FIXED_POINT_ROUND).latest(row))
                     for row in book.page(FIXED_POINT_ROUND).rows()
                     if row[:2] == (name, scope))
    last_index, last_ref, last = (0, None, None)
    if previous:
        last_index, last_ref, last = max(previous, key=lambda item: item[0])
        if last[0] is False and last[1] == hashlib.sha256(
            repr(initial).encode('utf-8')
        ).hexdigest()[:16]:
            return 0
    # Every new mapping read is owned by a canonical Call; every guarded
    # payload and callable belongs to one such read. The finite graph size
    # bounds those monotone row additions plus the final no-change round.
    nodes = sum(len(entry.graph.G) for entry in entries)
    bound = bound_for('planner', nodes=nodes,
                      callsites=len(tuple(book.page(SOURCE_MAPPING_GET_READ).rows())))
    guard = BoundedFixedPoint(
        name, last_index + bound, scope=scope, state=initial, book=book,
        start_index=last_index, previous_ref=last_ref,
    )
    started = time.monotonic()
    progress = f"source-evidence[{os.getpid()}]"
    print(f"{progress} start: functions={len(entries)} nodes={nodes}", flush=True)
    while True:
        before_cells = _snapshot(book)
        before = _digest(before_cells)
        round_number = guard.rounds + 1
        print(f"{progress} round={round_number} mapping reads", flush=True)
        post_mapping_get_reads(function_table, book=book)
        print(f"{progress} round={round_number} guarded evidence", flush=True)
        post_guarded_mapping_payloads(book)
        post_guarded_callables(source_graph, book)
        post_bytes_concat_classes(function_table, book)
        post_reached_bytes_return_classes(function_table, book)
        print(f"{progress} round={round_number} call and tuple domains", flush=True)
        post_formal_return_class_domains(book)
        post_tuple_member_demands(function_table, book=book)
        after_cells = _snapshot(book)
        after = _digest(after_cells)
        changed = before != after
        prior = {cell: entry for entry in before_cells
                 for cell in (entry[0],)}
        current = {cell: entry for entry in after_cells
                   for cell in (entry[0],)}
        changes = tuple((prior.get(cell), current.get(cell))
                        for cell in sorted(prior.keys() | current.keys(), key=repr)
                        if prior.get(cell) != current.get(cell))
        ids = tuple(new[0] if new is not None else old[0]
                    for old, new in changes)
        continuing = guard.round(changed, changed_ids=ids, state=after)
        guard_cell = book.latest_ref(FIXED_POINT_ROUND,
                                     (name, scope, guard.rounds))
        restarted = last_index > 0 and guard.rounds == last_index + 1
        causes = tuple(dict.fromkeys((
            *(entry[0] for pair in changes for entry in pair
              if entry is not None),
            *(source for pair in changes for entry in pair
              if entry is not None for source, _stage in entry[1]),
            guard_cell,
        )))
        book.post(SOURCE_EVIDENCE_FLUSH_ROUND, (scope, guard.rounds),
                  (before, after, changes, guard_cell,
                   restarted),
                  stage=FIXED_POINT_ROUND_STAGE,
                  provenance=Derived(causes),
                  mode=Mode.CONCORD)
        print(f"{progress} round={guard.rounds} changed={len(changes)} "
              f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        if not continuing:
            print(f"{progress} settled: rounds={guard.rounds - last_index} "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
            return guard.rounds - last_index

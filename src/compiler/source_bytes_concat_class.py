"""Publish the class of exact authored bytes additions before planning."""

from .concordance_declarations import (
    CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS, SOURCE_VALUE_CLASS,
)
from .identity_concordance import Derived
from .source_evidence_resolver import ClassEvidence, require_class


def _class_matches_operand(book, class_cell, operand):
    """Follow only class relabels to this operand's one ingestion origin."""

    from .concordance_declarations import INGESTION_VALUE
    from .concorded_literals import origin_cell

    try:
        expected = origin_cell(book, operand)
    except ValueError:
        return False
    if operand in {cause for cause, _ in book.edges_into(class_cell)}:
        return True
    pending = [class_cell]
    seen = set()
    origins = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for cause, _stage in book.edges_into(current):
            if cause.page == SOURCE_VALUE_CLASS:
                pending.append(cause)
            elif cause.page in {CANONICAL_VALUE, INGESTION_VALUE}:
                try:
                    origins.add(origin_cell(book, cause))
                except ValueError:
                    return False
    return origins == {expected}


def post_bytes_concat_classes(function_table, book) -> int:
    """Use the incumbent class writer for bytes + bytes source values.

    The operand snapshot and its canonical causes identify the Add. Each
    input's class cell must cite that canonical value or its exact identity
    lineage, so a same-numbered class from another fork cannot select it.
    """

    from ..common.tensors.topological_reducer import (
        _concord_source_value_class, _source_numeric_scope,
    )

    posted = 0
    for entry in function_table:
        graph = getattr(entry, "graph", None)
        if graph is None:
            continue
        nodes = graph.G
        source_scope = nodes.graph.get("lexical_read_scope")
        if source_scope is None:
            continue
        class_scope = _source_numeric_scope(graph)
        for node_id, data in sorted(nodes.nodes(data=True),
                                    key=lambda item: int(item[0])):
            if str(data.get("type") or data.get("op")) != "Add":
                continue
            result_id = int(data.get("value_id", node_id))
            result = book.latest_ref(
                CANONICAL_VALUE, (source_scope, result_id),
            )
            operands = book.latest_ref(
                PROCESS_GRAPH_OPERANDS, (source_scope, result_id),
            )
            if result is None or operands is None:
                continue
            result_fact = book.page(CANONICAL_VALUE).cell(
                result.row, result.column,
            )
            inputs = book.page(PROCESS_GRAPH_OPERANDS).cell(
                operands.row, operands.column,
            )
            if (
                "value:BinOp" not in result_fact
                or "value:Add" not in result_fact
                or not isinstance(inputs, tuple)
                or len(inputs) != 2
                or tuple(role for _cell, role in inputs) != ("lhs", "rhs")
                or result not in {
                    parent for parent, _stage in book.edges_into(operands)
                }
            ):
                continue
            classes = []
            for parent, _role in inputs:
                if parent.page != CANONICAL_VALUE:
                    break
                answer = require_class(
                    class_scope, int(parent.row[1]), book=book,
                )
                if (
                    not isinstance(answer, ClassEvidence)
                    or answer.class_identity != "builtins.bytes"
                    or answer.class_cell is None
                    or not _class_matches_operand(
                        book, answer.class_cell, parent,
                    )
                ):
                    break
                classes.append(answer.class_cell)
            if len(classes) != 2:
                continue
            if book.latest_ref(SOURCE_VALUE_CLASS,
                               (class_scope, result_id)) is not None:
                continue
            _concord_source_value_class(
                class_scope, result_id, "builtins.bytes",
                source="source bytes concatenation",
                provenance=Derived((result, operands, *classes)),
            )
            posted += 1
    return posted

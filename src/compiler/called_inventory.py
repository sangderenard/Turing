"""The called inventory: what the callers pass for each formal.

Albert's rule: where Python allows a parameter to be of several classes the
compiler does not guess and does not widen.  It GATHERS, from every call
record that reaches a formal, the exact Python class of the argument, posts
that set on the identity book (``CALLED_INVENTORY``, provenance = the call
record cells it read) and lowers from that record.

Sources of call records:
  * root-call contract rows (``root_calls=`` on ``lower_ast_source_to_ssa``):
    one row per known call of the entry, ``{formal: value-or-class}``;
    posted as ``ROOT_CALL_RECORD`` cells.
  * (open) internal call edges' actuals, today consumed by per-callsite
    specialization directly.

Reading: ``called_class(book, function_identity, formal)`` returns the class
the specialized copy under compilation is built for, or raises a NAMED
refusal.  One class in the inventory is that class; several classes require
a ``selected`` signature (one specialized copy per distinct signature, see
``distinct_signatures``); none is a refusal.  No default exists.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def class_spelling(value: Any) -> str:
    """The exact Python class of a call argument (a value or a class)."""

    cls = value if isinstance(value, type) else type(value)
    # A list call record names its element class: ``list[int]`` types the
    # formal's rows, where the bare ``list`` left every row dtype unsettled.
    # An empty or mixed-class list proves no element class and stays ``list``.
    if isinstance(value, list) and value:
        elements = {type(item).__name__ for item in value}
        if len(elements) == 1 and elements <= {"int", "float", "bool"}:
            return f"list[{elements.pop()}]"
    return cls.__name__


def post_root_calls(
    book: Any, function_identity: str, root_calls: Iterable[Mapping[str, Any]],
    *, selected: Mapping[str, str] | None = None,
) -> None:
    from .concordance_declarations import (
        CALLED_INVENTORY, CONTRACT_DEMAND, CalledInventoryFact,
        INGESTION, ROOT_CALL_RECORD,
    )
    from .identity_concordance import Derived, Mode, Novel

    cells: dict[str, list] = {}
    classes: dict[str, set] = {}
    for ordinal, call in enumerate(root_calls):
        record = tuple(sorted(
            (str(name), class_spelling(value)) for name, value in call.items()
        ))
        row = (function_identity, ordinal)
        ref = book.post(
            ROOT_CALL_RECORD, row, record, stage=INGESTION,
            provenance=Novel(CONTRACT_DEMAND, ()), mode=Mode.CONCORD,
        )
        for formal, cls in record:
            cells.setdefault(formal, []).append(ref)
            classes.setdefault(formal, set()).add(cls)
    for formal, found in classes.items():
        chosen = None if selected is None else selected.get(formal)
        if chosen is not None and chosen not in found:
            raise ValueError(
                f"selected class {chosen} for formal {formal} of "
                f"{function_identity} is not in its called inventory "
                f"{sorted(found)}"
            )
        if chosen is None and len(found) == 1:
            chosen = next(iter(found))
        book.post(
            CALLED_INVENTORY, (function_identity, formal),
            CalledInventoryFact(tuple(sorted(found)), chosen, True),
            stage=INGESTION, provenance=Derived(tuple(cells[formal])),
            mode=Mode.REVISE,
        )


def distinct_signatures(root_calls: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """One entry per distinct class signature among the root-call rows."""

    seen: dict[tuple, dict[str, str]] = {}
    for call in root_calls:
        sig = tuple(sorted((str(k), class_spelling(v)) for k, v in call.items()))
        seen.setdefault(sig, dict(sig))
    return list(seen.values())


def signature_suffix(signature: Mapping[str, str]) -> str:
    return "_".join(
        signature[name].replace("[", "_").replace("]", "")
        for name in sorted(signature)
    )


def called_class(book: Any, function_identity: str, formal: str):
    """The inventory cell for a formal: ``(class, ref)`` or ``None`` when no
    call record was supplied for this function (the formal then keeps
    whatever other evidence types it).  Several classes without a selected
    signature, or a selection outside the inventory, is a named refusal."""

    from .concordance_declarations import CALLED_INVENTORY

    row = (function_identity, formal)
    fact = book.page(CALLED_INVENTORY).latest(row)
    if fact is None or not fact.root:
        return None
    if not fact.classes:
        raise ValueError(
            f"no call record or contract reaches formal {formal} of "
            f"{function_identity}"
        )
    if fact.selected is None:
        raise ValueError(
            f"formal {formal} of {function_identity} is called with classes "
            f"{list(fact.classes)}; lower one specialized copy per class "
            "signature (lower_ast_source_by_signature)"
        )
    return fact.selected, book.latest_ref(CALLED_INVENTORY, row)


def annotate_from_inventory(tree: Any, function_identity: str, book: Any) -> None:
    """Give each unannotated formal of the entry the class its inventory
    names, as the annotation AST every later reader already consumes."""

    import ast

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_identity:
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                if arg.annotation is not None:
                    continue
                got = called_class(book, function_identity, arg.arg)
                if got is None:
                    continue
                arg.annotation = ast.copy_location(
                    ast.parse(got[0], mode="eval").body, arg)


def post_call_edge_actual(
    book: Any, callee_identity: str, formal: str, caller_scope: Any,
    call_id: int, class_spelling_: str, class_cell: Any,
) -> None:
    """Record one internal call edge's actual and fold it into the callee
    formal's inventory.  The edge cell is Derived from the caller's
    SOURCE_VALUE_CLASS cell; the inventory cell is Derived from every edge
    cell (and root-call record) that reaches the formal, so its provenance
    cites each call edge.  The set grows to exactly the classes found."""

    from .concordance_declarations import (
        CALL_EDGE_ACTUAL, CALLED_INVENTORY, CalledInventoryFact,
        PLANNER_TENSOR_SPECIALIZATION,
    )
    from .identity_concordance import Derived, Mode

    cls = str(class_spelling_)
    if cls.startswith("builtins."):
        cls = cls[len("builtins."):]
    edge_row = (caller_scope, int(call_id), str(formal))
    edge = book.latest_ref(CALL_EDGE_ACTUAL, edge_row)
    # Nested specialization revisits an edge; a visit that restates the same
    # actual from the same class cell has nothing to revise.
    if not (
        edge is not None
        and book.page(CALL_EDGE_ACTUAL).latest(edge_row) == cls
        and class_cell in [ref for ref, _stage in book.edges_into(edge)]
    ):
        book.post(
            CALL_EDGE_ACTUAL, edge_row, cls,
            stage=PLANNER_TENSOR_SPECIALIZATION,
            provenance=Derived((class_cell,)), mode=Mode.REVISE,
        )
        edge = book.latest_ref(CALL_EDGE_ACTUAL, edge_row)
    row = (str(callee_identity), str(formal))
    incumbent = book.page(CALLED_INVENTORY).latest(row)
    cell = book.latest_ref(CALLED_INVENTORY, row) if incumbent is not None else None
    cited = [ref for ref, _stage in book.edges_into(cell)] if cell is not None else []
    classes = set(incumbent.classes) if incumbent is not None else set()
    if incumbent is not None and cls in classes and edge in cited:
        return
    classes.add(cls)
    cited.append(edge)
    book.post(
        CALLED_INVENTORY, row,
        CalledInventoryFact(
            tuple(sorted(classes)),
            None if incumbent is None else incumbent.selected,
            False if incumbent is None else incumbent.root,
        ),
        stage=PLANNER_TENSOR_SPECIALIZATION,
        provenance=Derived(tuple(dict.fromkeys(cited))), mode=Mode.REVISE,
    )

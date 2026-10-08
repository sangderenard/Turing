"""Planner concordance posts: the single-api posts the deployment planner makes for regions, callsite binding, activation and specialization."""

from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping


def _post_deployment_region(graph: Any, subgraph: Any, region_index: int) -> None:
    """The book rows of one carved region (plan 80, A2.4)."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        DEPLOYMENT_REGION, DEPLOYMENT_REGION_MEMBER, DISPATCH_STORE,
        DISPATCH_STORE_PAGE, EXECUTABLE_NODE, MemberRole,
        PLANNER_REGION_CARVE, RegionFact,
    )
    from .identity_concordance import (
        Derived, Mode, Novel, current_identity_book,
    )

    planning_scope = graph.G.graph.get("planning_scope")
    if planning_scope is None:
        return
    book = current_identity_book()
    metadata = subgraph.G.graph
    inputs = tuple(map(int, metadata.get("deployment_inputs", ())))
    outputs = tuple(map(int, metadata.get("deployment_outputs", ())))
    nodes = tuple(map(int, metadata.get("deployment_nodes", ())))
    members = tuple(dict.fromkeys((*inputs, *nodes, *outputs)))

    def member_cell(node_id: int) -> Any:
        # The classifier's verdict for this node under this planning scope;
        # a boundary node the pass never classified is named by its identity.
        cell = book.latest_ref(EXECUTABLE_NODE, (planning_scope, node_id))
        if cell is not None:
            return cell
        try:
            return node_identity_cell(graph, node_id)
        except ValueError:
            return None

    member_cells = {
        node_id: cell for node_id in members
        for cell in (member_cell(node_id),) if cell is not None
    }
    if not member_cells:
        return
    region_row = (planning_scope, int(region_index))
    fact = RegionFact(
        str(metadata.get("compartment_schedule_preference") or "asap"),
        len(outputs), len(metadata.get("deployment_store_nodes", ())),
    )
    if book.latest_ref(DEPLOYMENT_REGION, region_row) is not None:
        # The same ordinal carved twice in one planning scope (a re-plan of
        # this graph without a new classifier pass) restates the region.
        book.post(
            DEPLOYMENT_REGION, region_row, fact, stage=PLANNER_REGION_CARVE,
            provenance=Derived(tuple(member_cells.values())),
            mode=Mode.CONCORD,
        )
        return
    region_cell = book.post(
        DEPLOYMENT_REGION, region_row, fact, stage=PLANNER_REGION_CARVE,
        provenance=Derived(tuple(member_cells.values())), mode=Mode.CONCORD,
    )
    output_member_cells: dict[int, Any] = {}
    for node_id, cell in member_cells.items():
        role = (
            MemberRole.OUTPUT if node_id in outputs
            else MemberRole.INPUT if node_id in inputs
            else MemberRole.NODE
        )
        member_ref = book.post(
            DEPLOYMENT_REGION_MEMBER, (planning_scope, int(region_index), node_id),
            role, stage=PLANNER_REGION_CARVE,
            provenance=Derived((cell, region_cell)), mode=Mode.CONCORD,
        )
        if role is MemberRole.OUTPUT:
            output_member_cells[node_id] = member_ref
    for output_id, store_id in zip(
        outputs, metadata.get("deployment_store_nodes", ()),
    ):
        member_ref = output_member_cells.get(int(output_id))
        if member_ref is None:
            continue
        book.post(
            DISPATCH_STORE_PAGE,
            (planning_scope, int(region_index), int(output_id)), int(store_id),
            stage=PLANNER_REGION_CARVE,
            provenance=Novel(DISPATCH_STORE, (member_ref,)), mode=Mode.CONCORD,
        )


def _post_if_changed(
    page: Any, row: tuple, fact: Any, *, stage: Any, cells: tuple,
) -> Any:
    """One REVISE post on ``page`` unless the row's latest cell already
    states ``fact`` from exactly these source cells.

    The planner's fixed points re-derive the same fact from the same cells
    on every round; ``post`` refuses such a revision (no changed source) and
    plan 80 R5 places the check at the writer, never in a ``try``.  A
    changed fact, or the same fact from a different cell set (a new
    callsite), is a revision with a cause and is posted.  Returns the Ref of
    the row's latest cell (the existing one when nothing was posted).
    """

    from .identity_concordance import Derived, Mode, current_identity_book

    book = current_identity_book()
    latest = book.latest_ref(page, row)
    if latest is not None:
        incumbent = book.pages[page.name].latest(row)
        if incumbent == fact and {
            source.key for source, _stage in book.edges_into(latest)
        } == {cell.key for cell in cells}:
            return latest
    return book.post(
        page, row, fact, stage=stage, provenance=Derived(tuple(cells)),
        mode=Mode.REVISE,
    )


def _argument_identity_cells(graph: Any, node_id: int) -> tuple:
    """The cells that identify one call argument as source data: the node's
    identity cell; for an authored literal its ``source_span`` cell; for a
    folded Constant the ``proven_literal`` cell the fold posted (the cause
    that turned a dynamic argument into a literal); for an aggregate literal
    the cells of its members."""

    from ..common.tensors.topological_reducer import (
        _post_source_span, node_identity_cell,
    )
    from .concordance_declarations import PROVEN_LITERAL
    from .identity_concordance import current_identity_book

    node_id = int(node_id)
    if node_id not in graph.G:
        return ()
    data = graph.G.nodes[node_id]
    cells: list = [node_identity_cell(graph, node_id)]
    expression = data.get("expr_obj")
    if isinstance(expression, ast.Constant):
        span = _post_source_span(expression)
        if span is not None:
            cells.append(span)
    literal = current_identity_book().latest_ref(
        PROVEN_LITERAL,
        (str(graph.G.graph.get("function_name")), int(data.get("value_id", node_id))),
    )
    if literal is not None:
        cells.append(literal)
    if data.get("type") == "Input":
        # A formal passed straight on is source-static when the caller's own
        # formal was specialized (``_source_static_value`` reads the
        # caller's ``planner_specializations``).  That decision is the
        # caller's ``planner_specialization`` row; cite it, so the callee's
        # row changes only with a cause.  Without it, dt_system_over's
        # lowering posted ``_propose_dt_pen``'s ``distribution`` as
        # dynamic, then -- once step_with_dt_control_used's ``distribution``
        # was specialized in the same pass -- as LITERAL from the same two
        # cells, and the book refused the causeless revision
        # (CONTINUATION_dt_compile_stall.md).
        from .concordance_declarations import PLANNER_SPECIALIZATION_PAGE

        scope = graph.G.graph.get("lexical_read_scope")
        binding = (data.get("attributes") or {}).get("binding_name")
        if scope is not None and binding:
            decided = current_identity_book().latest_ref(
                PLANNER_SPECIALIZATION_PAGE, (tuple(scope), str(binding)),
            )
            if decided is not None:
                cells.append(decided)
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        for parent, _role in data.get("parents") or ():
            cells.extend(_argument_identity_cells(graph, int(parent)))
    return tuple(dict.fromkeys(cells))


def _formal_identity_cells(callee: Any, parameter: str) -> tuple:
    """The cells that identify a callee formal bound to its signature
    default: the Input node's identity cell and its ``scalar_parameter``
    row when the reducer posted one (that row derives from the default's
    span; the def statement itself is not kept on the graph)."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import SCALAR_PARAMETER
    from .identity_concordance import current_identity_book

    cells: list = []
    book = current_identity_book()
    for node_id, data in callee.G.nodes(data=True):
        if data.get("type") != "Input" or str(
            (data.get("attributes") or {}).get("binding_name") or ""
        ) != str(parameter):
            continue
        cell = node_identity_cell(callee, int(node_id))
        cells.append(cell)
        scalar = book.latest_ref(SCALAR_PARAMETER, (cell.row[0], int(node_id)))
        if scalar is not None:
            cells.append(scalar)
    return tuple(dict.fromkeys(cells))


def _post_planner_specialization(
    callee: Any, parameter: str, fact: Any, cells: tuple, *, stage: Any = None,
) -> Any:
    """One ``planner_specialization`` row for ``callee``'s formal under the
    callee graph's own read scope (per copy, design 7.1); None when the
    graph has no scope (nothing to key the row by) or no cell proves it."""

    from .concordance_declarations import (
        PLANNER_SPECIALIZATION, PLANNER_SPECIALIZATION_PAGE,
    )

    scope = callee.G.graph.get("lexical_read_scope")
    if scope is None or not cells:
        return None
    return _post_if_changed(
        PLANNER_SPECIALIZATION_PAGE, (tuple(scope), str(parameter)), fact,
        stage=PLANNER_SPECIALIZATION if stage is None else stage, cells=cells,
    )


def _post_call_binding(
    graph: Any, node_id: int, reference: int, resolution: str,
) -> Any:
    """One ``call_binding`` row (caller read scope, call node) ->
    ``CallBinding(callee, resolution)`` DERIVED from the call node's identity
    cell and the callee's ``function_address`` cell (plan 80, A2.8).  None
    when the caller graph has no read scope; a callee with no address row
    derives from the call cell alone."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        CALL_BINDING, CallBinding, CallResolution, FUNCTION_ADDRESS,
        PLANNER_CALL_BINDING,
    )
    from .identity_concordance import (
        Derived, Mode, current_identity_book,
    )

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None or int(node_id) not in graph.G:
        return None
    book = current_identity_book()
    try:
        cells = [node_identity_cell(graph, int(node_id))]
    except ValueError:
        return None
    table = getattr(graph, "function_table", None)
    if table is not None:
        try:
            entry = table.entry(int(reference))
        except (KeyError, TypeError, ValueError):
            entry = None
        if entry is not None:
            address = book.latest_ref(
                FUNCTION_ADDRESS, (str(entry.qualified_name),),
            )
            if address is not None:
                cells.append(address)
    row = (tuple(scope), int(node_id))
    fact = CallBinding(int(reference), CallResolution(resolution))
    page = book.pages.get(CALL_BINDING.name)
    if page is not None and page.latest(row) is not None and page.latest(row) != fact:
        # A call re-resolved to another callee (a receiver class proven
        # later): a revision with the new address as its cause.
        return book.post(
            CALL_BINDING, row, fact, stage=PLANNER_CALL_BINDING,
            provenance=Derived(tuple(cells)), mode=Mode.REVISE,
        )
    return book.post(
        CALL_BINDING, row, fact, stage=PLANNER_CALL_BINDING,
        provenance=Derived(tuple(cells)), mode=Mode.CONCORD,
    )


def _post_callsite_activation(
    identity_row: tuple, activation_fact: tuple, binding_cell: Any,
) -> None:
    """The ``source_callsite_activation_concordance`` row DERIVED from the
    call's ``call_binding`` cell; raw (tagged) when there is none."""

    from .concordance_declarations import (
        PLANNER_CALL_BINDING, SOURCE_CALLSITE_ACTIVATION,
    )
    from .identity_concordance import (
        Derived, Mode, current_identity_book,
    )

    book = current_identity_book()
    if binding_cell is None:
        book.page(SOURCE_CALLSITE_ACTIVATION).set(identity_row, 0, activation_fact)
        return
    book.post(
        SOURCE_CALLSITE_ACTIVATION, identity_row, activation_fact,
        stage=PLANNER_CALL_BINDING, provenance=Derived((binding_cell,)),
        mode=Mode.CONCORD,
    )


def _post_control_specialization(
    graph: Any, control_id: int, record: dict, predicate_id: Any, *,
    predicate_known: bool,
) -> tuple:
    """One ``source_control_specialization_concordance`` row (function,
    retained control): the fold record DERIVED from the test's
    ``proven_literal`` cell (when the fold posted one) and the control node's
    identity cell; ``Unresolved(PREDICATE_NOT_KNOWN)`` reading the control
    cell when the test was not in ``known``.  Returns the cells posted."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        PLANNER_STRUCTURAL_FOLD, PREDICATE_NOT_KNOWN, PROVEN_LITERAL,
        SOURCE_CONTROL_SPECIALIZATION,
    )
    from .identity_concordance import (
        Mode, RAW_PRIMITIVE, Unresolved, Unsourced, current_identity_book,
    )

    book = current_identity_book()
    function_name = str(graph.G.graph.get("function_name"))
    row = (function_name, int(control_id))
    cells: list = []
    try:
        if int(control_id) in graph.G:
            cells.append(node_identity_cell(graph, int(control_id)))
        elif int(record.get("graph_control_id", -1)) in graph.G:
            cells.append(node_identity_cell(
                graph, int(record["graph_control_id"]),
            ))
    except ValueError:
        pass
    if predicate_id is not None and int(predicate_id) in graph.G:
        literal = book.latest_ref(PROVEN_LITERAL, (
            function_name,
            int(graph.G.nodes[int(predicate_id)].get("value_id", predicate_id)),
        ))
        if literal is not None:
            cells.append(literal)
    fact: Any = (
        record if predicate_known
        else Unresolved(PREDICATE_NOT_KNOWN, read=tuple(cells))
    )
    if cells:
        return (_post_if_changed(
            SOURCE_CONTROL_SPECIALIZATION, row, fact,
            stage=PLANNER_STRUCTURAL_FOLD, cells=tuple(cells),
        ),)
    page = book.pages.get(SOURCE_CONTROL_SPECIALIZATION.name)
    if page is not None and page.latest(row) == fact:
        return ()
    return (book.post(
        SOURCE_CONTROL_SPECIALIZATION, row, fact, stage=PLANNER_STRUCTURAL_FOLD,
        provenance=Unsourced(RAW_PRIMITIVE), mode=Mode.REVISE,
    ),)


def _post_pruned_return_sites(
    graph: Any, selected_sites: Mapping[Any, Any], control_id: int,
) -> None:
    """The structural fold proved one top-level arm terminal and kept only
    its return site: every other site's ``return_site_slot`` rows revise to
    ``Unresolved(RETURN_SITE_UNREACHABLE)`` reading the previous slot cell
    and the folded control's identity cell (plan 80, A2.7; plan 70, 4)."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        PLANNER_STRUCTURAL_FOLD, RETURN_SITE_SLOT, RETURN_SITE_UNREACHABLE,
    )
    from .identity_concordance import (
        Derived, Mode, Ref, Unresolved, current_identity_book,
    )

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    book = current_identity_book()
    slots = book.pages.get(RETURN_SITE_SLOT.name)
    if slots is None:
        return
    try:
        control_cell = (
            node_identity_cell(graph, int(control_id))
            if int(control_id) in graph.G else None
        )
    except ValueError:
        control_cell = None
    selected_spans = {
        tuple(site) for site in selected_sites if isinstance(site, tuple)
    }
    for row in tuple(slots.scope_rows(tuple(scope))):
        fact = slots.latest(row)
        if not isinstance(fact, Ref):
            continue
        site = row[1]
        # The return-site key is the ``source_span`` row of the returned
        # expression (a PAGE_REF); the ledger keys sites by its positions.
        if not isinstance(site, Ref) or site.page.name != "source_span":
            continue
        site_page = book.pages.get(site.page.name)
        span = None if site_page is None else site_page.latest(site.row)
        if span is None:
            continue
        positions = tuple(
            getattr(span, name, None) for name in (
                "lineno", "col_offset", "end_lineno", "end_col_offset",
            )
        )
        if positions in selected_spans:
            continue
        previous = book.latest_ref(RETURN_SITE_SLOT, row)
        cells = tuple(cell for cell in (previous, control_cell) if cell is not None)
        book.post(
            RETURN_SITE_SLOT, row,
            Unresolved(RETURN_SITE_UNREACHABLE, read=cells),
            stage=PLANNER_STRUCTURAL_FOLD, provenance=Derived(cells),
            mode=Mode.REVISE,
        )


def _fold_literal_source_cells(graph: Any, node_id: int) -> tuple:
    """The cells a structural fold of ``node_id`` evaluated: the node's own
    identity cell, each operand's identity cell, and the
    ``planner_specialization`` cell of an operand Input the planner fed."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import PLANNER_SPECIALIZATION_PAGE
    from .identity_concordance import current_identity_book

    if node_id not in graph.G:
        return ()
    book = current_identity_book()
    scope = graph.G.graph.get("lexical_read_scope")
    cells: list = []

    def specialization_cell(data: Any) -> Any:
        binding = (data.get("attributes") or {}).get("binding_name")
        if data.get("type") != "Input" or binding is None or scope is None:
            return None
        return book.latest_ref(
            PLANNER_SPECIALIZATION_PAGE, (tuple(scope), str(binding)),
        )

    try:
        cells.append(node_identity_cell(graph, node_id))
        # The node itself may be the planner-fed Input (``rollback`` folded
        # to its literal): its own specialization cell is the cause.
        own = specialization_cell(graph.G.nodes[node_id])
        if own is not None:
            cells.append(own)
        for parent, _role in graph.G.nodes[node_id].get("parents") or ():
            if int(parent) not in graph.G:
                continue
            cells.append(node_identity_cell(graph, int(parent)))
            specialization = specialization_cell(graph.G.nodes[int(parent)])
            if specialization is not None:
                cells.append(specialization)
    except ValueError:
        return ()
    return tuple(dict.fromkeys(cells))


def _post_identity_table_mutation(
    graph: Any, *, removed: Iterable[int] = (),
    aliases: Mapping[int, int] | None = None, cause_cells: tuple = (),
    stage: Any = None,
) -> None:
    """Record a post-reduction ``identity_table`` mutation on ``name_binding``
    (plan 80, A2.7; plan 60 R1).

    The reducer materialized the dict from the canonical ``name_binding``
    rows; every planner rewrite of the dict is a REVISE of the rows it
    changes, so the page and the dict stay one record:

    - a version whose value node was REMOVED revises to
      ``Unresolved(BINDING_VERSION_REMOVED)`` reading its previous cell and
      the removed node's ``canonical_value`` cell (the view skips
      ``Unresolved`` rows, which is today's filtered tuple);
    - a version whose value was ALIASED to another node revises to a
      ``BindingFact`` naming the alias source, DERIVED from its previous
      cell, the source node's identity cell and ``cause_cells`` (the
      ``identity_transition`` cells ``_set_operands`` wrote, when declared).

    A return slot (``return_site_slot``) whose value was aliased revises to
    the source's cell the same way.  The dict rewrite itself stays with the
    caller: unmigrated writers (the loop composer's port versions, the fold's
    output-slot rebinding) still write it raw, so the dict is not yet
    materialized from the page alone.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        BINDING_VERSION_REMOVED, BindingFact, CANONICAL_VALUE, NAME_BINDING,
        PLANNER_STRUCTURAL_FOLD, RETURN_SITE_SLOT,
    )
    from .identity_concordance import (
        Derived, Mode, Ref, Unresolved, current_identity_book,
    )

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    scope = tuple(scope)
    removed = {int(node_id) for node_id in removed}
    aliases = {int(old): int(new) for old, new in (aliases or {}).items()}
    if not removed and not aliases:
        return
    stage = PLANNER_STRUCTURAL_FOLD if stage is None else stage
    book = current_identity_book()
    page = book.pages.get(NAME_BINDING.name)
    if page is None:
        return
    source_cells: dict[int, Any] = {}

    def source_cell(node_id: int) -> Any:
        if node_id not in source_cells:
            try:
                source_cells[node_id] = (
                    node_identity_cell(graph, node_id)
                    if node_id in graph.G else
                    book.latest_ref(CANONICAL_VALUE, (scope, node_id))
                )
            except ValueError:
                source_cells[node_id] = None
        return source_cells[node_id]

    for row in tuple(page.scope_rows(scope)):
        fact = page.latest(row)
        if not isinstance(fact, BindingFact):
            continue
        value_id = int(fact.value_id)
        previous = book.latest_ref(NAME_BINDING, row)
        if value_id in removed:
            cells = tuple(
                cell for cell in (previous, source_cell(value_id))
                if cell is not None
            )
            book.post(
                NAME_BINDING, row,
                Unresolved(BINDING_VERSION_REMOVED, read=cells),
                stage=stage, provenance=Derived(cells), mode=Mode.REVISE,
            )
        elif value_id in aliases:
            replacement = aliases[value_id]
            new_fact = BindingFact(
                replacement, fact.authored, fact.span_positions,
                fact.context_sha256,
            )
            cells = tuple(dict.fromkeys(
                cell for cell in (
                    previous, source_cell(replacement), *cause_cells,
                ) if cell is not None
            ))
            book.post(
                NAME_BINDING, row, new_fact, stage=stage,
                provenance=Derived(cells), mode=Mode.REVISE,
            )
    if aliases:
        slots = book.pages.get(RETURN_SITE_SLOT.name)
        for row in tuple(slots.scope_rows(scope)) if slots is not None else ():
            fact = slots.latest(row)
            if not isinstance(fact, Ref) or fact.page is not CANONICAL_VALUE:
                continue
            old_value = fact.row[1]
            if not isinstance(old_value, int) or old_value not in aliases:
                continue
            replacement_cell = source_cell(aliases[old_value])
            if replacement_cell is None:
                continue
            previous = book.latest_ref(RETURN_SITE_SLOT, row)
            cells = tuple(dict.fromkeys(
                cell for cell in (previous, replacement_cell, *cause_cells)
                if cell is not None
            ))
            book.post(
                RETURN_SITE_SLOT, row, replacement_cell, stage=stage,
                provenance=Derived(cells), mode=Mode.REVISE,
            )


def _post_copy_planner_specializations(
    specialized: Any, specializations: Mapping[str, Any],
    specialization_cells: Mapping[str, tuple],
) -> None:
    """The rows of one callee COPY: a literal argument of its callsite
    (``SpecializationFact(value, LITERAL)`` from the argument cells the
    caller collected) or the signature default of an omitted argument
    (``DEFAULT``, from the copy's own formal cells); then the dict is the
    page's read view."""

    from .concordance_declarations import (
        SpecializationFact, SpecializationSource,
    )

    for parameter, value in specializations.items():
        cells = tuple(specialization_cells.get(str(parameter)) or ())
        source = SpecializationSource.LITERAL
        if not cells:
            cells = _formal_identity_cells(specialized, str(parameter))
            source = SpecializationSource.DEFAULT
        _post_planner_specialization(
            specialized, str(parameter), SpecializationFact(value, source),
            cells,
        )
    _materialize_planner_specializations(specialized)


def _materialize_planner_specializations(graph: Any) -> None:
    """Rebuild ``G.graph["planner_specializations"]`` as the read view of
    the graph's ``planner_specialization`` rows (latest fact per formal,
    ``Unresolved`` skipped).  A key only an unmigrated writer set survives:
    the page's facts win where both speak."""

    from .concordance_declarations import (
        PLANNER_SPECIALIZATION_PAGE, SpecializationFact,
    )
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is None:
        return
    page = current_identity_book().pages.get(PLANNER_SPECIALIZATION_PAGE.name)
    if page is None:
        return
    view = dict(graph.G.graph.get("planner_specializations") or {})
    for row in page.scope_rows(tuple(scope)):
        fact = page.latest(row)
        if isinstance(fact, SpecializationFact):
            view[str(row[1])] = fact.value
    graph.G.graph["planner_specializations"] = view


def _callsite_descriptor_receipt(descriptor: Any) -> Any:
    """Stable concordance spelling for one structured return descriptor."""

    if isinstance(descriptor, Mapping):
        return (
            tuple(descriptor.get("shape") or ()),
            str(descriptor.get("dtype") or "unknown"),
        )
    if isinstance(descriptor, tuple):
        return tuple(_callsite_descriptor_receipt(item) for item in descriptor)
    return None


def _post_callsite_return_member(
    caller: Any, call_id: int, index: int, member_id: int, descriptor: Any,
    cells: tuple, *, stage: Any,
) -> Any:
    """Post one aggregate-call member's descriptor on ``callsite_return_member``.

    The row is keyed by the caller COPY's ``lexical_read_scope`` (copies of
    one function never share it), the call, the slot and the member value.
    Both writers of a member's shape post here -- the return publication
    from the callee return value's cells, the structural fold from the
    member's own shape cells -- so a disagreement between them is a REVISE
    with no changed source, which the book refuses, instead of an endless
    fixed point (scalar_loss_join: bw_mul members flipping () <-> (2, 3)).
    """

    from .concordance_declarations import CALLSITE_RETURN_MEMBER
    from .identity_concordance import (
        Mode, RAW_PRIMITIVE, Unsourced, current_identity_book,
    )

    scope = caller.G.graph.get("lexical_read_scope")
    scope = (
        tuple(scope) if scope is not None
        else ("function", str(caller.G.graph.get("function_name") or ""))
    )
    row = (scope, int(call_id), int(index), int(member_id))
    fact = _callsite_descriptor_receipt(descriptor)
    cells = tuple(cell for cell in cells if cell is not None)
    if cells:
        return _post_if_changed(
            CALLSITE_RETURN_MEMBER, row, fact, stage=stage, cells=cells,
        )
    book = current_identity_book()
    latest = book.latest_ref(CALLSITE_RETURN_MEMBER, row)
    if latest is not None and book.pages[
        CALLSITE_RETURN_MEMBER.name
    ].latest(row) == fact:
        return latest
    return book.post(
        CALLSITE_RETURN_MEMBER, row, fact, stage=stage,
        provenance=Unsourced(RAW_PRIMITIVE), mode=Mode.REVISE,
    )

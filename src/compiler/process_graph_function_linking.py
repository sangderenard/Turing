"""Link already-authored ProcessGraph functions into a source graph.

This is the source-neutral function-table operation needed when different
front ends meet at ProcessGraph.  In particular, a Python control program may
call a numerical function authored by SymPy without rendering that numerical
function back into Python or recapturing it through a tape.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GraphFunctionSocketBinding:
    """One exact unresolved function class spliced from one graph body."""

    function: Any
    graph: Any
    call_node_ids: tuple[int, ...]
    inserted_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProcessGraphSplice:
    """One body graph copied into a host graph at one context.

    ``value_ids`` maps every body value the host now reads (operands,
    constants and copied nodes) to its host value id.  ``function_references``
    maps each body function-table address the copied nodes call to the
    address the host table gives that same function.
    """

    body: Any
    context_cells: tuple[Any, ...]
    value_ids: Mapping[int, int]
    inserted_node_ids: tuple[int, ...]
    function_references: Mapping[int, int]


#: Node attributes that hold a function-table address of the graph's table.
_FUNCTION_ADDRESS_ATTRIBUTES = ("callee_ref", "method_ref", "constructor_ref")


def _addresses_called(graph: Any, node_ids: Any = None) -> tuple[int, ...]:
    """Function-table addresses named by ``graph``'s nodes, in node order."""

    selected = graph.G if node_ids is None else node_ids
    found: dict[int, None] = {}
    for node_id in selected:
        attributes = graph.G.nodes[node_id].get("attributes") or {}
        for key in _FUNCTION_ADDRESS_ATTRIBUTES:
            value = attributes.get(key)
            if value is not None:
                found[int(value)] = None
    return tuple(found)


def _function_closure(table: Any, roots: Any) -> tuple[int, ...]:
    """``roots`` and every address their graph bodies call, transitively."""

    ordered: dict[int, None] = {}
    pending = list(dict.fromkeys(int(address) for address in roots))
    while pending:
        address = pending.pop(0)
        if address in ordered:
            continue
        ordered[address] = None
        entry = table.entry(address)
        if entry.graph is not None:
            pending.extend(
                callee for callee in _addresses_called(entry.graph)
                if callee not in ordered
            )
    return tuple(ordered)


def _call_resolution(graph: Any, node_id: int) -> str:
    """The ``call_binding`` resolution recorded for ``node_id``, or the
    planner's call-site resolution when its graph has no row for it."""

    from .concordance_declarations import CALL_BINDING
    from .identity_concordance import current_identity_book

    scope = graph.G.graph.get("lexical_read_scope")
    if scope is not None:
        page = current_identity_book().pages.get(CALL_BINDING.name)
        recorded = (
            page.latest((tuple(scope), int(node_id))) if page is not None else None
        )
        resolution = getattr(recorded, "resolution", None)
        if resolution is not None:
            return str(getattr(resolution, "value", resolution))
    return "callsite_shell"


def rebind_function_references(graph: Any, mapping: Mapping[int, int]) -> None:
    """Point ``graph``'s calls at the addresses ``mapping`` gives them.

    Each rebound call keeps its resolution; its ``call_binding`` row is
    revised to the new address by the planner's own writer.
    """

    from .planner_concordance_posts import _post_call_binding

    for node_id, data in graph.G.nodes(data=True):
        attributes = data.get("attributes") or {}
        for key in _FUNCTION_ADDRESS_ATTRIBUTES:
            value = attributes.get(key)
            if value is None or int(value) not in mapping:
                continue
            rebound = int(mapping[int(value)])
            if rebound == int(value):
                continue
            resolution = _call_resolution(graph, int(node_id))
            attributes[key] = rebound
            extra = data.get("extra_args")
            if isinstance(extra, dict) and key in extra:
                extra[key] = rebound
            _post_call_binding(graph, int(node_id), rebound, resolution)


def adopt_function_references(
    host_table: Any, body_table: Any, addresses: Any,
) -> dict[int, int]:
    """Declare the functions ``addresses`` reach in ``host_table``.

    Each body function keeps its graph and its authored identity; only its
    table-local address changes.  The host declaration derives from the body
    declaration's ``function_address`` cell.  The adopted graphs then point at
    the host table, and every call inside them is rebound to the host address
    (its ``call_binding`` row revised by the same writer the planner uses).

    The body table must own its graphs: the reverse builders and
    ``fuse_forward_loss_backward`` deep-copy the rule table for exactly this
    reason.  A graph already adopted into a different table is refused, and
    so is a host function of the same qualified name with a different
    authored identity.
    """

    from .concordance_declarations import FUNCTION_ADDRESS
    from .identity_concordance import current_identity_book
    from ..transmogrifier.function_table import (
        FunctionResolutionState, function_address_identity,
    )

    closure = _function_closure(body_table, addresses)
    if host_table is body_table:
        return {address: address for address in closure}
    book = current_identity_book()
    mapping: dict[int, int] = {}
    for address in closure:
        entry = body_table.entry(address)
        identity = function_address_identity(entry)
        declaration = book.latest_ref(FUNCTION_ADDRESS, (identity,))
        if declaration is None:
            raise ValueError(
                f"function {entry.qualified_name!r} has no function_address "
                "row in this identity book; it cannot be adopted by another "
                "table"
            )
        incumbent_reference = host_table._qualified.get(str(entry.qualified_name))
        if incumbent_reference is not None:
            incumbent = host_table.entry(incumbent_reference)
            incumbent_identity = (
                incumbent.graph.G.graph.get("authored_function_identity")
                if incumbent.graph is not None else None
            )
            body_identity = (
                entry.graph.G.graph.get("authored_function_identity")
                if entry.graph is not None else None
            )
            if incumbent.graph is not None and incumbent_identity != body_identity:
                raise ValueError(
                    f"function {entry.qualified_name!r} is already declared in "
                    f"the host table as {incumbent_identity!r}; the spliced "
                    f"body calls {body_identity!r}"
                )
        metadata = {
            key: value for key, value in dict(entry.metadata).items()
            if key != "function_address_identity"
        }
        reference = host_table.declare(
            entry.name,
            qualified_name=entry.qualified_name,
            external=entry.state == FunctionResolutionState.EXTERNAL,
            metadata=metadata,
            parameter_contracts=entry.parameter_contracts,
            source=(declaration,),
        )
        mapping[address] = int(reference.address)
    for address in closure:
        entry = body_table.entry(address)
        target = host_table.entry(mapping[address])
        graph = entry.graph
        if graph is not None:
            owner = getattr(graph, "function_table", None)
            if owner is not body_table and owner is not host_table:
                raise ValueError(
                    f"function graph {entry.qualified_name!r} belongs to a "
                    "third function table; only its own table can hand it on"
                )
            if target.graph is None:
                host_table.resolve_graph(mapping[address], graph)
            if target.graph is graph and owner is body_table:
                graph.function_table = host_table
                graph.G.graph["function_ref"] = int(mapping[address])
                rebind_function_references(graph, mapping)
                _revise_call_result_addresses(graph, host_table, mapping)
        if entry.python_callable is not None and target.python_callable is None:
            host_table.resolve_callable(mapping[address], entry.python_callable)
        for name, implementation in entry.implementations.items():
            target.implementations.setdefault(name, implementation)
    return mapping


def _revise_call_result_addresses(
    graph: Any, host_table: Any, mapping: Mapping[int, int],
) -> None:
    """Revise the adopted ``graph``'s call-result identities to host addresses.

    Program: a masked training loss ``(e*e*mask).sum() / (mask.sum()*3.0)``
    links ``bw_div`` (``ys = y + eps(); ...``).  The rule table was reduced
    first and posted ``source_call_result_identity`` row (bw_div scope, call
    ``eps()``) with ``eps``'s rule-table address 8.  Adoption declares ``eps``
    in the host table at address 7 and moves the graph; the host's reduction
    of the same row then proposed 7, and the reducer raised a disagreement.
    One callee, two table-local numbers.

    Rule: adoption is the renumbering.  As ``rebind_function_references``
    revises each moved call's ``call_binding``, each result-identity row of the
    moved graph whose callee address adoption changed is revised to the host
    address, DERIVED from the incumbent cell and the host's
    ``function_address`` cell.
    """

    from ..common.tensors.topological_reducer import _source_numeric_scope
    from ..transmogrifier.function_table import function_address_identity
    from .concordance_declarations import (
        FUNCTION_ADDRESS, REDUCTION, SOURCE_CALL_RESULT_IDENTITY,
    )
    from .identity_concordance import Derived, Mode, current_identity_book

    book = current_identity_book()
    page = book.pages.get(SOURCE_CALL_RESULT_IDENTITY.name)
    if page is None:
        return
    scope = _source_numeric_scope(graph)
    for row in tuple(page.rows()):
        if row[0] != scope:
            continue
        cell = book.latest_ref(SOURCE_CALL_RESULT_IDENTITY, row)
        if cell is None:
            continue
        fact = tuple(page.cell(cell.row, cell.column))
        old = int(fact[0])
        new = int(mapping.get(old, old))
        if new == old:
            continue
        address = book.latest_ref(
            FUNCTION_ADDRESS,
            (function_address_identity(host_table.entry(new)),),
        )
        book.post(
            SOURCE_CALL_RESULT_IDENTITY, row, (new, *fact[1:]),
            stage=REDUCTION,
            provenance=Derived(tuple(
                c for c in (cell, address) if c is not None
            )),
            mode=Mode.REVISE,
        )


def _copy_argument_role_rows(
    body: Any, body_id: int, host: Any, host_id: int, stage: Any,
) -> None:
    """Re-post a copied Call's ``callsite_argument_role`` rows in the host,
    each DERIVED from the body row it copies."""

    from .concordance_declarations import CALLSITE_ARGUMENT_ROLE
    from .identity_concordance import Derived, Mode, current_identity_book

    body_scope = body.G.graph.get("ingestion_value_scope")
    host_scope = host.G.graph.get("ingestion_value_scope")
    book = current_identity_book()
    page = book.pages.get(CALLSITE_ARGUMENT_ROLE.name)
    if body_scope is None or host_scope is None or page is None:
        return
    index = 0
    while True:
        cell = book.latest_ref(
            CALLSITE_ARGUMENT_ROLE, (tuple(body_scope), int(body_id), index),
        )
        if cell is None:
            return
        book.post(
            CALLSITE_ARGUMENT_ROLE, (tuple(host_scope), int(host_id), index),
            page.latest(cell.row), stage=stage,
            provenance=Derived((cell,)), mode=Mode.CONCORD,
        )
        index += 1


def splice_process_graph(
    host: Any,
    body: Any,
    *,
    operands: Mapping[int, int],
    outputs: Any,
    context_cells: Any,
    constants: Mapping[int, Any] | None = None,
) -> ProcessGraphSplice:
    """Copy the part of ``body`` that computes ``outputs`` into ``host``.

    ``operands`` maps body values to the host values that stand for them
    (call arguments, saved forward values): those body nodes are not copied.
    ``constants`` maps body values to the literal a host constant replaces
    them with (a unit gradient seed).  Every other required body node gets a
    fresh host id, an ``ingestion_value`` row NOVEL(GRAPH_FUNCTION_SPLICE,
    (context, body value)), its operand rows through ``_set_operands`` and,
    for a Call, its ``callsite_argument_role`` rows.  The functions the
    copied nodes call are adopted by the host's function table.

    ``context_cells`` is the identity the splice happens at: the call being
    replaced, the Derivative being resolved, the request that linked the
    graph.  The body graph is not rewritten; its function graphs move to the
    host table (``adopt_function_references``).
    """

    import networkx as nx

    from ..common.tensors.topological_reducer import (
        _set_operands, node_identity_cell,
    )
    from .concordance_declarations import (
        GRAPH_FUNCTION_SPLICE, GRAPH_FUNCTION_SPLICE_STAGE, INGESTION_VALUE,
        NodeFact,
    )
    from .identity_concordance import Mode, Novel, current_identity_book
    from .process_graph_value_ids import next_process_value_id

    context = tuple(cell for cell in tuple(context_cells) if cell is not None)
    if not context:
        raise ValueError("a graph splice needs the identity cell it happens at")
    if len(context) != 1:
        raise ValueError(
            "a graph splice derives from exactly one context cell; "
            f"got {len(context)}"
        )
    scope = host.G.graph.get("ingestion_value_scope")
    if scope is None:
        raise ValueError("graph splice host has no ingestion identity scope")
    output_ids = tuple(int(value) for value in outputs)
    missing = tuple(value for value in output_ids if value not in body.G)
    if missing:
        raise ValueError(f"graph splice outputs are not body values: {missing!r}")
    remap: dict[int, int] = {int(k): int(v) for k, v in operands.items()}
    absent = tuple(v for v in remap.values() if v not in host.G)
    if absent:
        raise ValueError(f"graph splice operands are not host values: {absent!r}")
    literal_values = {int(k): v for k, v in dict(constants or {}).items()}
    required: set[int] = set()
    for output_id in output_ids:
        required |= {int(n) for n in nx.ancestors(body.G, output_id)} | {output_id}
    copied = tuple(
        int(node_id) for node_id in nx.topological_sort(body.G.subgraph(required))
        if int(node_id) not in remap
    )
    host_table = getattr(host, "function_table", None)
    called = _addresses_called(body, copied)
    references = (
        adopt_function_references(host_table, body.function_table, called)
        if called else {}
    )
    book = current_identity_book()
    from .symbolic_process_graph import (
        law_value_axis_free, post_law_value_axes, resident_identity_nodes,
    )

    resident = resident_identity_nodes(host)
    inserted: list[int] = []
    for body_id in copied:
        source = body.G.nodes[body_id]
        data = copy.deepcopy(dict(source))
        attributes = dict(data.get("attributes") or {})
        if body_id in literal_values:
            value = literal_values[body_id]
            data.update(type="const", op="const", label=repr(value), constant=value)
            attributes = {"value": value}
            parents: list[tuple[int, str]] = []
        else:
            parents = []
            for parent_id, role in source.get("parents") or ():
                if int(parent_id) not in remap:
                    raise ValueError(
                        f"graph splice body value {body_id} depends on "
                        f"unmapped value {int(parent_id)}"
                    )
                parents.append((remap[int(parent_id)], str(role)))
        for key in _FUNCTION_ADDRESS_ATTRIBUTES:
            if attributes.get(key) is not None:
                attributes[key] = int(references[int(attributes[key])])
        attributes["graph_splice_body_value"] = int(body_id)
        new_id = next_process_value_id(host)
        remap[body_id] = new_id
        data["parents"] = []
        data["children"] = []
        data["value_id"] = new_id
        data["attributes"] = attributes
        data["extra_args"] = copy.deepcopy(attributes)
        host.G.add_node(new_id, **data)
        host.node_map[new_id] = data.get("expr_obj")
        body_cell = node_identity_cell(body, body_id)
        book.post(
            INGESTION_VALUE, (tuple(scope), int(new_id)),
            NodeFact(
                str(data.get("type") or ""), str(data.get("op") or ""),
                str(data.get("label") or ""),
            ),
            stage=GRAPH_FUNCTION_SPLICE_STAGE,
            provenance=Novel(GRAPH_FUNCTION_SPLICE, (context[0], body_cell)),
            mode=Mode.CONCORD,
        )
        if parents:
            _set_operands(host, new_id, parents, cause=GRAPH_FUNCTION_SPLICE)
        _copy_argument_role_rows(
            body, body_id, host, new_id, GRAPH_FUNCTION_SPLICE_STAGE,
        )
        # The copy's ``law_value_axes`` row is derived again in the host,
        # from its host operand rows: ``-s`` in ``k**(-s)`` is axis-free;
        # ``k*x`` over a Sum's index axis is not (``post_law_value_axes``).
        resident[node_identity_cell(host, new_id)] = new_id
        if str(data.get("type") or data.get("op") or "").casefold() in {
            "const", "constant",
        }:
            # A copied literal is the body's literal in the host: its
            # ``source_literal_value`` row derives from the copy's cell and
            # the body literal's own cells (a unit seed from ``constants``
            # is its own literal, caused by the copy and the splice context).
            from ..common.tensors.topological_reducer import _source_numeric_scope
            from .concorded_literals import MISSING, literal
            from .concordance_declarations import SOURCE_LITERAL_VALUE
            from .identity_concordance import Derived as _LiteralDerived

            if body_id in literal_values:
                literal_value, literal_cells = literal_values[body_id], context
            else:
                literal_value, literal_cells = literal(body, body_id)
            if literal_value is not MISSING:
                book.post(
                    SOURCE_LITERAL_VALUE,
                    (_source_numeric_scope(host), int(new_id)), literal_value,
                    stage=GRAPH_FUNCTION_SPLICE_STAGE,
                    provenance=_LiteralDerived(tuple(dict.fromkeys((
                        node_identity_cell(host, new_id), *literal_cells,
                    )))),
                    mode=Mode.CONCORD,
                )
        post_law_value_axes(host, new_id, resident)
        # The body's scalar stamp is the body's own declaration.  A copy with
        # a declared reduction axis among its host causes is not one value
        # of the law (``t*x`` over the Integral's (5, 1) rule axis): its
        # operator law states its shape, as the envelope stamp rule does.
        if not law_value_axis_free(host, new_id):
            host.G.nodes[new_id].get("tensor", {}).pop("shape", None)
        inserted.append(new_id)
    return ProcessGraphSplice(
        body=body,
        context_cells=context,
        value_ids=dict(remap),
        inserted_node_ids=tuple(inserted),
        function_references=dict(references),
    )


def _replace_graph_metadata_value(graph: Any, old_id: int, new_id: int) -> None:
    """Move public graph references when a socket itself is an output."""

    graph.roots = [new_id if int(value) == old_id else value for value in graph.roots]
    metadata = graph.G.graph
    for key in ("deployment_outputs",):
        if key in metadata:
            metadata[key] = tuple(
                new_id if int(value) == old_id else value
                for value in metadata[key]
            )
    identity_table = metadata.get("identity_table")
    if isinstance(identity_table, dict):
        for name, values in tuple(identity_table.items()):
            identity_table[name] = tuple(
                new_id if int(value) == old_id else value for value in values
            )


def _post_socket_binding(
    envelope: Any,
    call_id: int,
    body: Any,
    function_identity: str,
    parameter_ids: tuple[int, ...],
    argument_ids: tuple[int, ...],
    body_output_id: int,
    replacement_id: int,
) -> None:
    """Record the exact parameter and result wiring of one substitution."""

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        GRAPH_FUNCTION_SOCKET,
        GRAPH_FUNCTION_SPLICE,
        GRAPH_FUNCTION_SPLICE_STAGE,
        GraphFunctionSocketFact,
    )
    from .identity_concordance import Mode, Novel, current_identity_book

    scope = envelope.G.graph.get("ingestion_value_scope")
    body_scope = body.G.graph.get("ingestion_value_scope")
    if scope is None or body_scope is None:
        raise ValueError("graph function splice requires identified envelope and body scopes")
    call_cell = node_identity_cell(envelope, call_id)
    output_cell = node_identity_cell(body, body_output_id)
    current_identity_book().post(
        GRAPH_FUNCTION_SOCKET,
        (scope, int(call_id)),
        GraphFunctionSocketFact(
            str(function_identity), tuple(body_scope), tuple(parameter_ids),
            tuple(argument_ids), int(body_output_id), int(replacement_id),
        ),
        stage=GRAPH_FUNCTION_SPLICE_STAGE,
        provenance=Novel(GRAPH_FUNCTION_SPLICE, (call_cell, output_cell)),
        mode=Mode.CONCORD,
    )


def _ordered_socket_arguments(data: Mapping[str, Any], arity: int) -> tuple[int, ...]:
    arguments: dict[int, int] = {}
    for parent_id, role in data.get("parents") or ():
        prefix, separator, index = str(role).partition(":")
        if prefix != "arg" or not separator or not index.isdigit():
            raise ValueError("graph function socket arguments must use arg:N roles")
        position = int(index)
        if position in arguments:
            raise ValueError("graph function socket has a duplicate argument position")
        arguments[position] = int(parent_id)
    if tuple(sorted(arguments)) != tuple(range(arity)):
        raise ValueError(
            f"graph function socket has argument positions {tuple(sorted(arguments))}; "
            f"expected {tuple(range(arity))}"
        )
    return tuple(arguments[index] for index in range(arity))


def _splice_call(
    envelope: Any, call_id: int, body: Any, function_identity: str,
) -> tuple[int, ...]:
    """Replace one unresolved Call with a copied body subgraph."""

    from ..common.tensors.topological_reducer import (
        _redirect_value, node_identity_cell,
    )

    parameters = tuple(body.G.graph.get("function_parameters") or ())
    outputs = tuple(body.G.graph.get("function_outputs") or ())
    if not parameters or len(outputs) != 1:
        raise ValueError("a graph function socket body needs parameters and one output")
    identity_table = body.G.graph.get("identity_table") or {}
    try:
        parameter_ids = tuple(int(identity_table[name][-1]) for name in parameters)
        output_id = int(identity_table[outputs[0]][-1])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("graph function socket body has an incomplete identity table") from error
    if output_id not in body.G:
        raise ValueError("graph function socket output is not present in its body graph")

    arguments = _ordered_socket_arguments(envelope.G.nodes[call_id], len(parameters))
    splice = splice_process_graph(
        envelope, body,
        operands=dict(zip(parameter_ids, arguments)),
        outputs=(output_id,),
        context_cells=(node_identity_cell(envelope, call_id),),
    )
    for new_id in splice.inserted_node_ids:
        data = envelope.G.nodes[new_id]
        for key in ("attributes", "extra_args"):
            record = data.get(key)
            if isinstance(record, dict):
                record.update(
                    graph_socket_call=int(call_id),
                    graph_socket_body_value=int(record["graph_splice_body_value"]),
                )
    inserted = list(splice.inserted_node_ids)
    remap = splice.value_ids

    replacement = remap[output_id]
    _post_socket_binding(
        envelope, call_id, body, function_identity,
        parameter_ids, arguments, output_id, replacement,
    )
    _replace_graph_metadata_value(envelope, call_id, replacement)
    _redirect_value(envelope, call_id, replacement)
    envelope.node_map.pop(call_id, None)
    return tuple(inserted)


def bind_process_graph_function_sockets(
    envelope: Any,
    sockets: Mapping[Any, Any],
) -> tuple[GraphFunctionSocketBinding, ...]:
    """Splice graph bodies into unresolved function applications by identity.

    The symbolic envelope retains the applied undefined function object on
    each Call node.  That object -- not its spelling -- selects the supplied
    body.  Each occurrence receives fresh value identities whose receipts name
    both that exact application and the exact body values copied into it.
    Unmentioned unresolved functions remain external declarations.

    ``envelope`` is changed in place.  The body is not converted back through
    source or rebuilt from its expression labels; its existing nodes and edges
    are copied directly, with its parameter nodes wired to the call operands.
    """

    import sympy
    from sympy.core.function import AppliedUndef, UndefinedFunction

    requested = tuple(sockets.items())
    if not requested:
        return ()
    # Program: tests/test_graph_function_splicing.py run as a file (and on
    # clean main).  Wrong record: ``compile_sympy_equations`` served the
    # envelope from its disk cache; its node identities were posted in the
    # book of the compile that filled the cache, so splicing into it refused
    # "node 4 ... has no identity row under (('ingestion:sympy', 0),)".
    # Rule: an envelope with a symbolic recipe whose rows are not in the
    # active book is first replayed into it (the linker's own
    # ``rebuild_symbolic_process_graph_in_current_book``), in place, since
    # the caller holds this object.
    from .symbolic_equation_compiler import rebuild_symbolic_process_graph_in_current_book

    rebuilt = rebuild_symbolic_process_graph_in_current_book(envelope)
    if rebuilt is not envelope:
        envelope.G = rebuilt.G
        envelope.roots = list(rebuilt.roots)
    keys = []
    for function, _body in requested:
        if not isinstance(function, UndefinedFunction):
            raise TypeError("graph function socket must be an undefined function class")
        keys.append(sympy.srepr(function))
    if len(set(keys)) != len(keys):
        raise ValueError("graph function socket declaration identities must be unique")

    bindings = []
    for (function, body), socket_key in zip(requested, keys):
        name = getattr(function, "__name__", None)
        if not name:
            raise TypeError("graph function socket must be an undefined function class")
        parameters = tuple(body.G.graph.get("function_parameters") or ())
        outputs = tuple(body.G.graph.get("function_outputs") or ())
        if not parameters or len(outputs) != 1:
            raise ValueError(
                f"graph function socket {name!r} needs parameters and exactly one output"
            )
        matched = []
        for node_id, data in envelope.G.nodes(data=True):
            expression = data.get("expr_obj")
            attributes = data.get("attributes") or {}
            if (
                not isinstance(expression, AppliedUndef)
                or attributes.get("external_function_identity") != socket_key
            ):
                continue
            matched.append(int(node_id))
        if not matched:
            raise ValueError(f"graph function socket {name!r} has no callsite in the envelope")
        inserted = tuple(
            node_id
            for call_id in matched
            for node_id in _splice_call(envelope, call_id, body, socket_key)
        )
        bindings.append(GraphFunctionSocketBinding(
            function, body, tuple(matched), inserted,
        ))
    envelope.G.graph["graph_function_sockets"] = tuple(
        {
            "function": getattr(binding.function, "__name__", str(binding.function)),
            "calls": binding.call_node_ids,
            "inserted": binding.inserted_node_ids,
        }
        for binding in bindings
    )
    # Program: tests/test_graph_function_splicing.py::
    # test_spliced_outer_equation_executes_through_separate_linked_graph.
    # Wrong record: a bound envelope linked into another identity book could
    # not be replayed ("topology changed after compilation"): its recipe held
    # only the equations.  Rule: a binding whose bodies are authored programs
    # (``math_program_recipe``) is part of the envelope's recipe, with the
    # bound topology; a body without one leaves the recipe unchanged, so the
    # replay still refuses rather than drop the edit.
    recipe = envelope.G.graph.get("symbolic_compilation_recipe")
    portable = all(
        isinstance(binding.graph.G.graph.get("math_program_recipe"), Mapping)
        for binding in bindings
    )
    if isinstance(recipe, Mapping) and portable:
        envelope.G.graph["symbolic_compilation_recipe"] = {
            **dict(recipe),
            "socket_bindings": (
                *tuple(recipe.get("socket_bindings") or ()),
                *(
                    {"function": binding.function,
                     "body": dict(binding.graph.G.graph["math_program_recipe"])}
                    for binding in bindings
                ),
            ),
            "node_count": len(envelope.G),
            "edge_count": envelope.G.number_of_edges(),
        }
    return tuple(bindings)


def admit_process_graph_function(graph: Any, *, stage: Any) -> Any:
    """Post the records an AST function gets from reduction, for a graph body.

    A ProcessGraph built without source (a SymPy law, a spliced reverse, a
    training graph) is never reduced, so the reducer never gives it a
    lexical read scope, a declared activity graph, or version-zero formal
    bindings.  Control IR (``ingest_function_control``), the activity
    schedule and call linking read all three.  Here the read scope is the
    graph's own ingestion scope (its node cells are ingestion cells; there
    is no canonical relabel), the activity graph is declared empty -- the
    body has no authored control, and ``ActivitySchedule`` then orders it by
    the graph's own scheduler -- and each formal binds the input node that
    stands for it.  Each row derives from the cells it states.

    Program: ``lower_ast_source_to_ssa(..., linked_process_graphs=
    {name: sympy_graph})``.  Wrong record: since control IR moved into the
    concordance (cf57b8c1) a linked graph reached ``ingest_function_control``
    with no ``lexical_read_scope`` and raised KeyError; the training graph
    only passed because its two builders posted these rows by hand.
    Returns the read scope.
    """

    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import (
        BindingFact, NAME_BINDING, SEMANTIC_OUTPUT, SOURCE_ACTIVITY_SCOPE,
    )
    from .identity_concordance import Derived, Mode, current_identity_book

    metadata = graph.G.graph
    scope = metadata.get("ingestion_value_scope")
    if scope is None:
        raise ValueError(
            "a graph-native function has no ingestion identity scope to admit"
        )
    scope = tuple(scope)
    declared = metadata.get("lexical_read_scope")
    if declared is not None and tuple(declared) != scope:
        # Already reduced or admitted under its own read scope.
        return tuple(declared)
    metadata["lexical_read_scope"] = scope
    # An AST function is identified by (module, qualname), and every
    # specialized copy keeps it; call reachability and pursuit activation are
    # keyed by it.  A graph body is identified by the ingestion scope that
    # minted its values and the name it is declared under.
    if metadata.get("authored_function_identity") is None:
        name = metadata.get("function_name")
        if not name:
            raise ValueError("a graph-native function has no declared name")
        metadata["authored_function_identity"] = (
            f"process_graph:{scope!r}", str(name),
        )
    book = current_identity_book()
    roots = tuple(int(root) for root in (getattr(graph, "roots", None) or ()))
    if not roots:
        identity_table = metadata.get("identity_table") or {}
        roots = tuple(
            int(identity_table[name][-1])
            for name in tuple(metadata.get("function_outputs") or ())
            if identity_table.get(name)
        )
    if not roots:
        raise ValueError("a graph-native function has no output to admit")
    if book.latest_ref(SOURCE_ACTIVITY_SCOPE, (scope,)) is None:
        # The graph's own operand edges are its evaluation order; the read
        # scope then declares that activity graph, as the reducer declares
        # an AST function's from its entry point.
        from .concordance_declarations import SOURCE_ACTIVITY_NODE
        from .source_control_ingestion import ingest_graph_activity

        activity = ingest_graph_activity(graph, scope, stage=stage)
        book.post(
            SOURCE_ACTIVITY_SCOPE, (scope,), activity, stage=stage,
            provenance=Derived(tuple(dict.fromkeys(
                book.latest_ref(
                    SOURCE_ACTIVITY_NODE,
                    (activity, node_identity_cell(graph, root), "exit"),
                )
                for root in roots
            ))),
            mode=Mode.CONCORD,
        )
    identity_table = metadata.get("identity_table") or {}
    # As the reducer's return writer does for an AST function: output
    # position -> (name, the value the identity table ends at for it).
    for position, output_name in enumerate(
        tuple(metadata.get("function_outputs") or ())
    ):
        history = identity_table.get(output_name)
        if not history:
            raise ValueError(
                f"graph-native output {output_name!r} has no value identity"
            )
        value_id = int(history[-1])
        book.post(
            SEMANTIC_OUTPUT, (scope, int(position)),
            (str(output_name), value_id), stage=stage,
            provenance=Derived((node_identity_cell(graph, value_id),)),
            mode=Mode.CONCORD,
        )
    for parameter in tuple(metadata.get("function_parameters") or ()):
        history = identity_table.get(parameter)
        if not history:
            raise ValueError(
                f"graph-native formal {parameter!r} has no input identity"
            )
        value_id = int(history[0])
        book.post(
            NAME_BINDING, (scope, str(parameter), 0),
            BindingFact(value_id, False, (None,) * 4, ""),
            stage=stage,
            provenance=Derived((node_identity_cell(graph, value_id),)),
            mode=Mode.CONCORD,
        )
    _admit_graph_tensor_classes(graph)
    _admit_graph_call_reachability(graph, stage=stage)
    return scope


def _admit_graph_call_reachability(graph: Any, *, stage: Any) -> None:
    """Post the reachability the reducer's pursuit posts for an AST body.

    Program: ``tests/test_llvm_training_runtime.py::
    test_native_parameter_groups_step_only_the_selected_state`` (training
    graph lowered by ``lower_training_motion_to_repository_ssa``, no source
    pursuit).  Wrong record: the training graph's ``Call bw_mean`` had no
    ``source_function_reachability`` row, because only
    ``propagate_call_formal_numeric_types`` writes them and it runs for
    reduced bodies; callsite planning refused "selected return has no exact
    source call reachability authority".  Rule: each call node of a graph
    body whose ``callee_ref`` names a table entry posts
    ``(caller scope, call, callee scope) -> True``, DERIVED from the call
    node's identity cell and the callee's ``function_address`` cell, as the
    reducer's writer does.  CONCORD with that writer when both run.
    """

    from ..common.tensors.topological_reducer import (
        _source_numeric_scope, node_identity_cell,
    )
    from ..transmogrifier.function_table import function_address_identity
    from .concordance_declarations import (
        FUNCTION_ADDRESS, SOURCE_FUNCTION_REACHABILITY,
    )
    from .identity_concordance import Derived, Mode, current_identity_book

    table = getattr(graph, "function_table", None)
    if table is None:
        return
    book = current_identity_book()
    caller_scope = _source_numeric_scope(graph)
    page = book.page(SOURCE_FUNCTION_REACHABILITY)
    for node_id, data in graph.G.nodes(data=True):
        reference = (data.get("attributes") or {}).get("callee_ref")
        if reference is None:
            continue
        entry = table.entry(int(reference))
        if entry.graph is None:
            continue
        callee_scope = _source_numeric_scope(entry.graph.G, entry.name)
        row = (caller_scope, int(node_id), callee_scope)
        incumbent = page.latest(row)
        if incumbent is not None:
            if incumbent is not True:
                raise ValueError(
                    "source function reachability concordance disagreement "
                    f"for {row!r}: recorded={incumbent!r}, proposed=True"
                )
            continue
        address = book.latest_ref(
            FUNCTION_ADDRESS, (function_address_identity(entry),),
        )
        if address is None:
            raise ValueError(
                f"graph call {int(node_id)} names table entry {entry.name!r} "
                "with no function_address cell"
            )
        book.post(
            SOURCE_FUNCTION_REACHABILITY, row, True, stage=stage,
            provenance=Derived((node_identity_cell(graph, int(node_id)), address)),
            mode=Mode.CONCORD,
        )


def _admit_graph_tensor_classes(graph: Any) -> None:
    """Post the class of every graph-native tensor value.

    A graph body's values are AbstractTensor values when the book says so:
    their ``graph_tensor_descriptor`` row (posted by the stamping pass,
    derived from the recorded program's ``abstract_tensor_program_value``
    cells and the parents' rows) is the evidence, as an authored annotation
    or class inference is for an AST value.  A value without that row gets
    no class here.

    Program: linear-MSE training linked into ``lower_ast_source_to_ssa``
    with ``x``/``weight``/``bias``/``target`` declared AbstractTensor spans.
    Wrong record: the training graph's own values (``mul``, ``sub`` results
    passed to ``bw_mul``) had no ``source_value_class`` row, so the linked
    contracts of ``bw_mul`` and ``unbroadcast`` carried storage/shape with no
    ``python_type``, no ``value_type_abi`` row was written for their formals,
    and a planned region's ``sum`` over ``unbroadcast``'s ``G`` had no proven
    span ("dynamic reduction lacks a typed source span").  Rule: each
    non-call value with a ``graph_tensor_descriptor`` row and no incumbent
    class is AbstractTensor, DERIVED from that row and its identity cell.  A
    call node is the callee's result and keeps the class its callee states.
    """

    from ..common.tensors.abstraction import AbstractTensor
    from ..common.tensors.topological_reducer import (
        _concord_source_value_class, _source_numeric_scope, node_identity_cell,
    )
    from .concordance_declarations import (
        GRAPH_TENSOR_DESCRIPTOR, SOURCE_VALUE_CLASS,
    )
    from .identity_concordance import Derived, current_identity_book

    book = current_identity_book()
    numeric_scope = _source_numeric_scope(graph)
    descriptor_scope = tuple(graph.G.graph.get("ingestion_value_scope") or ())
    tensor_class = f"{AbstractTensor.__module__}.{AbstractTensor.__qualname__}"
    for node_id, data in graph.G.nodes(data=True):
        if str(data.get("op") or data.get("type") or "").casefold() == "call":
            continue
        value_id = int(data.get("value_id", node_id))
        descriptor = book.latest_ref(
            GRAPH_TENSOR_DESCRIPTOR, (descriptor_scope, value_id),
        )
        if descriptor is None:
            continue
        if book.latest_ref(SOURCE_VALUE_CLASS, (numeric_scope, value_id)):
            continue
        _concord_source_value_class(
            numeric_scope, value_id, tensor_class,
            source="graph-native tensor value",
            provenance=Derived((
                descriptor, node_identity_cell(graph, int(node_id)),
            )),
        )


def _graphs(root: Any) -> tuple[Any, ...]:
    seen: set[int] = set()
    pending = [root]
    result = []
    while pending:
        graph = pending.pop()
        if id(graph) in seen:
            continue
        seen.add(id(graph))
        result.append(graph)
        table = getattr(graph, "function_table", None)
        if table is not None:
            pending.extend(
                entry.graph for entry in table if entry.graph is not None
            )
    return tuple(result)


def link_process_graph_functions(
    root: Any,
    functions: Mapping[str, Any],
) -> Mapping[str, int]:
    """Resolve named source calls directly to supplied ProcessGraph bodies."""

    table = root.function_table
    references: dict[str, int] = {}
    for name, callee in sorted(functions.items()):
        function_name = str(name)
        from .symbolic_equation_compiler import (
            rebuild_symbolic_process_graph_in_current_book,
        )

        # A recipe (a zero-argument callable such as
        # ``training_function_recipe``) builds its graph here, inside the
        # compile's identity book; it returns the same graph to the second
        # link pass of the same book.
        if callable(callee) and not hasattr(callee, "G"):
            callee = callee()
        callee = rebuild_symbolic_process_graph_in_current_book(callee)
        parameters = tuple(callee.G.graph.get("function_parameters") or ())
        if not parameters:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no parameters"
            )
        outputs = tuple(callee.G.graph.get("function_outputs") or ())
        if not outputs:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no outputs"
            )
        from ..transmogrifier.graph.graph_express2 import (
            resolve_source_definition,
        )

        source_definitions = tuple(
            callee.G.graph.get("function_source_definitions") or ()
        )
        # A generated graph (a training motion) has no authored source; its
        # definition cells are its roots' ingestion cells in this book
        # (``training_motion_function``).
        definition_cells = tuple(
            cell for cell in tuple(
                callee.G.graph.get("function_definition_cells") or ()
            ) if cell is not None
        )
        if source_definitions:
            source_cells = tuple(
                resolve_source_definition(definition)
                for definition in source_definitions
            )
        elif definition_cells:
            source_cells = definition_cells
        else:
            raise ValueError(
                f"linked ProcessGraph {function_name!r} has no authored "
                "function source definitions or definition cells"
            )
        reference = table.declare(
            function_name,
            qualified_name=function_name,
            metadata={
                "source_language": callee.G.graph.get("symbolic_source"),
                "linked_process_graph": True,
            },
            source=source_cells,
        )
        # The linked graph may call functions of its own table (a spliced
        # graph reverse calls its backward rules).  Those functions move to
        # the root table and the graph's calls are rebound before it adopts
        # that table; replacing the table alone left ``callee_ref`` naming
        # whatever the root table held at the same address.
        own_table = getattr(callee, "function_table", None)
        called = _addresses_called(callee)
        adopted = (
            adopt_function_references(table, own_table, called)
            if own_table is not None and own_table is not table and called
            else {}
        )
        callee.function_table = table
        callee.external_function_table = root.external_function_table
        if adopted:
            rebind_function_references(callee, adopted)
        callee.G.graph.update(
            function_ref=int(reference.address),
            function_name=function_name,
        )
        from .concordance_declarations import GRAPH_FUNCTION_SPLICE_STAGE

        admit_process_graph_function(callee, stage=GRAPH_FUNCTION_SPLICE_STAGE)
        table.resolve_graph(reference, callee)
        references[function_name] = int(reference.address)

    bound_callers: list[Any] = []
    for graph in _graphs(root):
        # Source lowering registers linked functions before it ingests the
        # caller.  In that phase the root has no callsites to inspect.  Avoid
        # materializing its empty node view: under the disk-backed graph
        # container that view is not published until ingestion begins.
        if graph is root and not graph.node_map:
            continue
        for node_id, data in tuple(graph.G.nodes(data=True)):
            if data.get("op") not in {"Call", "call"}:
                continue
            if (data.get("attributes") or {}).get(
                "callee_resolution"
            ) == "linked-process-graph":
                # Bound by an earlier pass of this idempotent link.
                bound_callers.append(graph)
            parents = tuple(data.get("parents") or ())
            matched = None
            for parent_id, role in parents:
                if str(role) != "callee" or parent_id not in graph.G:
                    continue
                parent = graph.G.nodes[parent_id]
                binding_name = str(
                    (parent.get("attributes") or {}).get("binding_name") or ""
                )
                if binding_name in references:
                    matched = (int(parent_id), binding_name)
                    break
            if matched is None:
                continue
            callee_id, binding_name = matched
            data["parents"] = [
                (parent_id, role)
                for parent_id, role in parents
                if int(parent_id) != callee_id
            ]
            data.setdefault("attributes", {}).update(
                callee_ref=references[binding_name],
                callee_resolution="linked-process-graph",
            )
            # The attributes are the read view of the ``call_binding`` row
            # (plan 80, A2.8): DERIVED from the call node's identity cell and
            # the linked callee's ``function_address`` cell.
            from .glsl_deployment_strategy import _post_call_binding

            _post_call_binding(
                graph, int(node_id), references[binding_name],
                "linked_process_graph",
            )
            if graph.G.has_edge(callee_id, node_id):
                graph.G.remove_edge(callee_id, node_id)
            children = graph.G.nodes[callee_id].get("children") or []
            graph.G.nodes[callee_id]["children"] = [
                child for child in children if int(child[0]) != int(node_id)
            ]
            if not graph.G.nodes[callee_id]["children"]:
                graph.G.remove_node(callee_id)
                graph.node_map.pop(callee_id, None)
            unresolved = tuple(graph.G.graph.get("unresolved_ast_calls") or ())
            graph.G.graph["unresolved_ast_calls"] = tuple(
                record for record in unresolved
                if str(record.get("name")) != binding_name
            )
            bound_callers.append(graph)
    # Program: ``def root(n, x: AbstractTensor): return fsum(n, x)`` over a
    # declared SymPy Sum whose result is a static (3,) tensor. Wrong record:
    # this link binds the source call after topology reduction, so the
    # reducer's pursuit never posted its ``source_function_reachability``
    # row, and the callsite refused the static return ("no exact source call
    # reachability authority"). Rule: a call bound here is a call of a table
    # entry like any other; its caller graph posts the same row the
    # graph-native admission posts for its own calls (CONCORD with the
    # reducer's writer when both run).
    from .concordance_declarations import GRAPH_FUNCTION_SPLICE_STAGE

    # A call may reach a linked graph through this pass or through the
    # reducer's own name resolution of the registered table entry; either
    # way its callee_ref names a linked address.
    linked_addresses = set(references.values())
    for graph in _graphs(root):
        if graph is root and not graph.node_map:
            continue
        if any(
            (data.get("attributes") or {}).get("callee_ref") in linked_addresses
            for _node_id, data in graph.G.nodes(data=True)
        ):
            bound_callers.append(graph)
    for graph in {id(graph): graph for graph in bound_callers}.values():
        _admit_graph_call_reachability(graph, stage=GRAPH_FUNCTION_SPLICE_STAGE)
    return references


__all__ = [
    "ProcessGraphSplice",
    "adopt_function_references",
    "admit_process_graph_function",
    "rebind_function_references",
    "splice_process_graph",
    "GraphFunctionSocketBinding",
    "bind_process_graph_function_sockets",
    "link_process_graph_functions",
]

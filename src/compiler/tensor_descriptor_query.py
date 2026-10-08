"""Tensor descriptor query: the per-copy, book-memoised descriptor resolver (_tensor_descriptor and its rule), its operation tables, polymorphic-formal scan and the descriptor-dependent invalidation."""

from __future__ import annotations

import ast
import copy
import os
import sys
import numpy as np
from typing import Any, Mapping

from .source_node_facts import _constant_value
from .call_argument_binding import _authored_aggregate_leaves
from .region_scheduling import _dependency_levels
from .formal_shape_ledger import _proven_formal_shape


def _authored_aggregate_descriptor_tree(graph: Any, value_id: int) -> Any:
    """Return the exact nested descriptor tree of an authored aggregate.

    Aggregate identity is recursive: a tuple member may itself be a tuple.
    Flattening the outer ledger to one descriptor per member discarded the
    coefficient-limb rows in ``((c00, c01), ..., (c30, c31))``.  Preserve the
    authored tree; leaves remain ordinary tensor descriptors (or ``None``
    when no shape has yet been proved).
    """

    value_id = int(value_id)
    leaves = _authored_aggregate_leaves(graph, value_id)
    if leaves:
        return tuple(
            _authored_aggregate_descriptor_tree(graph, int(leaf))
            for leaf in leaves
        )
    descriptor = _tensor_descriptor(graph, value_id)
    return (
        None if descriptor is None
        else copy.deepcopy(dict(descriptor))
    )


def _invalidate_tensor_descriptor_dependents(
    graph: Any, source_ids: Any, reason: str,
) -> None:
    """Invalidate every cached descriptor derived from changed exact values."""

    sources = {int(source_id) for source_id in source_ids}
    affected = set(sources)
    changed = True
    while changed:
        changed = False
        for node_id, data in graph.G.nodes(data=True):
            node_id = int(node_id)
            if node_id in affected:
                continue
            parents = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) not in {"callee", "func", "definition"}
            )
            causes = tuple(parent for parent in parents if parent in affected)
            if not causes:
                continue
            affected.add(node_id)
            data.pop("tensor", None)
            try:
                from .identity_concordance import (
                    invalidate_proven_shape,
                    shape_scope_of,
                )

                invalidate_proven_shape(
                    shape_scope_of(graph),
                    int(data.get("value_id", node_id)),
                    int(causes[0]),
                    str(reason),
                )
            except Exception:
                pass
            changed = True


def _structured_output_descriptor(graph: Any, value_id: int) -> Any:
    """A tensor descriptor, or a nested tuple of them for an aggregate.

    ``return total, (hub, angle), valid`` publishes ``(hub, angle)`` as one
    output whose members are two tensors.  The structure is the exact leaf
    ledger; nothing is inferred from shapes.
    """

    value_id = int(value_id)
    if value_id not in graph.G:
        return None
    attributes = graph.G.nodes[value_id].get("attributes") or {}
    leaves = tuple(map(int, attributes.get("aggregate_leaf_value_ids", ())))
    if attributes.get("producer_kind") == "aggregate" and leaves:
        return tuple(
            _structured_output_descriptor(graph, leaf) for leaf in leaves
        )
    descriptor = _tensor_descriptor(graph, value_id)
    from .identity_concordance import (
        Unresolved as _Unresolved,
        authored_function_name,
        current_identity_book,
        proven_shape_contract_of,
        shape_scope_of,
    )

    # The shared proof is keyed by the AUTHORED name, so every callsite copy
    # of one function reads the same row.  When two call edges disagree on a
    # formal's shape (FORMAL_SHAPE_CONFLICT), the function is polymorphic and
    # that row is some other copy's answer: ``unbroadcast`` copies with
    # G = () (bw_add) read ('unbroadcast', 1) = (2, 3), cemented by the
    # G = (2, 3) copies, and published bw_add's scalar gradients as (2, 3)
    # (scalar_loss_join: NaN gradients).  A polymorphic copy answers with its
    # own descriptor -- the same rule ``_tensor_descriptor`` applies.
    if graph.G.graph.get("planner_tensor_descriptors"):
        owner = authored_function_name(graph.G.graph.get("function_name"))
        formal_page = current_identity_book().page("formal_shape")
        if _owner_has_polymorphic_formal(formal_page, owner):
            return descriptor
    proof = proven_shape_contract_of(
        shape_scope_of(graph),
        int(graph.G.nodes[value_id].get("value_id", value_id)),
    )
    if proof is None:
        return descriptor
    shape, dtype = proof
    # Call-result specialization is a read of the callee's final value
    # identity.  The shared proof for that exact identity is later and more
    # authoritative than a provisional descriptor left on an intermediate
    # reduction node.  Conflicting specializations answer ``None`` above, so
    # this cannot collapse a genuinely polymorphic return.
    return {
        **dict(descriptor or {}),
        "shape": tuple(shape),
        "dtype": str(dtype),
        "rank": len(shape),
    }


def _operand_descriptor(graph: Any, value_id: int) -> dict[str, Any] | None:
    """A tensor descriptor, treating an authored scalar literal as rank 0."""

    descriptor = _tensor_descriptor(graph, int(value_id))
    if descriptor is not None:
        return descriptor
    node = graph.G.nodes.get(int(value_id), {})
    if str(node.get("type")) not in {"Constant", "Const", "const"}:
        return None
    literal = node.get("constant")
    if literal is None:
        literal = (node.get("attributes") or {}).get("value")
    if isinstance(literal, bool):
        return {"shape": (), "dtype": "bool"}
    if isinstance(literal, int):
        return {"shape": (), "dtype": "int64"}
    if isinstance(literal, float):
        return {"shape": (), "dtype": "float64"}
    return None


_DTYPE_CAST_OPERATIONS = frozenset({
    "to_dtype", "astype", "cast", "type_as", "to", "float", "double",
    "int", "long", "bool_", "tensor",
})


# Elementwise operations whose result shape is the broadcast of its operands.
# ``matmul`` is deliberately absent: it contracts rather than broadcasts and
# has its own rule where it is lowered.
def _attribute_axis(data: Any) -> int | None:
    """The axis an operation names in its own attributes, when it names one."""

    attributes = data.get("attributes") or {}
    for key in ("dim", "axis", "dims", "axes"):
        value = attributes.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
    return None


# Spellings are the CATALOG's, not Python's: a comparison arrives as
# ``equal`` / ``greater_equal``, never ``eq`` / ``ge``.  Guessing the names
# here meant the rule silently never fired on any comparison, which is what
# builds every mask in axis-oriented source.
_ELEMENTWISE_BINARY_OPERATIONS = frozenset({
    "add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "eq", "ne", "lt", "le", "gt", "ge",
    "maximum", "minimum", "clamp_min", "clamp_max",
    "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor",
})


def _deepest_fact(page: Any, row: Any) -> Any:
    """The fact recorded at the deepest causal position for this row."""

    recorded = page.history(row)
    if not recorded:
        return None
    return max(recorded, key=lambda entry: int(entry[0]))[1]


def descriptor_states_a_shape(descriptor: Any) -> bool:
    """True only when a tensor descriptor actually states its extents.

    ``_tensor_descriptor`` answers every query, so its answer must be read
    for what it says.  Two of its answers state nothing:

    * ``rank`` larger than the extents listed -- a flat span whose extent is
      still unknown, which is not the rank-0 scalar ``shape=()`` reads as;
    * an empty shape with an unknown dtype -- what the query returns when
      recovery stops, as it does on a loop-carried result.

    An empty shape WITH a known dtype is a genuine rank-0 scalar and is a
    fact.  Treating the other two as facts retyped a matrix as one number at
    a call edge and folded ``get_shape()`` to ``()`` inside the callee.
    """

    if not isinstance(descriptor, Mapping):
        return False
    shape = tuple(descriptor.get("shape") or ())
    if int(descriptor.get("rank", len(shape))) != len(shape):
        return False
    return bool(shape) or str(
        descriptor.get("dtype") or "unknown"
    ) != "unknown"


#: The dtype a ``range`` loop target carries.  Python's ``range`` yields
#: ``int``; the loop composer's carried index is the SSA ``int`` scalar.
_ARITHMETIC_LOOP_TARGET_DTYPE = "int"


def _arithmetic_loop_target_owner(graph: Any, node_id: int) -> int | None:
    """The ``For`` node whose single arithmetic-sequence (``range``) target
    Input ``node_id`` is, else None.  The target's integer dtype is that
    loop's declaration (``iterator_kind``), read here by attribute."""

    data = graph.G.nodes.get(int(node_id)) or {}
    if data.get("type") != "Input" or (
        data.get("attributes") or {}
    ).get("binding_kind") != "loop":
        return None
    target_value = int(data.get("value_id", node_id))
    for loop_id, loop_data in graph.G.nodes(data=True):
        attributes = loop_data.get("attributes") or {}
        if attributes.get("iterator_kind") != "arithmetic_sequence":
            continue
        targets = [
            int(value) for value in dict(
                attributes.get("loop_target_bindings") or {}
            ).values()
        ]
        if len(targets) == 1 and targets[0] in {int(node_id), target_value}:
            return int(loop_id)
    return None


#: ``id(formal_shape page)`` -> (weak reference to that page, {authored owner
#: -> (cell count at the scan, answer)}).  See ``_owner_has_polymorphic_formal``.
_POLYMORPHIC_FORMAL_SCANS: dict[int, tuple[Any, dict[str, tuple[int, bool]]]] = {}


def _owner_has_polymorphic_formal(formal_page: Any, authored_owner: str) -> bool:
    """Whether any ``formal_shape`` row of ``authored_owner`` is
    ``Unresolved(FORMAL_SHAPE_CONFLICT)`` -- two call edges disagreed on a
    formal, so the function has no single descriptor graph.

    This is the scan ``_tensor_descriptor`` and ``_structured_output_descriptor``
    made on every outermost query (28% of the orbital profile), memoized on
    the page's content: every writer of ``formal_shape`` is a ``book.post``
    (``_publish_formal_shape``, mode REVISE: a new column per revision) or a
    scope fork's copy of a row (a new row), so the page's cell count only
    grows and is a version of its content; a hit at the same count is the
    same scan over the same cells.  Rows are keyed by the authored owner
    (``_publish_formal_shape`` writes ``authored_function_name``), so the
    owner's ``scope_rows`` are exactly the rows the filter admits.
    """

    import weakref

    from .identity_concordance import Unresolved as _Unresolved
    from .identity_concordance import authored_function_name

    key = id(formal_page)
    entry = _POLYMORPHIC_FORMAL_SCANS.get(key)
    if entry is None or entry[0]() is not formal_page:
        entry = (weakref.ref(formal_page), {})
        _POLYMORPHIC_FORMAL_SCANS[key] = entry
    version = len(formal_page.cells)
    held = entry[1].get(authored_owner)
    if held is not None and held[0] == version:
        return held[1]
    answer = any(
        isinstance(formal_row, tuple)
        and len(formal_row) >= 2
        and authored_function_name(formal_row[0]) == authored_owner
        and isinstance(formal_page.latest(formal_row), _Unresolved)
        for formal_row in formal_page.scope_rows(authored_owner)
    )
    entry[1][authored_owner] = (version, answer)
    return answer


class _DescriptorQuery(set):
    """The ``_seen`` path set of ONE outermost descriptor query.

    The graph is read, never mutated, for the whole of an outermost query, so
    a node whose descriptor the query already derived has the same answer
    when the query reaches it again by another path (a diamond).  This set
    is created by that outermost query, handed down the recursion, and gone
    when it returns: it holds the call's own working answers and nothing
    persists past it.  Every answer is still recorded on the book where it
    is derived; the book remains the only thing that outlives the query.

    An answer is held only when no cycle guard fired while it was derived
    (``cuts`` did not move), because a guarded answer depends on the path
    that reached it, not on the node.
    """

    __slots__ = ("answers", "cut_counter", "polymorphic")

    def __init__(self, members: Any = (), *, answers: Any = None,
                 cut_counter: Any = None, polymorphic: Any = None) -> None:
        super().__init__(members)
        self.answers = {} if answers is None else answers
        self.cut_counter = [0] if cut_counter is None else cut_counter
        #: One-element holder shared by every branch of the query.
        self.polymorphic = [None] if polymorphic is None else polymorphic

    def branch(self) -> "_DescriptorQuery":
        """This path, extended by the caller: same query, own path set."""
        return _DescriptorQuery(
            self, answers=self.answers, cut_counter=self.cut_counter,
            polymorphic=self.polymorphic,
        )

    def cut(self) -> None:
        """A cycle guard fired: answers derived across it are path-bound."""
        self.cut_counter[0] += 1


def _tensor_descriptor(
    graph: Any, node_id: int, _seen: set[int] | None = None,
) -> dict[str, Any] | None:
    """The compiler-owned shape query, with its answer recorded."""

    from .identity_concordance import Unresolved as _Unresolved
    from .identity_concordance import publish_program_abi_graph_identities

    # Published once per query: the recursion (``_seen`` carried down) asks
    # the same graph, which the publication already covers.  Its receipt
    # check costs O(nodes), and a 224k-node dt system asks this for every
    # node at every recursion depth.
    if _seen is None:
        publish_program_abi_graph_identities(graph.G)
        _seen = _DescriptorQuery()
    query = _seen if isinstance(_seen, _DescriptorQuery) else None
    if query is not None:
        held = query.answers.get(int(node_id), query)
        if held is not query:
            return None if held is None else dict(held)
        cuts_before = query.cut_counter[0]

    row = None
    page = None
    try:
        # The post-query block imports the same names locally, which makes
        # them function locals throughout; bind them here, before first use.
        from .identity_concordance import (
            authored_function_name,
            concordant_shape_transformation_state,
            current_identity_book,
            descriptor_from_shape_transformation_state,
            shape_scope_of,
        )

        data = graph.G.nodes[int(node_id)] if int(node_id) in graph.G else {}
        collection_result = bool(
            str(data.get("type") or data.get("op") or "").casefold()
            == "loopresult"
            and str((data.get("attributes") or {}).get("result_kind"))
            == "collection"
        )
        page = current_identity_book().page("proven_shape")
        row = (shape_scope_of(graph), int(data.get("value_id", node_id)))
        from .identity_concordance import proven_shape_of

        concorded_state = concordant_shape_transformation_state(
            row[0], row[1],
        )
        concorded_descriptor = descriptor_from_shape_transformation_state(
            concorded_state
        )
        settled = proven_shape_of(row[0], row[1])
        descriptor_operation = str(
            (data.get("attributes") or {}).get("tensor_operation")
            or (data.get("attributes") or {}).get("tensor")
            or (data.get("attributes") or {}).get("tensor_candidate")
            or data.get("op") or data.get("type") or ""
        ).casefold()
        specialized_operator = bool(
            graph.G.graph.get("planner_tensor_descriptors")
            and descriptor_operation in (
                {
                    "matmul", "dot", "sum", "mean", "prod", "min", "max",
                    "any", "all", "argmin", "argmax",
                    "zeros", "ones", "empty", "full",
                    # Shape-preserving operators whose catalogue descriptor
                    # (the graph domain's scalar default) predates this
                    # callsite's operands.
                    "cast_like", "cumsum", "cumprod",
                }
                | _ELEMENTWISE_BINARY_OPERATIONS
            )
        )
        binding_name = str(
            (data.get("attributes") or {}).get("binding_name") or ""
        )
        localized_formal = bool(
            str(data.get("type") or "") == "Input"
            and binding_name
            and binding_name in (
                graph.G.graph.get("planner_tensor_descriptors") or {}
            )
        )
        authored_owner = authored_function_name(
            graph.G.graph.get("function_name")
        )
        formal_conflict = False
        if str(data.get("type") or "") == "Input" and binding_name:
            formal_page = current_identity_book().page("formal_shape")
            formal_fact = formal_page.latest((
                authored_function_name(
                    graph.G.graph.get("function_name")
                ),
                binding_name,
            ))
            # The conflicting fact is ``Unresolved(FORMAL_SHAPE_CONFLICT)``.
            formal_conflict = isinstance(formal_fact, _Unresolved)
        # Once two exact call edges disagree on any formal, this authored
        # function has no single descriptor graph.  Every intermediate in a
        # callsite-specialized copy must then be derived from that copy's
        # exact operands.  Reusing a previously proven authored-row result for
        # an elementwise intermediate is precisely how `_row((2, 1), ...)`
        # inherited the `(2, 2)` lane's multiply and consequently returned
        # `(2,)` instead of `(1,)`.
        formal_page = current_identity_book().page("formal_shape")
        # The owner's formal rows do not change while one outermost query
        # runs (it posts shape states, never formal shapes), so the query
        # reads them once.
        if query is not None and query.polymorphic[0] is not None:
            polymorphic_specialization = query.polymorphic[0]
        else:
            polymorphic_specialization = bool(
                graph.G.graph.get("planner_tensor_descriptors")
                and _owner_has_polymorphic_formal(formal_page, authored_owner)
            )
            if query is not None:
                query.polymorphic[0] = polymorphic_specialization
        if (
            concorded_descriptor is not None
            and not formal_conflict
            and not localized_formal
            and not specialized_operator
            and not polymorphic_specialization
        ):
            # This is the current projection of the transformation graph,
            # including dynamic collection rank and row geometry.  Reading
            # it is the stage's decision; graph-local annotations are only
            # inputs to transformations that append a newer projection.
            return concorded_descriptor
        if (
            settled
            and not formal_conflict
            and not localized_formal
            and not specialized_operator
            and not polymorphic_specialization
            and not collection_result
        ):
            return {
                "shape": settled,
                "dtype": str(
                    (_deepest_fact(page, row) or ("", (), "float64"))[2]
                ),
                "rank": len(settled),
            }
    except Exception as error:
        if os.environ.get("TURING_DEBUG_SEQUENCE_ROW_LAYOUT"):
            print(
                "SEQUENCE-ROW-LAYOUT descriptor preamble failed: "
                f"function={graph.G.graph.get('function_name')!r} "
                f"value={int(node_id)} error={type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        row = None

    answer = _tensor_descriptor_rule(graph, node_id, _seen)
    if query is not None and query.cut_counter[0] == cuts_before:
        query.answers[int(node_id)] = (
            None if answer is None else dict(answer)
        )

    if row is not None and page is not None:
        try:
            from .identity_concordance import (
                concordant_shape_transformation_state,
                record_shape_transformation,
            )

            data = graph.G.nodes[int(node_id)]
            operation = str(
                (data.get("attributes") or {}).get("tensor_operation")
                or (data.get("attributes") or {}).get("tensor")
                or (data.get("attributes") or {}).get("tensor_candidate")
                or data.get("op") or data.get("type") or ""
            )
            semantic_sources = tuple(
                (
                    int(parent),
                    int(graph.G.nodes[int(parent)].get("value_id", parent)),
                    str(role),
                )
                for parent, role in data.get("parents") or ()
                if int(parent) in graph.G
                and str(role) not in {
                    "callee", "func", "function", "definition", "control",
                }
            )
            if answer is not None:
                source_cells: tuple = ()
                loop_owner = _arithmetic_loop_target_owner(graph, int(node_id))
                if not semantic_sources and loop_owner is not None:
                    # A ``range`` target is the loop's own product: the edge
                    # comes from the ``For`` node, derived from its identity
                    # cell, not from a descriptor root.
                    from ..common.tensors.topological_reducer import (
                        node_identity_cell,
                    )

                    semantic_sources = ((
                        loop_owner,
                        int(graph.G.nodes[loop_owner].get(
                            "value_id", loop_owner
                        )),
                        "loop_target",
                    ),)
                    source_cells = (node_identity_cell(graph, loop_owner),)
                if not semantic_sources:
                    semantic_sources = ((
                        int(node_id), ("descriptor_root", int(node_id)),
                        "root",
                    ),)
                for _source_node, source_value, source_role in semantic_sources:
                    record_shape_transformation(
                        row[0], source_value, row[0], row[1],
                        stage="graph_tensor_descriptor",
                        operation=operation,
                        source_state=concordant_shape_transformation_state(
                            row[0], source_value,
                        ),
                        source_cells=source_cells,
                        target_state=answer,
                        role=source_role,
                    )
            extents = tuple(answer.get("shape") or ()) if answer else ()
            dynamic = bool(
                answer and str(answer.get("metadata_state") or "") == "dynamic"
            )
            # Only an answer with real extents is a proof.  An empty shape is
            # never cemented: it is indistinguishable from "recovery stopped".
            if extents and not dynamic:
                from .identity_concordance import record_proven_shape

                record_proven_shape(
                    row[0], row[1], extents,
                    (answer or {}).get("dtype"),
                    int(_dependency_levels(graph).get(int(node_id), 0)),
                )
            row_shape = (
                answer.get("sequence_row_shape") if answer else None
            )
            if row_shape is not None:
                from .identity_concordance import (
                    commit_sequence_row_layout,
                )

                commit_sequence_row_layout(
                    row[0], row[1],
                    (tuple(map(int, row_shape)),),
                    ((answer or {}).get("dtype"),),
                    source="tensor descriptor transformation path",
                )
        except Exception as error:
            if os.environ.get("TURING_DEBUG_SEQUENCE_ROW_LAYOUT"):
                print(
                    "SEQUENCE-ROW-LAYOUT publication failed: "
                    f"function={row[0]!r} value={row[1]} "
                    f"error={type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
    return answer


def _promoting_sides(
    graph: Any, operands: tuple[int, ...], sides: list[Any],
) -> list[Any]:
    """The operand descriptors that decide an elementwise result dtype.

    An authored Python ``int``/``float`` literal is a weak scalar: against a
    floating tensor it does not promote (``1 - float32_tensor`` stays
    float32, as the eager reference does).  Giving it its own dtype made
    every ``1 - mask`` over a float32 value float64 at the graph stage, so a
    float32 program carried a float64 intermediate into a float32 formal."""

    def weak_scalar(operand: int, side: Any) -> bool:
        if tuple(side.get("shape") or ()):
            return False
        node = graph.G.nodes[operand]
        if str(node.get("type")) not in {"Constant", "Const", "const"}:
            return False
        literal = node.get("constant")
        if literal is None:
            literal = (node.get("attributes") or {}).get("value")
        return isinstance(literal, (bool, int, float)) and not isinstance(
            literal, np.generic
        )

    strong = [
        side for operand, side in zip(operands, sides)
        if not weak_scalar(operand, side)
    ]
    if strong and len(strong) < len(sides) and any(
        np.dtype(str(side.get("dtype"))).kind in "fc"
        for side in strong
        if str(side.get("dtype") or "unknown") != "unknown"
    ):
        return strong
    return list(sides)


def _dtype_argument_name(
    graph: Any, candidate: int, seen: Any = None,
) -> str | None:
    """The element dtype a ``dtype=`` argument node names, or ``None``.

    The argument may be a computed call -- ``other.get_dtype()`` -- which
    carries no literal at all.  ``get_dtype`` of a tensor IS that tensor's
    descriptor dtype, so name it from the tensor it reads; otherwise a cast
    (or a constructor such as ``arange``) would keep a default dtype instead
    of the reference tensor's (a bool comparison cast to the matrix's dtype
    stayed bool, and every consumer inherited it).
    """

    candidate_data = graph.G.nodes[candidate]
    if str(candidate_data.get("op") or "") == "get_dtype":
        readers = tuple(
            int(reader) for reader, _role
            in candidate_data.get("parents") or ()
            if int(reader) in graph.G
        )
        if len(readers) == 1:
            read = (
                _tensor_descriptor(graph, readers[0], seen)
                if seen is not None
                else _tensor_descriptor(graph, readers[0])
            )
            if read is not None and read.get("dtype"):
                return str(read["dtype"])
        return None
    try:
        literal = _constant_value(graph.G.nodes[candidate])
    except (KeyError, TypeError, ValueError):
        return None
    return str(literal) if isinstance(literal, str) else None


def _tensor_descriptor_rule(
    graph: Any, node_id: int, _seen: set[int] | None = None,
) -> dict[str, Any] | None:
    """Return compiler-owned tensor facts without inspecting a runtime value."""

    if int(node_id) not in graph.G:
        return None
    seen = (
        _seen.branch() if isinstance(_seen, _DescriptorQuery)
        else set(_seen or ())
    )
    if int(node_id) in seen:
        if isinstance(seen, _DescriptorQuery):
            seen.cut()
        return None
    seen.add(int(node_id))
    data = graph.G.nodes[int(node_id)]
    tensor = dict(data.get("tensor") or {})

    def authored_rectangular_shape(expression: ast.AST) -> tuple[int, ...] | None:
        """Return extents proved solely by nested aggregate syntax.

        Values inside a matrix literal need not themselves be constants. A
        tuple such as ``((cx, -sx), (sx, cx))`` still proves a rectangular
        2-by-2 container before any scalar expression is evaluated. This is
        a structural graph fact, unlike converting a literal payload with
        NumPy, which only applies once every leaf is a host constant.
        """

        if not isinstance(expression, (ast.Tuple, ast.List)):
            return ()
        child_shapes = tuple(
            authored_rectangular_shape(child) for child in expression.elts
        )
        if any(shape is None for shape in child_shapes):
            return None
        if child_shapes and len(set(child_shapes)) != 1:
            return None
        return (
            len(expression.elts),
            *(child_shapes[0] if child_shapes else ()),
        )

    expression = data.get("expr_obj")
    literal_operation = str(
        (data.get("attributes") or {}).get("tensor")
        or (data.get("attributes") or {}).get("tensor_candidate")
        or data.get("op")
        or data.get("type")
        or ""
    ).casefold()
    if "shape" not in tensor and (
        str(data.get("type")) in {"Constant", "Const", "const"}
        or literal_operation == "tensor_from_list"
    ):
        literal = data.get("constant")
        if literal is None:
            literal_attributes = data.get("attributes") or {}
            literal = literal_attributes.get(
                "values", literal_attributes.get("value")
            )
        literal_shape = getattr(literal, "shape", None)
        literal_dtype = getattr(literal, "dtype", None)
        if (
            literal_shape is not None
            and not callable(literal_shape)
            and literal_dtype is not None
            and not isinstance(literal, np.dtype)
        ):
            # Capture-time feed arrays can become Constant nodes after their
            # values are structurally bound, but their shape/dtype remain
            # physical boundary facts.  Publish those facts through the
            # descriptor query so a mutable tensor combined with a pointer
            # lane keeps its rank in the web hierarchy.  This reads no array
            # element and does not make the payload a planner literal.
            try:
                literal_shape = tuple(map(int, literal_shape))
            except (TypeError, ValueError):
                pass
            else:
                tensor = {
                    "shape": literal_shape,
                    "dtype": str(literal_dtype),
                    "rank": len(literal_shape),
                }
        elif isinstance(literal, (list, tuple)) and literal:
            # A source-authored nested numeric literal is the payload of a
            # ``tensor_from_list`` once it enters tensor dataflow.  Its
            # rectangular extents and numeric dtype are compile-time facts,
            # just like a captured NumPy constant's ``shape``/``dtype`` above.
            # Publishing them here lets rank-changing views and the following
            # matmul inherit one concordance descriptor instead of collapsing
            # the literal to an untyped structural aggregate.  Only a
            # rectangular nest of numeric leaves is such a payload; a ragged
            # or mixed tuple is ordinary Python structure, proven here from
            # the literal itself rather than by asking NumPy to fail on it.
            def rectangular_extents(value: Any) -> tuple[int, ...] | None:
                if isinstance(value, (bool, int, float, complex)):
                    return ()
                if not isinstance(value, (list, tuple)) or not value:
                    return None
                members = tuple(rectangular_extents(item) for item in value)
                if members[0] is None or any(
                    member != members[0] for member in members
                ):
                    return None
                return (len(value), *members[0])

            literal_array = (
                np.asarray(literal)
                if rectangular_extents(literal) is not None
                else np.asarray((), dtype=object)
            )
            if (
                literal_array.size
                and literal_array.dtype.kind in {"b", "i", "u", "f", "c"}
                and literal_array.dtype.kind != "O"
            ):
                literal_shape = tuple(map(int, literal_array.shape))
                tensor = {
                    "shape": literal_shape,
                    "dtype": str(literal_array.dtype),
                    "rank": len(literal_shape),
                }
    # A generic tensor annotation publishes flat native span storage before a
    # caller is known, so its graph node deliberately starts with dynamic
    # metadata and an empty static shape.  Whole-program planning later puts
    # the exact caller contract on this value identity.  Read that contract
    # before the provisional local descriptor: otherwise ``Tensor``'s flat
    # span rank wins over a caller's logical ``(m, n)`` shape and fundamental
    # operations such as get_shape()/dim() are frozen as runtime scalar feeds.
    linked = (
        graph.G.graph.get("linked_value_abi") or {}
    ).get(int(data.get("value_id", node_id)))
    if (
        isinstance(linked, Mapping)
        and linked.get("shape") is not None
    ):
        linked_shape = tuple(map(int, linked["shape"]))
        tensor = {
            **tensor,
            "shape": linked_shape,
            "dtype": str(
                linked.get("dtype") or tensor.get("dtype") or "float64"
            ),
            "rank": len(linked_shape),
        }
        tensor.pop("metadata_state", None)
    # A source-level tensor builtin remains a syntactic ``Call`` node while
    # its compiler-resolved numerical identity is recorded on the node.  The
    # descriptor law must follow that concordance identity: dispatching on
    # the word ``Call`` loses shape at perfectly ordinary views such as
    # ``abs(pixel_x)`` even though the tensor operand is already proven.
    descriptor_operation = str(
        (data.get("attributes") or {}).get("tensor_operation")
        or (data.get("attributes") or {}).get("tensor")
        or (data.get("attributes") or {}).get("tensor_candidate")
        or data.get("op")
        or data.get("type")
        or ""
    ).casefold()
    if descriptor_operation.startswith("precision_"):
        precision_operation = descriptor_operation.removeprefix(
            "precision_"
        )
        if precision_operation in (
            _ELEMENTWISE_BINARY_OPERATIONS
            | {"neg", "abs"}
        ):
            descriptor_operation = precision_operation
    provisional_tensor = None
    collection_loop_result = bool(
        descriptor_operation == "loopresult"
        and str((data.get("attributes") or {}).get("result_kind"))
        == "collection"
    )
    if collection_loop_result:
        # A collection LoopResult's inherited tensor fact commonly belongs
        # to the value appended on each iteration.  It is a row descriptor,
        # not the descriptor of the collection port.  Re-run the collection
        # law even when that row fact already has a concrete shape; accepting
        # it here erases the leading iteration axis and later makes ``[:, j]``
        # appear to over-index a vector.
        provisional_tensor = dict(tensor)
        tensor = {}
    elif (
        graph.G.graph.get("planner_tensor_descriptors")
        and descriptor_operation in (
            {
                "matmul", "dot", "sum", "mean", "prod", "min", "max",
                "any", "all", "argmin", "argmax",
                "zeros", "ones", "empty", "full",
                "cast_like", "cumsum", "cumprod",
            }
            | _ELEMENTWISE_BINARY_OPERATIONS
        )
        and "shape" in tensor
    ):
        # A catalogue node's result descriptor predates this exact callsite's
        # operand bindings.  Re-evaluate shape-changing operator laws from
        # those local operands, retaining the catalogue descriptor only as a
        # fallback when the law cannot prove a result.  The query records the
        # resulting exact shape in the concordance below.
        provisional_tensor = dict(tensor)
        tensor = {}
    elif tensor.get("metadata_state") == "dynamic" or (
        descriptor_operation == "tensor"
    ) or (
        descriptor_operation in {
            "identity", "loopresult", "loopexit", "loopstateport",
        }
        and not tuple(tensor.get("shape") or ())
        and str(tensor.get("dtype") or "unknown") == "unknown"
    ):
        # Dynamic metadata is a physical span/rank fallback, not a settled
        # logical shape.  Give exact operator identity (tensor normalization,
        # clone, loop result, indexing, cast, and call-result projection) a
        # chance to recover the caller-proven descriptor before returning
        # that fallback.  In particular, frontend ``np.asarray(x)`` is
        # canonical ``tensor(x)`` here; its empty graph-domain default must
        # not outrank the exact descriptor already attached to ``x``.
        provisional_tensor = dict(tensor)
        tensor = {}
    declared_rank = int(
        (data.get("attributes") or {}).get("program_abi_rank", 0) or 0
    )
    if "shape" not in tensor:
        # Whole-object lowering propagates exact Program-ABI contracts over
        # PlanCall argument bindings by SSA value identity. A callee input
        # need not have a useful local binding-name contract, so consult that
        # shared value ledger before attempting operation inference. This is
        # the same declared boundary fact, not backend shape recovery.
        if isinstance(linked, Mapping) and linked.get("shape") is not None:
            tensor = {
                "shape": tuple(map(int, linked["shape"])),
                "dtype": str(linked.get("dtype") or "float64"),
            }
    if "shape" not in tensor and str(
        data.get("op") or data.get("type") or ""
    ).casefold() == "getattr":
        # A declared record span is already a tensor boundary even when the
        # receiver is an opaque Python-shaped record. Resolve that exact
        # contract here, in the compiler-owned descriptor query used by both
        # callsite specialization and region shape planning. Waiting until
        # record-slot SSA materialization is too late: callees have already
        # partitioned gather/matmul regions by then.
        field_name = str(
            (data.get("attributes") or {}).get("attribute") or ""
        )
        receiver = next((
            int(parent)
            for parent, role in data.get("parents") or ()
            if str(role) in {"value", "base", "object", "receiver"}
            and int(parent) in graph.G
        ), None)
        receiver_attributes = (
            {} if receiver is None else
            graph.G.nodes[receiver].get("attributes") or {}
        )
        receiver_identity = receiver_attributes.get(
            "program_abi_record_identity"
        )
        records = dict(
            (graph.G.graph.get("program_abi") or {}).get("records") or {}
        )
        record_matches = tuple(
            record for name, record in records.items()
            if str(name) == str(receiver_identity)
            or str(record.get("identity") or "") == str(receiver_identity)
        )
        record = record_matches[0] if len(record_matches) == 1 else None
        if record is None:
            receiver_binding = str(
                receiver_attributes.get("binding_name") or ""
            )
            record = (
                (graph.G.graph.get("parameter_record_abi") or {}).get(
                    receiver_binding
                )
                if receiver_binding else None
            )
        field = (
            (record.get("fields") or {}).get(field_name)
            if isinstance(record, Mapping) else None
        )
        if isinstance(field, Mapping) and field.get("shape") is not None:
            tensor = {
                "shape": tuple(map(int, field["shape"])),
                "dtype": str(field.get("dtype") or "float64"),
            }
        elif field_name in {"real", "imag"} and receiver is not None:
            inherited = _tensor_descriptor(graph, receiver, seen)
            if inherited is not None:
                tensor = dict(inherited)
                # Both projections expose real-valued elements while
                # preserving the source tensor's storage/rank contract.
                tensor["dtype"] = "float64"
    if (
        "shape" not in tensor
        and data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_kind") == "loop"
    ):
        # A destructured loop target is column k of its iterable's row, k
        # being its position among the loop's targets (the same order the
        # loop composer enumerates as projected iterable bindings).  When the
        # iterable is a generator call, its yielded row columns are on the
        # concordance as that call's ``yield_row`` states.
        from .identity_concordance import (
            concordant_shape_transformation_state,
            descriptor_from_shape_transformation_state,
            shape_scope_of,
        )

        target_value = int(data.get("value_id", node_id))
        if _arithmetic_loop_target_owner(graph, int(node_id)) is not None:
            # ``for k in range(...)``: ``k`` is a Python int scalar, whatever
            # the tensors it is later compared with.
            return {
                "shape": (), "dtype": _ARITHMETIC_LOOP_TARGET_DTYPE,
                "rank": 0,
            }
        for loop_id, loop_data in graph.G.nodes(data=True):
            targets = [
                int(value) for value in dict(
                    (loop_data.get("attributes") or {}).get(
                        "loop_target_bindings"
                    ) or {}
                ).values()
            ]
            if int(node_id) not in targets and target_value not in targets:
                continue
            column = (
                targets.index(int(node_id))
                if int(node_id) in targets else targets.index(target_value)
            )
            iterable = next((
                int(parent)
                for parent, role in loop_data.get("parents") or ()
                if str(role) in {"iterable", "iter"}
                and int(parent) in graph.G
            ), None)
            if iterable is None:
                break
            iterable_value = int(
                graph.G.nodes[iterable].get("value_id", iterable)
            )
            state = concordant_shape_transformation_state(
                shape_scope_of(graph),
                ("yield_row", iterable_value, column),
            )
            described = descriptor_from_shape_transformation_state(state)
            if described is not None:
                return described
            break
    if "shape" not in tensor and data.get("type") == "Input":
        binding_name = (data.get("attributes") or {}).get("binding_name")
        # A copy with no caller was told nothing; the formal still has the
        # shape its callsites proved.
        agreed = _proven_formal_shape(graph, binding_name)
        if agreed is not None:
            return agreed
        boundary = (
            graph.G.graph.get("planner_tensor_descriptors") or {}
        ).get(str(binding_name))
        if boundary is None:
            boundary = (
                graph.G.graph.get("parameter_value_abi") or {}
            ).get(str(binding_name))
        if isinstance(boundary, Mapping) and (
            boundary.get("shape") is not None
            or str(boundary.get("storage") or "").casefold() == "scalar"
        ):
            # A scalar ProgramABI binding commonly omits ``shape`` because its
            # storage class already proves rank zero.  Treat it as the empty
            # tensor descriptor here so that the physical value fact survives
            # successive callsite specializations.  Requiring an explicit
            # shape silently lost scalar type/non-None facts at the first call
            # edge (dt_init -> ref_dt -> ref).
            tensor = {
                "shape": tuple(
                    int(extent) for extent in (boundary.get("shape") or ())
                ),
                "dtype": str(boundary.get("dtype") or "float64"),
            }
            # The declared Python type travels with the descriptor.  A
            # callee re-deriving the type from the dtype alone turned a
            # declared ``builtins.NoneType`` scalar into ``builtins.int``
            # and folded ``distribution is not None`` the wrong way.
            if boundary.get("python_type"):
                tensor["python_type"] = str(boundary["python_type"])
    if "shape" not in tensor:
        operation = descriptor_operation
        collection_expression = data.get("expr_obj")
        uncomposed_comprehension = bool(
            not collection_loop_result
            and isinstance(collection_expression, (
                ast.ListComp, ast.SetComp, ast.GeneratorExp,
            ))
        )
        if collection_loop_result or uncomposed_comprehension:
            # One collection law for one value: the comprehension node
            # before loop composition and its collection LoopResult port
            # after it describe the same sequence.  The port names its
            # element/iterable/materializer by declared attributes; the
            # comprehension by its own ``elt`` -> ``generators`` -> ``iter``
            # edges.  Describing only the port left every query issued
            # before composition (call-site return specialization included)
            # with no answer for the comprehension.
            attributes = dict(data.get("attributes") or {})
            if uncomposed_comprehension:
                element_sources = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "elt" and int(parent) in graph.G
                )
                generator_nodes = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role) == "generators" and int(parent) in graph.G
                )
                iterables = tuple(
                    int(parent)
                    for generator in generator_nodes
                    for parent, role in (
                        graph.G.nodes[generator].get("parents") or ()
                    )
                    if str(role) == "iter" and int(parent) in graph.G
                )
                if len(generator_nodes) == 1 and len(iterables) == 1:
                    attributes["collection_iterable_value_id"] = iterables[0]
                attributes["materializer_node_id"] = int(node_id)
            else:
                element_sources = tuple(
                    int(parent)
                    for parent, role in data.get("parents") or ()
                    if str(role).casefold() == "value"
                    and int(parent) in graph.G
                )
                declared_element_source = attributes.get("value_source_id")
                if declared_element_source is not None:
                    declared_element_source = int(declared_element_source)
                    if (
                        element_sources
                        and element_sources != (declared_element_source,)
                    ):
                        raise ValueError(
                            "collection LoopResult value-source identity "
                            "conflicts with its semantic edge: "
                            f"declared={declared_element_source}, "
                            f"edges={element_sources!r}"
                        )
                    element_sources = (declared_element_source,)
            element = None
            if len(element_sources) == 1:
                element_node_id = int(element_sources[0])
                element_data = graph.G.nodes[element_node_id]
                from .identity_concordance import (
                    proven_shape_contract_of,
                    shape_scope_of,
                )

                element_contract = proven_shape_contract_of(
                    shape_scope_of(graph),
                    int(element_data.get("value_id", element_node_id)),
                )
                if element_contract is not None:
                    element_shape, element_dtype = element_contract
                    element = {
                        "shape": tuple(map(int, element_shape)),
                        "dtype": str(element_dtype),
                        "rank": len(element_shape),
                    }
                else:
                    element = _tensor_descriptor(
                        graph, element_node_id, seen,
                    )
            iterable_id = attributes.get("collection_iterable_value_id")
            iterable = (
                _tensor_descriptor(graph, int(iterable_id), seen)
                if iterable_id is not None and int(iterable_id) in graph.G
                else None
            )
            materializer_id = attributes.get("materializer_node_id")
            materializer_expression = (
                graph.G.nodes[int(materializer_id)].get("expr_obj")
                if materializer_id is not None
                and int(materializer_id) in graph.G else None
            )
            exact_one_per_input = bool(
                isinstance(materializer_expression, (
                    ast.ListComp, ast.SetComp, ast.GeneratorExp,
                ))
                and len(materializer_expression.generators) == 1
                and not materializer_expression.generators[0].ifs
            )
            if element is not None:
                element_shape = tuple(map(
                    int, element.get("shape") or (),
                ))
                element_rank = int(element.get(
                    "rank", len(element_shape),
                ))
                iterable_shape = tuple(
                    map(int, (iterable or {}).get("shape") or ())
                )
                if exact_one_per_input and iterable_shape:
                    return {
                        "shape": (int(iterable_shape[0]), *element_shape),
                        "dtype": str(element.get("dtype") or "float64"),
                        "rank": 1 + element_rank,
                    }
                # The resident sequence's length cell owns the leading
                # extent when the iterable is dynamic.  Preserve the exact
                # rank here instead of publishing the element's row shape as
                # the entire collection; repository SSA later binds that
                # leading extent to the sequence descriptor.
                return {
                    "shape": (),
                    "dtype": str(element.get("dtype") or "float64"),
                    "rank": 1 + element_rank,
                    "metadata_state": "dynamic",
                    "sequence_row_shape": element_shape,
                }
        if operation in {
            "identity", "loopresult", "loopexit", "loopstateport",
        }:
            semantic_roles = (
                {"state", "value"}
                if operation == "loopstateport" else {"value"}
            )
            sources = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in semantic_roles
            )
            declared_source = (data.get("attributes") or {}).get(
                "value_source_id"
            )
            if declared_source is not None:
                declared_source = int(declared_source)
                if sources and sources != (declared_source,):
                    raise ValueError(
                        f"{operation} value-source identity conflicts with "
                        f"its semantic edge: declared={declared_source}, "
                        f"edges={sources!r}"
                    )
                sources = (declared_source,)
            inherited = (
                _tensor_descriptor(graph, sources[0], seen)
                if len(sources) == 1 else None
            )
            # EXTENTS, not merely "states a shape".  A carried matrix whose
            # update resolves to a bare rank-0 answer is not a scalar -- it is
            # the query giving up -- and accepting it here meant the seed,
            # which knows the real shape, was never consulted.  A genuinely
            # scalar carried value (a running sign, a counter) resolves to the
            # same rank-0 answer through the seed below, so preferring extents
            # costs it nothing.
            if inherited is not None and tuple(inherited.get("shape") or ()):
                return inherited
            # The carried chain is a CYCLE: the updated value is produced from
            # the port itself, so following it alone stops at the recursion
            # guard and publishes an empty descriptor.  The loop node declares
            # the pairing -- ``loop_carried_bindings[name] = (initial,
            # updated)`` -- and a carried binding is ONE storage across the
            # loop, so the seed states this port's storage exactly.  Reading
            # it here keeps a loop-carried matrix (LU) shaped for every callee
            # it is then passed to.
            attributes = data.get("attributes") or {}
            binding_name = str(attributes.get("binding_name") or "")
            loop_id = attributes.get("loop_id")
            if (
                str(attributes.get("result_kind") or "") == "carried"
                and binding_name
                and loop_id is not None
                and int(loop_id) in graph.G
            ):
                carried_binding = (
                    (graph.G.nodes[int(loop_id)].get("attributes") or {})
                    .get("loop_carried_bindings") or {}
                ).get(binding_name)
                if carried_binding:
                    initial_id = int(tuple(carried_binding)[0])
                    if initial_id in graph.G and initial_id in seen and isinstance(
                        seen, _DescriptorQuery
                    ):
                        seen.cut()
                    if initial_id in graph.G and initial_id not in seen:
                        seeded = _tensor_descriptor(graph, initial_id, seen)
                        if seeded is not None:
                            return seeded
            if inherited is not None:
                return inherited
        # Shape-preserving unary expressions are ordinary dynamic tensor
        # values too.  Callsite specialization used to see ``-g`` as having
        # no descriptor, specialize the callee only for its static shape
        # argument, and accidentally erase G from that callee's ABI.  Follow
        # the exact unary data edge so nested calls retain their numerical
        # argument without inspecting any runtime payload.
        if operation in {"unsqueeze", "squeeze", "expand", "broadcast_to"}:
            parents = data.get("parents") or ()
            source_id = next((
                int(parent)
                for parent, role in parents
                if str(role).casefold() in {
                    "operand", "value", "base", "input", "self", "receiver",
                }
                and int(parent) in graph.G
            ), None)

            def _axis_literal(candidate: int) -> Any:
                node = graph.G.nodes[int(candidate)]
                literal = node.get("constant")
                if literal is None:
                    literal = (node.get("attributes") or {}).get("value")
                return literal

            arguments = [
                _axis_literal(int(parent))
                for parent, role in parents
                if str(role).casefold().startswith("arg:")
                and int(parent) in graph.G
            ]
            declared_axis = _attribute_axis(data)
            if declared_axis is not None:
                arguments = [declared_axis, *arguments]
            if source_id is not None:
                base = _tensor_descriptor(graph, source_id, seen)
                if base is not None and descriptor_states_a_shape(base):
                    extents = tuple(int(e) for e in (base.get("shape") or ()))
                    dtype = str(base.get("dtype") or "float64")
                    settled = None
                    first = arguments[0] if arguments else None
                    if operation == "unsqueeze" and isinstance(first, int):
                        # A new axis of length one; a negative position counts
                        # from the END OF THE RESULT, which is one longer.
                        position = (
                            first if first >= 0 else first + len(extents) + 1
                        )
                        if 0 <= position <= len(extents):
                            settled = (
                                extents[:position] + (1,) + extents[position:]
                            )
                    elif operation == "squeeze":
                        if isinstance(first, int):
                            position = (
                                first if first >= 0 else first + len(extents)
                            )
                            if (
                                0 <= position < len(extents)
                                and extents[position] == 1
                            ):
                                settled = (
                                    extents[:position]
                                    + extents[position + 1:]
                                )
                        elif not arguments:
                            settled = tuple(e for e in extents if e != 1)
                    else:
                        requested = (
                            first
                            if isinstance(first, (tuple, list))
                            else tuple(arguments)
                        )
                        if requested and all(
                            isinstance(extent, int) and not isinstance(
                                extent, bool
                            )
                            for extent in requested
                        ):
                            settled = tuple(int(e) for e in requested)
                    if settled is not None:
                        return {
                            "shape": settled,
                            "dtype": dtype,
                            "rank": len(settled),
                        }
        if operation in _ELEMENTWISE_BINARY_OPERATIONS:
            operands = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in {
                    "lhs", "rhs", "left", "right", "operand", "value", "other",
                }
                and int(parent) in graph.G
            )
            if len(operands) == 2:
                sides = []
                for operand in operands:
                    side = _tensor_descriptor(graph, operand, seen)
                    if side is None:
                        # An authored scalar literal is rank 0, the same
                        # reading `_operand_descriptor` gives it.
                        node = graph.G.nodes[operand]
                        if str(node.get("type")) in {
                            "Constant", "Const", "const",
                        }:
                            literal = node.get("constant")
                            if literal is None:
                                literal = (
                                    node.get("attributes") or {}
                                ).get("value")
                            if isinstance(literal, (int, float, bool)):
                                side = {"shape": (), "dtype": "float64"}
                    sides.append(side)
                # A HOST SCALAR broadcasts against anything.  ``index >= k``
                # compares a vector with an ordinary Python ``int`` argument,
                # which has no tensor descriptor and never will; demanding one
                # from both sides refused valid source and left every mask
                # built that way shapeless.  A side with no descriptor counts
                # as rank 0 only when its node produces no tensor -- no tensor
                # metadata, no catalogued tensor operation -- so a tensor whose
                # shape merely failed to derive still refuses.
                for index, (side, operand) in enumerate(zip(sides, operands)):
                    if side is not None:
                        continue
                    node = graph.G.nodes[operand]
                    attributes = node.get("attributes") or {}
                    if node.get("tensor") or attributes.get("tensor_candidate"):
                        continue
                    if str(node.get("op") or node.get("type") or "").casefold() in (
                        _ELEMENTWISE_BINARY_OPERATIONS
                    ):
                        continue
                    sides[index] = {"shape": (), "dtype": "float64"}
                # An empty shape is a rank-0 scalar only when the node
                # produces a scalar.  For a tensor-producing node it is what
                # the query returns while its extents are still unsettled,
                # and broadcasting against it yields an answer that LOOKS
                # complete -- (2,) against an unknown gives (1,) -- which the
                # proof short-circuit then makes permanent, so no later round
                # ever revises it.  This is the rule already stated above for
                # a side with NO descriptor; a side whose descriptor STATES
                # nothing is the same case, and a known dtype does not turn an
                # unsettled extent into a fact.  A call's result is included:
                # its extents arrive when the callee settles, and until then
                # the honest answer is that we do not know.
                for index, (side, operand) in enumerate(zip(sides, operands)):
                    if side is None or tuple(side.get("shape") or ()):
                        continue
                    node = graph.G.nodes[operand]
                    attributes = node.get("attributes") or {}
                    semantic_operand = str(
                        attributes.get("tensor")
                        or attributes.get("tensor_candidate")
                        or node.get("op")
                        or node.get("type")
                        or ""
                    ).casefold()
                    if (
                        int(side.get("rank", 0)) == 0
                        and semantic_operand in {
                            "sum", "mean", "prod", "min", "max", "any", "all",
                        }
                    ):
                        # An all-axis reduction has an exact rank-zero result.
                        # It is not the empty placeholder returned by an
                        # unsettled tensor call, so retain the reduction law's
                        # descriptor when it enters later elementwise work.
                        continue
                    if (
                        node.get("tensor")
                        or attributes.get("tensor_candidate")
                        or str(
                            node.get("op") or node.get("type") or ""
                        ).casefold() == "call"
                    ):
                        sides[index] = None
                if all(side is not None for side in sides) and all(
                    descriptor_states_a_shape(side) for side in sides
                ):
                    left_shape = tuple(sides[0].get("shape") or ())
                    right_shape = tuple(sides[1].get("shape") or ())
                    rank = max(len(left_shape), len(right_shape))
                    left_aligned = (1,) * (rank - len(left_shape)) + left_shape
                    right_aligned = (
                        (1,) * (rank - len(right_shape)) + right_shape
                    )
                    broadcast: list[int] = []
                    compatible = True
                    for left_extent, right_extent in zip(
                        left_aligned, right_aligned
                    ):
                        if left_extent == right_extent:
                            broadcast.append(int(left_extent))
                        elif left_extent == 1:
                            broadcast.append(int(right_extent))
                        elif right_extent == 1:
                            broadcast.append(int(left_extent))
                        else:
                            # Not broadcastable: say nothing rather than
                            # invent an extent neither operand has.
                            compatible = False
                            break
                    if compatible:
                        dtype = (
                            "bool"
                            if operation in {
                                "equal", "not_equal", "less", "less_equal",
                                "greater", "greater_equal", "eq", "ne", "lt",
                                "le", "gt", "ge", "logical_and", "logical_or",
                                "logical_xor",
                            }
                            else str(np.result_type(*(
                                np.dtype(str(side.get("dtype")))
                                for side in _promoting_sides(
                                    graph, operands, sides,
                                )
                                if str(side.get("dtype") or "unknown")
                                != "unknown"
                            )))
                        )
                        return {
                            "shape": tuple(broadcast),
                            "dtype": dtype,
                            "rank": len(broadcast),
                        }
        if operation in {"zeros", "ones", "empty", "full"}:
            shape_source = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in {"arg:0", "arg0"}
                and int(parent) in graph.G
            ), None)
            if shape_source is not None:
                try:
                    authored_shape = _constant_value(
                        graph.G.nodes[shape_source]
                    )
                except (KeyError, TypeError, ValueError):
                    authored_shape = None
                if isinstance(authored_shape, int) and not isinstance(
                    authored_shape, bool
                ):
                    extents = (int(authored_shape),)
                elif isinstance(authored_shape, (tuple, list)) and all(
                    isinstance(extent, int) and not isinstance(extent, bool)
                    for extent in authored_shape
                ):
                    extents = tuple(map(int, authored_shape))
                else:
                    extents = None
                if extents is not None and all(extent >= 0 for extent in extents):
                    return {
                        "shape": extents,
                        "dtype": "float64",
                        "rank": len(extents),
                    }
        if operation in {"matmul", "dot"}:
            operands = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in {
                    "lhs", "rhs", "left", "right", "operand", "value",
                    "other", "self", "receiver",
                }
                and int(parent) in graph.G
            )
            if len(operands) == 2:
                sides = tuple(
                    _tensor_descriptor(graph, operand, seen)
                    for operand in operands
                )
                if all(
                    side is not None and descriptor_states_a_shape(side)
                    for side in sides
                ):
                    left = tuple(map(int, sides[0].get("shape") or ()))
                    right = tuple(map(int, sides[1].get("shape") or ()))
                    if operation == "dot" and left and right:
                        # NumPy's dot contract is a contraction over the last
                        # axis of the left operand and axis -2 of a ranked
                        # right operand (axis 0 for a vector).  Unlike matmul,
                        # its leading dimensions are concatenated rather than
                        # broadcast.  This is pure index geometry.
                        right_contract_axis = -1 if len(right) == 1 else -2
                        if left[-1] == right[right_contract_axis]:
                            result = (
                                left[:-1]
                                + (() if len(right) == 1 else right[:-2])
                                + (() if len(right) == 1 else right[-1:])
                            )
                            dtype = next((
                                str(side.get("dtype"))
                                for side in sides
                                if str(side.get("dtype") or "unknown")
                                != "unknown"
                            ), "float64")
                            return {
                                "shape": tuple(result),
                                "dtype": dtype,
                                "rank": len(result),
                            }
                    elif left and right:
                        left_vector = len(left) == 1
                        right_vector = len(right) == 1
                        matrix_left = (1, *left) if left_vector else left
                        matrix_right = (*right, 1) if right_vector else right
                        if matrix_left[-1] == matrix_right[-2]:
                            try:
                                batch = tuple(np.broadcast_shapes(
                                    matrix_left[:-2], matrix_right[:-2]
                                ))
                            except ValueError:
                                pass
                            else:
                                result = (
                                    *batch,
                                    matrix_left[-2],
                                    matrix_right[-1],
                                )
                                if left_vector:
                                    result = result[:-2] + result[-1:]
                                if right_vector:
                                    result = result[:-1]
                                dtype = next((
                                    str(side.get("dtype"))
                                    for side in sides
                                    if str(
                                        side.get("dtype") or "unknown"
                                    ) != "unknown"
                                ), "float64")
                                return {
                                    "shape": tuple(result),
                                    "dtype": dtype,
                                    "rank": len(result),
                                }
        if operation in {
            "sum", "mean", "prod", "min", "max", "any", "all",
            "argmin", "argmax",
        }:
            operand_roles = {
                "operand", "value", "base", "input", "self", "receiver",
            }
            source_id = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role).casefold() in operand_roles
                and int(parent) in graph.G
            ), None)
            source = (
                None if source_id is None
                else _tensor_descriptor(graph, source_id, seen)
            )
            if source is not None:
                source_shape = tuple(map(int, source.get("shape") or ()))
                source_rank = int(source.get("rank", len(source_shape)))
                row_shape = tuple(map(
                    int, source.get("sequence_row_shape") or (),
                ))
                complete_source_shape = descriptor_states_a_shape(source)
                dynamic_leading_extent = bool(
                    not complete_source_shape
                    and str(source.get("metadata_state") or "") == "dynamic"
                    and source_rank == 1 + len(row_shape)
                )
                if not complete_source_shape and not dynamic_leading_extent:
                    source = None
            if source is not None:
                attributes = data.get("attributes") or {}
                axis = attributes.get("axis", attributes.get("dim"))
                if axis is None:
                    # ``x.sum(0)``: the reduction axis is the first positional
                    # argument (``AbstractTensor.sum(self, dim, keepdim)``),
                    # read as the literal its node carries -- the same
                    # reading the unsqueeze rule below gives its position.
                    # Observed: the orbital quadrature ``t26.sum(t3)`` over a
                    # (5, 4) operand settled as rank 0 here (axis ignored),
                    # and the kernel's 4-element result landed in a scalar.
                    positional = [
                        int(parent)
                        for parent, role in data.get("parents") or ()
                        if str(role).casefold() == "arg:0"
                        and int(parent) in graph.G
                    ]
                    if positional:
                        node = graph.G.nodes[positional[0]]
                        literal = node.get("constant")
                        if literal is None:
                            literal = (node.get("attributes") or {}).get("value")
                        if isinstance(literal, int) and not isinstance(literal, bool):
                            axis = int(literal)
                keepdim = bool(attributes.get(
                    "keepdim", attributes.get("keepdims", False)
                ))
                if axis is None:
                    reduced = (1,) * source_rank if keepdim else ()
                else:
                    axes = (
                        tuple(axis)
                        if isinstance(axis, (tuple, list))
                        else (axis,)
                    )
                    if source_rank <= 0:
                        # An explicit reduction axis cannot be normalized
                        # against rank zero.  During callsite specialization
                        # this combination means the local operand descriptor
                        # and the authored reduction disagree; it is not a
                        # proof that the authored result is scalar.  Leave the
                        # result unsettled so the exact value identity falls
                        # through to its tracked provisional descriptor (and
                        # can be revised when the operand's concordance fact
                        # arrives) instead of dividing by zero or cementing a
                        # false shape.
                        reduced = None
                    elif all(
                        isinstance(item, int) and not isinstance(item, bool)
                        for item in axes
                    ):
                        normalized = tuple(sorted({
                            int(item) % source_rank for item in axes
                        }))
                        source_extents = (
                            (None, *row_shape)
                            if dynamic_leading_extent else source_shape
                        )
                        reduced_extents = tuple(
                            1 if keepdim and index in normalized else extent
                            for index, extent in enumerate(source_extents)
                            if keepdim or index not in normalized
                        )
                        reduced = (
                            tuple(map(int, reduced_extents))
                            if all(
                                extent is not None
                                for extent in reduced_extents
                            ) else None
                        )
                    else:
                        reduced = None
                if reduced is not None:
                    dtype = str(source.get("dtype") or "float64")
                    if operation in {"any", "all"}:
                        dtype = "bool"
                    elif operation in {"argmin", "argmax"}:
                        dtype = "int64"
                    return {
                        "shape": tuple(reduced),
                        "dtype": dtype,
                        "rank": len(reduced),
                    }
        if operation == "tensor":
            # Frontend array constructors/normalizers have already been
            # reduced to the canonical tensor operation at this point.  Its
            # payload is the exact first positional value; dtype/order-like
            # arguments describe representation and are not tensor operands.
            # Follow that transformation edge so an asarray-shaped source is
            # never mistaken for an opaque Python call or a fresh scalar.
            sources = tuple(dict.fromkeys(
                int(parent)
                for parent, role in data.get("parents") or ()
                if (
                    str(role).casefold() in {
                        "operand", "value", "base", "input", "self",
                        "receiver", "arg:0", "arg0",
                    }
                    and int(parent) in graph.G
                )
            ))
            if len(sources) == 1:
                source_expression = graph.G.nodes[sources[0]].get("expr_obj")
                if isinstance(source_expression, (ast.Tuple, ast.List)):
                    aggregate_shape = authored_rectangular_shape(
                        source_expression
                    )
                    if aggregate_shape is not None:
                        return {
                            "shape": tuple(map(int, aggregate_shape)),
                            "dtype": "float64",
                            "rank": len(aggregate_shape),
                        }
                inherited = _tensor_descriptor(graph, sources[0], seen)
                if inherited is not None:
                    return inherited
        if operation in {
            "neg", "abs", "sin", "cos", "tan", "exp", "log", "sqrt",
            "tanh", "clone", "copy", "identity", "real", "imag", "conj",
            # A prefix scan is not a reduction: it emits one element per
            # input element, so it keeps its operand's shape exactly.  With
            # no rule here it answered nothing, its result was never proven,
            # and the cast that consumed it read a rank-0 scalar and wrote
            # no elements at all.
            "cumsum", "cumprod",
        }:
            operand_roles = {"operand", "value", "base", "self", "receiver"}
            by_role = tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role).casefold() in operand_roles
                and int(parent) in graph.G
            )
            parents = by_role or tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role) not in {"callee", "func", "definition"}
                and int(parent) in graph.G
            )
            if len(parents) == 1:
                return _tensor_descriptor(graph, parents[0], seen)
        if operation == "cast_like":
            # ``value.cast_like(reference)`` keeps the VALUE's extents and
            # takes the REFERENCE's dtype.  Neither operand is optional: with
            # no rule here the result was never proven, so a consumer such as
            # the ``cumsum`` of ``_first_occurrence`` read a rank-0 scalar,
            # its tensor lowering declined (no extents), and the op reached
            # the LLVM emitter as an un-lowered ``cumsum``.
            operand_roles = {"operand", "value", "base", "self", "receiver"}
            parents = tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role) not in {"callee", "func", "definition"}
                and int(parent) in graph.G
            )
            by_role = tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role).casefold() in operand_roles
                and int(parent) in graph.G
            )
            references = tuple(
                parent for parent in parents if parent not in by_role
            )
            if len(by_role) == 1 and len(references) == 1:
                value_descriptor = _tensor_descriptor(graph, by_role[0], seen)
                reference_descriptor = _tensor_descriptor(
                    graph, references[0], seen
                )
                if value_descriptor is not None and descriptor_states_a_shape(
                    value_descriptor
                ):
                    result = dict(value_descriptor)
                    if (
                        reference_descriptor is not None
                        and reference_descriptor.get("dtype")
                    ):
                        result["dtype"] = reference_descriptor["dtype"]
                    return result
        if operation in _DTYPE_CAST_OPERATIONS:
            # ``x.to_dtype("int64")`` preserves the shape exactly and only
            # changes the element type.  Leaving it descriptor-less made
            # every subscript on a cast value, and every value after it,
            # shapeless (``cell_index[:, :, 1]`` in the periodic terrain).
            parents = tuple(
                int(parent) for parent, role in data.get("parents") or ()
                if str(role) not in {"callee", "func", "definition"}
                and int(parent) in graph.G
            )
            # The VALUE being cast is the one in an operand role.  Counting
            # every non-constant parent made ``x.to_dtype(y.get_dtype())``
            # look like two tensor operands -- the dtype argument is a
            # computed call, not a literal -- so the rule declined and every
            # mask built with a dtype taken from another tensor lost its
            # shape at the cast.
            operand_roles = {"operand", "value", "base", "self", "receiver"}
            by_role = tuple(
                int(parent)
                for parent, role in (data.get("parents") or ())
                if str(role).casefold() in operand_roles
                and int(parent) in graph.G
            )
            tensor_parents = by_role or tuple(
                parent for parent in parents
                if str(
                    graph.G.nodes[parent].get("type") or ""
                ).casefold() not in {"constant", "const"}
            )
            if len(tensor_parents) == 1:
                inherited = _tensor_descriptor(graph, tensor_parents[0], seen)
                if inherited is not None:
                    def _literal_dtype(candidate: int) -> str | None:
                        return _dtype_argument_name(graph, candidate, seen)

                    cast_dtype = next((
                        named
                        for parent in parents
                        if parent not in tensor_parents
                        for named in (_literal_dtype(parent),)
                        if named is not None
                    ), None)
                    result = dict(inherited)
                    if cast_dtype:
                        result["dtype"] = cast_dtype
                    return result
        if operation in {"indexedstore", "index_set", "setitem"}:
            base = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) == "base"
            ), None)
            if base is not None:
                return _tensor_descriptor(graph, base, seen)
        if operation in {"indexed", "getitem", "subscript"}:
            base = next((
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) == "base"
            ), None)
            indices = tuple(
                int(parent)
                for parent, role in data.get("parents") or ()
                if str(role) == "index"
            )
            if base is not None and base in graph.G and len(indices) == 1:
                base_attributes = (
                    graph.G.nodes[base].get("attributes") or {}
                )
                leaves = tuple(map(
                    int,
                    base_attributes.get("aggregate_leaf_value_ids", ()),
                ))
                index_attributes = (
                    graph.G.nodes[indices[0]].get("attributes") or {}
                )
                index = index_attributes.get("value")
                if index is None:
                    index = graph.G.nodes[indices[0]].get("constant")
                if isinstance(index, slice):
                    base_descriptor = _tensor_descriptor(graph, base, seen)
                    base_shape = tuple(
                        (base_descriptor or {}).get("shape") or ()
                    )
                    if base_shape:
                        start, stop, step = index.indices(int(base_shape[0]))
                        sliced = len(range(start, stop, step))
                        return {
                            "shape": (int(sliced), *base_shape[1:]),
                            "dtype": str(
                                (base_descriptor or {}).get("dtype")
                                or "float64"
                            ),
                            "rank": len(base_shape),
                        }
                output_descriptors = base_attributes.get(
                    "tensor_output_descriptors"
                ) or ()
                if (
                    output_descriptors
                    and isinstance(index, int)
                    and not isinstance(index, bool)
                    and -len(output_descriptors) <= index < len(output_descriptors)
                ):
                    descriptor = output_descriptors[
                        index % len(output_descriptors)
                    ]
                    if isinstance(descriptor, Mapping):
                        return copy.deepcopy(dict(descriptor))
                # ``r[1][0]``: the base is itself a projection of a call
                # whose output descriptor at that position is a nested tuple.
                # Follow the exact literal-index path; a structural path is
                # an identity, not an inference.
                path = [index] if isinstance(index, int) and not isinstance(
                    index, bool
                ) else None
                top = base
                while path is not None and top in graph.G:
                    top_data = graph.G.nodes[top]
                    top_attributes = top_data.get("attributes") or {}
                    nested = top_attributes.get("tensor_output_descriptors")
                    if nested:
                        structure: Any = tuple(nested)
                        for step in path:
                            if not isinstance(structure, tuple) or not (
                                -len(structure) <= step < len(structure)
                            ):
                                structure = None
                                break
                            structure = structure[step % len(structure)]
                        if isinstance(structure, Mapping):
                            return copy.deepcopy(dict(structure))
                        break
                    if str(
                        top_data.get("op") or top_data.get("type") or ""
                    ).casefold() not in {"indexed", "getitem", "subscript"}:
                        break
                    top_base = next((
                        int(parent)
                        for parent, role in top_data.get("parents") or ()
                        if str(role) == "base"
                    ), None)
                    top_indices = tuple(
                        int(parent)
                        for parent, role in top_data.get("parents") or ()
                        if str(role) == "index"
                    )
                    if top_base is None or len(top_indices) != 1:
                        break
                    top_index = (
                        graph.G.nodes[top_indices[0]].get("attributes") or {}
                    ).get("value")
                    if top_index is None:
                        top_index = graph.G.nodes[top_indices[0]].get(
                            "constant"
                        )
                    if not isinstance(top_index, int) or isinstance(
                        top_index, bool
                    ):
                        break
                    path.insert(0, int(top_index))
                    top = top_base
                if (
                    leaves
                    and isinstance(index, int)
                    and not isinstance(index, bool)
                    and -len(leaves) <= index < len(leaves)
                ):
                    return _tensor_descriptor(
                        graph, leaves[index % len(leaves)], seen
                    )
        if provisional_tensor is not None:
            tensor = provisional_tensor
        elif declared_rank > 0:
            tensor = {
                "shape": (),
                "dtype": str(
                    (data.get("attributes") or {}).get(
                        "program_abi_dtype", "unknown"
                    )
                ),
                "rank": declared_rank,
                "metadata_state": "dynamic",
            }
        else:
            return None
    static_shape = tuple(tensor.get("shape") or ())
    return {
        "shape": static_shape,
        "dtype": str(tensor.get("dtype") or "float64"),
        "rank": int(tensor.get("rank", len(static_shape))),
        **(
            {"metadata_state": str(tensor["metadata_state"])}
            if tensor.get("metadata_state") is not None else {}
        ),
        **(
            {"device": tensor["device"]}
            if tensor.get("device") is not None else {}
        ),
        # A declared Python type is part of the boundary fact; it rides
        # along so a callee specialization sees ``builtins.NoneType`` and
        # not a type re-derived from the dtype.
        **(
            {"python_type": str(tensor["python_type"])}
            if tensor.get("python_type") else {}
        ),
        **(
            {"sequence_row_shape": tuple(tensor["sequence_row_shape"])}
            if tensor.get("sequence_row_shape") is not None else {}
        ),
    }

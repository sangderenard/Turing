"""Lexical control placement: control-expression attachment, lexical insertion of plan callsites, resident control values (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
import re
from typing import (
    Any,
    Iterable,
)
from dataclasses import replace

from .ssa_fortran_backend import FortranEmissionError


def _graph_control_expression(
    graph_obj: Any, node_id: int, visiting: frozenset[int] = frozenset(),
    *, resident: frozenset[int] = frozenset(),
):
    """Translate a retained scalar predicate from ProcessGraph structure.

    ``resident`` names every value a scheduled region owns.  The shell
    consumes such a value through the region's publication; re-deriving it
    from the node's parents would give one source value a second physical
    definition (and, for a loop-carried update, a stale one).
    """

    from .control_source import ControlExpression

    node_id = int(node_id)
    if node_id in visiting or node_id not in graph_obj:
        return ControlExpression("value", value_id=node_id)
    if node_id in resident:
        return ControlExpression("value", value_id=node_id)
    data = graph_obj.nodes[node_id]
    expression = data.get("expr_obj")
    attributes = data.get("attributes") or {}
    literal = attributes.get("value", data.get("constant"))
    if isinstance(expression, ast.Constant):
        literal = expression.value
        if literal is None:
            return ControlExpression("const", value_id=node_id, literal=None)
    if isinstance(literal, (bool, int, float)) and str(
        data.get("type") or ""
    ).casefold() in {"constant", "const"}:
        return ControlExpression("const", value_id=node_id, literal=literal)
    if attributes.get("aggregate_kind") in {
        "list", "set", "dict", "tuple", "bytes", "bytearray",
    }:
        return ControlExpression(
            "sequence_nonempty",
            (ControlExpression("value", value_id=node_id),),
            value_id=node_id,
            literal=attributes.get("aggregate_kind") in {"set", "dict"},
        )
    parents = tuple(data.get("parents") or ())
    ordered = tuple(
        int(parent)
        for parent, role in sorted(
            parents,
            key=lambda item: (
                int(str(item[1]).split(":")[-1])
                if str(item[1]).split(":")[-1].isdigit() else 0
            ),
        )
        if str(role) not in {"callee", "ops", "operator"}
    )
    if isinstance(expression, ast.BoolOp):
        operator = "and" if isinstance(expression.op, ast.And) else "or"
        operands = tuple(
            _graph_control_expression(
                graph_obj, parent, visiting | {node_id}, resident=resident,
            )
            for parent in ordered
        )
        if operands:
            result = operands[0]
            for operand in operands[1:]:
                result = ControlExpression(
                    operator, (result, operand), value_id=node_id
                )
            return result
    operation = {
        "add": "add", "sub": "sub", "mul": "mul",
        "div": "div", "truediv": "div",
        "less": "lt", "lt": "lt",
        "lessequal": "le", "less_equal": "le", "le": "le",
        "greater": "gt", "gt": "gt",
        "greaterequal": "ge", "greater_equal": "ge", "ge": "ge",
        "equal": "eq", "eq": "eq",
        "notequal": "ne", "not_equal": "ne", "ne": "ne",
        "logical_and": "and", "land": "and",
        "logical_or": "or", "lor": "or",
        "logical_not": "not", "lnot": "not",
        "neg": "neg", "usub": "neg",
        "bitand": "bitand", "bitor": "bitor",
        "bitxor": "bitxor", "shl": "shl", "shr": "shr",
        "invert": "invert",
        "item": "item",
        "float": "float", "int": "int", "bool": "bool",
        "isfinite": "isfinite",
    }.get(str(data.get("op") or data.get("type") or "").casefold())
    if operation is not None:
        arity = 1 if operation in {
            "not", "neg", "invert", "item", "float", "int", "bool",
            "isfinite",
        } else 2
        operands = tuple(
            _graph_control_expression(
                graph_obj, parent, visiting | {node_id}, resident=resident,
            )
            for parent in ordered[:arity]
        )
        if len(operands) == arity:
            return ControlExpression(operation, operands, value_id=node_id)
    return ControlExpression("value", value_id=node_id)


def _region_resident_value_ids(dispatch_subgraphs: Iterable[Any]) -> frozenset[int]:
    """Every graph value a scheduled region actually publishes.

    ``deployment_nodes`` includes shader-local intermediates.  The coordinator
    cannot consume those identities; only ``deployment_outputs`` receive Store
    nodes and cross the region boundary.  Treating ownership as publication
    suppresses control-expression lowering and leaves the unpublished local as
    a fabricated function formal.
    """

    return frozenset(
        int(node_id)
        for subgraph in dispatch_subgraphs
        for node_id in subgraph.G.graph.get("deployment_outputs", ())
    )


def _attach_graph_control_expressions(
    control: Any, graph_obj: Any, *, resident: frozenset[int] = frozenset(),
):
    """Recover structured predicates for every retained conditional."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock, WhileBlock,
    )

    def owns_optional_presence(expression: Any) -> bool:
        if expression is None:
            return False
        return (
            str(expression.op) == "optional_presence"
            or any(owns_optional_presence(item) for item in expression.operands)
        )

    def attach(block):
        if isinstance(block, ConditionalBlock):
            predicate_expression = (
                block.predicate_expression
                if owns_optional_presence(block.predicate_expression)
                else _graph_control_expression(
                    graph_obj, int(block.predicate_value_id),
                    resident=resident,
                )
            )
            return replace(
                block,
                body=attach(block.body),
                orelse=(None if block.orelse is None else attach(block.orelse)),
                predicate_expression=predicate_expression,
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block, blocks=tuple(attach(item) for item in block.blocks)
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=attach(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=attach(block.condition),
                body=attach(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=attach(block.callee))
        return block

    return replace(control, root=attach(control.root))


def _nest_lexical_conditionals_in_loops(
    control: Any, graph: Any, dispatch_subgraphs: Iterable[Any],
):
    """Move a conditional authored inside a retained loop into that loop.

    The control overlay embeds a nested conditional by replacing its region
    span inside the parent.  A conditional that owns no numerical region
    (its arm holds only effects or calls) has no span to replace, so it fell
    to the root sequence after the loop: its predicate then read loop-local
    values in ``while_exit`` and nothing it guarded ran per iteration.  The
    authored span is the identity: a conditional whose source position lies
    inside a retained loop's span belongs in that loop's body, ordered among
    the body's statements by source position.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, LoopControlBlock,
        SequenceBlock, SequenceMutationBlock, StatementBlock, WhileBlock,
    )

    def node_position(node_id: int) -> tuple[int, int, int]:
        return _authored_node_position(graph.G, int(node_id))

    # Sequence effects and numerical regions share one authored ordering.
    # A region is atomic, so its position is the last authored member it must
    # complete, not the first member that happened to seed it.  Reuse the
    # canonical rule used by scalar-field effects; independently taking the
    # minimum here moved a late mapping store ahead of its mapping clear.
    region_positions = _dispatch_region_completion_positions(
        graph.G, dispatch_subgraphs,
    )

    def block_position(block) -> tuple[int, int, int]:
        if isinstance(block, SequenceMutationBlock):
            return node_position(int(block.mutation.effect_node_id))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            line = block.lines[0]
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                return region_positions.get(
                    int(line[len("__scheduled_region_"):-2]),
                    (1 << 30, 0, 0),
                )
            if line.startswith("__plan_callsite_") and line.endswith("__"):
                return node_position(int(line[len("__plan_callsite_"):-2]))
        if isinstance(block, ConditionalBlock) and block.source_node_id is not None:
            return node_position(int(block.source_node_id))
        if isinstance(block, LoopControlBlock) and block.site_node_id is not None:
            return _loop_control_block_position(graph.G, block, node_position)
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                return node_position(int(source_id))
        if isinstance(block, CallBlock):
            return node_position(int(block.callsite_id))
        if isinstance(block, SequenceBlock) and block.blocks:
            return min(map(block_position, block.blocks))
        return (1 << 30, 0, 0)

    def loop_span(block) -> tuple[int, int] | None:
        source_id = getattr(block, "source_loop_node_id", None)
        if source_id is None:
            return None
        expression = graph.G.nodes.get(int(source_id), {}).get("expr_obj")
        start = getattr(expression, "lineno", None)
        end = getattr(expression, "end_lineno", None)
        if start is None or end is None:
            return None
        return int(start), int(end)

    spans: list[tuple[int, int, int]] = []  # (start, end, loop source id)

    def collect_loops(block):
        if isinstance(block, (LoopBlock, WhileBlock)):
            span = loop_span(block)
            if span is not None:
                spans.append((span[0], span[1], int(block.source_loop_node_id)))
            collect_loops(block.body)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect_loops(child)
        elif isinstance(block, ConditionalBlock):
            collect_loops(block.body)
            if block.orelse is not None:
                collect_loops(block.orelse)
        elif isinstance(block, CallBlock):
            collect_loops(block.callee)

    collect_loops(control.root)
    if not spans:
        return control

    def innermost_loop(line: int) -> int | None:
        candidates = [
            (end - start, loop_id) for start, end, loop_id in spans
            if start <= line <= end
        ]
        return min(candidates)[1] if candidates else None

    misplaced: list = []

    def strip(block, ancestors: tuple[int, ...]):
        """Remove conditionals that belong in a loop not among ``ancestors``."""

        if isinstance(block, SequenceBlock):
            kept = []
            for child in block.blocks:
                projected = strip(child, ancestors)
                if projected is not None:
                    kept.append(projected)
            return SequenceBlock(tuple(kept))
        if isinstance(block, ConditionalBlock):
            if block.source_node_id is not None:
                owner = innermost_loop(
                    node_position(int(block.source_node_id))[0]
                )
                if owner is not None and owner not in ancestors:
                    misplaced.append((owner, block))
                    return None
            return replace(
                block,
                body=strip(block.body, ancestors),
                orelse=(
                    None if block.orelse is None
                    else strip(block.orelse, ancestors)
                ),
            )
        if isinstance(block, (LoopBlock, WhileBlock)):
            own = getattr(block, "source_loop_node_id", None)
            inner = ancestors if own is None else (*ancestors, int(own))
            return replace(block, body=strip(block.body, inner))
        if isinstance(block, CallBlock):
            return replace(block, callee=strip(block.callee, ancestors))
        return block

    root = strip(control.root, ())
    if not misplaced:
        return control

    def insert_ordered(block, item):
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(_flattened_sequence_children(block))
        ]
        decorated.append((block_position(item), len(decorated), item))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                decorated, key=lambda entry: (entry[0], entry[1])
            )
        ))

    def insert_into_loop(block, loop_id: int, item):
        if isinstance(block, (LoopBlock, WhileBlock)):
            if int(getattr(block, "source_loop_node_id", -1) or -1) == loop_id:
                return replace(block, body=insert_ordered(block.body, item)), True
            body, inserted = insert_into_loop(block.body, loop_id, item)
            return (replace(block, body=body), True) if inserted else (block, False)
        if isinstance(block, SequenceBlock):
            for index, child in enumerate(block.blocks):
                projected, inserted = insert_into_loop(child, loop_id, item)
                if inserted:
                    return replace(block, blocks=(
                        *block.blocks[:index], projected,
                        *block.blocks[index + 1:],
                    )), True
            return block, False
        if isinstance(block, ConditionalBlock):
            body, inserted = insert_into_loop(block.body, loop_id, item)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_into_loop(block.orelse, loop_id, item)
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, CallBlock):
            callee, inserted = insert_into_loop(block.callee, loop_id, item)
            return (replace(block, callee=callee), True) if inserted else (block, False)
        return block, False

    # Outer conditionals first, so an inner one lands inside its parent's
    # already-placed arm by ordinary source ordering.
    for loop_id, block in sorted(
        misplaced,
        key=lambda item: node_position(int(item[1].source_node_id)),
    ):
        root, inserted = insert_into_loop(root, loop_id, block)
        if not inserted:
            raise FortranEmissionError(
                f"conditional {block.source_node_id} is authored inside loop "
                f"{loop_id} but that loop is absent from the control tree"
            )
    return replace(control, root=root)


def _plan_callsite_projection_ids(graph_obj: Any, hierarchy_plan: Any) -> dict[int, tuple[int, ...]]:
    """Every caller-side projection of each planned call's bound results.

    A call result that is a record or tuple has no physical value of its
    own; the caller reads it through ``GetAttr``/``Indexed`` nodes.  Those
    nodes are the values the rest of the caller consumes, so a scheduler
    that orders statements by value dependencies must see them as the
    call's publications.
    """

    from .hierarchical_plan import PlanCall

    if hierarchy_plan is None:
        return {}
    result: dict[int, tuple[int, ...]] = {}
    for item in getattr(hierarchy_plan, "items", ()):
        if not isinstance(item, PlanCall):
            continue
        roots = [
            int(caller_id) for _callee_id, caller_id in item.result_bindings
        ] or [int(value_id) for value_id in item.result_value_ids]
        found: set[int] = set()
        pending = list(roots)
        root_ids = set(roots)
        while pending:
            current = pending.pop()
            if current not in graph_obj:
                continue
            # Only an aggregate is projected: the call's own results, or a
            # projection the reducer declared a record (``result_class_ref``).
            # Anything else is a value the call delivers, and reading into it
            # is ordinary computation the planner places in a region.
            # Descending into ``metrics.control_values`` claimed its element
            # ``control_values[0]`` for the call too; region 17 computes it,
            # so one id had two producers and the region reading it bound to
            # the call: run_superstep's ``if float(metrics.control_values[0]
            # .item()) > 0.0`` and its feeding regions ran before region 17
            # (woodshop dt system, use-not-dominated).
            if current not in root_ids and not (
                graph_obj.nodes[current].get("attributes") or {}
            ).get("result_class_ref"):
                continue
            for successor in graph_obj.successors(current):
                data = graph_obj.nodes[successor]
                operation = str(
                    data.get("op") or data.get("type") or ""
                ).casefold()
                if operation not in {"getattr", "indexed"}:
                    continue
                if not any(
                    int(parent) == current
                    and str(role) in {"value", "base", "operand", "object"}
                    for parent, role in data.get("parents") or ()
                ):
                    continue
                value_id = int(data.get("value_id", successor))
                if value_id in found:
                    continue
                found.add(value_id)
                pending.append(int(successor))
        if found:
            result[int(item.callsite_id)] = tuple(sorted(found))
    return result


def _flattened_sequence_children(block) -> tuple:
    """A sequence's children with nested sequences opened up.

    Nested ``SequenceBlock``s are grouping only.  Ordering a sequence by
    source position must see each authored statement on its own: a nested
    group takes the position of its earliest member, and dragged a later
    conditional ahead of the statement producing its predicate.
    """

    from .control_source import SequenceBlock

    if not isinstance(block, SequenceBlock):
        return (block,)
    flat = []
    for child in block.blocks:
        flat.extend(_flattened_sequence_children(child))
    return tuple(flat)


def _loop_control_block_position(
    graph_obj: Any, block: Any, node_position: Any,
) -> tuple[int, int, int]:
    """Lexical position of a ``LoopControlBlock`` for source-ordered placement.

    A ``break``/``continue`` sits at its statement.  A ``return`` is anchored
    at its returned expression -- for ``return a, f(x), g(y)`` that is the
    Tuple, whose source position is the START of the expression (the opening
    column), i.e. BEFORE every call written inside it.  Sorting the edge by
    that start put it ahead of the ``__plan_callsite_N__`` markers for
    ``f(x)`` and ``g(y)``, so the edge consumed values produced after it and
    the SSA scheduler refused with ``control effect order conflicts with
    value dependencies`` (step_with_dt_control_used's in-loop ``return
    metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)``).
    A return executes once its expression is fully evaluated, so its position
    is the END of that expression: the latest end coordinate over the site
    and the exact slot values the edge carries (``return_value_ids``), which
    every call inside the expression precedes.
    """

    site_id = int(block.site_node_id)
    candidates = [node_position(site_id)]
    if block.action != "return":
        return candidates[0]
    for value_id in (site_id, *(
        int(value_id) for value_id in block.return_value_ids
        if value_id is not None
    )):
        if value_id not in graph_obj:
            continue
        expression = graph_obj.nodes[value_id].get("expr_obj")
        end_line = getattr(expression, "end_lineno", None)
        if end_line is None:
            continue
        candidates.append((
            int(end_line),
            int(getattr(expression, "end_col_offset", 0) or 0),
            site_id,
        ))
    return max(candidates)


def _authored_node_position(graph_obj: Any, node_id: int) -> tuple[int, int, int]:
    """Source position of a graph node, or of the control record it left.

    A conditional whose ``If`` node the structural fold removed still exists
    as a control block naming that node; its authored position is retained
    in the graph's source control records.  Without this fallback such a
    block had no position, so no lexical placement could order it, and it
    was lowered ahead of its predicate's producer.
    """

    data = graph_obj.nodes.get(int(node_id)) if int(node_id) in graph_obj else None
    expression = None if data is None else data.get("expr_obj")
    if isinstance(expression, ast.comprehension):
        # comprehension is a control AST node without its own location;
        # its authored target still places the producer before its consumers.
        expression = expression.target
    span = {} if data is None else (data.get("source_span") or {})
    if data is None or (
        getattr(expression, "lineno", None) is None and not span.get("line")
    ):
        from .glsl_deployment_strategy import _source_control_records
        record = _source_control_records(graph_obj).get(int(node_id))
        recorded = None if record is None else record.get("expression")
        if getattr(recorded, "lineno", None) is not None:
            return (
                int(recorded.lineno),
                int(getattr(recorded, "col_offset", 0) or 0),
                int(node_id),
            )
    return (
        int(getattr(expression, "lineno", span.get("line", 1 << 30)) or (1 << 30)),
        int(getattr(expression, "col_offset", span.get("column", 0)) or 0),
        int(node_id),
    )


def _dispatch_region_completion_positions(
    graph_obj: Any,
    dispatch_subgraphs: Iterable[Any],
) -> dict[int, tuple[int, int, int]]:
    """Place each atomic region where its last authored member completes."""

    return {
        index: max(
            (
                _authored_node_position(graph_obj, int(node_id))
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            ),
            default=(1 << 30, 0, index),
        )
        for index, subgraph in enumerate(dispatch_subgraphs)
    }


def _insert_lexically(root, graph: Any, dispatch_subgraphs: Iterable[Any], item, node_id: int):
    """Insert ``item`` at the authored position of graph node ``node_id``.

    Innermost conditional arm that owns the node (branch compartments), else
    the innermost retained loop whose span contains it, else the root
    sequence; within the chosen sequence, ordered by source position.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, LoopControlBlock,
        ScalarFieldWriteBlock, SequenceBlock, SequenceMutationBlock, SequenceQueryBlock,
        StatementBlock, WhileBlock,
    )
    from .glsl_deployment_strategy import _branch_compartments

    def node_position(candidate: int) -> tuple[int, int, int]:
        return _authored_node_position(graph.G, int(candidate))

    region_positions = {
        int(index): min(
            (
                node_position(int(member))
                for member in subgraph.G.graph.get("deployment_nodes", ())
            ),
            default=(1 << 30, 0, int(index)),
        )
        for index, subgraph in enumerate(dispatch_subgraphs)
    }

    def block_position(block) -> tuple[int, int, int]:
        if isinstance(block, ScalarFieldWriteBlock):
            return node_position(int(block.effect_node_id))
        if isinstance(block, SequenceMutationBlock):
            return node_position(int(block.mutation.effect_node_id))
        if isinstance(block, SequenceQueryBlock) and block.source_call_node_id is not None:
            return node_position(int(block.source_call_node_id))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            line = block.lines[0]
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                return region_positions.get(
                    int(line[len("__scheduled_region_"):-2]),
                    (1 << 30, 0, 0),
                )
            if line.startswith("__plan_callsite_") and line.endswith("__"):
                return node_position(int(line[len("__plan_callsite_"):-2]))
        if isinstance(block, ConditionalBlock) and block.source_node_id is not None:
            return node_position(int(block.source_node_id))
        if isinstance(block, LoopControlBlock) and block.site_node_id is not None:
            return _loop_control_block_position(graph.G, block, node_position)
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                return node_position(int(source_id))
        if isinstance(block, CallBlock):
            return node_position(int(block.callsite_id))
        if isinstance(block, SequenceBlock) and block.blocks:
            return min(map(block_position, block.blocks))
        return (1 << 30, 0, 0)

    def insert_ordered(block, candidate):
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(_flattened_sequence_children(block))
        ]
        decorated.append((block_position(candidate), len(decorated), candidate))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                decorated, key=lambda entry: (entry[0], entry[1])
            )
        ))

    present: set[int] = set()
    loop_spans: list[tuple[int, int, int]] = []

    def survey(block):
        if isinstance(block, ConditionalBlock):
            if block.source_node_id is not None:
                present.add(int(block.source_node_id))
            survey(block.body)
            if block.orelse is not None:
                survey(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                survey(child)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                expression = graph.G.nodes.get(int(source_id), {}).get("expr_obj")
                start = getattr(expression, "lineno", None)
                end = getattr(expression, "end_lineno", None)
                if start is not None and end is not None:
                    loop_spans.append((int(start), int(end), int(source_id)))
            survey(block.body)
        elif isinstance(block, CallBlock):
            survey(block.callee)

    survey(root)

    def insert_in_conditional(block, owner_id, arm):
        if isinstance(block, ConditionalBlock):
            if int(block.source_node_id or -1) == int(owner_id):
                if str(arm) == "body":
                    return replace(block, body=insert_ordered(block.body, item)), True
                return replace(
                    block,
                    orelse=insert_ordered(block.orelse or SequenceBlock(()), item),
                ), True
            body, inserted = insert_in_conditional(block.body, owner_id, arm)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_in_conditional(block.orelse, owner_id, arm)
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = (
                    insert_in_conditional(child, owner_id, arm)
                    if not inserted else (child, False)
                )
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, (LoopBlock, WhileBlock)):
            body, inserted = insert_in_conditional(block.body, owner_id, arm)
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_in_conditional(block.callee, owner_id, arm)
            return replace(block, callee=callee), inserted
        return block, False

    def insert_into_loop(block, loop_id: int):
        if isinstance(block, (LoopBlock, WhileBlock)):
            if int(getattr(block, "source_loop_node_id", -1) or -1) == loop_id:
                return replace(block, body=insert_ordered(block.body, item)), True
            body, inserted = insert_into_loop(block.body, loop_id)
            return (replace(block, body=body), True) if inserted else (block, False)
        if isinstance(block, SequenceBlock):
            for index, child in enumerate(block.blocks):
                projected, inserted = insert_into_loop(child, loop_id)
                if inserted:
                    return replace(block, blocks=(
                        *block.blocks[:index], projected, *block.blocks[index + 1:],
                    )), True
            return block, False
        if isinstance(block, ConditionalBlock):
            body, inserted = insert_into_loop(block.body, loop_id)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_into_loop(block.orelse, loop_id)
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, CallBlock):
            callee, inserted = insert_into_loop(block.callee, loop_id)
            return (replace(block, callee=callee), True) if inserted else (block, False)
        return block, False

    memberships = _branch_compartments(graph)
    guards = [
        (int(owner), str(arm))
        for owner, arm in memberships.get(int(node_id), ())
        if str(arm) in {"body", "orelse"} and int(owner) in present
    ]
    if guards:
        owner, arm = max(guards, key=lambda entry: node_position(entry[0]))
        root, inserted = insert_in_conditional(root, owner, arm)
        if inserted:
            return root, True
    line = node_position(int(node_id))[0]
    candidates = [
        (end - start, loop_id) for start, end, loop_id in loop_spans
        if start <= line <= end
    ]
    if candidates:
        root, inserted = insert_into_loop(root, min(candidates)[1])
        if inserted:
            return root, True
    return insert_ordered(root, item), True


def _place_plan_callsites_lexically(
    control: Any, graph: Any, hierarchy_plan: Any, dispatch_subgraphs: Iterable[Any],
):
    """Insert every planned callsite marker at its authored position.

    The callsite scheduler placed a pending call before the next retained
    region in hierarchy order.  When intervening control (a ``for`` loop
    that consumes the call's results) sits between the call and that region,
    the marker landed after that control: ``advance`` was emitted in the
    exit block of the loop over ``targets.error_limits`` that reads the
    metrics it publishes.  The authored position is the identity: the
    marker goes into the innermost conditional arm that owns the call, else
    the innermost loop whose span contains it, ordered by source position.
    The scheduler treats a marker already present as placed.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, LoopControlBlock,
        SequenceBlock, SequenceMutationBlock, StatementBlock, WhileBlock,
    )
    from .glsl_deployment_strategy import _branch_compartments
    from .hierarchical_plan import PlanCall

    if hierarchy_plan is None:
        return control
    planned_calls = {
        int(item.callsite_id): item
        for item in getattr(hierarchy_plan, "items", ())
        if isinstance(item, PlanCall)
    }
    from .control_source import control_callsite_markers

    placed_markers = control_callsite_markers(control)
    callsite_ids = [
        callsite_id for callsite_id in planned_calls
        if int(callsite_id) not in placed_markers
    ]
    if not callsite_ids:
        return control

    def node_position(node_id: int) -> tuple[int, int, int]:
        return _authored_node_position(graph.G, int(node_id))

    region_positions = {
        int(index): min(
            (
                node_position(int(node_id))
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            ),
            default=(1 << 30, 0, int(index)),
        )
        for index, subgraph in enumerate(dispatch_subgraphs)
    }

    def block_position(block) -> tuple[int, int, int]:
        if isinstance(block, SequenceMutationBlock):
            return node_position(int(block.mutation.effect_node_id))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            line = block.lines[0]
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                return region_positions.get(
                    int(line[len("__scheduled_region_"):-2]),
                    (1 << 30, 0, 0),
                )
            if line.startswith("__plan_callsite_") and line.endswith("__"):
                return node_position(int(line[len("__plan_callsite_"):-2]))
        if isinstance(block, ConditionalBlock) and block.source_node_id is not None:
            return node_position(int(block.source_node_id))
        if isinstance(block, LoopControlBlock) and block.site_node_id is not None:
            return _loop_control_block_position(graph.G, block, node_position)
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                return node_position(int(source_id))
        if isinstance(block, CallBlock):
            return node_position(int(block.callsite_id))
        if isinstance(block, SequenceBlock) and block.blocks:
            return min(map(block_position, block.blocks))
        return (1 << 30, 0, 0)

    def insert_ordered(block, item):
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(_flattened_sequence_children(block))
        ]
        decorated.append((block_position(item), len(decorated), item))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                # A return anchored at its result call has the same source
                # position as that call. Evaluate the value before taking
                # the terminal edge, even when the edge was inserted first.
                decorated, key=lambda entry: (
                    entry[0], isinstance(entry[2], LoopControlBlock), entry[1]
                )
            )
        ))

    present: set[int] = set()
    present_loops: set[int] = set()
    loop_spans: list[tuple[int, int, int]] = []

    def survey(block):
        if isinstance(block, ConditionalBlock):
            if block.source_node_id is not None:
                present.add(int(block.source_node_id))
            survey(block.body)
            if block.orelse is not None:
                survey(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                survey(child)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            source_id = getattr(block, "source_loop_node_id", None)
            if source_id is not None:
                present_loops.add(int(source_id))
                expression = graph.G.nodes.get(int(source_id), {}).get("expr_obj")
                start = getattr(expression, "lineno", None)
                end = getattr(expression, "end_lineno", None)
                if start is not None and end is not None:
                    loop_spans.append((int(start), int(end), int(source_id)))
            survey(block.body)
        elif isinstance(block, CallBlock):
            survey(block.callee)

    survey(control.root)

    def insert_in_conditional(block, owner_id, arm, item):
        if isinstance(block, ConditionalBlock):
            if int(block.source_node_id or -1) == int(owner_id):
                if str(arm) == "body":
                    return replace(block, body=insert_ordered(block.body, item)), True
                return replace(
                    block,
                    orelse=insert_ordered(block.orelse or SequenceBlock(()), item),
                ), True
            body, inserted = insert_in_conditional(block.body, owner_id, arm, item)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_in_conditional(
                    block.orelse, owner_id, arm, item
                )
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = (
                    insert_in_conditional(child, owner_id, arm, item)
                    if not inserted else (child, False)
                )
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, (LoopBlock, WhileBlock)):
            body, inserted = insert_in_conditional(block.body, owner_id, arm, item)
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_in_conditional(block.callee, owner_id, arm, item)
            return replace(block, callee=callee), inserted
        return block, False

    def insert_into_loop(block, loop_id: int, item):
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_loop_id = getattr(block, "source_loop_node_id", None)
            if (
                source_loop_id is not None
                and int(source_loop_id) == int(loop_id)
            ):
                return replace(block, body=insert_ordered(block.body, item)), True
            body, inserted = insert_into_loop(block.body, loop_id, item)
            return (replace(block, body=body), True) if inserted else (block, False)
        if isinstance(block, SequenceBlock):
            for index, child in enumerate(block.blocks):
                projected, inserted = insert_into_loop(child, loop_id, item)
                if inserted:
                    return replace(block, blocks=(
                        *block.blocks[:index], projected,
                        *block.blocks[index + 1:],
                    )), True
            return block, False
        if isinstance(block, ConditionalBlock):
            body, inserted = insert_into_loop(block.body, loop_id, item)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_into_loop(block.orelse, loop_id, item)
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, CallBlock):
            callee, inserted = insert_into_loop(block.callee, loop_id, item)
            return (replace(block, callee=callee), True) if inserted else (block, False)
        return block, False

    memberships = _branch_compartments(graph)
    root = control.root
    for callsite_id in sorted(callsite_ids, key=node_position):
        if int(callsite_id) not in graph.G:
            continue
        marker = StatementBlock((f"__plan_callsite_{int(callsite_id)}__",))
        guards = [
            (int(owner), str(arm))
            for owner, arm in memberships.get(int(callsite_id), ())
            if str(arm) in {"body", "orelse"} and int(owner) in present
        ]
        inserted = False
        if guards:
            owner, arm = max(guards, key=lambda item: node_position(item[0]))
            root, inserted = insert_in_conditional(root, owner, arm, marker)
        if not inserted:
            # PlanCall carries the exact lexical loop ancestry.  Prefer it to
            # source spans: comprehension AST nodes commonly have no usable
            # span in persisted graphs, which previously hoisted their calls
            # to function entry before loop-target projections existed.
            planned_loop_id = next((
                int(loop_id)
                for loop_id in reversed(
                    planned_calls[int(callsite_id)].enclosing_loop_ids
                )
                if int(loop_id) in present_loops
            ), None)
            if planned_loop_id is not None:
                root, inserted = insert_into_loop(
                    root, planned_loop_id, marker
                )
        if not inserted:
            line = node_position(int(callsite_id))[0]
            candidates = [
                (end - start, loop_id) for start, end, loop_id in loop_spans
                if start <= line <= end
            ]
            if candidates:
                root, inserted = insert_into_loop(root, min(candidates)[1], marker)
        if not inserted:
            root = insert_ordered(root, marker)
    return replace(control, root=root)


def _stamp_conditional_callsite_ownership(
    control: Any, graph: Any, hierarchy_plan: Any,
):
    """Record, on each retained conditional, the planned calls its arms own.

    Conditionals overlaid on numerical regions already carry
    ``body_callsite_ids``/``orelse_callsite_ids`` from the planner.  A
    conditional inside a retained loop body is built by the loop composer
    and carries none, so the callsite scheduler treated a call authored in
    its arm as a trailing call of the loop and appended its marker after the
    body's terminal edge, in unreachable code.  The graph's branch
    compartments are the exact authored ownership; stamp the innermost
    conditional present in the tree with each such callsite.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock, WhileBlock,
    )
    from .glsl_deployment_strategy import _branch_compartments
    from .hierarchical_plan import PlanCall

    if hierarchy_plan is None:
        return control
    callsite_ids = {
        int(item.callsite_id)
        for item in getattr(hierarchy_plan, "items", ())
        if isinstance(item, PlanCall)
    }
    if not callsite_ids:
        return control

    present: set[int] = set()

    def collect(block):
        if isinstance(block, ConditionalBlock):
            if block.source_node_id is not None:
                present.add(int(block.source_node_id))
            collect(block.body)
            if block.orelse is not None:
                collect(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                collect(child)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            collect(block.body)
        elif isinstance(block, CallBlock):
            collect(block.callee)

    collect(control.root)
    if not present:
        return control

    def node_position(node_id: int) -> tuple[int, int, int]:
        return _authored_node_position(graph.G, int(node_id))

    memberships = _branch_compartments(graph)
    owned: dict[tuple[int, str], set[int]] = {}
    for callsite_id in callsite_ids:
        guards = [
            (int(owner), str(arm))
            for owner, arm in memberships.get(int(callsite_id), ())
            if str(arm) in {"body", "orelse"} and int(owner) in present
        ]
        if not guards:
            continue
        # The innermost owner starts last in source order.
        owner, arm = max(guards, key=lambda item: node_position(item[0]))
        owned.setdefault((owner, arm), set()).add(int(callsite_id))
    if not owned:
        return control

    def stamp(block):
        if isinstance(block, ConditionalBlock):
            body = stamp(block.body)
            orelse = None if block.orelse is None else stamp(block.orelse)
            if block.source_node_id is None:
                return replace(block, body=body, orelse=orelse)
            source = int(block.source_node_id)
            return replace(
                block, body=body, orelse=orelse,
                body_callsite_ids=tuple(sorted({
                    *map(int, block.body_callsite_ids),
                    *owned.get((source, "body"), ()),
                })),
                orelse_callsite_ids=tuple(sorted({
                    *map(int, block.orelse_callsite_ids),
                    *owned.get((source, "orelse"), ()),
                })),
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block, blocks=tuple(stamp(child) for child in block.blocks)
            )
        if isinstance(block, (LoopBlock, WhileBlock)):
            return replace(block, body=stamp(block.body))
        if isinstance(block, CallBlock):
            return replace(block, callee=stamp(block.callee))
        return block

    return replace(control, root=stamp(control.root))


def _consume_resident_control_values(control: Any, resident: frozenset[int]):
    """Consume every region-owned value a control expression names.

    Control expressions are composed in two places: the loop composer builds
    return/break guards, while predicates and mutation guards before regions
    are carved, and the shell recovers conditional predicates afterwards
    (``_graph_control_expression``).  Both re-derive a scalar from its graph
    parents.  Once a scheduled region owns that node, the region is its one
    physical producer: the shell consumes the publication.  Re-deriving it
    would define the same source value twice, and for a loop-carried update
    the shell's copy would read the pre-update Phi.

    A subtree rooted at a resident value becomes ``value``.  A predicate whose
    root is itself resident and equal to the block's predicate id keeps no
    expression at all; for a while loop this is the designed path, where the
    condition region publishes the latch predicate.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, ControlExpression, ControlSequenceMutation,
        LoopBlock, LoopControlBlock, ParallelDeployment, ResourceScopeBlock,
        SequenceBlock, SequenceMutationBlock, StateMachineTick,
        ValidationBlock, WhileBlock,
    )

    if not resident:
        return control

    def expression(item):
        if item is None:
            return None
        if (
            item.op not in {"value", "const", "sequence_nonempty"}
            and item.value_id is not None
            and int(item.value_id) in resident
        ):
            return ControlExpression("value", value_id=int(item.value_id))
        operands = tuple(expression(operand) for operand in item.operands)
        if all(a is b for a, b in zip(operands, item.operands)):
            return item
        return replace(item, operands=operands)

    def predicate(item, predicate_value_id, carried_initials=frozenset()):
        rewritten = expression(item)
        if (
            rewritten is not None
            and rewritten.op == "value"
            and predicate_value_id is not None
            and rewritten.value_id is not None
            and int(rewritten.value_id) == int(predicate_value_id)
            and int(predicate_value_id) in resident
            # ``while go:`` with ``go`` rebound in the body: the resident
            # region publishes only the pre-loop ``go``; the next test is
            # the carried binding's update, which only the leaf can name.
            and int(predicate_value_id) not in carried_initials
        ):
            return None
        return rewritten

    def mutation(item: ControlSequenceMutation) -> ControlSequenceMutation:
        return replace(
            item,
            predicate_expression=expression(item.predicate_expression),
            argument_expressions=tuple(
                expression(argument) for argument in item.argument_expressions
            ),
        )

    def visit(block):
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(visit(child) for child in block.blocks))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=visit(block.body),
                orelse=None if block.orelse is None else visit(block.orelse),
                predicate_expression=predicate(
                    block.predicate_expression, block.predicate_value_id
                ),
            )
        if isinstance(block, LoopControlBlock):
            return replace(
                block,
                predicate_expression=predicate(
                    block.predicate_expression, block.predicate_value_id
                ),
            )
        if isinstance(block, ValidationBlock):
            return replace(
                block,
                predicate_expression=predicate(
                    block.predicate_expression, block.predicate_value_id
                ),
            )
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=visit(block.condition),
                body=visit(block.body),
                predicate_expression=predicate(
                    block.predicate_expression, block.predicate_value_id,
                    frozenset(
                        int(initial)
                        for _updated, initial in block.carried_aliases
                    ),
                ),
                sequence_mutations=tuple(map(mutation, block.sequence_mutations)),
                terminal_controls=tuple(map(visit, block.terminal_controls)),
            )
        if isinstance(block, LoopBlock):
            return replace(
                block,
                body=visit(block.body),
                sequence_mutations=tuple(map(mutation, block.sequence_mutations)),
            )
        if isinstance(block, SequenceMutationBlock):
            return replace(block, mutation=mutation(block.mutation))
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple((value, visit(body)) for value, body in block.cases),
                default=None if block.default is None else visit(block.default),
            )
        if isinstance(block, ParallelDeployment):
            return replace(block, lanes=tuple(map(visit, block.lanes)))
        if isinstance(block, CallBlock):
            return replace(block, callee=visit(block.callee))
        if isinstance(block, ResourceScopeBlock):
            return replace(block, body=visit(block.body))
        return block

    return replace(control, root=visit(control.root))


def _rehome_structured_while_predicate_regions(
    control: Any,
    dispatch_subgraphs: Iterable[Any],
) -> tuple[Any, tuple[tuple[int, int, tuple[int, ...], str], ...]]:
    """Move exact structured-predicate regions into their owning while.

    Persisted shell plans can predate structural control recovery.  In that
    form, an arithmetic region used by ``(tensor - carried).item() > eps``
    remains in the flat schedule even though the WhileBlock's expression
    consumes its result on both the entry and latch evaluations.  Rehome only
    a region whose complete deployment-node set consists of computed (not
    captured leaf) identities in exactly one while predicate.  Ambiguous or
    mixed regions retain their incumbent placement.
    """

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        LoopBlock,
        SequenceBlock,
        StatementBlock,
        WhileBlock,
    )

    region_nodes = {
        int(index): frozenset(map(
            int, subgraph.G.graph.get("deployment_nodes", ())
        ))
        for index, subgraph in enumerate(dispatch_subgraphs)
    }

    def marker_region(block: Any) -> int | None:
        if not isinstance(block, StatementBlock) or len(block.lines) != 1:
            return None
        match = re.fullmatch(r"__scheduled_region_(\d+)__", str(block.lines[0]))
        return None if match is None else int(match.group(1))

    present_regions: set[int] = set()

    def computed_expression_ids(expression: Any) -> frozenset[int]:
        if expression is None or str(expression.op) == "value":
            return frozenset()
        found = {
            int(expression.value_id)
            for _ in (0,)
            if expression.value_id is not None
        }
        for operand in expression.operands:
            found.update(computed_expression_ids(operand))
        return frozenset(found)

    loop_candidates: dict[int, set[int]] = {}

    def discover(block: Any) -> None:
        region = marker_region(block)
        if region is not None:
            present_regions.add(region)
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                discover(child)
        elif isinstance(block, ConditionalBlock):
            discover(block.body)
            if block.orelse is not None:
                discover(block.orelse)
        elif isinstance(block, LoopBlock):
            discover(block.body)
        elif isinstance(block, WhileBlock):
            loop_id = block.source_loop_node_id
            computed = computed_expression_ids(block.predicate_expression)
            if loop_id is not None and computed:
                loop_candidates[int(loop_id)] = {
                    region_index
                    for region_index, nodes in region_nodes.items()
                    if nodes and nodes.issubset(computed)
                }
            discover(block.condition)
            discover(block.body)
        elif isinstance(block, CallBlock):
            discover(block.callee)

    discover(control.root)
    owners: dict[int, list[int]] = {}
    for loop_id, candidates in loop_candidates.items():
        for region_index in candidates & present_regions:
            owners.setdefault(int(region_index), []).append(int(loop_id))
    owned = {
        int(region_index): int(loop_ids[0])
        for region_index, loop_ids in owners.items()
        if len(set(loop_ids)) == 1
    }
    if not owned:
        return control, ()

    region_order = {
        int(region_index): position
        for position, region_index in enumerate(control.region_indices)
    }
    by_loop: dict[int, list[int]] = {}
    for region_index, loop_id in owned.items():
        by_loop.setdefault(loop_id, []).append(region_index)
    for regions in by_loop.values():
        regions.sort(key=lambda value: region_order.get(value, value))

    def rewrite(block: Any) -> Any:
        region = marker_region(block)
        if region in owned:
            return SequenceBlock(())
        if isinstance(block, SequenceBlock):
            children = tuple(rewrite(child) for child in block.blocks)
            return replace(block, blocks=tuple(
                grandchild
                for child in children
                for grandchild in (
                    child.blocks if isinstance(child, SequenceBlock) else (child,)
                )
            ))
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=rewrite(block.body),
                orelse=(
                    None if block.orelse is None else rewrite(block.orelse)
                ),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=rewrite(block.body))
        if isinstance(block, WhileBlock):
            condition = rewrite(block.condition)
            prefix = tuple(
                StatementBlock((f"__scheduled_region_{region_index}__",))
                for region_index in by_loop.get(
                    (
                        -1
                        if block.source_loop_node_id is None
                        else int(block.source_loop_node_id)
                    ),
                    (),
                )
            )
            existing = (
                condition.blocks
                if isinstance(condition, SequenceBlock) else (condition,)
            )
            return replace(
                block,
                condition=SequenceBlock((*prefix, *existing)),
                body=rewrite(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=rewrite(block.callee))
        return block

    receipts = tuple(
        (
            int(loop_id),
            int(region_index),
            tuple(sorted(region_nodes[region_index])),
            "structured_while_predicate_operation",
        )
        for region_index, loop_id in sorted(owned.items())
    )
    return replace(control, root=rewrite(control.root)), receipts


def _apply_phi_initial_identity_repairs_to_control(control, receipts):
    """Project graph Phi-incumbent repairs into already-planned Control IR."""
    if control is None or not receipts:
        return control
    repairs = {
        (
            int(receipt["phi_value_id"]),
            int(receipt["missing_initial_value_id"]),
        ): int(receipt["resident_initial_value_id"])
        for receipt in receipts
    }
    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, ParallelDeployment,
        ResourceScopeBlock, SequenceBlock, StateMachineTick, WhileBlock,
    )

    def rewrite(block):
        if isinstance(block, ConditionalBlock):
            carried = []
            for true_id, false_id, initial_id, merged_id in (
                block.carried_aliases
            ):
                replacement = repairs.get((
                    int(merged_id), int(initial_id)
                ))
                if replacement is None:
                    carried.append((
                        true_id, false_id, initial_id, merged_id,
                    ))
                    continue
                carried.append((
                    replacement if int(true_id) == int(initial_id)
                    else int(true_id),
                    replacement if int(false_id) == int(initial_id)
                    else int(false_id),
                    replacement,
                    int(merged_id),
                ))
            return replace(
                block,
                body=rewrite(block.body),
                orelse=(
                    None if block.orelse is None else rewrite(block.orelse)
                ),
                carried_aliases=tuple(carried),
            )
        if isinstance(block, SequenceBlock):
            return replace(
                block, blocks=tuple(rewrite(item) for item in block.blocks)
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=rewrite(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=rewrite(block.condition),
                body=rewrite(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=rewrite(block.callee))
        if isinstance(block, ParallelDeployment):
            return replace(
                block, lanes=tuple(rewrite(lane) for lane in block.lanes)
            )
        if isinstance(block, StateMachineTick):
            return replace(
                block,
                cases=tuple((case, rewrite(body)) for case, body in block.cases),
                default=(
                    None if block.default is None else rewrite(block.default)
                ),
            )
        if isinstance(block, ResourceScopeBlock):
            return replace(block, body=rewrite(block.body))
        return block

    return replace(control, root=rewrite(control.root))

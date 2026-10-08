"""Lexical control installation: install sequence mutations, queries and external reference calls into their authored control scope (relocated from fortran_c_shell)"""

from __future__ import annotations
import ast
import os
import re
import sys
from typing import (
    Any,
    Iterable,
)
from dataclasses import replace

from .ssa_fortran_backend import FortranEmissionError
from .lexical_control_placement import (
    _authored_node_position,
    _flattened_sequence_children,
    _graph_control_expression,
    _insert_lexically,
    _loop_control_block_position,
    _region_resident_value_ids,
)
from .resident_sequence_materialization import _sequence_append_call_mutations
from ..transmogrifier.graph.edge_roles import (
    keyword_argument_name,
    ordered_arguments,
)


def _install_lexical_sequence_mutations(
    control: Any,
    graph: Any,
    dispatch_subgraphs: Iterable[Any],
    *,
    extra_mutations: Iterable[Any] = (),
    scalar_writes: Iterable[Any] = (),
):
    """Place non-loop sequence effects into their authored control scope."""

    from .control_source import (
        CallBlock,
        ConditionalBlock,
        ControlExpression,
        LoopBlock,
        LoopControlBlock,
        SequenceBlock,
        SequenceMutationBlock,
        ScalarFieldWriteBlock,
        StatementBlock,
        WhileBlock,
    )
    from .glsl_deployment_strategy import (
        _branch_compartments,
        _retained_control_value_id,
        _source_control_records,
    )

    mutations = tuple((
        *_sequence_append_call_mutations(graph.G),
        *tuple(extra_mutations),
        *tuple(scalar_writes),
    ))
    if not mutations:
        return control, ()

    memberships = _branch_compartments(graph)

    def enclosing_loop_span(loop_block) -> tuple[int, int] | None:
        source_id = getattr(loop_block, "source_loop_node_id", None)
        if source_id is None:
            return None
        expression = graph.G.nodes.get(int(source_id), {}).get("expr_obj")
        start = getattr(expression, "lineno", None)
        end = getattr(expression, "end_lineno", None)
        if start is None or end is None:
            return None
        return int(start), int(end)

    def insert_in_enclosing_loop(block, item_block, node_id: int):
        """Insert ``item_block`` ordered by position inside the innermost loop
        whose authored span contains ``node_id``; at this level otherwise.

        A guard synthesized for an effect authored inside a loop body used to
        be ordered among the ROOT statements, which put it after the loop:
        the effect never ran and its predicate read loop-local values in
        ``while_exit``.
        """

        line = node_position(int(node_id))[0]

        def descend(candidate):
            if isinstance(candidate, SequenceBlock):
                for index, child in enumerate(candidate.blocks):
                    replaced, inserted = descend(child)
                    if inserted:
                        return replace(
                            candidate,
                            blocks=(*candidate.blocks[:index], replaced,
                                    *candidate.blocks[index + 1:]),
                        ), True
                return candidate, False
            if isinstance(candidate, ConditionalBlock):
                body, inserted = descend(candidate.body)
                if inserted:
                    return replace(candidate, body=body), True
                if candidate.orelse is not None:
                    orelse, inserted = descend(candidate.orelse)
                    if inserted:
                        return replace(candidate, orelse=orelse), True
                return candidate, False
            if isinstance(candidate, CallBlock):
                callee, inserted = descend(candidate.callee)
                return (replace(candidate, callee=callee), True) if inserted else (candidate, False)
            if isinstance(candidate, (LoopBlock, WhileBlock)):
                span = enclosing_loop_span(candidate)
                if span is None or not (span[0] <= line <= span[1]):
                    return candidate, False
                body, inserted = descend(candidate.body)
                if not inserted:
                    body = insert_ordered(candidate.body, item_block)
                return replace(candidate, body=body), True
            return candidate, False

        replaced, inserted = descend(block)
        if inserted:
            return replaced
        return insert_ordered(block, item_block)

    existing_effect_ids: set[int] = set()

    def gather(block):
        if isinstance(block, ScalarFieldWriteBlock):
            existing_effect_ids.add(int(block.effect_node_id))
        elif isinstance(block, SequenceMutationBlock):
            existing_effect_ids.add(int(block.mutation.effect_node_id))
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather(child)
        elif isinstance(block, ConditionalBlock):
            gather(block.body)
            if block.orelse is not None:
                gather(block.orelse)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            existing_effect_ids.update(
                int(item.effect_node_id) for item in block.sequence_mutations
            )
            if isinstance(block, WhileBlock):
                gather(block.condition)
            gather(block.body)
        elif isinstance(block, CallBlock):
            gather(block.callee)

    gather(control.root)
    mutations = tuple(
        item for item in mutations
        if int(item.effect_node_id) not in existing_effect_ids
    )

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
        if isinstance(block, ScalarFieldWriteBlock):
            return node_position(int(block.effect_node_id))
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

    def insert_ordered(block, mutation_block):
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(_flattened_sequence_children(block))
        ]
        decorated.append((
            block_position(mutation_block), len(decorated), mutation_block,
        ))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                decorated, key=lambda item: (item[0], item[1])
            )
        ))

    retained_mutation_records = (
        graph.G.graph.get("source_sequence_mutation_records") or {}
    )

    def insert_in_conditional(block, owner_id, arm, mutation_block):
        if isinstance(block, ConditionalBlock):
            if int(block.source_node_id or -1) == int(owner_id):
                if str(arm) == "body":
                    return replace(
                        block,
                        body=insert_ordered(block.body, mutation_block),
                    ), True
                return replace(
                    block,
                    orelse=insert_ordered(
                        block.orelse or SequenceBlock(()), mutation_block,
                    ),
                ), True
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_in_conditional(
                    block.orelse, owner_id, arm, mutation_block
                )
                if inserted:
                    return replace(block, orelse=orelse), True
            return block, False
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = insert_in_conditional(
                    child, owner_id, arm, mutation_block
                )
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, LoopBlock):
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            return replace(block, body=body), inserted
        if isinstance(block, WhileBlock):
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, mutation_block
            )
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_in_conditional(
                block.callee, owner_id, arm, mutation_block
            )
            return replace(block, callee=callee), inserted
        return block, False

    # A retained loop's effects were lowered after its body, each behind its
    # own guard predicate.  When the body is lexical control ending in a
    # terminal edge (break, continue, return), "after the body" is
    # unreachable code: the effect never runs, and every value it consumes
    # there (the guard's own operands above all) drags the call producing
    # that value out of the body with it.  An authored effect belongs at its
    # authored position inside the body.  When its guarding conditional is
    # present in the tree it is installed in that arm like any other effect
    # (the arm is its guard).  Otherwise -- the conditional owned no region
    # and was projected away -- it keeps its composed guard and is ordered
    # among the body's statements by source position.
    def conditional_owner_ids(block) -> set[int]:
        owners: set[int] = set()
        if isinstance(block, ConditionalBlock):
            if block.source_node_id is not None:
                owners.add(int(block.source_node_id))
            owners |= conditional_owner_ids(block.body)
            if block.orelse is not None:
                owners |= conditional_owner_ids(block.orelse)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                owners |= conditional_owner_ids(child)
        elif isinstance(block, (LoopBlock, WhileBlock)):
            owners |= conditional_owner_ids(block.body)
        elif isinstance(block, CallBlock):
            owners |= conditional_owner_ids(block.callee)
        return owners

    arm_relocated: list = []

    def relocate(block):
        if isinstance(block, SequenceBlock):
            return replace(
                block, blocks=tuple(relocate(child) for child in block.blocks)
            )
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=relocate(block.body),
                orelse=None if block.orelse is None else relocate(block.orelse),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=relocate(block.callee))
        if isinstance(block, (LoopBlock, WhileBlock)):
            body = relocate(block.body)
            owners = conditional_owner_ids(body)
            kept = []
            for item in block.sequence_mutations:
                effect_id = int(item.effect_node_id)
                guards = tuple(
                    (int(owner), str(arm))
                    for owner, arm in memberships.get(effect_id, ())
                    if str(arm) in {"body", "orelse"} and int(owner) in owners
                )
                if guards:
                    arm_relocated.append(
                        replace(item, predicate_expression=None)
                    )
                    continue
                if (
                    effect_id in graph.G
                    and node_position(effect_id)[0] < (1 << 30)
                ):
                    body = insert_ordered(body, SequenceMutationBlock(item))
                    continue
                kept.append(item)
            return replace(block, body=body, sequence_mutations=tuple(kept))
        return block

    control = replace(control, root=relocate(control.root))
    if arm_relocated:
        relocated_ids = {int(item.effect_node_id) for item in arm_relocated}
        mutations = (
            *arm_relocated,
            *(
                item for item in mutations
                if int(item.effect_node_id) not in relocated_ids
            ),
        )

    root = control.root
    unplaced = []
    for mutation in mutations:
        mutation_block = (mutation if isinstance(mutation, ScalarFieldWriteBlock)
                          else SequenceMutationBlock(mutation))
        guarded = tuple(dict.fromkeys((
            *tuple(
            (int(owner), str(arm))
            for owner, arm in memberships.get(int(mutation.effect_node_id), ())
            if str(arm) in {"body", "orelse"}
            ),
            *tuple(
                (int(owner), str(arm))
                for owner, arm in (
                    retained_mutation_records.get(
                        int(mutation.effect_node_id), {}
                    ).get("branch_memberships", ())
                )
                if str(arm) in {"body", "orelse"}
            ),
        )))
        if os.environ.get("TURING_DEBUG_SEQUENCE_MUTATION"):
            print(
                "DEBUGMUTATION "
                f"fn={graph.G.graph.get('function_name')} "
                f"effect={int(mutation.effect_node_id)} "
                f"sequence={getattr(mutation, 'sequence_value_id', None)} "
                f"guarded={guarded}",
                file=sys.stderr,
            )
        inserted = False
        # Innermost authored condition has the latest source position.
        for owner_id, arm in sorted(
            guarded, key=lambda item: node_position(item[0]), reverse=True,
        ):
            root, inserted = insert_in_conditional(
                root, owner_id, arm, mutation_block
            )
            if inserted:
                break
        if not inserted:
            # A branch containing only structural storage effects owns no
            # numerical region, so ordinary region overlay has no marker from
            # which to materialize its ConditionalBlock.  Reconstruct that
            # authored branch from the retained pre-fold control record, then
            # place the complete wrapper inside its nearest surviving outer
            # conditional (or at the lexical root).
            synthesized = None
            remaining_guards = guarded
            if guarded:
                owner_id, arm = sorted(
                    guarded,
                    key=lambda item: node_position(item[0]),
                    reverse=True,
                )[0]
                record = _source_control_records(graph.G).get(owner_id)
                owner_node_id = (
                    int(owner_id) if int(owner_id) in graph.G else next((
                        int(node_id)
                        for node_id, data in graph.G.nodes(data=True)
                        if int(data.get("value_id", node_id)) == int(owner_id)
                    ), None)
                )
                owner_data = (
                    {} if owner_node_id is None
                    else graph.G.nodes[int(owner_node_id)]
                )
                expression = (
                    record.get("expression")
                    if record is not None
                    else owner_data.get("expr_obj")
                )
                recorded_predicate_id = (
                    None if record is None else record.get("predicate_id")
                )
                if recorded_predicate_id is None:
                    recorded_predicate_id = next((
                        int(parent)
                        for parent, role in owner_data.get("parents", ())
                        if str(role) == "test"
                    ), None)
                predicate_id = (
                    None if not isinstance(expression, (ast.If, ast.IfExp))
                    else _retained_control_value_id(
                        graph.G,
                        recorded_predicate_id,
                        expression.test,
                    )
                )
                if predicate_id is None and isinstance(
                    expression, (ast.If, ast.IfExp)
                ):
                    direct_test = next((
                        int(parent)
                        for parent, role in owner_data.get("parents", ())
                        if str(role) == "test" and int(parent) in graph.G
                    ), None)
                    if direct_test is not None:
                        predicate_id = int(
                            graph.G.nodes[direct_test].get(
                                "value_id", direct_test
                            )
                        )
                if (
                    predicate_id is None
                    and recorded_predicate_id is not None
                    and isinstance(expression, (ast.If, ast.IfExp))
                ):
                    # The conditional node may have been folded out after its
                    # immutable source-control record was published.  That
                    # record already speaks in the canonical value identity
                    # consumed by region/control SSA; retain it directly.
                    predicate_id = int(recorded_predicate_id)
                if os.environ.get("TURING_DEBUG_SEQUENCE_MUTATION"):
                    print(
                        "DEBUGMUTATION-GUARD "
                        f"effect={int(mutation.effect_node_id)} "
                        f"owner={owner_id} owner_node={owner_node_id} "
                        f"recorded_predicate={recorded_predicate_id} "
                        f"predicate={predicate_id} "
                        f"expression={type(expression).__name__}",
                        file=sys.stderr,
                    )
                if predicate_id is not None:
                    empty = SequenceBlock(())
                    synthesized = ConditionalBlock(
                        int(predicate_id),
                        mutation_block if arm == "body" else empty,
                        mutation_block if arm == "orelse" else None,
                        predicate_expression=_graph_control_expression(
                            graph.G, int(predicate_id),
                            resident=_region_resident_value_ids(
                                dispatch_subgraphs
                            ),
                        ),
                        source_node_id=int(owner_id),
                    )
                    remaining_guards = tuple(
                        item for item in guarded if item != (owner_id, arm)
                    )
            if synthesized is not None:
                for owner_id, arm in sorted(
                    remaining_guards,
                    key=lambda item: node_position(item[0]),
                    reverse=True,
                ):
                    root, inserted = insert_in_conditional(
                        root, owner_id, arm, synthesized
                    )
                    if inserted:
                        break
                if not inserted:
                    root = insert_in_enclosing_loop(
                        root, synthesized, int(mutation.effect_node_id)
                    )
            elif guarded:
                unplaced.append(int(mutation.effect_node_id))
            else:
                root = insert_in_enclosing_loop(
                    root, mutation_block, int(mutation.effect_node_id)
                )
    return replace(control, root=root), tuple(unplaced)


def _control_expression_value_ids(expression: Any) -> frozenset[int]:
    if expression is None:
        return frozenset()
    return frozenset((
        *((int(expression.value_id),) if expression.value_id is not None else ()),
        *(
            value_id
            for operand in expression.operands
            for value_id in _control_expression_value_ids(operand)
        ),
    ))


def _install_external_reference_calls(control: Any, graph: Any, dispatch_subgraphs):
    """Retain native boundaries and dispatcher operations as lexical control.

    This is intentionally the last control transformation. Earlier region and
    hierarchy passes therefore cannot mistake an external capability for a
    Python function to pursue or for a numerical region to fuse.
    """

    from .control_source import (
        CallBlock, ConditionalBlock, DispatchBlock, ExternalReferenceCallBlock, LoopBlock,
        ResourceScopeBlock, SequenceBlock, StatementBlock, WhileBlock,
    )
    from .glsl_deployment_strategy import _ast_source_signature, _branch_compartments

    graph_obj = graph.G
    memberships = {node: set(owners) for node, owners in _branch_compartments(graph).items()}
    expression_nodes = {id(data.get("expr_obj")): int(node_id)
                        for node_id, data in graph_obj.nodes(data=True)
                        if isinstance(data.get("expr_obj"), ast.AST)}
    signature_nodes = {}
    for node_id, data in graph_obj.nodes(data=True):
        expression = data.get("expr_obj")
        if isinstance(expression, ast.AST) and getattr(expression, "lineno", None) is not None:
            signature_nodes.setdefault(_ast_source_signature(expression), []).append(int(node_id))
    resource_nodes = {
        int(node_id): data["expr_obj"]
        for node_id, data in graph_obj.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Try)
        and getattr(data["expr_obj"], "_turing_resource_scope", None)
    }
    source_order = {}
    node_source_order = {}
    def index_source(node, loop_owners=()):
        if not isinstance(node, ast.AST):
            return
        source_order[id(node)] = len(source_order)
        node_id = expression_nodes.get(id(node))
        if node_id is None and getattr(node, "lineno", None) is not None:
            candidates = signature_nodes.get(_ast_source_signature(node), ())
            if len(candidates) == 1:
                node_id = candidates[0]
        if node_id is not None:
            node_source_order[node_id] = source_order[id(node)]
            memberships.setdefault(node_id, set()).update(loop_owners)
        for field, value in ast.iter_fields(node):
            owners = loop_owners
            if isinstance(node, (ast.For, ast.While)) and node_id is not None:
                owners += ((node_id, field),)
            for child in value if isinstance(value, list) else (value,):
                if isinstance(child, ast.AST):
                    index_source(child, owners)
    for statement in graph_obj.graph.get("function_body", ()):
        index_source(statement)

    def node_position(node_id):
        data = graph_obj.nodes.get(int(node_id), {})
        expression = data.get("expr_obj")
        span = data.get("source_span") or {}
        return (
            int(getattr(expression, "lineno", span.get("line", 1 << 30)) or (1 << 30)),
            int(getattr(expression, "col_offset", span.get("column", 0)) or 0),
            node_source_order.get(int(node_id), int(node_id)),
        )

    region_positions = {
        int(index): min(
            (node_position(node_id) for node_id in subgraph.G.graph.get(
                "deployment_nodes", ()
            )),
            default=(1 << 30, 0, int(index)),
        )
        for index, subgraph in enumerate(dispatch_subgraphs)
    }

    def block_position(block):
        if isinstance(block, ResourceScopeBlock):
            return node_position(block.source_scope_id)
        if isinstance(block, (ExternalReferenceCallBlock, DispatchBlock)):
            return node_position(block.callsite_id)
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            line = block.lines[0]
            if line.startswith("__scheduled_region_") and line.endswith("__"):
                return region_positions.get(
                    int(line[len("__scheduled_region_"):-2]),
                    (1 << 30, 0, 0),
                )
        if isinstance(block, ConditionalBlock) and block.source_node_id is not None:
            return node_position(block.source_node_id)
        if isinstance(block, (LoopBlock, WhileBlock)):
            source_id = block.source_loop_node_id
            if source_id is not None:
                return node_position(source_id)
        if isinstance(block, CallBlock):
            return node_position(block.callsite_id)
        if isinstance(block, SequenceBlock) and block.blocks:
            return min(map(block_position, block.blocks))
        return (1 << 30, 0, 0)

    def insert_ordered(block, external):
        decorated = [
            (block_position(child), index, child)
            for index, child in enumerate(_flattened_sequence_children(block))
        ]
        decorated.append((block_position(external), len(decorated), external))
        return SequenceBlock(tuple(
            child for _position, _index, child in sorted(
                decorated, key=lambda item: (item[0], item[1])
            )
        ))

    def insert_in_conditional(block, owner_id, arm, external):
        if isinstance(block, ResourceScopeBlock):
            if block.source_scope_id == owner_id and arm == "body":
                return replace(block, body=insert_ordered(block.body, external)), True
            body, inserted = insert_in_conditional(block.body, owner_id, arm, external)
            return replace(block, body=body), inserted
        if isinstance(block, ConditionalBlock):
            if int(block.source_node_id or -1) == int(owner_id):
                if arm == "body":
                    return replace(block, body=insert_ordered(block.body, external)), True
                return replace(
                    block,
                    orelse=insert_ordered(block.orelse or SequenceBlock(()), external),
                ), True
            body, inserted = insert_in_conditional(block.body, owner_id, arm, external)
            if inserted:
                return replace(block, body=body), True
            if block.orelse is not None:
                orelse, inserted = insert_in_conditional(
                    block.orelse, owner_id, arm, external
                )
                if inserted:
                    return replace(block, orelse=orelse), True
        elif isinstance(block, SequenceBlock):
            children = []
            for child in block.blocks:
                projected, inserted = insert_in_conditional(
                    child, owner_id, arm, external
                )
                children.append(projected)
                if inserted:
                    children.extend(block.blocks[len(children):])
                    return SequenceBlock(tuple(children)), True
        elif isinstance(block, (LoopBlock, WhileBlock)):
            if block.source_loop_node_id == owner_id:
                if arm == "body":
                    return replace(block, body=insert_ordered(block.body, external)), True
                if arm == "test" and isinstance(block, WhileBlock):
                    return replace(block, condition=insert_ordered(block.condition, external)), True
            body, inserted = insert_in_conditional(
                block.body, owner_id, arm, external
            )
            if inserted:
                return replace(block, body=body), True
        elif isinstance(block, CallBlock):
            callee, inserted = insert_in_conditional(
                block.callee, owner_id, arm, external
            )
            if inserted:
                return replace(block, callee=callee), True
        return block, False

    root = control.root
    unplaced = []
    installed = []
    pending_dispatch = {}
    for node_id, data in sorted(graph_obj.nodes(data=True), key=lambda item: node_position(item[0])):
        expression = data.get("expr_obj")
        contract = dict(
            data.get("extraction_contract")
            or (data.get("attributes") or {}).get("extraction_contract")
            or getattr(expression, "_extraction_contract", None)
            or {}
        )
        dispatch = (data.get("attributes") or {}).get("dispatch_operation")
        if not isinstance(expression, ast.Call) or (
            contract.get("action") != "use_native" and not dispatch
        ):
            continue
        if dispatch and (
            any(not (
                scope[0] in {"Try", "If"}
                or scope[0] in {"For", "While"} and scope[2] == "body"
                or scope[0] == "While" and scope[2] == "test"
            ) for scope in dispatch.get("control_scopes", ()))
            or any(
                isinstance(graph_obj.nodes.get(owner, {}).get("expr_obj"), ast.Try)
                and owner not in resource_nodes
                for owner, _arm in memberships.get(int(node_id), ())
            )
        ):
            raise FortranEmissionError(
                "dispatcher operation requires retained control scope lowering: "
                f"operation={dispatch['operation']}, scopes={dispatch['control_scopes']}"
            )
        if os.environ.get("TURING_DEBUG_EXTERNAL_REFERENCE"):
            print(
                "DEBUG-EXTERNAL-REFERENCE "
                f"function={graph_obj.graph.get('function_name')} "
                f"callsite={int(node_id)} identity={contract.get('identity')} "
                f"parents={data.get('parents')} "
                f"parameters={contract.get('parameters')}",
                file=sys.stderr,
            )
        parameters = dict(contract.get("parameters") or {})
        result_dtype = parameters.get("result_dtype")
        if dispatch:
            operation = str(dispatch["operation"])
            result_dtype = (
                "opaque_ref" if operation.endswith("_create") else
                "bool" if operation in {
                    "condition_acquire", "condition_wait", "event_wait",
                    "event_is_set", "thread_is_alive",
                } else "none"
            )
        if result_dtype is None:
            unplaced.append((int(node_id), "native boundary has no recorded result_dtype"))
            continue

        parents = tuple(data.get("parents") or ())

        def parent_value_id(parent_id):
            if parent_id not in graph_obj:
                return None
            return int(graph_obj.nodes[parent_id].get("value_id", parent_id))

        arguments = tuple(
            result for parent_id in ordered_arguments(parents)
            for result in (parent_value_id(parent_id),) if result is not None
        )
        keywords = tuple(
            (name, result)
            for parent_id, role in parents
            for name in (keyword_argument_name(role),)
            if name is not None
            for result in (parent_value_id(parent_id),)
            if result is not None
        )
        if len(arguments) != len(expression.args) or len(keywords) != len(expression.keywords):
            if dispatch:
                raise FortranEmissionError(
                    f"dispatcher arguments lack SSA identities: {dispatch['operation']} at node {node_id}"
                )
            unplaced.append((int(node_id), "native boundary arguments lack SSA identities"))
            continue
        external = DispatchBlock(
            callsite_id=int(node_id), operation=str(dispatch["operation"]),
            argument_value_ids=arguments, keyword_argument_value_ids=keywords,
            result_value_id=int(data.get("value_id", node_id)),
            result_dtype=str(result_dtype),
        ) if dispatch else ExternalReferenceCallBlock(
            callsite_id=int(node_id),
            identity=str(contract["identity"]),
            argument_value_ids=arguments,
            keyword_argument_value_ids=keywords,
            result_value_id=int(data.get("value_id", node_id)),
            result_dtype=str(result_dtype),
            shell_abi=str(parameters["shell_abi"]),
            external_domain=str(parameters["external_domain"]),
            native_abi=str(parameters.get("native_abi") or ""),
            runtime_owner=str(parameters.get("runtime_owner") or ""),
            shell_profiles=tuple(map(str, parameters.get("shell_profiles") or ())),
        )
        if dispatch:
            pending_dispatch[int(node_id)] = external
            continue
        guarded = tuple(
            (int(owner), str(arm))
            for owner, arm in memberships.get(int(node_id), ())
            if str(arm) in {"body", "orelse"}
        )
        inserted = False
        for owner_id, arm in sorted(
            guarded, key=lambda item: node_position(item[0]), reverse=True
        ):
            root, inserted = insert_in_conditional(root, owner_id, arm, external)
            if inserted:
                break
        if guarded and not inserted:
            unplaced.append((int(node_id), "native boundary conditional scope is not retained"))
            continue
        if not inserted:
            root = insert_ordered(root, external)
        if not dispatch:
            installed.append(int(node_id))

    def anchors(block):
        if isinstance(block, ResourceScopeBlock):
            return {block.source_scope_id}
        if isinstance(block, (DispatchBlock, ExternalReferenceCallBlock, CallBlock)):
            return {block.callsite_id}
        if isinstance(block, ConditionalBlock):
            return {block.source_node_id} if block.source_node_id is not None else set()
        if isinstance(block, (LoopBlock, WhileBlock)):
            return {block.source_loop_node_id} if block.source_loop_node_id is not None else set()
        if isinstance(block, SequenceBlock):
            return set().union(*(anchors(child) for child in block.blocks))
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            match = re.fullmatch(r"__scheduled_region_(\d+)__", block.lines[0])
            if match:
                return set(dispatch_subgraphs[int(match[1])].G.graph.get("deployment_nodes", ()))
        return set()

    def wrap_scope(block, scope_id, cleanup):
        def owned(child):
            nodes = anchors(child)
            inside = {node for node in nodes if (scope_id, "body") in memberships.get(node, ())}
            if inside and inside != nodes and not isinstance(child, SequenceBlock):
                raise FortranEmissionError("resource scope crosses a compiled region boundary")
            return bool(nodes) and inside == nodes
        if owned(block):
            return ResourceScopeBlock(block, cleanup, scope_id), True
        if isinstance(block, SequenceBlock):
            indices = [index for index, child in enumerate(block.blocks) if owned(child)]
            if indices:
                first, last = indices[0], indices[-1]
                if indices != list(range(first, last + 1)):
                    raise FortranEmissionError("resource scope has noncontiguous control ownership")
                grouped = ResourceScopeBlock(SequenceBlock(block.blocks[first:last + 1]), cleanup, scope_id)
                return SequenceBlock((*block.blocks[:first], grouped, *block.blocks[last + 1:])), True
            for index, child in enumerate(block.blocks):
                projected, found = wrap_scope(child, scope_id, cleanup)
                if found:
                    return SequenceBlock((*block.blocks[:index], projected, *block.blocks[index + 1:])), True
        elif isinstance(block, (ConditionalBlock, LoopBlock, WhileBlock, ResourceScopeBlock)):
            body, found = wrap_scope(block.body, scope_id, cleanup)
            if found:
                return replace(block, body=body), True
            if isinstance(block, ConditionalBlock) and block.orelse is not None:
                orelse, found = wrap_scope(block.orelse, scope_id, cleanup)
                if found:
                    return replace(block, orelse=orelse), True
        return block, False

    # Materialize lexical cleanup around the existing numerical/control body.
    # Nested scopes are wrapped first; their source node anchors then belong
    # to the enclosing scope just like any other authored body statement.
    for scope_id, expression in sorted(resource_nodes.items(),
                                      key=lambda item: len(memberships.get(item[0], ())), reverse=True):
        cleanup_ids = [node_id for node_id in pending_dispatch
                       if (scope_id, "finalbody") in memberships.get(node_id, ())]
        if len(cleanup_ids) != 1 or pending_dispatch[cleanup_ids[0]].operation != "condition_release":
            raise FortranEmissionError("resource scope has no exact condition release cleanup")
        cleanup = (replace(pending_dispatch.pop(cleanup_ids[0]), result_value_id=None),)
        root, found = wrap_scope(root, scope_id, cleanup)
        if not found:
            # A scope containing only dispatcher effects has no numeric marker.
            # Root scopes can still be anchored explicitly; a missing enclosing
            # branch/loop must not be guessed into existence here.
            if any(owner not in resource_nodes for owner, _ in memberships.get(scope_id, ())):
                raise FortranEmissionError("resource scope's enclosing control is not retained")
            root = insert_ordered(root, ResourceScopeBlock(SequenceBlock(()), cleanup, scope_id))
    for node_id, operation in pending_dispatch.items():
        guarded = [(owner, arm) for owner, arm in memberships.get(node_id, ())
                   if arm in {"body", "orelse", "test"}]
        inserted = False
        for owner, arm in sorted(guarded, key=lambda item: len(memberships.get(item[0], ())), reverse=True):
            root, inserted = insert_in_conditional(root, owner, arm, operation)
            # A missing inner owner cannot be replaced with an outer one:
            # that would hoist a wait out of its loop or conditional.
            break
        if guarded and not inserted:
            raise FortranEmissionError("dispatcher operation's control owner is not retained")
        if not inserted:
            root = insert_ordered(root, operation)
    return replace(control, root=root), tuple(installed), tuple(unplaced)


def _control_block_consumes_values(block: Any, value_ids: Iterable[int]) -> bool:
    """Whether a control subtree observes any exact SSA identity."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        ScalarFieldWriteBlock, SequenceMutationBlock, SequenceQueryBlock, ValidationBlock,
        WhileBlock,
    )

    wanted = set(map(int, value_ids))
    if not wanted:
        return False
    if isinstance(block, ScalarFieldWriteBlock):
        return bool(_control_expression_value_ids(block.value_expression).intersection(wanted))
    if isinstance(block, ConditionalBlock):
        if (
            int(block.predicate_value_id) in wanted
            or _control_expression_value_ids(
                block.predicate_expression
            ).intersection(wanted)
        ):
            return True
        return (
            _control_block_consumes_values(block.body, wanted)
            or block.orelse is not None
            and _control_block_consumes_values(block.orelse, wanted)
        )
    if isinstance(block, ValidationBlock):
        return (
            int(block.predicate_value_id) in wanted
            or bool(_control_expression_value_ids(
                block.predicate_expression
            ).intersection(wanted))
        )
    if isinstance(block, SequenceMutationBlock):
        mutation = block.mutation
        return bool(
            set(map(int, mutation.argument_value_ids)).intersection(wanted)
            or _control_expression_value_ids(
                mutation.predicate_expression
            ).intersection(wanted)
        )
    if isinstance(block, SequenceQueryBlock):
        consumed = {
            int(block.sequence_value_id),
            *map(int, block.reduction_prefix_value_ids),
            *map(int, block.reduction_suffix_value_ids),
        }
        if block.default_value_id is not None:
            consumed.add(int(block.default_value_id))
        return bool(consumed.intersection(wanted))
    if isinstance(block, CallBlock):
        return bool(
            {int(caller) for caller, _callee in block.argument_bindings}
            .intersection(wanted)
            or _control_block_consumes_values(block.callee, wanted)
        )
    if isinstance(block, SequenceBlock):
        return any(
            _control_block_consumes_values(child, wanted)
            for child in block.blocks
        )
    if isinstance(block, LoopBlock):
        return _control_block_consumes_values(block.body, wanted)
    if isinstance(block, WhileBlock):
        return (
            _control_block_consumes_values(block.condition, wanted)
            or _control_block_consumes_values(block.body, wanted)
        )
    return False


def _schedule_sequence_query_dependencies(root: Any) -> Any:
    """Place a producer-loop/query unit before its first lexical consumer."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceQueryBlock, WhileBlock,
    )

    if isinstance(root, SequenceBlock):
        scheduled_children = [
            _schedule_sequence_query_dependencies(child)
            for child in root.blocks
        ]
        # Sequential composition is associative: nested SequenceBlocks carry
        # no scope or closure.  Flatten them before dependency scheduling so
        # a generator loop appended in one planner group can move ahead of a
        # consumer retained in an earlier group.
        children = [
            grandchild
            for child in scheduled_children
            for grandchild in (
                child.blocks if isinstance(child, SequenceBlock) else (child,)
            )
        ]
        changed = True
        while changed:
            changed = False
            for loop_index, loop in enumerate(children):
                if (
                    not isinstance(loop, LoopBlock)
                    or loop.source_loop_node_id is None
                ):
                    continue
                query_indexes = [
                    index
                    for index, candidate in enumerate(children)
                    if isinstance(candidate, SequenceQueryBlock)
                    and candidate.producer_loop_node_id is not None
                    and int(candidate.producer_loop_node_id)
                    == int(loop.source_loop_node_id)
                ]
                queries = [children[index] for index in query_indexes]
                if not queries:
                    continue
                produced = {
                    int(value_id)
                    for query in queries
                    for value_id in (
                        query.result_value_id, *query.result_alias_ids,
                    )
                }
                unit_indexes = {loop_index, *query_indexes}
                consumer_index = next((
                    index
                    for index, candidate in enumerate(children)
                    if index not in unit_indexes
                    and _control_block_consumes_values(candidate, produced)
                ), None)
                # Source-position insertion and later structural recovery may
                # separate a generator loop from its query, or even leave the
                # query before the loop.  The pair is one semantic producer
                # unit.  Normalize it at its earliest existing position, and
                # move it farther forward when an earlier consumer requires
                # that dominance edge.
                anchor = min(unit_indexes)
                if consumer_index is not None:
                    anchor = min(anchor, consumer_index)
                rebuilt = [
                    candidate
                    for index, candidate in enumerate(children)
                    if index not in unit_indexes
                ]
                adjusted_anchor = sum(
                    1
                    for index in range(anchor)
                    if index not in unit_indexes
                )
                rebuilt[adjusted_anchor:adjusted_anchor] = [loop, *queries]
                if rebuilt != children:
                    children = rebuilt
                    changed = True
                    break
        return replace(root, blocks=tuple(children))
    if isinstance(root, ConditionalBlock):
        return replace(
            root,
            body=_schedule_sequence_query_dependencies(root.body),
            orelse=(
                None if root.orelse is None
                else _schedule_sequence_query_dependencies(root.orelse)
            ),
        )
    if isinstance(root, LoopBlock):
        return replace(
            root, body=_schedule_sequence_query_dependencies(root.body)
        )
    if isinstance(root, WhileBlock):
        return replace(
            root,
            condition=_schedule_sequence_query_dependencies(root.condition),
            body=_schedule_sequence_query_dependencies(root.body),
        )
    if isinstance(root, CallBlock):
        return replace(
            root, callee=_schedule_sequence_query_dependencies(root.callee)
        )
    return root


def _install_lexical_sequence_queries(
    control: Any,
    graph: Any,
    dispatch_subgraphs: Iterable[Any],
):
    """Replace supported generator consumers with resident sequence queries."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
        SequenceQueryBlock, StatementBlock, WhileBlock,
    )

    graph_obj = graph.G
    queries = []
    unsupported = []
    for node_id, data in sorted(graph_obj.nodes(data=True)):
        expression = data.get("expr_obj")
        if not (
            isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Name)
            and expression.func.id in {"bool", "len", "next", "sum", "max"}
        ):
            continue
        positioned_arguments = tuple(sorted(
            (
                int(str(role).split(":", 1)[1]),
                int(parent),
            )
            for parent, role in data.get("parents") or ()
            if str(role).startswith("arg:") and int(parent) in graph_obj
        ))
        arguments = tuple(parent for _position, parent in positioned_arguments)
        if not arguments:
            continue
        identity = (data.get("attributes") or {}).get("extraction_identity")
        result_id = int(data.get("value_id", node_id))
        result_alias_ids = tuple(dict.fromkeys(
            int(graph_obj.nodes[child].get("value_id", child))
            for child, role in data.get("children") or ()
            if str(role) == "lhs"
            and int(child) in graph_obj
            and isinstance(
                graph_obj.nodes[int(child)].get("expr_obj"), ast.Name
            )
        ))
        if expression.func.id == "max":
            starred = tuple(
                (position, wrapper_id)
                for position, wrapper_id in positioned_arguments
                if isinstance(
                    graph_obj.nodes[wrapper_id].get("expr_obj"), ast.Starred
                )
            )
            if len(starred) != 1:
                continue
            starred_position, wrapper_id = starred[0]
            sequence_id = next((
                int(parent)
                for parent, role in graph_obj.nodes[wrapper_id].get("parents") or ()
                if str(role) == "value" and int(parent) in graph_obj
            ), None)
            if sequence_id is None:
                unsupported.append(int(node_id))
                continue
            sequence_data = graph_obj.nodes[sequence_id]
            sequence_attributes = sequence_data.get("attributes") or {}
            if not (
                sequence_data.get("type") == "LoopResult"
                and sequence_attributes.get("result_kind") == "collection"
            ):
                unsupported.append(int(node_id))
                continue
            prefix = tuple(
                parent for position, parent in positioned_arguments
                if position < starred_position
            )
            suffix = tuple(
                parent for position, parent in positioned_arguments
                if position > starred_position
            )
            if not prefix:
                unsupported.append(int(node_id))
                continue
            queries.append(SequenceQueryBlock(
                result_value_id=result_id,
                sequence_value_id=sequence_id,
                operation="maximum",
                source_call_node_id=int(node_id),
                extraction_identity=identity,
                result_alias_ids=result_alias_ids,
                producer_loop_node_id=int(sequence_attributes["loop_id"]),
                reduction_prefix_value_ids=prefix,
                reduction_suffix_value_ids=suffix,
            ))
            continue
        sequence_id = arguments[0]
        sequence_data = graph_obj.nodes[sequence_id]
        sequence_attributes = sequence_data.get("attributes") or {}
        producer_loop_node_id = None
        if (
            sequence_data.get("type") == "LoopResult"
            and sequence_attributes.get("result_kind") == "collection"
        ):
            producer_loop_node_id = int(sequence_attributes["loop_id"])
        elif (
            expression.func.id == "len"
            and sequence_attributes.get("aggregate_kind")
            in {"list", "bytes", "bytearray", "tuple"}
            and sequence_attributes.get("producer_kind")
            in {"sequence_materialization", "aggregate_materialization"}
        ):
            producer_loop_node_id = next((
                int(parent)
                for parent, role in sequence_data.get("parents") or ()
                if str(role) == "generators"
            ), None)
        local_list_truth = (
            expression.func.id in {"bool", "len"}
            and len(arguments) == 1
            and sequence_attributes.get("aggregate_kind") in {"list", "tuple"}
            and sequence_attributes.get("producer_kind") in {"aggregate", "aggregate_materialization"}
            and (data.get("attributes") or {}).get("extraction_identity")
                == f"builtins.{expression.func.id}"
        )
        if producer_loop_node_id is None and not local_list_truth:
            continue
        if os.environ.get("TURING_DEBUG_BUILTIN_SELECTION"):
            print(
                "DEBUGQUERY "
                f"fn={graph_obj.graph.get('function_name')} "
                f"call={int(node_id)} result={result_id} "
                f"aliases={result_alias_ids} sequence={sequence_id}",
                file=sys.stderr,
            )
        if expression.func.id in {"len", "bool"}:
            queries.append(SequenceQueryBlock(
                result_value_id=result_id,
                sequence_value_id=sequence_id,
                operation="truth" if expression.func.id == "bool" else "length",
                source_call_node_id=int(node_id),
                extraction_identity=identity,
                result_alias_ids=result_alias_ids,
                producer_loop_node_id=producer_loop_node_id,
            ))
            continue
        if expression.func.id == "next":
            if len(arguments) != 2:
                unsupported.append(int(node_id))
                continue
            queries.append(SequenceQueryBlock(
                result_value_id=result_id,
                sequence_value_id=sequence_id,
                operation="first_or_default",
                default_value_id=int(arguments[1]),
                source_call_node_id=int(node_id),
                extraction_identity=identity,
                result_alias_ids=result_alias_ids,
                producer_loop_node_id=producer_loop_node_id,
            ))
            continue

        materializer_id = sequence_attributes.get("materializer_node_id")
        materializer = (
            {} if materializer_id not in graph_obj
            else graph_obj.nodes[int(materializer_id)]
        )
        element_id = next((
            int(parent)
            for parent, role in materializer.get("parents") or ()
            if str(role) == "elt" and int(parent) in graph_obj
        ), None)
        element_data = (
            {} if element_id is None else graph_obj.nodes[element_id]
        )
        element_literal = (element_data.get("attributes") or {}).get(
            "value", element_data.get("constant")
        )
        explicit_start = (
            None if len(arguments) == 1
            else (graph_obj.nodes[arguments[1]].get("attributes") or {}).get(
                "value", graph_obj.nodes[arguments[1]].get("constant")
            )
        )
        if element_literal != 1 or explicit_start not in {None, 0}:
            unsupported.append(int(node_id))
            continue
        queries.append(SequenceQueryBlock(
            result_value_id=result_id,
            sequence_value_id=sequence_id,
            operation="length",
            source_call_node_id=int(node_id),
            extraction_identity=identity,
            result_alias_ids=result_alias_ids,
            producer_loop_node_id=producer_loop_node_id,
        ))

    if not queries:
        return control, tuple(unsupported)

    def insert_after_producer(block, query):
        if (
            os.environ.get("TURING_DEBUG_BUILTIN_SELECTION")
            and isinstance(block, LoopBlock)
        ):
            print(
                "DEBUGQUERYLOOP "
                f"call={query.source_call_node_id} "
                f"candidate={block.source_loop_node_id}",
                file=sys.stderr,
            )
        if (
            isinstance(block, LoopBlock)
            and block.source_loop_node_id is not None
            and int(block.source_loop_node_id)
            == int(query.producer_loop_node_id)
        ):
            return SequenceBlock((block, query)), True
        if isinstance(block, SequenceBlock):
            children = []
            inserted = False
            for child in block.blocks:
                projected, child_inserted = insert_after_producer(child, query)
                children.append(projected)
                inserted |= child_inserted
            return SequenceBlock(tuple(children)), inserted
        if isinstance(block, ConditionalBlock):
            body, inserted = insert_after_producer(block.body, query)
            orelse = block.orelse
            if not inserted and orelse is not None:
                orelse, inserted = insert_after_producer(orelse, query)
            return replace(block, body=body, orelse=orelse), inserted
        if isinstance(block, LoopBlock):
            body, inserted = insert_after_producer(block.body, query)
            return replace(block, body=body), inserted
        if isinstance(block, WhileBlock):
            body, inserted = insert_after_producer(block.body, query)
            return replace(block, body=body), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_after_producer(block.callee, query)
            return replace(block, callee=callee), inserted
        return block, False

    region_nodes = {
        index: set(map(int, region.G.graph.get("deployment_nodes", ())))
        for index, region in enumerate(dispatch_subgraphs)
    }

    def insert_at_query_region(block, query):
        # The region marker retains the authored use position, including its
        # enclosing branch/loop. Never hoist truth to sequence construction.
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            match = re.fullmatch(r"__scheduled_region_(\d+)__", block.lines[0])
            if match and query.source_call_node_id in region_nodes.get(int(match[1]), set()):
                return SequenceBlock((query, block)), True
        if isinstance(block, SequenceBlock):
            children, inserted = [], False
            for child in block.blocks:
                child, found = insert_at_query_region(child, query)
                children.append(child)
                inserted |= found
            return replace(block, blocks=tuple(children)), inserted
        if isinstance(block, (ConditionalBlock, LoopBlock, WhileBlock)):
            body, inserted = insert_at_query_region(block.body, query)
            updates = {"body": body}
            if isinstance(block, ConditionalBlock) and block.orelse is not None:
                updates["orelse"], found = insert_at_query_region(block.orelse, query)
                inserted |= found
            if isinstance(block, WhileBlock):
                updates["condition"], found = insert_at_query_region(block.condition, query)
                inserted |= found
            return replace(block, **updates), inserted
        if isinstance(block, CallBlock):
            callee, inserted = insert_at_query_region(block.callee, query)
            return replace(block, callee=callee), inserted
        return block, False

    root = control.root
    unplaced = list(unsupported)
    for query in queries:
        if query.producer_loop_node_id is None:
            # A local list's truth (``if reasons:``) is an authored
            # statement: it goes at its source position in its own scope.
            # Anchoring it to whichever region happens to contain the
            # ``bool`` node put it after the conditionals that consume it
            # once regions were ordered by the hierarchy, and their merge
            # rebinding then made the query define the merged id.
            if int(query.source_call_node_id) in graph_obj:
                root, inserted = _insert_lexically(
                    root, graph, dispatch_subgraphs, query,
                    int(query.source_call_node_id),
                )
            else:
                root, inserted = insert_at_query_region(root, query)
            if not inserted:
                unplaced.append(int(query.source_call_node_id))
            continue
        root, inserted = insert_after_producer(root, query)
        if os.environ.get("TURING_DEBUG_BUILTIN_SELECTION"):
            print(
                "DEBUGQUERYPLACE "
                f"fn={graph_obj.graph.get('function_name')} "
                f"call={query.source_call_node_id} "
                f"producer={query.producer_loop_node_id} inserted={inserted}",
                file=sys.stderr,
            )
        if not inserted:
            unplaced.append(int(query.source_call_node_id))
    # When a numerical region contains only source calls replaced by resident
    # queries, its old marker would emit a second, now-dead helper call.  That
    # helper still names the starred wrapper as a scalar capture, recreating
    # the invented formal this transform exists to remove.  Retire only whole
    # regions whose complete source-node set is covered by installed queries;
    # mixed regions retain their downstream numerical work.
    query_call_ids = {
        int(query.source_call_node_id)
        for query in queries
        if query.source_call_node_id is not None
    }
    replaced_region_indices = {
        int(index)
        for index, nodes in region_nodes.items()
        if nodes and nodes.issubset(query_call_ids)
    }

    def remove_replaced_regions(block):
        if isinstance(block, StatementBlock) and len(block.lines) == 1:
            match = re.fullmatch(r"__scheduled_region_(\d+)__", block.lines[0])
            if match and int(match[1]) in replaced_region_indices:
                return SequenceBlock(())
        if isinstance(block, SequenceBlock):
            return replace(
                block,
                blocks=tuple(
                    projected
                    for child in block.blocks
                    for projected in (remove_replaced_regions(child),)
                    if not (
                        isinstance(projected, SequenceBlock)
                        and not projected.blocks
                    )
                ),
            )
        if isinstance(block, ConditionalBlock):
            return replace(
                block,
                body=remove_replaced_regions(block.body),
                orelse=(
                    None if block.orelse is None
                    else remove_replaced_regions(block.orelse)
                ),
            )
        if isinstance(block, LoopBlock):
            return replace(block, body=remove_replaced_regions(block.body))
        if isinstance(block, WhileBlock):
            return replace(
                block,
                condition=remove_replaced_regions(block.condition),
                body=remove_replaced_regions(block.body),
            )
        if isinstance(block, CallBlock):
            return replace(block, callee=remove_replaced_regions(block.callee))
        return block

    root = remove_replaced_regions(root)
    root = _schedule_sequence_query_dependencies(root)
    return replace(
        control,
        root=root,
        region_indices=tuple(
            int(index) for index in control.region_indices
            if int(index) not in replaced_region_indices
        ),
    ), tuple(unplaced)

"""Conditional control programs: authored conditional/loop control lowered to ControlPrograms, nesting hints, the subdivision-required refusal and the control partition keys."""

from __future__ import annotations

import ast
import os
import re
import sys
import networkx as nx
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .control_source import (
    CallBlock,
    ControlExpression,
    ControlOverlayScopeError,
    ControlProgram,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
    WhileBlock,
    overlay_scheduled_control,
)
from ..common.tensors.topological_reducer import indexed_store_site_span
from .source_node_facts import (
    _ast_source_location,
    _ast_source_signature,
)
from .source_control_retention import (
    _source_control_records,
    _source_control_expression,
    _retained_control_value_id,
    _optional_presence_control_expression,
    _record_field_state_keys,
)


class CompilationSubdivisionRequired(ValueError):
    """A correct control owner exists, but this unit is too coarse to lower.

    This remains a ``ValueError`` for callers which historically treated the
    refusal as an invalid lowering. The structured receipt lets the bounded
    project compiler distinguish a partition frontier from an opaque worker
    crash and creep at the named source/control boundary.
    """

    def __init__(self, message: str, *, boundaries=()):
        super().__init__(message)
        self.boundaries = tuple(dict(item) for item in boundaries)

    def to_failure_mapping(self) -> dict[str, Any]:
        return {
            "frontier_kind": "compilation-subdivision-required",
            "subdivision_boundaries": [
                dict(item) for item in self.boundaries
            ],
        }


def _loop_induction_name(loop_node_id: int) -> str:
    return f"iteration_{loop_node_id}"


def _find_nested_loop_node_ids_in_block(
    block: Any,
) -> "frozenset[int]":
    """Collect the node_ids of all LoopBlock/WhileBlock induction names in a
    block tree by parsing the ``iteration_{node_id}`` induction convention."""

    found: set[int] = set()

    def walk(b: Any) -> None:
        if isinstance(b, LoopBlock):
            name = str(b.induction)
            if name.startswith("iteration_"):
                try:
                    found.add(int(name[len("iteration_"):]))
                except ValueError:
                    pass
            walk(b.body)
        elif isinstance(b, WhileBlock):
            walk(b.condition)
            walk(b.body)
        elif isinstance(b, SequenceBlock):
            for child in b.blocks:
                walk(child)
        elif isinstance(b, StateMachineTick):
            for _value, body in b.cases:
                walk(body)
            if b.default is not None:
                walk(b.default)
        elif isinstance(b, ParallelDeployment):
            for lane in b.lanes:
                walk(lane)
        elif isinstance(b, CallBlock):
            walk(b.callee)

    walk(block)
    return frozenset(found)


def _loop_reduction_nesting_hints(
    reductions: "Sequence[LoopShaderReduction]",
    loop_plans: "Iterable[LoopPlan]",
    graph: Any | None = None,
) -> dict[int, frozenset[int]]:
    """Direct-child hints for ``overlay_scheduled_control``'s
    ``known_nesting``, keyed by index into ``reductions``.

    Region-set containment is only a proxy for "this loop is lexically
    nested inside that one" -- it fails when the outer loop's entire body
    is the inner loop (``while a: while b: ...`` with nothing of the
    outer's own between them), since then both compute the identical
    region set rather than the outer being a superset. The real signal is
    the same one ``analyze_shader_loop_reductions`` already used to
    compute each reduction's own ``region_indices`` in the first place:
    ``LoopDescriptor.body_nodes``, the set of graph nodes lexically inside
    a loop's body. A loop is a direct concern of this hint only if its own
    node id is a member of another loop's ``body_nodes`` -- real AST
    nesting, not inferred from schedule position or edge traversal.

    Fallback: when ``loop_plans`` does not provide ``body_nodes`` for a
    candidate outer loop (e.g. the plan was evaporated or belongs to a
    different shell), equal-region-set pairs are disambiguated by
    inspecting the ``control_program.root`` block tree.  The outer loop's
    un-projected ``LoopBlock`` body must contain the inner loop's
    ``LoopBlock`` as a direct or indirect child; this is always true for
    genuinely nested reductions because the planner embeds the inner loop's
    ``LoopBlock`` inside the outer one before any region projection occurs.
    """

    loop_plans = tuple(loop_plans or ())
    body_nodes_by_loop_id = {
        int(plan.loop.node_id): frozenset(map(int, plan.loop.body_nodes))
        for plan in loop_plans
    }
    loop_ids = [int(reduction.loop_node_id) for reduction in reductions]
    hints: dict[int, frozenset[int]] = {}
    for parent_index, parent_loop_id in enumerate(loop_ids):
        body = body_nodes_by_loop_id.get(parent_loop_id)
        if not body:
            continue
        children = frozenset(
            child_index
            for child_index, child_loop_id in enumerate(loop_ids)
            if child_index != parent_index and child_loop_id in body
        )
        if children:
            hints[parent_index] = children

    # Fallback for equal-region-set pairs missed by the body_nodes pass.
    # Two reductions whose projected region sets are equal but whose loop
    # plans did not provide a nesting signal are resolved structurally: the
    # outer reduction's un-projected control_program.root should embed the
    # inner reduction's LoopBlock (identifiable by its ``iteration_{id}``
    # induction name) somewhere in its tree.
    already_covered_as_child = frozenset(
        child for children in hints.values() for child in children
    )
    for parent_index, parent_reduction in enumerate(reductions):
        if parent_reduction.control_program is None:
            continue
        parent_region_set = frozenset(parent_reduction.region_indices)
        if not parent_region_set:
            continue
        parent_nested_ids = _find_nested_loop_node_ids_in_block(
            parent_reduction.control_program.root
        )
        if not parent_nested_ids:
            continue
        fallback_children = frozenset(
            child_index
            for child_index, child_reduction in enumerate(reductions)
            if child_index != parent_index
            and child_index not in already_covered_as_child
            and frozenset(child_reduction.region_indices) == parent_region_set
            and int(child_reduction.loop_node_id) in parent_nested_ids
        )
        if fallback_children:
            merged = hints.get(parent_index, frozenset()) | fallback_children
            hints[parent_index] = merged

    # Comprehension clauses are represented as sibling ``generators`` parents
    # of one aggregate materializer, not as one clause's body containing the
    # next clause's node. Their ordered parent list is nevertheless the exact
    # Python lexical nesting order: ``for outer ... for inner ...``. Preserve
    # that order explicitly when multiple clauses reduce to the same region.
    if graph is not None:
        reduction_index_by_loop_id = {
            int(reduction.loop_node_id): index
            for index, reduction in enumerate(reductions)
        }
        for _node_id, data in graph.G.nodes(data=True):
            generator_indices = tuple(
                reduction_index_by_loop_id[int(parent_id)]
                for parent_id, role in data.get("parents", ())
                if str(role) == "generators"
                and int(parent_id) in reduction_index_by_loop_id
            )
            for parent_index, child_index in zip(
                generator_indices, generator_indices[1:]
            ):
                parent_regions = frozenset(
                    reductions[parent_index].region_indices
                )
                child_regions = frozenset(
                    reductions[child_index].region_indices
                )
                if parent_regions and child_regions == parent_regions:
                    hints[parent_index] = (
                        hints.get(parent_index, frozenset()) | {child_index}
                    )

    return hints


def _branch_compartments(graph: Any) -> dict[int, frozenset[tuple[int, str]]]:
    """Map every node to the conditional branches that guard it.

    A branch body only runs when its test selects it.  The coordinator honours
    that by withholding controlled nodes from the ordinary topological sweep,
    but a shader region is executed as a unit: pulling any of its members
    forces every member to run.  Recording branch membership lets region
    planning refuse to mix guarded work with work that runs unconditionally,
    or to mix two branches of the same test with each other.
    """

    # ``ast.Load``/``ast.Store`` and operator/context nodes are locationless
    # singleton helpers.  They are shared by otherwise unrelated expressions,
    # so correlating them by object identity (or by their identical dump)
    # makes one small branch appear to guard every name load in the function.
    # Branch ownership belongs only to source-positioned syntax.  Real names,
    # calls, expressions, and statements all carry ``lineno``; helper tokens
    # do not.
    def source_positioned(expression: Any) -> bool:
        return (
            isinstance(expression, ast.AST)
            and getattr(expression, "lineno", None) is not None
        )

    expression_nodes = {
        id(data.get("expr_obj")): node_id
        for node_id, data in graph.G.nodes(data=True)
        if source_positioned(data.get("expr_obj"))
    }
    location_nodes = {}
    signatures = {}

    def signature(expression):
        key = id(expression)
        if key not in signatures:
            signatures[key] = _ast_source_signature(expression)
        return signatures[key]
    effect_span_nodes: dict[tuple[Any, ...], set[int]] = {}
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        for source_expression in (expression, data.get("authored_expr_obj")):
            if source_positioned(source_expression):
                location_nodes.setdefault(_ast_source_location(source_expression), []).append((source_expression, int(node_id)))
        if not source_positioned(expression) and str(data.get("type") or "").casefold() == "setattr":
            span = data.get("source_span") or {}
            if span.get("line") is not None:
                effect_span_nodes.setdefault(tuple(span.get(key) for key in
                    ("line", "column", "end_line", "end_column")), set()).add(int(node_id))
        if not source_positioned(expression) and str(data.get("type") or "") == "IndexedStore":
            # An indexed store with no authored node of its own: its site
            # is the ``indexed_store_site`` book cell its identity derives
            # from; the site's span only places it.
            site_span = indexed_store_site_span(graph, int(node_id))
            if site_span is not None:
                effect_span_nodes.setdefault(site_span, set()).add(int(node_id))
    memberships: dict[int, set[tuple[int, str]]] = {}
    structurally_specialized = frozenset(map(
        int,
        graph.G.graph.get(
            "structurally_specialized_conditional_node_ids", ()
        ),
    ))
    for control_id, record in _source_control_records(graph.G).items():
        if int(control_id) in structurally_specialized:
            # Structural folding has selected one arm and made its producer
            # unconditional. Keeping the old branch compartment would withhold
            # that producer from the topological sweep even though the matching
            # ControlProgram is deliberately omitted below. A later live
            # branch can then consume an unavailable value (the selected
            # ``self.dt_min`` payload in ``pi_update`` exposed this).
            continue
        expression = record.get("expression")
        if isinstance(expression, ast.If):
            branches = {
                "body": tuple(expression.body),
                "orelse": tuple(expression.orelse),
            }
        elif isinstance(expression, ast.IfExp):
            branches = {
                "body": (expression.body,),
                "orelse": (expression.orelse,),
            }
        elif isinstance(expression, ast.Try):
            branches = {
                "body": tuple(expression.body),
                "handlers": tuple(expression.handlers),
                "orelse": tuple(expression.orelse),
                "finalbody": tuple(expression.finalbody),
            }
        elif isinstance(expression, ast.BoolOp):
            # Only the first operand of a short-circuiting operator is
            # guaranteed to run; the rest are guarded by its outcome.
            branches = {
                f"operand:{index}": (value,)
                for index, value in enumerate(expression.values)
                if index
            }
        else:
            continue
        for role, statements in branches.items():
            for statement in statements:
                for member in ast.walk(statement):
                    if not source_positioned(member):
                        continue
                    guarded_nodes = (
                        (int(expression_nodes[id(member)]),)
                        if id(member) in expression_nodes
                        else tuple(sorted({candidate_id
                            for candidate, candidate_id in location_nodes.get(_ast_source_location(member), ())
                            if signature(candidate) == signature(member)}))
                    )
                    guarded_nodes = (*guarded_nodes, *effect_span_nodes.get(
                        (member.lineno, member.col_offset,
                         getattr(member, "end_lineno", None), getattr(member, "end_col_offset", None)), ()))
                    for guarded in guarded_nodes:
                        if guarded != control_id:
                            memberships.setdefault(guarded, set()).add(
                                (control_id, role)
                            )
    # Reducer-authored Phi values do not own an AST expression, but they run
    # at the continuation of their exact source conditional.  When that
    # conditional is itself nested in another branch, its merge is therefore
    # guarded by the enclosing branch and must be a candidate outgoing value
    # for the enclosing Phi.  Inherit only the source conditional's outer
    # memberships; never mark the merge as belonging to either of its own
    # mutually exclusive arms.
    for node_id, data in graph.G.nodes(data=True):
        source_conditional_id = int((data.get("attributes") or {}).get(
            "source_conditional_id", -1
        ))
        if source_conditional_id in memberships:
            memberships.setdefault(int(node_id), set()).update(
                memberships[source_conditional_id]
            )
    return {
        node_id: frozenset(roles)
        for node_id, roles in memberships.items()
    }


def _ordinary_conditional_control_programs(
    graph: Any,
    retained_regions: Iterable[int],
    dispatch_subgraphs: Iterable[Any],
) -> tuple[ControlProgram, ...]:
    """Preserve source conditional arms and their lexical SSA merges."""

    from .control_source import ConditionalBlock, LoopControlBlock

    retained_regions = tuple(map(int, retained_regions))
    retained = frozenset(retained_regions)
    subgraphs = tuple(dispatch_subgraphs)
    memberships = _branch_compartments(graph)
    node_by_value = {
        int(data.get("value_id", node_id)): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("value_id", node_id), int)
    }
    reducer_phi_record_fields = _record_field_state_keys(
        graph.G,
        node_by_value,
        (
            int(data.get("value_id", node_id))
            for node_id, data in graph.G.nodes(data=True)
            if str(data.get("type") or data.get("op") or "").casefold()
            == "phi"
            and (data.get("attributes") or {}).get("record_field_state")
            is not None
        ),
    )
    structurally_specialized = frozenset(map(
        int,
        graph.G.graph.get(
            "structurally_specialized_conditional_node_ids", ()
        ),
    ))
    programs = []
    program_expressions: list[tuple[int, ast.AST]] = []
    for control_id, record in _source_control_records(graph.G).items():
        if int(control_id) in structurally_specialized:
            # Structural specialization has already selected one authored arm,
            # removed the other, and redirected every merge identity to the
            # selected producer.  Reconstructing this retained source record
            # would put that producer back behind a runtime branch while its
            # post-merge consumers remain straight-line.
            continue
        expression = record.get("expression")
        if not isinstance(expression, (ast.If, ast.IfExp)):
            continue
        is_expression = isinstance(expression, ast.IfExp)
        body_statements = (expression.body,) if is_expression else expression.body
        else_statements = (expression.orelse,) if is_expression else expression.orelse
        result_aliases = ()
        if is_expression:
            data = graph.G.nodes.get(int(control_id))
            if data is None:
                continue
            parents = dict((str(role), int(parent))
                           for parent, role in data.get("parents", ()))
            if not all(parents.get(role) in graph.G for role in ("body", "orelse")):
                continue
            result_aliases = (data.get("attributes") or {}).get("conditional_member_bindings") or ((
                int(graph.G.nodes[parents["body"]].get("value_id", parents["body"])),
                int(graph.G.nodes[parents["orelse"]].get("value_id", parents["orelse"])),
                int(data.get("value_id", control_id)),
            ),)
        predicate_id = _retained_control_value_id(
            graph.G,
            record.get("predicate_id"),
            expression.test,
        )
        # A retained source comparison may still exist even though its Python
        # ``None`` operand has no native representation.  Prefer the exact
        # payload/presence relationship whenever it is available; lowering
        # the surviving comparison itself would compare the filler payload
        # and manufacture a constant truth value.
        predicate_expression = _optional_presence_control_expression(
            graph.G, expression.test,
        )
        if predicate_expression is not None:
            predicate_value_id = int(predicate_expression.value_id)
            predicate_id = predicate_value_id
        elif predicate_id is None:
            continue
        else:
            predicate_value_id = int(
                graph.G.nodes[predicate_id].get("value_id", predicate_id)
            )
            predicate_expression = ControlExpression(
                "value", value_id=int(predicate_value_id)
            )
        body_regions = tuple(
            region_index
            for region_index, subgraph in enumerate(subgraphs)
            if region_index in retained
            and any(
                (int(control_id), "body") in memberships.get(int(node_id), ())
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
        )
        else_regions = tuple(
            region_index
            for region_index, subgraph in enumerate(subgraphs)
            if region_index in retained
            and any(
                (int(control_id), "orelse") in memberships.get(int(node_id), ())
                for node_id in subgraph.G.graph.get("deployment_nodes", ())
            )
        )
        # The comparison node is only the last link in the predicate.  Every
        # retained numerical region containing one of its transitive graph
        # producers must execute before the branch.  Selecting only the region
        # that contains ``predicate_id`` leaves earlier producer regions in
        # the flat tail, and control lowering then mistakes their canonical
        # result IDs for additional public arguments.
        predicate_scope = memberships.get(int(predicate_id), frozenset())
        predicate_dependencies = {
            int(candidate)
            for candidate in {
                int(predicate_id),
                *map(int, nx.ancestors(graph.G, int(predicate_id))),
            }
            # A nested predicate can read a value produced outside its source
            # arm, but that producer remains owned by the enclosing scope.
            # Only dependencies physically evaluated in the predicate's own
            # branch compartment belong in this conditional prelude.  Exact
            # source membership is the lexical proof; graph ancestry alone is
            # deliberately insufficient.
            if memberships.get(int(candidate), frozenset()) == predicate_scope
        }
        predicate_regions = tuple(
            region_index
            for region_index in retained_regions
            if int(region_index) in retained
            for subgraph in (subgraphs[int(region_index)],)
            if predicate_dependencies.intersection(map(
                int, subgraph.G.graph.get("deployment_nodes", ())
            ))
        )
        predicate_terminal_regions = tuple(
            region_index
            for region_index in retained_regions
            if int(region_index) in retained
            for subgraph in (subgraphs[int(region_index)],)
            if int(predicate_id) in set(map(
                int, subgraph.G.graph.get("deployment_nodes", ())
            ))
        )
        # A ``return`` ending an arm is a control effect of that arm: it
        # leaves the function carrying the arm's values. The overlay removes
        # the loop schedule's copy when the exact anchor and slots agree.
        return_slot_values = dict(graph.G.graph.get("return_slot_values") or {})
        enclosing_loop_spans = tuple(
            (int(loop_expression.lineno), int(getattr(
                loop_expression, "end_lineno", loop_expression.lineno
            )))
            for _loop_id, loop_data in graph.G.nodes(data=True)
            for loop_expression in (loop_data.get("expr_obj"),)
            if isinstance(loop_expression, (ast.For, ast.While))
        )
        inside_loop = any(
            start <= int(expression.lineno) <= end
            for start, end in enclosing_loop_spans
        )
        # A ``break``/``continue`` ending an arm is that arm's exit edge: it
        # leaves the loop carrying the arm's OWN bindings, which dominate
        # the edge only inside the arm.  Place it there (the loop schedule
        # skips arm-owned sites, `LoopDescriptor.control_sites`), with the
        # site's live bindings from the owning loop's ``loop_break_sites``.
        enclosing_loops = tuple(
            (
                int(loop_expression.lineno),
                int(getattr(
                    loop_expression, "end_lineno", loop_expression.lineno
                )),
                dict((loop_data.get("attributes") or {}).get(
                    "loop_break_sites"
                ) or {}),
            )
            for _loop_id, loop_data in graph.G.nodes(data=True)
            for loop_expression in (loop_data.get("expr_obj"),)
            if isinstance(loop_expression, (ast.For, ast.While))
        )
        retained_region_nodes = {
            int(member)
            for region_index in sorted(retained)
            for member in subgraphs[int(region_index)].G.graph.get(
                "deployment_nodes", ()
            )
        }
        control_node_by_statement = {
            id(node_data.get("expr_obj")): int(node_id)
            for node_id, node_data in graph.G.nodes(data=True)
            if isinstance(
                node_data.get("expr_obj"), (ast.Break, ast.Continue)
            )
        }

        def arm_loop_control(statements, arm_name):
            if not inside_loop or not statements:
                return None
            terminal = statements[-1]
            if not isinstance(terminal, (ast.Break, ast.Continue)):
                return None
            action = "break" if isinstance(terminal, ast.Break) else "continue"
            line = int(getattr(terminal, "lineno", -1))
            span = (
                line,
                int(getattr(terminal, "col_offset", -1)),
                int(getattr(terminal, "end_lineno", -1)),
                int(getattr(terminal, "end_col_offset", -1)),
            )
            # Innermost loop whose source range holds the site owns it.
            owner = min(
                (
                    (end - start, break_sites)
                    for start, end, break_sites in enclosing_loops
                    if start <= line <= end
                ),
                key=lambda item: item[0],
                default=None,
            )
            declared_identity_ids = {
                int(value_id)
                for history in (graph.G.graph.get("identity_table") or {}).values()
                for value_id in history
            }
            site_values = {
                int(initial): int(value)
                for initial, value in (
                    (owner[1].get(span) or {}) if owner else {}
                ).items()
                if int(value) in graph.G
                and (int(initial) in graph.G or int(initial) in declared_identity_ids)
            }
            # Same rule as the loop schedule (``arm_owned_site`` in
            # loop_composer): the arm owns the site only when a value it
            # carries is produced by a retained region INSIDE this arm.
            owned = any(
                int(value) in retained_region_nodes
                and int(value) in node_by_value
                and (int(control_id), arm_name) in memberships.get(
                    node_by_value[int(value)], ()
                )
                for value in site_values.values()
            )
            if os.environ.get("TURING_DEBUG_BREAK_EDGE"):
                print(
                    "DEBUG-ARM-OWNED builder "
                    f"fn={graph.G.graph.get('function_name')} "
                    f"control={int(control_id)} arm={arm_name} "
                    f"site={control_node_by_statement.get(id(terminal))} "
                    f"site_values={site_values} owned={owned}",
                    file=sys.stderr, flush=True,
                )
            if not owned:
                return None
            return LoopControlBlock(
                action, None, True, None,
                source_action=action,
                site_node_id=control_node_by_statement.get(id(terminal)),
                site_values=tuple(sorted(
                    (int(initial), int(value))
                    for initial, value in site_values.items()
                )),
            )

        def arm_return_control(statements):
            if not statements:
                return None
            terminal = statements[-1]
            if not isinstance(terminal, ast.Return) or terminal.value is None:
                return None
            returned = terminal.value
            slots = return_slot_values.get((
                int(getattr(returned, "lineno", -1)),
                int(getattr(returned, "col_offset", -1)),
                int(getattr(returned, "end_lineno", -1)),
                int(getattr(returned, "end_col_offset", -1)),
            ))
            if slots is None:
                return None
            # ``site_node_id`` is a position hint (the returned expression's
            # node, as LoopComposer.describe's hint); the SITE is
            # ``return_site_cell`` below. Tuple containers can be reduced
            # away, in which case the last surviving exact slot is the hint.
            signature = lambda expr: (
                *(getattr(expr, key, None) for key in (
                    "lineno", "col_offset", "end_lineno", "end_col_offset")),
                ast.dump(expr, include_attributes=False),
            )
            sites = [int(node_id) for node_id, node_data in graph.G.nodes(data=True)
                     if isinstance(node_data.get("expr_obj"), ast.AST)
                     and signature(node_data["expr_obj"]) == signature(returned)]
            site_id = min(sites) if sites else next((
                int(value_id) for value_id in reversed(slots)
                if value_id is not None and int(value_id) in graph.G
            ), None)
            if inside_loop and site_id is None:
                return None
            # The site's identity is the return construct's cell, the same
            # cell ``LoopComposer.describe`` keys the loop's copy by: the
            # overlay matches the two by site, never by the value returned.
            from .ssa_record_return_state import return_site_cell_for
            return LoopControlBlock(
                "return", None, True, None,
                source_action="return",
                site_node_id=site_id,
                return_value_ids=tuple(slots),
                return_site_cell=return_site_cell_for(graph, returned),
            )

        body_return_control = (
            arm_return_control(body_statements)
            or arm_loop_control(body_statements, "body")
        )
        else_return_control = (
            arm_return_control(else_statements)
            or arm_loop_control(else_statements, "orelse")
        )
        def arm_callsites(arm):
            return tuple(sorted(
                int(node_id)
                for node_id, roles in memberships.items()
                if (int(control_id), arm) in roles
                and int(node_id) in graph.G
                for data in (graph.G.nodes[int(node_id)],)
                if str(data.get("op") or data.get("type") or "").casefold() == "call"
                or (
                    isinstance(data.get("expr_obj"), ast.Call)
                    and any((data.get("attributes") or {}).get(key) is not None
                            for key in ("method_ref", "callee_ref"))
                )
            ))

        # A resolved method keeps its method name as the graph operation.
        # Its lexical control ownership does not depend on numerical regions
        # or on the generic operation spelling "call".
        body_callsites = arm_callsites("body")
        else_callsites = arm_callsites("orelse")
        if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
            terminal_spans = [
                (
                    int(getattr(arm[-1].value, "lineno", -1)),
                    int(getattr(arm[-1].value, "col_offset", -1)),
                    int(getattr(arm[-1].value, "end_lineno", -1)),
                    int(getattr(arm[-1].value, "end_col_offset", -1)),
                )
                for arm in (body_statements, else_statements)
                if arm and isinstance(arm[-1], ast.Return)
                and arm[-1].value is not None
            ]
            print(
                "DEBUG-RETURN-ARM "
                f"fn={graph.G.graph.get('function_name')} "
                f"control={int(control_id)} inside_loop={inside_loop} "
                f"terminal_return_spans={terminal_spans!r} "
                f"table_keys={sorted(return_slot_values)!r} "
                f"body={body_return_control is not None} "
                f"else={else_return_control is not None}",
                file=sys.stderr,
            )
        has_structural_branch_effect = (
            body_return_control is not None
            or else_return_control is not None
            or any(
                isinstance(member, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                or (
                    isinstance(member, ast.Call)
                    and isinstance(member.func, ast.Attribute)
                    and member.func.attr in {"append", "add", "extend"}
                )
                for statement in (*body_statements, *else_statements)
                for member in ast.walk(statement)
            )
        )
        # The reducer's field-state merge for THIS conditional is a
        # retention reason on its own.  ``if rejected: m.hard_failure =
        # True`` (tools/compiler_probes/probe_scalar_write_only_arm.py): the
        # arm holds nothing but a scalar record-field write, so it has no
        # numerical region, no callsite, no terminal control and no result
        # alias, and ``rejected`` is a bare formal with no predicate region.
        # Every term below was false and the conditional was dropped; the
        # ScalarFieldWriteBlock then landed in the ENCLOSING arm and the
        # MERGED cell of this merge never received a version
        # (``carried-field-arm-missing``).  The Phi the reducer posted for
        # this conditional -- ``source_conditional_id`` equal to it, with
        # ``field_state_arms`` -- says the conditional owns a field-state
        # merge; consuming that record here (before the ``continue``) builds
        # the ConditionalBlock whose ``carried_field_cells`` the control
        # builder already reads, and the mutation pass then finds the arm.
        field_state_merge = any(
            str(phi_data.get("type") or phi_data.get("op") or "")
            .casefold() == "phi"
            and int((phi_data.get("attributes") or {}).get(
                "source_conditional_id", -1
            )) == int(control_id)
            and isinstance(
                (phi_data.get("attributes") or {}).get("field_state_arms"),
                (tuple, list),
            )
            for _phi_node_id, phi_data in graph.G.nodes(data=True)
        )
        if (
            not body_regions
            and not else_regions
            and body_return_control is None
            and else_return_control is None
            and not body_callsites
            and not else_callsites
            and not result_aliases
            and not (predicate_regions and has_structural_branch_effect)
            and not field_state_merge
        ):
            if os.environ.get("TURING_DEBUG_CONTROL_OVERLAY"):
                print(
                    "DEBUGIFMISS "
                    f"fn={graph.G.graph.get('function_name')} "
                    f"control={int(control_id)} "
                    f"source={ast.unparse(expression)!r} "
                    f"predicate={predicate_id} "
                    f"memberships={sum(1 for roles in memberships.values() if any(owner == int(control_id) for owner, _arm in roles))}",
                    file=sys.stderr,
                )
            continue

        # The reducer already owns the path-specific SSA values for an
        # authored assignment merge.  Consume those Phi parents directly
        # whenever it published the pre-branch identity.  Reconstructing the
        # same relation from a flat spelling history loses nesting and, for
        # attribute assignments, used to confuse SetAttr event ids with the
        # values they store.
        carried = []
        # Aligned with ``carried``: the reducer's field-state cells for a
        # field-carried merge, read from the Phi node's own attributes
        # (``field_state_arms`` = (true-arm cell, false-arm cell, test cell),
        # ``field_state_cell`` = the MERGED cell) so the lowering names the
        # cells the merge derived from instead of copying graph ids.
        carried_field_cells = []
        direct_phi_bindings: set[str] = set()
        direct_phi_record_fields: set[tuple[int, str]] = set()
        for phi_node_id, phi_data in graph.G.nodes(data=True):
            phi_attributes = phi_data.get("attributes") or {}
            if (
                str(phi_data.get("type") or phi_data.get("op") or "")
                .casefold() != "phi"
                or int(phi_attributes.get("source_conditional_id", -1))
                != int(control_id)
            ):
                continue
            phi_parents = {
                str(role): int(graph.G.nodes[parent].get(
                    "value_id", parent
                ))
                for parent, role in phi_data.get("parents") or ()
                if parent in graph.G
            }
            if not {"body", "orelse"}.issubset(phi_parents):
                continue
            merged_id = int(phi_data.get("value_id", phi_node_id))
            binding_name = phi_attributes.get("binding_name")
            if (
                binding_name is not None
                and str(binding_name).startswith("__turing_single_exit_result")
            ):
                # Guard-return normalization creates a fresh result only in
                # the two arms.  Its reducer Phi may name the first arm as an
                # ``initial_value_id`` compatibility fallback, but there is
                # no incumbent value before the branch.  Lower it through
                # the exact conditional-result path so each arm resolves its
                # own source identity instead of the missing arm falling
                # back to the first arm's constant.
                result_aliases = (*result_aliases, (
                    int(phi_parents["body"]),
                    int(phi_parents["orelse"]),
                    merged_id,
                ))
                direct_phi_bindings.add(str(binding_name))
                continue
            if phi_attributes.get("initial_value_id") is None:
                continue
            carried.append((
                int(phi_parents["body"]),
                int(phi_parents["orelse"]),
                int(phi_attributes["initial_value_id"]),
                merged_id,
            ))
            field_state_arms = phi_attributes.get("field_state_arms")
            carried_field_cells.append(
                (
                    field_state_arms[0], field_state_arms[1],
                    phi_attributes.get("field_state_cell"),
                )
                if isinstance(field_state_arms, (tuple, list))
                and len(field_state_arms) >= 2
                else None
            )
            if binding_name is not None:
                direct_phi_bindings.add(str(binding_name))
            direct_phi_record_fields.update(
                _record_field_state_keys(
                    graph.G, node_by_value, (merged_id,),
                )
            )
        # Merges are the reducer's Phis (above), which it creates for every
        # runtime conditional whose arms bind a name differently and never
        # for an arm that leaves by return/raise/continue/break.  Merges were
        # also synthesized here from a name's identity history ("the first
        # version after the arms"); for a continue arm that version is the
        # loop's fall-through update, which was then bound as the if-merge
        # and the loop lost the update.
        body = SequenceBlock((
            *(
                StatementBlock((f"__scheduled_region_{index}__",))
                for index in body_regions
            ),
            *((body_return_control,) if body_return_control is not None else ()),
        ))
        orelse = (
            SequenceBlock((
                *(
                    StatementBlock((f"__scheduled_region_{index}__",))
                    for index in else_regions
                ),
                *(
                    (else_return_control,)
                    if else_return_control is not None else ()
                ),
            ))
            if else_regions or else_return_control is not None or else_callsites else None
        )
        arm_regions = tuple(dict.fromkeys((*body_regions, *else_regions)))
        split_regions = set(predicate_regions).intersection(arm_regions)
        if split_regions:
            raise ValueError(
                "a dispatch region crosses a source conditional boundary: "
                f"conditional={int(control_id)}, "
                f"regions={sorted(split_regions)!r}. The region must be "
                "divided before it can be scheduled atomically."
            )
        # Predicate-producing regions are dataflow predecessors, not lexical
        # property of this branch. They can be shared by sibling conditions
        # or owned by an enclosing arm. Leave them in the flat dependency
        # schedule whenever this conditional has real arm regions.
        predicate_owned = not arm_regions
        # A structural-only arm still needs the region that materializes its
        # predicate under conditional control, but the predicate's transitive
        # numerical ancestors are ordinary dataflow predecessors.  Owning all
        # of them here pulls an enclosing loop (and often the function prefix)
        # into a post-loop ``if`` merely because its test reads a carried
        # result.  Leave those predecessor regions in the flat dependency
        # schedule and claim only the region containing the predicate value.
        owned_predicate_regions = (
            predicate_terminal_regions if predicate_owned else ()
        )
        root = SequenceBlock((
            *(
                StatementBlock((f"__scheduled_region_{index}__",))
                for index in owned_predicate_regions
            ),
            ConditionalBlock(
                int(predicate_value_id), body, orelse,
                predicate_expression=predicate_expression,
                carried_aliases=tuple(carried),
                carried_field_cells=tuple(carried_field_cells),
                source_node_id=int(control_id),
                # Call nodes lexically inside each arm: the callsite
                # scheduler anchors a planned call on numerical regions,
                # which an arm consisting only of a call does not have.
                body_callsite_ids=body_callsites,
                orelse_callsite_ids=else_callsites,
                result_aliases=result_aliases,
                predicate_region_indices=tuple(
                    int(region) for region in predicate_terminal_regions
                ),
            ),
        ))
        anchor_region = None
        if not owned_predicate_regions and not arm_regions:
            control_end = (
                int(getattr(expression, "end_lineno", expression.lineno)),
                int(getattr(expression, "end_col_offset", expression.col_offset)),
            )
            for region_index in retained_regions:
                positions = []
                for node_id in subgraphs[region_index].G.graph.get("deployment_nodes", ()):
                    span = graph.G.nodes[int(node_id)].get("source_span") or {}
                    if span.get("line") is not None:
                        positions.append((int(span["line"]), int(span.get("column", 0))))
                if positions and min(positions) > control_end:
                    anchor_region = region_index
                    break
        programs.append(ControlProgram(
            root=root,
            region_indices=tuple(dict.fromkeys((
                *owned_predicate_regions,
                *arm_regions,
            ))),
            # A call-only arm has no numerical marker to replace. Its source
            # position must still precede later mutation/read regions. Publish
            # the existing overlay anchor at planning time, rather than relying
            # on a later shell reconstruction to happen to rebuild this branch.
            anchor_region=anchor_region,
        ))
        program_expressions.append((int(control_id), expression))
    nested = _nest_region_less_conditionals(
        graph, tuple(programs), tuple(program_expressions), subgraphs,
    )
    # Step 9 (plan 100, 2.3): one ``control_program`` row per authored
    # conditional, its blocks on ``control_block`` with their placements;
    # the label is the conditional's own node cell (derived by the helper).
    from .concordance_declarations import CONTROL_PROGRAM_BUILD
    from .control_source import post_control_program

    for program in nested:
        post_control_program(graph, program, stage=CONTROL_PROGRAM_BUILD)
    return nested


def _nest_region_less_conditionals(
    graph: Any,
    programs: tuple[ControlProgram, ...],
    program_expressions: tuple[tuple[int, ast.AST], ...],
    subgraphs: tuple[Any, ...],
) -> tuple[ControlProgram, ...]:
    """Place a region-less conditional inside the arm that lexically owns it.

    The overlay (``overlay_scheduled_control``) embeds a nested control by
    replacing the region markers it owns inside its parent's arm.  A
    conditional with no region of its own has no marker; the overlay then
    places it before its anchor region "within this root, or appends it",
    and the root of a conditional program is ``(prefix..., ConditionalBlock)``
    -- so a region-less guard nested in an arm was appended AFTER its parent
    conditional, as a sibling.  For ``if bool(m.hard_failure): ... else: if
    rejected: m.hard_failure = True`` (the single-exit form of
    probe_scalar_write_only_arm.py) the reducer's outer merge names the
    inner MERGED field-state cell as its orelse arm; a sibling inner
    conditional is lowered after the outer merge, which then finds no
    version for that arm (``ARM_VERSION_MISSING``) and the inner conditional
    later disagrees with it.  The planner knows both source constructs, so
    it nests here: the innermost containing retained conditional's arm, at
    the lexical position among that arm's region markers (before the first
    region whose earliest source line follows the child; before a trailing
    terminal control; else at the end).  Children with regions keep the
    overlay's marker embedding, which this does not touch.
    """

    from .control_source import ConditionalBlock, LoopControlBlock

    if len(programs) < 2:
        return programs

    def span(expression: ast.AST) -> tuple[int, int]:
        start = int(getattr(expression, "lineno", -1))
        return start, int(getattr(expression, "end_lineno", start))

    def arm_statements(expression: ast.AST, arm: str):
        statements = getattr(expression, arm)
        return statements if isinstance(statements, list) else [statements]

    def owning_arm(parent: ast.AST, child: ast.AST):
        child_signature = _ast_source_signature(child)
        for arm in ("body", "orelse"):
            if any(
                _ast_source_signature(member) == child_signature
                for statement in arm_statements(parent, arm)
                for member in ast.walk(statement)
            ):
                return arm
        return None

    region_lines: dict[int, int | None] = {}

    def earliest_region_line(region_index: int) -> int | None:
        if region_index in region_lines:
            return region_lines[region_index]
        lines = []
        if region_index < len(subgraphs):
            for node_id in subgraphs[region_index].G.graph.get(
                "deployment_nodes", ()
            ):
                node = graph.G.nodes.get(int(node_id)) or {}
                line = (node.get("source_span") or {}).get("line")
                if line is None:
                    line = getattr(node.get("expr_obj"), "lineno", None)
                if line is not None and int(line) >= 0:
                    lines.append(int(line))
        region_lines[region_index] = min(lines) if lines else None
        return region_lines[region_index]

    def insert_in_arm(arm: SequenceBlock | None, child_blocks, child_line):
        blocks = list(arm.blocks) if isinstance(arm, SequenceBlock) else (
            [] if arm is None else [arm]
        )
        position = len(blocks)
        for index, block in enumerate(blocks):
            if (
                isinstance(block, StatementBlock)
                and len(block.lines) == 1
                and block.lines[0].startswith("__scheduled_region_")
            ):
                region_index = int(block.lines[0][len("__scheduled_region_"):-2])
                line = earliest_region_line(region_index)
                if line is not None and line > child_line:
                    position = index
                    break
        if position == len(blocks) and blocks and isinstance(
            blocks[-1], LoopControlBlock,
        ):
            position = len(blocks) - 1
        return SequenceBlock((
            *blocks[:position], *child_blocks, *blocks[position:],
        ))

    remaining = {index: program for index, program in enumerate(programs)}
    expressions = dict(enumerate(expression for _id, expression in program_expressions))
    # Innermost children first, so a grandchild joins its parent before that
    # parent joins its own.
    region_less = sorted(
        (
            index for index, program in remaining.items()
            if not program.region_indices
        ),
        key=lambda index: (
            span(expressions[index])[1] - span(expressions[index])[0],
            span(expressions[index])[0],
        ),
    )
    for child_index in region_less:
        child = remaining.get(child_index)
        if child is None:
            continue
        child_expression = expressions[child_index]
        candidates = []
        for parent_index, parent in remaining.items():
            if parent_index == child_index:
                continue
            parent_expression = expressions[parent_index]
            arm = owning_arm(parent_expression, child_expression)
            if arm is not None:
                start, end = span(parent_expression)
                candidates.append((end - start, parent_index, arm))
        if not candidates:
            continue
        _extent, parent_index, arm = min(candidates)
        parent = remaining[parent_index]
        conditional_position = next((
            index for index, block in enumerate(parent.root.blocks)
            if isinstance(block, ConditionalBlock)
        ), None)
        if conditional_position is None:
            continue
        conditional = parent.root.blocks[conditional_position]
        child_line = span(child_expression)[0]
        if arm == "body":
            conditional = replace(conditional, body=insert_in_arm(
                conditional.body, child.root.blocks, child_line,
            ))
        else:
            conditional = replace(conditional, orelse=insert_in_arm(
                conditional.orelse, child.root.blocks, child_line,
            ))
        remaining[parent_index] = replace(parent, root=SequenceBlock((
            *parent.root.blocks[:conditional_position],
            conditional,
            *parent.root.blocks[conditional_position + 1:],
        )))
        del remaining[child_index]
    return tuple(program for _index, program in sorted(remaining.items()))


def _ordinary_conditional_nesting(
    graph: Any,
    programs: Iterable[ControlProgram],
    *,
    offset: int = 0,
    root_index: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return exact lexical parentage for source-derived conditional IR."""

    from .control_source import ConditionalBlock

    controls = tuple(programs)
    blocks = tuple(
        next((
            block for block in program.root.blocks
            if isinstance(block, ConditionalBlock)
        ), None)
        if isinstance(program.root, SequenceBlock)
        else (
            program.root
            if isinstance(program.root, ConditionalBlock) else None
        )
        for program in controls
    )
    expressions = {
        index: graph.G.nodes[int(block.source_node_id)].get("expr_obj")
        for index, block in enumerate(blocks)
        if block is not None and block.source_node_id is not None
    }
    parent_by_child: dict[int, int] = {}
    for child_index, child_expression in expressions.items():
        candidates = []
        for parent_index, parent_expression in expressions.items():
            if parent_index == child_index:
                continue
            descendants = {
                id(member)
                for statement in (
                    (parent_expression.body, parent_expression.orelse)
                    if isinstance(parent_expression, ast.IfExp)
                    else (*parent_expression.body, *parent_expression.orelse)
                )
                for member in ast.walk(statement)
            }
            if id(child_expression) not in descendants:
                continue
            span = int(getattr(
                parent_expression, "end_lineno", parent_expression.lineno
            )) - int(parent_expression.lineno)
            candidates.append((span, parent_index))
        if candidates:
            parent_by_child[child_index] = min(candidates)[1]
    children: dict[int, list[int]] = {}
    for child_index in range(len(blocks)):
        parent = parent_by_child.get(child_index)
        if parent is not None:
            children.setdefault(offset + parent, []).append(
                offset + child_index
            )
        elif root_index is not None:
            children.setdefault(int(root_index), []).append(
                offset + child_index
            )
    return {parent: tuple(items) for parent, items in children.items()}


def _source_control_nesting_hints(
    reductions: "Sequence[LoopShaderReduction]",
    conditional_programs: "Sequence[ControlProgram]",
    loop_plans: "Iterable[LoopPlan]",
    graph: Any,
) -> dict[int, frozenset[int]]:
    """Correlate loop and conditional controls by authored AST identity.

    Overlay receives a heterogeneous control catalogue.  Region containment
    proves most nesting, but equal-region controls and comprehension clauses
    require the authored relation.  Compute one index space for both kinds so
    a top-level ``if`` can own a loop in its body and a loop can own an inner
    ``if`` without either construct being flattened.
    """

    reductions = tuple(reductions)
    conditionals = tuple(conditional_programs)
    offset = len(reductions)
    expressions: dict[int, ast.AST] = {}
    for index, reduction in enumerate(reductions):
        if int(reduction.loop_node_id) not in graph.G:
            continue
        expression = graph.G.nodes[int(reduction.loop_node_id)].get("expr_obj")
        if isinstance(expression, ast.AST):
            expressions[index] = expression
    from .control_source import ConditionalBlock
    for local_index, program in enumerate(conditionals):
        block = next((
            item for item in (
                program.root.blocks
                if isinstance(program.root, SequenceBlock)
                else (program.root,)
            )
            if isinstance(item, ConditionalBlock)
        ), None)
        if (
            block is None
            or block.source_node_id is None
        ):
            continue
        expression = _source_control_expression(
            graph.G, int(block.source_node_id)
        )
        if isinstance(expression, ast.AST):
            expressions[offset + local_index] = expression

    children: dict[int, set[int]] = {
        int(parent): set(map(int, nested))
        for parent, nested in _loop_reduction_nesting_hints(
            reductions, loop_plans, graph,
        ).items()
    }
    # Select the closest authored container for every control.  AST object
    # identity is the compiler's exact ingestion correlation; node count and
    # source span merely rank multiple real ancestors to find the direct one.
    for child_index, child_expression in expressions.items():
        candidates = []
        for parent_index, parent_expression in expressions.items():
            if parent_index == child_index:
                continue
            descendants = tuple(ast.walk(parent_expression))
            child_signature = _ast_source_signature(child_expression)
            if not any(
                id(member) == id(child_expression)
                or _ast_source_signature(member) == child_signature
                for member in descendants
            ):
                continue
            span = int(getattr(
                parent_expression, "end_lineno",
                getattr(parent_expression, "lineno", 0),
            ) or 0) - int(getattr(parent_expression, "lineno", 0) or 0)
            candidates.append((len(descendants), span, parent_index))
        if candidates:
            _size, _span, parent_index = min(candidates)
            children.setdefault(int(parent_index), set()).add(int(child_index))
    return {
        parent: frozenset(sorted(nested))
        for parent, nested in children.items() if nested
    }


def _overlay_control_or_require_subdivision(
    graph: Any,
    runtime_regions: Iterable[int],
    reductions: Sequence[Any],
    loop_controls: Sequence[ControlProgram],
    conditional_controls: Sequence[ControlProgram],
    nesting: Mapping[int, Iterable[int]],
    *, region_dependencies: Iterable[tuple[int, int]] = (),
) -> ControlProgram:
    """Turn a named cross-scope overlay refusal into loop-owned frontier data."""

    from .control_source import (
        CallBlock, ConditionalBlock, LoopBlock, LoopControlBlock, SequenceBlock, WhileBlock,
    )

    def children(block):
        if isinstance(block, SequenceBlock):
            return block.blocks
        if isinstance(block, ConditionalBlock):
            return (block.body,) + (() if block.orelse is None else (block.orelse,))
        if isinstance(block, (LoopBlock, WhileBlock)):
            return (block.body, *block.terminal_controls)
        if isinstance(block, CallBlock):
            return (block.callee,)
        return ()

    # A return is matched by its SITE: the return construct's cell when the
    # block carries one (several sites may return one value), else the
    # legacy position hint.
    def return_site_key(block):
        if block.return_site_cell is not None:
            return ("site", block.return_site_cell)
        if block.site_node_id is not None:
            return ("node", int(block.site_node_id))
        return None

    owned_returns = {}
    def collect(block):
        if (isinstance(block, LoopControlBlock) and block.action == "return"
                and return_site_key(block) is not None and block.predicate_value_id is None
                and block.predicate_expression is None):
            owned_returns.setdefault(return_site_key(block), set()).add(tuple(block.return_value_ids))
        for child in children(block):
            collect(child)

    for conditional in conditional_controls:
        collect(conditional.root)

    def remove_duplicate_return(block):
        if (isinstance(block, LoopControlBlock) and block.action == "return"
                and return_site_key(block) is not None
                and owned_returns.get(return_site_key(block)) == {tuple(block.return_value_ids)}):
            return SequenceBlock(())
        if isinstance(block, SequenceBlock):
            return replace(block, blocks=tuple(remove_duplicate_return(child) for child in block.blocks))
        if isinstance(block, ConditionalBlock):
            return replace(block, body=remove_duplicate_return(block.body),
                           orelse=None if block.orelse is None else remove_duplicate_return(block.orelse))
        if isinstance(block, (LoopBlock, WhileBlock)):
            return replace(block, body=remove_duplicate_return(block.body),
                           terminal_controls=tuple(remove_duplicate_return(child) for child in block.terminal_controls))
        if isinstance(block, CallBlock):
            return replace(block, callee=remove_duplicate_return(block.callee))
        return block

    # A terminal return belongs inside its authored conditional arm. The
    # loop's flat guarded copy is the same source site, not another return.
    # Keeping both postponed the real edge until after the branch merge and
    # left its return value without a dominating definition.
    if owned_returns:
        loop_controls = tuple(replace(control, root=remove_duplicate_return(control.root))
                              for control in loop_controls)

    try:
        from .control_source import order_control_region_dependencies
        program = overlay_scheduled_control(
            runtime_regions,
            (*loop_controls, *conditional_controls),
            known_nesting=nesting,
        )
        return order_control_region_dependencies(program, region_dependencies)
    except ValueError as error:
        if not isinstance(error, ControlOverlayScopeError):
            implicated_regions = {
                int(value)
                for value in re.findall(r"\d+", str(error))
            }

            def source_nodes(block):
                from .control_source import (
                    CallBlock, ConditionalBlock, LoopBlock, SequenceBlock,
                    WhileBlock,
                )
                found = []
                if isinstance(block, ConditionalBlock):
                    if block.source_node_id is not None:
                        found.append(int(block.source_node_id))
                    found.extend(source_nodes(block.body))
                    if block.orelse is not None:
                        found.extend(source_nodes(block.orelse))
                elif isinstance(block, (LoopBlock, WhileBlock)):
                    if block.source_loop_node_id is not None:
                        found.append(int(block.source_loop_node_id))
                    if isinstance(block, WhileBlock):
                        found.extend(source_nodes(block.condition))
                    found.extend(source_nodes(block.body))
                elif isinstance(block, SequenceBlock):
                    for child in block.blocks:
                        found.extend(source_nodes(child))
                elif isinstance(block, CallBlock):
                    found.extend(source_nodes(block.callee))
                return tuple(dict.fromkeys(found))

            catalogue = []
            for control_index, control in enumerate((
                *loop_controls, *conditional_controls,
            )):
                regions = tuple(map(int, control.region_indices))
                if implicated_regions and not implicated_regions.intersection(
                    regions
                ):
                    continue
                nodes = source_nodes(control.root)
                catalogue.append({
                    "index": control_index,
                    "regions": regions,
                    "source_nodes": nodes,
                    "source": tuple(
                        ast.unparse(expression)
                        for node_id in nodes
                        for expression in (
                            _source_control_expression(graph.G, node_id),
                        )
                        if isinstance(expression, ast.AST)
                    ),
                })
            raise ValueError(
                f"{error}; function="
                f"{graph.G.graph.get('function_name', '?')!r}; "
                f"controls={catalogue!r}"
            ) from error
        conflict_regions = frozenset(map(int, error.region_indices))
        overlapping = tuple(
            (reduction, control)
            for reduction, control in zip(
                reductions, loop_controls, strict=True,
            )
            if conflict_regions.intersection(map(int, control.region_indices))
        )
        named_scope_inductions = {
            component[len("loop("):-1]
            for scope in error.scopes
            for component in scope
            if component.startswith("loop(") and component.endswith(")")
        }

        def loop_inductions(block: Any) -> frozenset[str]:
            if isinstance(block, LoopBlock):
                return frozenset((str(block.induction),)) | loop_inductions(
                    block.body
                )
            if isinstance(block, SequenceBlock):
                return frozenset().union(*(
                    loop_inductions(child) for child in block.blocks
                )) if block.blocks else frozenset()
            return frozenset()

        named = tuple(
            (reduction, control)
            for reduction, control in overlapping
            if named_scope_inductions.intersection(loop_inductions(control.root))
        )
        implicated = named or overlapping
        # A conditional/conditional conflict has no loop owner at which the
        # subdivision planner can make progress. Preserve the original named
        # refusal in that case instead of fabricating a boundary.
        if not implicated:
            raise
        owner_reference = graph.G.graph.get("function_ref")
        owner_qualified_name = None
        owner_table = getattr(graph, "function_table", None)
        if owner_reference is not None and owner_table is not None:
            try:
                owner_qualified_name = str(
                    owner_table.entry(int(owner_reference)).qualified_name
                )
            except (IndexError, KeyError, TypeError, ValueError):
                owner_qualified_name = None
        scope_blockers = tuple(
            "scope:" + ">".join(scope) for scope in sorted(error.scopes)
        )
        boundaries = tuple({
            "kind": "loop-control-owner",
            "loop_node_id": int(reduction.loop_node_id),
            "region_indices": list(map(
                int, sorted(control.region_indices),
            )),
            "blockers": [
                "control-overlay-sequence-scope",
                *scope_blockers,
            ],
            **({
                "function_reference": int(owner_reference),
            } if owner_reference is not None else {}),
            **({
                "qualified_name": owner_qualified_name,
            } if owner_qualified_name is not None else {}),
        } for reduction, control in implicated)
        raise CompilationSubdivisionRequired(
            str(error), boundaries=boundaries,
        ) from error


def _control_partition_keys(
    graph: Any,
    loop_plans: Iterable[Any],
    node_ids: Iterable[int],
) -> dict[int, tuple[Any, ...]]:
    """Key nodes by lexical control and nested-call execution frontiers.

    A parent numerical region cannot span a nested function call.  Hierarchy
    composition inserts the child at the callsite, while one region marker is
    indivisible; allowing a region to contain work from both sides causes its
    post-call stages to execute before the child has produced their inputs.
    Descendant membership supplies the exact semantic frontier without
    separating independent work merely because of an arbitrary topological
    ordering.
    """

    plans = tuple(loop_plans)
    branches = _branch_compartments(graph)
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    # A while predicate is evaluated again at the latch. Its numerical
    # region cannot also own pre-loop initialization: the control overlay
    # replaces that whole region with the loop, moving intervening cap/
    # limit computations behind the loop that consumes them.
    condition_owners_by_node: dict[int, list[int]] = {}
    for plan in plans:
        loop_expression = graph.G.nodes[int(plan.loop.node_id)].get("expr_obj")
        if not isinstance(loop_expression, ast.While):
            continue
        signatures = {
            _ast_source_signature(member)
            for member in ast.walk(loop_expression.test)
            if getattr(member, "lineno", None) is not None
        }
        members = set(map(int, getattr(plan.loop, "condition_nodes", ())))
        members.update(int(node_id) for node_id, data in graph.G.nodes(data=True)
                       if isinstance(data.get("expr_obj"), ast.AST)
                       and _ast_source_signature(data["expr_obj"]) in signatures)
        for node_id in members:
            condition_owners_by_node.setdefault(node_id, []).append(int(plan.loop.node_id))
    ordinary_if_frontiers = tuple(
        (
            int(control_id),
            int(getattr(expression, "end_lineno", expression.lineno)),
        )
        for control_id, data in graph.G.nodes(data=True)
        for expression in (data.get("expr_obj"),)
        if isinstance(expression, ast.If)
    )
    branch_frontiers_by_node: dict[int, list[int]] = {}
    for control_id, end_line in ordinary_if_frontiers:
        for node_id, data in graph.G.nodes(data=True):
            expression = data.get("expr_obj")
            if not isinstance(expression, ast.AST):
                continue
            if int(getattr(expression, "lineno", -1)) > end_line:
                branch_frontiers_by_node.setdefault(
                    int(node_id), []
                ).append(int(control_id))
    callsites = tuple(
        int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        for attributes in ((data.get("attributes") or {}),)
        if (
            attributes.get("callee_ref") is not None
            or attributes.get("method_ref") is not None
        )
    )
    call_descendants = {
        callsite: nx.descendants(graph.G, callsite)
        for callsite in callsites
    }
    comprehension_owners = tuple(dict.fromkeys(
        int(node_id)
        for plan in plans
        if isinstance(
            graph.G.nodes[int(plan.loop.node_id)].get("expr_obj"),
            ast.comprehension,
        )
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(
            data.get("expr_obj"),
            (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
        )
        and any(
            int(parent) == int(plan.loop.node_id)
            and str(role) == "generators"
            for parent, role in data.get("parents", ())
        )
    ))
    comprehension_descendants = {
        owner: nx.descendants(graph.G, owner)
        for owner in comprehension_owners
    }
    expression_nodes = {
        id(data.get("expr_obj")): int(node_id)
        for node_id, data in graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.AST)
    }
    comprehension_body_members = {}
    for owner in comprehension_owners:
        aggregate = graph.G.nodes[owner].get("expr_obj")
        body_roots = (
            (aggregate.key, aggregate.value)
            if isinstance(aggregate, ast.DictComp)
            else (aggregate.elt,)
        )
        comprehension_body_members[owner] = frozenset(
            expression_nodes[id(member)]
            for body_root in body_roots
            for member in ast.walk(body_root)
            if id(member) in expression_nodes
        )
    loop_control_frontiers = tuple(
        (
            int(plan.loop.node_id),
            int(control_id),
            int(getattr(expression, "end_lineno", expression.lineno)),
            frozenset(map(int, plan.loop.body_nodes)),
        )
        for plan in plans
        for control_id in plan.loop.body_nodes
        if control_id in graph.G
        for expression in (graph.G.nodes[control_id].get("expr_obj"),)
        if isinstance(expression, ast.If)
        and any(
            isinstance(member, (ast.Break, ast.Continue))
            for member in ast.walk(expression)
        )
    )
    # Invert every owner->members relation into member->owners once, in owner
    # order, so each node's key is a set of dict lookups rather than a rescan of
    # every plan/callsite/comprehension/frontier. The old per-node membership
    # scans were O(nodes x owners) and dominated the planning pass; this is
    # O(sum of membership sizes + nodes) with byte-identical output.
    def _invert(owners, member_sets):
        by_node: dict[int, list[Any]] = {}
        for owner in owners:
            for member in member_sets(owner):
                by_node.setdefault(int(member), []).append(owner)
        return by_node

    loops_by_node = _invert(
        plans, lambda plan: plan.loop.body_nodes,
    )
    loops_by_node = {
        node_id: tuple(plan.loop.node_id for plan in owners)
        for node_id, owners in loops_by_node.items()
    }
    comp_body_by_node = _invert(
        comprehension_owners, lambda owner: comprehension_body_members[owner],
    )
    callsites_by_node = _invert(
        callsites, lambda callsite: call_descendants[callsite],
    )
    comp_desc_by_node = _invert(
        comprehension_owners, lambda owner: comprehension_descendants[owner],
    )
    frontiers_by_node: dict[int, list[tuple[int, int]]] = {}
    for loop_id, control_id, end_line, body in loop_control_frontiers:
        for member in body:
            member = int(member)
            if member not in graph.G:
                continue
            if int(getattr(
                graph.G.nodes[member].get("expr_obj"), "lineno", -1,
            )) > end_line:
                frontiers_by_node.setdefault(member, []).append(
                    (loop_id, control_id)
                )

    return {
        node_id: (
            tuple(loops_by_node.get(int(node_id), ())),
            tuple(sorted(branches.get(node_id, ()))),
            tuple(comp_body_by_node.get(int(node_id), ())),
            tuple(callsites_by_node.get(int(node_id), ())),
            tuple(comp_desc_by_node.get(int(node_id), ())),
            tuple(frontiers_by_node.get(int(node_id), ())),
            tuple(branch_frontiers_by_node.get(int(node_id), ())),
            tuple(condition_owners_by_node.get(int(node_id), ())),
        )
        for node_id in node_ids
    }

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from ..transmogrifier.ssa import SSAValue, Instr

if TYPE_CHECKING:  # pragma: no cover - optional heavy deps
    from ..transmogrifier.graph.graph_express2 import ProcessGraph


def process_graph_to_ssa_instrs(pg: ProcessGraph, schedule: str = "alap") -> List[Instr]:
    """Convert a ProcessGraph into a linear SSA instruction list.

    The graph's embedded scheduler is executed using ``schedule`` ("alap" by
    default) and nodes are emitted in the resulting level order. This preserves
    any memory/storage nodes inserted by the scheduler and mirrors the intended
    execution order.

    Loop analysis is intentionally omitted – ProcessGraphs are expected to be
    acyclic once scheduled. Cyclic behaviour must be resolved prior to invoking
    this helper.

    NOT the compiler entrypoint for a general Python function. This reads
    whatever ``op``/``label`` a node already carries (falling back to the
    raw ``expr_obj``'s Python type name, or worse, the object itself, if
    neither is set) -- it does not resolve control flow, attribute access,
    or free names itself. Feeding it a ProcessGraph built by
    graph_express2.ProcessGraph.build_from_ast() without first running the
    real pipeline's control/region/binding passes silently emits degenerate
    instructions (``op`` holding a literal ``<ast.Name object at 0x...>``)
    for anything past straight-line scalar expressions, without raising.
    For real Python-function compilation use
    src.common.tensors.accelerator_backends.aot_compile.compile_ast_aot,
    which drives this same underlying machinery correctly.
    """

    # Run the embedded scheduler. ``compute_levels`` populates ``pg.levels`` and
    # performs side effects such as inserting memory nodes or interference
    # graphs.  It returns ``None`` in the full implementation, but our minimal
    # stub used in tests may return the level mapping directly.  We honour both
    # behaviours.
    ret = pg.compute_levels(method=schedule, order="dependency")
    levels = ret if ret is not None else pg.levels
    order = sorted(levels, key=lambda n: levels[n])
    distinct_levels = tuple(sorted({int(level) for level in levels.values()}))
    schedule_groups = {
        level: ordinal for ordinal, level in enumerate(distinct_levels)
    }

    values: Dict[int, SSAValue] = {}
    instrs: List[Instr] = []

    for nid in order:
        data = pg.G.nodes[nid]
        op = data.get("op") or data.get("label")
        expr_obj = data.get("expr_obj")
        if op is None and expr_obj is not None:
            op = type(expr_obj).__name__
        parent_items = list(data.get("parents", []))
        parents = [p for p, _ in parent_items]
        roles = [role for _, role in parent_items]
        tensor = data.get("tensor") or {}
        accounting = data.get("bit_quanta") or {}
        if hasattr(accounting, "__dict__"):
            accounting = dict(accounting.__dict__)
        res = values.setdefault(
            nid,
            SSAValue(
                nid,
                dtype=tensor.get("dtype"),
                shape=tuple(tensor.get("shape", ())),
                device=tensor.get("device"),
                accounting=dict(accounting),
            ),
        )

        # Detect back-edges (loop-carried dependencies).  Any parent scheduled
        # at the same or a later level feeds a previous iteration and must be
        # merged via a ``phi`` node before the actual operation executes.
        back_parents = [p for p in parents if levels.get(p, -1) >= levels[nid]]
        if back_parents:
            phi_args = [values.setdefault(p, SSAValue(p)) for p in back_parents]
            instrs.append(Instr("phi", phi_args, res))
            keep = [(p, role) for p, role in zip(parents, roles) if p not in back_parents]
            parents = [p for p, _ in keep]
            roles = [role for _, role in keep]

        args = [values.setdefault(p, SSAValue(p)) for p in parents]
        attributes = dict(data.get("attributes") or data.get("extra_args") or {})
        attributes.update({
            "schedule_level": int(levels[nid]),
            "schedule_group": schedule_groups[int(levels[nid])],
            "schedule_method": str(schedule),
        })
        from .semantic_translation import (
            SemanticRepresentation, semantic_identity,
        )
        source_identity = semantic_identity(
            str(op), SemanticRepresentation.PROCESS_GRAPH,
            facets=dict(attributes.get("semantic_facets") or {}),
        )
        source_family = attributes.get("semantic_family")
        if source_family is not None:
            from .semantic_translation import SemanticOperationIdentity
            source_identity = SemanticOperationIdentity(
                str(source_family), SemanticRepresentation.PROCESS_GRAPH,
                str(attributes.get("semantic_spelling") or op),
                dict(attributes.get("semantic_facets") or {}),
            )
        from .semantic_translation import SemanticOperationIdentity
        target_identity = SemanticOperationIdentity(
            source_identity.family,
            SemanticRepresentation.REPOSITORY_SSA,
            str(op), dict(source_identity.facets),
        )
        attributes.update(target_identity.attributes())
        attributes["semantic_source_representation"] = (
            source_identity.representation.value
        )
        if data.get("constant") is not None:
            attributes["value"] = data["constant"]
        source = data.get("source_span")
        instrs.append(
            Instr(
                op,
                args,
                res,
                arg_roles=roles,
                attributes=attributes,
                source_span=source,
            )
        )

    return instrs

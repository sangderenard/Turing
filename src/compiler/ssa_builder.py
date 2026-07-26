from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

from ..transmogrifier.ssa import SSAValue, Instr
from ..transmogrifier.process_op import ProcessOp

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
    """

    # Run the embedded scheduler. ``compute_levels`` populates ``pg.levels`` and
    # performs side effects such as inserting memory nodes or interference
    # graphs.  It returns ``None`` in the full implementation, but our minimal
    # stub used in tests may return the level mapping directly.  We honour both
    # behaviours.
    ret = pg.compute_levels(method=schedule, order="dependency")
    levels = ret if ret is not None else pg.levels
    order = sorted(levels, key=lambda n: levels[n])

    values: Dict[int, SSAValue] = {}
    instrs: List[Instr] = []

    for nid in order:
        data = pg.G.nodes[nid]
        process_op = data.get("process_op")
        op = process_op.op if isinstance(process_op, ProcessOp) else data.get("label")
        expr_obj = data.get("expr_obj")
        if op is None and expr_obj is not None:
            op = type(expr_obj).__name__
        parent_items = list(data.get("parents", []))
        parents = [p for p, _ in parent_items]
        roles = [role for _, role in parent_items]
        tensor = process_op.tensor if isinstance(process_op, ProcessOp) else None
        res = values.setdefault(
            nid,
            SSAValue(
                nid,
                dtype=tensor.dtype if tensor else None,
                shape=tensor.shape if tensor else (),
                device=tensor.device if tensor else None,
                accounting=(
                    process_op.bit_quanta.__dict__
                    if isinstance(process_op, ProcessOp)
                    and process_op.bit_quanta is not None
                    else {}
                ),
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
        attributes = dict(process_op.attributes) if isinstance(process_op, ProcessOp) else {}
        if isinstance(process_op, ProcessOp) and process_op.constant is not None:
            attributes["value"] = process_op.constant
        source = (
            process_op.source.__dict__
            if isinstance(process_op, ProcessOp) and process_op.source is not None
            else None
        )
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

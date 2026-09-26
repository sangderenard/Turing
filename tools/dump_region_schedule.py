"""Dump one function's region schedule during the managed balloon-tire compile.

``python tools/dump_region_schedule.py <function-name-substring>`` hooks the
planner's ``_build_shell_hierarchy_plan`` and ``_topological_region_order``
and prints, for every shell whose function name contains the substring:
the plan item order with captures/outputs per region, the ordering
routine's candidates, dependency edges and result, and every GetAttr /
SetAttr / IndexedStore node with its parents and edge roles so field
ordering edges (``after_write``) are visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler import glsl_deployment_strategy as planner  # noqa: E402
from src.compiler.hierarchical_plan import PlanClosure, PlanLine  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    lower_balloon_tire_managed_python_ssa,
)


def _field_nodes(graph) -> None:
    for node_id, data in graph.nodes(data=True):
        operation = str(data.get("type") or data.get("op") or "")
        if operation.casefold() not in {"getattr", "setattr", "indexedstore"}:
            continue
        print(f"    node {node_id} value_id={data.get('value_id')} op={operation} "
              f"attr={(data.get('attributes') or {}).get('attribute')} "
              f"parents={tuple(data.get('parents') or ())} span={data.get('source_span')}")


def main() -> int:
    wanted = sys.argv[1]
    original_plan = planner._build_shell_hierarchy_plan
    original_order = planner._topological_region_order

    def plan_hook(shell):
        plan = original_plan(shell)
        graph = shell.process_graph.G
        name = str(graph.graph.get("function_name"))
        if wanted in name:
            print(f"PLAN {name}")
            for item in plan.items:
                if isinstance(item, PlanClosure):
                    lines = [
                        (line.opcode, tuple(map(int, line.inputs)), tuple(map(int, line.outputs)))
                        for line in item.items if isinstance(line, PlanLine)
                    ]
                    print(f"  item {item.name} captures={item.captures} lines={lines}")
                else:
                    print(f"  item {type(item).__name__} {getattr(item, 'opcode', '')} "
                          f"in={getattr(item, 'inputs', ())} out={getattr(item, 'outputs', ())}")
            for index, subgraph in enumerate(getattr(shell, "dispatch_subgraphs", ())):
                meta = subgraph.G.graph
                print(f"  dispatch[{index}] nodes={sorted(map(int, meta.get('deployment_nodes', ())))} "
                      f"outputs={sorted(map(int, meta.get('deployment_outputs', ())))}")
            recursion = graph.graph.get("recursion_table") or {}
            for key, record in recursion.items():
                print(f"  recursion {key}: control_ir={record.get('control_ir', True)} "
                      f"control_members={sorted(map(int, record.get('control_members', ())))}")
            print("  field nodes:")
            _field_nodes(graph)
        return plan

    def order_hook(shell, candidate_regions):
        result = original_order(shell, candidate_regions)
        graph = shell.process_graph.G
        name = str(graph.graph.get("function_name"))
        if wanted in name:
            print(f"ORDER {name}: candidates={tuple(candidate_regions)} -> {result}")
        return result

    planner._build_shell_hierarchy_plan = plan_hook
    planner._topological_region_order = order_hook
    try:
        lower_balloon_tire_managed_python_ssa(progress=lambda message: None)
    finally:
        planner._build_shell_hierarchy_plan = original_plan
        planner._topological_region_order = original_order
    print("LOWERED OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

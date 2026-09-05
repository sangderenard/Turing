"""Which scheduled regions publish the same value id?

``_propose_dt_pen`` (real dt_controller source, real ABI) lowers with SSA
value 50 (``dt_cfl / penalty``, the fall-through return value) defined by TWO
region result loads: one after the ``max(...)`` loop, one after the
``if energy_limit is not None`` merge.  Hook the hierarchy plan and print
every ``region_*`` closure's lines for that function so the duplicated
producer is visible at the plan level (before SSA emission).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler import glsl_deployment_strategy as planner  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.hierarchical_plan import PlanClosure, PlanLine  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import _propose_dt_pen, _energy_time_limit  # noqa: E402
from src.common.dt_system.dt_scaler import _scalar  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


class _Done(Exception):
    pass


def _base_records():
    import numpy as np

    stub = BalloonTireManagedState.__new__(BalloonTireManagedState)
    for name in (
        "inputs", "state", "output", "wheel_input_indices", "rest",
        "face_vertices", "face_rest", "face_scatter", "bending_incidence",
        "bending_scatter", "bending_weight", "vertex_area", "bead_mask",
        "face_material", "telemetry",
    ):
        setattr(stub, name, np.zeros((1,), dtype=np.float64))
    return balloon_tire_managed_extraction_contract(stub).program_abi.receipt()


def main() -> int:
    source = "\n\n".join((
        inspect.getsource(_scalar),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_propose_dt_pen),
        "def root(metrics, targets, dx, distribution):\n"
        "    return _propose_dt_pen(metrics, targets, dx, distribution)\n",
    ))
    base = _base_records()
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
        "values": [],
    })
    original = planner._build_shell_hierarchy_plan
    seen = 0

    def capture(shell):
        nonlocal seen
        plan = original(shell)
        graph = shell.process_graph.G
        if graph.graph.get("function_name") == "_propose_dt_pen":
            seen += 1
            producers: dict[int, list[str]] = {}
            for item in plan.items:
                if not isinstance(item, PlanClosure) or not item.name.startswith("region_"):
                    continue
                print(f"{item.name}: captures={item.captures}")
                for line in item.items:
                    if isinstance(line, PlanLine):
                        print(f"    {line.opcode} inputs={line.inputs} outputs={line.outputs}")
                        for output in line.outputs:
                            producers.setdefault(int(output), []).append(item.name)
            duplicated = {k: v for k, v in producers.items() if len(v) > 1}
            print(f"DUPLICATED-OUTPUTS {duplicated}")
            for index, subgraph in enumerate(getattr(shell, "dispatch_subgraphs", ())):
                meta = subgraph.G.graph
                print(f"dispatch[{index}] nodes={sorted(map(int, meta.get('deployment_nodes', ())))} "
                      f"outputs={sorted(map(int, meta.get('deployment_outputs', ())))} "
                      f"required={sorted(map(int, meta.get('required_outputs', ())))}")
            memberships = planner._branch_compartments(shell.process_graph)
            for value_id in duplicated:
                node = graph.nodes.get(int(value_id)) or {}
                print(f"  node {value_id}: op={node.get('op')} span={node.get('source_span')} "
                      f"parents={node.get('parents')} memberships={sorted(memberships.get(int(value_id), ()))}")
            raise _Done()
        return plan

    planner._build_shell_hierarchy_plan = capture
    try:
        lower_ast_source_to_ssa(
            source, "root", name="region_dupes", extraction_contract=policy,
        )
        print("lowering finished without capturing _propose_dt_pen?!")
    except _Done:
        pass
    finally:
        planner._build_shell_hierarchy_plan = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

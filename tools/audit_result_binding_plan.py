"""Plan-level audit: why ``run_superstep``'s call to
``step_with_dt_control_used`` gets ``result_bindings=()``.

The semantic result bindings of a call are frozen in
``glsl_deployment_strategy._build_shell_hierarchy_plan`` (PlanCall
construction).  That plan is (re)built several times during shell
preparation, before repository-SSA lowering and before the
'opaque-state-effect' blocker the existing repro tools hit.  This script hooks
the plan builder and, every time a ``run_superstep`` shell WITH attached
callsite shells is planned, prints:

* the callee's ``function_outputs`` and identity histories per output name;
* the child output paths the planner expands (with aggregate leaves);
* the caller's successors of the call node (the unpack projections);
* the resulting ``PlanCall.result_bindings``.

Real sources (``inspect.getsource``) for every dt_controller collaborator,
exactly as tools/repro_run_superstep.py; ``advance`` is the one fabricated
collaborator (it does not participate in the call under audit).  Set
AUDIT_STOP_AFTER=N to abort after the N-th printed plan (default 1).
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler import glsl_deployment_strategy as planner  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.hierarchical_plan import PlanCall, PlanClosure  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import (  # noqa: E402
    _restore_type, Targets, _shadow_dt_limit, _energy_time_limit,
    _no_exchange_observed, _apply_energy_sidechain, STController,
    _propose_dt_pen, step_with_dt_control_used, run_superstep,
)
from src.common.dt_system.dt_scaler import Metrics, _scalar, coerce_metrics  # noqa: E402
from src.common.dt_system.shadow import shadow_dt_limit  # noqa: E402

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


def _flat_calls(items):
    for item in items:
        if isinstance(item, PlanCall):
            yield item
        elif isinstance(item, PlanClosure):
            yield from _flat_calls(item.items)


def _describe(graph, value_id):
    if value_id not in graph:
        return f"{value_id}:<absent>"
    data = graph.nodes[value_id]
    expr = data.get("expr_obj")
    where = ""
    if isinstance(expr, ast.AST) and getattr(expr, "lineno", None):
        where = f"@L{expr.lineno}"
    attrs = data.get("attributes") or {}
    leaves = attrs.get("aggregate_leaf_value_ids")
    extra = f" leaves={tuple(leaves)}" if leaves else ""
    return (
        f"{value_id}:{data.get('type')}/{data.get('op')}"
        f"[{data.get('label')!r}]{where}{extra}"
    )


def _report(shell, plan, t0) -> bool:
    graph = shell.process_graph.G
    calls = list(_flat_calls(plan.items))
    printed = False
    for call in calls:
        child = shell.callsite_function_shells.get(call.callsite_id)
        if child is None:
            continue
        child_graph = child.process_graph.G
        child_name = child_graph.graph.get("function_name")
        if child_name != "step_with_dt_control_used":
            continue
        printed = True
        print(f"\n=== [{time.time()-t0:.1f}s] run_superstep plan: callsite "
              f"{call.callsite_id} -> {child_name} "
              f"(shell {type(shell).__name__})", flush=True)
        outputs = tuple(child_graph.graph.get("function_outputs", ()))
        print(f"function_outputs = {outputs}")
        identities = child_graph.graph.get("identity_table") or {}
        for output_name in outputs:
            history = identities.get(output_name, ())
            print(f"  identity[{output_name!r}] ({len(history)} versions):")
            for value_id in history:
                print(f"      {_describe(child_graph, int(value_id))}")
        for probe in ("metrics", "result_1", "result_2", "dt_next"):
            history = identities.get(probe, ())
            print(f"  identity[{probe!r}] ({len(history)} versions):")
            for value_id in history[-6:]:
                print(f"      {_describe(child_graph, int(value_id))}")
        print(f"  captured_return_value_ids = "
              f"{sorted(getattr(child, '_captured_return_value_ids', ()))}")
        nested_returns = [
            node for node in ast.walk(ast.parse(inspect.getsource(
                step_with_dt_control_used
            )))
            if isinstance(node, ast.Return)
        ]
        print(f"  ast.Return statements in callee source: "
              f"{[node.lineno for node in nested_returns]} "
              f"(top-level body Returns: "
              f"{[type(s).__name__ for s in ast.parse(inspect.getsource(step_with_dt_control_used)).body[0].body]})")
        child_outputs = tuple(
            int(identities[name][-1]) for name in outputs
            if identities.get(name)
        )
        print(f"  child_outputs (last identity per name) = {child_outputs}")
        for position, output_id in enumerate(child_outputs):
            leaves = planner._authored_aggregate_leaves(
                child.process_graph, output_id
            )
            print(f"    path ({position},) -> "
                  f"{_describe(child_graph, output_id)}  leaves={leaves}")
        print("  caller call node:", _describe(graph, int(call.callsite_id)))
        print("  caller successors of the call node:")
        for successor in graph.successors(int(call.callsite_id)):
            data = graph.nodes[successor]
            parents = tuple(
                (int(parent), str(role))
                for parent, role in data.get("parents") or ()
            )
            print(f"      {_describe(graph, successor)} parents={parents}")
            for parent, role in parents:
                if role == "index":
                    pdata = graph.nodes.get(parent, {})
                    print(f"          index parent {parent}: "
                          f"type={pdata.get('type')} "
                          f"constant={pdata.get('constant')!r} "
                          f"attrs={pdata.get('attributes')!r}")
        print(f"  result_value_ids   = {call.result_value_ids}")
        print(f"  result_bindings    = {call.result_bindings}")
        print(f"  enclosing_loop_ids = {call.enclosing_loop_ids}", flush=True)
    return printed


def main() -> int:
    real_source = "\n\n".join((
        inspect.getsource(_scalar),
        inspect.getsource(coerce_metrics),
        inspect.getsource(_restore_type),
        inspect.getsource(shadow_dt_limit),
        inspect.getsource(_shadow_dt_limit),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_no_exchange_observed),
        inspect.getsource(_apply_energy_sidechain),
        inspect.getsource(_propose_dt_pen),
        inspect.getsource(step_with_dt_control_used),
        inspect.getsource(run_superstep),
    ))
    advance_source = (
        "def advance(state, dt):\n"
        "    return True, Metrics(\n"
        "        max_vel=float(state.state[0]),\n"
        "        max_flux=float(state.state[0]),\n"
        "        div_inf=0.0,\n"
        "        mass_err=0.0,\n"
        "    )\n"
    )
    root_source = (
        "def root(state, round_max, dt_init, dx, targets, ctrl):\n"
        "    return run_superstep(\n"
        "        state, round_max, dt_init, dx, targets, ctrl, advance,\n"
        "        rollback=True,\n"
        "    )\n"
    )
    source = real_source + "\n\n" + advance_source + "\n\n" + root_source
    base = _base_records()
    contract_abi = {
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
            "STController": base["records"]["STController"],
            "BalloonTireManagedState": base["records"][
                "BalloonTireManagedState"
            ],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
            {"function": "*", "parameter": "ctrl", "record": "STController"},
            {
                "function": "*", "parameter": "state",
                "record": "BalloonTireManagedState",
            },
        ],
        "values": [],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi)

    stop_after = int(os.environ.get("AUDIT_STOP_AFTER", "1"))
    printed = 0
    original = planner._build_shell_hierarchy_plan
    t0 = time.time()

    def capture(shell):
        nonlocal printed
        plan = original(shell)
        name = shell.process_graph.G.graph.get("function_name")
        if name == "run_superstep" and shell.callsite_function_shells:
            if _report(shell, plan, t0):
                printed += 1
                if printed >= stop_after:
                    raise _Done()
        return plan

    planner._build_shell_hierarchy_plan = capture
    try:
        lower_ast_source_to_ssa(
            source, "root", name="audit_result_binding",
            extraction_contract=policy,
        )
        print(f"lowering finished in {time.time()-t0:.1f}s")
    except _Done:
        print(f"\nstopped after {printed} plan report(s) at {time.time()-t0:.1f}s")
    except Exception as error:  # noqa: BLE001
        print(f"\naborted after {time.time()-t0:.1f}s: "
              f"{type(error).__name__}: {str(error)[:600]}")
    finally:
        planner._build_shell_hierarchy_plan = original
    if not printed:
        print("no run_superstep plan with a step_with_dt_control_used "
              "callsite was ever built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

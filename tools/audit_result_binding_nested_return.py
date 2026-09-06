"""Minimal synthetic confirmation of the ``result_bindings=()`` root cause.

``topological_reducer`` derives ``function_outputs`` only from a Return that
is a TOP-LEVEL body statement of the function (topological_reducer.py
~1272-1287).  A function whose every ``return`` is nested (inside
``while True:``, like ``step_with_dt_control_used``) therefore gets
``function_outputs=()``, no output identities, and the hierarchy planner has
no child output paths to correlate with the caller's ``Indexed`` unpack
projections -> ``PlanCall.result_bindings=()``.

Two variants of the same two-output callee, called from a ``while`` loop:

* NESTED:   both returns live inside ``while True:``  -> expected ()
* TOPLEVEL: one extra top-level return after the loop -> expected 2 bindings

Only the plan is inspected; lowering is aborted right after the caller's
plan is built, so this runs in seconds regardless of later SSA stages.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler import glsl_deployment_strategy as planner  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.hierarchical_plan import PlanCall, PlanClosure  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

NESTED = """
def leaf(x, n):
    i = 0
    while True:
        i = i + 1
        if i > n:
            return x * 2.0, i
        if i > 100:
            return x, i


def tick(x, n):
    total = x * 0.0
    k = 0
    j = 0
    while j < 2:
        y, k = leaf(x, n)
        total = total + y
        j = j + 1
    return total, k
"""

TOPLEVEL = """
def leaf(x, n):
    i = 0
    while True:
        i = i + 1
        if i > n:
            return x * 2.0, i
        if i > 100:
            break
    return x, i


def tick(x, n):
    total = x * 0.0
    k = 0
    j = 0
    while j < 2:
        y, k = leaf(x, n)
        total = total + y
        j = j + 1
    return total, k
"""


class _Done(Exception):
    pass


def _contract():
    values = [
        {
            "function": "tick", "parameter": "x", "storage": "span",
            "dtype": "float64", "rank": 1, "shape": [4],
            "python_type": "src.common.tensors.abstraction.AbstractTensor",
        },
        {
            "function": "tick", "parameter": "n", "storage": "scalar",
            "dtype": "int64", "rank": 0, "python_type": "builtins.int",
        },
    ]
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(
        {"records": {}, "bindings": [], "values": values}
    ).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")


def _flat_calls(items):
    for item in items:
        if isinstance(item, PlanCall):
            yield item
        elif isinstance(item, PlanClosure):
            yield from _flat_calls(item.items)


def run(label: str, source: str) -> None:
    original = planner._build_shell_hierarchy_plan
    t0 = time.time()

    def capture(shell):
        plan = original(shell)
        graph = shell.process_graph.G
        if (
            graph.graph.get("function_name") == "tick"
            and shell.callsite_function_shells
        ):
            for call in _flat_calls(plan.items):
                child = shell.callsite_function_shells.get(call.callsite_id)
                if child is None:
                    continue
                child_graph = child.process_graph.G
                identities = child_graph.graph.get("identity_table") or {}
                outputs = tuple(child_graph.graph.get("function_outputs", ()))
                print(f"[{label}] callee function_outputs = {outputs}")
                for name in outputs:
                    print(f"[{label}]   identity[{name!r}] = "
                          f"{tuple(identities.get(name, ()))}")
                print(f"[{label}]   caller Indexed successors = "
                      f"{[s for s in graph.successors(int(call.callsite_id)) if str(graph.nodes[s].get('op')).lower() == 'indexed']}")
                print(f"[{label}]   PlanCall.result_value_ids = "
                      f"{call.result_value_ids}")
                print(f"[{label}]   PlanCall.result_bindings  = "
                      f"{call.result_bindings}  ({time.time()-t0:.1f}s)")
            raise _Done()
        return plan

    planner._build_shell_hierarchy_plan = capture
    try:
        lower_ast_source_to_ssa(
            source, "tick",
            python_bindings={"AbstractTensor": AbstractTensor},
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            name=f"audit_{label.lower()}", runtime_closure_only=True,
            extraction_contract=_contract(),
        )
        print(f"[{label}] lowering finished without a tick plan?!")
    except _Done:
        pass
    except Exception as error:  # noqa: BLE001
        print(f"[{label}] aborted: {type(error).__name__}: {str(error)[:400]}")
    finally:
        planner._build_shell_hierarchy_plan = original


def main() -> int:
    run("NESTED", NESTED)
    run("TOPLEVEL", TOPLEVEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

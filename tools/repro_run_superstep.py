"""Lower the real ``run_superstep`` source closure with a tiny advance kernel.

Use the callable's real module namespace, as the managed tire lowering does.
Copying selected function bodies into an anonymous module loses method
resolution and produces a harness-only ``opaque-state-effect`` for
``ctrl.update_dt_max``. ``--diagnose-effects`` prints opaque loop effects
without changing their classification.

``advance`` is the one fabricated collaborator here, exactly as every
existing dt-controller unit test already fabricates one.
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import run_superstep  # noqa: E402
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


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
    if "--diagnose-effects" in sys.argv:
        from src.compiler.loop_composer import LoopComposer
        from src.compiler.loop_ir import LoopStateEffectMode

        describe = LoopComposer.describe

        def describe_with_effects(self, graph, node_id):
            loop = describe(self, graph, node_id)
            for effect in loop.state_effects:
                if effect.mode is not LoopStateEffectMode.OPAQUE:
                    continue
                data = graph.G.nodes.get(effect.effect_node_id, {})
                expression = data.get("expr_obj")
                print(
                    f"OPAQUE {graph.G.graph.get('function_name')} "
                    f"loop={node_id}: {effect} "
                    f"source={ast.unparse(expression) if isinstance(expression, ast.AST) else None}",
                    flush=True,
                )
            return loop

        LoopComposer.describe = describe_with_effects
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
    # Resolve the real callable with its module globals, as the managed
    # lowering does. Copying function text into an anonymous module loses
    # the STController class needed to source-link its bound methods.
    source = advance_source + "\n\n" + root_source
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
            # Preserve the managed contract's function-scoped bindings too:
            # coerce_metrics calls its Metrics receiver ``value``.
            *base["bindings"],
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

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="run_superstep_repro", extraction_contract=policy,
            python_bindings={"run_superstep": run_superstep, "Metrics": Metrics},
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:2500]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

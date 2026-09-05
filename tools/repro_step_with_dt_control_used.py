"""Fast, real repro: the actual ``step_with_dt_control_used`` (its full
``while True:`` body, all 4 real return statements, real ``rollback=True``
retry/rollback path) called once, via ``inspect.getsource`` on every real
collaborator it needs -- reproducing the exact caller shape from
``run_superstep`` that the full managed-tire compile reports as
``unresolved_calls``: ``callee_output_count=15, physical_result_count=0,
semantic_result_count=0`` (every one of the callee's 15 real outputs -- the
13 Metrics fields plus dt_next/dt_used -- unbound at the call site).

``advance`` is the one fabricated collaborator here, exactly as every
existing dt-controller unit test already fabricates one (see
tests/dt_system/test_dt_controller.py); the function actually under
suspicion, ``step_with_dt_control_used`` itself, is the real, unmodified
source from dt_controller.py.
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import (  # noqa: E402
    _restore_type, Targets, _shadow_dt_limit, _energy_time_limit,
    _no_exchange_observed, _apply_energy_sidechain, STController,
    _propose_dt_pen, step_with_dt_control_used,
)
from src.common.dt_system.dt_scaler import Metrics, _scalar, coerce_metrics  # noqa: E402
from src.common.dt_system.shadow import shadow_dt_limit  # noqa: E402

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
    ))
    # advance is the one fabricated collaborator (see module docstring); its
    # body is trivial by design so the only thing under test is
    # step_with_dt_control_used's own real control/record flow.
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
        "def root(state, dt, dx, targets, ctrl):\n"
        "    return step_with_dt_control_used(\n"
        "        state, dt, dx, targets, ctrl, advance,\n"
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

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="step_used", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:2000]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast, real repro: the actual ``run_superstep`` (its real while-loop,
calling the real ``step_with_dt_control_used``) called once, via
``inspect.getsource`` on every real collaborator it needs -- reproducing
the exact 4-level real call chain the full managed-tire compile reports as
blocked: ``run_superstep -> step_with_dt_control_used -> _apply_energy_
sidechain -> _no_exchange_observed``. tools/repro_step_with_dt_control_used.py
(3 levels: root calls step_with_dt_control_used directly) hit a DIFFERENT,
repro-only artifact ("opaque-state-effect"/CompilationSubdivisionRequired)
that the real full compile never reports -- confirmed by running the real
compile with all fixes applied and seeing the exact same, unrelated
``unresolved_calls`` diagnostic as before. This one goes one level deeper
(through run_superstep) to see whether that changes the region shape enough
to reach the real remaining blocker: ``_no_exchange_observed``'s output
getting registered as a bogus empty-fields record placeholder instead of a
plain boolean, and ``step_with_dt_control_used``'s own 15 real outputs
never getting bound at its call site.

``advance`` is the one fabricated collaborator here, exactly as every
existing dt-controller unit test already fabricates one.
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
    _propose_dt_pen, step_with_dt_control_used, run_superstep,
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

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="run_superstep_repro", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:2500]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

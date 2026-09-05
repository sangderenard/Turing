"""Fast, real repro: the actual ``_apply_energy_sidechain`` calling the real
``_no_exchange_observed`` inside an ``if`` (not returned directly), via
``inspect.getsource`` -- reproducing the exact caller shape from
dt_controller.py that the isolated ``root() -> _no_exchange_observed(...)``
repro does not exercise (there the boolean is directly returned; here it
only steers a branch), to find why the full managed-tire compile still
shows ``_no_exchange_observed``'s output classified as a record with an
empty layout instead of a plain boolean value.
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
    _apply_energy_sidechain, _no_exchange_observed, _energy_time_limit,
    _shadow_dt_limit,
)
from src.common.dt_system.dt_scaler import _scalar  # noqa: E402
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
        inspect.getsource(shadow_dt_limit),
        inspect.getsource(_shadow_dt_limit),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_no_exchange_observed),
        inspect.getsource(_apply_energy_sidechain),
    ))
    root_source = (
        "def root(dt_next, dt_tensor, metrics, targets):\n"
        "    return _apply_energy_sidechain(dt_next, dt_tensor, metrics, targets)\n"
    )
    source = real_source + "\n\n" + root_source
    base = _base_records()
    contract_abi = {
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
        "values": [],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi)

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="sidechain", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1500]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

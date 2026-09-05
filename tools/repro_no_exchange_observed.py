"""Fast, real repro: the actual ``_no_exchange_observed`` calling the real
``_scalar`` (default-parameter helper) from dt_controller.py/dt_scaler.py,
via ``inspect.getsource`` -- the second shape harmonization uncovered once
the first (targets expansion) was fixed.
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
from src.common.dt_system.dt_controller import _no_exchange_observed  # noqa: E402
from src.common.dt_system.dt_scaler import _scalar  # noqa: E402

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
        inspect.getsource(_no_exchange_observed),
    ))
    root_source = (
        "def root(metrics, targets):\n"
        "    return _no_exchange_observed(metrics, targets)\n"
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
            source, "root", name="no_exchange", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:800]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

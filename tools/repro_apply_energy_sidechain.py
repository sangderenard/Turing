"""Lower the real DT energy-sidechain closure and retain C emission failures.

Uses the callable's actual globals and the complete managed record ABI under
native-only execution. Saves repository SSA so emitter fixes can be checked
without repeating source lowering. This is a focused diagnostic, not DT parity.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import _apply_energy_sidechain, _propose_dt_pen  # noqa: E402

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry", choices=("sidechain", "proposal"), default="sidechain")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root_source = (
        "def root(dt_next, dt_tensor, metrics, targets):\n"
        "    return _apply_energy_sidechain(dt_next, dt_tensor, metrics, targets)\n"
    )
    bindings = {"_apply_energy_sidechain": _apply_energy_sidechain}
    if args.entry == "proposal":
        root_source = (
            "def root(metrics, targets, dx):\n"
            "    return _propose_dt_pen(metrics, targets, dx, None)\n"
        )
        bindings = {"_propose_dt_pen": _propose_dt_pen}
    source = root_source
    base = _base_records()
    contract_abi = {**base,
        "bindings": [*base.get("bindings", []),
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi).with_execution_file(
        CONTRACTS / "vehicle_full_native_execution.yaml")

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="sidechain", extraction_contract=policy,
            python_bindings=bindings,
        )
        (args.output / "repository-ssa.pkl").write_bytes(pickle.dumps((module, outputs, exports), protocol=5))
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, exports[0], entry_name="sidechain_native")
        (args.output / "module.c").write_text(artifact.source, encoding="utf-8")
        failures = [{"operation": item.operation, "reason": item.reason} for item in artifact.shortfalls]
        (args.output / "shortfalls.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(json.dumps(failures, indent=2), flush=True)
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0 if artifact.complete else 1
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1500]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

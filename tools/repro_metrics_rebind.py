"""Fast, real repro: does `metrics = coerce_metrics(metrics)` (a call result
reassigned to the same source name) get a fresh SSA identity distinct from
the pre-call `metrics`, or does the compiler treat pre- and post-coercion
`metrics` as one shared value? Isolates the exact pattern from
step_with_dt_control_used (dt_controller.py) that's producing the
"actual value N is bound to two contracted formals wanting different
dtypes" conflict in the full managed-tire compile, via inspect.getsource on
the real functions -- not a fabricated stand-in.
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
from src.common.dt_system.dt_controller import _propose_dt_pen  # noqa: E402
from src.common.dt_system.dt_scaler import coerce_metrics, _scalar  # noqa: E402

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
        inspect.getsource(_propose_dt_pen),
    ))
    # Mirrors step_with_dt_control_used's own real pattern exactly: metrics
    # is coerced, THEN reassigned to the same name, THEN passed to
    # _propose_dt_pen -- inside a real while-loop, since the real function's
    # entire body is `while True:` and the first isolated (loop-free) repro
    # of this same rebind lowered cleanly, ruling that shape out.
    root_source = (
        "def root(metrics, targets, dx, distribution, attempts):\n"
        "    dt_pen = 0.0\n"
        "    count = 0\n"
        "    while count < attempts:\n"
        "        metrics = coerce_metrics(metrics)\n"
        "        dt_pen = _propose_dt_pen(metrics, targets, dx, distribution)\n"
        "        count = count + 1\n"
        "    return dt_pen\n"
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
            {"function": "*", "parameter": "value", "record": "Metrics"},
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
            source, "root", name="metrics_rebind", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
        return 0
    except Exception as error:
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1200]}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

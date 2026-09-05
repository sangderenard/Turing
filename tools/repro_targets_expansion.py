"""Fast, real repro: the actual ``coerce_metrics`` and ``_propose_dt_pen``
from dt_scaler.py/dt_controller.py, via ``inspect.getsource`` (not retyped),
called from one small real wrapper that forwards ``metrics``/``targets``
without touching their fields itself -- the exact shape
``step_with_dt_control_used`` produces, isolated from that function's
separate ``while True:`` retry-loop issue.
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.ssa_reference_evaluator import SSAReferenceEvaluator  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import _propose_dt_pen, _energy_time_limit  # noqa: E402
from src.common.dt_system.dt_scaler import coerce_metrics, _scalar, Metrics  # noqa: E402

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
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_propose_dt_pen),
    ))
    # Real forwarding shape: root receives metrics/targets and passes them
    # on to two DIFFERENT real functions, neither of which it touches
    # itself -- exactly what step_with_dt_control_used does, minus its
    # separate while-loop retry logic.
    root_source = (
        "def root(metrics, targets, dx, distribution):\n"
        "    coerced = coerce_metrics(metrics)\n"
        "    dt_pen = _propose_dt_pen(coerced, targets, dx, distribution)\n"
        "    return coerced.hard_failure + dt_pen\n"
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
        "values": [
            {"function": "root", "parameter": "dx", "storage": "scalar",
             "dtype": "float64", "rank": 0, "python_type": "builtins.float"},
            {"function": "root", "parameter": "distribution", "storage": "scalar",
             "dtype": "int64", "rank": 0, "python_type": "builtins.NoneType"},
        ],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi)

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="targets_expansion", extraction_contract=policy,
        )
        print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    except Exception as error:
        print(f"LOWERING FAILED after {time.time()-t0:.2f}s: "
              f"{type(error).__name__}: {str(error)[:800]}", flush=True)
        return 1

    entry = next(name for name in module.functions if name.endswith("__root"))
    function = module.functions[entry]
    field_lookup: dict[tuple, object] = {}
    for value in function.args:
        accounting = value.accounting or {}
        field = accounting.get("program_abi_field")
        if field is not None:
            field_lookup[(accounting.get("program_abi_record"), field)] = value

    metrics = Metrics(max_vel=2.0, max_flux=2.0, div_inf=0.1, mass_err=0.05,
                       hard_failure=0.0)
    targets_values = {"cfl": 0.3, "div_max": 1.0, "mass_max": 1.0}
    dx = 0.02

    arguments: dict[int, object] = {}
    for name, value in (("max_vel", metrics.max_vel), ("max_flux", metrics.max_flux),
                        ("div_inf", metrics.div_inf), ("mass_err", metrics.mass_err),
                        ("hard_failure", float(metrics.hard_failure))):
        target = field_lookup.get(("Metrics", name))
        if target is not None:
            arguments[int(target.id)] = value
    for name, value in targets_values.items():
        target = field_lookup.get(("Targets", name))
        if target is not None:
            arguments[int(target.id)] = value
    error_limits_length = field_lookup.get(("Targets", "error_limits.length"))
    if error_limits_length is not None:
        arguments[int(error_limits_length.id)] = 0
    error_limits_keys = field_lookup.get(("Targets", "error_limits.keys"))
    if error_limits_keys is not None:
        arguments[int(error_limits_keys.id)] = []
    error_limits_values = field_lookup.get(("Targets", "error_limits.values"))
    if error_limits_values is not None:
        arguments[int(error_limits_values.id)] = []
    error_channels_length = field_lookup.get(("Metrics", "error_channels.length"))
    if error_channels_length is not None:
        arguments[int(error_channels_length.id)] = 0
    error_channels_keys = field_lookup.get(("Metrics", "error_channels.keys"))
    if error_channels_keys is not None:
        arguments[int(error_channels_keys.id)] = []
    error_channels_values = field_lookup.get(("Metrics", "error_channels.values"))
    if error_channels_values is not None:
        arguments[int(error_channels_values.id)] = []
    dx_value = next((v for v in function.args
                      if (v.accounting or {}).get("program_abi_field") is None
                      and v.dtype == "float64"), None)
    if dx_value is not None:
        arguments[int(dx_value.id)] = dx

    evaluator = SSAReferenceEvaluator(module)
    try:
        result = evaluator.run(entry, arguments)
    except Exception as error:
        print(f"EXECUTION FAILED: {type(error).__name__}: {str(error)[:800]}",
              flush=True)
        return 1

    got = result.returned[0] if result.returned else None
    coerced = coerce_metrics(metrics)
    targets_obj = type("T", (), targets_values | {"error_limits": {}})()
    expected = float(coerced.hard_failure) + _propose_dt_pen(
        coerced, targets_obj, dx, None,
    )
    ok = got is not None and abs(float(got) - float(expected)) < 1e-9
    print(f"got={got} expected={expected} {'OK' if ok else 'MISMATCH'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

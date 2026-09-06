"""Record identities used as physical operands by row effects.

Isolates the DT blocker from ``step_with_dt_control_used`` (dt_controller.py):
``advance`` returns a whole ``Metrics`` record (the record id has no physical
definition, by design; the call publishes its member fields), the caller then

* appends ``(float(dt), metrics, tuple(reasons))`` to ``failures`` inside the
  retry loop (line 449), and
* passes the record to a callee that rewrites its keyed ``error_channels``
  field (the ``dt_unresolved`` write-back at lines 407-410 / coerce_metrics).

Lowered with the real Metrics/Targets ABI and the full-native execution
contract.  The lowered caller must have no operand without a definition (a
record identity must resolve to its published member formals or keyed part
slots), and every sequence effect must sit inside the loop that owns it,
never in ``entry`` ahead of the advance call.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

SOURCE = '''
def advance(metrics, dt):
    return True, Metrics(
        max_vel=float(metrics.max_vel),
        max_flux=float(metrics.max_flux),
        div_inf=0.0,
        mass_err=float(metrics.mass_err) + float(dt),
        error_channels={
            "maximum_substep_displacement_m": float(dt),
            "energy_j": float(metrics.max_flux),
        },
    )


def tag_unresolved(metrics, dt, attempts):
    metrics.error_channels["dt_unresolved"] = float(dt)
    metrics.error_channels["dt_unresolved_attempts"] = float(attempts)
    return metrics


def root(metrics, targets, dt,
         failures: list[tuple[float, Metrics, tuple[str, ...]]] | None = None):
    if failures is None:
        failures = []
    x = dt
    while True:
        ok, m = advance(metrics, x)
        reasons = []
        if m.mass_err > targets.mass_max:
            reasons.append("mass_err")
        if not ok:
            reasons.append("advance reported a physical-bound violation")
        if reasons:
            m = tag_unresolved(m, x, len(failures) + 1)
            failures.append((float(x), m, tuple(reasons)))
            x = x * 0.5
            continue
        break
    return x, len(failures)
'''


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


def _contract():
    base = _base_records()
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": base["records"]["Targets"],
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
        "values": [
            {
                "function": "root", "parameter": "dt", "storage": "scalar",
                "dtype": "float64", "rank": 0, "python_type": "builtins.float",
            },
        ],
    }).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")


def main() -> int:
    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            SOURCE, "root", name="record_row_effects",
            python_bindings={"Metrics": Metrics},
            extraction_contract=_contract(),
        )
    except Exception as error:  # noqa: BLE001
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:2500]}", flush=True)
        return 1
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    gate = (getattr(module, "metadata", {}) or {}).get("full_native_link_gate")
    print(f"full_native_link_gate={gate}")
    ok = True
    for name, function in (getattr(module, "functions", {}) or {}).items():
        if "root" not in name or "__planned_region_" in name:
            continue
        defined = {int(a.id) for a in function.args}
        defined.update(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        undefined = sorted({
            int(a.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for a in instruction.args
            if int(a.id) not in defined
        })
        effects = [
            (block_name, instruction.op,
             (instruction.attributes or {}).get("binding"),
             [int(a.id) for a in instruction.args],
             (instruction.attributes or {}).get("source_effect_node_id"))
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if str((instruction.attributes or {}).get("binding", "")).startswith(
                "ssa_sequence_"
            ) or (instruction.attributes or {}).get("ssa_sequence_operation")
        ]
        print(f"function {name}")
        print(f"  blocks={list(function.blocks)}")
        print(f"  undefined_operands={undefined}")
        for effect in effects:
            print(f"  effect {effect}")
        if undefined:
            ok = False
            for block_name, block in function.blocks.items():
                for instruction in block.instrs:
                    if any(int(a.id) in undefined for a in instruction.args):
                        attributes = {
                            k: v for k, v in (instruction.attributes or {}).items()
                            if k in ("binding", "callee", "source_effect_node_id",
                                     "sequence_id", "plan_callsite_id", "callsite_id")
                        }
                        print(f"  uses-undefined {block_name}: {instruction.op} "
                              f"{[int(a.id) for a in instruction.args]} {attributes}")
        in_entry = [effect for effect in effects if effect[0] == "entry"]
        if in_entry:
            ok = False
            print(f"  !! sequence effects placed in entry: {in_entry}")
    if gate is not None and not gate.get("complete", True):
        ok = False
    print("OK" if ok else "CHECK FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Control-aware result merging for a ``return`` inside a TOP-LEVEL ``if``.

Real function: ``_propose_dt_pen`` (dt_controller.py) has
``if distribution is not None: return distribution(...)`` followed by the
ordinary fall-through return. Lowered with the real Metrics/Targets ABI
(same harness as tools/repro_metrics_rebind.py), then the lowered
``_propose_dt_pen`` must show exactly one ``Ret`` in ``function_exit``
reading a ``return_merge`` Phi with one incoming per return site, and no
SSA result id defined twice.
"""

from __future__ import annotations

import inspect
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_controller import _propose_dt_pen, _energy_time_limit  # noqa: E402
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
    source = "\n\n".join((
        inspect.getsource(_scalar),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_propose_dt_pen),
        "def root(metrics, targets, dx, distribution):\n"
        "    return _propose_dt_pen(metrics, targets, dx, distribution)\n",
    ))
    base = _base_records()
    policy = ExtractionContract(
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
        "values": [],
    })
    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name="return_merge_toplevel",
            extraction_contract=policy,
        )
    except Exception as error:  # noqa: BLE001
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:1500]}", flush=True)
        return 1
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    ok = True
    for name, function in (getattr(module, "functions", {}) or {}).items():
        if "_propose_dt_pen" not in name or "__planned_region_" in name:
            continue
        rets = [
            (block_name, [int(a.id) for a in instruction.args])
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret"}
        ]
        merges = [
            (block_name, [int(a.id) for a in instruction.args],
             int(instruction.res.id),
             (instruction.attributes or {}).get("incoming_blocks"))
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
            if instruction.op in {"Phi", "phi"}
            and (instruction.attributes or {}).get("binding") == "return_merge"
        ]
        counts = Counter(
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )
        duplicates = sorted((k, v) for k, v in counts.items() if v > 1)
        print(f"function {name}")
        print(f"  blocks={list(function.blocks)}")
        print(f"  rets={rets}")
        for merge in merges:
            print(f"  return_merge={merge}")
        print(f"  duplicate_result_ids={duplicates}")
        for duplicate_id, _count in duplicates:
            for block_name, block in function.blocks.items():
                for index, instruction in enumerate(block.instrs):
                    if instruction.res is not None and int(instruction.res.id) == duplicate_id:
                        attributes = {
                            key: value for key, value in (instruction.attributes or {}).items()
                            if key in ("binding", "incoming_blocks", "callee", "structural_operation", "semantic_family", "source_control", "initial_value_id")
                        }
                        print(f"  producer[{duplicate_id}] {block_name}: {instruction.op} "
                              f"{[int(a.id) for a in instruction.args]} {attributes}")
                        for previous in block.instrs[max(0, index - 3):index]:
                            print(f"      before: {previous.op} {[int(a.id) for a in previous.args]} "
                                  f"-> {None if previous.res is None else int(previous.res.id)} "
                                  f"{(previous.attributes or {}).get('callee', '')}")
    for name, function in (getattr(module, "functions", {}) or {}).items():
        if "_propose_dt_pen" in name and "__planned_region_" in name:
            region_rets = [
                [int(a.id) for a in instruction.args]
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Ret", "ret"}
            ]
            print(f"  region {name[-40:]}: outputs={region_rets} args={[int(a.id) for a in function.args]}")
    if len(rets) != 1 or rets[0][0] != "function_exit" or not merges or duplicates:
        ok = False
    print("OK" if ok else "CHECK FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

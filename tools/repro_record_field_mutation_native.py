"""A record's array field written two calls deep must change the ROOT buffer.

Native parity of the managed DT binary (build/managed_dt_record_rows9_20260906)
runs to completion but every write through the material record --
``material.telemetry[1] = material.telemetry[1] + 1.0`` inside
``balloon_tire_managed_advance`` called from ``step_with_dt_control_used``
called from ``run_superstep`` called from the window -- is invisible at the
root buffers, and even the returned ``dt_next`` is 0.  This is the same shape
with the real ``BalloonTireManagedState`` ABI: ``root -> middle -> deep``,
``deep`` writes two telemetry cells, ``middle`` returns a scalar derived from
its argument, ``root`` returns it.

Lowers to repository SSA, emits and compiles the standalone C, runs it in a
bounded subprocess (as the counter test does) and checks that the root
telemetry buffer holds the writes and the returned scalar is right.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.ssa_c_backend import emit_ssa_module_to_c  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

SOURCE = '''
def deep(material, dt):
    material.telemetry[1] = material.telemetry[1] + 1.0
    material.telemetry[7] = dt
    return True


def middle(material, dt):
    ok = deep(material, dt)
    if not ok:
        return 0.0
    return dt * 0.5


def root(material, dt):
    dt_next = middle(material, dt)
    return dt_next
'''

PROBE = r'''
import json, pickle, sys
import numpy as np
with open(sys.argv[1], "rb") as stream:
    artifact, feeds, telemetry_id, result_id = pickle.load(stream)
native = artifact.prepare_execution(feeds).run()
telemetry = native.buffers[telemetry_id]
keys = sorted(int(k) for k in native.buffers)
result = native.buffers.get(result_id)
if result is None:
    # The root output may be published under the artifact's own buffer id.
    candidates = [k for k in keys if k not in feeds]
    result = native.buffers[candidates[0]] if len(candidates) == 1 else None
print(json.dumps({
    "telemetry_1": float(np.asarray(telemetry).reshape(-1)[1]),
    "telemetry_7": float(np.asarray(telemetry).reshape(-1)[7]),
    "result": None if result is None else float(np.asarray(result).reshape(-1)[0]),
    "buffer_keys": keys, "buffer_order": [int(v) for v in artifact.buffer_order],
}))
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
    stub.telemetry = np.zeros((20,), dtype=np.float64)
    return balloon_tire_managed_extraction_contract(stub).program_abi.receipt()


def _contract():
    base = _base_records()
    return ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": {
            "BalloonTireManagedState": base["records"]["BalloonTireManagedState"],
        },
        "bindings": [
            {"function": "*", "parameter": "material",
             "record": "BalloonTireManagedState"},
        ],
        "values": [
            {
                "function": name, "parameter": "dt", "storage": "scalar",
                "dtype": "float64", "rank": 0, "python_type": "builtins.float",
            }
            for name in ("root", "middle", "deep")
        ],
    }).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")


def main() -> int:
    import numpy as np

    t0 = time.time()
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            SOURCE, "root", name="record_field_mutation",
            extraction_contract=_contract(),
        )
    except Exception as error:  # noqa: BLE001
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:2500]}", flush=True)
        return 1
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    entry = exports[0]
    function = module.functions[entry]
    telemetry_id = next(
        int(a.id) for a in function.args
        if (a.accounting or {}).get("program_abi_field") == "telemetry"
    )
    dt_id = next(
        int(a.id) for a in function.args
        if (a.accounting or {}).get("program_abi_field") is None
        and str(a.dtype) == "float64" and not tuple(a.shape or ())
    )
    result_id = int(outputs[entry][0].id)
    print(f"  entry={entry} args={[(int(a.id), (a.accounting or {}).get('program_abi_field'), a.dtype, a.shape) for a in function.args]}")
    print(f"  telemetry_id={telemetry_id} dt_id={dt_id} result_id={result_id}")
    artifact = emit_ssa_module_to_c(module, entry)
    if not artifact.complete:
        print("C EMISSION INCOMPLETE:", [(i.operation, i.reason) for i in artifact.shortfalls][:6])
        return 2
    work = Path(tempfile.mkdtemp(prefix="record_field_mutation_"))
    artifact.compile(work / "record_field_mutation")
    feeds = {
        int(a.id): (
            np.zeros((20,), dtype=np.float64)
            if int(a.id) == telemetry_id
            else np.array([0.25], dtype=np.float64) if int(a.id) == dt_id
            else np.zeros(tuple(a.shape or (1,)), dtype=np.float64)
        )
        for a in function.args
    }
    saved = work / "artifact.pkl"
    saved.write_bytes(pickle.dumps((artifact, feeds, telemetry_id, result_id)))
    probe = subprocess.run(
        [sys.executable, "-c", PROBE, str(saved)],
        capture_output=True, text=True, timeout=20,
    )
    if probe.returncode != 0:
        print("NATIVE RUN FAILED:", probe.stderr[-1500:])
        return 3
    observed = json.loads(probe.stdout.splitlines()[-1])
    expected = {"telemetry_1": 1.0, "telemetry_7": 0.25, "result": 0.125}
    print(f"  native={observed} expected={expected}")
    ok = all(
        observed.get(k) is not None and abs(observed[k] - v) < 1e-12
        for k, v in expected.items()
    )
    print("OK" if ok else "CHECK FAILED")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())

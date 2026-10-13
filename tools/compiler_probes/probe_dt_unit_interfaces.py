"""Small authored DT helper slice through the public parallel compiler entry."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if "--baseline" in sys.argv:
    if "--serial" not in sys.argv:
        raise ValueError("baseline comparison requires the serial control path")
    sys.path.insert(0, "C:/dev/tg-base")
    import src.compiler.fortran_c_shell as baseline_compiler
    print("BASELINE_COMPILER=" + baseline_compiler.__file__, flush=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


def main(directory, *, proposal=False, serial=False):
    from llvm_dt_system import dt_system_contract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import concordance_report
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference
    source = (
        "from src.common.dt_system.dt_controller import _energy_time_limit, _no_exchange_observed\n"
        "def root(metrics, targets):\n"
        "    limit, present = _energy_time_limit(metrics, targets)\n"
        "    hold = _no_exchange_observed(metrics, targets)\n"
        "    return limit + present + float(hold)\n"
    )
    if proposal:
        source = (
            "from src.common.dt_system.dt_controller import _propose_dt_pen\n"
            "def root(metrics, targets):\n"
            "    return _propose_dt_pen(metrics, targets, 1.0, None, 0.5)\n"
        )
    started = time.perf_counter()
    unit_options = {} if serial else {
        "partitioned_directory": Path(directory),
        "partitioned_options": {"jobs": 2, "max_unit_source_nodes": 256,
            "worker_resident_reservation_bytes": 512 * 1024 ** 2,
            "max_worker_memory_bytes": 2 * 1024 ** 3},
    }
    module, outputs, exports = lower_ast_source_to_ssa(
        source, "root", extraction_contract=dt_system_contract("root", (), 1, 1),
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        **unit_options,
        progress=lambda message: print(message, flush=True),
    )
    print(concordance_report(module), flush=True)
    import dataclasses
    import subprocess
    from src.compiler.project_compilation_product import _dump_pickle_artifact
    from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
    artifact = emit_ssa_function_to_llvm(module, exports[0], entry_name="dt_unit_slice")
    if not artifact.complete:
        raise RuntimeError(artifact.shortfalls)
    artifact = compile_artifact(artifact, directory=Path(directory) / "native")
    artifact = dataclasses.replace(artifact, emission=None, _entry=None, _library=None,
        _validation_error_reset=None, _validation_error_take=None)
    payload = Path(directory) / "native-check.pkl"
    cases = [({}, [0.0])]
    if proposal:
        import numpy as np
        root = module.functions[exports[0]]
        records = module.record_tables[root.name].records
        cases = []
        # dt_current=.5, dx=1.0. All errors absent: min(CFL=1, .5*1.8)=.9.
        # With velocity=2 and CFL=.5, CFL=.25 wins over the penalty proposal.
        for velocity, cfl, expected in ((1.0, 1.0, [0.9]), (2.0, 0.5, [0.25])):
            feeds = {}
            values = {int(value.id): value for value in root.args}
            for record in records.values():
                for field in record.fields:
                    scalar = {"max_vel": velocity, "cfl": cfl, "div_max": 1.0, "mass_max": 1.0,
                              "error_limits": 1.0, "pub_limits": 1.0}.get(field.name, 0.0)
                    for value_id in field.value_ids:
                        value = values[int(value_id)]
                        feeds[int(value_id)] = np.full(tuple(value.shape or ()), scalar, dtype=np.float64)
            cases.append((feeds, expected))
    else:
        # Declared pub_exchange_time extent proves both helper early returns.
        assert not module.functions[exports[0]].args
    import json
    for feeds, expected in cases:
        _dump_pickle_artifact(payload, (artifact, {}, ("program_outputs",), {
            "inputs": feeds, "outputs": tuple(value.id for value in outputs[exports[0]]),
        }))
        child = subprocess.run([sys.executable, "-u", str(ROOT / "tools/compiler_probes/graph_kernel_native_worker.py"),
                                str(payload)], text=True, capture_output=True, timeout=10)
        print(child.stdout, end="", flush=True)
        if child.returncode:
            raise RuntimeError(child.stderr or child.stdout)
        actual = json.loads(next(line.split("=", 1)[1] for line in child.stdout.splitlines()
                                 if line.startswith("GRAPH_PROGRAM_OUTPUTS=")))
        if actual != expected:
            raise RuntimeError(f"native DT proposal disagreement: {actual!r} != {expected!r}")
    print({"seconds": time.perf_counter() - started, "exports": exports,
           "outputs": {name: [value.id for value in values] for name, values in outputs.items()},
           "counts": module.metadata.get("partitioned_source_product", {}).get("counts")}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], proposal="--proposal" in sys.argv, serial="--serial" in sys.argv)

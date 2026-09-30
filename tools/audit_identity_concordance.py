"""Run the identity concordance audit over small, real lowerings (seconds).

Cases (any subset as arguments; default all):

  view      a record with one contiguous ``memory`` span whose fields are
            static slice views, snapshot/restore as whole-span copies
            (the PieceState view model probe)
  toplevel  the real ``_propose_dt_pen`` with its top-level early return,
            under the real Metrics/Targets ABI (repro_return_merge_toplevel)
  energy    the real ``_energy_time_limit`` with its shape-15 Metrics ABI and
            repository tensor kernels
  controller the real ``STController`` update/PI methods through one record
             receiver and repository tensor kernels
  controller_untyped the same controller path without root scalar Python-type
             declarations, matching callers whose identities must arrive
             through whole-program callsite concordance
  mapping   ``for name, limit in channels.items()`` over a bare keyed
            mapping parameter (repro_loop_dominance)
  oscillator a Van der Pol oscillator stepped by Runge-Kutta 4 in a
            data-bounded ``while``, with a Heron square-root loop, a cosine
            Taylor series over ``range`` and a running peak behind an ``if``:
            three helpers, nested loops, branches, all scalar

``--pickle PATH`` audits a pickled SSA module instead (what
``TURING_REPRO_SSA=... tools/repro_step_with_dt_control_used.py`` writes).

Each case prints the concordance report: rows counted, then every finding
kind with its entries.  A clean lowering prints zero findings; a finding is
one concrete disagreement between two of the compiler's own records about
one value, reported where it is written rather than where it later fails.
"""

from __future__ import annotations

import inspect
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.identity_concordance import concordance_report  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


def _lower_view():
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    batch = 4
    source = f'''
class Cell:
    def __init__(self, memory, telemetry):
        self.memory = memory
        self.telemetry = telemetry

    def copy_shallow(self):
        return self.memory.copy()

    def restore(self, snapshot):
        self.memory[...] = snapshot


def law(x, v, dt):
    return x + dt * v


def advance(state, dt):
    x = state.memory[0:{batch}]
    v = state.memory[{batch}:{2 * batch}]
    x_next = law(x, v, dt)
    state.memory[0:{batch}] = x_next
    return float(x_next.max())


def tick(state, dt):
    saved = state.copy_shallow()
    peak = advance(state, dt)
    state.telemetry[0] = peak
    state.telemetry[1] = float(state.memory[0:{batch}].max())
    state.restore(saved)
    return peak
'''
    span = lambda length: {  # noqa: E731
        "storage": "span", "dtype": "float64", "rank": 1,
        "shape": [int(length)], "mutable": True,
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": {"Cell": {
            "identity": "concordance_view.Cell",
            "fields": {"memory": span(2 * batch), "telemetry": span(2)},
        }},
        "bindings": [{"function": "*", "parameter": "state", "record": "Cell"}],
        "values": [
            {"function": "tick", "parameter": "dt", "storage": "scalar",
             "dtype": "float64", "rank": 0, "python_type": "builtins.float"},
            {"function": "restore", "parameter": "snapshot", "storage": "span",
             "dtype": "float64", "rank": 1, "shape": [2 * batch],
             "python_type": "src.common.tensors.abstraction.AbstractTensor"},
        ],
    }).with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "tick", python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        name="concordance_view", runtime_closure_only=True,
        extraction_contract=policy,
    )
    return module


def _lower_toplevel():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.vehicle_python_compilation import (
        balloon_tire_managed_extraction_contract, BalloonTireManagedState,
    )
    from src.common.dt_system.dt_controller import (
        _propose_dt_pen, _energy_time_limit,
    )
    from src.common.dt_system.dt_scaler import _scalar
    import numpy as np

    stub = BalloonTireManagedState.__new__(BalloonTireManagedState)
    for name in (
        "inputs", "state", "output", "wheel_input_indices", "rest",
        "face_vertices", "face_rest", "face_scatter", "bending_incidence",
        "bending_scatter", "bending_weight", "vertex_area", "bead_mask",
        "face_material", "telemetry",
    ):
        setattr(stub, name, np.zeros((1,), dtype=np.float64))
    base = balloon_tire_managed_extraction_contract(stub).program_abi.receipt()
    source = "\n\n".join((
        inspect.getsource(_scalar),
        inspect.getsource(_energy_time_limit),
        inspect.getsource(_propose_dt_pen),
        "def root(metrics, targets, dx, distribution):\n"
        "    return _propose_dt_pen(metrics, targets, dx, distribution)\n",
    ))
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
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="concordance_toplevel", extraction_contract=policy,
    )
    return module


def _lower_energy():
    from src.common.dt_system.dt_controller import _energy_time_limit
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = "\n\n".join((
        inspect.getsource(_energy_time_limit),
        "def root(metrics, targets):\n"
        "    return _energy_time_limit(metrics, targets)\n",
    ))
    base = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).program_abi.receipt()
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
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="concordance_energy",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=policy,
    )
    return module


def _lower_controller(*, declare_root_types: bool = True):
    from src.common.dt_system.dt_controller import STController, _restore_type
    from src.common.tensors import AbstractTensor
    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = "\n\n".join((
        inspect.getsource(_restore_type),
        inspect.getsource(STController),
        "def root(ctrl, max_vel, dx, dt_prev, dt_pen, osc):\n"
        "    ctrl.update_dt_max(max_vel, dx)\n"
        "    return ctrl.pi_update(dt_prev, dt_pen, osc)\n",
    ))
    base = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).program_abi.receipt()
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi({
        "records": {"STController": base["records"]["STController"]},
        "bindings": [
            {"function": "*", "parameter": "ctrl", "record": "STController"},
        ],
        "values": ([
            {
                "function": "root", "parameter": name,
                "storage": "scalar", "dtype": dtype, "rank": 0,
                "python_type": python_type,
            }
            for name, dtype, python_type in (
                ("max_vel", "float64", "builtins.float"),
                ("dx", "float64", "builtins.float"),
                ("dt_prev", "float64", "builtins.float"),
                ("dt_pen", "float64", "builtins.float"),
                ("osc", "bool", "builtins.bool"),
            )
        ] if declare_root_types else []),
    })
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="concordance_controller",
        python_bindings={"AbstractTensor": AbstractTensor},
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        extraction_contract=policy,
    )
    return module


def _lower_controller_untyped():
    return _lower_controller(declare_root_types=False)


def _lower_mapping():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def root(channels):\n"
        "    total = 0.0\n"
        "    for name, limit in channels.items():\n"
        "        total = total + limit\n"
        "    return total\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="concordance_mapping",
        extraction_contract=CONTRACTS / "program_extraction.yaml",
    )
    return module


def _lower_oscillator():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def accel(mu, x, v):\n"
        "    return mu * (1.0 - x * x) * v - x\n"
        "\n"
        "def heron_sqrt(value):\n"
        "    guess = value * 0.5 + 0.5\n"
        "    error = guess * guess - value\n"
        "    if error < 0.0:\n"
        "        error = 0.0 - error\n"
        "    while error > 1e-12:\n"
        "        guess = 0.5 * (guess + value / guess)\n"
        "        error = guess * guess - value\n"
        "        if error < 0.0:\n"
        "            error = 0.0 - error\n"
        "    return guess\n"
        "\n"
        "def cos_series(x):\n"
        "    term = 1.0\n"
        "    total = 1.0\n"
        "    for k in range(6):\n"
        "        term = 0.0 - term * x * x / ((2.0 * k + 1.0) * (2.0 * k + 2.0))\n"
        "        total = total + term\n"
        "    return total\n"
        "\n"
        "def root(mu, x0, v0, dt, steps):\n"
        "    x = x0\n"
        "    v = v0\n"
        "    energy = 0.0\n"
        "    peak = 0.0\n"
        "    turns = 0\n"
        "    while turns < steps:\n"
        "        a1 = accel(mu, x, v)\n"
        "        x2 = x + 0.5 * dt * v\n"
        "        v2 = v + 0.5 * dt * a1\n"
        "        a2 = accel(mu, x2, v2)\n"
        "        x3 = x + 0.5 * dt * v2\n"
        "        v3 = v + 0.5 * dt * a2\n"
        "        a3 = accel(mu, x3, v3)\n"
        "        x4 = x + dt * v3\n"
        "        v4 = v + dt * a3\n"
        "        a4 = accel(mu, x4, v4)\n"
        "        x = x + dt * (v + 2.0 * v2 + 2.0 * v3 + v4) / 6.0\n"
        "        v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0\n"
        "        radius = heron_sqrt(x * x + v * v)\n"
        "        if radius > peak:\n"
        "            peak = radius\n"
        "        energy = energy + 0.5 * (x * x + v * v) * dt\n"
        "        turns = turns + 1\n"
        "    return energy * cos_series(peak) + peak\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="concordance_oscillator",
        extraction_contract=CONTRACTS / "program_extraction.yaml",
    )
    return module


CASES = {
    "view": _lower_view,
    "toplevel": _lower_toplevel,
    "energy": _lower_energy,
    "controller": _lower_controller,
    "controller_untyped": _lower_controller_untyped,
    "mapping": _lower_mapping,
    "oscillator": _lower_oscillator,
}


def main() -> int:
    arguments = sys.argv[1:]
    if arguments[:1] == ["--pickle"]:
        module = pickle.loads(Path(arguments[1]).read_bytes())
        print(concordance_report(module), flush=True)
        return 0
    wanted = arguments or list(CASES)
    failures = 0
    for case in wanted:
        print(f"===== {case} =====", flush=True)
        started = time.perf_counter()
        try:
            module = CASES[case]()
        except Exception as error:  # noqa: BLE001 -- report, keep going
            print(f"  LOWERING FAILED after {time.perf_counter()-started:.1f}s: "
                  f"{type(error).__name__}: {str(error)[:300]}", flush=True)
            failures += 1
            continue
        report = concordance_report(module)
        print(f"  lowered in {time.perf_counter()-started:.1f}s", flush=True)
        print("  " + report.replace("\n", "\n  "), flush=True)
        if "finding(s)" in report.splitlines()[0] and not report.splitlines()[0].endswith(" 0 finding(s)"):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

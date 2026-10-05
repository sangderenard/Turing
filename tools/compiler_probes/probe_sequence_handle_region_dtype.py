"""A record's table field handle produced inside a planned region keeps its
declared column dtype.

Shape of ``dt_controller.step_with_dt_control_used``: a call-result record
whose ``unresolved_report`` field (``storage: table, columns: [token:
int64]``) is written under a guard inside a retry loop.  The reducer seeds the
field's pre-branch state as a GetAttr stamped ``sequence_column_dtypes =
('int64',)`` and the planner places that GetAttr in a numerical region; the
region's output IS the sequence handle (the arena).

The planner (``hierarchical_plan.plan_region_to_ssa_instrs``) used to ignore
the declared row contract and type the handle ``float64`` (its default), so
the caller's region projection overrode the int64 arena the control builder
had declared from the same contract while every callee formal for that arena
stayed int64: "10 incompatible final physical call inputs; storage types are
immutable" on the N=2 orbital dt system.

Green when every lowered instruction that carries ``sequence_column_dtypes``
produces a value typed by its column-0 dtype, and the program lowers through
the full-native gate.

    python -u tools/compiler_probes/probe_sequence_handle_region_dtype.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from repro_record_row_effects import _contract  # noqa: E402  real Metrics ABI
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402
from src.compiler import hierarchical_plan  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

SOURCE = '''
from src.common.tensors import AbstractTensor


def advance(metrics, dt):
    return True, Metrics(
        max_vel=float(metrics.max_vel),
        max_flux=float(metrics.max_flux),
        div_inf=0.0,
        mass_err=float(metrics.mass_err) + float(dt),
        error_channels=AbstractTensor.tensor([float(metrics.max_flux), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(dt), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        error_present=AbstractTensor.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def coerce_metrics(value) -> Metrics:
    return value


def propose(m, targets, x):
    return x * 0.5 if m.mass_err > targets.mass_max else x


def step(metrics, targets, dt, allow_unresolved: bool = True):
    x = dt
    while True:
        ok, m = advance(metrics, x)
        m = coerce_metrics(m)
        rejected = m.mass_err > targets.mass_max
        if bool(m.hard_failure):
            rejected = True
        if rejected and allow_unresolved:
            lines = ["timestep controller proceeded unresolved"]
            lines.append("  attempt rejected")
            m.unresolved_report = list(lines)
            m.hard_failure = True
            rejected = False
        if rejected:
            x = x * 0.5
            continue
        y = propose(m, targets, x)
        return m, y


def root(metrics, targets, dt):
    m, x = step(metrics, targets, dt)
    return x + float(m.mass_err)
'''

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


# Read-only observation of the planner's region expansion: every PlanLine
# that declares a row contract (``sequence_column_dtypes``) and the dtype of
# the repository SSA value the expansion emits for it.  The wrapper calls
# straight through; nothing is patched.
handles: list[tuple[str, str, str, int, str, str]] = []
_original_expand = hierarchical_plan.plan_region_to_ssa_instrs


def _observing_expand(region, **kwargs):
    instructions = _original_expand(region, **kwargs)
    for item in region.items:
        attributes = dict(getattr(item, "attributes", ()) or ())
        columns = tuple(attributes.get("sequence_column_dtypes") or ())
        outputs = tuple(map(int, getattr(item, "outputs", ()) or ()))
        if not columns or not outputs:
            continue
        for instruction in instructions:
            if instruction.res is not None and int(instruction.res.id) == outputs[0]:
                handles.append((
                    str(kwargs.get("function_scope")), str(region.name),
                    str(getattr(item, "opcode", "")), outputs[0],
                    str(instruction.res.dtype), str(columns[0]),
                ))
    return instructions


hierarchical_plan.plan_region_to_ssa_instrs = _observing_expand


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            module, _outputs, _exports = lower_ast_source_to_ssa(
                SOURCE, "root", name="sequence_handle_region_dtype",
                python_bindings={"Metrics": Metrics},
                extraction_contract=_contract(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"lowering raised {type(exc).__name__}: {str(exc)[:600]}")
            check("the program lowers", False)
            return 1
    check("the program lowers", True)
    gate = (module.metadata or {}).get("full_native_link_gate") or {}
    check("the full-native gate is complete", bool(gate.get("complete")))
    print(f"planned lines declaring a row contract: {len(handles)}")
    for scope, region, opcode, value_id, dtype, column in handles:
        print(f"    {scope} {region}: {opcode} -> {value_id} "
              f"emitted dtype={dtype} declared column 0={column}")
    check("a planned region produces a declared sequence handle "
          "(the seeded unresolved_report GetAttr)", bool(handles))
    check("every such handle is typed by its declared column-0 dtype", all(
        dtype == column for *_rest, dtype, column in handles
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

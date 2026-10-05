"""A record literal's table field is typed by its declared column across a call.

Shape of ``llvm_dt_system.dt_system_over -> run_superstep``: the callee
builds ``Metrics(..., unresolved_report=[])`` (``unresolved_report`` is
``storage: table, columns: [token: int64]`` in the program ABI) and returns
it; the caller links the callee and must supply the literal's arena formal
(a ``default_literal`` frame binding, minted as a ``Const`` with the formal's
dtype at link time).

``materialize_program_abi_record_literals`` typed that arena from the
field's top-level ``dtype`` -- which a table field does not have -- leaving
the provisional float64 of the empty ``[]``; the late sequence-descriptor
reconcile corrected the formal to int64 only after the caller had copied
float64 into its Const: "incompatible final physical call inputs; storage
types are immutable" on the N=2 orbital dt system (the last of ten pairs).

Green when the program lowers through the full-native gate and, at every
call, the actual handed to a ``program_abi_field == 'unresolved_report'``
formal has the formal's dtype (the declared token dtype).

    python -u tools/compiler_probes/probe_record_literal_table_field_dtype.py
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
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

SOURCE = '''
def inner(targets, dt):
    last = Metrics(0.0, 0.0, 0.0, 0.0, hard_failure=True, unresolved_report=[])
    total = 0.0
    while total < dt:
        total = total + dt * 0.5
        last.mass_err = total
    return total, last


def root(targets, dt):
    advanced, metrics = inner(targets, dt)
    return advanced + float(metrics.mass_err) + float(metrics.hard_failure)
'''

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            module, _outputs, _exports = lower_ast_source_to_ssa(
                SOURCE, "root", name="record_literal_table_field_dtype",
                python_bindings={"Metrics": Metrics},
                extraction_contract=_contract(),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"lowering raised {type(exc).__name__}: {str(exc)[:700]}")
            check("the program lowers", False)
            return 1
    check("the program lowers", True)
    gate = (module.metadata or {}).get("full_native_link_gate") or {}
    check("the full-native gate is complete", bool(gate.get("complete")))
    pairs = []
    for caller_name, caller in module.functions.items():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                callee = module.functions.get(
                    str((instruction.attributes or {}).get("callee", ""))
                )
                if (instruction.op not in {"Call", "call"} or callee is None
                        or len(instruction.args) != len(callee.args)):
                    continue
                for actual, formal in zip(instruction.args, callee.args):
                    field = (formal.accounting or {}).get("program_abi_field")
                    if field is None or not str(field).startswith("unresolved_report"):
                        continue
                    pairs.append((str(caller_name), str(callee.name), str(field),
                                  int(actual.id), str(actual.dtype),
                                  int(formal.id), str(formal.dtype)))
    print(f"call inputs feeding an unresolved_report formal: {len(pairs)}")
    for caller, callee, field, actual_id, actual_dtype, formal_id, formal_dtype in pairs:
        print(f"    {caller} -> {callee} {field}: actual {actual_id} {actual_dtype} "
              f"/ formal {formal_id} {formal_dtype}")
    check("the literal's table-field arena crosses at least one call", bool(pairs))
    check("every such actual has its formal's dtype", all(
        actual_dtype == formal_dtype
        for *_rest, actual_dtype, _formal_id, formal_dtype in pairs
    ))
    check("every such formal is typed by the declared token column (int64)", all(
        formal_dtype == "int64" for *_rest, formal_dtype in pairs
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

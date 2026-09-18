"""Fast, real repro for the `ssa_deferred_record_row` undefined-operand defect.

The full managed-tire compile rejects run_superstep's
`unresolved.append(metrics)` (dt_controller.py:644) with an undefined
deferred-row operand: the append Call still carries
``ssa_deferred_record_row=(239, 'Metrics', 14)`` and its final operand 239
is defined by nothing.  There, ``metrics`` is the record-valued first
element of the 3-tuple returned by step_with_dt_control_used(...) at
dt_controller.py:627, inside a ``while`` loop, appended under an ``if``.

This script isolates the sequence-row half of that shape with the REAL
coerce_metrics/_scalar (inspect.getsource) and the real Metrics record from
the program ABI, and asks one question per variant: does appending a
call-produced record to a ``list[Metrics]`` leave an unresolved deferred row
on its own, or only when the producing call is a multi-result tuple?

Variants (pass the name as argv[1]; each runs in its own process):
  flat        metrics = coerce_metrics(value); rows.append(metrics)         (control: the proven test shape)
  loop        same, inside `while`, appended under `if`                      (single result, loop + branch)
  tuple       metrics, a, b = pair(value); rows.append(metrics)              (multi-result, no loop)
  tuple_loop  same, inside `while`, appended under `if`                      (the run_superstep shape)
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import (  # noqa: E402
    lower_ast_source_to_ssa, _undefined_repository_ssa_operands,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system.dt_scaler import (  # noqa: E402
    Metrics, coerce_metrics, _scalar,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"

# A real multi-result producer whose first element is the coerced record --
# the exact result shape of step_with_dt_control_used (metrics, dt_next,
# dt_used) without its callbacks/state machinery.
PAIR_SOURCE = (
    "def pair(metrics, scale):\n"
    "    metrics = coerce_metrics(metrics)\n"
    "    dt_next = metrics.max_vel * scale\n"
    "    dt_used = dt_next * 0.5\n"
    "    return metrics, dt_next, dt_used\n"
)

ROOTS = {
    "flat": (
        "def root(metrics, scale):\n"
        "    rows: list[Metrics] = []\n"
        "    metrics = coerce_metrics(metrics)\n"
        "    rows.append(metrics)\n"
        "    return scale\n"
    ),
    "loop": (
        "def root(metrics, scale, attempts):\n"
        "    rows: list[Metrics] = []\n"
        "    count = 0\n"
        "    while count < attempts:\n"
        "        metrics = coerce_metrics(metrics)\n"
        "        if metrics.max_vel > 0.0:\n"
        "            rows.append(metrics)\n"
        "        count = count + 1\n"
        "    return scale\n"
    ),
    # No-rebind twins: the call result goes to a FRESH name.  Isolates the
    # `metrics = f(metrics)` same-name rebind from the loop/tuple factors.
    "flat_fresh": (
        "def root(metrics, scale):\n"
        "    rows: list[Metrics] = []\n"
        "    coerced = coerce_metrics(metrics)\n"
        "    rows.append(coerced)\n"
        "    return scale\n"
    ),
    "loop_fresh": (
        "def root(metrics, scale, attempts):\n"
        "    rows: list[Metrics] = []\n"
        "    count = 0\n"
        "    while count < attempts:\n"
        "        coerced = coerce_metrics(metrics)\n"
        "        if coerced.max_vel > 0.0:\n"
        "            rows.append(coerced)\n"
        "        count = count + 1\n"
        "    return scale\n"
    ),
    "tuple_loop_fresh": (
        "def root(metrics, scale, attempts):\n"
        "    rows: list[Metrics] = []\n"
        "    count = 0\n"
        "    total = 0.0\n"
        "    while count < attempts:\n"
        "        coerced, dt_next, dt_used = pair(metrics, scale)\n"
        "        if coerced.max_vel > 0.0:\n"
        "            rows.append(coerced)\n"
        "        total = total + dt_used\n"
        "        count = count + 1\n"
        "    return total\n"
    ),
    # Single-result twin whose callee does NOT take/return a record parameter:
    # pair() wrapped so the caller sees one record result built from a scalar.
    # Distinguishes "single-result call" from "callee returns its own
    # mutated record parameter" (coerce_metrics's `return value` branch).
    "flat_scalar_in": (
        "def root(metrics, scale):\n"
        "    rows: list[Metrics] = []\n"
        "    coerced = fresh(scale)\n"
        "    rows.append(coerced)\n"
        "    return scale\n"
    ),
    "tuple": (
        "def root(metrics, scale):\n"
        "    rows: list[Metrics] = []\n"
        "    metrics, dt_next, dt_used = pair(metrics, scale)\n"
        "    rows.append(metrics)\n"
        "    return dt_used\n"
    ),
    "tuple_loop": (
        "def root(metrics, scale, attempts):\n"
        "    rows: list[Metrics] = []\n"
        "    count = 0\n"
        "    total = 0.0\n"
        "    while count < attempts:\n"
        "        metrics, dt_next, dt_used = pair(metrics, scale)\n"
        "        if metrics.max_vel > 0.0:\n"
        "            rows.append(metrics)\n"
        "        total = total + dt_used\n"
        "        count = count + 1\n"
        "    return total\n"
    ),
}


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
    variant = sys.argv[1] if len(sys.argv) > 1 else "flat"
    root_source = ROOTS[variant]
    parts = [inspect.getsource(_scalar), inspect.getsource(coerce_metrics)]
    if variant.startswith("tuple"):
        parts.append(PAIR_SOURCE)
    if variant == "flat_scalar_in":
        parts.append(
            "def fresh(scale):\n"
            "    return Metrics(max_vel=scale, max_flux=scale, div_inf=scale, "
            "mass_err=scale, osc_flag=False, stiff_flag=False, sim_frame=0, "
            "proc_ms=scale, dt_limit=scale, error_channels={'residual': scale}, "
            "hard_failure=False, advanced_dt=scale)\n"
        )
    parts.append(root_source)
    source = "\n\n".join(parts)

    base = _base_records()
    contract_abi = {
        "records": {"Metrics": base["records"]["Metrics"]},
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "value", "record": "Metrics"},
        ],
        "values": [],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi)

    t0 = time.time()
    try:
        module, _outputs, _exports = lower_ast_source_to_ssa(
            source, "root", name=f"audit_row_{variant}",
            python_bindings={"Metrics": Metrics},
            extraction_contract=policy,
        )
    except Exception as error:
        print(f"[{variant}] FAILED after {time.time()-t0:.2f}s: "
              f"{type(error).__name__}: {str(error)[:1500]}", flush=True)
        return 1
    print(f"[{variant}] lowered in {time.time()-t0:.2f}s", flush=True)

    root = module.functions[f"audit_row_{variant}__root"]
    appends = [
        (block_name, instruction)
        for block_name, block in root.blocks.items()
        for instruction in block.instrs
        if instruction.attributes.get("ssa_sequence_operation") == "append"
    ]
    for block_name, append in appends:
        print(f"  append in block {block_name!r}: callee="
              f"{append.attributes.get('callee')!r} nargs={len(append.args)} "
              f"deferred={append.attributes.get('ssa_deferred_record_row')!r} "
              f"expanded_from="
              f"{append.attributes.get('ssa_record_row_expanded_from')!r} "
              f"last_operand={int(append.args[-1].id)} "
              f"last_accounting={dict(append.args[-1].accounting or {})!r}",
              flush=True)
    print(f"  unresolved_record_sequence_rows="
          f"{root.metadata.get('unresolved_record_sequence_rows')!r}",
          flush=True)
    aliases = dict(root.metadata.get("value_aliases") or {})
    print(f"  value_aliases={aliases!r}", flush=True)
    print(f"  value_names={tuple(root.metadata.get('value_names') or ())!r}",
          flush=True)
    definitions = {
        int(instruction.res.id): (block_name, instruction)
        for block_name, block in root.blocks.items()
        for instruction in block.instrs
        if instruction.res is not None
    }
    argument_ids = tuple(int(value.id) for value in root.args)
    print(f"  root.args={argument_ids!r}", flush=True)
    for _block_name, append in appends:
        for probe in dict.fromkeys((
            int(append.args[-1].id),
            *(int(append.attributes.get("ssa_deferred_record_row", (-1,))[0]),),
        )):
            if probe in definitions:
                block_name, instruction = definitions[probe]
                print(f"  value {probe} defined in {block_name!r} by "
                      f"{instruction.op} args="
                      f"{tuple(int(a.id) for a in instruction.args)!r} "
                      f"attrs={ {k: v for k, v in instruction.attributes.items() if k in ('binding', 'callee', 'source_output_id', 'updated_value_id', 'incoming_blocks', 'plan_callsite_marker_projection')} !r}",
                      flush=True)
            elif probe in argument_ids:
                print(f"  value {probe} is a root parameter", flush=True)
            else:
                print(f"  value {probe} has NO definition and is not a parameter",
                      flush=True)
        users = [
            (block_name, instruction.op,
             None if instruction.res is None else int(instruction.res.id),
             instruction.attributes.get("binding"))
            for block_name, block in root.blocks.items()
            for instruction in block.instrs
            if any(int(a.id) == int(append.args[-1].id) for a in instruction.args)
        ]
        print(f"  users of final operand {int(append.args[-1].id)}: {users!r}",
              flush=True)
    record_ids = tuple(module.record_tables[root.name].records)
    print(f"  root record descriptors: {record_ids!r}", flush=True)
    root_values = set(definitions) | set(argument_ids)
    for _block_name, append in appends:
        deferred = append.attributes.get("ssa_deferred_record_row")
        semantic_id = (
            int(deferred[0]) if deferred is not None
            else int(append.attributes.get("ssa_record_row_expanded_from", -1))
        )
        descriptor = module.record_tables[root.name].records.get(semantic_id)
        if descriptor is None:
            print(f"  descriptor for semantic {semantic_id}: NONE", flush=True)
            continue
        print(f"  descriptor for semantic {semantic_id}: identity="
              f"{descriptor.identity!r} fields="
              + repr(tuple(
                  (field.name, tuple(map(int, field.value_ids)),
                   all(int(v) in root_values for v in field.value_ids))
                  for field in descriptor.fields
              )), flush=True)
    # Callee-side facts: does the producing callee publish a physical record
    # return layout at all?
    for callee_name, callee in module.functions.items():
        if callee_name == root.name or callee_name.startswith("ssa_sequence"):
            continue
        rets = [
            instruction
            for block in callee.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ]
        layouts = dict(callee.metadata.get("record_return_layouts", ()))
        callee_formals = {int(value.id) for value in callee.args}
        if layouts:
            for output_id, layout in layouts.items():
                print(f"  callee {callee_name!r} layout {output_id}: "
                      f"{sum(1 for v in layout if int(v) in callee_formals)} of "
                      f"{len(layout)} layout ids are the callee's OWN INPUT FORMALS",
                      flush=True)
        print(f"  callee {callee_name!r}: ret_arity="
              f"{[len(r.args) for r in rets]!r} "
              f"record_return_layouts="
              f"{dict(callee.metadata.get('record_return_layouts', ()))!r} "
              f"source_output_value_ids="
              f"{tuple(callee.metadata.get('source_output_value_ids', ()))!r} "
              f"structural_shortfalls="
              f"{callee.metadata.get('structural_output_shortfalls')!r}",
              flush=True)
    calls = [
        (block_name, instruction)
        for block_name, block in root.blocks.items()
        for instruction in block.instrs
        if instruction.op == "Call" and instruction.attributes.get("source_linked")
        and not instruction.attributes.get("ssa_sequence_operation")
    ]
    for block_name, call in calls:
        print(f"  linked call in {block_name!r}: "
              f"callee={call.attributes.get('callee')!r} "
              f"res={None if call.res is None else int(call.res.id)} "
              f"output_ids={call.attributes.get('output_ids')!r} "
              f"result_convention={call.attributes.get('result_convention')!r}",
              flush=True)
    for record in (getattr(module, "call_table", {}) or {}).get(root.name, ()):
        print(f"  call record: callsite={record.callsite_id} "
              f"callee={record.callee_symbol!r} resolution={record.resolution!r} "
              f"result_bindings={tuple(record.result_bindings)!r} "
              f"n_argument_bindings={len(record.argument_bindings)}",
              flush=True)
    undefined = _undefined_repository_ssa_operands(module)
    print(f"  undefined_operands={len(undefined)}", flush=True)
    for finding in undefined[:8]:
        print(f"    {finding['function']} {finding['block']} {finding['operation']} "
              f"value_id={finding['value_id']} callee={finding['callee']!r} "
              f"names={finding['value_names']!r} "
              f"operand_ids={finding['operand_ids']!r}", flush=True)
    return 0 if not undefined else 2


if __name__ == "__main__":
    raise SystemExit(main())

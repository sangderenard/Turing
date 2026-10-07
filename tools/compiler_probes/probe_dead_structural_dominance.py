"""A recovered predicate chain nothing reads must not outlive its operands.

``recover_structural_source_outputs`` turns every anonymous control-predicate
formal into a structural producer chain and inserts the chain before the
function's ``Ret``.  When the arm's own ``control_expression`` already computes
the predicate, nothing reads the recovered chain; it is dead, and
``drop_dead_pure_structural_instructions`` -- the sweep the pipeline runs right
after the recovery -- is the pass that retires dead structural values.  Its
operation vocabulary (``_PURE_REGION_OPS``) lacked ``LNot``, ``Select`` and
``isfinite``, so the chain's tail survived in ``while_exit`` / ``function_exit``
and its ``LNot`` read ``isfinite(...)``, a value defined in a loop body or an
arm that does not dominate the exit (N=2 orbital dt system:
``step_with_dt_control_used``).

    python -u tools/compiler_probes/probe_dead_structural_dominance.py

A real program through ``lower_ast_source_to_ssa`` under the program
extraction contract (the retry loop of ``step_with_dt_control_used`` in
miniature: a ``not math.isfinite(..) or ..`` guard and a
``limit is not None and math.isfinite(limit) and limit > 0`` guard).  The
module must have no operand whose definition does not dominate it, and every
swept value is one ``dead_structural_retirement`` row DERIVED from a cell.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "compiler_probes"))

SOURCE = '''
import math


def advance(x):
    return x * 0.5 - 1.0


def root(x, dt, limit):
    step = 0
    while step < 4:
        v = advance(x)
        if (not math.isfinite(v) or v <= 0.0 or v >= dt):
            dt = dt * 0.5
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0.0:
            dt = min(dt, limit)
        step = step + 1
    return dt
'''

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label)
    if not condition:
        failures.append(label)


def main() -> int:
    from probe_module_dominance import module_dominance_violations
    from src.compiler.concordance_declarations import DEAD_STRUCTURAL_RETIREMENT
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = REPO / "extraction_contracts" / "program_extraction.yaml"
    module, _outputs, _exports = lower_ast_source_to_ssa(
        SOURCE, "root", name="dead_structural", extraction_contract=contract,
        progress=lambda message: None,
    )
    violations = module_dominance_violations(module)
    for check_name, items in violations.items():
        for item in items:
            print(f"  {check_name}: {item}")
    check("no operand reads a value that is undefined",
          not violations["undefined_operand"])
    check("every operand's definition dominates its use",
          not violations["definition_does_not_dominate_use"])

    root = next(
        function for name, function in module.functions.items()
        if str(name).endswith("__root")
    )
    unused = [
        (str(block_name), int(index), str(instruction.op))
        for block_name, block in root.blocks.items()
        for index, instruction in enumerate(block.instrs)
        if instruction.res is not None
        and instruction.attributes.get("structural_operation") is not None
        and not any(
            int(argument.id) == int(instruction.res.id)
            for other in root.blocks.values()
            for consumer in other.instrs
            for argument in consumer.args
        )
    ]
    print(f"  unconsumed structural values left in root: {unused}")
    check("no unconsumed structural value survives the sweep", not unused)

    book = module.metadata["identity_book"]
    page = book.page(DEAD_STRUCTURAL_RETIREMENT)
    rows = [row for row in page.rows() if row[0] == (
        root.metadata.get("tensor_shape_concordance_scope") or root.name
    )]
    ops = sorted({page.latest(row)[0] for row in rows})
    print(f"  dead_structural_retirement rows for root: {len(rows)} ops={ops}")
    check("the chain's LNot, Select and isfinite were retired through rows",
          {"LNot", "Select", "isfinite"} <= set(ops))
    derived = [
        row for row in rows
        if book.latest_ref(DEAD_STRUCTURAL_RETIREMENT, row) is not None
        and book.edges_into(book.latest_ref(DEAD_STRUCTURAL_RETIREMENT, row))
    ]
    check("every retirement row has an inbound edge", len(derived) == len(rows))
    print("FAILED: " + "; ".join(failures) if failures else "all green")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
